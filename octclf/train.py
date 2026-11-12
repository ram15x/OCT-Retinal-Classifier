# train: python -m octclf.train --data <dataset folder> --out runs/convnext

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from sklearn.metrics import f1_score

from .config import CLASSES, DEFAULT_SEED
from .data import (
    balanced_sampler,
    class_counts,
    find_dataset_root,
    make_dataset,
    make_loader,
    seed_everything,
)
from .losses import FocalLoss
from .model import BACKBONES, build_model


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    ys, ps, probs = [], [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        p = torch.softmax(logits.float(), 1).cpu()
        probs.append(p)
        ps.append(p.argmax(1))
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(probs).numpy()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder containing train/ val/ test/")
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--backbone", default="convnext_tiny", choices=BACKBONES)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--samples-per-epoch", type=int, default=20000)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args(argv)

    seed_everything(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    root = find_dataset_root(a.data)
    train_ds = make_dataset(root, "train", train=True)
    val_ds = make_dataset(root, "val", train=False)
    print(f"device={device} root={root}")
    print("train class counts:", dict(zip(CLASSES, class_counts(train_ds))))

    n_epoch = min(a.samples_per_epoch, len(train_ds))
    train_loader = make_loader(train_ds, a.batch_size,
                               sampler=balanced_sampler(train_ds, n_epoch), workers=a.workers)
    val_loader = make_loader(val_ds, a.batch_size, workers=a.workers)

    model = build_model(a.backbone, pretrained=not a.no_pretrained).to(device)
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    history, best_f1, bad = [], -1.0, 0
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        yv, pv, _ = predict_loader(model, val_loader, device)
        acc = float((yv == pv).mean())
        f1 = float(f1_score(yv, pv, average="macro"))
        rec = {"epoch": epoch, "train_loss": total / len(train_loader),
               "val_acc": acc, "val_macro_f1": f1, "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if f1 > best_f1:
            best_f1, bad = f1, 0
            torch.save({"model": model.state_dict(), "backbone": a.backbone,
                        "classes": list(CLASSES), "val_macro_f1": f1}, out / "best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping")
                break
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best val macro-F1: {best_f1:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
