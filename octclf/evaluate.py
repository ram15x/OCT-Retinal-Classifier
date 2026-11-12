# evaluate on the test split: python -m octclf.evaluate --data <dataset folder> --ckpt runs/convnext/best.pt

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from .config import CLASSES
from .data import find_dataset_root, make_dataset, make_loader
from .gradcam import GradCAM, denormalize, overlay
from .model import build_model, gradcam_layer
from .train import predict_loader


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ck["backbone"], pretrained=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck["backbone"]


def metrics_dict(y, pred, probs) -> dict:
    rep = classification_report(y, pred, target_names=CLASSES, output_dict=True, zero_division=0)
    auc = {c: float(roc_auc_score((y == i).astype(int), probs[:, i])) for i, c in enumerate(CLASSES)}
    return {"accuracy": float((y == pred).mean()), "per_class": {c: rep[c] for c in CLASSES},
            "macro_f1": rep["macro avg"]["f1-score"], "auc_ovr": auc,
            "macro_auc": float(np.mean(list(auc.values()))),
            "confusion_matrix": confusion_matrix(y, pred).tolist(), "n": len(y)}


def plot_confusion(cm, path) -> None:
    cm = np.array(cm)
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4), CLASSES)
    ax.set_yticks(range(4), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_title("Test confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_gradcam_grid(model, backbone, ds, path, per_class: int = 2, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    cam_engine = GradCAM(model, gradcam_layer(model, backbone))
    picks = []
    for c in range(len(CLASSES)):
        idx = [i for i, (_, y) in enumerate(ds.samples) if y == c]
        picks += list(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
    fig, axes = plt.subplots(2, len(picks), figsize=(2.1 * len(picks), 4.6))
    for k, i in enumerate(picks):
        x, y = ds[i]
        cam, probs, pred = cam_engine(x.unsqueeze(0).to(device))
        img = denormalize(x)
        axes[0, k].imshow(img)
        axes[0, k].set_title(f"true {CLASSES[y]}", fontsize=8)
        axes[1, k].imshow(overlay(img, cam))
        axes[1, k].set_title(f"pred {CLASSES[pred]} {probs[pred]:.0%}", fontsize=8,
                             color="green" if pred == y else "red")
        axes[0, k].axis("off")
        axes[1, k].axis("off")
    cam_engine.close()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True, nargs="+", help="one or more checkpoints (ensemble = mean of probabilities)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out or Path(a.ckpt[0]).parent)
    out.mkdir(parents=True, exist_ok=True)
    ds = make_dataset(find_dataset_root(a.data), "test", train=False)
    loader = make_loader(ds, a.batch_size, workers=a.workers)
    all_probs, first = [], None
    for path in a.ckpt:
        model, backbone = load_checkpoint(path, device)
        first = first or (model, backbone)
        y, p_single, probs_single = predict_loader(model, loader, device)
        print(f"{Path(path).parent.name}/{backbone}: acc={float((y == p_single).mean()):.4f}")
        all_probs.append(probs_single)
    probs = np.mean(all_probs, axis=0)
    pred = probs.argmax(1)
    model, backbone = first
    m = metrics_dict(y, pred, probs)
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    plot_confusion(m["confusion_matrix"], out / "confusion_matrix.png")
    save_gradcam_grid(model, backbone, ds, out / "gradcam_examples.png")
    print(classification_report(y, pred, target_names=CLASSES, digits=3))
    print("macro AUC:", round(m["macro_auc"], 4), "accuracy:", round(m["accuracy"], 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
