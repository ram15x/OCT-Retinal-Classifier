# predict one image: python -m octclf.predict --ckpt best.pt --image scan.png --out out.png

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

from .config import CLASSES
from .data import eval_transform
from .evaluate import load_checkpoint
from .gradcam import GradCAM, denormalize, overlay
from .model import gradcam_layer


def predict_image(model, backbone, image: Image.Image, device):
    x = eval_transform()(image.convert("RGB")).unsqueeze(0).to(device)
    engine = GradCAM(model, gradcam_layer(model, backbone))
    cam, probs, pred = engine(x)
    engine.close()
    return {"label": CLASSES[pred], "probs": dict(zip(CLASSES, map(float, probs)))}, cam, denormalize(x[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="prediction.png")
    a = ap.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, backbone = load_checkpoint(a.ckpt, device)
    res, cam, img = predict_image(model, backbone, Image.open(a.image), device)
    print(res)
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.6))
    ax[0].imshow(img)
    ax[1].imshow(overlay(img, cam))
    ax[1].set_title(f"{res['label']} ({res['probs'][res['label']]:.0%})")
    for x in ax:
        x.axis("off")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
