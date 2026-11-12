# quick cpu tests on a tiny fake dataset, nothing gets downloaded

import numpy as np
import pytest
import torch
from PIL import Image

from octclf import train as train_mod
from octclf.config import CLASSES
from octclf.data import find_dataset_root, make_dataset
from octclf.evaluate import metrics_dict
from octclf.gradcam import GradCAM
from octclf.losses import FocalLoss
from octclf.model import build_model, gradcam_layer


@pytest.fixture(scope="module")
def tiny_root(tmp_path_factory):
    base = tmp_path_factory.mktemp("zip") / "Dataset - train+val+test"
    rng = np.random.default_rng(0)
    for split, n in (("train", 8), ("val", 4), ("test", 4)):
        for ci, c in enumerate(CLASSES):
            d = base / split / c
            d.mkdir(parents=True)
            for k in range(n):
                arr = (rng.random((64, 64)) * 60 + ci * 40).astype("uint8")
                Image.fromarray(arr).save(d / f"{k}.jpeg")
    return base.parent


def test_find_root_handles_nested_zip(tiny_root):
    assert find_dataset_root(tiny_root).name == "Dataset - train+val+test"


def test_dataset_has_all_four_classes_including_normal(tiny_root):
    ds = make_dataset(find_dataset_root(tiny_root), "test", train=False)
    assert tuple(ds.classes) == CLASSES and "NORMAL" in ds.classes
    x, _ = ds[0]
    assert x.shape == (3, 224, 224)


def test_model_outputs_four_logits():
    m = build_model("resnet18", pretrained=False)
    assert m(torch.zeros(2, 3, 224, 224)).shape == (2, 4)


def test_focal_loss_downweights_easy_examples():
    logits = torch.tensor([[10.0, 0, 0, 0], [0.1, 0, 0, 0]])
    t = torch.tensor([0, 0])
    fl, ce = FocalLoss(gamma=2.0), FocalLoss(gamma=0.0)
    assert fl(logits, t) < ce(logits, t)


def test_gradcam_heatmap_shape_and_range():
    m = build_model("resnet18", pretrained=False).eval()
    eng = GradCAM(m, gradcam_layer(m, "resnet18"))
    cam, probs, idx = eng(torch.randn(1, 3, 224, 224))
    eng.close()
    assert cam.shape == (224, 224) and 0 <= cam.min() and cam.max() <= 1
    assert abs(probs.sum() - 1) < 1e-4 and 0 <= idx < 4


def test_metrics_dict_perfect_predictions():
    y = np.repeat(np.arange(4), 5)
    probs = np.eye(4)[y]
    m = metrics_dict(y, y, probs)
    assert m["accuracy"] == 1.0 and m["macro_auc"] == 1.0


def test_train_runs_end_to_end_on_cpu(tiny_root, tmp_path):
    rc = train_mod.main(["--data", str(tiny_root), "--out", str(tmp_path), "--backbone", "resnet18",
                         "--no-pretrained", "--epochs", "2", "--batch-size", "8",
                         "--samples-per-epoch", "32", "--workers", "0"])
    assert rc == 0 and (tmp_path / "best.pt").exists() and (tmp_path / "history.json").exists()


def test_maxvit_hybrid_builds_and_gradcam_works():
    m = build_model("maxvit_t", pretrained=False).eval()
    assert m(torch.zeros(1, 3, 224, 224)).shape == (1, 4)
    eng = GradCAM(m, gradcam_layer(m, "maxvit_t"))
    cam, _, _ = eng(torch.randn(1, 3, 224, 224))
    eng.close()
    assert cam.shape == (224, 224)


def test_ensemble_evaluate_runs(tiny_root, tmp_path):
    from octclf import evaluate as ev
    for n in ("a", "b"):
        train_mod.main(["--data", str(tiny_root), "--out", str(tmp_path / n), "--backbone", "resnet18",
                        "--no-pretrained", "--epochs", "1", "--batch-size", "8",
                        "--samples-per-epoch", "16", "--workers", "0"])
    rc = ev.main(["--data", str(tiny_root), "--ckpt", str(tmp_path / "a/best.pt"),
                  str(tmp_path / "b/best.pt"), "--out", str(tmp_path / "ens"), "--workers", "0"])
    assert rc == 0 and (tmp_path / "ens/metrics.json").exists()
