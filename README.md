# OCT Retinal Classifier

**Read the scan. Show where it looked.**

A retinal OCT classifier that sorts scans into four groups: CNV, DME, DRUSEN and NORMAL. It combines a CNN (ConvNeXt-Tiny) and a convolution-attention hybrid (MaxViT-Tiny), averages their predictions, and draws a Grad-CAM heatmap so you can see which part of the scan drove each call.

Built for research and learning. It is not a medical device.

## How it helps

OCT scans are the standard way to look for the retinal conditions behind vision loss, and reading them takes trained eyes. This project shows how well a modern vision model can sort the four most common scan types, and the heatmaps make each prediction inspectable instead of a bare label.

## Results

Test set: 1,000 images, 250 per class. Seed 42.

| model | accuracy | macro F1 | macro AUC |
|---|---|---|---|
| ConvNeXt-Tiny (CNN) | 95.5% | 0.955 | 0.9965 |
| MaxViT-Tiny (conv + attention) | 95.1% | 0.951 | 0.9971 |
| **both averaged (hybrid)** | **95.8%** | **0.958** | **0.9974** |

Per class, for the averaged model:

| class | precision | recall |
|---|---|---|
| CNV | 0.871 | 0.996 |
| DME | 1.000 | 0.952 |
| DRUSEN | 0.995 | 0.884 |
| NORMAL | 0.984 | 1.000 |

![confusion matrix](docs/confusion_matrix.png)

Every NORMAL scan in the test set was recognised as normal, and CNV recall is 99.6%.

### What the model looks at

Grad-CAM heatmaps from the averaged model. The top row is the scan and the bottom row is where the model looked. The examples are random picks, 2 per class.

![gradcam](docs/gradcam.png)

## How it works

```
OCT scan (224x224) --> ConvNeXt-Tiny -->\
                                          average probabilities --> class + Grad-CAM heatmap
OCT scan (224x224) --> MaxViT-Tiny  -->/
```

- `octclf/data.py` loads the folders, with a class-balanced sampler so the small classes are seen as often as the big ones.
- `octclf/model.py` builds ConvNeXt-Tiny or MaxViT-Tiny (or ResNet / EfficientNetV2) from ImageNet weights.
- `octclf/losses.py` has a focal loss (gamma 2) with light label smoothing.
- `octclf/train.py` trains with AdamW, a one-cycle learning rate and mixed precision, and keeps the checkpoint with the best validation macro F1.
- `octclf/evaluate.py` computes metrics, the confusion matrix and the Grad-CAM grid. Give it several checkpoints and it averages their probabilities.
- `octclf/predict.py` takes one image and returns a label plus a heatmap.

## Data

OCTMNIST from MedMNIST at 224x224, which is the Kermany OCT2017 data with a fixed split.

| split | images |
|---|---|
| train | 97,477 (CNV 33,484 / DME 10,213 / DRUSEN 7,754 / NORMAL 46,026) |
| val | 10,832 |
| test | 1,000 (250 each) |

## Tech stack

Python, PyTorch and torchvision (pretrained ConvNeXt-Tiny and MaxViT-Tiny), scikit-learn, OpenCV, Matplotlib, MedMNIST, pytest and GitHub Actions. Training takes about 16 minutes for ConvNeXt and 26 for MaxViT on a free Colab T4: 6 epochs, 16,000 images per epoch, batch 64 (48 for MaxViT).

## Run it

```
pip install -r requirements.txt
python scripts/prepare_octmnist.py --out data/octmnist     # about 4 GB download

python -m octclf.train --data data/octmnist --out runs/convnext_tiny --backbone convnext_tiny \
    --epochs 6 --samples-per-epoch 16000
python -m octclf.train --data data/octmnist --out runs/maxvit_t --backbone maxvit_t \
    --epochs 6 --batch-size 48 --lr 2e-4 --samples-per-epoch 16000

python -m octclf.evaluate --data data/octmnist \
    --ckpt runs/convnext_tiny/best.pt runs/maxvit_t/best.pt --out runs/ensemble

python -m octclf.predict --ckpt runs/convnext_tiny/best.pt --image some_scan.png
```

Training is comfortable on a GPU. The tests run on CPU and need no download:

```
pip install -r requirements-dev.txt
pytest
```

The metrics from the runs above are in `results/`.

## Credits

Kermany et al., OCT2017 dataset. MedMNIST (Yang et al.) for the packaged OCTMNIST version. PyTorch and torchvision for the model weights.

## More

Notes on the results and on running it: [docs/notes.md](docs/notes.md).

Released under the [MIT License](LICENSE).
