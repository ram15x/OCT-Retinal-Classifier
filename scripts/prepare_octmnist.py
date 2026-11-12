# downloads OCTMNIST (224px) and writes it as folders: <out>/{train,val,test}/<class>/<i>.png
# OCTMNIST is the Kermany OCT2017 data repackaged by MedMNIST, same four classes
# usage: python scripts/prepare_octmnist.py --out data/octmnist
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from medmnist import OCTMNIST
from PIL import Image

NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]  # same order as the OCTMNIST labels


def write_split(split, out):
    ds = OCTMNIST(split=split, download=True, size=224)
    imgs, labels = ds.imgs, ds.labels.reshape(-1)
    print(split, imgs.shape, np.bincount(labels))
    for n in NAMES:
        os.makedirs(f"{out}/{split}/{n}", exist_ok=True)

    def save(i):
        Image.fromarray(imgs[i]).save(f"{out}/{split}/{NAMES[labels[i]]}/{i}.png", compress_level=1)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(save, range(len(imgs))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/octmnist")
    a = ap.parse_args()
    for split in ("train", "val", "test"):
        write_split(split, a.out)


if __name__ == "__main__":
    main()
