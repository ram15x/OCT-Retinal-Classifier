# Notes on the results

- This is a learning project and not a medical device. It should not be used to diagnose anything.
- All numbers come from one training run (seed 42) on the 1,000-image MedMNIST test split. With that many images the 95% interval on an accuracy is about plus or minus 1.2 points, so the hybrid is shown to be no worse than the single models, not clearly better.
- The test set is balanced (250 per class) and real data is not (almost half the training images are normal), so precision on real patients would look different.
- Almost all errors go one way: 25 of 250 DRUSEN scans and 12 of 250 DME scans are called CNV. No NORMAL scan was called a disease, and 4 DRUSEN scans were called normal.
- The Grad-CAM examples are random picks (2 per class) and all happen to be correct. The wrong ones are in the confusion matrix.
- The trained weights (about 110 to 120 MB each) are not in the repository.
