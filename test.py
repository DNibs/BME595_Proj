# BME595 Project
# Author: David Niblick
# Date: 05DEC18
# test.py


import cupy as cp
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import random
from glob import glob
import data
import torch
import torchvision

# Set Hyperparameters
FLAG_USE_GPU = False


if torch.cuda.is_available() and FLAG_USE_GPU:
    device = torch.device('cuda:0')
    pin_mem = True
    workers = 4
else:
    device = torch.device('cpu')
    pin_mem = False
    workers = 0
print('device = {}'.format(device))


dir_data = '../MSCOCO/unlabeled2017/'

train_dataset, val_dataset, test_dataset = data.build_datasets(dir_data)

train_dataset.device = device
val_dataset.device = device
test_dataset.device = device

print(train_dataset.__len__())
print(val_dataset.__len__())
print(test_dataset.__len__())

im1, t1 = train_dataset.__getitem__(5)

print(im1.size())
