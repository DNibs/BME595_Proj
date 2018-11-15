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
import data_prep
import torch
import torchvision

# Set Hyperparameters
FLAG_USE_GPU = True
dir_data = '../MSCOCO/unlabeled2017/'
batch_sz = 32

# Prepare Data
print('Preparing Data...')

dataset_train, dataset_val = data_prep.build_datasets_debugging(dir_data)


loader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                           batch_size=batch_sz,
                                           shuffle=False,
                                           drop_last=True)

loader_val = torch.utils.data.DataLoader(dataset=dataset_val,
                                         batch_size=batch_sz,
                                         shuffle=False,
                                         drop_last=True)

for batch_idx, (data, target) in enumerate(loader_train):

    print(data.size())
    print(data.type())
    plt.figure(0)
    plt.imshow(data[0, 0, :, :])
    plt.figure(1)
    plt.imshow(data[0, 1, :, :])
    plt.show()
