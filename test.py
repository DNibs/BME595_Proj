# BME595 Project
# Author: David Niblick
# Date: 05DEC18
# test.py


import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import random
from glob import glob
import data_prep
import torch
import torchvision
import data_prep
import nibs_homography_net as nib


# Set Hyperparameters
FLAG_USE_GPU = True
dir_data = '../MSCOCO/unlabeled2017/'
batch_sz = 64
learn_rt = 0.005
wt_decay = 0.001355
momentum = 0.9

dir_model = './model/'
dir_metric = './metrics/'

net = nib.NibsNet1()

dataset_train, dataset_val = data_prep.build_datasets_debugging(dir_data)

loader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                           batch_size=batch_sz,
                                           shuffle=False,
                                           drop_last=True)

for batch_idx, (data, target) in enumerate(loader_train):
    print(data.size())
    print(data[0, 0, :, :])
    print(data.max())
    print(data.mean())
    plt.imshow(data[0, 0, :, :])
    plt.show()


