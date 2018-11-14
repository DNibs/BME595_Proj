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


def main():

    if torch.cuda.is_available() and FLAG_USE_GPU:
        device = torch.device('cuda:0')
        pin_mem = True
        workers = 8
    else:
        device = torch.device('cpu')
        pin_mem = False
        workers = 0
    print('device = {}'.format(device))

    dir_data = '../MSCOCO/unlabeled2017/'

    train_dataset, val_dataset, test_dataset = data_prep.build_datasets(dir_data)

    # print(train_dataset.__len__())
    # print(val_dataset.__len__())
    # print(test_dataset.__len__())
    #
    # im1, t1 = train_dataset.__getitem__(5)
    #
    # print(im1.size())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=32,
                                               shuffle=False,
                                               pin_memory=pin_mem,
                                               num_workers=workers)

    i = 0
    for images, target in train_loader:
        print(images.size())
        images = images.to(device)
        print(images.get_device())
        i += 1
        print(i)

if __name__ == '__main__':
    main()
