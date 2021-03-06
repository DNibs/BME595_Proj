# BME595 Project
# Author: David Niblick
# Date: 05DEC18
# data_prep.py


import cupy as cp
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import random
from glob import glob
import torch
import torchvision


class CustomDataset(torch.utils.data.dataset.Dataset):

    """
    Custom dataset that creates labels 'on the fly'.
    Pulls image, randomly selects 4 points (no closer than rho from edges),
    randomly perturbs four points (from -rho to rho of original point),
    calculates homography between pts and perturbed pts (to be passed as label)
    stacks PATCHES from images using the points as the training input image
    """

    def __init__(self, data_directory, idx_begin, idx_end, rho=32, patch_sz=128, height=240, width=320):
        data_lst = glob(data_directory+'*.jpg')
        data_lst.sort()
        self.data_lst = data_lst[idx_begin:idx_end]
        self.rho = rho
        self.patch_sz = patch_sz
        self.height = height
        self.width = width


    def __getitem__(self, index):
        # Get random image, resize
        idx = random.randint(0, len(self.data_lst) - 1)
        fp = self.data_lst[idx]
        img = cv2.imread(fp, 0)
        img = cv2.resize(img, (self.width, self.height))

        # Create point for corners of image patch
        m = random.randint(self.rho, self.width - self.rho - self.patch_sz)  # col
        n = random.randint(self.rho, self.height - self.rho - self.patch_sz)  # row
        # define corners of image patch
        top_left_point = (m, n)
        bottom_left_point = (m, self.patch_sz + n)
        bottom_right_point = (self.patch_sz + m, self.patch_sz + n)
        top_right_point = (self.patch_sz + m, n)
        four_points = [top_left_point, top_right_point, bottom_left_point, bottom_right_point]
        perturbed_four_points = []
        for point in four_points:
            perturbed_four_points.append(
                (point[0] + random.randint(-self.rho, self.rho), point[1] + random.randint(-self.rho, self.rho)))

        # calculate H
        H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
        # H_inverse = np.linalg.pinv(H)
        # inv_warped_image = cv2.warpPerspective(img, H_inverse, (320, 240))
        warped_image = cv2.warpPerspective(img, H, (320, 240))

        # grab image patches
        original_patch = img[n:n + self.patch_sz, m:m + self.patch_sz]
        warped_patch = warped_image[n:n + self.patch_sz, m:m + self.patch_sz]

        # Stack patches to create input
        img_train = np.dstack([((original_patch / 255.0) - 0.456) / 0.224, ((warped_patch / 255.0) - 0.456) / 0.224])
        img_train = img_train.swapaxes(0, 2)
        img_train = img_train.swapaxes(1, 2)
        img_train = torch.from_numpy(img_train).float()

        H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
        target = torch.from_numpy(H_four_points.reshape(-1)).float()

        return img_train, target

    def __len__(self):
        return len(self.data_lst)


def build_datasets(dir_data, train_fraction=0.85):
    """
    Create training, validation, and test data lists
    :param dir_data: folder containing all images
    :param train_fraction:   share of images used for training
    :param test_number:    number of images resrved for testing  (validation is built from leftovers
    :return: lists of file paths for train, val, test data
    """

    # Read in images and sort
    loc_list = glob(dir_data+'*.jpg')
    loc_list.sort()
    lst_len = len(loc_list)

    # Partition dataset
    train_idx_end = int(lst_len * train_fraction)

    train_dataset = CustomDataset(dir_data, 0, train_idx_end)
    val_dataset = CustomDataset(dir_data, train_idx_end + 1, lst_len)

    return train_dataset, val_dataset


def build_datasets_debugging(dir_data):
    """
    Smaller dataset for debugging purposes
    :param dir_data:
    :return:
    """

    # Read in images and sort
    loc_list = glob(dir_data+'*.jpg')
    loc_list.sort()
    lst_len = len(loc_list)

    # Partition dataset
    train_idx_end = int(1000)

    train_dataset = CustomDataset(dir_data, 0, train_idx_end)
    val_dataset = CustomDataset(dir_data, train_idx_end + 1, 1101)

    return train_dataset, val_dataset



