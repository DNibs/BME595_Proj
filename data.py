# BME595 Project
# Author: David Niblick
# Date: 05DEC18
# data.py


import cupy as cp
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import random
from glob import glob
import torch
import torchvision


# Filepaths
dir_data = '../MSCOCO/unlabeled2017/'


class CustomDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, dir_data, idx_begin, idx_end, rho, patch_sz, height, width):
        data_lst = glob(dir_data+'*.jpg')
        data_lst.sort()
        self.data_lst = data_lst[idx_begin:idx_end]
        self.rho = rho
        self.patch_sz = patch_sz
        self.height = height
        self.width = width

    def __getitem__(self, index):
        # Get random image
        idx = random.randint(0, len(self.data_lst) - 1)
        img_file_location = self.data_lst[idx]
        color_image = plt.imread(img_file_location)
        color_image = cv2.resize(color_image, (self.width, self.height))
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)

        # Create point for corners of image patch
        y = random.randint(self.rho, self.height - self.rho - self.patch_sz)  # row
        x = random.randint(self.rho, self.width - self.rho - self.patch_sz)  # col
        # define corners of image patch
        top_left_point = (x, y)
        bottom_left_point = (self.patch_sz + x, y)
        bottom_right_point = (self.patch_sz + x, self.patch_sz + y)
        top_right_point = (x, self.patch_sz + y)
        four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
        perturbed_four_points = []
        for point in four_points:
            perturbed_four_points.append(
                (point[0] + random.randint(-self.rho, self.rho), point[1] + random.randint(-self.rho, self.rho)))

        # calculate H
        H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
        H_inverse = np.linalg.inv(H)
        inv_warped_image = cv2.warpPerspective(gray_image, H_inverse, (320, 240))
        warped_image = cv2.warpPerspective(gray_image, H, (320, 240))

        # grab image patches
        original_patch = gray_image[y:y + self.patch_sz, x:x + self.patch_sz]
        warped_patch = inv_warped_image[y:y + self.patch_sz, x:x + self.patch_sz]
        # Stack patches to create input
        training_image = np.dstack((original_patch, warped_patch))
        H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
        X = training_image
        Y = H_four_points.reshape(-1)

        return X, Y

    def __len__(self):
        return len(self.data_lst)



def build_lists(filepath = dir_data, train_fraction = 0.85, test_number = 1000):
    """
    Create training, validation, and test data lists
    :param filepath:   folder containing all images
    :param train_fraction:   share of images used for training
    :param test_number:    number of images resrved for testing  (validation is built from leftovers
    :return: lists of file paths for train, val, test data
    """

    # Read in images and sort
    loc_list = glob(filepath+'*.jpg')
    loc_list.sort()

    # Partition dataset
    train_len = int(len(loc_list) * train_fraction)
    train_lst = loc_list[0:train_len]
    val_lst = loc_list[train_len + 1:-(test_number+1)]
    test_lst = loc_list[-test_number:]

    print('Number of Training Images: {}'.format(len(train_lst)))
    print('Number of Validation Images: {}'.format(len(val_lst)))
    print('Number of Test Images: {}'.format(len(test_lst)))

    return train_lst, val_lst, test_lst


def dataloader(data_lst, batch_sz=1024, rho=32, patch_size = 128, height=240, width=320):

    """
    Loads random image, resize, create random patche coordinates, create random perturbed coordinates,
    concatenate point sets to H_4pts (target), compute homography using coordinate sets, warp image per homography,
    use coordinates to pull patches from original and warped imgs, concatenate patches
    :param data_lst: Filepaths to images in dataset
    :param batch_sz:
    :param rho:
    :param patch_size:
    :param height:
    :param width:
    :return: X = concatenated patches (model input), Y = H_4pts (model target
    """

    while 1:

        X = np.zeros((batch_sz, 128, 128, 2))  # Input to model - concatenate patches of original and warped imgs
        Y = np.zeros((batch_sz, 8)) # Target of model - 4pt parameterized homography
        for i in range(batch_sz):
            # Get random image
            index = random.randint(0, len(data_lst) - 1)
            img_file_location = data_lst[index]
            color_image = plt.imread(img_file_location)
            color_image = cv2.resize(color_image, (width, height))
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)

            # Create point for corners of image patch
            y = random.randint(rho, height - rho - patch_size)  # row
            x = random.randint(rho, width - rho - patch_size)  # col
            # define corners of image patch
            top_left_point = (x, y)
            bottom_left_point = (patch_size + x, y)
            bottom_right_point = (patch_size + x, patch_size + y)
            top_right_point = (x, patch_size + y)
            four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
            perturbed_four_points = []
            for point in four_points:
                perturbed_four_points.append(
                    (point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))

            # calculate H
            H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
            H_inverse = np.linalg.inv(H)
            inv_warped_image = cv2.warpPerspective(gray_image, H_inverse, (320, 240))
            warped_image = cv2.warpPerspective(gray_image, H, (320, 240))

            # grab image patches
            original_patch = gray_image[y:y + patch_size, x:x + patch_size]
            warped_patch = inv_warped_image[y:y + patch_size, x:x + patch_size]
            # Stack patches to create input
            training_image = np.dstack((original_patch, warped_patch))
            H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
            X[i, :, :] = training_image
            Y[i, :] = H_four_points.reshape(-1)
        yield (X, Y)

