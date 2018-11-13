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

"""
Data functions
    Create train, val, test lists
        read all imgs
        sort numerically
        partition 80/15/5
        return three dif lists
    Preprocessing
        read file at random from list
        resize
        get random patch coordinates
        get random perturbed patch coordinates
        concatenate points into H_4pts
        compute 
        warp image
        get patch from orignal image and warp image, concatenate depthwise
        return X=img_patches, Y=H_4pts
"""


def get_test(path):
    rho = 32
    patch_size = 128
    height = 240
    width = 320
    # random read image
    loc_list = glob(path)
    print(len(loc_list))
    print(loc_list[0:5])
    random.shuffle(loc_list)
    print(loc_list[0:5])
    loc_list.sort()
    print(loc_list[0:5])
    train_len = int(len(loc_list) * 0.9)
    val_len = int(len(loc_list) * 0.95)

    train_lst = loc_list[0:train_len]
    val_lst = loc_list[train_len+1:-1001]
    test_lst = loc_list[-1000:]

    print('train len {}'.format(len(train_lst)))
    print('val len {}'.format(len(val_lst)))
    print('test len {}'.format(len(test_lst)))

    index = random.randint(0, len(loc_list) - 1)
    img_file_location = loc_list[index]
    color_image = plt.imread(img_file_location)
    color_image = cv2.resize(color_image, (width, height))
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    # points
    y = random.randint(rho, height - rho - patch_size)  # row
    x = random.randint(rho, width - rho - patch_size)  # col
    top_left_point = (x, y)
    bottom_left_point = (patch_size + x, y)
    bottom_right_point = (patch_size + x, patch_size + y)
    top_right_point = (x, patch_size + y)
    four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
    four_points_array = np.array(four_points)
    perturbed_four_points = []
    for point in four_points:
        perturbed_four_points.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))

    # compute H
    H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
    H_inverse = np.linalg.inv(H)
    inv_warped_image = cv2.warpPerspective(gray_image, H_inverse, (width, height))
    # grab image patches
    original_patch = gray_image[y:y + patch_size, x:x + patch_size]
    warped_patch = inv_warped_image[y:y + patch_size, x:x + patch_size]
    # make into dataset
    training_image = np.dstack((original_patch, warped_patch))
    val_image = training_image.reshape((1, 128, 128, 2))

    return color_image, H_inverse, val_image, four_points_array


color_image, H_matrix,val_image,four_points_array = get_test("./datasets/unlabeled2017/*.jpg")
four_points_array_ = four_points_array.reshape((1,4,2))
rectangle_image = cv2.polylines(color_image, four_points_array_, 1, (0,0,255),2)
warped_image = cv2.warpPerspective(rectangle_image, H_matrix, (color_image.shape[1], color_image.shape[0]))

plt.imshow(rectangle_image)
plt.title('original image')
plt.show()

plt.imshow(warped_image)
plt.title('warped_image image')
plt.show()
