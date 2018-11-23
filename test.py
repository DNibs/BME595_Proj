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
dir_data = '../MSCOCO/unlabeled2017/'


def test_NibsNet():
    rho = 32
    patch_sz = 128
    height = 240
    width = 320

    data_lst = glob(dir_data + '*.jpg')
    idx = random.randint(0, len(data_lst) - 1)
    fp = data_lst[idx]
    img = cv2.imread(fp)
    img_g = cv2.imread(fp, 0)
    img = cv2.resize(img, (width, height))

    # Create point for corners of image patch
    m = random.randint(rho, height - rho - patch_sz)  # row
    n = random.randint(rho, width - rho - patch_sz)  # col
    # define corners of image patch
    top_left_point = (m, n)
    bottom_left_point = (patch_sz + m, n)
    bottom_right_point = (patch_sz + m, patch_sz + n)
    top_right_point = (m, patch_sz + n)
    four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
    perturbed_four_points = []
    for point in four_points:
        perturbed_four_points.append(
            (point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))

    # calculate H
    H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
    H_inverse = np.linalg.pinv(H)
    inv_warped_image = cv2.warpPerspective(img_g, H_inverse, (320, 240))
    # warped_image = cv2.warpPerspective(img, H, (320, 240))

    # grab image patches
    original_patch = img_g[m:m + patch_sz, n:n + patch_sz]
    warped_patch = inv_warped_image[m:m + patch_sz, n:n + patch_sz]

    # Stack patches to create input
    img_train = np.dstack([((original_patch / 255.0) - 0.456) / 0.224, ((warped_patch / 255.0) - 0.456) / 0.224])
    img_train = img_train.swapaxes(0, 2)
    img_train = img_train.swapaxes(1, 2)
    img_train = torch.from_numpy(img_train).float()

    H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
    target = torch.from_numpy(H_four_points.reshape(-1)).float()

    print(H_four_points)
    print(target)


    net = nib.NibsNet1()
    cp = torch.load(dir_model + 'best_val_model.tar')
    net.load_state_dict(cp['model_state_dict'])
    net.eval()

    model_input = img_train.unsqueeze(0)
    out = net(model_input)

    print(out)



    pts = [(b, a) for a, b in four_points]
    pts_wpd = [(b, a) for a, b in perturbed_four_points]

    img_cpy1 = img.copy()
    cv2.line(img_cpy1, pts[0], pts[1], (0, 0, 255), thickness=2)
    cv2.line(img_cpy1, pts[0], pts[3], (0, 0, 255), thickness=2)
    cv2.line(img_cpy1, pts[3], pts[2], (0, 0, 255), thickness=2)
    cv2.line(img_cpy1, pts[1], pts[2], (0, 0, 255), thickness=2)

    img_cpy2 = img.copy()
    cv2.line(img_cpy2, pts_wpd[0], pts_wpd[1], (0, 255, 0), thickness=2)
    cv2.line(img_cpy2, pts_wpd[0], pts_wpd[3], (0, 255, 0), thickness=2)
    cv2.line(img_cpy2, pts_wpd[3], pts_wpd[2], (0, 255, 0), thickness=2)
    cv2.line(img_cpy2, pts_wpd[1], pts_wpd[2], (0, 255, 0), thickness=2)

    img_wpd_cpy = inv_warped_image.copy()
    cv2.line(img_wpd_cpy, pts_wpd[0], pts_wpd[1], (0, 0, 255), thickness=2)
    cv2.line(img_wpd_cpy, pts_wpd[0], pts_wpd[3], (0, 0, 255), thickness=2)
    cv2.line(img_wpd_cpy, pts_wpd[3], pts_wpd[2], (0, 0, 255), thickness=2)
    cv2.line(img_wpd_cpy, pts_wpd[1], pts_wpd[2], (0, 0, 255), thickness=2)

    plt.figure(0)
    plt.imshow(img_cpy1)

    plt.figure(1)
    plt.imshow(original_patch)

    plt.figure(2)
    plt.imshow(img_wpd_cpy)

    plt.figure(3)
    plt.imshow(warped_patch)

    plt.show()



test_NibsNet()




# for batch_idx, (data, target) in enumerate(loader_train):
#     print(data.size())
#     print(data[0, 0, :, :])
#     print(data.max())
#     print(data.mean())
#     plt.imshow(data[0, 0, :, :])
#     plt.show()




