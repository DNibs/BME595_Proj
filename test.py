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


def test_single_NibsNet():
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
    img_g = cv2.resize(img_g, (width, height))

    # Create point for corners of image patch
    m = random.randint(rho, width - rho - patch_sz)  # col
    n = random.randint(rho, height - rho - patch_sz)  # row
    # define corners of image patch
    top_left_point = (m, n)
    top_right_point = (patch_sz + m, n)
    bottom_right_point = (patch_sz + m, patch_sz + n)
    bottom_left_point = (m, patch_sz + n)
    four_points = [top_left_point, top_right_point, bottom_left_point, bottom_right_point]
    perturbed_four_points = []
    for point in four_points:
        perturbed_four_points.append(
            (point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))

    # calculate H
    H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
    # H = cv2.getPerspectiveTransform(np.float32(perturbed_four_points), np.float32(four_points))
    H_inverse = np.linalg.pinv(H)
    # inv_warped_image = cv2.warpPerspective(img_g, H_inverse, (320, 240))
    # inv_warped_image_c = cv2.warpPerspective(img, H_inverse, (320, 240))
    warped_image = cv2.warpPerspective(img_g, H, (320, 240))
    warped_image_c = cv2.warpPerspective(img, H, (320, 240))

    # grab image patches
    original_patch = img_g[n:n + patch_sz, m:m + patch_sz]
    # warped_patch = inv_warped_image[n:n + patch_sz, m:m + patch_sz]
    warped_patch = warped_image[n:n + patch_sz, m:m + patch_sz]


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


    dif_out = [[out[0, 0].item(), out[0, 1].item()],
               [out[0, 2].item(), out[0, 3].item()],
               [out[0, 4].item(), out[0, 5].item()],
               [out[0, 6].item(), out[0, 7].item()]]
    pts_out_flip = ((int(out[0, 0].item()) + m, int(out[0, 1].item()) + n),
               (int(out[0, 2].item()) + m + patch_sz, int(out[0, 3].item()) + n),
               (int(out[0, 4].item()) + m, int(out[0, 5].item()) + n + patch_sz),
               (int(out[0, 6].item()) + m + patch_sz, int(out[0, 7].item()) + n + patch_sz))
    print(out)
    print(dif_out)

    error = H_four_points - dif_out
    print('error {}'.format(error))
    mace = np.linalg.norm(error) / 4
    print('MACE {}'.format(mace))

    pt_error = np.linalg.norm(error, axis=1)
    print(pt_error)
    pt_error_avg = np.average(pt_error)
    print(pt_error_avg)


    # pts = [(b, a) for a, b in four_points]
    pts = four_points
    # pts_wpd = [(b, a) for a, b in perturbed_four_points]
    pts_wpd = perturbed_four_points
    # pts_out = [(b, a) for a, b in pts_out_flip]
    pts_out = pts_out_flip

    img_cpy1 = img.copy()
    cv2.line(img_cpy1, pts[0], pts[1], (0, 0, 255), thickness=2)
    cv2.line(img_cpy1, pts[0], pts[2], (0, 0, 255), thickness=2)
    cv2.line(img_cpy1, pts[1], pts[3], (0, 0, 255), thickness=2)
    cv2.line(img_cpy1, pts[2], pts[3], (0, 0, 255), thickness=2)

    # img_wpd_cpy = inv_warped_image_c.copy()
    img_wpd_cpy = warped_image_c.copy()
    # img_wpd_cpy = img.copy()
    cv2.line(img_wpd_cpy, pts_wpd[0], pts_wpd[1], (0, 0, 255), thickness=2)
    cv2.line(img_wpd_cpy, pts_wpd[0], pts_wpd[2], (0, 0, 255), thickness=2)
    cv2.line(img_wpd_cpy, pts_wpd[1], pts_wpd[3], (0, 0, 255), thickness=2)
    cv2.line(img_wpd_cpy, pts_wpd[2], pts_wpd[3], (0, 0, 255), thickness=2)

    cv2.line(img_wpd_cpy, pts_out[0], pts_out[1], (255, 0, 0), thickness=2)
    cv2.line(img_wpd_cpy, pts_out[0], pts_out[2], (255, 0, 0), thickness=2)
    cv2.line(img_wpd_cpy, pts_out[1], pts_out[3], (255, 0, 0), thickness=2)
    cv2.line(img_wpd_cpy, pts_out[2], pts_out[3], (255, 0, 0), thickness=2)

    plt.figure(0)
    plt.imshow(img_cpy1)

    plt.figure(1)
    plt.imshow(original_patch)

    plt.figure(2)
    plt.imshow(img_wpd_cpy)

    plt.figure(3)
    plt.imshow(warped_patch)

    plt.show()


def testNibsNet1(data_lst, net, fn, num_iter):
    rho = 32
    patch_sz = 128
    height = 240
    width = 320

    idx = random.randint(0, len(data_lst) - 1)
    fp = data_lst[idx]
    img = cv2.imread(fp)
    img_g = cv2.imread(fp, 0)
    img = cv2.resize(img, (width, height))
    img_g = cv2.resize(img_g, (width, height))

    # Create point for corners of image patch
    m = random.randint(rho, width - rho - patch_sz)  # col
    n = random.randint(rho, height - rho - patch_sz)  # row
    # define corners of image patch
    top_left_point = (m, n)
    top_right_point = (patch_sz + m, n)
    bottom_right_point = (patch_sz + m, patch_sz + n)
    bottom_left_point = (m, patch_sz + n)
    four_points = [top_left_point, top_right_point, bottom_left_point, bottom_right_point]
    perturbed_four_points = []
    for point in four_points:
        perturbed_four_points.append(
            (point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))

    # calculate H
    H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
    warped_image = cv2.warpPerspective(img_g, H, (320, 240))
    warped_image_c = cv2.warpPerspective(img, H, (320, 240))

    # grab image patches
    original_patch = img_g[n:n + patch_sz, m:m + patch_sz]
    original_patch_c = img[n:n + patch_sz, m:m + patch_sz]
    warped_patch = warped_image[n:n + patch_sz, m:m + patch_sz]
    warped_patch_c = warped_image_c[n:n + patch_sz, m:m + patch_sz]

    # Stack patches to create input
    img_train = np.dstack([((original_patch / 255.0) - 0.456) / 0.224, ((warped_patch / 255.0) - 0.456) / 0.224])
    img_train = img_train.swapaxes(0, 2)
    img_train = img_train.swapaxes(1, 2)
    img_train = torch.from_numpy(img_train).float()

    H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
    target = torch.from_numpy(H_four_points.reshape(-1)).float()

    net.eval()
    model_input = img_train.unsqueeze(0)
    out = net(model_input)

    dif_out = [[out[0, 0].item(), out[0, 1].item()],
               [out[0, 2].item(), out[0, 3].item()],
               [out[0, 4].item(), out[0, 5].item()],
               [out[0, 6].item(), out[0, 7].item()]]
    pts_out_flip = ((int(out[0, 0].item()) + m, int(out[0, 1].item()) + n),
                    (int(out[0, 2].item()) + m + patch_sz, int(out[0, 3].item()) + n),
                    (int(out[0, 4].item()) + m, int(out[0, 5].item()) + n + patch_sz),
                    (int(out[0, 6].item()) + m + patch_sz, int(out[0, 7].item()) + n + patch_sz))

    error = H_four_points - dif_out
    pt_error = np.linalg.norm(error, axis=1)
    pt_error_avg = np.average(pt_error)
    print('IMG: {}   MACE: {}'.format(num_iter, pt_error_avg))
    fn.write('IMG: {}   MACE: {} \n'.format(num_iter, pt_error_avg))

    if num_iter % 25 == 0:
        pts = four_points
        pts_wpd = perturbed_four_points
        pts_out = pts_out_flip
        font = cv2.FONT_HERSHEY_SIMPLEX

        img_cpy1 = img.copy()
        cv2.line(img_cpy1, pts[0], pts[1], (0, 0, 255), thickness=2)
        cv2.line(img_cpy1, pts[0], pts[2], (0, 0, 255), thickness=2)
        cv2.line(img_cpy1, pts[1], pts[3], (0, 0, 255), thickness=2)
        cv2.line(img_cpy1, pts[2], pts[3], (0, 0, 255), thickness=2)

        img_wpd_cpy = warped_image_c.copy()
        cv2.line(img_wpd_cpy, pts_wpd[0], pts_wpd[1], (0, 0, 255), thickness=2)
        cv2.line(img_wpd_cpy, pts_wpd[0], pts_wpd[2], (0, 0, 255), thickness=2)
        cv2.line(img_wpd_cpy, pts_wpd[1], pts_wpd[3], (0, 0, 255), thickness=2)
        cv2.line(img_wpd_cpy, pts_wpd[2], pts_wpd[3], (0, 0, 255), thickness=2)

        cv2.line(img_wpd_cpy, pts_out[0], pts_out[1], (255, 0, 0), thickness=2)
        cv2.line(img_wpd_cpy, pts_out[0], pts_out[2], (255, 0, 0), thickness=2)
        cv2.line(img_wpd_cpy, pts_out[1], pts_out[3], (255, 0, 0), thickness=2)
        cv2.line(img_wpd_cpy, pts_out[2], pts_out[3], (255, 0, 0), thickness=2)
        cv2.putText(img_wpd_cpy, 'Avg Corner Error: {}'.format('%.3f'%pt_error_avg),
                    (15, 15), font, 0.4, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

        cv2.imwrite('./results/{}_orig.tiff'.format(num_iter), img_cpy1)
        cv2.imwrite('./results/{}_patch_orig.tiff'.format(num_iter), original_patch_c)
        cv2.imwrite('./results/{}_wpd.tiff'.format(num_iter), img_wpd_cpy)
        cv2.imwrite('./results/{}_patch_wpd.tiff'.format(num_iter), warped_patch_c)

    return pt_error_avg


def main():
    fn = open('./results/results.txt', 'w')

    net = nib.NibsNet1()
    cp = torch.load(dir_model + 'best_val_model.tar')
    net.load_state_dict(cp['model_state_dict'])
    net.eval()

    data_lst = glob(dir_data + '*.jpg')

    err = 0
    num_iter = 0

    for i in range(0, 1000):
        num_iter += 1
        err += testNibsNet1(data_lst, net, fn, num_iter)

    MACE = err / num_iter

    print('FINAL MACE: {}'.format(MACE))
    fn.write('FINAL MACE: {}'.format(MACE))
    fn.close()



if __name__ == '__main__':
    main()


