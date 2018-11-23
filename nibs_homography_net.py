# BME595 Project
# Author: David Niblick
# Date: 05DEC18
# nibs_homography_net.py


import cupy as cp
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import random
from glob import glob
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as fn


class NibsNet1(nn.Module):
    def __init__(self):
        super(NibsNet1, self).__init__()
        # Input img size [128, 128, 2]
        self.conv1 = nn.Conv2d(2, 64, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.bnorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bnorm3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.bnorm4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bnorm5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.bnorm6 = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128, 128, 3, padding=1)
        self.bnorm7 = nn.BatchNorm2d(128)

        self.conv8 = nn.Conv2d(128, 128, 3, padding=1)
        self.drop1 = nn.Dropout(p=0.5)
        self.bnorm8 = nn.BatchNorm1d(128*16*16)

        self.fc1 = nn.Linear(128*16*16, 1024)
        self.drop2 = nn.Dropout(p=0.5)
        self.bnorm9 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 8)

    def forward(self, x):
        x = fn.relu(self.conv1(x))
        x = self.bnorm1(x)

        x = fn.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.bnorm2(x)

        x = fn.relu(self.conv3(x))
        x = self.bnorm3(x)

        x = fn.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.bnorm4(x)

        x = fn.relu(self.conv5(x))
        x = self.bnorm5(x)

        x = fn.relu(self.conv6(x))
        x = self.pool3(x)
        x = self.bnorm6(x)

        x = fn.relu(self.conv7(x))
        x = self.bnorm7(x)

        x = fn.relu(self.conv8(x))
        x = x.view(-1, 128*16*16)
        x = self.drop1(x)
        x = self.bnorm8(x)

        x = torch.tanh(self.fc1(x))
        x = self.drop2(x)
        x = self.bnorm9(x)

        x = self.fc2(x)

        return x

