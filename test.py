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

net = nib.NibsNet1()
optimizer = torch.optim.SGD(net.parameters(), learn_rt, momentum=0.9, weight_decay=wt_decay)


cp = torch.load(dir_model + 'cp.tar')
net.load_state_dict(cp['model_state_dict'])
optimizer.load_state_dict(cp['optimizer_state_dict'])
epoch = cp['epoch']
train_loss = cp['train_loss']
val_loss = cp['val_loss']
epoch_time = cp['epoch_time']
learn_rt = cp['learn_rt']
best_model_loss = cp['best_model_loss']

print('Current Epoch: {}'.format(epoch))
print('Best loss achieved: {}'.format(best_model_loss))

plt.figure(0)
plt.plot(train_loss)

plt.plot(val_loss)
plt.ylabel('Error')
plt.xlabel('Epochs')
plt.legend(['Training', 'Validation'])
plt.suptitle('Validate and Training loss')

plt.figure(1)
plt.plot(epoch_time)
plt.ylabel('Time (s)')
plt.xlabel('Epochs')
plt.suptitle('Train and Validation Time per Epoch')

plt.figure(2)
plt.plot(learn_rt)
plt.ylabel('Learn Rate')
plt.xlabel('Epochs')
plt.suptitle('Learn Rate Adjustments per Epoch')

plt.show()

