# BME595 Project
# Author: David Niblick
# Date: 05DEC18
# main.py


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
import data_prep
import nibs_homography_net as nib


# Set FLAGS
FLAG_USE_GPU = True
FLAG_LOAD_CP = True
FLAG_LOAD_BEST_MODEL = False
FLAG_TRAIN = True
FLAG_DEBUG = False

# Set Hyper Parameters
batch_sz = 64
init_learn_rt = 0.005
learn_rt_decay_epoch = 10
wt_decay = 0.001355
momentum = 0.9

# Set folder directories
dir_model = './model/'
dir_data = '../MSCOCO/unlabeled2017/'


def train(net, device, loader_train, optimizer, loss_fn, epoch, log_interval=10):
    net.train().to(device=device)
    loss_epoch = 0

    for batch_idx, (data, target) in enumerate(loader_train):
        data, target = data.to(device), target.to(device=device)
        optimizer.zero_grad()
        out = net(data)
        loss = loss_fn(out, target)
        loss_epoch += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss for batch: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader_train.dataset),
                100.0 * batch_idx / len(loader_train), loss.item()
            ), end='')

    return loss_epoch / len(loader_train)


def validate(net, device, loader_val, loss_fn, epoch, log_interval=10):
    net.eval().to(device=device)
    loss_epoch = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader_val):
            data, target = data.to(device=device), target.to(device=device)
            out = net(data)
            loss = loss_fn(out, target)
            loss_epoch += loss.item()
            if batch_idx % log_interval == 0:
                print('\rValidate Epoch: {} [{}/{} ({:.0f}%)]\tLoss for batch: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(loader_val.dataset),
                    100.0 * batch_idx / len(loader_val), loss.item()
                ), end='')

    return loss_epoch / len(loader_val)


def exp_lr_scheduler(optimizer, epoch, init_lr, lr_decay_epoch):

    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer, lr


def main():

    print('Welcome to NibsNet1!')

    # Prepare GPU
    if torch.cuda.is_available() and FLAG_USE_GPU:
        device = torch.device('cuda:0')
        pin_mem = True
        workers = 8
    else:
        device = torch.device('cpu')
        pin_mem = False
        workers = 0
    print('Device = {}'.format(device))

    # Prepare Data
    print('Preparing Data...')
    if FLAG_DEBUG:
        print('Using dataset for debugging purposes...')
        dataset_train, dataset_val = data_prep.build_datasets_debugging(dir_data)
    else:
        dataset_train, dataset_val = data_prep.build_datasets(dir_data)

    loader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=batch_sz,
                                               shuffle=False,
                                               pin_memory=pin_mem,
                                               num_workers=workers,
                                               drop_last=True)

    loader_val = torch.utils.data.DataLoader(dataset=dataset_val,
                                             batch_size=batch_sz,
                                             shuffle=False,
                                             pin_memory=pin_mem,
                                             num_workers=workers,
                                             drop_last=True)

    # Manually set seed for consistent initialization
    # torch.manual_seed(1)

    # Initialize Neural Network
    print('Initializing NibsNet1...')
    net = nib.NibsNet1().to(device=device)

    # Initialize Optimizer and Loss Function
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), init_learn_rt, momentum=0.9, weight_decay=wt_decay)

    # Initialize running counters and helpful variables
    epoch = 0
    train_loss = []
    val_loss = []
    epoch_time = []
    best_model_loss = 10000

    # If continuing previous training
    if FLAG_LOAD_BEST_MODEL:
        print('Loading model and parameters from best model...')
        cp = torch.load(dir_model+'best_model.tar')
        net.load_state_dict(cp['model_state_dict'])
        optimizer.load_state_dict(cp['optimizer_state_dict'])
        epoch = cp['epoch']
        train_loss = cp['train_loss']
        val_loss = cp['val_loss']
        epoch_time = cp['epoch_time']
        learn_rt = cp['learn_rt']
        best_model_loss = cp['best_model_loss']
    elif FLAG_LOAD_CP:
        print('Loading model and parameters from previous training...')
        cp = torch.load(dir_model+'cp.tar')
        net.load_state_dict(cp['model_state_dict'])
        optimizer.load_state_dict(cp['optimizer_state_dict'])
        epoch = cp['epoch']
        train_loss = cp['train_loss']
        val_loss = cp['val_loss']
        epoch_time = cp['epoch_time']
        learn_rt = cp['learn_rt']
        best_model_loss = cp['best_model_loss']
    else:
        print('Starting with fresh model...')

    # Train and Validate
    print("Begin Training and Validation...")
    while epoch < 500:

        epoch += 1
        optimizer, learn_rt = exp_lr_scheduler(optimizer, epoch,
                                               init_lr=init_learn_rt, lr_decay_epoch=learn_rt_decay_epoch)
        start_time = time.time()
        train_loss_epoch = train(net, device, loader_train, optimizer, loss_fn, epoch)
        print('')
        train_loss.append(train_loss_epoch)
        val_loss_epoch = validate(net, device, loader_val, loss_fn, epoch)
        val_loss.append(val_loss_epoch)
        end_time = time.time()
        epoch_time.append(end_time - start_time)
        print('\nEpoch: {}  \tTime {} \tLearn Rate: {} \tTrain Loss: {} \tVal Loss: {}'.format(epoch,
                                                                                               end_time - start_time,
                                                                                               learn_rt,
                                                                                               train_loss_epoch,
                                                                                               val_loss_epoch))
        print('')

        # Save model and metrics
        torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'epoch_time': epoch_time,
            'learn_rt': learn_rt,
            'best_model_loss': best_model_loss
        }, dir_model+'cp.tar'.format(epoch))

        if val_loss_epoch < best_model_loss:
            best_model_loss = val_loss_epoch
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch_time': epoch_time,
                'learn_rt': learn_rt,
                'best_model_loss': best_model_loss
            }, dir_model + 'best_model.tar'.format(epoch))


if __name__ == '__main__':
    main()
