# BME595 Project
# Author: David Niblick
# Date: 05DEC18
# main.py

# Notes: Try ADAM again, normalize imgs in dataloader because loss should be way less, if that doesn't work
#   then try dividing loss by static number prior to feeding into optimizer
#   adjust the dropout/BN layers... ie remove BN after the first dropout as a paper says they create disharmony


import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import data_prep
import nibs_homography_net as nib


# Set FLAGS
FLAG_USE_GPU = True
FLAG_LOAD_CP = True
FLAG_LOAD_BEST_VAL_MODEL = False
FLAG_LOAD_BEST_TRAIN_MODEL = False
FLAG_TRAIN = True
FLAG_DEBUG = False

# Set Hyper Parameters
batch_sz = 64
init_learn_rt_adam = 0.002
learn_rt_decay_epoch = 20
wt_decay = 0.001355
momentum = 0.9

# Set folder directories
dir_model = './model/'
dir_data = '../MSCOCO/unlabeled2017/'
dir_metric = './metrics/'


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

        # print(loss_epoch)
        # print(len(loader_val))

    return loss_epoch / len(loader_val)


def exp_lr_scheduler(optimizer, epoch, init_lr, lr_decay_epoch):

    lr = init_lr * (0.5**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer, lr


def plot_metrics(train_loss, val_loss, epoch_time, learn_rt, epoch):

    plt.figure(0)
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.legend(['Train Loss, Val Loss'])
    plt.suptitle('Training vs Validation Loss')
    plt.savefig(dir_metric+'{}_train_val.tiff'.format(epoch))

    plt.figure(1)
    plt.plot(train_loss)
    plt.ylabel('Error')
    plt.xlabel('Epochs')
    plt.suptitle('Training loss')
    plt.savefig(dir_metric+'{}_train.tiff'.format(epoch))

    plt.figure(2)
    plt.plot(val_loss)
    plt.ylabel('Error')
    plt.xlabel('Epochs')
    plt.suptitle('Validate loss')
    plt.savefig(dir_metric+'{}_val.tiff'.format(epoch))

    plt.figure(3)
    plt.plot(epoch_time)
    plt.ylabel('Time (s)')
    plt.xlabel('Epochs')
    plt.suptitle('Train and Validation Time per Epoch')
    plt.savefig(dir_metric+'{}_time.tiff'.format(epoch))

    plt.figure(4)
    plt.plot(learn_rt)
    plt.ylabel('Learn Rate')
    plt.xlabel('Epochs')
    plt.suptitle('Learn Rate Adjustments per Epoch')
    plt.savefig(dir_metric+'{}learn_rt.tiff'.format(epoch))


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
    optimizer = torch.optim.Adam(net.parameters(), lr=init_learn_rt_adam, weight_decay=wt_decay)

    # Initialize running counters and helpful variables
    epoch = 0
    train_loss = []
    val_loss = []
    epoch_time = []
    learn_rt = []
    best_val_model_loss = 0
    best_train_model_loss = 0

    # If continuing previous training
    if FLAG_LOAD_BEST_VAL_MODEL:
        print('Loading model and parameters from best model...')
        cp = torch.load(dir_model+'best_val_model.tar')
        net.load_state_dict(cp['model_state_dict'])
        optimizer.load_state_dict(cp['optimizer_state_dict'])
        epoch = cp['epoch']
        train_loss = cp['train_loss']
        val_loss = cp['val_loss']
        epoch_time = cp['epoch_time']
        learn_rt = cp['learn_rt']
        best_val_model_loss = cp['best_val_model_loss']
        best_train_model_loss = cp['best_train_model_loss']
    elif FLAG_LOAD_BEST_TRAIN_MODEL:
        print('Loading model and parameters from best model...')
        cp = torch.load(dir_model + 'best_train_model.tar')
        net.load_state_dict(cp['model_state_dict'])
        optimizer.load_state_dict(cp['optimizer_state_dict'])
        epoch = cp['epoch']
        train_loss = cp['train_loss']
        val_loss = cp['val_loss']
        epoch_time = cp['epoch_time']
        learn_rt = cp['learn_rt']
        best_val_model_loss = cp['best_val_model_loss']
        best_train_model_loss = cp['best_train_model_loss']
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
        best_val_model_loss = cp['best_val_model_loss']
        best_train_model_loss = cp['best_train_model_loss']
    else:
        print('Starting with fresh model...')

    # Train and Validate
    print("Begin Training and Validation...")
    while epoch < 500:

        epoch += 1
        optimizer, learn_rt_epoch = exp_lr_scheduler(optimizer,
                                                     epoch,
                                                     init_lr=init_learn_rt_adam,
                                                     lr_decay_epoch=learn_rt_decay_epoch)
        start_time = time.time()
        train_loss_epoch = train(net, device, loader_train, optimizer, loss_fn, epoch)
        print('')
        val_loss_epoch = validate(net, device, loader_val, loss_fn, epoch)
        end_time = time.time()

        print('\nEpoch: {}  \tTime {} \tLearn Rate: {} \tTrain Loss: {} \tVal Loss: {}'.format(epoch,
                                                                                               end_time - start_time,
                                                                                               learn_rt_epoch,
                                                                                               train_loss_epoch,
                                                                                               val_loss_epoch))
        print('')

        # Save model and metrics
        train_loss.append(train_loss_epoch)
        val_loss.append(val_loss_epoch)
        epoch_time.append(end_time - start_time)
        learn_rt.append(learn_rt_epoch)
        plot_metrics(train_loss, val_loss, epoch_time, learn_rt, epoch)

        torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'epoch_time': epoch_time,
            'learn_rt': learn_rt,
            'best_train_model_loss': best_train_model_loss,
            'best_val_model_loss': best_val_model_loss
        }, dir_model+'cp.tar'.format(epoch))

        if val_loss_epoch < best_val_model_loss:
            best_val_model_loss = val_loss_epoch
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch_time': epoch_time,
                'learn_rt': learn_rt,
                'best_train_model_loss': best_train_model_loss,
                'best_val_model_loss': best_val_model_loss
            }, dir_model + 'best_val_model.tar'.format(epoch))

        if train_loss_epoch < best_train_model_loss:
            best_train_model_loss = train_loss_epoch
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch_time': epoch_time,
                'learn_rt': learn_rt,
                'best_train_model_loss': best_train_model_loss,
                'best_val_model_loss': best_val_model_loss
            }, dir_model + 'best_train_model.tar'.format(epoch))


if __name__ == '__main__':
    main()
