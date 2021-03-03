import argparse
import os
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from dataset import get_train_val_loaders
from train_common import *
from utils import *


best_acc1 = 0
random.seed(0)
torch.manual_seed(0)


def train_epoch(data_loader, model, criterion, optimizer, device):
    """
    Train the `model` for one epoch of data from `data_loader`
    Use `optimizer` to optimize the specified `criterion`.
    """
    model.train()

    for i, (X, y) in enumerate(data_loader, 0): # 0 indicates 0-index
        # get the inputs
        # inputs, labels = data
        # inputs, labels = inputs.to(device), labels.to(device)
        X, y = X.to(device), y.to(device)

        # clear parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()


def evaluate_epoch(axes, tr_loader, val_loader, model, criterion, epoch,
    stats, device):
    """
    Evaluates the `model` on the train and validation set.
    """
    model.eval()

    y_true, y_pred = [], []
    correct, total = 0, 0
    running_loss = []
    
    for X, y in tr_loader:
        X, y = X.to(device), y.to(device)

        with torch.no_grad():
            output = model(X)
            predicted = predictions(output.data)

            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())

    train_loss = np.mean(running_loss)
    train_acc = correct / total

    y_true, y_pred = [], []
    correct, total = 0, 0
    running_loss = []

    for X, y in val_loader:
        X, y = X.to(device), y.to(device)

        with torch.no_grad():
            output = model(X)
            predicted = predictions(output.data)

            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())

    val_loss = np.mean(running_loss)
    val_acc = correct / total
    stats.append([val_acc, val_loss, train_acc, train_loss])
    log_training(epoch, stats)
    update_training_plot(axes, epoch, stats)


def main_worker(device):
    # create model
    if config('pretrained'):
        print("=> using pre-trained model '{}'".format(config('arch')))
        model = models.__dict__[config('arch')](pretrained=True)
    else:
        print("=> creating model '{}'".format(config('arch')))
        model = models.__dict__[config('arch')]()

    # Transfer learning: pretrained model as fixed feature extractor
    for param in model.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    model = model.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config('lr'))
    # optimizer = torch.optim.SGD(model.parameters(), config('lr'), momentum=config('momentum'), weight_decay=config('weight_decay'))

    print('Number of float-valued parameters:', count_parameters(model))

    # Automate the training for 21 binary classifiers.
    data_dirs = ['n01818515', 'n02123045', 'n02981792', 'n03100240', 'n03594945', 'n03642806', 'n03769881', 'n03770679', 'n03791053', 'n03792782', 'n03977966', 'n03868863', 'n03902125', 'n03930630', 'n04399382', 'n04456115', 'n06794110', 'n06874185', 'n07583066', 'n07614500', 'n07753592']

    sementic_labels = ['macaw', 'tabby', 'catamaran', 'convertible', 'jeep', 'laptop', 'minibus', 'minivan', 'scooter', 'mountain_bike', 'police_van', 'oxygen_mask', 'pay-phone', 'pickup', 'teddy', 'torch', 'street_sign', 'traffic_light', 'guacamole', 'icecream', 'banana']

    for i, data_dir in enumerate(data_dirs):
        class_name = sementic_labels[i]
        data_path = os.path.join(config("data_path"), data_dir)
        
        tr_loader, val_loader = get_train_val_loaders(data_path, num_classes=config('num_classes'))

        if config('test'):
            evaluate_epoch(val_loader, model, criterion, device)
            return

        plot_title = config("arch") + " training (" + class_name + ")"
        axes = make_training_plot(plot_title)
        tr_start_time = time.time()
        stats = []

        for epoch in range(config('start_epoch'), config("epoch")):
            # train for one epoch
            train_epoch(tr_loader, model, criterion, optimizer, device)

            # evaluate on validation set
            evaluate_epoch(axes, tr_loader, val_loader, model, criterion, epoch+1, stats, device)

            checkpoint_path = os.path.join(config('checkpoint_path'), class_name)
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)

            save_model(model, epoch+1, checkpoint_path)
        
        tr_end_time = time.time()
        print('Training Time: ', tr_end_time - tr_start_time)
        print('Finished Training')

        plot_path = os.path.join(config('plot_path'), class_name)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        save_training_plot(plot_path)


def main():
    torch.cuda.set_device(get_emptiest_gpu())
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    ''' Only have 1 worker'''
    main_worker(device)


if __name__ == '__main__':
    main()
