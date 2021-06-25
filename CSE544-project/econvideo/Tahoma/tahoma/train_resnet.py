import json
import os
import shutil
import time

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from dataset import get_train_val_loaders
from train_common import *
import utils
from utils import config


def get_emptiest_gpu():
	# Command line to find memory usage of GPUs. Return the one with most mem available.
	output = os.popen('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free').read().strip()
	mem_avail = [int(x.split()[2]) for x in output.split('\n')]
	return mem_avail.index(max(mem_avail))

def _train_epoch(data_loader, model, criterion, optimizer, device):
    """
    Train the 'model' for one epoch of data from 'data_loader'.
    Use 'optimizer' to optimize the specified 'criterion'.
    """
    model.train()

    for i, data in enumerate(data_loader, 0): # 0 indicates 0-index
        # Get the inputs and pass them into GPU.
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Clear parameter gradients.
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def _evaluate_epoch(axes, tr_loader, val_loader, model, criterion, epoch,
    stats, device):
    """
    Evaluates the 'model' on the train and validation set.
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
    utils.log_cnn_training(epoch, stats)
    utils.update_cnn_training_plot(axes, epoch, stats)

def main():
    # Get the emptiest GPU.
    torch.cuda.device(get_emptiest_gpu())

    # Set up the device.
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    """
    To do: fix the data path, and add early stopping.
    """

    # Model hyper-parameters.
    batch_size = config('resnet.batch_size')
    learning_rate = config('resnet.learning_rate')
    num_epochs = config('resnet.num_epochs')
    num_classes = config('resnet.num_classes')

    # Fine-tune a pretrained model.
    model = models.resnet50(pretrained=False)
    # Freeze model parameters.
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # Retrieve the model output size before the fully collected layer.
    fc_input_size = model.fc.in_features
    # Repalce the original fully connected layer with customized layers. 
    # model.fc = nn.Sequential(OrderedDict([
    #     ('fc1', nn.Linear(fc_input_size, 64)),
    #     ('fc1_relu', nn.ReLu()),
    #     ('fc2', nn.Linear(64, num_classes))
    #     #('fc2_softmax', nn.Softmax(dim=1))
    # ]))
    model.fc = nn.Sequential(
        nn.Linear(fc_input_size, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes))

    # model.fc = nn.Linear(fc_input_size, num_classes)
    

    # Pass the model into GPU.
    model = model.to(device)

    # Set up the criterion and the optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Attempts to restore the latest checkpoint if exists.
    # print('Loading resnet...')
    # model, start_epoch, stats = restore_checkpoint(model, config('cnn.checkpoint'))

    data_dir = "../is_winter_resnet_0908/FAST1/labeled_images/"

    tr_loader, va_loader, _, standardizer = get_train_val_loaders(num_classes=num_classes, data_dir=data_dir, color_mode='ColorImage', image_dim=[224,224], batch_size=batch_size)

    # Start training
    axes = utils.make_cnn_training_plot()
    stats = []

    # Loop over the entire dataset multiple times
    for epoch in range(num_epochs):
        # Train model
        _train_epoch(tr_loader, model, criterion, optimizer, device)

        # Evaluate model
        _evaluate_epoch(axes, tr_loader, va_loader, model, criterion, epoch+1,
            stats, device)

        # Save model parameters
        # save_checkpoint(model, epoch+1, config('resne.checkpoint'), stats)

    # Save figure and keep plot open
    utils.save_cnn_training_plot()

if __name__ == '__main__':
    main()
