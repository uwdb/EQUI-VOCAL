"""
Train CNN
    Trains a convolutional neural network to classify images
    Periodically outputs training information, and saves model checkpoints
    Usage: python train_cnn.py
"""
import json
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import copy

from lib_cnn import CNN
from dataset import get_train_val_loaders
from lottery import *
from train_common import *
import utils
from utils import config

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def get_emptiest_gpu():
	# Command line to find memory usage of GPUs.
    # Return the one with most mem available.
	output = os.popen('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free').read().strip()
	mem_avail = [int(x.split()[2]) for x in output.split('\n')]

	return mem_avail.index(max(mem_avail))

def _train_epoch(data_loader, model, criterion, optimizer, device):
    """
    Train the `model` for one epoch of data from `data_loader`
    Use `optimizer` to optimize the specified `criterion`
    """
    model.train()

    for i, (inputs, labels) in enumerate(data_loader, 0): # 0 indicates 0-index
        # put inputs and labels on GPU
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def _train_epoch_kd(teacher_data_loader, data_loader, teacher, student, criterion, optimizer, device):
    """
    Train the `model` for one epoch of data from `data_loader`
    Use `optimizer` to optimize the specified `criterion`
    """
    teacher.eval()
    student.train()

    for (inputs, labels), (teacher_inputs, _) in zip(data_loader, teacher_data_loader):
        inputs, labels, teacher_inputs = inputs.to(device), labels.to(device), teacher_inputs.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            # logits_t = teacher(teacher_inputs)
            logits_t = teacher(inputs)
        
        logits_s = student(inputs)
        
        loss = criterion(input=F.log_softmax(logits_s/config('cnn.temperature'), dim=1), target=F.softmax(logits_t/config('cnn.temperature'), dim=1)) * config('cnn.alpha') * config('cnn.temperature') * config('cnn.temperature') + F.cross_entropy(logits_s, labels) * (1 - config('cnn.alpha'))
        
        loss.backward()
        optimizer.step()

def _evaluate_epoch(val_loader, model, criterion, device):
    """
    Evaluates the 'model' on the train and validation set.
    """
    model.eval()
    
    y_true, y_pred = [], []
    pred_time = 0
    correct, total = 0, 0
    running_loss = []

    for X, y in val_loader:
        pred_start = time.time()
        
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            output = model(X)
            predicted = predictions(output.data)

            pred_end = time.time()
            pred_time += (pred_end - pred_start)

            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())

    val_loss = np.mean(running_loss)
    val_acc = correct / total
    infer_fps = total / pred_time
    print("Predicted %d items in %.3f seconds. Infer fps: %.2f." % (total, pred_time, infer_fps))

    return val_acc, val_loss, y_pred, y_true, infer_fps


def _evaluate_epoch_kd(teacher_data_loader, val_loader, teacher, student, criterion, device, standardizer=None):
    """
    Evaluates the `model` on the train and validation set.
    """
    # To compute load times, we will save images to a tmp dir and then read
    # them all. This is to simulate loading them after preprocessing has been
    # done as an offline process.
    teacher.eval()
    student.eval()

    y_true, y_pred = [], []
    pred_time = 0
    correct, total = 0, 0
    running_loss = []

    for (X, y), (X_teacher, _) in zip(val_loader, teacher_data_loader):
        pred_start = time.time()

        X, y, X_teacher = X.to(device), y.to(device), X_teacher.to(device)

        with torch.no_grad():
            output = student(X)
            batch_prob = F.softmax(output.data, dim=1)
            pred_prob = batch_prob[:, 1]
            # print('pred_prob', pred_prob)
            predicted = predictions(output.data)

            pred_end = time.time()
            pred_time += (pred_end - pred_start)

            y_true.append(y)
            y_pred.append(pred_prob)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            # knowledge distillation
            # logits_t = teacher(X_teacher)
            logits_t = teacher(X)

            loss = criterion(input=F.log_softmax(output/config('cnn.temperature'), dim=1), target=F.softmax(logits_t/config('cnn.temperature'), dim=1)) * config('cnn.alpha') * config('cnn.temperature') * config('cnn.temperature') + F.cross_entropy(output, y) * (1 - config('cnn.alpha'))

            running_loss.append(loss.item())

    val_loss = np.mean(running_loss)
    val_acc = correct / total
    infer_fps = total / pred_time
    print("Predicted %d items in %.3f seconds. Infer fps: %.2f." % (total, pred_time, infer_fps))

    return val_acc, val_loss, y_pred, y_true, infer_fps


def train_cnn(model_dir, prefix, device, image_dim):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    layer_opts = [[16],[16,16],[16,16,16,16], [32],[32,32],[32,32,32,32]]
    dense_opts = [[32, 16], [64, 16], [128, 16]]
    # size_opts = [[image_dim, 0], [image_dim, 1], [image_dim, 2], [image_dim, 3], [30,30], [60,60], [120,120], [224,224]]
    size_opts = [[30,30], [60,60], [120,120], [224,224]]
    processors = ["ColorImage", "BWImage", "RedChannel", "GreenChannel", "BlueChannel"]

    # Reset for knowledge distillation.
    # size_opts = [[224,224]]
    # processors = ["ColorImage"]

    # teacher_data_dir = "./tmp/" + prefix + "/FAST1/labeled_images/"

    # teacher_tr_loader, teacher_va_loader, _, _ = get_train_val_loaders(num_classes=2,data_dir=teacher_data_dir, color_mode='ColorImage', image_dim=[224,224], batch_size=config('cnn.batch_size'))
    
    info_models_data = {}
    training_weight = []

    with open('./tmp/' + prefix + '/FAST1/labeled_images/training_weight.json', 'r') as f:
        data = json.load(f)
        training_weight = data['training_weight']

    with open(os.path.join(model_dir, 'validation_acc.txt'), 'w') as fout:
        for input_shape in size_opts:
            res_mean = [0,0,0,0]
            res_std = [0,0,0,0]

            for preprocessor in processors:
                # Data loaders
                print("loading data...")
                data_dir = "./tmp/" + prefix + "/FAST1/labeled_images/"
                tr_loader, va_loader, _, standardizer = get_train_val_loaders(num_classes=2, data_dir=data_dir, color_mode=preprocessor, image_dim=input_shape, batch_size=config('cnn.batch_size'))

                if preprocessor == 'ColorImage':
                    res_mean[0:3] = standardizer.image_mean # In BGR order
                    res_std[0:3] = standardizer.image_std
                    # print('res_mean', res_mean)
                    # print('res_std', res_std)
                elif preprocessor == 'BWImage':
                    res_mean[3] = standardizer.image_mean[0]
                    res_std[3] = standardizer.image_std[0]
                    # print('res_mean', res_mean)
                    # print('res_std', res_std)

                for cnn_layer in layer_opts:
                    for fc_layer in dense_opts:
                        dropout_rate = config("cnn.dropout_rate")
                        # process = processor(input_shape[0], input_shape[1])
                        train_start_time = time.time()
                        
                        suffix = '-'.join(map(str,cnn_layer)) + "_" + '-'.join(map(str,fc_layer)) + "_" + str(dropout_rate) + "_" + '-'.join(map(str,input_shape)) + "_" + preprocessor
                        # uniquely define the model name
                        model_file_name = os.path.join(model_dir, prefix + '_' + suffix + '.pth.tar')

                        if os.path.exists(model_file_name):
                            continue

                        attempt_num = 0
                        acc = 0.5

                        while attempt_num < 10 and acc <= 0.5:
                            attempt_num += 1
                            print('---------', preprocessor, cnn_layer, fc_layer, input_shape, 'attempt:', attempt_num,'---------')
                            
                            # Model
                            model = CNN(input_shape, preprocessor, cnn_layer, fc_layer, dropout_rate)
                            model.to(device)

                            # Reset for lottery
                            # initial_state_dict = copy.deepcopy(model.state_dict())

                            criterion = torch.nn.CrossEntropyLoss()
                            # Reset for knowledge distillation.
                            # criterion = torch.nn.KLDivLoss()
                            optimizer = torch.optim.Adam(model.parameters(), lr=config('cnn.learning_rate'))

                            print('Loading cnn...')

                            # Loop over the entire dataset multiple times
                            best_loss = np.inf
                            best_acc = 0

                            for epoch in range(config('cnn.num_epochs')):
                                # Train model
                                _train_epoch(tr_loader, model, criterion, optimizer, device)
                                # Reset for knowledge distillation.
                                # _train_epoch_kd(teacher_tr_loader, tr_loader, teacher, model, criterion, optimizer, device)

                                # Evaluate model
                                val_acc, val_loss, _, _, _ = _evaluate_epoch(va_loader, model, criterion, device)
                                # Reset for knowledge distillation.
                                # val_acc, val_loss, _, _, _ = _evaluate_epoch_kd(teacher_va_loader, va_loader, teacher, model, criterion, device)

                                print("val_acc", val_acc)
                                print("val_loss", val_loss)

                                # if val_loss < best_loss: 
                                #     best_loss = val_loss
                                #     acc = val_acc
                                #     save_checkpoint(model, epoch, best_loss, model_file_name)

                                if best_acc < val_acc: 
                                    best_acc = val_acc
                                    acc = val_acc
                                    save_checkpoint(model, epoch, val_loss, model_file_name)

                            if acc > 0.5:
                                # Reset for lottery
                                # acc = lottery(model, tr_loader, va_loader, initial_state_dict, model_file_name, device)
                                fout.write(str(attempt_num) + "\t" + prefix + "_" + suffix + "\t" + str(acc) + "\n")
                        train_end_time = time.time()
                        print("Training time: %.3f s" % (train_end_time - train_start_time))

            info_models_data['x'.join([str(a) for a in input_shape])] = {'mean': res_mean, 'std': res_std}

    with open("./tmp/" + prefix + "/info_models.json", "w") as f:
        json.dump(info_models_data, f)
    
if __name__ == '__main__':
    train_cnn()


