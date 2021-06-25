import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import utils
import sys
from utils import config 
from torch.utils.data import Dataset, DataLoader, Subset, random_split, TensorDataset
from lstm import LSTM
from train_common import *
from dataset import get_train_val_loaders, get_train_val_loaders_no_sampling
import time
import json
import matplotlib.pyplot as plt
import csv
from math import floor
from lstm import *
import re

torch.manual_seed(1)
eps = 1e-12
bin_mid = []
# These will usually be more like 32 or 64 dimensional.
EMBEDDING_DIM = config("lstm.embedding_dim")
HIDDEN_DIM = config("lstm.hidden_dim")
BATCH_SIZE = config("lstm.batch_size")


def get_emptiest_gpu():
	# Command line to find memory usage of GPUs. Return the one with most mem available.
	output = os.popen('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free').read().strip()
	mem_avail = [int(x.split()[2]) for x in output.split('\n')]
	return mem_avail.index(max(mem_avail))

def _train_epoch(tr_loader, model, criterion, optimizer, device):
    model.train()
    for i, (description, length, labels, true_value, _) in enumerate(tr_loader, 0): # 0 indicates 0-index
        description, length, labels = description.to(device), length.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(description, length)
        # loss = criterion(torch.log(outputs.squeeze() + eps), torch.log(labels + eps))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def _evaluate_epoch(axes, tr_loader, val_loader, model, criterion, epoch, stats, device):
    model.eval()
    y_true, y_pred = [], []
    correct, total, correct_near = 0, 0, 0
    running_loss = []
    y_pred_mid, y_true_values = [], []
    for i, (X, length, y, true_values, _) in enumerate(tr_loader, 0):
        X, length, y = X.to(device), length.to(device), y.to(device)
        with torch.no_grad():
            output = model(X, length)
            predicted = predictions(output.data)
            y_true.append(y)
            y_pred.append(predicted)
            y_pred_mid.extend([bin_mid[bin] for bin in predicted])
            y_true_values.extend(true_values.tolist())
            # print(y_true_values)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            correct_near += (torch.abs(predicted - y) <= 1).sum().item()
            running_loss.append(criterion(output, y).item())
    train_loss = np.mean(running_loss)
    train_acc = correct / total
    train_acc_near = correct_near / total
    train_mse = ((np.array(y_pred_mid) - np.array(y_true_values))**2).mean()
    print("MSE loss for training set is: ", train_mse)

    y_true, y_pred = [], []
    correct, total, correct_near = 0, 0, 0
    running_loss = []
    y_pred_mid, y_true_values = [], []
    for i, (X, length, y, true_values, _) in enumerate(val_loader, 0):
        X, length, y = X.to(device), length.to(device), y.to(device)
        with torch.no_grad():
            output = model(X, length)
            predicted = predictions(output.data)
            y_true.append(y)
            y_pred.append(predicted)
            y_pred_mid.extend([bin_mid[bin] for bin in predicted])
            y_true_values.extend(list(true_values))
            total += y.size(0)
            correct += (predicted == y).sum().item()
            correct_near += (torch.abs(predicted - y) <= 1).sum().item()
            running_loss.append(criterion(output, y).item())
    val_mse = ((np.array(y_pred_mid) - np.array(y_true_values))**2).mean()
    print("MSE loss for valisation set is: ",val_mse)

    
    # Accuracy for each class
    class_correct = list(0. for i in range(config("lstm.num_classes")))
    class_correct_near = list(0. for i in range(config("lstm.num_classes")))
    class_total = list(0. for i in range(config("lstm.num_classes")))
    with torch.no_grad():
        for _, (X, length, y, true_values, _) in enumerate(val_loader, 0):
            X, length, y = X.to(device), length.to(device), y.to(device)
            output = model(X, length)
            predicted = predictions(output.data)
            c = (predicted == y)
            c_near = (torch.abs(predicted - y) <= 1)
            for i in range(y.size(0)):
                label = y[i]
                class_correct[label] += c[i].item()
                class_correct_near[label] += c_near[i].item()
                class_total[label] += 1
    # for i in range(config("lstm.num_classes")):
    #     if class_total[i] > 0:
    #         print('Accuracy of y = %d : %f %%' % (i, 100 * class_correct[i] / class_total[i]))
    #         print('Accuracy (near) of y = %d : %f %%' % (i, 100 * class_correct_near[i] / class_total[i]))
    val_loss = np.mean(running_loss)
    val_acc = correct / total
    val_acc_near = correct_near / total
    stats.append([val_acc, val_acc_near, val_loss, train_acc, train_acc_near, train_loss])
    utils.log_lstm_training(epoch, stats)
    utils.update_lstm_training_plot(axes, epoch, stats)
    return val_acc, val_acc_near, correct, correct_near, total

def read_bin_mid(category_idx):
    df = pd.read_csv(os.path.join(config("bin_range_csv"), str(category_idx) + '.csv'), delimiter = ',')
    for index, row in df.iterrows():
        bin_mid.append((row["begin"] + row["end"]) / 2)
    print(bin_mid)
    
def output_mjc_csv(inverted_dict, data_loader, model, device, in_sample, category_idx):
    train_dir_name = 'results/intermediate_results/all_products_train/'
    val_dir_name = 'results/intermediate_results/all_products/'
    if not os.path.exists(train_dir_name):
        os.makedirs(train_dir_name)
    if not os.path.exists(val_dir_name):
        os.makedirs(val_dir_name)
    # Create a new csv file to record data statistics
    count = 0
    if in_sample == 1:
        file_name = train_dir_name + str(category_idx) + '.csv'
    else:
        file_name = val_dir_name + str(category_idx) + '.csv'

    with open(file_name, mode='w') as csv_file:
        field_names = ['index', 'Desc', 'True Price', 'Predicted Price', 'In Sample', 'True Bin', 'Top1 Predicted Bin', 'Top2 Predicted Bin']
        csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
        csv_writer.writeheader()

        model.eval()
        for i, (X, length, y, true_values, index_values) in enumerate(data_loader, 0):
            X, length, y = X.to(device), length.to(device), y.to(device)

            with torch.no_grad():
                output = model(X, length)
                topk_predicted = topk_pred(output.data, k=2)
                # print(type(topk_predicted))
                # print("prob", topk_predicted[0])
                # print("indices", topk_predicted[1])

                top1_indices = [ row[0].item() for row in topk_predicted[1] ]
                top1_probs = [ row[0].item() for row in topk_predicted[0] ]
                top2_indices = [ row[1].item() for row in topk_predicted[1] ]
                top2_probs = [ row[1].item() for row in topk_predicted[0] ]

                for j in range(y.size(0)):
                    count += 1
                    top1_mid_price = bin_mid[top1_indices[j]]
                    top2_mid_price = bin_mid[top2_indices[j]]
                    top1_weight = top1_probs[j] / (top1_probs[j] + top2_probs[j])
                    top2_weight = top2_probs[j] / (top1_probs[j] + top2_probs[j])
                    pred_price = (top1_weight * top1_mid_price + top2_weight * top2_mid_price)

                    csv_writer.writerow({'index': index_values[j].item(),
                                         'Desc': " ".join([inverted_dict[idx.item()] for idx in X[j] if idx.item() != 0]), 
                                         'True Price': true_values[j].item(),
                                         'Predicted Price': pred_price,
                                         'In Sample': in_sample,
                                         'True Bin': int(y[j].item()), 
                                         'Top1 Predicted Bin': top1_indices[j], 
                                         'Top2 Predicted Bin': top2_indices[j]
                                        })
    print(f"Epoch example count: {count}")

def train_lstm(category_idx, error_analysis_flag):
    # Set emptiest GPU
    read_bin_mid(category_idx)
    torch.cuda.set_device(get_emptiest_gpu())

    # Training on GPU
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    if not error_analysis_flag:
        tr_loader, va_loader, word_len, word_dict= get_train_val_loaders(category_idx)
    else:
        test_loader, tr_loader, word_len, word_dict = get_train_val_loaders_no_sampling(category_idx) # Predict all products, without sampling 
    model = LSTM(EMBEDDING_DIM, HIDDEN_DIM, word_len, BATCH_SIZE, 0.2, 0.8, 0.6)

    print('Loading lstm...')
    if not os.path.exists(os.path.join(config('lstm.checkpoint'), str(category_idx))):
        os.makedirs(os.path.join(config('lstm.checkpoint'), str(category_idx)))
    # model, start_epoch, stats = restore_checkpoint(model, os.path.join(config('lstm.checkpoint'), str(category_idx)))
    start_epoch = 0
    stats = []

    # Start training
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = config('lstm.learning_rate'))
    # optimizer = torch.optim.SGD(model.parameters(), lr = config('lstm.learning_rate'), momentum=0.6)
    a = time.time()
    axes = utils.make_lstm_training_plot()
    # Error analysis only
    if error_analysis_flag:
        print('start error analysis...')
        # upc_validation_set(va_loader, category_idx)
        checkpoint_path = os.path.join(config('lstm.checkpoint'), f"{category_idx}", f"epoch={25}.checkpoint.pth.tar")
        checkpoint = torch.load(checkpoint_path)

        try:
            start_epoch = checkpoint['epoch']
            stats = checkpoint['stats']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> Successfully restored checkpoint (trained for {} epochs)".format(checkpoint['epoch']))
        except:
            print("=> Checkpoint not successfully restored")
            raise
        inverted_dict = dict([[v,k] for k,v in word_dict.items()])
        output_mjc_csv(inverted_dict, tr_loader, model, device, 1, category_idx)
        output_mjc_csv(inverted_dict, test_loader, model, device, 0, category_idx)
        return

    # Evaluate the randomly initialized model
    _evaluate_epoch(axes, tr_loader, va_loader, model, criterion, start_epoch, stats, device)
        
    for epoch in range(start_epoch, config("lstm.num_epochs")):
        _train_epoch(tr_loader, model, criterion, optimizer, device)

        _evaluate_epoch(axes, tr_loader, va_loader, model, criterion, epoch+1, stats, device)
        
        save_checkpoint(model, epoch+1, os.path.join(config('lstm.checkpoint'), str(category_idx)), stats)

    b = time.time()
    print('Training Time: ', b - a)
    print('Finished Training')

    # Save figure and keep plot open
    utils.save_lstm_training_plot(category_idx)

if __name__ == '__main__':
    print("Step 4 begins...")
    error_analysis_flag = int(sys.argv[1])
    predictor_idx = int(sys.argv[2])
    # for i in range(28, 461):
    # for i in [31]:
    print('Start training category #', predictor_idx)
    bin_mid = []
    train_lstm(predictor_idx, error_analysis_flag)