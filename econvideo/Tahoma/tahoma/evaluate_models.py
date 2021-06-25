import json
import os
import random
import time

import cv2
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import torch

from dataset import get_train_val_loaders
from lib_cnn import CNN
from train_cnn import _evaluate_epoch, _evaluate_epoch_kd
from train_common import *
import utils
from utils import config

CACHE_DATA = {}

def get_emptiest_gpu():
	# Command line to find memory usage of GPUs. Return the one with most mem available.
	output = os.popen('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free').read().strip()
	mem_avail = [int(x.split()[2]) for x in output.split('\n')]
	return mem_avail.index(max(mem_avail))

# def make_plot():
#     fps = np.load(config('root_dir') + result_folder + prefix + '_fps.npy')
#     load_fps = np.load(config('root_dir') + result_folder + prefix + '_load_fps.npy')
#     models = np.load(config('root_dir') + result_folder + prefix + '_models.npy')
#     acc = np.load(config('root_dir') + result_folder + prefix + '_acc.npy')
#     plot_types = models[0].keys()

#     for plot_type in plot_types:
#         num_models = 0

#         model_fps = {}
#         model_acc = {}
#         for i, m in enumerate(models):
#             plot_key = str(m[plot_type])

#             if plot_key not in model_fps:
#                 model_fps[plot_key] = []
#                 model_acc[plot_key] = []
#             model_fps[plot_key].append(1./(1./fps[i] + 1./load_fps[i]))
#             model_acc[plot_key].append(acc[i])


#         plt.figure(figsize=[8,6])
#         leg = []
#         for key in model_fps.keys():
#             plt.plot(model_fps[key], model_acc[key], linestyle='None', marker='o', alpha=0.7)
#             leg.append(key)
#             num_models += len(model_fps[key])


#         plt.title('FPS vs Fidelity - ' + plot_type)
#         plt.ylabel('Fidelity %')
#         plt.xlabel('Frames per second')
#         plt.legend(leg, loc='lower right')
#         if not os.path.exists(config('root_dir') + 'plots/'):
#             os.makedirs(config('root_dir') + 'plots/')

#         plt.savefig(config('root_dir') + 'plots/evaluate_models_'+ plot_type + '.png', dpi=200)
#     print("Number of models:", num_models)

def eval_models(result_dir, model_dir, prefix, image_dim, device):
    processors = ["ColorImage", "BWImage", "RedChannel", "GreenChannel", "BlueChannel"]

    # Set emptiest GPU
    #torch.cuda.device(get_emptiest_gpu())
    #torch.cuda.set_device(get_emptiest_gpu())

    # Training on GPU
    #device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    models = []
    infer_fps_vals = []
    load_fps_vals = []
    acc_vals = []
    results = []
    ground_truth = None
    cascade_results = []
    cascade_ground_truth = None
    count = 1
    training_weight = []

    with open('./tmp/' + prefix + '/FAST1/labeled_images/training_weight.json') as f:
        data = json.load(f)
        training_weight = data['training_weight']

    for f in os.listdir(model_dir):
        if 'pth.tar' not in f or prefix not in f:
            continue

        print('--------', f, '--------')
        print("count", count)
        count += 1

        model_file_name = os.path.join(model_dir, f)
        print(model_file_name)

        params = f.split(".pth.tar")[0].split(prefix)[1].split("_")
        cnn_layer = list(map(int, params[1].split("-")))
        fc_layer = list(map(int, params[2].split("-")))
        # fc_layer = params[2].split(' ')
        # fc_layer[0] = int(fc_layer[0])
        # fc_layer[1] = int(fc_layer[1])
        dropout_rate = float(params[3])
        input_shape = list(map(int, params[4].split("-")))
        preprocessor = params[5]
        
        if preprocessor not in processors:
            print('Processor not found:', preprocessor)
            continue

        cache_key = preprocessor + '_' + str(input_shape[0]) + '_' + str(input_shape[1])
        tr_loader, val_loader, standardizer = None, None, None

        # Reset for knowledge distillation.
        # teacher_data_dir = prefix + "/FAST1/labeled_images/"
        # teacher_tr_loader, teacher_va_loader, teacher_cascade_loader, _ = get_train_val_loaders(num_classes=2,data_dir=teacher_data_dir, color_mode='ColorImage', image_dim=[224,224], batch_size=config('cnn.batch_size'))
        
        load_start = time.time()

        if cache_key not in CACHE_DATA:
            data_dir = "./tmp/" + prefix + "/FAST1/labeled_images/"
            tr_loader, val_loader, cascade_loader, standardizer = get_train_val_loaders(num_classes=config('cnn.num_classes'), data_dir=data_dir, color_mode=preprocessor, image_dim=input_shape, batch_size=config('cnn.batch_size'))
            CACHE_DATA[cache_key] = (tr_loader, val_loader, cascade_loader, standardizer)
        else:
            tr_loader, val_loader, cascade_loader, standardizer = CACHE_DATA[cache_key]

        load_end = time.time()
        print("data loading time (not useful): ", load_end - load_start)

        model = CNN(input_shape, preprocessor, cnn_layer, fc_layer, dropout_rate)
        # weight=torch.FloatTensor(training_weight).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        # Reset for knowledge distillation. 
        # criterion = torch.nn.KLDivLoss()

        model = restore_checkpoint(model, model_file_name)
        model.to(device)

        val_acc, _, predictions, actuals, _ = _evaluate_epoch(val_loader=val_loader, model=model, criterion=criterion, device=device)
        # Reset for knowledge distillation. 
        # val_acc, _, predictions, actuals, _ = _evaluate_epoch_kd(teacher_data_loader=teacher_va_loader, val_loader=val_loader, teacher=teacher, student=model, criterion=criterion, device=device, standardizer=standardizer)
        
        # use training data to get fps on bigger dataset
        _, _, _, _, infer_fps_train = _evaluate_epoch(val_loader=tr_loader, model=model, criterion=criterion, device=device)
        # Reset for knowledge distillation.
        # _, _, _, _, infer_fps_train = _evaluate_epoch_kd(teacher_data_loader=teacher_tr_loader, val_loader=tr_loader, teacher=teacher, student=model, criterion=criterion, device=device, standardizer=standardizer)

        _, _, predictions_cascade, actuals_cascade, _ = _evaluate_epoch(val_loader=cascade_loader, model=model, criterion=criterion, device=device)
        # Reset for knowledge distillation. 
        # _, _, predictions_cascade, actuals_cascade, _ = _evaluate_epoch_kd(teacher_data_loader=teacher_cascade_loader, val_loader=cascade_loader, teacher=teacher, student=model, criterion=criterion, device=device, standardizer=standardizer)

        print("statistics: val_acc: ", val_acc, "infer_fps_train: ", infer_fps_train)
        acc_vals.append(val_acc)
        infer_fps_vals.append(infer_fps_train)
        # load_fps_vals.append(load_fps_train)
        results.append(predictions)
        cascade_results.append(predictions_cascade)
        ground_truth = np.array([elem.cpu().numpy() for elem in actuals])
        cascade_ground_truth = np.array([elem.cpu().numpy() for elem in actuals_cascade])
        models.append({'cnn_layer': cnn_layer, 'fc_layer': fc_layer, 'dropout_rate': dropout_rate, 'input_shape': input_shape, 'preprocessor': preprocessor})
    
    models.append({'name': 'user', 'preprocessor': 'ColorImage', 'input_shape': [image_dim, image_dim]})
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    results = [[elem.cpu().numpy() for elem in list_] for list_ in results]
    cascade_results = [[elem.cpu().numpy() for elem in list_] for list_ in cascade_results]

    # np.save(result_dir + prefix + '_models', models)
    np.save(result_dir + prefix + '_actual', ground_truth)
    np.save(result_dir + prefix + '_results', np.array(results))
    np.save(result_dir + prefix + '_acc', acc_vals)
    np.save(result_dir + prefix + '_cascade_results', np.array(cascade_results))
    np.save(result_dir + prefix + '_cascade_actual', cascade_ground_truth)
    np.save(result_dir + prefix + '_infer_fps', infer_fps_vals)
    # np.save(result_dir + prefix + '_load_fps', load_fps_vals)

    return models

if __name__ == '__main__':
    eval_models()