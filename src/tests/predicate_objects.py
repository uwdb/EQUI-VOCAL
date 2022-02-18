from __future__ import print_function, division

import json
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from time import time
import statistics
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset
import torchvision
from torchvision import datasets, models, transforms
import os
import copy
import math

def construct_spatial_feature(bbox):
    if bbox == (0, 0, 0, 0):
        return np.array([0, 0, 0, 0, 0])
    x1, y1, x2, y2 = bbox
    centroid_x = (x1 + x2) / 2
    centroid_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    wh_ratio = width / height
    return np.array([centroid_x, centroid_y, width, height, wh_ratio])

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensor_x, tensor_y, transform):
        self.tensor_x = tensor_x
        self.tensor_y = tensor_y
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensor_x[index]

        if self.transform:
            x = self.transform(x)

        y = self.tensor_y[index]

        return x, y

    def __len__(self):
        return self.tensor_x.size(0)

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, dataset_sizes, num_epochs=25):
    since = time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):

        y_test = []
        y_pred = []
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if phase == 'val':
                    y_test.append(labels.data.cpu())
                    y_pred.append(preds.cpu())
            if phase == 'train':
                scheduler.step()
            else:
                y_test = np.concatenate(y_test)
                y_pred = np.concatenate(y_pred)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print("[Metrics] Accuracy: {}, Balanced Accuracy: {}, F1 Score: {}, Precision: {}, Recall: {}".format(metrics.accuracy_score(y_test, y_pred), metrics.balanced_accuracy_score(y_test, y_pred), metrics.f1_score(y_test, y_pred), metrics.precision_score(y_test, y_pred), metrics.recall_score(y_test, y_pred)))

        print()

    time_elapsed = time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return metrics.accuracy_score(y_test, y_pred), metrics.balanced_accuracy_score(y_test, y_pred), metrics.f1_score(y_test, y_pred), metrics.precision_score(y_test, y_pred), metrics.recall_score(y_test, y_pred), time_elapsed

def predicate_objects_vocal(train_size, test_size=300, sampling_rate=30, dataset="visualroad_traffic2"):

        """Read in object detection bounding box information
        """
        if dataset == "visualroad_traffic2":
            with open("../../data/car_turning_traffic2/bbox.json", 'r') as f:
                maskrcnn_bboxes = json.loads(f.read())
            n_frames = len(maskrcnn_bboxes)
        elif dataset == "meva":
            pass

        """Prepare data
        """
        X, y = [], []
        for frame_id in range(n_frames):
            if frame_id % sampling_rate:
                continue
            res_per_frame = maskrcnn_bboxes["frame_{}.jpg".format(frame_id)]
            has_car = 0
            positive_car_boxes = []
            car_boxes = [(0, 0, 0, 0)] # This is the case when no car is detected
            for x1, y1, x2, y2, class_name, _ in res_per_frame:
                if class_name in ["car", "truck"]:
                    car_boxes.append((x1, y1, x2, y2))
                    if (x1 + x2) / 2 <= 480 and (y1 + y2) / 2 >= 270 and (y1 + y2) / 2 <= 540: # Car in the bottom-left of the frame
                        positive_car_boxes.append((x1, y1, x2, y2))
                    break
            if positive_car_boxes: # Positive sample
                X.append(construct_spatial_feature(random.choice(positive_car_boxes)))
                y.append(1)
            else: # Negative sample
                X.append(construct_spatial_feature(random.choice(car_boxes)))
                y.append(0)
        X = np.asarray(X)
        y = np.asarray(y)

        """Training model
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size)
        print("Training set size: {}, # positive samples: {}, # negative samples: {}".format(X_train.shape, np.sum(y_train), np.count_nonzero(y_train==0)))
        print("Test set size: {}, # positive samples: {}, # negative samples: {}".format(X_test.shape, np.sum(y_test), np.count_nonzero(y_test==0)))
        start_ = time()
        clf = RandomForestClassifier(
            # criterion="entropy",
            # max_depth=10,
            n_estimators=100,
            # min_samples_split=32,
            class_weight="balanced",
            # min_impurity_decrease=0.1
        )
        clf = clf.fit(X_train, y_train)
        training_time = time() - start_
        print("Training time: ", training_time)
        """Make prediction
        """
        y_pred = clf.predict(X_test)
        print("[Metrics] Accuracy: {}, Balanced Accuracy: {}, F1 Score: {}, Precision: {}, Recall: {}".format(metrics.accuracy_score(y_test, y_pred), metrics.balanced_accuracy_score(y_test, y_pred), metrics.f1_score(y_test, y_pred), metrics.precision_score(y_test, y_pred), metrics.recall_score(y_test, y_pred)))
        return metrics.accuracy_score(y_test, y_pred), metrics.balanced_accuracy_score(y_test, y_pred), metrics.f1_score(y_test, y_pred), metrics.precision_score(y_test, y_pred), metrics.recall_score(y_test, y_pred), training_time

def predicate_objects_deep(train_size, test_size=300, sampling_rate=30, dataset="visualroad_traffic2"):

        """Read in object detection bounding box information
        """
        if dataset == "visualroad_traffic2":
            with open("../../data/car_turning_traffic2/bbox.json", 'r') as f:
                maskrcnn_bboxes = json.loads(f.read())
            n_frames = len(maskrcnn_bboxes)
        elif dataset == "meva":
            pass

        """Ingest video into numpy array
        """
        print("Ingest video into numpy array")
        cap = cv2.VideoCapture("/gscratch/balazinska/enhaoz/complex_event_video/data/visual_road/traffic-2.mp4")
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        buf = np.empty((math.ceil(frameCount / sampling_rate), frameHeight, frameWidth, 3), np.dtype('uint8'))
        fc = 0
        ret = True
        fc_sampled = 0
        while (fc < frameCount and ret):
            ret, frame = cap.read()
            if fc % sampling_rate:
                fc += 1
                continue
            fc += 1
            buf[fc_sampled] = frame / 255.
            fc_sampled += 1
        cap.release()

        """Prepare data
        """
        print("Prepare data")
        X = buf
        y = []
        for frame_id in range(n_frames):
            if frame_id % sampling_rate:
                continue
            res_per_frame = maskrcnn_bboxes["frame_{}.jpg".format(frame_id)]
            has_car = 0
            positive_car_boxes = []
            car_boxes = [(0, 0, 0, 0)] # This is the case when no car is detected
            for x1, y1, x2, y2, class_name, _ in res_per_frame:
                if class_name in ["car", "truck"]:
                    car_boxes.append((x1, y1, x2, y2))
                    if (x1 + x2) / 2 <= 480 and (y1 + y2) / 2 >= 270 and (y1 + y2) / 2 <= 540: # Car in the bottom-left of the frame
                        positive_car_boxes.append((x1, y1, x2, y2))
                    break
            if positive_car_boxes: # Positive sample
                y.append(1)
            else: # Negative sample
                y.append(0)
        tensor_x = torch.tensor(X, dtype=torch.float) # transform to torch tensor
        tensor_x = tensor_x.permute(0, 3, 1, 2)
        tensor_y = torch.tensor(y)
        """Load data for Pytorch
        """
        print("Load data for Pytorch")
        print(tensor_x.shape)
        # Data augmentation and normalization for training
        # Just normalization for validation
        data_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop(224),
            # transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        my_dataset = CustomTensorDataset(tensor_x, tensor_y, data_transform) # create your datset
        train_idx, val_idx = train_test_split(list(range(len(my_dataset))), train_size=train_size, test_size=test_size)
        my_datasets = {}
        my_datasets['train'] = Subset(my_dataset, train_idx)
        my_datasets['val'] = Subset(my_dataset, val_idx)
        print("Training set size:", len(my_datasets['train']))
        print("Test set size:", len(my_datasets['val']))
        my_dataloaders = {x: DataLoader(my_datasets[x], batch_size=8, shuffle=True, num_workers=1) for x in ['train','val']}
        dataset_sizes = {x: len(my_datasets[x]) for x in ['train', 'val']}
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        """Finetuning the convnet
        """
        print("Finetuning the convnet")
        model_ft = models.resnet50(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        # Here the size of each output sample is set to 2.
        model_ft.fc = nn.Linear(num_ftrs, 2)

        model_ft = model_ft.to(device)

        criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        return train_model(model_ft, my_dataloaders, criterion, optimizer_ft, exp_lr_scheduler, device, dataset_sizes, num_epochs=25)


def run_vocal():
    out_list = []
    acc_list, balanced_acc_list, f1_list, precision_list, recall_list, training_time_list = ([] for _ in range(6))
    for train_size in [10, 20, 50, 100, 150, 200]:
        print("Train size: ", train_size)
        for i in range(10):
            print("Iteration: ", i)
            acc, balanced_acc, f1, precision, recall, training_time = predicate_objects_vocal(train_size)
            acc_list.append(acc)
            balanced_acc_list.append(balanced_acc)
            f1_list.append(f1)
            precision_list.append(precision)
            recall_list.append(recall)
            training_time_list.append(training_time)
        print("[acc] mean: {:.3f}, stdev: {:.3f}".format(statistics.mean(acc_list), statistics.stdev(acc_list)))
        print("[balanced_acc] mean: {:.3f}, stdev: {:.3f}".format(statistics.mean(balanced_acc_list), statistics.stdev(balanced_acc_list)))
        print("[f1] mean: {:.3f}, stdev: {:.3f}".format(statistics.mean(f1_list), statistics.stdev(f1_list)))
        print("[precision] mean: {:.3f}, stdev: {:.3f}".format(statistics.mean(precision_list), statistics.stdev(precision_list)))
        print("[recall] mean: {:.3f}, stdev: {:.3f}".format(statistics.mean(recall_list), statistics.stdev(recall_list)))
        print("[training_time] mean: {:.3f}, stdev: {:.3f}".format(statistics.mean(training_time_list), statistics.stdev(training_time_list)))
        out_list.append([train_size, statistics.mean(acc_list), statistics.stdev(acc_list), statistics.mean(balanced_acc_list), statistics.stdev(balanced_acc_list), statistics.mean(f1_list), statistics.stdev(f1_list), statistics.mean(precision_list), statistics.stdev(precision_list), statistics.mean(recall_list), statistics.stdev(recall_list), statistics.mean(training_time_list), statistics.stdev(training_time_list)])

    with open("/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/predicates/predicate_objects_vocal.json", 'w') as f:
            f.write(json.dumps(out_list))

def run_deep():
    out_list = []
    acc_list, balanced_acc_list, f1_list, precision_list, recall_list, training_time_list = ([] for _ in range(6))
    for train_size in [10, 20, 50, 100, 150, 200]:
        print("Train size: ", train_size)
        for i in range(1):
            print("Iteration: ", i)
            acc, balanced_acc, f1, precision, recall, training_time = predicate_objects_deep(train_size)
            acc_list.append(acc)
            balanced_acc_list.append(balanced_acc)
            f1_list.append(f1)
            precision_list.append(precision)
            recall_list.append(recall)
            training_time_list.append(training_time)
        print("[acc] mean: {:.3f}, stdev: {:.3f}".format(statistics.mean(acc_list), statistics.stdev(acc_list)))
        print("[balanced_acc] mean: {:.3f}, stdev: {:.3f}".format(statistics.mean(balanced_acc_list), statistics.stdev(balanced_acc_list)))
        print("[f1] mean: {:.3f}, stdev: {:.3f}".format(statistics.mean(f1_list), statistics.stdev(f1_list)))
        print("[precision] mean: {:.3f}, stdev: {:.3f}".format(statistics.mean(precision_list), statistics.stdev(precision_list)))
        print("[recall] mean: {:.3f}, stdev: {:.3f}".format(statistics.mean(recall_list), statistics.stdev(recall_list)))
        print("[training_time] mean: {:.3f}, stdev: {:.3f}".format(statistics.mean(training_time_list), statistics.stdev(training_time_list)))
        out_list.append([train_size, statistics.mean(acc_list), statistics.stdev(acc_list), statistics.mean(balanced_acc_list), statistics.stdev(balanced_acc_list), statistics.mean(f1_list), statistics.stdev(f1_list), statistics.mean(precision_list), statistics.stdev(precision_list), statistics.mean(recall_list), statistics.stdev(recall_list), statistics.mean(training_time_list), statistics.stdev(training_time_list)])

    with open("/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/predicates/predicate_objects_deep.json", 'w') as f:
            f.write(json.dumps(out_list))

if __name__ == '__main__':
    # run_vocal()
    run_deep()