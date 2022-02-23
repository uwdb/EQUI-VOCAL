from __future__ import print_function, division

import json
from turtle import pos
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
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset, random_split
import torchvision
from torchvision import datasets, models, transforms
import os
import copy
import math
from glob import glob
import yaml
import cv2
from random import sample

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

        print("[Metrics] Accuracy: {}, Balanced Accuracy: {}, F1 Score: {}, Precision: {}, Recall: {}".format(metrics.accuracy_score(y_test, y_pred), metrics.balanced_accuracy_score(y_test, y_pred), metrics.f1_score(y_test, y_pred), metrics.precision_score(y_test, y_pred, zero_division=0), metrics.recall_score(y_test, y_pred)))

        print()

    time_elapsed = time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return metrics.accuracy_score(y_test, y_pred), metrics.balanced_accuracy_score(y_test, y_pred), metrics.f1_score(y_test, y_pred), metrics.precision_score(y_test, y_pred, zero_division=0), metrics.recall_score(y_test, y_pred), time_elapsed

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
        print("[Metrics] Accuracy: {}, Balanced Accuracy: {}, F1 Score: {}, Precision: {}, Recall: {}".format(metrics.accuracy_score(y_test, y_pred), metrics.balanced_accuracy_score(y_test, y_pred), metrics.f1_score(y_test, y_pred), metrics.precision_score(y_test, y_pred, zero_division=0), metrics.recall_score(y_test, y_pred)))
        return metrics.accuracy_score(y_test, y_pred), metrics.balanced_accuracy_score(y_test, y_pred), metrics.f1_score(y_test, y_pred), metrics.precision_score(y_test, y_pred, zero_division=0), metrics.recall_score(y_test, y_pred), training_time

def query_car_location(sampling_rate=25):
    """Read in object detection bounding box information
    """
    with open("../../data/car_turning_traffic2/bbox.json", 'r') as f:
        maskrcnn_bboxes = json.loads(f.read())
    n_frames = len(maskrcnn_bboxes)

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
        positive_car_boxes = []
        for x1, y1, x2, y2, class_name, _ in res_per_frame:
            if class_name in ["car", "truck"]:
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
    return tensor_x, tensor_y

def prepare_person_stands_up():
    activity_name = "person_stands_up"
    video_camera = "school.G421"
    video_basenames = [os.path.basename(y).replace(".activities.yml", "") for x in os.walk("../../data/meva/meva-data-repo/annotation/DIVA-phase-2/MEVA/kitware") for y in glob(os.path.join(x[0], '*.{}.activities.yml'.format(video_camera)))]
    print("video_basenames", video_basenames)
    middle_index = int(len(video_basenames) * 0.75)
    activities_files = [y for x in os.walk("../../data/meva/meva-data-repo/annotation/DIVA-phase-2/MEVA/kitware") for y in glob(os.path.join(x[0], '*.{}.activities.yml'.format(video_camera)))]
    video_files = [y for x in os.walk("../../data/meva") for y in glob(os.path.join(x[0], '*.avi'))]
    for i, video_basename in enumerate(video_basenames):
        if i < 34:
            continue
        print(video_basename)
        train_or_test = "train" if i < middle_index else "test"
        # Ingest video into numpy array, stored in ``buf''
        matching = [f for f in video_files if video_basename + ".r13.avi" in f]
        assert(len(matching) == 1)
        video_file = matching[0]
        cap = cv2.VideoCapture(video_file)
        all_actor_id = []
        pos_actor_id = []
        matching = [f for f in activities_files if video_basename + ".activities" in f]
        assert(len(matching) == 1)
        activities_file = matching[0]
        geom_file = activities_file.replace(".activities", ".geom")
        types_file = activities_file.replace(".activities", ".types")
        with open(activities_file, 'r') as f:
            activities_annotation = yaml.load(f, Loader=yaml.CLoader)
        print("loaded activities file")
        with open(geom_file, 'r') as f:
            geom_annotation = yaml.load(f, Loader=yaml.CLoader)
        print("loaded geom file")
        with open(types_file, 'r') as f:
            types_annotation = yaml.load(f, Loader=yaml.CLoader)
        print("loaded types file")
        # Construct all_actor_id list
        for types_row in types_annotation:
            if "types" in types_row and "cset3" in types_row["types"] and "person" in types_row["types"]["cset3"]:
                all_actor_id.append(types_row["types"]["id1"])
        # Construct pos_actor_id list
        num_pos = 0
        for row in activities_annotation:
            if "act" in row and activity_name in row["act"]["act2"]:
                actor_id = row["act"]["actors"][0]["id1"]
                pos_actor_id.append(actor_id)
                start_frame, end_frame = row["act"]["timespan"][0]["tsr0"]
                pos_frames = []
                for i in range(start_frame, end_frame+1):
                    pos_frames.append(i)
                for geom_row in geom_annotation:
                    if "geom" in geom_row and geom_row["geom"]["id1"] == actor_id and geom_row["geom"]["ts0"] in pos_frames:
                        bbox = list(map(int, geom_row["geom"]["g0"].split()))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, geom_row["geom"]["ts0"])
                        ret, frame = cap.read()
                        cropped_image = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                        cv2.imwrite(os.path.join("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/data/meva_person_stands_up", train_or_test, "pos", '{}_{}_{}.jpg'.format(video_basename, str(actor_id).zfill(4), geom_row["geom"]["ts0"])), cropped_image)
                        num_pos += 1
        # Construct neg_actor_id list
        neg_candidates = []
        for geom_row in geom_annotation:
            if "geom" in geom_row and geom_row["geom"]["id1"] in all_actor_id and geom_row["geom"]["id1"] not in pos_actor_id:
                bbox = list(map(int, geom_row["geom"]["g0"].split()))
                neg_candidates.append([geom_row["geom"]["ts0"], bbox, geom_row["geom"]["id1"]])
        for fid, bbox, actor_id in sample(neg_candidates, num_pos):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            cropped_image = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            cv2.imwrite(os.path.join("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/data/meva_person_stands_up", train_or_test, "neg", '{}_{}_{}.jpg'.format(video_basename, str(actor_id).zfill(4), fid)), cropped_image)
        cap.release()


def query_person_stands_up(train_size=None, test_size=None):
    """train_size: number of training images. If None, use all training images.
    test_size: number of test images. If None, use all test images.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    # data_transform = transforms.Compose([
    #     transforms.Resize((224,224)),
    #     # transforms.CenterCrop(224),
    #     # transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    data_dir = '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/data/meva_person_stands_up'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    if train_size:
        image_datasets["train"] = random_split(image_datasets["train"], [train_size, len(image_datasets["train"])-train_size])[0]
    if test_size:
        image_datasets["val"] = random_split(image_datasets["val"], [test_size, len(image_datasets["val"])-test_size])[0]

    print("Training set size:", len(image_datasets['train']))
    print("Test set size:", len(image_datasets['val']))
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    """Finetuning the convnet
    """
    print("Finetuning the convnet")
    model_conv = models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
    # optimizer_conv = optim.Adam(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    return train_model(model_conv, dataloaders, criterion, optimizer_conv, exp_lr_scheduler, device, dataset_sizes, num_epochs=25)



def predicate_objects_deep(train_size, test_size=200):
    tensor_x, tensor_y = query_car_location()
    # Use first 500 seconds of video for training, and remainder (400 seconds) for testing.
    tensor_x_train = tensor_x[:500]
    tensor_x_test = tensor_x[500:]
    tensor_y_train = tensor_y[:500]
    tensor_y_test = tensor_y[500:]

    # Construct balanced dataset
    tensor_x_pos_train = tensor_x_train[(tensor_y_train == 1).nonzero(as_tuple=True)[0]]
    tensor_x_neg_train = tensor_x_train[(tensor_y_train == 0).nonzero(as_tuple=True)[0]]
    indices_pos_train = torch.randperm(len(tensor_x_pos_train))[:int(train_size/2)]
    tensor_x_pos_train = tensor_x_pos_train[indices_pos_train]
    indices_neg_train = torch.randperm(len(tensor_x_neg_train))[:int(train_size/2)]
    tensor_x_neg_train = tensor_x_neg_train[indices_neg_train]

    tensor_x_pos_test = tensor_x_test[(tensor_y_test == 1).nonzero(as_tuple=True)[0]]
    tensor_x_neg_test = tensor_x_test[(tensor_y_test == 0).nonzero(as_tuple=True)[0]]
    indices_pos_test = torch.randperm(len(tensor_x_pos_test))[:int(test_size/2)]
    tensor_x_pos_test = tensor_x_pos_test[indices_pos_test]
    indices_neg_test = torch.randperm(len(tensor_x_neg_test))[:int(train_size/2)]
    tensor_x_neg_test = tensor_x_neg_test[indices_neg_test]

    tensor_x_train = torch.cat((tensor_x_pos_train, tensor_x_neg_train), 0)
    tensor_y_train = torch.cat((torch.ones(tensor_x_pos_train.size(dim=0)), torch.zeros(tensor_x_neg_train.size(dim=0))), 0)
    tensor_x_test = torch.cat((tensor_x_pos_test, tensor_x_neg_test), 0)
    tensor_y_test = torch.cat((torch.ones(tensor_x_pos_test.size(dim=0)), torch.zeros(tensor_x_neg_test.size(dim=0))), 0)
    print("after balancing:", tensor_x_train.size(), tensor_y_train.size(), tensor_x_test.size(), tensor_y_test.size())
    tensor_y_train = tensor_y_train.type(torch.LongTensor)
    tensor_y_test = tensor_y_test.type(torch.LongTensor)
    """Load data for Pytorch
    """
    print("Load data for Pytorch")
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transform = transforms.Compose([
        transforms.Resize((224,224)),
        # transforms.CenterCrop(224),
        # transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    my_datasets = {}
    my_datasets['train'] = CustomTensorDataset(tensor_x_train, tensor_y_train, data_transform)
    my_datasets['val'] = CustomTensorDataset(tensor_x_test, tensor_y_test, data_transform)
    print("Training set size:", len(my_datasets['train']))
    print("Test set size:", len(my_datasets['val']))
    my_dataloaders = {x: DataLoader(my_datasets[x], batch_size=8, shuffle=True, num_workers=1) for x in ['train','val']}
    dataset_sizes = {x: len(my_datasets[x]) for x in ['train', 'val']}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    """Finetuning the convnet
    """
    print("Finetuning the convnet")
    model_conv = models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    return train_model(model_conv, my_dataloaders, criterion, optimizer_conv, exp_lr_scheduler, device, dataset_sizes, num_epochs=10)


def run_vocal():
    out_list = []
    for train_size in [10, 20, 50, 100, 150, 200]:
        print("Train size: ", train_size)
        acc_list, balanced_acc_list, f1_list, precision_list, recall_list, training_time_list = ([] for _ in range(6))
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
    # for train_size in [10, 20, 50, 100, 150, 200]:
    for train_size in [200]:
        print("Train size: ", train_size)
        acc_list, balanced_acc_list, f1_list, precision_list, recall_list, training_time_list = ([] for _ in range(6))
        for i in range(10):
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
    # run_deep()
    # prepare_person_stands_up()
    query_person_stands_up()