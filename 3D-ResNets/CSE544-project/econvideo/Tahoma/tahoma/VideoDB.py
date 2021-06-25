import json
import os
import pickle
import random
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from Classifier import Classifier
from VideoStream import VideoStream

from cascades import *
from evaluate_models import *
from multiproc_cascades import compute_all_cascades
from train_common import *
from train_cnn import *
from utils import config

# Some basic setup:
# Setup detectron2 logger
import torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
from tqdm import tqdm

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer


class VideoDB(object):
    """
    This is a module that takes client's classifier and return a video clip of required range.
    We are aiming at reducing runtime and will provide 4 modes:
        -dumb scan
        -tahoma multiresolution
        -zombie indexing
        -combination of tahoma multiresolution and zombie indexing
    """
    class Error(Exception):
        def __init__(self, error_msg):
            self.error_msg = error_msg

        def __str__(self):
            return error_msg

    def __init__(self, device):
        """
        Create an instance of the VideoDB class.
        Initialization:
            Resize all videos in the database to various sizes.
        """

        self.classifier_name = None
        self.mode = None
        self.info = None
        self.images = None
        self.classifier_dir = None
        self.user_classifier = None
        self.device = device
        self.topK = None
        self.image_dim = 1920
        self.mean = None 
        self.std = None 
        self.mean_models = {}
        self.std_models = {} 
        self.classifier = None
        self.videostream = None 
        self.load_time = {}
        self.standardize_time = {}
        self.interval = None
        # Path of the video database.
        self.videoDB_path = config('video_path')
        # Path of the root directory which is to store all resized videos.
        self.resized_videoDB_path = config("resized_video_path")
        self.user_infer_time = 0
        sizes = [30, 60, 120, 224]
        self._resize_database_videos(sizes)
        print("Initialization done")

    # def __del__(self):
    #     # Release cv2 video capture objects and close display windows
    #     cv2.destroyAllWindows()
        
    def _resize_database_videos(self, sizes):
        '''
        Called once when the database is initialized;
        resize videos to 4 sizes and store them on disk
        '''
        # Create directories to store resized videos.
        size_directories = []
        i = 0
        while i < len(sizes):
            size_directory = os.path.join(self.resized_videoDB_path, "{}".format(sizes[i]))
            if not os.path.exists(size_directory):
                print("Cannot find resized video database for size: ", sizes[i], ". Resizing videos...")
                os.makedirs(size_directory)
                size_directories.append(size_directory)
                i += 1
            else:
                del sizes[i]

        if len(sizes) == 0:
            return    

        # Read and write resized videos.
        for file in os.listdir(self.videoDB_path):
            cap = cv2.VideoCapture(os.path.join(self.videoDB_path, file))
            if not cap.isOpened():
                print("Error opening video stream or file: ", file)
            else:
                (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
                fps = -1
                if int(major_ver) < 3:
                    fps = round(cap.get(cv2.cv.CV_CAP_PROP_FPS))
                else:
                    fps = round(cap.get(cv2.CAP_PROP_FPS))

                file = file.split('.')[0]
                fourcc = cv2.VideoWriter_fourcc(*'XVID')

                out = []
                video_paths = []
                
                # Initialize video writers of different sizes for each video.
                for i in range(len(sizes)):
                    video_paths.append(os.path.join(size_directories[i], '{}.{}'.format(file, "avi")))
                    out.append(cv2.VideoWriter(video_paths[i] ,fourcc, fps, (sizes[i], sizes[i])))
                
                # Read and resize frames.
                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret == True: # current frame is effective
                        for i in range(len(sizes)):
                            resized_frame = cv2.resize(frame, (sizes[i], sizes[i]), interpolation=cv2.INTER_CUBIC)
                            out[i].write(resized_frame)
                    else:
                        break                                                         
            cap.release()
            # Write resized videos to the disk.

            for each in out:                     
                each.release()                                                                              
        # cv2.destroyAllWindows()      
        print("Resized videos for:", sizes)
    
        
    def query(self, classifier_name, user_classifier, mode, accuracy=0, topK=100, info=None, images=None, interval=60):
        '''
        The actual query function
        Parameters 
        ----------
        classifier_name : string
            Used to create unique local cache
        user_classifier : torch.nn.Module 
            The pre-trained binary user classifier
        mode : 'NAIVE' | 'FAST1' | 'TRUE NAIVE'
            Specifies which mode to run the query on
        accuracy : float
            The minimum accuracy that query needs to achieve (compared to user classifier result)
            Invalid under naive modes
        topK : int
            Number of frames to return
        info : string
            Path to file that records the mean and stdev of the user training set
            Used to preprocess images
        images : string
            Path to sample of the user training set
            Only one of "images" and "info" is used
        interval : int
            Extract one frame for training set every "interval" frames
        '''
        # if (info and images) or (not info and not images):
        #     print("Exactly one of the arguments info and images can be filled")
        #     raise Error("Exactly one of the arguments info and images can be filled")

        self.interval = interval
        self.classifier_name = classifier_name
        # self.user_classifier = user_classifier
        # self.user_classifier.eval()
        self.topK = topK

        if not os.path.exists("./tmp/" + self.classifier_name):
            os.mkdir("./tmp/" + self.classifier_name)
        
        # load mean and std of the user training set from either "info" or "images"
        # self.load_user_info(info, images)
        
        # self._resize_database_videos([self.image_dim])

        print("resized")
        if mode == "TRUE-NAIVE":
            tik = time.time()
            total, count = 0, 0
            for file in os.listdir(self.videoDB_path):  
                cap = cv2.VideoCapture(os.path.join(self.videoDB_path, file))
                if not cap.isOpened():
                    print("Error opening video stream or file: ", file)
                else:
                    # Read and resize frames.
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if ret == True: # current frame is effective
                            frame = self._preprocess(frame, False)
                            output = self.user_classifier(frame)
                            result = predictions(output.data)
                            total += 1
                            if result[0]:
                                count += 1
                                # print("total:", total)
                                # print('count:', count)
                            if count >= topK:
                                print('found:', count)
                                print("total:", total)
                                tok = time.time()
                                print('query time:', tok - tik)
                                return
                        else:
                            break
            print('found:', count)
            print("total:", total)
            tok = time.time()
            print('query time:', tok - tik)
            return

        elif mode == "NAIVE":
            self.classifier = Classifier([self.user_classifier], [-1.0, 2,0])
            self.videostream = iter(VideoStream([[self.image_dim, self.image_dim]], ['ColorImage'], config("resized_videodb_path"), self.mean_models, self.std_models, self.interval))

        elif mode == "FAST1":
            if not os.path.exists("./tmp/" + self.classifier_name + "/FAST1/"):
                # FAST1 hasn't been initialized. Start the training process of FAST1.
                os.mkdir("./tmp/" + self.classifier_name + "/FAST1/")
            self._train_fast()
            
            with open("./tmp/" + self.classifier_name + "/info_models.json", "r") as json_file:
                data = json.load(json_file)
                for key, value in data.items():
                    self.mean_models[key] = value['mean']
                    self.std_models[key] = value['std']

            # Select cascade from the previously trained ones
            # open the pickled file that contains the pareto frontier of all models
            with open('./tmp/' + self.classifier_name + '/FAST1/frontiers.pkl', 'rb') as f:
                frontiers = pickle.load(f)
                print("frontiers: ", frontiers)

            # now select the cascade that satisfies the user given accuracy and has the highest throughput
            models = list(np.load(os.path.join("./tmp/" + self.classifier_name + "/FAST1/results", self.classifier_name +'_models.npy'), allow_pickle=True))

            for item in frontiers:
                print('item in frontiers:', item)
                if item[2] >= accuracy:
                # if item[2] >= accuracy and item[3] >= 0.9:
                    list_of_models = []
                    list_of_low_high = []
                    list_of_img_size = []
                    list_of_preprocessor = []
                    for model_id in item[0]:
                        print('model_id', model_id)
                        list_of_models.append(self._load_model_from_dict(models[model_id])) # model object
                        list_of_low_high.append(models[model_id][item[4]]) # p_low and p_high
                        list_of_img_size.append(models[model_id]["input_shape"])
                        list_of_preprocessor.append(models[model_id]["preprocessor"])
                    self.classifier = Classifier(list_of_models, list_of_low_high)
                    self.videostream = iter(VideoStream(list_of_img_size, list_of_preprocessor, config("resized_videodb_path"), self.mean_models, self.std_models, self.interval))
                    break

        # Begin query
        print('Begin Query...')
        result = []
        count = 0
        total = 0
        frame_read = 0
        query_time = 0
        frame_list_time = 0 
        start_time = time.time()
        frame_list_start = time.time()
        for frame_list in self.videostream:
            frame_list_time += time.time() - frame_list_start
            query_time += time.time() - start_time
            frame_read += 1

            if frame_read % self.interval != 0:
                start_time = time.time()
                frame_list_start = time.time()
                continue
            # print(frame_list[0].shape)

            pred_start = time.time()
            frame_list = [frame.to(self.device) for frame in frame_list]
            pred = self.classifier(frame_list)  
            query_time += time.time() - pred_start

            total += 1
            result.append(pred)

            if pred == 1:
                count += 1
                # print('total', total)  
                # print('count', count)
            if count == self.topK:
                break

            start_time = time.time()
            frame_list_start = time.time()

        # cv2.destroyAllWindows()

        print('query time:', query_time)
        print('frame_list_time', frame_list_time)
        print("found:", count)
        print('total', total)
        print('frame_read', frame_read)

        return np.array(result)

    def load_user_info(self, info, images):
        '''
        Load the mean and std of the user training dataset from either "info" and "images" provided by the user. The results are written into "info_user.json".
        -----------------------------
        info : string
            Path to file that records the mean and stdev of the user training set
            Used to preprocess images
        images : string
            Path to sample of the user training set
            Only one of "images" and "info" is used
        '''
        # User provides mean and std
        if info: 
            with open(info) as json_file:
                data = json.load(json_file)
                image_dim = data['image_dim']
                mean = data['mean']
                std = data['std']
            with open("./tmp/" + self.classifier_name + "/info_user.json", 'w') as f:
                print("mean", mean, "std", std)
                json.dump({"mean": mean, "std": std, "image_dim": image_dim}, f)
        else: # User provides image dir
            X = []
            img_list = os.listdir(images)
            random.shuffle(img_list)
            count = 0

            for file in img_list:
                if file.endswith(('.jpg','.jpeg','.png','.ppm','.bmp', '.tiff')):
                    image = cv2.imread(os.path.join(images, file))
                    X.append(image)
                    count += 1
                if count == 100:
                    break

            X = np.array(X)
            image_dim = X.shape[1]
            color_channel = X.shape[3]
            image_mean = np.zeros(color_channel)
            image_std = np.zeros(color_channel)

            for i in range(0, color_channel):
                layer = X[:, :, :, i]
                image_mean[i] = layer.mean()
                image_std[i] = layer.std()

            with open("./tmp/" + self.classifier_name + "/info_user.json", 'w') as f:
                json.dump({"mean": image_mean.tolist(), "std": image_std.tolist(), "image_dim": image_dim}, f)

        with open("./tmp/" + self.classifier_name + "/info_user.json", "r") as json_file:
            data = json.load(json_file)
            self.image_dim = data['image_dim']
            self.mean = data['mean']
            self.std = data['std']

        self.mean_models[str(self.image_dim) + 'x' + str(self.image_dim)] = self.mean
        self.std_models[str(self.image_dim) + 'x' + str(self.image_dim)] = self.std
        
        
    def _load_model_from_dict(self, model): 
        '''
        Load pytorch model given the model dict. Return the torch.nn.Module of the correponding model.
        --------------------------------------
        model: dict 
            A example of model dict is: {'name': 'user', 'preprocessor': 'ColorImage', 'input_shape': [image_dim, image_dim]}
        '''
        if 'name' in model and model['name'] == 'user':
            return self.user_classifier
            # return torch.load(self.classifier_dir)
        suffix = '-'.join(map(str,model['cnn_layer'])) + "_" + '-'.join(map(str,model['fc_layer'])) + "_" + str(model['dropout_rate']) + "_" + '-'.join(map(str,model['input_shape'])) + "_" + model['preprocessor']
        filename = self.classifier_name + '_' + suffix + '.pth.tar'
        print('filename', filename)
        filepath = os.path.join("./tmp/" + self.classifier_name + "/FAST1/models", filename)
        checkpoint = torch.load(filepath)
        return checkpoint['model'].to(self.device)

    def _train_fast(self):
        ''' 
        The training process of FAST1.
        '''
        # Construct labeled_images dir
        self._generate_labels([500, 500], [200, 200], [500, 500], 30)
        #self._exp_generate_labels([4000, 4000], [500, 500], [500, 500], 30)

        # Read in user_infer_time
        with open("./tmp/" + self.classifier_name + "/FAST1/labeled_images/user_infer_time.json", "r") as json_file:
            data = json.load(json_file)
            self.user_infer_time = data["user_infer_time"]
        
        # Train 360 models
        model_directory = './tmp/' + self.classifier_name + '/FAST1/models/'
        train_cnn(model_directory, self.classifier_name, self.device, self.image_dim)

        with open("./tmp/" + self.classifier_name + "/info_models.json", "r") as json_file:
            data = json.load(json_file)
            for key, value in data.items():
                self.mean_models[key] = value['mean']
                self.std_models[key] = value['std']

        self._compute_loading_time()

        # Evaluate all models
        result_directory = './tmp/' + self.classifier_name + '/FAST1/results/'
        models = eval_models(result_directory, model_directory, self.classifier_name, self.image_dim, self.device)

        # Evaluate all cascades
        threshes = [0.91, 0.93, 0.95, 0.97, 0.99]
        all_cascades = []

        for thresh in threshes:
            cascade_result = compute_all_cascades(self.classifier_name, float(thresh), models, self.load_time, self.standardize_time, self.user_infer_time, result_directory)
            all_cascades.extend(cascade_result)
            frontier_result, frontier_result_precision, user_accuracy_cascade, user_precision_cascade = compute_pareto_frontier(cascade_result)
            frontier_result.sort(key=lambda tup: (tup[2]))
            frontier_result_precision.sort(key=lambda tup: (tup[3]))
            self._plot_pareto_frontiers(frontier_result, frontier_result_precision, user_accuracy_cascade, user_precision_cascade, cascade_result, thresh)

        # Find frontier
        frontiers, frontiers_precision, user_accuracy_cascade, user_precision_cascade = compute_pareto_frontier(all_cascades)
        # Sort frontier by accuracy (ascending order)
        frontiers.sort(key=lambda tup: (tup[2]))
        frontiers_precision.sort(key=lambda tup: (tup[3]))
        self._plot_pareto_frontiers(frontiers, frontiers_precision, user_accuracy_cascade, user_precision_cascade, all_cascades, 1.00)

        with open('./tmp/' + self.classifier_name + '/frontier_result.txt', 'w') as f:
            json.dump(frontiers_precision, f)
        with open('./tmp/' + self.classifier_name + '/FAST1/frontiers.pkl', 'wb') as f:
            pickle.dump(frontiers, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        np.save('./tmp/' + self.classifier_name + '/FAST1/results/' + self.classifier_name + '_models', models)
        
    def _generate_labels(self, tr_split=[500, 500], va_split=[200, 200], cas_split=[500, 500], interval=1):
        # Ensure training and validation images are extracted from different videos
        tr_pos_count, tr_neg_count = 0, 0
        tr_pos, tr_neg = [], []

        va_cas_pos_count, va_cas_neg_count = 0, 0
        va_cas_pos, va_cas_neg = [], []

        va_cas_pos_total = va_split[0] + cas_split[0]
        va_cas_neg_total = va_split[1] + cas_split[1]

        frame_read, frame_extracted = 0, 0

        filenames = os.listdir(self.videoDB_path)
        random.shuffle(filenames)

        dir_list = []
        size_directories = []
        img_sizes = [30, 60, 120, self.image_dim]
        labeled_imgs_dir = './tmp/' + self.classifier_name + '/FAST1/labeled_images'

        for i, size in enumerate(img_sizes):
            size_directories.append(os.path.join(labeled_imgs_dir, "{}".format(size)))
            if not os.path.exists(size_directories[i]):
                os.makedirs(size_directories[i])
            else:
                return

            dir_list.append([
                os.path.join(size_directories[i], "train", "pos"), 
                os.path.join(size_directories[i], "validation", "pos"),
                os.path.join(size_directories[i], "cascade", "pos"),
                os.path.join(size_directories[i], "train", "neg"),
                os.path.join(size_directories[i], "validation", "neg"),
                os.path.join(size_directories[i], "cascade", "neg")
            ])

        # if labeled_imgs_dir not in os.listdir('.') or self.classifier_name not in os.listdir(labeled_imgs_dir):
        for each_list in dir_list:
            for each in each_list:
                if not os.path.exists(each):
                    os.makedirs(each)
        
        user_infer_time = 0

        # Load detectron2 model 
        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)

        for i, file in enumerate(filenames):
            print(file)
            cap = cv2.VideoCapture(os.path.join(self.videoDB_path, file))

            while True:
                # Capture frame-by-frame
                ret = cap.grab()
                if not ret:
                    break

                frame_read += 1

                if frame_read % interval == 0:
                    # the above line won't work for interval == 1
                    ret, frame = cap.retrieve()
                    start_time = time.time()
                    outputs = predictor(frame)
                    instances = outputs["instances"]
                    # pred_class is idx, not string; need coco_names[pred_class.item()] to access class name 
                    pred_classes = instances.pred_classes
                    result = 0
                    for pred_class in pred_classes:
                        # 1: bicycle, 2: car
                        if pred_class.item() == 2:
                            result = 1
                            break
                    user_infer_time += (time.time() - start_time)
                    frame_extracted += 1

                    # Frame predicted as positive
                    if result == 1:
                        # Frame assigned to the training dataset
                        if i % 2 == 0:
                            tr_pos.append(frame)
                            tr_pos_count += 1
                            print("Train positive:", tr_pos_count)
                        # Frame assigned to the validation dataset
                        else:
                            va_cas_pos.append(frame)
                            va_cas_pos_count += 1
                            print("Validation positive:", va_cas_pos_count)
                    # Frame predicted as false
                    else:
                        # Frame assigned to the training dataset
                        if i % 2 == 0:
                            # Avoid using too much cpu and being killed
                            if tr_neg_count > 10000:
                                break
                            tr_neg.append(frame)
                            tr_neg_count += 1
                            print("Train negtive:", tr_neg_count)
                        # Frame assigned to the validation dataset
                        else:
                            # Avoid using too much cpu and being killed
                            if va_cas_neg_count > 10000:
                                break
                            va_cas_neg.append(frame)
                            va_cas_neg_count += 1
                            print("Validation negtive:", va_cas_neg_count)

                    if tr_pos_count >= tr_split[0] and va_cas_pos_count >= va_cas_pos_total and tr_neg_count >= tr_split[1] and va_cas_neg_count >= va_cas_neg_total:
                        break
            
            # Release the Video Device if ret is false
            cap.release()
            # Message to be displayed after releasing the device
            print("Released Video Resource")

            if tr_pos_count >= tr_split[0] and va_cas_pos_count >= va_cas_pos_total and tr_neg_count >= tr_split[1] and va_cas_neg_count >= va_cas_neg_total:
                break

        # cv2.destroyAllWindows()

        user_infer_time /= frame_extracted
        
        # Ratio for constructing the unbalanced cascade dataset
        ratio = tr_pos_count / (tr_neg_count + tr_pos_count)
        assert ratio <= 1

        # Shuffle and prune the training dataset to ensure its balance
        random.shuffle(tr_pos)
        random.shuffle(tr_neg)

        if len(tr_pos) > tr_split[0]:
            tr_pos = tr_pos[:tr_split[0]]
        if len(tr_neg) > tr_split[1]:
            tr_neg = tr_neg[:tr_split[1]]

        # Construct the balanced validation dataset and the unbalanced cascade dataset
        random.shuffle(va_cas_pos)
        random.shuffle(va_cas_neg)

        va_pos = va_cas_pos[:va_split[0]]
        va_neg = va_cas_neg[:va_split[1]]

        # cas_pos_count = int((cas_split[0] + cas_split[1]) * ratio)
        # cas_neg_count = cas_split[0] + cas_split[1] - cas_pos_count
        cas_pos_count = cas_split[0] // 5
        cas_neg_count = cas_split[0] - cas_pos_count
        cas_pos = va_cas_pos[-cas_pos_count:]
        cas_neg = va_cas_neg[-cas_neg_count:]

        datasets = [tr_pos, va_pos, cas_pos, tr_neg, va_neg, cas_neg]

        for i in range(len(datasets)):
            for j, frame in enumerate(datasets[i]):
                for k, size in enumerate(img_sizes):
                    new_frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_CUBIC)
                    filename = dir_list[k][i] + "/{}.jpg".format(j)
                    cv2.imwrite(filename, new_frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    # print("Write to disk {}".format(filename))

        with open(os.path.join(labeled_imgs_dir, "user_infer_time.json"), "w") as f:
            json.dump({'user_infer_time': user_infer_time}, f)

        tr_weight = [len(tr_pos)/(len(tr_pos) + len(tr_neg)), len(tr_neg)/(len(tr_pos) + len(tr_neg))] 
        with open(os.path.join(labeled_imgs_dir, "training_weight.json"), "w") as f:
            json.dump({'training_weight': tr_weight}, f)


    def _exp_generate_labels(self, tr_split=[500, 500], va_split=[200, 200], cas_split=[500, 500], interval=1):
        # Ensure training and validation images are extracted from different videos
        validation_interval = interval * 5
        tr_pos_count, tr_neg_count = 0, 0
        tr_pos, tr_neg = [], []

        va_cas_pos_count, va_cas_neg_count = 0, 0
        va_cas_pos, va_cas_neg = [], []

        va_cas_pos_total = va_split[0] + cas_split[0]
        va_cas_neg_total = va_split[1] + cas_split[1]

        frame_read, frame_extracted = 0, 0

        filenames = os.listdir(self.videoDB_path)
        random.shuffle(filenames)

        dir_list = []
        size_directories = []
        img_sizes = [30, 60, 120, 224, self.image_dim]
        labeled_imgs_dir = './tmp/' + self.classifier_name + '/FAST1/labeled_images'
        misclassified_imgs_dir = './tmp/' + self.classifier_name + '/misclassified_images'

        for i, size in enumerate(img_sizes):
            size_directories.append(os.path.join(labeled_imgs_dir, "{}".format(size)))
            if not os.path.exists(size_directories[i]):
                os.makedirs(size_directories[i])
            else:
                return

            dir_list.append([
                os.path.join(size_directories[i], "train", "pos"), 
                os.path.join(size_directories[i], "validation", "pos"),
                os.path.join(size_directories[i], "cascade", "pos"),
                os.path.join(size_directories[i], "train", "neg"),
                os.path.join(size_directories[i], "validation", "neg"),
                os.path.join(size_directories[i], "cascade", "neg")
            ])

        # if labeled_imgs_dir not in os.listdir('.') or self.classifier_name not in os.listdir(labeled_imgs_dir):
        for each_list in dir_list:
            for each in each_list:
                if not os.path.exists(each):
                    os.makedirs(each)

        if not os.path.exists(misclassified_imgs_dir):
            os.makedirs(misclassified_imgs_dir)
        
        user_infer_time = 0
        misclassified_count = 0
        df = pd.read_csv(config("video_season_label_csv_file"))

        for i, file in enumerate(filenames):
            video_path = os.path.join(self.videoDB_path, file)
            cap = cv2.VideoCapture(video_path)

            try:
                # For video label of season, use this line
                if df.loc[df['video_path'] == video_path].iloc[0]['label'] == config("true_label"):
                    label = 1 
                else:
                    label = 0
                # For video label of day-night, use this line
                # label = df.loc[df['video_name'] == video_path].iloc[0]['label']
            except (KeyError, IndexError) as error:
                continue

            while True:
                # Capture frame-by-frame
                ret = cap.grab()
                if not ret:
                    break

                frame_read += 1 # how many frames we have grabbed 

                if frame_read % interval == 0:
                    ret, frame = cap.retrieve()
                    frame_copy = frame.copy()
                    processed = self._preprocess(frame_copy, False)
                    # print("subtraction: ", frame - frame_copy)
                    start_time = time.time()
                    result = self._predict(processed)
                    user_infer_time += (time.time() - start_time)

                    frame_extracted += 1 # how many frames we have actually read

                    if label != result.item():
                        # Misclassified image
                        filename = misclassified_imgs_dir + f"/{misclassified_count}_{result.item()}-{label}.jpg"
                        cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                        misclassified_count += 1
                        continue
                    
                    #### We only want correctly classified image ####

                    # Frame predicted as positive
                    if result.item() == 1:
                        # Frame assigned to the training dataset
                        if i % 2 == 0:
                            tr_pos.append(frame)
                            tr_pos_count += 1
                            print("Train positive:", tr_pos_count)
                        # Frame assigned to the validation dataset
                        else:
                            # add every 1 out of 180 frames to validation set
                            if frame_read % validation_interval == 0: 
                                va_cas_pos.append(frame)
                                va_cas_pos_count += 1
                                print("Validation positive:", va_cas_pos_count)
                    # Frame predicted as false
                    else:
                        # Frame assigned to the training dataset
                        if i % 2 == 0:
                            # Avoid using too much cpu and being killed
                            if tr_neg_count > 10000:
                                break
                            tr_neg.append(frame)
                            tr_neg_count += 1
                            print("Train negtive:", tr_neg_count)
                        # Frame assigned to the validation dataset
                        else:
                            # add every 1 out of 180 frames to validation set
                            if frame_read % validation_interval == 0: 
                                # Avoid using too much cpu and being killed
                                if va_cas_neg_count > 10000:
                                    break
                                va_cas_neg.append(frame)
                                va_cas_neg_count += 1
                                print("Validation negtive:", va_cas_neg_count)

                    if tr_pos_count >= tr_split[0] and va_cas_pos_count >= va_cas_pos_total and tr_neg_count >= tr_split[1] and va_cas_neg_count >= va_cas_neg_total:
                        break

            # Release the capture
            cap.release()
            # Message to be displayed after releasing the capture
            print("Released Video Resource")

            if tr_pos_count >= tr_split[0] and va_cas_pos_count >= va_cas_pos_total and tr_neg_count >= tr_split[1] and va_cas_neg_count >= va_cas_neg_total:
                break

        # cv2.destroyAllWindows()

        user_infer_time /= frame_extracted
        print("misclassification ratio:", misclassified_count / frame_extracted)
        # Ratio for constructing the unbalanced cascade dataset
        ratio = tr_pos_count / (tr_neg_count + tr_pos_count)
        assert ratio <= 1

        # Shuffle and prune the training dataset to ensure its balance
        random.shuffle(tr_pos)
        random.shuffle(tr_neg)

        if len(tr_pos) > tr_split[0]:
            tr_pos = tr_pos[:tr_split[0]]
        if len(tr_neg) > tr_split[1]:
            tr_neg = tr_neg[:tr_split[1]]

        # Construct the balanced validation dataset and the unbalanced cascade dataset
        random.shuffle(va_cas_pos)
        random.shuffle(va_cas_neg)

        va_pos = va_cas_pos[:va_split[0]]
        va_neg = va_cas_neg[:va_split[1]]

        #cas_pos_count = int((cas_split[0] + cas_split[1]) * ratio)
        #cas_neg_count = cas_split[0] + cas_split[1] - cas_pos_count
        cas_pos_count = cas_split[0] // 5
        cas_neg_count = cas_split[0] - cas_pos_count
        cas_pos = va_cas_pos[-cas_pos_count:]
        cas_neg = va_cas_neg[-cas_neg_count:]

        datasets = [tr_pos, va_pos, cas_pos, tr_neg, va_neg, cas_neg]

        for i in range(len(datasets)):
            for j, frame in enumerate(datasets[i]):
                for k, size in enumerate(img_sizes):
                    new_frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_CUBIC)
                    filename = dir_list[k][i] + "/{}.jpg".format(j)
                    cv2.imwrite(filename, new_frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    # print("Write to disk {}".format(filename))

        with open(os.path.join(labeled_imgs_dir, "user_infer_time.json"), "w") as f:
            json.dump({'user_infer_time': user_infer_time}, f)

        tr_weight = [len(tr_pos)/(len(tr_pos) + len(tr_neg)), len(tr_neg)/(len(tr_pos) + len(tr_neg))] 
        with open(os.path.join(labeled_imgs_dir, "training_weight.json"), "w") as f:
            json.dump({'training_weight': tr_weight}, f)

    def _generate_balanced_set(self, num_instance, interval):
        pos_count, neg_count, pos, neg = 0, 0, [], []
        total_count = 0 
        filenames = os.listdir(self.videoDB_path)
        random.shuffle(filenames)
        frames_read = 0
        img_sizes = [30, 60, 120, 128, 224]

        contrad_count = 0

        for file in filenames:
            cap = cv2.VideoCapture(os.path.join(self.videoDB_path, file))
            while True:
                frames_read += 1
                # Capture frame-by-frame
                ret = cap.grab()
                if not ret:
                    # Release the Video Device if ret is false
                    cap.release()
                    # Message to be displayed after releasing the device
                    print("Released Video Resource")
                    break

                if frames_read % interval == 0:
                    ret, frame = cap.retrieve()
                    
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    processed = self._preprocess(frame, False)
                    # cv2.imwrite('./abc.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    # frame_second = cv2.imread('./abc.jpg')
                    # if not np.array_equal(frame, frame_second): print('in trouble')
                    result = self._predict(processed)

                    total_count += 1
                    if result[0]:
                        pos.append((frame, 1))
                        pos_count += 1
                        print("positive:", pos_count)
                    elif not result[0]:
                        neg.append((frame, 0))
                        neg_count += 1
                        print("negative:", neg_count)
                    if pos_count >= num_instance and neg_count >= num_instance:
                        break
                    
            if pos_count >= num_instance and neg_count >= num_instance:
                break
        random.shuffle(pos)
        #print(len(pos))
        if len(pos) > num_instance:
            pos = pos[:num_instance]
            #print(len(pos))
        random.shuffle(neg)
        #print(len(neg))
        if len(neg) > num_instance:
            neg = neg[:num_instance]
            #print(len(neg))
        
        pos.extend(neg)
        #print(len(pos))
        frame_list = pos
        random.shuffle(frame_list)
        total, count = 0, 0
        for (frame, pred) in frame_list:
            processed = self._preprocess(frame, False)
            result = self._predict(processed)
            total += 1
            if result[0]:
                count += 1
            if count >= 100000:
                print('found:', count)
                return
        print('found:', count)
        print('contradict count', contrad_count)
        print('total', total_count)
        return 

        for size in img_sizes:
            os.makedirs("./tmp/" + self.classifier_name + "/balanced_set_img/{}".format(size))

        for size in img_sizes:
            for i, frame in enumerate(frame_list):
                filename = "./tmp/" + self.classifier_name + "/balanced_set_img/{}".format(size) + "/{}.jpg".format(i)
                cv2.imwrite(filename, cv2.resize(frame, (size, size), interpolation=cv2.INTER_CUBIC))


    def _preprocess(self, frame, resized):
        # Resize the frame if it is not already resized
        if not resized:
            frame = cv2.resize(frame, (self.image_dim, self.image_dim), interpolation=cv2.INTER_CUBIC)
        
        # Normalize the frame
        frame = frame.astype(np.float)
        frame_new = (frame - self.mean) / self.std

        # Transpose the color channel
        frame_new = frame_new.transpose(2, 0, 1)

        return torch.from_numpy(frame_new).float().unsqueeze(0).to(self.device)

    def _predict(self, frame):
        #Frame is a pre-processed tensor
        with torch.no_grad():
            output = self.user_classifier(frame)
            result = predictions(output.data)
        return result

    def getVariables(self):
        return self.num_process, self.classify_time, self.fps_time, self.read_cap, self.construct_cap

    def _compute_loading_time(self):
        ''' 
        This function computes the loading time by sequentially read 50000 frames from all sizes of videos
        '''
        load_time_json = "./tmp/" + self.classifier_name + "/loading_time.json"
        stad_time_json = "./tmp/" + self.classifier_name + "/standardize_time.json"

        if os.path.exists(load_time_json):
            with open(stad_time_json) as f:
                self.standardize_time = json.load(f)
            with open(load_time_json) as f:
                self.load_time = json.load(f)
                # if the loading time of current user classifier's image size has already been measured, return
                if self.image_dim in self.load_time:
                    return
        
        # if the loading time of the 4 sizes are already measured, we only need to measure ones related to the user classifiers' image size, i.e. self.image_dim
        # size_list = [(30, 30), (60, 60), (120, 120), (224, 224), (self.image_dim, 0), (self.image_dim, 1), (self.image_dim, 2), (self.image_dim, 3), (self.image_dim, self.image_dim)] if 30 not in self.load_time else [(self.image_dim,0), (self.image_dim,1), (self.image_dim,2), (self.image_dim,3), (self.image_dim,self.image_dim)]
        # size_list = [(224, 224)]
        size_list = [(30, 30), (60, 60), (120, 120), (224, 224)]
        FRAME_LIMIT = 3000
        preprocessors = ['BWImage', 'ColorImage', 'SingleChannelImage']

        for size in size_list:            
            key = 'x'.join([str(a) for a in size]) # stringify the size representation to store
            self.load_time[key] = 0
            self.standardize_time[key] = {'BWImage': 0, 'ColorImage': 0, 'SingleChannelImage': 0}

            load_start_time = time.time()
            frame_count = 0

            for file in os.listdir(self.videoDB_path):
                cap = cv2.VideoCapture(os.path.join(self.resized_videoDB_path, key, file))
                if not cap.isOpened():
                    print("Error opening video stream or file: ", file)
                else:
                    # Read and resize frames.
                    while cap.isOpened():
                        ret, frame = cap.read()

                        if ret == True: # current frame is effective
                            self.load_time[key] += (time.time() - load_start_time)
                            # standardize
                            for preprocessor in preprocessors:
                                frame_copy = np.copy(frame)
                                standardize_start_time = time.time()

                                if preprocessor == 'BWImage' and size[1] != self.image_dim:
                                    frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
                                    frame_copy = np.expand_dims(frame_copy, axis=2)
                                    frame_copy = frame_copy.astype(np.float)
                                    arr = frame_copy[:, :, 0]
                                    new_arr = (arr - self.mean_models[key][3]) / self.std_models[key][3]
                                    frame_copy[:, :, 0] = new_arr
                                elif preprocessor == 'SingleChannelImage':
                                    frame_copy = frame_copy[:, :, 0]
                                    frame_copy = np.expand_dims(frame_copy, axis=2)
                                    frame_copy = frame_copy.astype(np.float)
                                    arr = frame_copy[:, :, 0]
                                    new_arr = (arr - self.mean_models[key][0]) / self.std_models[key][0]
                                    frame_copy[:, :, 0] = new_arr
                                else:
                                    frame_copy = frame_copy.astype(np.float)
                                    for i in range(0, 3):
                                        arr = frame_copy[:, :, i]
                                        new_arr = (arr - self.mean_models[key][i]) / self.std_models[key][i]
                                        frame_copy[:, :, i] = new_arr

                                frame_copy = frame_copy.transpose(2, 0, 1)
                                frame_copy = torch.from_numpy(frame_copy).float().unsqueeze(0)
                                self.standardize_time[key][preprocessor] += (time.time() - standardize_start_time)

                            frame_count += 1

                            if frame_count == FRAME_LIMIT:
                                break

                            load_start_time = time.time()
                        else:
                            break                                                         
                cap.release()

                if frame_count == FRAME_LIMIT:
                    break

            self.load_time[key] /= FRAME_LIMIT
 
            for preprocessor in preprocessors:
                self.standardize_time[key][preprocessor] /= FRAME_LIMIT
                           
        # cv2.destroyAllWindows()      
        print("Measured video load time,", self.load_time)
  
        with open(load_time_json, "w") as f:
            json.dump(self.load_time, f)

        with open(stad_time_json, "w") as f:
            json.dump(self.standardize_time, f)

    def _plot_pareto_frontiers(self, frontiers, frontiers_precision, user_accuracy_cascade, user_precision_cascade, all_cascades, thresh):
        # list of (id, fps, accuracy, threshold)
        print("Pareto Optimal: (Accuracy, FPS)")
        print(["({:.10f}, {:.10f})".format(x[2], x[1]) for x in frontiers])

        print("Pareto Optimal Precision: (Precision, FPS)")
        print(["({:.10f}, {:.10f})".format(x[3], x[1]) for x in frontiers_precision])

        plt.figure(figsize=[12,10])

        fps_all = [x[1] for x in all_cascades]
        acc_all = [x[2] for x in all_cascades]

        fps_dominant = [x[1] for x in frontiers]
        acc_dominant = [x[2] for x in frontiers]

        plt.plot(fps_all, acc_all, linestyle='None', marker='o', markersize=4, color='C0', alpha=.5)
        plt.plot(fps_dominant, acc_dominant, linestyle='None', marker='o',color='red', alpha=.9, markersize=4, markeredgecolor='darkred')
        plt.plot(user_accuracy_cascade[1], user_accuracy_cascade[2], linestyle='None', marker='x', markersize=4, color='k')
        
        # plt.xlim([-.05*max(fps_maxes),max(fps_maxes)+.05*max(fps_maxes)])
        # plt.ylim([min(acc_mins)-.01,1.01])

        plt.xlabel("Throughput (fps)")
        plt.ylabel("Accuracy")
        # plt.legend(['Us','NoScope Options', 'NoScope Choice', 'ResNet', '0 cost', 'Pareto'])

        # plt.title('Accuracy vs. throughput - %s, User Thresh: %.2f, Cost Multiplier: %.2f' % (imagenet_name, thresh, load_multiplier))
        plots_dir = f'./tmp/{self.classifier_name}/frontier_plots/'

        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        acc_plot = os.path.join(plots_dir, f'frontier_plot_{str(thresh)}.png')
        plt.savefig(acc_plot, dpi=200)

        plt.close()
        plt.figure(figsize=[12,10])

        precision_all = [x[3] for x in all_cascades]
        precision_dominant = [x[3] for x in frontiers]
        fps_dominant_pre = [x[1] for x in frontiers_precision]
        precision_dominant_pre = [x[3] for x in frontiers_precision]
        #acc_dominant_pre = [x[2] for x in frontiers_precision]

        plt.plot(fps_all, precision_all, linestyle='None', marker='o', markersize=4, color='C0', alpha=.5)
        plt.plot(fps_dominant_pre, precision_dominant_pre, linestyle='None', marker='o',color='red', alpha=.9, markersize=4, markeredgecolor='darkred')
        plt.plot(user_precision_cascade[1], user_precision_cascade[3], linestyle='None', marker='x', markersize=4, color='k')

        plt.xlabel("Throughput (fps)")
        plt.ylabel("Precision")

        prec_plot = os.path.join(plots_dir, f'precision_frontier_plot_{str(thresh)}.png')
        plt.savefig(prec_plot, dpi=200)




        



      