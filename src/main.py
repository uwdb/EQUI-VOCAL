from time import time
from shutil import copyfile
import json
from typing import List, Tuple
import torch
import math
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import csv
from tqdm import tqdm
from sklearn import tree, metrics
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import graphviz
import random
import more_itertools as mit
import matplotlib.pyplot as plt
import sys
from utils.utils import isInsideIntersection, isOverlapping
from utils import tools
from frame_selection import RandomFrameSelection
from user_feedback import UserFeedback
from proxy_model_training import ProxyModelTraining
from query_initialization import RandomInitialization

annotated_batch_size = 1
materialized_batch_size = 16
init_sampling_step = 100

class ComplexEventVideoDB:
    def __init__(self, bbox_file = "/home/ubuntu/complex_event_video/data/car_turning_traffic2/bbox.json"):
        self.spatial_feature_dim = 5
        self.edge_corner_bbox = (367, 345, 540, 418)
        # Read in bbox info
        with open(bbox_file, 'r') as f:
            self.maskrcnn_bboxes = json.loads(f.read())

        # Read labels
        self.pos_frames = []
        self.pos_frames_per_instance = {}
        with open("/home/ubuntu/complex_event_video/data/annotation.csv", "r") as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)
            for i, row in enumerate(csvreader):
                start_frame, end_frame = int(row[0]), int(row[1])
                self.pos_frames += list(range(start_frame, end_frame+1))
                self.pos_frames_per_instance[i] = (start_frame, end_frame+1, 0) # The third value is a flag: 0 represents no detections have been found; 1 represents detection with only one match

        self.num_positive_instances_found = 0
        # frames_unseen --> raw_frames
        self.raw_frames = np.full(len(self.maskrcnn_bboxes), True, dtype=np.bool)
        self.materialized_frames = np.full(len(self.maskrcnn_bboxes), False, dtype=np.bool)

        # Initially, materialize 1/init_sampling_step (e.g. 1%) of all frames.
        for i in range(0, len(self.maskrcnn_bboxes), init_sampling_step):
            self.raw_frames[i] = False
            self.materialized_frames[i] = True

        self.negative_frames_seen = []
        self.positive_frames_seen = []
        self.p = np.ones(len(self.maskrcnn_bboxes))
        self.candidates = np.full(len(self.maskrcnn_bboxes), False, dtype=np.bool)
        self.spatial_features = np.zeros((len(self.maskrcnn_bboxes), self.spatial_feature_dim), dtype=np.float64)
        self.Y = np.zeros(len(self.maskrcnn_bboxes), dtype=np.int)
        for i in range(len(self.maskrcnn_bboxes)):
            if i in self.pos_frames:
                self.Y[i] = 1
        self.plot_data_y_annotated = np.array([0])
        self.plot_data_y_materialized = np.array([self.materialized_frames.nonzero()[0].size])
        self.clf = RandomForestClassifier(
            # criterion="entropy",
            # max_depth=10,
            n_estimators=10,
            # min_samples_split=32,
            class_weight="balanced"
        )
        self.get_candidates()

        # ExSample initialization
        self.number_of_chunks = 2
        self.stats_per_chunk = [[0, 0] for _ in range(self.number_of_chunks)] # N^1 and n


        self.query_initialization = RandomInitialization(self.pos_frames, self.spatial_features, self.Y, self.candidates)
        self.frame_selection = RandomFrameSelection(self.spatial_features, self.Y, materialized_batch_size, annotated_batch_size)
        self.user_feedback = UserFeedback(self.pos_frames, self.candidates)
        self.proxy_model_training = ProxyModelTraining(self.spatial_features, self.Y)

    def run(self):
        self.materialized_frames, self.positive_frames_seen, self.negative_frames_seen, self.clf, self.pos_frames_per_instance, self.num_positive_instances_found, self.plot_data_y_annotated, self.plot_data_y_materialized, self.stats_per_chunk = self.query_initialization.run(self.raw_frames, self.materialized_frames, self.positive_frames_seen, self.negative_frames_seen, self.pos_frames_per_instance, self.num_positive_instances_found, self.plot_data_y_annotated, self.plot_data_y_materialized, self.stats_per_chunk)

        frame_id_arr_to_annotate = np.empty(annotated_batch_size)  # Construct a pseudo list of length BATCH_SIZE to pass the While condition.
        while self.num_positive_instances_found < 15 and frame_id_arr_to_annotate.size >= annotated_batch_size:

            frame_id_arr_to_annotate, self.raw_frames, self.materialized_frames, self.stats_per_chunk = self.frame_selection.run(self.clf, self.raw_frames, self.materialized_frames, self.positive_frames_seen, self.negative_frames_seen, self.stats_per_chunk)

            self.positive_frames_seen, self.negative_frames_seen, self.pos_frames_per_instance, self.num_positive_instances_found, self.plot_data_y_annotated, self.plot_data_y_materialized, self.stats_per_chunk = self.user_feedback.run(frame_id_arr_to_annotate, self.positive_frames_seen, self.negative_frames_seen, self.pos_frames_per_instance, self.num_positive_instances_found, self.plot_data_y_annotated, self.plot_data_y_materialized, self.materialized_frames.nonzero()[0].size, len(self.maskrcnn_bboxes), self.stats_per_chunk)

            self.clf = self.proxy_model_training.run(self.raw_frames, self.materialized_frames)

            self.get_frames_stats()
        print("stats_per_chunk", self.stats_per_chunk)
        return self.plot_data_y_annotated, self.plot_data_y_materialized

    def get_candidates(self):
        for frame_id in range(len(self.maskrcnn_bboxes)):
            is_candidate, bbox = self.frame_has_objects_of_interest(frame_id)
            if is_candidate:
                self.candidates[frame_id] = True
                self.spatial_features[frame_id] = self.construct_spatial_feature(bbox)

    def frame_has_objects_of_interest(self, frame_id):
        has_car = 0
        has_pedestrian = 0
        res_per_frame = self.maskrcnn_bboxes["frame_{}.jpg".format(frame_id)]
        for x1, y1, x2, y2, class_name, score in res_per_frame:
            if (class_name == "person" and isOverlapping(self.edge_corner_bbox, (x1, y1, x2, y2))):
                has_pedestrian = 1
            elif (class_name in ["car", "truck"] and isInsideIntersection((x1, y1, x2, y2))):
                # Watch the video and identify the correct cars. Hardcode the correct car bbox to use.
                if frame_id >= 14043 and frame_id <= 14079 and (x1 < 500 or x1 > 800):
                    continue
                if frame_id >= 15312 and frame_id <= 15365 and y2 < 450:
                    continue
                if frame_id >= 15649 and frame_id <= 15722 and x1 < 200:
                    continue
                if frame_id >= 16005 and frame_id <= 16044 and y2 < 430:
                    continue
                if frame_id >= 16045 and frame_id <= 16072 and x1 < 250:
                    continue
                if frame_id >= 16073 and frame_id <= 16090 and y2 < 450:
                    continue
                if frame_id >= 16091 and frame_id <= 16122 and x1 < 245:
                    continue
                if frame_id >= 16123 and frame_id <= 16153 and x1 > 500:
                    continue
                if frame_id >= 22375 and frame_id <= 22430 and y2 < 500:
                    continue
                has_car = 1
                car_x1, car_y1, car_x2, car_y2 = x1, y1, x2, y2
            if has_car and has_pedestrian:
                return True, (car_x1, car_y1, car_x2, car_y2)
        return False, None

    def construct_spatial_feature(self, bbox):
        x1, y1, x2, y2 = bbox
        centroid_x = (x1 + x2) / 2
        centroid_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        wh_ratio = width / height
        return np.array([centroid_x, centroid_y, width, height, wh_ratio])

    def get_frames_stats(self):
        print("raw frames: {0}, materialized frames: {1}, negative frames seen: {2}, positive frames seen: {3}.".format(self.raw_frames.nonzero()[0].size, self.materialized_frames.nonzero()[0].size, len(self.negative_frames_seen), len(self.positive_frames_seen)))
        if len(self.positive_frames_seen) == 1:
            print(self.positive_frames_seen)
        # print("The user has seen {0} frames and found {1} positive frames so far.".format(len(self.positive_frames_seen) + len(self.negative_frames_seen), len(self.positive_frames_seen)))

    @staticmethod
    def save_data(plot_data_y_list, method):
        with open("outputs/{}.json".format(method), 'w') as f:
            f.write(json.dumps([arr.tolist() for arr in plot_data_y_list]))

class ComplexEventVideoDBSkewed(ComplexEventVideoDB):
   def __init__(self, bbox_file="/home/ubuntu/complex_event_video/data/car_turning_traffic2/bbox.json"):
        self.spatial_feature_dim = 5
        self.edge_corner_bbox = (367, 345, 540, 418)
        # Read in bbox info
        with open(bbox_file, 'r') as f:
            self.maskrcnn_bboxes = json.loads(f.read())

        # Manually append 22500 negative frames
        bbox_of_last_frame = self.maskrcnn_bboxes["frame_22500.jpg"]
        for i in range(22501, 45001):
            self.maskrcnn_bboxes["frame_{}.jpg".format(i)] = bbox_of_last_frame

        # Read labels
        self.pos_frames = []
        self.pos_frames_per_instance = {}
        with open("/home/ubuntu/complex_event_video/data/annotation.csv", "r") as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)
            for i, row in enumerate(csvreader):
                start_frame, end_frame = int(row[0]), int(row[1])
                self.pos_frames += list(range(start_frame, end_frame+1))
                self.pos_frames_per_instance[i] = (start_frame, end_frame+1, 0) # The third value is a flag: 0 represents no detections have been found; 1 represents detection with only one match

        self.num_positive_instances_found = 0
        # frames_unseen --> raw_frames
        self.raw_frames = np.full(len(self.maskrcnn_bboxes), True, dtype=np.bool)
        self.materialized_frames = np.full(len(self.maskrcnn_bboxes), False, dtype=np.bool)

        # Initially, materialize 1/init_sampling_step (e.g. 1%) of all frames.
        for i in range(0, len(self.maskrcnn_bboxes), init_sampling_step):
            self.raw_frames[i] = False
            self.materialized_frames[i] = True

        self.negative_frames_seen = []
        self.positive_frames_seen = []
        self.p = np.ones(len(self.maskrcnn_bboxes))
        self.candidates = np.full(len(self.maskrcnn_bboxes), False, dtype=np.bool)
        self.spatial_features = np.zeros((len(self.maskrcnn_bboxes), self.spatial_feature_dim), dtype=np.float64)
        self.Y = np.zeros(len(self.maskrcnn_bboxes), dtype=np.int)
        for i in range(len(self.maskrcnn_bboxes)):
            if i in self.pos_frames:
                self.Y[i] = 1
        self.plot_data_y_annotated = np.array([0])
        self.plot_data_y_materialized = np.array([self.materialized_frames.nonzero()[0].size])
        self.clf = RandomForestClassifier(
            # criterion="entropy",
            # max_depth=10,
            n_estimators=10,
            # min_samples_split=32,
            class_weight="balanced"
        )
        self.get_candidates()

        # ExSample initialization
        self.number_of_chunks = 2
        self.stats_per_chunk = [[0, 0] for _ in range(self.number_of_chunks)] # N^1 and n


        self.query_initialization = RandomInitialization(self.pos_frames, self.spatial_features, self.Y, self.candidates)
        self.frame_selection = RandomFrameSelection(self.spatial_features, self.Y, materialized_batch_size, annotated_batch_size)
        self.user_feedback = UserFeedback(self.pos_frames, self.candidates)
        self.proxy_model_training = ProxyModelTraining(self.spatial_features, self.Y)

if __name__ == '__main__':
    plot_data_y_annotated_list = []
    plot_data_y_materialized_list = []
    for _ in range(20):
        cevdb = ComplexEventVideoDBSkewed()
        plot_data_y_annotated, plot_data_y_materialized = cevdb.run()
        plot_data_y_annotated_list.append(plot_data_y_annotated)
        plot_data_y_materialized_list.append(plot_data_y_materialized)
    cevdb.save_data(plot_data_y_annotated_list, "baseline_annotated_exsample_skewed")
    cevdb.save_data(plot_data_y_materialized_list, "baseline_materialized_exsample_skewed")