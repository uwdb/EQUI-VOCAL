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
from utils import tools
from frame_selection import RandomFrameSelection
from user_feedback import UserFeedback
from proxy_model_training import ProxyModelTraining
from query_initialization import RandomInitialization
from prepare_query_ground_truth import *
import filter 

annotated_batch_size = 1
materialized_batch_size = 16
# init_sampling_step = 100
init_sampling_step = 1

class ComplexEventVideoDB:
    def __init__(self, bbox_file = "/home/ubuntu/complex_event_video/data/car_turning_traffic2/bbox.json"):
        self.spatial_feature_dim = 5
        # Read in bbox info
        with open(bbox_file, 'r') as f:
            self.maskrcnn_bboxes = json.loads(f.read())
        self.n_frames = len(self.maskrcnn_bboxes)

        # Get ground-truth labels
        # self.pos_frames, self.pos_frames_per_instance = turning_car_and_pedestrain_at_intersection()
        self.pos_frames, self.pos_frames_per_instance = test_a(self.maskrcnn_bboxes)
        self.n_positive_instances = len(self.pos_frames_per_instance)
        self.avg_duration = 1.0 * len(self.pos_frames) / self.n_positive_instances
        print(self.avg_duration)
        self.num_positive_instances_found = 0
        self.raw_frames = np.full(self.n_frames, True, dtype=np.bool)
        self.materialized_frames = np.full(self.n_frames, False, dtype=np.bool)

        # Initially, materialize 1/init_sampling_step (e.g. 1%) of all frames.
        for i in range(0, self.n_frames, init_sampling_step):
            self.raw_frames[i] = False
            self.materialized_frames[i] = True

        self.negative_frames_seen = []
        self.positive_frames_seen = []
        self.p = np.ones(self.n_frames)
        
        self.spatial_features = np.zeros((self.n_frames, self.spatial_feature_dim), dtype=np.float64)
        self.Y = np.zeros(self.n_frames, dtype=np.int)
        for i in range(self.n_frames):
            if i in self.pos_frames:
                self.Y[i] = 1
        self.plot_data_y_annotated = np.array([0])
        self.plot_data_y_materialized = np.array([self.materialized_frames.nonzero()[0].size])

        # Filtering stage 
        self.candidates = np.full(self.n_frames, True, dtype=np.bool)
        for frame_id in range(self.n_frames):
            res_per_frame = self.maskrcnn_bboxes["frame_{}.jpg".format(frame_id)]
            # is_candidate, bbox = filter.car_and_pedestrain_at_intersection(res_per_frame, frame_id)
            is_candidate, bbox = filter.test_a(res_per_frame, frame_id)
            if not is_candidate:
                self.candidates[frame_id] = False
            else:
                self.spatial_features[frame_id] = self.construct_spatial_feature(bbox)

        # ExSample initialization
        self.number_of_chunks = 1
        self.stats_per_chunk = [[0, 0] for _ in range(self.number_of_chunks)] # N^1 and n

        self.query_initialization = RandomInitialization(self.pos_frames, self.candidates)
        self.frame_selection = RandomFrameSelection(self.spatial_features, self.Y, materialized_batch_size, annotated_batch_size, self.avg_duration, self.candidates)
        self.user_feedback = UserFeedback(self.n_frames, self.pos_frames)
        self.proxy_model_training = ProxyModelTraining(self.spatial_features, self.Y)

    def run(self):
        self.materialized_frames, self.positive_frames_seen, self.negative_frames_seen, self.pos_frames_per_instance, self.num_positive_instances_found, self.plot_data_y_annotated, self.plot_data_y_materialized, self.stats_per_chunk = self.query_initialization.run(self.materialized_frames, self.positive_frames_seen, self.negative_frames_seen, self.pos_frames_per_instance, self.num_positive_instances_found, self.plot_data_y_annotated, self.plot_data_y_materialized, self.stats_per_chunk)

        self.clf = self.proxy_model_training.run(self.raw_frames, self.materialized_frames)

        frame_id_arr_to_annotate = np.empty(annotated_batch_size)  # Construct a pseudo list of length BATCH_SIZE to pass the While condition.
        while self.num_positive_instances_found < self.n_positive_instances and frame_id_arr_to_annotate.size >= annotated_batch_size:
            self.update_random_choice_p()
            frame_id_arr_to_annotate, self.raw_frames, self.materialized_frames, self.stats_per_chunk = self.frame_selection.run(self.p, self.clf, self.raw_frames, self.materialized_frames, self.positive_frames_seen, self.stats_per_chunk)

            self.positive_frames_seen, self.negative_frames_seen, self.pos_frames_per_instance, self.num_positive_instances_found, self.plot_data_y_annotated, self.plot_data_y_materialized, self.stats_per_chunk = self.user_feedback.run(frame_id_arr_to_annotate, self.positive_frames_seen, self.negative_frames_seen, self.pos_frames_per_instance, self.num_positive_instances_found, self.plot_data_y_annotated, self.plot_data_y_materialized, self.materialized_frames.nonzero()[0].size, self.stats_per_chunk)

            self.clf = self.proxy_model_training.run(self.raw_frames, self.materialized_frames)

            self.get_frames_stats()
        print("stats_per_chunk", self.stats_per_chunk)
        return self.plot_data_y_annotated, self.plot_data_y_materialized

    def construct_spatial_feature(self, bbox):
        x1, y1, x2, y2 = bbox
        centroid_x = (x1 + x2) / 2
        centroid_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        wh_ratio = width / height
        return np.array([centroid_x, centroid_y, width, height, wh_ratio])

    def update_random_choice_p(self):
        """Given positive frames seen, compute the probabilities associated with each frame for random choice.
        Frames that are close to a positive frame that have been seen are more likely to be positive as well, thus should have a smaller probability of returning to user for annotations.
        Heuristic: For each observed positive frame, the probability function is 0 at that observed frame, grows ``linearly'' as the distance from the observed frame increases, and becomes constantly 1 after AVG_DURATION distance on each side.
        TODO: considering cases when two observed positive frames are close enough that their probability functions overlap.
        Return: Numpy_Array[proba]
        """
        self.p = np.ones(self.n_frames)

        scale = 0.5
        func = lambda x : (x ** 2) / (int(self.avg_duration * scale) ** 2)
        for frame_id in self.positive_frames_seen:
           # Right half of the probability function
            for i in range(int(self.avg_duration * scale) + 1):
                if frame_id + i < self.n_frames:
                    self.p[frame_id + i] = min(func(i), self.p[frame_id + i])
            # Left half of the probability function
            for i in range(int(self.avg_duration * scale) + 1):
                if frame_id - i >= 0:
                    self.p[frame_id - i] = min(func(i), self.p[frame_id - i])

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

        self.n_frames = len(self.maskrcnn_bboxes)
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
        self.raw_frames = np.full(self.n_frames, True, dtype=np.bool)
        self.materialized_frames = np.full(self.n_frames, False, dtype=np.bool)

        # Initially, materialize 1/init_sampling_step (e.g. 1%) of all frames.
        for i in range(0, self.n_frames, init_sampling_step):
            self.raw_frames[i] = False
            self.materialized_frames[i] = True

        self.negative_frames_seen = []
        self.positive_frames_seen = []
        self.p = np.ones(self.n_frames)
        self.candidates = np.full(self.n_frames, False, dtype=np.bool)
        self.spatial_features = np.zeros((self.n_frames, self.spatial_feature_dim), dtype=np.float64)
        self.Y = np.zeros(self.n_frames, dtype=np.int)
        for i in range(self.n_frames):
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

class IterativeProcessing(ComplexEventVideoDB):
    def __init__(self) -> None:
        super().__init__()

    def run(self):
        while self.num_positive_instances_found < self.n_positive_instances:
            self.update_random_choice_p()
            arr = self.materialized_frames * self.candidates
            normalized_p = self.p[arr] / self.p[arr].sum()
            frame_id = np.random.choice(arr.nonzero()[0], p=normalized_p)
            self.materialized_frames[frame_id] = False

            if frame_id in self.pos_frames:
                for key, (start_frame, end_frame, flag) in self.pos_frames_per_instance.items():
                    if start_frame <= frame_id and frame_id < end_frame:
                        if flag == 0:
                            self.num_positive_instances_found += 1
                            print("num_positive_instances_found:", self.num_positive_instances_found)
                            self.plot_data_y_materialized = np.append(self.plot_data_y_materialized, self.materialized_frames.nonzero()[0].size)
                            self.pos_frames_per_instance[key] = (start_frame, end_frame, 1)
                        else:
                            del self.pos_frames_per_instance[key]
                        break
                self.positive_frames_seen.append(frame_id)
                self.plot_data_y_annotated = np.append(self.plot_data_y_annotated, self.num_positive_instances_found)
                self.get_frames_stats()
            else:
                self.negative_frames_seen.append(frame_id)
                self.plot_data_y_annotated = np.append(self.plot_data_y_annotated, self.num_positive_instances_found)
        return self.plot_data_y_annotated, self.plot_data_y_materialized


class RandomProcessing(ComplexEventVideoDB):
    def __init__(self) -> None:
        super().__init__()

    def run(self):
        while self.num_positive_instances_found < self.n_positive_instances:
            self.update_random_choice_p()
            arr = self.materialized_frames
            normalized_p = self.p[arr] / self.p[arr].sum()
            frame_id = np.random.choice(arr.nonzero()[0], p=normalized_p)
            self.materialized_frames[frame_id] = False

            if frame_id in self.pos_frames:
                for key, (start_frame, end_frame, flag) in self.pos_frames_per_instance.items():
                    if start_frame <= frame_id and frame_id < end_frame:
                        if flag == 0:
                            self.num_positive_instances_found += 1
                            print("num_positive_instances_found:", self.num_positive_instances_found)
                            self.plot_data_y_materialized = np.append(self.plot_data_y_materialized, self.materialized_frames.nonzero()[0].size)
                            self.pos_frames_per_instance[key] = (start_frame, end_frame, 1)
                        else:
                            del self.pos_frames_per_instance[key]
                        break
                self.positive_frames_seen.append(frame_id)
                self.plot_data_y_annotated = np.append(self.plot_data_y_annotated, self.num_positive_instances_found)
                self.get_frames_stats()
            else:
                self.negative_frames_seen.append(frame_id)
                self.plot_data_y_annotated = np.append(self.plot_data_y_annotated, self.num_positive_instances_found)
        return self.plot_data_y_annotated, self.plot_data_y_materialized

        
if __name__ == '__main__':
    plot_data_y_annotated_list = []
    plot_data_y_materialized_list = []
    for _ in range(20):
        cevdb = ComplexEventVideoDB()
        # cevdb = RandomProcessing()
        plot_data_y_annotated, plot_data_y_materialized = cevdb.run()
        plot_data_y_annotated_list.append(plot_data_y_annotated)
        plot_data_y_materialized_list.append(plot_data_y_materialized)
    cevdb.save_data(plot_data_y_annotated_list, "iterative_test_a")
    # cevdb.save_data(plot_data_y_annotated_list, "random_test_a")
    # cevdb.save_data(plot_data_y_materialized_list, "filtered_materialized")