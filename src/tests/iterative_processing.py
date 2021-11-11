"""Target query:
A car turning at the intersection while a pedestrian also appears at the intersection.

Test video: traffic-2.mp4
Stats: duration: 15 mins. 25 fps. 22500 frames in total.
positive events: 15 instances; 912 frames

Goal:
Find all instances of the target event from the test video. We say the system finds an instance of the target event as long as it finds one frame of that instance; the user can later on play the video directly to find all other frames of that instance since they are temporally adjacent.

Performance metric:
The number of frames users have seen in order to find (1/ one half/ all) positive frames (instances). A plot will be made to visualize the result, where x-axis is the number of frames the user has seen, and y-axis is the number of positive instances the user finds.

To start with, the system returns frames via random sampling. The user annotates them (whether the frame is positive or negative; label the car of interest, etc).
"""
from time import time
from shutil import copyfile
import json
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

np.set_printoptions(threshold=sys.maxsize)

# When BATCH_SIZE > 1, simulate_user_annotation() will break
BATCH_SIZE = 1
EPSILON = 1e-7

# Average duration of the target event (60.8 frames per event)
AVG_DURATION = 61

class IterativeProcessing:
    def __init__(self, n_annotations):
        self.n_annotations = n_annotations
        self.spatial_feature_dim = 5
        self.edge_corner_bbox = (367, 345, 540, 418)
        # Read in bbox info
        bbox_file = "/home/ubuntu/complex_event_video/data/car_turning_traffic2/bbox.json"
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
                self.pos_frames_per_instance[i] = (start_frame, end_frame+1)

        self.num_positive_instances_found = 0
        self.frames_unseen = np.full(len(self.maskrcnn_bboxes), True, dtype=np.bool)
        self.negative_frames_seen = []
        self.positive_frames_seen = []
        self.p = np.ones(len(self.maskrcnn_bboxes))
        self.candidates = np.full(len(self.maskrcnn_bboxes), False, dtype=np.bool)
        self.spatial_features = np.zeros((len(self.maskrcnn_bboxes), self.spatial_feature_dim), dtype=np.float64)
        self.Y = np.zeros(len(self.maskrcnn_bboxes), dtype=np.int)
        for i in range(len(self.maskrcnn_bboxes)):
            if i in self.pos_frames:
                self.Y[i] = 1
        self.plot_data_y = np.array([0])
        # self.clf = tree.DecisionTreeClassifier(
        #     criterion="entropy",
        #     max_depth=10,
        #     min_samples_split=32,
        #     class_weight="balanced"
        #     )
        self.clf = RandomForestClassifier(
            # criterion="entropy",
            # max_depth=10,
            n_estimators=10,
            # min_samples_split=32,
            class_weight="balanced"
        )
        self.get_candidates()

    def get_candidates(self):
        for frame_id in range(len(self.maskrcnn_bboxes)):
            is_candidate, bbox = self.frame_has_objects_of_interest(frame_id)
            if is_candidate:
                self.candidates[frame_id] = True
                self.spatial_features[frame_id] = self.construct_spatial_feature(bbox)

    def get_num_positive_instances_found(self):
        return self.num_positive_instances_found

    def update_random_choice_p(self):
        """Given positive frames seen, compute the probabilities associated with each frame for random choice.
        Frames that are close to a positive frame that have been seen are more likely to be positive as well, thus should have a smaller probability of returning to user for annotations.
        Heuristic: For each observed positive frame, the probability function is 0 at that observed frame, grows ``linearly'' as the distance from the observed frame increases, and becomes constantly 1 after AVG_DURATION distance on each side.
        TODO: considering cases when two observed positive frames are close enough that their probability functions overlap.
        Return: Numpy_Array[proba]
        """
        scale = 2
        func = lambda x : (x ** 2) / (int(AVG_DURATION * scale) ** 2)
        p = np.ones(len(self.maskrcnn_bboxes))
        for frame_id in self.positive_frames_seen:
           # Right half of the probability function
            for i in range(int(AVG_DURATION * scale) + 1):
                if frame_id + i < len(self.maskrcnn_bboxes):
                    p[frame_id + i] = min(func(i), p[frame_id + i])
            # Left half of the probability function
            for i in range(int(AVG_DURATION * scale) + 1):
                if frame_id - i >= 0:
                    p[frame_id - i] = min(func(i), p[frame_id - i])
        self.p = p

    def get_frames_stats(self):
        print("Unseen frames: {0}, negative frames seen: {1}, positive frames seen: {2}.".format(self.frames_unseen.nonzero()[0].size, len(self.negative_frames_seen), len(self.positive_frames_seen)))
        # print("The user has seen {0} frames and found {1} positive frames so far.".format(len(self.positive_frames_seen) + len(self.negative_frames_seen), len(self.positive_frames_seen)))

    def get_plot_data_y(self):
        return self.plot_data_y

    def random_sampling(self):
        while not (self.positive_frames_seen and self.negative_frames_seen) and self.frames_unseen.nonzero()[0].size:
            # TODO: normalized_p seems to be useless
            normalized_p = self.p[self.frames_unseen] / self.p[self.frames_unseen].sum()
            frame_id = np.random.choice(self.frames_unseen.nonzero()[0], p=normalized_p)
            self.simulate_user_annotation(frame_id)
        self.fit_decision_tree()

    def simulate_user_annotation(self, frame_id):
        """Given an input frame, simulate the process where the user annotates the frame. If the frame is labelled as positive, add it to the positive_frames_seen list (and all other consecutive frames that are positive), otherwise add it to the negative_frames_seen list. Remove the frame from frames_unseen list after completion.
        """
        self.frames_unseen[frame_id] = False
        if frame_id in self.pos_frames:
            for key, (start_frame, end_frame) in self.pos_frames_per_instance.items():
                if start_frame <= frame_id and frame_id < end_frame:
                    self.num_positive_instances_found += 1
                    print("num_positive_instances_found:", self.num_positive_instances_found)
                    del self.pos_frames_per_instance[key]
                    break
            self.positive_frames_seen.append(frame_id)
            self.plot_data_y = np.append(self.plot_data_y, self.num_positive_instances_found)
            self.get_frames_stats()
        else:
            self.negative_frames_seen.append(frame_id)
            self.plot_data_y = np.append(self.plot_data_y, self.num_positive_instances_found)

    # @tools.tik_tok
    def get_next_batch(self):
        """Iterate through "frames_unseen", and find the first BATCH_SIZE most confident frames that evaluate positive by the decision tree classifier.
        """
        if self.frames_unseen.nonzero()[0].size % self.n_annotations == 0:
            self.fit_decision_tree()
        preds = self.clf.predict_proba(self.spatial_features)[:, 1]
        # print("Get next batch before:", np.argsort(-preds)[:5], preds[np.argsort(-preds)][:5])
        self.update_random_choice_p()
        preds = preds * self.p
        frames = self.candidates * self.frames_unseen
        scores = preds[frames]
        frames = frames.nonzero()[0]
        ind = np.argsort(-scores)
        # print("Get next batch:", frames[ind][:5], scores[ind][:5])
        return frames[ind][:BATCH_SIZE]

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

    # @tools.tik_tok
    def fit_decision_tree(self):
        self.clf = self.clf.fit(self.spatial_features[~self.frames_unseen], self.Y[~self.frames_unseen])

    @staticmethod
    def save_data(plot_data_y_list, method):
        with open("{}.json".format(method), 'w') as f:
            f.write(json.dumps([arr.tolist() for arr in plot_data_y_list]))

class IterativeProcessingWithoutHeuristic(IterativeProcessing):
    def __init__(self) -> None:
        super().__init__()

    def update_random_choice_p(self):
        p = np.ones(len(self.maskrcnn_bboxes))
        self.p = p


if __name__ == '__main__':
    n_annotations_list = [1, 2, 4, 8, 16]
    for n_annotations in n_annotations_list:
        plot_data_y_list = []
        for _ in range(100):
            ip = IterativeProcessing(n_annotations)
            # ip = IterativeProcessingWithoutHeuristic()
            # Cold start
            print("Cold start:")
            ip.random_sampling()
            print("Cold start done.")
            # Iterative processing
            batched_frames = np.empty(BATCH_SIZE)  # Construct a pseudo list of length BATCH_SIZE to pass the While condition.
            while ip.get_num_positive_instances_found() < 15 and batched_frames.size >= BATCH_SIZE:
                batched_frames = ip.get_next_batch()
                for frame_id in batched_frames:
                    ip.simulate_user_annotation(frame_id)
            plot_data_y_list.append(ip.get_plot_data_y())
        ip.save_data(plot_data_y_list, "retraining_freq/{}".format(n_annotations))