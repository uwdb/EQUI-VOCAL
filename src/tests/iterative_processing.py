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
import numpy as np
import graphviz
import random
import more_itertools as mit
import matplotlib.pyplot as plt

from utils.utils import isInsideIntersection, isOverlapping
from utils import tools

# When BATCH_SIZE > 1, simulate_user_annotation() will break
BATCH_SIZE = 1
EPSILON = 1e-7

class IterativeProcessing:
    def __init__(self):
        self.spatial_feature_dim = 5
        self.edge_corner_bbox = (367, 345, 540, 418)
        # Read in bbox info
        bbox_file = "/home/ubuntu/complex_event_video/data/car_turning_traffic2/bbox.json"
        with open(bbox_file, 'r') as f:
            self.maskrcnn_bboxes = json.loads(f.read())

        # Read labels
        self.pos_frames = []
        with open("/home/ubuntu/complex_event_video/data/annotation.csv", "r") as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)
            for row in csvreader:
                start_frame, end_frame = int(row[0]), int(row[1])
                self.pos_frames += list(range(start_frame, end_frame+1))

        self.frames_unseen = np.full(len(self.maskrcnn_bboxes), True, dtype=np.bool)
        self.negative_frames_seen = []
        self.positive_frames_seen = []
        self.candidates = np.full(len(self.maskrcnn_bboxes), False, dtype=np.bool)
        self.spatial_features = np.zeros((len(self.maskrcnn_bboxes), self.spatial_feature_dim), dtype=np.float64)
        self.Y = np.zeros(len(self.maskrcnn_bboxes), dtype=np.int)
        for i in range(len(self.maskrcnn_bboxes)):
            if i in self.pos_frames:
                self.Y[i] = 1
        self.plot_data_y = np.array([0])
        self.plot_data_y_only_neg_frames = np.array([0])
        self.clf = tree.DecisionTreeClassifier(
            # criterion="entropy",
            max_depth=None,
            min_samples_split=32,
            class_weight="balanced"
            )
        self.get_candidates()

    @staticmethod
    def get_dist(p1, p2):
        return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

    def get_candidates(self):
        for frame_id in range(len(self.maskrcnn_bboxes)):
            is_candidate, bbox = self.frame_has_objects_of_interest(frame_id)
            if is_candidate:
                self.candidates[frame_id] = True
                self.spatial_features[frame_id] = self.construct_spatial_feature(bbox)

    def get_all_frames_of_instance(self, frame_id):
        '''
        Input: one frame of the target instance
        Output: all frames of the target instance
        ------
        Return: List[frame_id]
        '''
        for group in mit.consecutive_groups(self.pos_frames):
            consecutive_frames_list = list(group)
            if frame_id in consecutive_frames_list:
                return consecutive_frames_list
        raise ValueError("Input frame_id is invalid")

    def get_frames_stats(self):
        print("Unseen frames: {0}, negative frames seen: {1}, positive frames seen: {2}.".format(self.frames_unseen.nonzero()[0].size, len(self.negative_frames_seen), len(self.positive_frames_seen)))
        # print("The user has seen {0} frames and found {1} positive frames so far.".format(len(self.positive_frames_seen) + len(self.negative_frames_seen), len(self.positive_frames_seen)))

    def get_num_positive_frames_seen(self):
        return len(self.positive_frames_seen)

    def get_plot_data_y(self):
        # return self.plot_data_y
        return self.plot_data_y_only_neg_frames

    def random_sampling(self):
        while not (self.positive_frames_seen and self.negative_frames_seen) and self.frames_unseen.nonzero()[0].size:
            frame_id = np.random.choice(self.frames_unseen.nonzero()[0])
            self.simulate_user_annotation(frame_id)

    def simulate_user_annotation(self, frame_id):
        """Given an input frame, simulate the process where the user annotates the frame. If the frame is labelled as positive, add it to the positive_frames_seen list (and all other consecutive frames that are positive), otherwise add it to the negative_frames_seen list. Remove the frame from frames_unseen list after completion.
        """
        if frame_id in self.pos_frames:
            positive_frames = self.get_all_frames_of_instance(frame_id)
            num_instances_found = self.plot_data_y[-1] + 1
            for f in positive_frames:
                self.frames_unseen[f] = False
                self.positive_frames_seen.append(f)
                self.plot_data_y = np.append(self.plot_data_y, num_instances_found)
            self.plot_data_y_only_neg_frames[-1] += 1
            self.get_frames_stats()
        else:
            self.frames_unseen[frame_id] = False
            self.negative_frames_seen.append(frame_id)
            self.plot_data_y = np.append(self.plot_data_y, self.plot_data_y[-1])
            self.plot_data_y_only_neg_frames = np.append(self.plot_data_y_only_neg_frames, self.plot_data_y_only_neg_frames[-1])

    # @tools.tik_tok
    def get_next_batch(self):
        """Iterate through "frames_unseen", and find the first BATCH_SIZE most confident frames that evaluate positive by the decision tree classifier.
        """
        self.fit_decision_tree()
        preds = self.clf.predict_proba(self.spatial_features)
        frames = self.candidates * self.frames_unseen
        scores = preds[frames][:, 1]
        frames = frames.nonzero()[0]
        ind = np.argsort(-scores)
        return frames[ind][:BATCH_SIZE]

    def frame_has_objects_of_interest(self, frame_id):
        has_car = 0
        has_pedestrian = 0
        res_per_frame = self.maskrcnn_bboxes["frame_{}.jpg".format(frame_id)]
        for x1, y1, x2, y2, class_name, score in res_per_frame:
            if (class_name == "person" and isOverlapping(self.edge_corner_bbox, (x1, y1, x2, y2))):
                has_pedestrian = 1
            elif (class_name in ["car", "truck"] and isInsideIntersection((x1, y1, x2, y2))):
                has_car = 1
            if has_car and has_pedestrian:
                return True, (x1, y1, x2, y2)
        return False, None

    def construct_spatial_feature(self, bbox):
        x1, y1, x2, y2 = bbox
        centroid_x = (x1 + x2) / 2
        centroid_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        wh_ratio = width / height
        return np.array([centroid_x, centroid_y, width, height, wh_ratio])

    def fit_decision_tree(self):
        self.clf = self.clf.fit(self.spatial_features[~self.frames_unseen], self.Y[~self.frames_unseen])


def plot_data(plot_data_y_list, method):
    with open("{}.json".format(method), 'w') as f:
        f.write(json.dumps([arr.tolist() for arr in plot_data_y_list]))
    fig, ax = plt.subplots(1)
    for plot_data_y in plot_data_y_list:
        x_values = range(plot_data_y.size)
        ax.plot(x_values, plot_data_y, color='tab:blue')
    ax.set_ylabel('number of positive instances the user finds')
    # ax.set_xlabel('number of frames that user has seen')
    ax.set_xlabel('number of negative frames that user has seen')
    ax.grid()
    plt.savefig(method)

if __name__ == '__main__':
    plot_data_y_list = []
    for _ in range(20):
        ip = IterativeProcessing()
        # Cold start
        print("Cold start:")
        ip.random_sampling()
        print("Cold start done.")
        # Iterative processing
        batched_frames = np.empty(BATCH_SIZE)  # Construct a pseudo list of length BATCH_SIZE to pass the While condition.
        # while len(batched_frames) >= BATCH_SIZE and ip.get_num_positive_frames_seen() < 821:
        while ip.get_num_positive_frames_seen() < 912 and batched_frames.size >= BATCH_SIZE:
            batched_frames = ip.get_next_batch()
            for frame_id in batched_frames:
                ip.simulate_user_annotation(frame_id)
        plot_data_y_list.append(ip.get_plot_data_y())
    plot_data(plot_data_y_list, "iterative_only_neg")