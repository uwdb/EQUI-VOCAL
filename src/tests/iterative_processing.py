"""Target query:
A car turning at the intersection while a pedestrian also appears at the intersection.

Test video: traffic-2.mp4
Stats: duration: 15 mins. 25 fps. 22500 frames in total.
positive events: 15 instances; 912 frames

Goal:
Find all instances of the target event from the test video. We say the system finds an instance of the target event as long as it finds one frame of that instance; the user can later on play the video directly to find all other frames of that instance since they are temporally adjacent.

Performance metric:
The number of frames users have seen in order to find (1/ one half/ all) positive frames (instances). A plot will be made to visualize the result, where x-axis is the number of frames the user has seen, and y-axis is the number of positive frames the user finds.

To start with, the system returns frames via random sampling. The user annotates them (whether the frame is positive or negative; label the car of interest, etc).
"""

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

        self.frames_unseen = list(range(len(self.maskrcnn_bboxes)))
        self.negative_frames_seen = []
        self.positive_frames_seen = []
        self.plot_data_y = np.array([0])
        self.car_centroid_mean = [0, 0, 0, 0, 0]  # (centroid_x_mean, centroid_y_mean, h_mean, w_mean, num_cars)
        self.person_centroid_mean = [0, 0, 0, 0, 0]  # (centroid_x_mean, centroid_y_mean, h_mean, w_mean, num_persons)
        self.resort_frames_unseen = False
        self.x_train = np.empty((0, self.spatial_feature_dim), dtype=np.float64)
        self.y_train = np.empty(0, dtype=np.int)

        self.clf = tree.DecisionTreeClassifier(
            # criterion="entropy",
            max_depth=None,
            min_samples_split=32,
            class_weight="balanced"
            )

    def get_all_frames_of_instance(self, frame_idx):
        '''
        Input: one frame of the target instance
        Output: all frames of the target instance
        ------
        Return: List[frame_idx]
        '''
        for group in mit.consecutive_groups(self.pos_frames):
            consecutive_frames_list = list(group)
            if frame_idx in consecutive_frames_list:
                return consecutive_frames_list
        raise ValueError("Input frame_idx is invalid")

    def get_frames_stats(self):
        print("Unseen frames: {0}, negative frames seen: {1}, positive frames seen: {2}.".format(len(self.frames_unseen), len(self.negative_frames_seen), len(self.positive_frames_seen)))
        # print("The user has seen {0} frames and found {1} positive frames so far.".format(len(self.positive_frames_seen) + len(self.negative_frames_seen), len(self.positive_frames_seen)))

    def get_num_positive_frames_seen(self):
        return len(self.positive_frames_seen)

    def get_plot_data_y(self):
        return self.plot_data_y

    def update_objects_centroid_mean(self, positive_frames):
        for frame_idx in positive_frames:
            pedestrian_found = 0
            car_found = 0
            res_per_frame = self.maskrcnn_bboxes["frame_{}.jpg".format(frame_idx)]
            for x1, y1, x2, y2, class_name, score in res_per_frame:
                if (class_name == "person" and isOverlapping(self.edge_corner_bbox, (x1, y1, x2, y2)) and not pedestrian_found):
                    pedestrian_found = 1
                    self.person_centroid_mean = self._update_objects_centroid_mean_helper(self.person_centroid_mean, (x1 + x2) / 2, (y1 + y2) / 2, y2 - y1, x2 - x1)
                elif (class_name in ["car", "truck"] and isInsideIntersection((x1, y1, x2, y2)) and not car_found):
                    car_found = 1
                    self.car_centroid_mean = self._update_objects_centroid_mean_helper(self.car_centroid_mean, (x1 + x2) / 2, (y1 + y2) / 2, y2 - y1, x2 - x1)

    def _update_objects_centroid_mean_helper(self, old_centroid_mean, centroid_x, centorid_y, height, weight):
        centroid_x_mean, centroid_y_mean, height_mean, weight_mean, count = old_centroid_mean
        return [
            (centroid_x_mean * count + centroid_x) / (count + 1),
            (centroid_y_mean * count + centorid_y) / (count + 1),
            (height_mean * count + height) / (count + 1),
            (weight_mean * count + weight) / (count + 1),
            count + 1
            ]

    def random_sampling(self):
        while not self.positive_frames_seen and self.frames_unseen:
            frame_idx = random.choice(self.frames_unseen)
            self.simulate_user_annotation(frame_idx)

    def simulate_user_annotation(self, frame_idx):
        """
        Given an input frame, simulate the process where the user annotates the frame. If the frame is labelled as positive, add it to the positive_frames_seen list (and all other consecutive frames that are positive), otherwise add it to the negative_frames_seen list. Remove the frame from frames_unseen list after completion.
        """
        if frame_idx in self.pos_frames:
            positive_frames = self.get_all_frames_of_instance(frame_idx)
            for f in positive_frames:
                self.frames_unseen.remove(f)
                self.positive_frames_seen.append(f)
                # construct positive data
                self.x_train = np.vstack((self.x_train, self.construct_spatial_feature_for_one_frame(f)))
                self.y_train = np.append(self.y_train, 1)
                self.plot_data_y = np.append(self.plot_data_y, len(self.negative_frames_seen) + len(self.positive_frames_seen))
            self.update_objects_centroid_mean(positive_frames)
            self.resort_frames_unseen = True
            self.get_frames_stats()
        else:
            self.frames_unseen.remove(frame_idx)
            self.negative_frames_seen.append(frame_idx)
            # construct negative data
            self.x_train = np.vstack((self.x_train, self.construct_spatial_feature_for_one_frame(frame_idx)))
            self.y_train = np.append(self.y_train, 0)

    # @tools.tik_tok
    def get_next_batch(self):
        # Sort unseen frames based on the distance between the mean centroid and the car (person) closest to the mean centroid in each frame.
        if self.resort_frames_unseen:
            print("Resorting...")
            distances = []
            for frame_idx in self.frames_unseen:
                person_normalized_distance = float('inf')
                car_normalized_distance = float('inf')
                res_per_frame = self.maskrcnn_bboxes["frame_{}.jpg".format(frame_idx)]
                for x1, y1, x2, y2, class_name, score in res_per_frame:
                    if (class_name == "person"):
                        person_normalized_distance = min(self.get_dist(self.person_centroid_mean[:2], [(x1 + x2) / 2, (y1 + y2) / 2]) / (self.person_centroid_mean[2] * self.person_centroid_mean[3]), person_normalized_distance)
                    elif (class_name in ["car", "truck"]):
                        car_normalized_distance = min(self.get_dist(self.car_centroid_mean[:2], [(x1 + x2) / 2, (y1 + y2) / 2]) / (self.car_centroid_mean[2] * self.car_centroid_mean[3]), car_normalized_distance)
                distances.append(person_normalized_distance + car_normalized_distance)
            self.frames_unseen = [frame for (_, frame) in sorted(zip(distances, self.frames_unseen))]
            self.resort_frames_unseen = False
        # Iterate through "sorted_frames_unseen", and find the first BATCH_SIZE frames that evaluate positive by the decision tree classifier.
        batched_frames = []
        self.fit_decision_tree()
        for frame_idx in self.frames_unseen:
            # Construct test data
            spatial_feature = self.construct_spatial_feature_for_one_frame(frame_idx)
            pred_label = self.clf.predict([spatial_feature])[0]
            if pred_label == 1:
                batched_frames.append(frame_idx)
                if len(batched_frames) >= BATCH_SIZE:
                    break
        return batched_frames

    def construct_spatial_feature_for_one_frame(self, frame_idx):
        target_car = (0, 0, 0, 0, float('inf'))  # (x1, y1, x2, y2, distance to centroid mean)
        res_per_frame = self.maskrcnn_bboxes["frame_{}.jpg".format(frame_idx)]
        for x1, y1, x2, y2, class_name, score in res_per_frame:
            if (class_name in ["car", "truck"]):
                dist = self.get_dist(self.car_centroid_mean[:2], [(x1 + x2) / 2, (y1 + y2) / 2])
                if dist < target_car[4]:
                    target_car = (x1, y1, x2, y2, dist)
        # After finding the target car, construct spatial feature of that car.
        x1, y1, x2, y2, _ = target_car
        centroid_x = (x1 + x2) / 2
        centroid_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        wh_ratio = width / (height + EPSILON)
        return np.array([centroid_x, centroid_y, width, height, wh_ratio])

    def fit_decision_tree(self):
        self.clf = self.clf.fit(self.x_train, self.y_train)

    @staticmethod
    def get_dist(p1, p2):
        return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

def plot_data(plot_data_y_list):
    np.savetxt('iterative_processing.csv', plot_data_y_list, fmt='%d', delimiter=',')
    y_upper = np.max(plot_data_y_list, axis=0)
    y_lower = np.min(plot_data_y_list, axis=0)
    y_mean = np.mean(plot_data_y_list, axis=0)
    x_values = range(plot_data_y_list.shape[1])
    plt.fill_between(x_values, y_lower, y_upper, alpha=0.2)
    plt.plot(x_values, y_mean)
    plt.savefig("iterative_processing_plot")

if __name__ == '__main__':
    plot_data_y_list = np.empty((0, 822))
    for _ in range(20):
        ip = IterativeProcessing()
        # Cold start
        print("Cold start:")
        ip.random_sampling()
        print("Cold start done.")
        # Iterative processing
        batched_frames = list(range(BATCH_SIZE))  # Construct a pseudo list of length BATCH_SIZE to pass the While condition.
        while len(batched_frames) >= BATCH_SIZE and ip.get_num_positive_frames_seen() < 821:
            batched_frames = ip.get_next_batch()
            for frame_idx in batched_frames:
                ip.simulate_user_annotation(frame_idx)
        if ip.get_plot_data_y().shape[0] < 822:
            print("Warning: failed to find 90% positive frames. Ignore this iteration.")
        else:
            plot_data_y_list = np.vstack((plot_data_y_list, ip.get_plot_data_y()[:822]))
    plot_data(plot_data_y_list)