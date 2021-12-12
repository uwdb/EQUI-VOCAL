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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
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
from sklearn.metrics import RocCurveDisplay
from sklearn.tree._tree import TREE_LEAF, TREE_UNDEFINED
from copy import deepcopy
import cv2
from shapely.geometry import Polygon
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from glob import glob

annotated_batch_size = 1
materialized_batch_size = 16
# init_sampling_step = 100
init_sampling_step = 1

class ComplexEventVideoDB:
    def __init__(self, dataset="visualroad_traffic2", query="test_b", temporal_heuristic=True):
        self.query = query
        self.temporal_heuristic = temporal_heuristic
        self.base_image_path = "/home/ubuntu/complex_event_video/data/car_turning_traffic2/neg/frame_0.jpg"

        """Read in object detection bounding box information
        Properties initialized:
            self.maskrcnn_bboxes,
            self.n_frames
        """
        if dataset == "visualroad_traffic2":
            self.ingest_bbox_info()
        elif dataset == "meva":
            self.ingest_bbox_info_meva()

        """Ingest ground-truth labels of the target event
        Properties initialized:
            self.pos_frames,
            self.pos_frames_per_instance,
            self.n_positive_instances,
            self.avg_duration
        """
        self.ingest_gt_labels()

        """1. Apply initial query to filter out unnecessary frames
        2. Construct spatial features for candidate frames
        Properties initialized:
            self.candidates,
            self.spatial_feature_dim,
            self.feature_names,
            self.spatial_features
        """
        self.filtering_stage()

        self.num_positive_instances_found = 0
        self.raw_frames = np.full(self.n_frames, True, dtype=np.bool)
        self.materialized_frames = np.full(self.n_frames, False, dtype=np.bool)
        self.iteration = 0
        self.vis_decision_output = []
        self.Y = np.zeros(self.n_frames, dtype=np.int)
        for i in range(self.n_frames):
            if i in self.pos_frames:
                self.Y[i] = 1
        self.plot_data_y_annotated = np.array([0])
        self.plot_data_y_materialized = np.array([self.materialized_frames.nonzero()[0].size])
        self.negative_frames_seen = []
        self.positive_frames_seen = []
        self.p = np.ones(self.n_frames)

        # Initially, materialize 1/init_sampling_step (e.g. 1%) of all frames.
        for i in range(0, self.n_frames, init_sampling_step):
            self.raw_frames[i] = False
            self.materialized_frames[i] = True

        # ExSample initialization
        self.number_of_chunks = 1
        self.stats_per_chunk = [[0, 0] for _ in range(self.number_of_chunks)] # N^1 and n

        self.query_initialization = RandomInitialization(self.pos_frames, self.candidates)
        self.frame_selection = RandomFrameSelection(self.spatial_features, self.Y, materialized_batch_size, annotated_batch_size, self.avg_duration, self.candidates)
        self.user_feedback = UserFeedback(self.n_frames, self.pos_frames)
        self.proxy_model_training = ProxyModelTraining(self.spatial_features, self.Y)

    def ingest_bbox_info(self):
        # Read in bbox info
        with open("/home/ubuntu/complex_event_video/data/car_turning_traffic2/bbox.json", 'r') as f:
            self.maskrcnn_bboxes = json.loads(f.read())
        self.n_frames = len(self.maskrcnn_bboxes)

        # with open("ms_coco_classnames.txt") as f:
        #     coco_names = f.read().splitlines()
        # bboxes_lst = []
        # with open("/home/ubuntu/complex_event_video/data/car_turning_traffic2/traffic-2.txt", 'r') as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         # [fid, tid, cid, x1, y1, x2, y2], types: int
        #         bboxes_lst.append(list(map(int, line.split())))
        # self.n_frames = bboxes_lst[-1][0] + 1

        # self.maskrcnn_bboxes = {"frame_{}.jpg".format(key): [] for key in range(self.n_frames)}
        # for elem in bboxes_lst:
        #     # x1, y1, x2, y2, class_name, tid
        #     self.maskrcnn_bboxes["frame_{}.jpg".format(elem[0])].append([
        #         elem[3],
        #         elem[4],
        #         elem[5],
        #         elem[6],
        #         coco_names[elem[2]],
        #         elem[1]
        #     ])

    def ingest_bbox_info_meva(self):
        files = [y for x in os.walk("/home/ubuntu/complex_event_video/data/meva") for y in glob(os.path.join(x[0], '*.json'))]
        gt_annotations = [os.path.basename(y).replace(".activities.yml", "") for x in os.walk("/home/ubuntu/complex_event_video/data/meva/meva-data-repo/annotation/DIVA-phase-2/MEVA/kitware") for y in glob(os.path.join(x[0], '*.yml'))]
        self.maskrcnn_bboxes = {}
        self.video_list = [] # (video_basename, frame_offset, n_frames)
        for file in files:
            if "school.G421.r13.json" not in file:
                continue
            # If the video doesn't have annotations, skip.
            video_basename = os.path.basename(file).replace(".r13.json", "")
            if video_basename not in gt_annotations:
                continue
            frame_offset = len(self.maskrcnn_bboxes)
            # Read in bbox info
            with open(file, 'r') as f:
                bbox_dict = json.loads(f.read())
                for local_frame_id, v in bbox_dict.items():
                    self.maskrcnn_bboxes[video_basename + "_" + local_frame_id] = v
            self.video_list.append((video_basename, frame_offset, len(bbox_dict)))
        self.n_frames = len(self.maskrcnn_bboxes)
        print("# all frames: ", self.n_frames, "; # video files: ", len(self.video_list))

    def ingest_gt_labels(self):
        # Get ground-truth labels
        if self.query in ["test_a", "test_b", "test_c"]:
            self.pos_frames, self.pos_frames_per_instance = eval(self.query + "(self.maskrcnn_bboxes)")
        elif self.query == "turning_car_and_pedestrain_at_intersection":
            self.pos_frames, self.pos_frames_per_instance = turning_car_and_pedestrain_at_intersection()
        elif self.query in ["test_d"]:
            self.pos_frames, self.pos_frames_per_instance = eval(self.query + "(self.maskrcnn_bboxes)")
        elif self.query in ["test_e"]:
            self.pos_frames, self.pos_frames_per_instance = eval(self.query + "(self.maskrcnn_bboxes)")
        elif self.query in ["meva_person_stands_up"]:
            self.pos_frames, self.pos_frames_per_instance = eval(self.query + "(self.video_list)")

        self.n_positive_instances = len(self.pos_frames_per_instance)
        self.avg_duration = 1.0 * len(self.pos_frames) / self.n_positive_instances
        print("# positive frames: ", len(self.pos_frames), "; # distinct instances: ", self.n_positive_instances, "; average duration: ", self.avg_duration)
        print("Durations of all instances: ")
        print(" ".join([str(v[1]- v[0]) for _, v in self.pos_frames_per_instance.items()]))

    def filtering_stage(self):
        if self.query in ["meva_person_stands_up"]:
            self.feature_names, self.spatial_feature_dim, self.spatial_features, self.candidates = getattr(filter, self.query)(self.maskrcnn_bboxes, self.video_list, self.pos_frames)
            return
        elif self.query in ["test_a", "test_b", "test_c", "turning_car_and_pedestrain_at_intersection"]:
            self.spatial_feature_dim = 5
            self.feature_names = ["x", "y", "w", "h", "r"]
        elif self.query in ["test_d"]:
            self.spatial_feature_dim = 10
            self.feature_names = ["x1", "y1", "w1", "h1", "r1", "x2", "y2", "w2", "h2", "r2"]
        elif self.query in ["test_e"]:
            self.spatial_feature_dim = 8
            self.feature_names = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
        self.spatial_features = np.zeros((self.n_frames, self.spatial_feature_dim), dtype=np.float64)
        # Filtering stage
        self.candidates = np.full(self.n_frames, True, dtype=np.bool)
        if self.query in ["test_a", "test_b", "test_c", "test_d", "test_e", "turning_car_and_pedestrain_at_intersection"]:
            for frame_id in range(self.n_frames):
                res_per_frame = self.maskrcnn_bboxes["frame_{}.jpg".format(frame_id)]
                if self.query in ["test_a", "test_b", "test_c"]:
                    is_candidate, bbox = getattr(filter, self.query)(res_per_frame, frame_id)
                elif self.query == "turning_car_and_pedestrain_at_intersection":
                    is_candidate, bbox = filter.car_and_pedestrain_at_intersection(res_per_frame, frame_id)
                elif self.query in ["test_d"]:
                    is_candidate, bbox1, bbox2 = getattr(filter, self.query)(res_per_frame)
                elif self.query in ["test_e"]:
                    is_candidate, car_box, person_box = getattr(filter, self.query)(res_per_frame)

                if not is_candidate:
                    self.candidates[frame_id] = False
                else:
                    if self.query in ["test_a", "test_b", "test_c", "turning_car_and_pedestrain_at_intersection"]:
                        self.spatial_features[frame_id] = self.construct_spatial_feature(bbox)
                    elif self.query in ["test_d"]:
                        self.spatial_features[frame_id] = self.construct_spatial_feature_two_objects(bbox1, bbox2)
                    elif self.query in ["test_e"]:
                        self.spatial_features[frame_id] = self.construct_spatial_feature_spatial_relationship(car_box, person_box)

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

            if not self.iteration % 20:
                self.visualize_classifier_decision()
            self.iteration += 1

        print("stats_per_chunk", self.stats_per_chunk)

        # Write out decision tree text report
        with open('outputs/tree_report_{}.txt'.format(self.query), 'w') as f:
            for element, rules in self.vis_decision_output:
                # print("rules", rules)
                f.write(element + "\nExtracted rule: ")
                for i, rule in enumerate(rules):
                    if not rule:
                        continue
                    f.write("(")
                    for j, pred in enumerate(rule):
                        if pred[1] != -float('inf') and pred[2] != float('inf'):
                            pred_str = "{:.2f} < {} <= {:.2f}".format(pred[1], pred[0], pred[2])
                        elif pred[1] == -float('inf'):
                            pred_str = "{} <= {:.2f}".format(pred[0], pred[2])
                        elif pred[2] == float('inf'):
                            pred_str = "{} > {:.2f}".format(pred[0], pred[1])
                        if j < len(rule) - 1:
                            pred_str += ", "
                        f.write(pred_str)
                    f.write(")")
                    if i < len(rules) - 1:
                        f.write(" or ")
                f.write("\n")

        return self.plot_data_y_annotated, self.plot_data_y_materialized

    def construct_spatial_feature(self, bbox):
        x1, y1, x2, y2 = bbox
        centroid_x = (x1 + x2) / 2
        centroid_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        wh_ratio = width / height
        return np.array([centroid_x, centroid_y, width, height, wh_ratio])
        # return np.array([x1, y1, x2, y2])

    def construct_spatial_feature_two_objects(self, bbox1, bbox2):
        x11, y11, x21, y21 = bbox1
        centroid_x1 = (x11 + x21) / 2
        centroid_y1 = (y11 + y21) / 2
        width1 = x21 - x11
        height1 = y21 - y11
        wh_ratio1 = width1 / height1

        x12, y12, x22, y22 = bbox2
        centroid_x2 = (x12 + x22) / 2
        centroid_y2 = (y12 + y22) / 2
        width2 = x22 - x12
        height2 = y22 - y12
        wh_ratio2 = width2 / height2
        return np.array([centroid_x1, centroid_y1, width1, height1, wh_ratio1, centroid_x2, centroid_y2, width2, height2, wh_ratio2])

    def construct_spatial_feature_spatial_relationship(self, car_box, person_box):
        x, y, x2, y2 = car_box
        xp, yp, x4, y4 = person_box
        w = x2 - x
        h = y2 - y
        wp = x4 - xp
        hp = y4 - yp
        s1 = (x - xp) / w
        s2 = (y - yp) / h
        s3 = (y + h - yp - hp) / h
        s4 = (x + w - xp - wp) / w
        s5 = hp / h
        s6 = wp / w
        s7 = (wp * hp) / (w * h)
        s8 = (wp + hp) / (w + h)
        return np.array([s1, s2, s3, s4, s5, s6, s7, s8])

    def update_random_choice_p(self):
        """Given positive frames seen, compute the probabilities associated with each frame for random choice.
        Frames that are close to a positive frame that have been seen are more likely to be positive as well, thus should have a smaller probability of returning to user for annotations.
        Heuristic: For each observed positive frame, the probability function is 0 at that observed frame, grows ``linearly'' as the distance from the observed frame increases, and becomes constantly 1 after AVG_DURATION distance on each side.
        TODO: considering cases when two observed positive frames are close enough that their probability functions overlap.
        Return: Numpy_Array[proba]
        """
        self.p = np.ones(self.n_frames)
        if not self.temporal_heuristic:
            # No temporal heuristic. Set same probability dacay for all frames.
            return

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

    def visualize_classifier_decision(self):
        # Make a copy
        clf_copy = deepcopy(self.clf)

        # Find smallest decision tree in the random forest classifier
        smallest_tree = clf_copy.estimators_[0]
        min_leaf_node_count = sum(clf_copy.estimators_[0].tree_.children_left < 0)
        for est in clf_copy.estimators_:
            if sum(est.tree_.children_left < 0) < min_leaf_node_count:
                min_leaf_node_count = sum(est.tree_.children_left < 0)
                smallest_tree = est

        # Prune the decision tree
        self.prune_duplicate_leaves(smallest_tree)
        # r1 = tree.export_text(smallest_tree, feature_names=self.feature_names)
        rules = self.extract_rules(smallest_tree, self.feature_names)
        # self.visualize_rules_one_object(rules)
        # self.visualize_rules_spatial_relationship(rules)

        # Evaluate metrics on training data
        x_train = self.spatial_features[~(self.raw_frames | self.materialized_frames)]
        y_train = self.Y[~(self.raw_frames | self.materialized_frames)]
        y_pred = self.clf.predict(x_train)
        tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
        training_data_str = "[training data] accuracy: {}; f1_score: {}; tn, fp, fn, tp: {}, {}, {}, {}".format(accuracy_score(y_train, y_pred), f1_score(y_train, y_pred), tn, fp, fn, tp)

        # Evaluate metrics on all (filtered) data
        y_pred = self.clf.predict(self.spatial_features[self.candidates])
        tn, fp, fn, tp = confusion_matrix(self.Y[self.candidates], y_pred).ravel()
        all_data_str = "[all data] accuracy: {}; f1_score: {}; tn, fp, fn, tp: {}, {}, {}, {}".format(accuracy_score(self.Y[self.candidates], y_pred), f1_score(self.Y[self.candidates], y_pred), tn, fp, fn, tp)
        # rfc_disp = RocCurveDisplay.from_estimator(self.clf, self.spatial_features, self.Y, alpha=0.8)
        # plt.savefig("outputs/roc_{}.pdf".format(self.iteration), bbox_inches='tight', pad_inches=0)
        self.vis_decision_output.append(("========== Iteration: {} ==========\n".format(self.iteration) + training_data_str + "\n" + all_data_str, rules))

    @staticmethod
    def extract_rules(tree, feature_names):
        tree_ = tree.tree_
        rules = []
        def traverse(node_id, pred):
            if tree_.children_left[node_id] != tree_.children_right[node_id]:
                name = feature_names[tree_.feature[node_id]]
                threshold = tree_.threshold[node_id]
                traverse(tree_.children_left[node_id], pred + [(name, "<=", threshold)])
                traverse(tree_.children_right[node_id], pred + [(name, ">", threshold)])
            else:
                if np.argmax(tree_.value[node_id]) == 1:
                    rule = pred.copy()
                    # Simplify the rule by grouping together predicates on the same feature
                    values = set(map(lambda x: x[0], rule))
                    # simplified_rule: [(feature name, > value, <= value)]
                    simplified_rule = [(x, np.max([y[2] for y in rule if y[0] == x and y[1] == ">"], initial=-float('inf')), np.min([y[2] for y in rule if y[0]==x and y[1] == "<="], initial=float('inf'))) for x in values]
                    simplified_rule.sort(key=lambda tup: tup[0])
                    rules.append(simplified_rule)
                    # print(simplified_rule)
        traverse(0, [])
        # print("rules", rules)

        return rules

    def visualize_rules_one_object(self, rules):
        img = cv2.imread(self.base_image_path)
        # pts = []
        overlay = img.copy()
        for i, rule in enumerate(rules):
            local_overlay = img.copy()
            # pts.append()
            # print("pts", pts)
            color = list(np.random.random(size=3) * 256)
            overlay = cv2.fillPoly(overlay, [self.find_polygon_points(rule)], color)
            local_overlay = cv2.fillPoly(local_overlay, [self.find_polygon_points(rule)], color)
            w = 200
            h = 100
            spacing = 50
            num_per_row = 4
            x = 10 + (w + spacing) * (i % 5)
            y = 10 + (h + spacing) * (i // 5)
            overlay = cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)  # A filled rectangle
            local_overlay = cv2.rectangle(local_overlay, (x, y), (x+w, y+h), color, -1)  # A filled rectangle
            # for pred in rule:
            #     if pred[0] == "r":

            out_img = cv2.addWeighted(local_overlay, 0.7, img, 0.3, 0)
            cv2.imwrite("{0}/{1}_{2}.jpg".format("/home/ubuntu/complex_event_video/src/outputs", self.iteration, i), out_img)
        out_img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
        cv2.imwrite("{0}/{1}.jpg".format("/home/ubuntu/complex_event_video/src/outputs", self.iteration), out_img)

    def visualize_rules_spatial_relationship(self, rules):
        for i, rule in enumerate(rules):
            x1, y1, x2, y2 = 50, 50, 60, 60
            s1_min, s1_max, s2_min, s2_max = 0, 100, 0, 100
            for pred in rule:
                if pred[0] == "s1":
                    s1_min = max(pred[1], s1_min)
                    s1_max = min(pred[2], s1_max)
                elif pred[0] == "s2":
                    s2_min = max(pred[1], s2_min)
                    s2_max = min(pred[2], s2_max)
                # elif pred[0] == "s3":
                #     s3_min = max(pred[1], s3_min)
                #     s3_max = min(pred[2], s3_max)
                # elif pred[0] == "s4":
                #     s4_min = max(pred[1], s4_min)
                #     s4_max = min(pred[2], s4_max)

            img = np.zeros([110, 110, 3],dtype=np.uint8)
            img.fill(255)
            color = list(np.random.random(size=3) * 256)
            # Draw base box
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 1)
            # Draw visualization for s1 and s2
            print(x1 + s2_min, y1 + s1_min, x1 + s2_max, y1 + s1_max)
            img = cv2.rectangle(img, (x1 + int(s2_min*10), y1 + int(s1_min*10)), (x1 + int(s2_max*10), y1 + int(s1_max*10)), (0,0,255), 1)
            # # Draw visualization for s3 and s4
            # img = cv2.rectangle(img, (x1 + s2_min, y1 + s1_min), (x1 + s2_max, y1 + s1_max), color, 2)
            cv2.imwrite("{0}/spatial_rel_{1}_{2}.jpg".format("/home/ubuntu/complex_event_video/src/outputs", self.iteration, i), img)

    @staticmethod
    def find_polygon_points(rule):
        x_min, y_min, x_max, y_max = 0, 0, 960, 540
        for pred in rule:
            if pred[0] == "x":
                x_min = max(pred[1], 0)
                x_max = min(pred[2], 960)
            elif pred[0] == "y":
                y_min = max(pred[1], 0)
                y_max = min(pred[2], 540)
        poly1 = Polygon([(0, 480), (450, 394), (782, 492)])
        poly2 = Polygon([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)])
        poly3 = poly1.intersection(poly2)
        return np.array([[x, y] for x, y in poly3.exterior.coords], np.int32)

    @staticmethod
    def is_leaf(inner_tree, index):
        # Check whether node is leaf node
        return (inner_tree.children_left[index] == TREE_LEAF and
                inner_tree.children_right[index] == TREE_LEAF)

    @classmethod
    def prune_index(cls, inner_tree, decisions, index=0):
        # Start pruning from the bottom - if we start from the top, we might miss
        # nodes that become leaves during pruning.
        # Do not use this directly - use prune_duplicate_leaves instead.
        if not cls.is_leaf(inner_tree, inner_tree.children_left[index]):
            cls.prune_index(inner_tree, decisions, inner_tree.children_left[index])
        if not cls.is_leaf(inner_tree, inner_tree.children_right[index]):
            cls.prune_index(inner_tree, decisions, inner_tree.children_right[index])

        # Prune children if both children are leaves now and make the same decision:
        if (cls.is_leaf(inner_tree, inner_tree.children_left[index]) and
            cls.is_leaf(inner_tree, inner_tree.children_right[index]) and
            (decisions[index] == decisions[inner_tree.children_left[index]]) and
            (decisions[index] == decisions[inner_tree.children_right[index]])):
            # turn node into a leaf by "unlinking" its children
            inner_tree.children_left[index] = TREE_LEAF
            inner_tree.children_right[index] = TREE_LEAF
            inner_tree.feature[index] = TREE_UNDEFINED
            ##print("Pruned {}".format(index))

    @classmethod
    def prune_duplicate_leaves(cls, mdl):
        # Remove leaves if both
        decisions = mdl.tree_.value.argmax(axis=2).flatten().tolist() # Decision for each node
        cls.prune_index(mdl.tree_, decisions)

    def get_frames_stats(self):
        print("raw frames: {0}, materialized frames: {1}, negative frames seen: {2}, positive frames seen: {3}.".format(self.raw_frames.nonzero()[0].size, self.materialized_frames.nonzero()[0].size, len(self.negative_frames_seen), len(self.positive_frames_seen)))
        if len(self.positive_frames_seen) == 1:
            print(self.positive_frames_seen)
        # print("The user has seen {0} frames and found {1} positive frames so far.".format(len(self.positive_frames_seen) + len(self.negative_frames_seen), len(self.positive_frames_seen)))

    def tsne_plot(self):
        spatial_features_embedded = TSNE(n_components=3, learning_rate="auto", init="random").fit_transform(self.spatial_features[self.candidates])
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')
        # fig, ax = plt.subplots(1, figsize=(4, 3))
        ax.scatter(spatial_features_embedded[:, 0], spatial_features_embedded[:, 1], spatial_features_embedded[:, 2], c=self.Y[self.candidates], label=self.Y[self.candidates], alpha=0.5)
        ax.legend()
        plt.savefig("tsne.pdf", bbox_inches='tight', pad_inches=0)

    @staticmethod
    def save_data(plot_data_y_list, method):
        with open("outputs/{}.json".format(method), 'w') as f:
            f.write(json.dumps([arr.tolist() for arr in plot_data_y_list]))


class FilteredProcessing(ComplexEventVideoDB):
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
    for _ in range(1):
        cevdb = ComplexEventVideoDB(dataset="meva", query="meva_person_stands_up", temporal_heuristic=True)
        # cevdb = ComplexEventVideoDB(dataset="visualroad_traffic2", query="turning_car_and_pedestrain_at_intersection", temporal_heuristic=True)
        # cevdb.tsne_plot()
        # cevdb = FilteredProcessing(dataset="meva", query="meva_person_stands_up", temporal_heuristic=False)
        plot_data_y_annotated, plot_data_y_materialized = cevdb.run()
        # plot_data_y_annotated_list.append(plot_data_y_annotated)
        # plot_data_y_materialized_list.append(plot_data_y_materialized)
    # cevdb.save_data(plot_data_y_annotated_list, "iterative_test_b")
    # cevdb.save_data(plot_data_y_annotated_list, "random_test_a")
    # cevdb.save_data(plot_data_y_materialized_list, "filtered_materialized")