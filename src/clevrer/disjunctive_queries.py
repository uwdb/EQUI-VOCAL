"""
This is a copy of topk_queries.py and is used for searching disjunctive queries.
"""

import json
import itertools
from ntpath import join
from turtle import left
import joblib
from filter import construct_spatial_feature_spatial_relationship
from itertools import groupby, count
from multiprocessing import Pool
import multiprocessing as mp
import random
import argparse
from functools import partial
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import f1_score
from utils import tools
from time import time

# num_workers = mp.cpu_count()
num_workers = 4
print("# of workers: ", num_workers)
segment_length = 128
n_chunks = int(128 / segment_length)
random.seed(1234)
np.random.seed(5)

class ObjectsCollision:
    def __init__(self):
        self.methods_str = ["0_1", "0_2", "0_3", "0_4", "0_1;1_1", "0_1;1_2", "0_1;1_3", "0_2;1_1", "0_2;1_2", "0_3;1_1", "1_1;0_1", "1_1;0_2", "1_1;0_3", "1_2;0_1", "1_2;0_2", "1_3;0_1"]
        self.methods = []
        for i, method_str in enumerate(self.methods_str):
            method = [[int(y) for y in x.split('_')] for x in method_str.split(';')]
            self.methods.append(method)
        num_train = 200
        self.objects_far = joblib.load("/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/clevrer_far/models/random_forest-{}-least_confidence-original.joblib".format(num_train))
        self.objects_near = joblib.load("/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/clevrer_near/models/random_forest-{}-least_confidence-original.joblib".format(num_train))

    def generic_matching(self, sequence, left_list=None):
        if left_list is None and len(sequence) == 1:
            filtered = [lst for lst in self.list_of_list[sequence[0][0]] if lst[1] - lst[0] + 1 >= sequence[0][1]]
            return len(filtered) > 0
        if left_list is None:
            left_list = self.list_of_list[sequence[0][0]]
        if len(sequence) == 1:
            return len(left_list) > 0
        right_list = self.list_of_list[sequence[1][0]]
        join_list = []
        for left_segment in left_list:
            for right_segment in right_list:
                if left_segment[1] < right_segment[0] and left_segment[1] - left_segment[0] + 1 >= sequence[0][1] and right_segment[1] - right_segment[0] + 1 >= sequence[1][1]:
                    join_list.append([left_segment[0], right_segment[1]])
        return self.generic_matching(sequence[1:], join_list)

    @tools.tik_tok
    def detect_collision(self):
        with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_bboxes_evaluation_1000.json", 'r') as f:
            maskrcnn_bboxes_evaluation = json.loads(f.read())
        with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_video_list_evaluation_1000.json", 'r') as f:
            video_list_evaluation = json.loads(f.read())

        prediction_matrix = []
        true_labels = []
        construct_obj_list_time = 0
        read_bbox_info_time = 0
        quering_time = 0
        y_label_time = 0
        prediction_time = 0
        matching_time = 0
        for video_i, (video_basename, _, _) in enumerate(video_list_evaluation):
            print(video_i, video_basename)
            # Construct object list
            _start = time()
            obj_set = set()
            for frame_id in range(128):
                res_per_frame = maskrcnn_bboxes_evaluation["{}_{}".format(video_basename, frame_id)]
                # [81.0, 109.0, 128.0, 164.0, "metal", "blue", "cylinder"]
                for obj in res_per_frame:
                    obj_set.add("{}_{}_{}".format(obj[4], obj[5], obj[6]))
            construct_obj_list_time += time() - _start

            _start = time()
            file = "/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_proposals/sim_{}.json".format(video_basename)
            # Read in bbox info
            with open(file, 'r') as f:
                data = json.load(f)
            collisions = data["ground_truth"]["collisions"]
            objects = data["ground_truth"]["objects"]
            read_bbox_info_time += time() - _start

            _start = time()
            # Start querying
            for obj1_str_id, obj2_str_id in itertools.combinations(obj_set, 2):
                obj1_id = obj1_str_id.split("_")
                obj2_id = obj2_str_id.split("_")
                Y_true = 0

                _start_y_label = time()
                for collision in collisions:
                    pos_obj1_id, pos_obj2_id = collision["object"]
                    for obj in objects:
                        if obj["id"] == pos_obj1_id:
                            pos_obj1 = obj
                        if obj["id"] == pos_obj2_id:
                            pos_obj2 = obj
                    if (obj1_id[0] == pos_obj1["material"] and obj1_id[1] == pos_obj1["color"] and obj1_id[2] == pos_obj1["shape"] and obj2_id[0] == pos_obj2["material"] and obj2_id[1] == pos_obj2["color"] and obj2_id[2] == pos_obj2["shape"]) or (obj1_id[0] == pos_obj2["material"] and obj1_id[1] == pos_obj2["color"] and obj1_id[2] == pos_obj2["shape"] and obj2_id[0] == pos_obj1["material"] and obj2_id[1] == pos_obj1["color"] and obj2_id[2] == pos_obj1["shape"]):
                        # Positive example
                        Y_true = 1
                        break
                y_label_time += time() - _start_y_label

                true_labels.append(Y_true)
                for chunk_id in range(n_chunks):
                    _start_prediction = time()
                    row = []
                    near_fid_list = []
                    far_fid_list = []
                    spatial_features = []
                    frame_ids = []
                    for offest in range(segment_length):
                        frame_id = offest + chunk_id * 32
                        obj1 = None
                        obj2 = None
                        res_per_frame = maskrcnn_bboxes_evaluation["{}_{}".format(video_basename, frame_id)]
                        for obj in res_per_frame:
                            if obj[4] == obj1_id[0] and obj[5] == obj1_id[1] and obj[6] == obj1_id[2]:
                                obj1 = obj
                            if obj[4] == obj2_id[0] and obj[5] == obj2_id[1] and obj[6] == obj2_id[2]:
                                obj2 = obj
                        # If both objects are present in the frame, then check for collision
                        if obj1 and obj2:
                            spatial_feature = construct_spatial_feature_spatial_relationship(obj1[:4], obj2[:4])
                            spatial_features.append(spatial_feature)
                            frame_ids.append(frame_id)
                    if (len(spatial_features) > 0):
                        near_preds = self.objects_near.predict(spatial_features)
                        far_preds = self.objects_far.predict(spatial_features)
                        for pred_i, (near_pred, far_pred) in enumerate(zip(near_preds, far_preds)):
                            if near_pred == 1:
                                near_fid_list.append(frame_ids[pred_i])
                            if far_pred == 1:
                                far_fid_list.append(frame_ids[pred_i])
                    prediction_time += time() - _start_prediction

                    _start_matching = time()
                    # If there are collisions, then add the pair to the list
                    groups = groupby(near_fid_list, key=lambda item, c=count():item-next(c))
                    near_fid_list = []
                    for k, g in groups:
                        g_list = list(g)
                        near_fid_list.append([g_list[0], g_list[-1]])
                    groups = groupby(far_fid_list, key=lambda item, c=count():item-next(c))
                    far_fid_list = []
                    for k, g in groups:
                        g_list = list(g)
                        far_fid_list.append([g_list[0], g_list[-1]])
                    self.list_of_list = [near_fid_list, far_fid_list]
                    for method in self.methods:
                        if self.generic_matching(method):
                            # print(video_basename, obj1_str_id, obj2_str_id)
                            row.append(1)
                        else:
                            row.append(0)
                    prediction_matrix.append(row)
                    matching_time += time() - _start_matching
            quering_time += time() - _start

        # print all time
        print("construct_obj_list_time: {}; read_bbox_info_time: {}; quering_time: {}; y_label_time: {}; prediction_time: {}; matching_time: {}".format(construct_obj_list_time, read_bbox_info_time, quering_time, y_label_time, prediction_time, matching_time))

        with open("/gscratch/balazinska/enhaoz/complex_event_video/src/clevrer/disjunctive/prediction-1000.json", 'w') as f:
            f.write(json.dumps(prediction_matrix))

        with open("/gscratch/balazinska/enhaoz/complex_event_video/src/clevrer/disjunctive/true_labels-1000.json", 'w') as f:
            f.write(json.dumps(true_labels))

        # construct_obj_list_time: 0.33950257301330566; read_bbox_info_time: 5.088515758514404; quering_time: 102.15306210517883; y_label_time: 0.056902408599853516; prediction_time: 101.6325330734253; matching_time: 0.41710400581359863
        # detect_collision took time: 109.220s
        # find_best_queries took time: 2.603s

    @tools.tik_tok
    def find_best_queries(self):
        methods_str = ["0_1", "0_2", "0_3", "0_4", "0_1;1_1", "0_1;1_2", "0_1;1_3", "0_2;1_1", "0_2;1_2", "0_3;1_1", "1_1;0_1", "1_1;0_2", "1_1;0_3", "1_2;0_1", "1_2;0_2", "1_3;0_1"]
        methods_verbose = []
        for method_str in methods_str:
            method = [[int(y) for y in x.split('_')] for x in method_str.split(';')]
            method_verbose = ""
            for predicate in method:
                method_verbose += "{} (d={});".format("far" if predicate[0] else "near", predicate[1])
            methods_verbose.append(method_verbose)
        with open("/gscratch/balazinska/enhaoz/complex_event_video/src/clevrer/disjunctive/prediction-1000.json", 'r') as f:
            prediction_matrix = json.load(f)
        with open("/gscratch/balazinska/enhaoz/complex_event_video/src/clevrer/disjunctive/true_labels-1000.json", 'r') as f:
            true_labels = json.load(f)
        pred = np.array(prediction_matrix)
        Y = np.array(true_labels)
        f1_scores = []
        for i in range(16):
            pred_ = pred[:, i]
            # print(i, f1_score(Y, pred_))
            f1_scores.append(["{}".format(methods_verbose[i]), f1_score(Y, pred_)])
        for i, j in itertools.combinations(range(16), 2):
            pred_ = pred[:, i] | pred[:, j]
            # print(i, j, f1_score(Y, pred_))
            f1_scores.append(["{} or {}".format(methods_verbose[i], methods_verbose[j]), f1_score(Y, pred_)])
        for i, j, k in itertools.combinations(range(16), 3):
            pred_ = pred[:, i] | pred[:, j] | pred[:, k]
            # print(i, j, k, f1_score(Y, pred_))
            f1_scores.append(["{} or {} or {}".format(methods_verbose[i], methods_verbose[j], methods_verbose[k]), f1_score(Y, pred_)])

        f1_scores.sort(key=lambda x: x[1], reverse=True)
        with open("/gscratch/balazinska/enhaoz/complex_event_video/src/clevrer/disjunctive/f1_scores-1000.json", 'w') as f:
            f.write(json.dumps(f1_scores))

class CubeCollidesCylinder(ObjectsCollision):
    @tools.tik_tok
    def detect_collision(self):
        with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_bboxes_evaluation_1000.json", 'r') as f:
            maskrcnn_bboxes_evaluation = json.loads(f.read())
        with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_video_list_evaluation_1000.json", 'r') as f:
            video_list_evaluation = json.loads(f.read())

        prediction_matrix = []
        true_labels = []
        construct_obj_list_time = 0
        read_bbox_info_time = 0
        quering_time = 0
        y_label_time = 0
        prediction_time = 0
        matching_time = 0
        for video_i, (video_basename, _, _) in enumerate(video_list_evaluation):
            print(video_i, video_basename)
            # Construct object list
            _start = time()
            obj_set = set()
            for frame_id in range(128):
                res_per_frame = maskrcnn_bboxes_evaluation["{}_{}".format(video_basename, frame_id)]
                # [81.0, 109.0, 128.0, 164.0, "metal", "blue", "cylinder"]
                for obj in res_per_frame:
                    obj_set.add("{}_{}_{}".format(obj[4], obj[5], obj[6]))
            construct_obj_list_time += time() - _start

            _start = time()
            file = "/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_proposals/sim_{}.json".format(video_basename)
            # Read in bbox info
            with open(file, 'r') as f:
                data = json.load(f)
            collisions = data["ground_truth"]["collisions"]
            objects = data["ground_truth"]["objects"]
            read_bbox_info_time += time() - _start

            _start = time()
            # Start querying
            for obj1_str_id, obj2_str_id in itertools.combinations(obj_set, 2):
                obj1_id = obj1_str_id.split("_")
                obj2_id = obj2_str_id.split("_")
                Y_true = 0

                _start_y_label = time()
                for collision in collisions:
                    pos_obj1_id, pos_obj2_id = collision["object"]
                    for obj in objects:
                        if obj["id"] == pos_obj1_id:
                            pos_obj1 = obj
                        if obj["id"] == pos_obj2_id:
                            pos_obj2 = obj
                    if (obj1_id[0] == pos_obj1["material"] and obj1_id[1] == pos_obj1["color"] and obj1_id[2] == pos_obj1["shape"] and obj2_id[0] == pos_obj2["material"] and obj2_id[1] == pos_obj2["color"] and obj2_id[2] == pos_obj2["shape"]) or (obj1_id[0] == pos_obj2["material"] and obj1_id[1] == pos_obj2["color"] and obj1_id[2] == pos_obj2["shape"] and obj2_id[0] == pos_obj1["material"] and obj2_id[1] == pos_obj1["color"] and obj2_id[2] == pos_obj1["shape"]):
                        # Positive example
                        Y_true = 1
                        break
                y_label_time += time() - _start_y_label

                true_labels.append(Y_true)
                for chunk_id in range(n_chunks):
                    _start_prediction = time()
                    row = []
                    near_fid_list = []
                    far_fid_list = []
                    spatial_features = []
                    frame_ids = []
                    for offest in range(segment_length):
                        frame_id = offest + chunk_id * 32
                        obj1 = None
                        obj2 = None
                        res_per_frame = maskrcnn_bboxes_evaluation["{}_{}".format(video_basename, frame_id)]
                        for obj in res_per_frame:
                            if obj[4] == obj1_id[0] and obj[5] == obj1_id[1] and obj[6] == obj1_id[2]:
                                obj1 = obj
                            if obj[4] == obj2_id[0] and obj[5] == obj2_id[1] and obj[6] == obj2_id[2]:
                                obj2 = obj
                        # If both objects are present in the frame, then check for collision
                        if obj1 and obj2:
                            spatial_feature = construct_spatial_feature_spatial_relationship(obj1[:4], obj2[:4])
                            spatial_features.append(spatial_feature)
                            frame_ids.append(frame_id)
                    if (len(spatial_features) > 0):
                        near_preds = self.objects_near.predict(spatial_features)
                        far_preds = self.objects_far.predict(spatial_features)
                        for pred_i, (near_pred, far_pred) in enumerate(zip(near_preds, far_preds)):
                            if near_pred == 1:
                                near_fid_list.append(frame_ids[pred_i])
                            if far_pred == 1:
                                far_fid_list.append(frame_ids[pred_i])
                    prediction_time += time() - _start_prediction

                    _start_matching = time()
                    # If there are collisions, then add the pair to the list
                    groups = groupby(near_fid_list, key=lambda item, c=count():item-next(c))
                    near_fid_list = []
                    for k, g in groups:
                        g_list = list(g)
                        near_fid_list.append([g_list[0], g_list[-1]])
                    groups = groupby(far_fid_list, key=lambda item, c=count():item-next(c))
                    far_fid_list = []
                    for k, g in groups:
                        g_list = list(g)
                        far_fid_list.append([g_list[0], g_list[-1]])
                    self.list_of_list = [near_fid_list, far_fid_list]
                    for method in self.methods:
                        if self.generic_matching(method):
                            # print(video_basename, obj1_str_id, obj2_str_id)
                            row.append(1)
                        else:
                            row.append(0)
                    prediction_matrix.append(row)
                    matching_time += time() - _start_matching
            quering_time += time() - _start

        # print all time
        print("construct_obj_list_time: {}; read_bbox_info_time: {}; quering_time: {}; y_label_time: {}; prediction_time: {}; matching_time: {}".format(construct_obj_list_time, read_bbox_info_time, quering_time, y_label_time, prediction_time, matching_time))

        with open("/gscratch/balazinska/enhaoz/complex_event_video/src/clevrer/disjunctive/prediction-1000.json", 'w') as f:
            f.write(json.dumps(prediction_matrix))

        with open("/gscratch/balazinska/enhaoz/complex_event_video/src/clevrer/disjunctive/true_labels-1000.json", 'w') as f:
            f.write(json.dumps(true_labels))

if __name__ == '__main__':
    test_query = ObjectsCollision()
    test_query.detect_collision()
    test_query.find_best_queries()