"""
This is a copy of topk_queries.py and is used for searching disjunctive queries.
"""

from calendar import c
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
from sklearn.metrics import f1_score, accuracy_score
from utils import tools
from time import time
import os
import math

segment_length = 128
n_chunks = int(128 / segment_length)
random.seed(1234)
np.random.seed(5)

class ObjectsCollision:
    def __init__(self, n_validation_videos=1000):
        self.predicate_names = ["near", "far"]
        self.output_dir = "/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/clevrer_query_exploration"
        self.target_query_name="objects_collision"
        self.n_validation_videos = n_validation_videos
        self.methods_str = ["0_1", "0_2", "0_3", "0_4", "0_1;1_1", "0_2;1_2", "0_3;1_3", "0_4;1_4", "1_1;0_1", "1_2;0_2", "1_3;0_3", "1_4;0_4", "1_1;0_1;1_1", "1_1;0_2;1_1", "1_1;0_3;1_1"]
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

    def attribute_predicate(self, obj1_id, obj2_id):
        return True

    @tools.tik_tok
    def detect_collision(self):
        with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_bboxes_evaluation_1000.json", 'r') as f:
            maskrcnn_bboxes_evaluation = json.loads(f.read())
        with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_video_list_evaluation_1000.json", 'r') as f:
            video_list_evaluation = json.loads(f.read())

        prediction_matrix_pair_level = []
        prediction_matrix_video_level = []
        true_labels_pair_level = []
        true_labels_video_level = []
        read_bbox_info_time = 0
        prediction_time = 0

        # sample_pos_inputs = []
        # sample_neg_inputs = []

        for video_i, (video_basename, _, _) in enumerate(video_list_evaluation):
            print(video_i, video_basename)
            prediction_matrix_pair_level_per_video = []
            true_labels_pair_level_per_video = []
            # Construct object list
            obj_set = set()
            for frame_id in range(128):
                res_per_frame = maskrcnn_bboxes_evaluation["{}_{}".format(video_basename, frame_id)]
                # [81.0, 109.0, 128.0, 164.0, "metal", "blue", "cylinder"]
                for obj in res_per_frame:
                    obj_set.add("{}_{}_{}".format(obj[4], obj[5], obj[6]))

            _start = time()
            file = "/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_proposals/sim_{}.json".format(video_basename)
            # Read in bbox info
            with open(file, 'r') as f:
                data = json.load(f)
            collisions = data["ground_truth"]["collisions"]
            objects = data["ground_truth"]["objects"]
            read_bbox_info_time += time() - _start

            # Start querying
            for obj1_str_id, obj2_str_id in itertools.combinations(obj_set, 2):
                obj1_id = obj1_str_id.split("_")
                obj2_id = obj2_str_id.split("_")
                if not self.attribute_predicate(obj1_id, obj2_id):
                    continue
                Y_true = 0

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

                # if Y_true and len(sample_pos_inputs) >= 20:
                #     continue
                # if not Y_true and len(sample_neg_inputs) >= 20:
                #     continue
                # sample_input = [[], []]
                true_labels_pair_level_per_video.append(Y_true)
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
                            # sample_input[0].append(obj1[:4])
                            # sample_input[1].append(obj2[:4])
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

                    # if Y_true:
                    #     sample_pos_inputs.append(sample_input)
                    # else:
                    #     sample_neg_inputs.append(sample_input)
                    # if len(sample_pos_inputs) >= 20 and len(sample_neg_inputs) >= 20:
                    #     with open(os.path.join(self.output_dir, self.target_query_name, "sample_inputs.json"), 'w') as f:
                    #         sample_pos_inputs.extend(sample_neg_inputs)
                    #         f.write(json.dumps(sample_pos_inputs))
                    #     with open(os.path.join(self.output_dir, self.target_query_name, "sample_outputs.json"), 'w') as f:
                    #         f.write(json.dumps([1] * 20 + [0] * 20))
                    #     return
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
                    prediction_matrix_pair_level_per_video.append(row)
            prediction_matrix_pair_level = [*prediction_matrix_pair_level, *prediction_matrix_pair_level_per_video]
            if len(prediction_matrix_pair_level_per_video):
                prediction_matrix_video_level.append(np.max(np.asarray(prediction_matrix_pair_level_per_video), axis=0).tolist())
            else:
                prediction_matrix_video_level.append([0] * len(self.methods))
            true_labels_pair_level = [*true_labels_pair_level, *true_labels_pair_level_per_video]
            if len(true_labels_pair_level_per_video):
                true_labels_video_level.append(max(true_labels_pair_level_per_video))
            else:
                true_labels_video_level.append(0)

        # print all time
        print("read_bbox_info_time: {}; prediction_time: {}".format(read_bbox_info_time, prediction_time))

        # if folder not exists, then create it,
        if not os.path.exists(os.path.join(self.output_dir, self.target_query_name)):
            os.makedirs(os.path.join(self.output_dir, self.target_query_name))

        with open(os.path.join(self.output_dir, self.target_query_name, "prediction-pair_level-{}.json".format(self.n_validation_videos)), 'w') as f:
            f.write(json.dumps(prediction_matrix_pair_level))

        with open(os.path.join(self.output_dir, self.target_query_name, "prediction-video_level-{}.json".format(self.n_validation_videos)), 'w') as f:
            f.write(json.dumps(prediction_matrix_video_level))

        with open(os.path.join(self.output_dir, self.target_query_name, "true_labels-pair_level-{}.json".format(self.n_validation_videos)), 'w') as f:
            f.write(json.dumps(true_labels_pair_level))

        with open(os.path.join(self.output_dir, self.target_query_name, "true_labels-video_level-{}.json".format(self.n_validation_videos)), 'w') as f:
            f.write(json.dumps(true_labels_video_level))

        # construct_obj_list_time: 0.3353233337402344; read_bbox_info_time: 17.369605779647827; quering_time: 101.47075414657593; y_label_time: 0.055441856384277344; prediction_time: 100.84484648704529; matching_time: 0.40783190727233887
        # detect_collision took time: 120.725s
        # find_best_queries took time: 3.228s

    @tools.tik_tok
    def find_best_queries(self):
        methods_verbose = []
        for method_str in self.methods_str:
            method = [[int(y) for y in x.split('_')] for x in method_str.split(';')]
            method_verbose = ""
            for predicate in method:
                method_verbose += "{} (d={});".format(self.predicate_names[1] if predicate[0] else self.predicate_names[0], predicate[1])
            methods_verbose.append(method_verbose)

        with open(os.path.join(self.output_dir, self.target_query_name, "prediction-pair_level-{}.json".format(self.n_validation_videos)), 'r') as f:
            prediction_matrix_pair_level = json.load(f)
        with open(os.path.join(self.output_dir, self.target_query_name, "prediction-video_level-{}.json".format(self.n_validation_videos)), 'r') as f:
            prediction_matrix_video_level = json.load(f)
        with open(os.path.join(self.output_dir, self.target_query_name, "true_labels-pair_level-{}.json".format(self.n_validation_videos)), 'r') as f:
            true_labels_pair_level = json.load(f)
        with open(os.path.join(self.output_dir, self.target_query_name, "true_labels-video_level-{}.json".format(self.n_validation_videos)), 'r') as f:
            true_labels_video_level = json.load(f)

        # Compute pair-level f1 score
        pred = np.array(prediction_matrix_pair_level)
        Y = np.array(true_labels_pair_level)
        f1_scores_pair_level_disjunctive = []
        f1_scores_pair_level_simple = []
        for i in range(len(self.methods_str)):
            pred_ = pred[:, i]
            # print(i, f1_score(Y, pred_))
            f1_scores_pair_level_simple.append(["{}".format(methods_verbose[i]), accuracy_score(Y, pred_)])
            f1_scores_pair_level_disjunctive.append(["{}".format(methods_verbose[i]), accuracy_score(Y, pred_)])
        for i, j in itertools.combinations(range(len(self.methods_str)), 2):
            pred_ = pred[:, i] | pred[:, j]
            # print(i, j, f1_score(Y, pred_))
            f1_scores_pair_level_disjunctive.append(["{} or {}".format(methods_verbose[i], methods_verbose[j]), accuracy_score(Y, pred_)])
        for i, j, k in itertools.combinations(range(len(self.methods_str)), 3):
            pred_ = pred[:, i] | pred[:, j] | pred[:, k]
            # print(i, j, k, f1_score(Y, pred_))
            f1_scores_pair_level_disjunctive.append(["{} or {} or {}".format(methods_verbose[i], methods_verbose[j], methods_verbose[k]), accuracy_score(Y, pred_)])

        f1_scores_pair_level_disjunctive.sort(key=lambda x: x[1], reverse=True)
        f1_scores_pair_level_simple.sort(key=lambda x: x[1], reverse=True)
        with open(os.path.join(self.output_dir, self.target_query_name, "accuracy_score-pair_level-disjunctive-{}.json".format(self.n_validation_videos)), 'w') as f:
            f.write(json.dumps(f1_scores_pair_level_disjunctive))
        with open(os.path.join(self.output_dir, self.target_query_name, "accuracy_score-pair_level-simple-{}.json".format(self.n_validation_videos)), 'w') as f:
            f.write(json.dumps(f1_scores_pair_level_simple))

        # Compute video-level f1 score
        pred = np.array(prediction_matrix_video_level)
        Y = np.array(true_labels_video_level)
        f1_scores_video_level_disjunctive = []
        f1_scores_video_level_simple = []
        for i in range(len(self.methods_str)):
            pred_ = pred[:, i]
            # print(i, f1_score(Y, pred_))
            f1_scores_video_level_simple.append(["{}".format(methods_verbose[i]), f1_score(Y, pred_)])
            f1_scores_video_level_disjunctive.append(["{}".format(methods_verbose[i]), f1_score(Y, pred_)])
        for i, j in itertools.combinations(range(len(self.methods_str)), 2):
            pred_ = pred[:, i] | pred[:, j]
            # print(i, j, f1_score(Y, pred_))
            f1_scores_video_level_disjunctive.append(["{} or {}".format(methods_verbose[i], methods_verbose[j]), f1_score(Y, pred_)])
        for i, j, k in itertools.combinations(range(len(self.methods_str)), 3):
            pred_ = pred[:, i] | pred[:, j] | pred[:, k]
            # print(i, j, k, f1_score(Y, pred_))
            f1_scores_video_level_disjunctive.append(["{} or {} or {}".format(methods_verbose[i], methods_verbose[j], methods_verbose[k]), f1_score(Y, pred_)])

        f1_scores_video_level_disjunctive.sort(key=lambda x: x[1], reverse=True)
        f1_scores_video_level_simple.sort(key=lambda x: x[1], reverse=True)
        with open(os.path.join(self.output_dir, self.target_query_name, "f1_scores-video_level-disjunctive-{}.json".format(self.n_validation_videos)), 'w') as f:
            f.write(json.dumps(f1_scores_video_level_disjunctive))
        with open(os.path.join(self.output_dir, self.target_query_name, "f1_scores-video_level-simple-{}.json".format(self.n_validation_videos)), 'w') as f:
            f.write(json.dumps(f1_scores_video_level_simple))

class CubeCollidesCylinder(ObjectsCollision):
    def __init__(self):
        super().__init__()
        self.target_query_name = "cube_collides_cylinder"

    def attribute_predicate(self, obj1_id, obj2_id):
        return (obj1_id[2] == "cube" and obj2_id[2] == "cylinder") or (obj1_id[2] == "cylinder" and obj2_id[2] == "cube")

class MetalCubeCollidesSphere(ObjectsCollision):
    def __init__(self):
        super().__init__()
        self.target_query_name = "metal_cube_collides_sphere"

    def attribute_predicate(self, obj1_id, obj2_id):
        return (obj1_id[2] == "cube" and obj1_id[0] == "metal" and obj2_id[2] == "sphere") or (obj2_id[2] == "cube" and obj2_id[0] == "metal" and obj1_id[2] == "sphere")
# construct_obj_list_time: 0.29500699043273926; read_bbox_info_time: 20.272231101989746; quering_time: 13.35873794555664; y_label_time: 0.007788896560668945; prediction_time: 13.25111985206604; matching_time: 0.05533146858215332
# detect_collision took time: 70.302s
# find_best_queries took time: 1.052s

class MetalCubeCollidesRubberSphere(ObjectsCollision):
    def __init__(self):
        super().__init__()
        self.target_query_name = "metal_cube_collides_rubber_sphere"

    def attribute_predicate(self, obj1_id, obj2_id):
        return (obj1_id[2] == "cube" and obj1_id[0] == "metal" and obj2_id[2] == "sphere" and obj2_id[0] == "rubber") or (obj2_id[2] == "cube" and obj2_id[0] == "metal" and obj1_id[2] == "sphere" and obj1_id[0] == "rubber")
# construct_obj_list_time: 0.2992737293243408; read_bbox_info_time: 20.876301765441895; quering_time: 6.520001411437988; y_label_time: 0.0037567615509033203; prediction_time: 6.456456661224365; matching_time: 0.027746915817260742
# detect_collision took time: 29.365s
# find_best_queries took time: 0.093s

class RedCubeCollidesSphere(ObjectsCollision):
    def __init__(self):
        super().__init__()
        self.target_query_name = "red_cube_collides_sphere"

    def attribute_predicate(self, obj1_id, obj2_id):
        return (obj1_id[2] == "cube" and obj1_id[1] == "red" and obj2_id[2] == "sphere") or (obj2_id[2] == "cube" and obj2_id[1] == "red" and obj1_id[2] == "sphere")
# construct_obj_list_time: 0.28928112983703613; read_bbox_info_time: 19.98965573310852; quering_time: 2.673443078994751; y_label_time: 0.0014295578002929688; prediction_time: 2.640381097793579; matching_time: 0.011377811431884766
# detect_collision took time: 24.663s
# find_best_queries took time: 0.083s

class RedCubeCollidesGreenSphere(ObjectsCollision):
    def __init__(self):
        super().__init__()
        self.target_query_name = "red_cube_collides_green_sphere"

    def attribute_predicate(self, obj1_id, obj2_id):
        return (obj1_id[2] == "cube" and obj1_id[1] == "red" and obj2_id[2] == "sphere" and obj2_id[1] == "green") or (obj2_id[2] == "cube" and obj2_id[1] == "red" and obj1_id[2] == "sphere" and obj1_id[1] == "green")

class ObjectsCollisionPerfect(ObjectsCollision):
    def __init__(self, near_thresh, far_thresh):
        super().__init__()
        self.target_query_name = "object_collision_perfect-{}-{}".format(near_thresh, far_thresh)
        self.near_thresh = near_thresh
        self.far_thresh = far_thresh

    @tools.tik_tok
    def detect_collision(self):
        with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_bboxes_evaluation_1000.json", 'r') as f:
            maskrcnn_bboxes_evaluation = json.loads(f.read())
        with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_video_list_evaluation_1000.json", 'r') as f:
            video_list_evaluation = json.loads(f.read())

        prediction_matrix_pair_level = []
        prediction_matrix_video_level = []
        true_labels_pair_level = []
        true_labels_video_level = []
        read_bbox_info_time = 0
        prediction_time = 0
        misclassified_results = []
        for video_i, (video_basename, _, _) in enumerate(video_list_evaluation):
            print(video_i, video_basename)
            prediction_matrix_pair_level_per_video = []
            true_labels_pair_level_per_video = []
            # Construct object list
            obj_set = set()
            for frame_id in range(128):
                res_per_frame = maskrcnn_bboxes_evaluation["{}_{}".format(video_basename, frame_id)]
                # [81.0, 109.0, 128.0, 164.0, "metal", "blue", "cylinder"]
                for obj in res_per_frame:
                    obj_set.add("{}_{}_{}".format(obj[4], obj[5], obj[6]))

            _start = time()
            file = "/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_proposals/sim_{}.json".format(video_basename)
            # Read in bbox info
            with open(file, 'r') as f:
                data = json.load(f)
            collisions = data["ground_truth"]["collisions"]
            objects = data["ground_truth"]["objects"]
            read_bbox_info_time += time() - _start

            # Start querying
            for obj1_str_id, obj2_str_id in itertools.combinations(obj_set, 2):
                obj1_id = obj1_str_id.split("_")
                obj2_id = obj2_str_id.split("_")
                if not self.attribute_predicate(obj1_id, obj2_id):
                    continue
                Y_true = 0

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

                true_labels_pair_level_per_video.append(Y_true)
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
                            spatial_feature = [*obj1[:4], *obj2[:4]]
                            spatial_features.append(spatial_feature)
                            frame_ids.append(frame_id)
                    if (len(spatial_features) > 0):
                        near_preds = self.objects_near_perfect(spatial_features)
                        far_preds = self.objects_far_perfect(spatial_features)
                        for pred_i, (near_pred, far_pred) in enumerate(zip(near_preds, far_preds)):
                            if near_pred == 1:
                                near_fid_list.append(frame_ids[pred_i])
                            if far_pred == 1:
                                far_fid_list.append(frame_ids[pred_i])
                    prediction_time += time() - _start_prediction

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
                            if method == [[1, 1], [0, 1]] and Y_true == 0:
                                misclassified_results.append(["FP", video_basename, obj1_str_id, obj2_str_id, near_fid_list, far_fid_list])
                        else:
                            row.append(0)
                            if method == [[1, 1], [0, 1]] and Y_true == 1:
                                misclassified_results.append(["FN", video_basename, obj1_str_id, obj2_str_id, near_fid_list, far_fid_list])
                    prediction_matrix_pair_level_per_video.append(row)
            prediction_matrix_pair_level = [*prediction_matrix_pair_level, *prediction_matrix_pair_level_per_video]
            if len(prediction_matrix_pair_level_per_video):
                prediction_matrix_video_level.append(np.max(np.asarray(prediction_matrix_pair_level_per_video), axis=0).tolist())
            else:
                prediction_matrix_video_level.append([0] * len(self.methods))
            true_labels_pair_level = [*true_labels_pair_level, *true_labels_pair_level_per_video]
            if len(true_labels_pair_level_per_video):
                true_labels_video_level.append(max(true_labels_pair_level_per_video))
            else:
                true_labels_video_level.append(0)

        # print all time
        print("read_bbox_info_time: {}; prediction_time: {}".format(read_bbox_info_time, prediction_time))

        # if folder not exists, then create it,
        if not os.path.exists(os.path.join(self.output_dir, self.target_query_name)):
            os.makedirs(os.path.join(self.output_dir, self.target_query_name))

        with open(os.path.join(self.output_dir, self.target_query_name, "prediction-pair_level-{}.json".format(self.n_validation_videos)), 'w') as f:
            f.write(json.dumps(prediction_matrix_pair_level))

        with open(os.path.join(self.output_dir, self.target_query_name, "prediction-video_level-{}.json".format(self.n_validation_videos)), 'w') as f:
            f.write(json.dumps(prediction_matrix_video_level))

        with open(os.path.join(self.output_dir, self.target_query_name, "true_labels-pair_level-{}.json".format(self.n_validation_videos)), 'w') as f:
            f.write(json.dumps(true_labels_pair_level))

        with open(os.path.join(self.output_dir, self.target_query_name, "true_labels-video_level-{}.json".format(self.n_validation_videos)), 'w') as f:
            f.write(json.dumps(true_labels_video_level))

        with open(os.path.join(self.output_dir, self.target_query_name, "misclassified-{}.json".format(self.n_validation_videos)), 'w') as f:
            f.write(json.dumps(misclassified_results))

    def objects_near_perfect(self, spatial_features):
        """
        This function is used to check if the objects are near each other by directly reasoning on their bounding boxes.
        """
        near_preds = []
        for spatial_feature in spatial_features:
            near_preds.append(self.obj_distance(spatial_feature[:4], spatial_feature[4:]) <= self.near_thresh)
        return near_preds

    def objects_far_perfect(self, spatial_features):
        """
        This function is used to check if the objects are far away from each other by directly reasoning on their bounding boxes.
        """
        far_preds = []
        for spatial_feature in spatial_features:
            far_preds.append(self.obj_distance(spatial_feature[:4], spatial_feature[4:]) > self.far_thresh)
        return far_preds

    @staticmethod
    def obj_distance(bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        cx1 = (x1 + x2) / 2
        cy1 = (y1 + y2) / 2
        cx2 = (x3 + x4) / 2
        cy2 = (y3 + y4) / 2
        return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) / ((x2 - x1 + x4 - x3) / 2)

    # @staticmethod
    # def obj_distance(bbox1, bbox2):
    #     x1, y1, x2, y2 = bbox1
    #     x3, y3, x4, y4 = bbox2
    #     cx1 = (x1 + x2) / 2
    #     cy1 = (y1 + y2) / 2
    #     cx2 = (x3 + x4) / 2
    #     cy2 = (y3 + y4) / 2
    #     weight_y = abs(cy1 - cy2) / (abs(cx1 - cx2) + abs(cy1 - cy2) + 1e-6)
    #     weight_x = abs(cx1 - cx2) / (abs(cx1 - cx2) + abs(cy1 - cy2) + 1e-6)
    #     return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) / (weight_x * (x2 - x1 + x4 - x3) / 2 + weight_y * (y2 - y1 + y4 - y3) / 2 + 1e-6)


class InOutPerfect(ObjectsCollision):
    def __init__(self, edge_thresh, center_thresh, n_validation_videos=1000):
        self.predicate_names = ["edge", "center"]
        self.misclassified_method = None
        self.edge_thresh = edge_thresh
        self.center_thresh = center_thresh
        self.output_dir = "/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/clevrer_query_exploration"
        self.n_validation_videos = n_validation_videos
        self.methods_str = ["0_1", "0_2", "0_3", "0_4", "0_1;1_1", "0_2;1_2", "0_3;1_3", "0_4;1_4", "1_1;0_1", "1_2;0_2", "1_3;0_3", "1_4;0_4", "1_1;0_1;1_1", "1_1;0_2;1_1", "1_1;0_3;1_1"]
        self.methods = []
        for i, method_str in enumerate(self.methods_str):
            method = [[int(y) for y in x.split('_')] for x in method_str.split(';')]
            self.methods.append(method)

    @tools.tik_tok
    def detect_collision(self, in_or_out):
        with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_bboxes_evaluation_1000.json", 'r') as f:
            maskrcnn_bboxes_evaluation = json.loads(f.read())
        with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_video_list_evaluation_1000.json", 'r') as f:
            video_list_evaluation = json.loads(f.read())

        prediction_matrix_pair_level = []
        prediction_matrix_video_level = []
        true_labels_pair_level = []
        true_labels_video_level = []
        read_bbox_info_time = 0
        prediction_time = 0
        misclassified_results = []
        for video_i, (video_basename, _, _) in enumerate(video_list_evaluation):
            print(video_i, video_basename)
            prediction_matrix_pair_level_per_video = []
            true_labels_pair_level_per_video = []
            # Construct object list
            obj_set = set()
            for frame_id in range(128):
                res_per_frame = maskrcnn_bboxes_evaluation["{}_{}".format(video_basename, frame_id)]
                # [81.0, 109.0, 128.0, 164.0, "metal", "blue", "cylinder"]
                for obj in res_per_frame:
                    obj_set.add("{}_{}_{}".format(obj[4], obj[5], obj[6]))

            _start = time()
            file = "/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_proposals/sim_{}.json".format(video_basename)
            # Read in bbox info
            with open(file, 'r') as f:
                data = json.load(f)
            in_outs = data["ground_truth"]["in_outs"]
            objects = data["ground_truth"]["objects"]
            read_bbox_info_time += time() - _start

            # Start querying
            for obj1_str_id in obj_set:
                obj1_id = obj1_str_id.split("_")
                Y_true = 0

                for in_out in in_outs:
                    if in_out["type"] != in_or_out:
                        continue
                    pos_obj1_id = in_out["object"]
                    for obj in objects:
                        if obj["id"] == pos_obj1_id:
                            pos_obj1 = obj
                    if (obj1_id[0] == pos_obj1["material"] and obj1_id[1] == pos_obj1["color"] and obj1_id[2] == pos_obj1["shape"]):
                        # Positive example
                        Y_true = 1
                        break

                true_labels_pair_level_per_video.append(Y_true)
                for chunk_id in range(n_chunks):
                    _start_prediction = time()
                    row = []
                    edge_fid_list = []
                    center_fid_list = []
                    spatial_features = []
                    frame_ids = []
                    for offest in range(segment_length):
                        frame_id = offest + chunk_id * 32
                        obj1 = None
                        res_per_frame = maskrcnn_bboxes_evaluation["{}_{}".format(video_basename, frame_id)]
                        for obj in res_per_frame:
                            if obj[4] == obj1_id[0] and obj[5] == obj1_id[1] and obj[6] == obj1_id[2]:
                                obj1 = obj
                        # If both objects are present in the frame, then check for collision
                        if obj1:
                            spatial_feature = obj1[:4]
                            spatial_features.append(spatial_feature)
                            frame_ids.append(frame_id)
                    if (len(spatial_features) > 0):
                        edge_preds = self.objects_edge_perfect(spatial_features)
                        center_preds = self.objects_center_perfect(spatial_features)
                        for pred_i, (edge_pred, center_pred) in enumerate(zip(edge_preds, center_preds)):
                            if edge_pred == 1:
                                edge_fid_list.append(frame_ids[pred_i])
                            if center_pred == 1:
                                center_fid_list.append(frame_ids[pred_i])
                    prediction_time += time() - _start_prediction

                    # If there are collisions, then add the pair to the list
                    groups = groupby(edge_fid_list, key=lambda item, c=count():item-next(c))
                    edge_fid_list = []
                    for k, g in groups:
                        g_list = list(g)
                        edge_fid_list.append([g_list[0], g_list[-1]])
                    groups = groupby(center_fid_list, key=lambda item, c=count():item-next(c))
                    center_fid_list = []
                    for k, g in groups:
                        g_list = list(g)
                        center_fid_list.append([g_list[0], g_list[-1]])
                    self.list_of_list = [edge_fid_list, center_fid_list]
                    for method in self.methods:
                        if self.pattern_matching(method):
                            # print(video_basename, obj1_str_id, obj2_str_id)
                            row.append(1)
                            if method == self.misclassified_method and Y_true == 0:
                                misclassified_results.append(["FP", video_basename, obj1_str_id, edge_fid_list, center_fid_list])
                        else:
                            row.append(0)
                            if method == self.misclassified_method and Y_true == 1:
                                misclassified_results.append(["FN", video_basename, obj1_str_id, edge_fid_list, center_fid_list])
                    prediction_matrix_pair_level_per_video.append(row)
            prediction_matrix_pair_level = [*prediction_matrix_pair_level, *prediction_matrix_pair_level_per_video]
            if len(prediction_matrix_pair_level_per_video):
                prediction_matrix_video_level.append(np.max(np.asarray(prediction_matrix_pair_level_per_video), axis=0).tolist())
            else:
                prediction_matrix_video_level.append([0] * len(self.methods))
            true_labels_pair_level = [*true_labels_pair_level, *true_labels_pair_level_per_video]
            if len(true_labels_pair_level_per_video):
                true_labels_video_level.append(max(true_labels_pair_level_per_video))
            else:
                true_labels_video_level.append(0)

        # print all time
        print("read_bbox_info_time: {}; prediction_time: {}".format(read_bbox_info_time, prediction_time))

        # if folder not exists, then create it,
        if not os.path.exists(os.path.join(self.output_dir, self.target_query_name)):
            os.makedirs(os.path.join(self.output_dir, self.target_query_name))

        with open(os.path.join(self.output_dir, self.target_query_name, "prediction-pair_level-{}.json".format(self.n_validation_videos)), 'w') as f:
            f.write(json.dumps(prediction_matrix_pair_level))

        with open(os.path.join(self.output_dir, self.target_query_name, "prediction-video_level-{}.json".format(self.n_validation_videos)), 'w') as f:
            f.write(json.dumps(prediction_matrix_video_level))

        with open(os.path.join(self.output_dir, self.target_query_name, "true_labels-pair_level-{}.json".format(self.n_validation_videos)), 'w') as f:
            f.write(json.dumps(true_labels_pair_level))

        with open(os.path.join(self.output_dir, self.target_query_name, "true_labels-video_level-{}.json".format(self.n_validation_videos)), 'w') as f:
            f.write(json.dumps(true_labels_video_level))

        with open(os.path.join(self.output_dir, self.target_query_name, "misclassified-{}.json".format(self.n_validation_videos)), 'w') as f:
            f.write(json.dumps(misclassified_results))

    def pattern_matching(self, method):
        return self.generic_matching(method)

    def objects_edge_perfect(self, spatial_features):
        """
        This function is used to check if the objects are near each other by directly reasoning on their bounding boxes.
        """
        edge_preds = []
        for spatial_feature in spatial_features:
            edge_preds.append(self.distance_to_boundary(spatial_feature) <= self.edge_thresh)
        return edge_preds

    def objects_center_perfect(self, spatial_features):
        """
        This function is used to check if the objects are far away from each other by directly reasoning on their bounding boxes.
        """
        center_preds = []
        for spatial_feature in spatial_features:
            center_preds.append(self.distance_to_boundary(spatial_feature) > self.center_thresh)
        return center_preds

    @staticmethod
    def distance_to_boundary(bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return min(cx, 480 - cx, cy, 320 - cy)

class InPerfect(InOutPerfect):
    def __init__(self, edge_thresh, center_thresh, n_validation_videos=1000):
        super().__init__(edge_thresh, center_thresh, n_validation_videos)
        self.misclassified_method = [[0, 1], [1, 1]]
        self.target_query_name = "in_perfect-{}-{}".format(edge_thresh, center_thresh)

    def detect_collision(self):
        super().detect_collision("in")

class InTrivial(InOutPerfect):
    def __init__(self, edge_thresh, center_thresh, n_validation_videos=1000):
        super().__init__(edge_thresh, center_thresh, n_validation_videos)
        self.misclassified_method = [[0, 1], [1, 1]]
        self.target_query_name = "in_trivial-{}-{}".format(edge_thresh, center_thresh)

    def detect_collision(self):
        with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_bboxes_evaluation_1000.json", 'r') as f:
            maskrcnn_bboxes_evaluation = json.loads(f.read())
        with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_video_list_evaluation_1000.json", 'r') as f:
            video_list_evaluation = json.loads(f.read())

        prediction_matrix_pair_level = []
        true_labels_pair_level = []
        read_bbox_info_time = 0
        misclassified_results = []
        for video_i, (video_basename, _, _) in enumerate(video_list_evaluation):
            print(video_i, video_basename)
            # Construct object list
            obj_set = set()
            for frame_id in range(128):
                res_per_frame = maskrcnn_bboxes_evaluation["{}_{}".format(video_basename, frame_id)]
                # [81.0, 109.0, 128.0, 164.0, "metal", "blue", "cylinder"]
                for obj in res_per_frame:
                    obj_set.add("{}_{}_{}".format(obj[4], obj[5], obj[6]))

            _start = time()
            file = "/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_proposals/sim_{}.json".format(video_basename)
            # Read in bbox info
            with open(file, 'r') as f:
                data = json.load(f)
            in_outs = data["ground_truth"]["in_outs"]
            objects = data["ground_truth"]["objects"]
            read_bbox_info_time += time() - _start

            # Start querying
            for obj1_str_id in obj_set:
                obj1_id = obj1_str_id.split("_")
                Y_true = 0

                for in_out in in_outs:
                    if in_out["type"] != "in":
                        continue
                    pos_obj1_id = in_out["object"]
                    for obj in objects:
                        if obj["id"] == pos_obj1_id:
                            pos_obj1 = obj
                    if (obj1_id[0] == pos_obj1["material"] and obj1_id[1] == pos_obj1["color"] and obj1_id[2] == pos_obj1["shape"]):
                        # Positive example
                        Y_true = 1
                        break

                true_labels_pair_level.append(Y_true)
                Y_pred = 0
                for chunk_id in range(n_chunks):
                    frame_ids = []
                    for offest in range(segment_length):
                        frame_id = offest + chunk_id * 32
                        obj1 = None
                        res_per_frame = maskrcnn_bboxes_evaluation["{}_{}".format(video_basename, frame_id)]
                        for obj in res_per_frame:
                            if obj[4] == obj1_id[0] and obj[5] == obj1_id[1] and obj[6] == obj1_id[2]:
                                obj1 = obj
                        # If both objects are present in the frame, then check for collision
                        if obj1:
                            frame_ids.append(frame_id)
                if len(frame_ids) and frame_ids[0] > 1:
                    Y_pred = 1
                prediction_matrix_pair_level.append(Y_pred)
                groups = groupby(frame_ids, key=lambda item, c=count():item-next(c))
                frame_ids = []
                for k, g in groups:
                    g_list = list(g)
                    frame_ids.append([g_list[0], g_list[-1]])
                if Y_true == 1 and Y_pred == 0:
                    misclassified_results.append(["FN", video_basename, obj1_str_id, frame_ids])
                if Y_true == 0 and Y_pred == 1:
                    misclassified_results.append(["FP", video_basename, obj1_str_id, frame_ids])
        print("f1 score:", f1_score(true_labels_pair_level, prediction_matrix_pair_level))

        # if folder not exists, then create it,
        if not os.path.exists(os.path.join(self.output_dir, self.target_query_name)):
            os.makedirs(os.path.join(self.output_dir, self.target_query_name))

        with open(os.path.join(self.output_dir, self.target_query_name, "misclassified-{}.json".format(self.n_validation_videos)), 'w') as f:
            f.write(json.dumps(misclassified_results))

        with open(os.path.join(self.output_dir, self.target_query_name, "f1_score-{}.json".format(self.n_validation_videos)), 'w') as f:
            f.write(json.dumps(f1_score(true_labels_pair_level, prediction_matrix_pair_level)))

class InPerfectR1(InPerfect):
    def __init__(self, edge_thresh, center_thresh, n_validation_videos=1000):
        super().__init__(edge_thresh, center_thresh, n_validation_videos)
        self.target_query_name = "in_perfect-{}-{}-r1".format(edge_thresh, center_thresh)

    def pattern_matching(self, method):
        edge_fid_list = self.list_of_list[0]
        return self.generic_matching(method) and edge_fid_list[0][0] > 0

class InPerfectR2(InPerfect):
    def __init__(self, edge_thresh, center_thresh, n_validation_videos=1000):
        super().__init__(edge_thresh, center_thresh, n_validation_videos)
        self.target_query_name = "in_perfect-{}-{}-r2".format(edge_thresh, center_thresh)

    def pattern_matching(self, method):
        edge_fid_list = self.list_of_list[0]
        center_fid_list = self.list_of_list[1]
        if len(edge_fid_list) and len(center_fid_list):
            return self.generic_matching(method) and center_fid_list[0][0] > edge_fid_list[0][0]
        else:
            return self.generic_matching(method)

class InPerfectR4(InPerfect):
    def __init__(self, edge_thresh, center_thresh, n_validation_videos=1000):
        super().__init__(edge_thresh, center_thresh, n_validation_videos)
        self.target_query_name = "in_perfect-{}-{}-r4".format(edge_thresh, center_thresh)
        self.misclassified_method = [[0, 1]]

    def pattern_matching(self, method):
        if method == [[0, 1]]:
            return self.generic_matching(method) and self.list_of_list[0][0][0] > 120 and len(self.list_of_list[1]) == 0
        return self.generic_matching(method)

class InPerfectR5(InPerfect):
    def __init__(self, edge_thresh, center_thresh, n_validation_videos=1000):
        super().__init__(edge_thresh, center_thresh, n_validation_videos)
        self.target_query_name = "in_perfect-{}-{}-r5".format(edge_thresh, center_thresh)

    def pattern_matching(self, method):
        edge_fid_list = self.list_of_list[0]
        center_fid_list = self.list_of_list[1]
        if len(edge_fid_list) and len(center_fid_list):
            return self.generic_matching(method) and center_fid_list[0][0] > edge_fid_list[0][0] and edge_fid_list[0][0] > 2
        return self.generic_matching(method) and edge_fid_list[0][0] > 2

class OutPerfect(InOutPerfect):
    def __init__(self, edge_thresh, center_thresh, n_validation_videos=1000):
        super().__init__(edge_thresh, center_thresh, n_validation_videos)
        self.misclassified_method = [[1, 1], [0, 1]]
        self.target_query_name = "out_perfect-{}-{}".format(edge_thresh, center_thresh)

    def detect_collision(self):
        super().detect_collision("out")

class OutPerfectR1(OutPerfect):
    def __init__(self, edge_thresh, center_thresh, n_validation_videos=1000):
        super().__init__(edge_thresh, center_thresh, n_validation_videos)
        self.target_query_name = "out_perfect-{}-{}-r1".format(edge_thresh, center_thresh)

    def pattern_matching(self, method):
        edge_fid_list = self.list_of_list[0]
        center_fid_list = self.list_of_list[1]
        if len(edge_fid_list) and len(center_fid_list):
            return self.generic_matching(method) and edge_fid_list[-1][-1] < 125
        return self.generic_matching(method) and edge_fid_list[-1][-1] < 125

if __name__ == '__main__':
    # test_query = ObjectsCollision("disjunctive")
    test_query = ObjectsCollision("simple")
    # test_query = CubeCollidesCylinder("disjunctive")
    # test_query = RedCubeCollidesSphere("disjunctive")
    # test_query = ObjectsCollisionPerfect(1.05, 1.1)
    # test_query = InPerfectR5(20, 20)
    # test_query = OutPerfectR1(20, 20)
    # test_query = InTrivial(10, 10)
    test_query.detect_collision()
    # test_query.find_best_queries()