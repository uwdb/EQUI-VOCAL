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

# num_workers = mp.cpu_count()
num_workers = 4
print("# of workers: ", num_workers)
segment_length = 128
n_chunks = int(128 / segment_length)
random.seed(1234)

def collision_matching_by_far_near_far(near_fid_list, far_fid_list):
    # generic_matching([near_fid_list, far_fid_list], [1, 0, 1])
    for near_segment in near_fid_list:
        for i in range(len(far_fid_list) - 1):
            for j in range(i+1, len(far_fid_list)):
                far_segment1 = far_fid_list[i]
                far_segment2 = far_fid_list[j]
                if far_segment1[1] < near_segment[0] and near_segment[1] < far_segment2[0]:
                    return True
    return False


def collision_matching_by_far_near(near_fid_list, far_fid_list):
    # generic_matching([near_fid_list, far_fid_list], [1, 0])
    for near_segment in near_fid_list:
        for far_segment in far_fid_list:
            if far_segment[1] < near_segment[0]:
                return True
    return False


def collision_matching_by_far_near_or_near_far(near_fid_list, far_fid_list):
    for near_segment in near_fid_list:
        for far_segment in far_fid_list:
            if far_segment[1] < near_segment[0] or near_segment[1] < far_segment[0]:
                return True
    return False


def collision_matching_by_near(near_fid_list, far_fid_list):
    # generic_matching([near_fid_list, far_fid_list], [0])
    if len(near_fid_list) > 0:
        return True
    return False

def generic_matching(list_of_list, sequence, left_list=None):
    if left_list is None:
        left_list = list_of_list[sequence[0]]
    if len(sequence) == 1:
        return len(left_list) > 0
    right_list = list_of_list[sequence[1]]
    join_list = []
    for left_segment in left_list:
        for right_segment in right_list:
            if left_segment[1] < right_segment[0]:
                join_list.append([left_segment[0], right_segment[1]])
    return generic_matching(list_of_list, sequence[1:], join_list)


def detect_collision(method_name, method, duration, objects_near, objects_far):
    with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_bboxes_evaluation_1000.json", 'r') as f:
        maskrcnn_bboxes_evaluation = json.loads(f.read())
    with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_video_list_evaluation_1000.json", 'r') as f:
        video_list_evaluation = json.loads(f.read())
    collision_list = []
    args= []
    for video_basename, _, _ in video_list_evaluation:
        args.append(video_basename)
    with Pool(processes=num_workers) as pool:
        for i, local_collision_list in enumerate(pool.imap_unordered(partial(_detect_collision, maskrcnn_bboxes_evaluation=maskrcnn_bboxes_evaluation, method=method, duration=duration, objects_near=objects_near, objects_far=objects_far), args)):
            print("\rdone {0:%}".format(i/len(args)))
            collision_list += local_collision_list

    with open("/gscratch/balazinska/enhaoz/complex_event_video/src/clevrer/{}.json".format(method_name), 'w') as f:
        f.write(json.dumps(collision_list))


def _detect_collision(video_basename, maskrcnn_bboxes_evaluation, method, duration, objects_near, objects_far):
    local_collision_list = []
    # Construct object list
    obj_set = set()
    for frame_id in range(128):
        res_per_frame = maskrcnn_bboxes_evaluation["{}_{}".format(video_basename, frame_id)]
        # [81.0, 109.0, 128.0, 164.0, "metal", "blue", "cylinder"]
        for obj in res_per_frame:
            obj_set.add("{}_{}_{}".format(obj[4], obj[5], obj[6]))
    # Start querying
    for obj1_str_id, obj2_str_id in itertools.combinations(obj_set, 2):
        obj1_id = obj1_str_id.split("_")
        obj2_id = obj2_str_id.split("_")
        for chunk_id in range(n_chunks):
            near_fid_list = []
            far_fid_list = []
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
                    if objects_near.predict([spatial_feature])[0] == 1:
                        near_fid_list.append(frame_id)
                    if objects_far.predict([spatial_feature])[0] == 1:
                        far_fid_list.append(frame_id)

            # If there are collisions, then add the pair to the list
            groups = groupby(near_fid_list, key=lambda item, c=count():item-next(c))
            near_fid_list = []
            for k, g in groups:
                g_list = list(g)
                if len(g_list) >= duration:
                    near_fid_list.append([g_list[0], g_list[-1]])
            groups = groupby(far_fid_list, key=lambda item, c=count():item-next(c))
            far_fid_list = []
            for k, g in groups:
                g_list = list(g)
                if len(g_list) >= duration:
                    far_fid_list.append([g_list[0], g_list[-1]])
            # if collision_matching_by_far_near_far(near_fid_list, far_fid_list):
            # if collision_matching_by_far_near(near_fid_list, far_fid_list):
            if generic_matching([near_fid_list, far_fid_list], method):
            # if (method == "using_far_near_far" and generic_matching([near_fid_list, far_fid_list], [1, 0, 1])) or (method == "using_far_near" and generic_matching([near_fid_list, far_fid_list], [1, 0])) or (method == "using_near" and generic_matching([near_fid_list, far_fid_list], [0])) or (method == "using_far_near_or_near_far" and collision_matching_by_far_near_or_near_far(near_fid_list, far_fid_list)):
                print(video_basename, obj1_str_id, obj2_str_id)
                local_collision_list.append([video_basename, chunk_id, obj1_str_id, obj2_str_id, near_fid_list, far_fid_list])
    return local_collision_list


def compute_f1_score(method_name, misclassified_report):
    tp, fp, fn, tn = 0, 0, 0, 0
    with open("/gscratch/balazinska/enhaoz/complex_event_video/src/clevrer/{}.json".format(method_name), 'r') as f:
        collision_list = json.loads(f.read())
    collision_list = [item[:4] for item in collision_list]
    with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_video_list_evaluation_1000.json", 'r') as f:
        video_list_evaluation = json.loads(f.read())
    args = []
    for video_basename, _, _ in video_list_evaluation:
        for chunk_id in range(n_chunks):
            args.append((video_basename, chunk_id))
    misclassified_filename = "/gscratch/balazinska/enhaoz/complex_event_video/src/clevrer/misclassified_report-{}.txt".format(method_name)
    with Pool(processes=num_workers) as pool:
        for i, ret in enumerate(pool.imap_unordered(partial(_compute_f1_score, misclassified_report=misclassified_report, collision_list=collision_list, misclassified_filename=misclassified_filename), args)):
            print("\rdone {0:%}".format(i/len(args)))
            tp += ret[0]
            fp += ret[1]
            fn += ret[2]
            tn += ret[3]
    # tp: 1118; fp: 303; fn: 1339; tn: 7383
    # F1 score: 0.5765858690046416
    # append the score to the file
    with open("/gscratch/balazinska/enhaoz/complex_event_video/src/clevrer/f1_scores-topk.txt", 'a') as f:
        f.write("method: {}; tp: {}; fp: {}; fn: {}; tn: {}; f1_score: {}\n".format(method_name, tp, fp, fn, tn, 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn != 0 else 0))
    # Random guessing: F1 score:  0.3279734219269103


def _compute_f1_score(args, misclassified_report, collision_list, misclassified_filename):
    video_basename, chunk_id = args
    tp, fp, fn, tn = 0, 0, 0, 0
    pred_positives = []
    for row in collision_list:
        if row[0] == video_basename and row[1] == chunk_id:
            pred_positives.append([row[2], row[3]])
    file = "/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_proposals/sim_{}.json".format(video_basename)
    # Read in bbox info
    with open(file, 'r') as f:
        data = json.load(f)
    collisions = data["ground_truth"]["collisions"]
    objects = data["ground_truth"]["objects"]
    positive_pairs = []
    for collision in collisions:
        if collision["frame"] >= chunk_id * segment_length and collision["frame"] < (chunk_id + 1) * segment_length:
            positive_pairs.append(collision["object"])
    for obj1, obj2 in itertools.combinations(objects, 2):
        obj1_str_id = "{}_{}_{}".format(obj1["material"], obj1["color"], obj1["shape"])
        obj2_str_id = "{}_{}_{}".format(obj2["material"], obj2["color"], obj2["shape"])
        true_label = 0
        for pos_obj1_id, pos_obj2_id in positive_pairs:
            if (obj1["id"] == pos_obj1_id and obj2["id"] == pos_obj2_id) or (obj1["id"] == pos_obj2_id and obj2["id"] == pos_obj1_id):
                # Positive example
                true_label = 1
                if [obj1_str_id, obj2_str_id] in pred_positives or [obj2_str_id, obj1_str_id] in pred_positives:
                    tp += 1
                else:
                    fn += 1
                    if misclassified_report:
                        with open(misclassified_filename, 'a') as f:
                            f.write("[fn]: {} {} {} {}\n".format(video_basename, chunk_id, obj1, obj2))
                break
        if true_label == 0:
            # Negative example
            if [obj1_str_id, obj2_str_id] in pred_positives or [obj2_str_id, obj1_str_id] in pred_positives:
                fp += 1
                if misclassified_report:
                    with open(misclassified_filename, 'a') as f:
                        f.write("[fp]: {} {} {} {}\n".format(video_basename, chunk_id, obj1_str_id, obj2_str_id))
            else:
                tn += 1
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    return tp, fp, fn, tn


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    # ap.add_argument('--num_train', type=int)
    # ap.add_argument('--duration', type=int)
    ap.add_argument('--method', type=str)
    # ap.add_argument('--misclassified_report', type=bool, default=False)
    args = ap.parse_args()
    # print(args.num_train, args.duration, args.method, args.misclassified_report)
    # method_name = "test_collision-32_frames-{}-n{}-d{}".format(args.method, args.num_train, args.duration)
    # k = 5
    # methods = [[0], [1]]
    # current_methods = methods.copy()
    # for _ in range(k-1):
    #     new_methods = []
    #     for i in [0, 1]:
    #         for method in current_methods:
    #             new_methods.append(method + [i])
    #     current_methods = new_methods.copy()
    #     for method in new_methods:
    #         methods.append(method)
    # methods_str = []
    # for method in methods:
    #     method_str = '_'.join(str(x) for x in method)
    #     methods_str.append(method_str)
    method_str = args.method
    method = [int(x) for x in method_str.split('_')]
    duration = 1
    num_train = 200
    method_name = "enumeration/test_collision-{}-n{}-d{}".format(method_str, num_train, 1)
    objects_far = joblib.load("/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/clevrer_far/models/random_forest-{}-least_confidence-original.joblib".format(num_train))
    objects_near = joblib.load("/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/clevrer_near/models/random_forest-{}-least_confidence-original.joblib".format(num_train))
    detect_collision(method_name, method, duration, objects_near, objects_far)
    compute_f1_score(method_name, False)