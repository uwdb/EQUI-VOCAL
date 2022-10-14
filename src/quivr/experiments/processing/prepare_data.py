import json
from operator import neg
import random
import itertools
import shutil
import numpy as np
import quivr.dsl as dsl
import os
from quivr.utils import str_to_program, print_program
import csv
from itertools import repeat
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from threading import current_thread, get_ident, get_native_id
import scipy.stats as stats
from glob import glob
from torchvision.ops import masks_to_boxes
import pycocotools._mask as _mask
import torch

segment_length = 128
# random.seed(1234)
# np.random.seed(10)


def generate_queries(n_queries, ratio_lower_bound, ratio_upper_bound, npred, depth, max_duration, predicate_dict, max_workers, dataset_name):
    """
    Generate (n_queries) queries with the same complexity (npred, depth), removing those that don't have enough positive data (i.e., highly imbalanced)
    """
    queries = []
    while len(queries) < n_queries:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for res in executor.map(generate_one_query, repeat(npred, max_workers), repeat(depth, max_workers), repeat(max_duration, max_workers), repeat(predicate_dict, max_workers), repeat(ratio_lower_bound, max_workers), repeat(ratio_upper_bound, max_workers), repeat(dataset_name, max_workers)):
                if res:
                    queries.append(res)
                    print("Generated {} queries".format(len(queries)))
    # write queries to csv file
    with open("inputs/{}/queries.csv".format(dataset_name), "w") as csvfile:
        writer = csv.writer(csvfile)
        # write header
        writer.writerow(["query", "npos", "nneg", "ratio"])
        writer.writerows(queries)


def generate_one_query(npred, depth, max_duration, predicate_dict, ratio_lower_bound, ratio_upper_bound, dataset_name):
    """
    Generate one query with the specific complexity (npred, depth), using predicates from predicate_dict.
    """
    def print_scene_graph(predicate_list):
        # Conj(Conj(p13, p12), p11)
        if len(predicate_list) == 1:
            if predicate_dict[predicate_list[0]]:
                theta = random.choice(predicate_dict[predicate_list[0]])
                return predicate_list[0]().name + "_" + str(abs(theta))
            else:
                return predicate_list[0]().name
        else:
            if predicate_dict[predicate_list[-1]]:
                theta = random.choice(predicate_dict[predicate_list[-1]])
                return "Conjunction({}, {})".format(print_scene_graph(predicate_list[:-1]), predicate_list[-1]().name + "_" + str(abs(theta)))
            else:
                return "Conjunction({}, {})".format(print_scene_graph(predicate_list[:-1]), predicate_list[-1]().name)

    def print_query(scene_graphs):
        return "Sequencing({}, True*)".format(print_query_helper(scene_graphs))

    def print_query_helper(scene_graphs):
        # "Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Conjunction(Front, Left)), True*), Duration(Left, 2)), True*), Conjunction(Far_0.9, Left)), True*)"
        if len(scene_graphs) == 1:
            return "Sequencing(True*, {})".format(scene_graphs[0])
        else:
            return "Sequencing(Sequencing({}, True*), {})".format(print_query_helper(scene_graphs[:-1]), scene_graphs[-1])

    assert(npred >= depth)
    assert(npred <= len(predicate_dict) * depth)
    npred_per_scene_graph = [1] * depth
    for _ in range(npred - depth):
        candidates = [i for i in range(depth) if npred_per_scene_graph[i] < len(predicate_dict)]
        npred_per_scene_graph[random.choice(candidates)] += 1

    x = np.arange(1, max_duration + 1)
    weights = x ** (-1.6)
    weights /= weights.sum()
    bounded_zipf = stats.rv_discrete(name='bounded_zipf', values=(x, weights))
    duration_per_scene_graph = bounded_zipf.rvs(size=depth)

    scene_graphs = []
    for i in range(depth):
        predicate_list = random.sample(list(predicate_dict.keys()), npred_per_scene_graph[i])
        # Sort predicate_list in descending order (alphabetically)
        predicate_list.sort(key=lambda x: x(), reverse=True)
        scene_graph_str = print_scene_graph(predicate_list)
        if duration_per_scene_graph[i] > 1:
            scene_graph_str = "Duration({}, {})".format(scene_graph_str, duration_per_scene_graph[i])
        scene_graphs.append(scene_graph_str)
    query = print_query(scene_graphs)
    print(query)
    return prepare_trajectory_pairs_given_target_query(query, ratio_lower_bound, ratio_upper_bound, dataset_name)


def prepare_trajectory_pairs():
    with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_bboxes_evaluation_1000.json", 'r') as f:
        maskrcnn_bboxes_evaluation = json.loads(f.read())
    with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_video_list_evaluation_1000.json", 'r') as f:
        video_list_evaluation = json.loads(f.read())

    sample_inputs = []

    for video_i, (video_basename, _, _) in enumerate(video_list_evaluation):
        print(video_i, video_basename)
        # Construct object list
        obj_set = set()
        for frame_id in range(128):
            res_per_frame = maskrcnn_bboxes_evaluation["{}_{}".format(video_basename, frame_id)]
            # [81.0, 109.0, 128.0, 164.0, "metal", "blue", "cylinder"]
            for obj in res_per_frame:
                obj_set.add("{}_{}_{}".format(obj[4], obj[5], obj[6]))

        file = "/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_proposals/sim_{}.json".format(video_basename)
        # Read in bbox info
        with open(file, 'r') as f:
            data = json.load(f)
        objects = data["ground_truth"]["objects"]

        # Start querying
        for obj1_str_id, obj2_str_id in itertools.combinations(obj_set, 2):
            obj1_id = obj1_str_id.split("_")
            obj2_id = obj2_str_id.split("_")

            # Check if both objects are registered in the video (i.e. not a misclassification)
            obj1 = None
            obj2 = None
            for obj in objects:
                if obj["color"] == obj1_id[1] and obj["material"] == obj1_id[0] and obj["shape"] == obj1_id[2]:
                    obj1 = obj
                if obj["color"] == obj2_id[1] and obj["material"] == obj2_id[0] and obj["shape"] == obj2_id[2]:
                    obj2 = obj

            if obj1 and obj2:
                sample_input = [[], []]

                for frame_id in range(segment_length):
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
                        sample_input[0].append(obj1[:4])
                        sample_input[1].append(obj2[:4])

                sample_inputs.append(sample_input)
    print("Generated {} input pairs".format(len(sample_inputs)))

    with open(os.path.join("inputs/trajectory_pairs.json"), 'w') as f:
        f.write(json.dumps(sample_inputs))

def prepare_trajectory_pairs_given_target_query(program_str, ratio_lower_bound, ratio_upper_bound, dataset_name):
    """
    Given the target query (in string format), generate inputs.json and labels.json files containing
    inputs.json.

    ratio: minimum ratio of positive examples to negative examples
    """
    program = str_to_program(program_str)
    # read in trajectory pairs from file
    with open(os.path.join("inputs/trajectory_pairs.json"), 'r') as f:
        sample_inputs = json.loads(f.read())

    min_npos = int(len(sample_inputs) * ratio_lower_bound)
    max_npos = int(len(sample_inputs) * ratio_upper_bound)
    max_nneg = len(sample_inputs) - min_npos

    positive_inputs = []
    negative_inputs = []

    label = -1 # dummy label

    for i, sample_input in enumerate(sample_inputs):
        result, _ = program.execute(sample_input, label, {}, {})
        if result[0, len(sample_input[0])] > 0:
            positive_inputs.append(sample_input)
            # print("So far: {} positive inputs and {} negative inputs".format(len(positive_inputs), len(negative_inputs)))
        else:
            negative_inputs.append(sample_input)
        if len(negative_inputs) > max_nneg:
            print("Query {} doesn't have enough positive examples".format(program_str))
            return None
        if len(positive_inputs) > max_npos:
            print("Query {} doesn't have enough negative examples".format(program_str))
            return None
        if i % 100 == 0:
            print("Processed {} inputs: {} positives and {} negatives".format(i+1, len(positive_inputs), len(negative_inputs)))
    print("Generated {} positive inputs and {} negative inputs".format(len(positive_inputs), len(negative_inputs)))

    labels = [1] * len(positive_inputs) + [0] * len(negative_inputs)
    if not os.path.exists("inputs/{}".format(dataset_name)):
        os.makedirs("inputs/{}".format(dataset_name), exist_ok=True)
    # write positive and negative inputs to one file
    with open("inputs/{}/{}_inputs.json".format(dataset_name, program_str), 'w') as f:
        positive_inputs.extend(negative_inputs)
        f.write(json.dumps(positive_inputs))
    with open("inputs/{}/{}_labels.json".format(dataset_name, program_str), 'w') as f:
        f.write(json.dumps(labels))
    return program_str, sum(labels), len(labels) - sum(labels), sum(labels) / (len(labels) - sum(labels))

def prepare_noisy_data(fn_error_rate, fp_error_rate, dataset_name):
    source_folder_name = os.path.join("inputs", dataset_name)
    target_folder_name = "inputs/{}-fn_error_rate_{}-fp_error_rate_{}".format(dataset_name, fn_error_rate, fp_error_rate)
    if not os.path.exists(target_folder_name):
        os.makedirs(target_folder_name, exist_ok=True)

    for filename in os.listdir(source_folder_name):
        if filename.endswith("_labels.json"):
            with open(os.path.join(source_folder_name, filename), 'r') as f:
                labels = json.load(f)
            # flip the label with probability error_rate
            for i in range(len(labels)):
                if labels[i] and random.random() < fn_error_rate:
                    labels[i] = 0
                elif not labels[i] and random.random() < fp_error_rate:
                    labels[i] = 1
            with open(os.path.join(target_folder_name, filename), 'w') as f:
                f.write(json.dumps(labels))
            # copy file
            shutil.copy(os.path.join(source_folder_name, filename.replace("_labels", "_inputs")), os.path.join(target_folder_name, filename.replace("_labels", "_inputs")))


def prepare_postgres_data():
    """
    Store scene graphs of 10000 videos from Clevrer dataset into obj_clevrer.csv, which will be loaded into Postgres database.
    Table schema: (oid INT, vid INT, fid INT, shape varchar, color varchar, material varchar, x1 float, y1 float, x2 float, y2 float)
    Results are stored in obj_clevrer.csv
    """
    def decode(rleObjs):
        if type(rleObjs) == list:
            return _mask.decode(rleObjs)
        else:
            return _mask.decode([rleObjs])[:,:,0]
    # schema: oid, vid, fid, shape, color, material, x1, y1, x2, y2
    csv_data = []
    # iterate all files in the folder
    for file in glob("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_proposals/*.json"):
        encountered_objects = {}
        video_basename = os.path.basename(file).replace(".json", "").replace("sim_", "")
        if int(video_basename) >= 10000:
            continue
        # read in the json file
        with open(file, 'r') as f:
            data = json.loads(f.read())

        if len(data['frames']) != 128:
            print("Video {} has {} frames".format(video_basename, len(data['frames'])))
        # iterate all videos in the json file
        for frame in data['frames']:
            frame_id = frame['frame_index']
            objects = frame['objects']
            for i in range(len(objects)):
                # print(objects[i]['material'], objects[i]['color'], objects[i]['shape'])
                mask = decode(objects[i]['mask'])
                # O represents black, 1 represents white.
                box = masks_to_boxes(torch.from_numpy(mask[np.newaxis, :]))
                box = np.squeeze(box.numpy(), axis=0).tolist()
                obj_name = objects[i]['shape'] + "_" + objects[i]['color'] + "_" + objects[i]['material']
                if obj_name not in encountered_objects:
                    encountered_objects[obj_name] = len(encountered_objects)
                csv_data.append([encountered_objects[obj_name], int(video_basename), frame_id, objects[i]['shape'], objects[i]['color'], objects[i]['material'], box[0], box[1], box[2], box[3]])

    with open('postgres/obj_clevrer.csv', 'w') as f:
        writer = csv.writer(f)
        # write multiple rows
        writer.writerows(csv_data)

def prepare_postgres_data_test_trajectories():
    # schema: oid, vid, fid, shape, color, material, x1, y1, x2, y2
    csv_data = []
    with open("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/trajectory_pairs.json", 'r') as f:
        data = json.loads(f.read())

    for vid, pair in enumerate(data):
        t1 = pair[0]
        t2 = pair[1]
        assert(len(t1) == len(t2))
        for fid, (bbox1, bbox2) in enumerate(zip(t1, t2)):
            csv_data.append([0, vid, fid, "cube", "red", "metal", bbox1[0], bbox1[1], bbox1[2], bbox1[3]])
            csv_data.append([1, vid, fid, "cube", "red", "metal", bbox2[0], bbox2[1], bbox2[2], bbox2[3]])

    with open('postgres/obj_trajectories.csv', 'w') as f:
        writer = csv.writer(f)
        # write multiple rows
        writer.writerows(csv_data)

if __name__ == '__main__':
    pass
    # predicate_dict = {dsl.Near: [-1.05], dsl.Far: [0.9], dsl.LeftOf: None, dsl.BackOf: None, dsl.RightQuadrant: None, dsl.TopQuadrant: None}
    # generate_queries(n_queries=50, ratio_lower_bound=0.05, ratio_upper_bound=0.1, npred=5, depth=3, max_duration=5, predicate_dict=predicate_dict, max_workers=32, dataset_name="synthetic_rare")

    # prepare_noisy_data(fn_error_rate=0.1, fp_error_rate=0.01, dataset_name="synthetic")
    # prepare_trajectory_pairs_given_target_query("Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Conjunction(Front, Left)), True*), Duration(Left, 2)), True*), Conjunction(Far_0.9, Left)), True*)")
    prepare_postgres_data_test_trajectories()