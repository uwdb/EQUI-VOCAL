import json
import random
import itertools
import shutil
import numpy as np
import os
from src.utils import rewrite_program_postgres, str_to_program_postgres, postgres_execute, postgres_execute_cache_sequence
import csv
from itertools import repeat
from concurrent.futures import ThreadPoolExecutor
import scipy.stats as stats
import time
import psycopg2 as psycopg
import multiprocessing
from lru import LRU
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse

# random.seed(1234)
# np.random.seed(10)

m = multiprocessing.Manager()
lock = m.Lock()
memo = [LRU(10000) for _ in range(72159)]

def generate_queries(n_queries, ratio_lower_bound, ratio_upper_bound, npred, depth, max_duration, nvars, predicate_list, attr_predicate_list, max_workers, dataset_name, nattr_pred, port):
    """
    Generate (n_queries) queries with the same complexity (npred, depth), removing those that don't have enough positive data (i.e., highly imbalanced)
    """
    queries = []
    while len(queries) < n_queries:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for res in executor.map(generate_one_query, repeat(npred, max_workers), repeat(depth, max_workers), repeat(max_duration, max_workers), repeat(nvars, max_workers), repeat(predicate_list, max_workers), repeat(attr_predicate_list, max_workers), repeat(ratio_lower_bound, max_workers), repeat(ratio_upper_bound, max_workers), repeat(dataset_name, max_workers), repeat(nattr_pred, max_workers), repeat(port, max_workers)):
                if res:
                    queries.append(res)
                    print("Generated {} queries".format(len(queries)))
    # write queries to csv file
    with open("/gscratch/balazinska/enhaoz/complex_event_video/inputs/{}/queries.csv".format(dataset_name), "w") as csvfile:
        writer = csv.writer(csvfile)
        # write header
        writer.writerow(["query", "npos", "nneg", "ratio"])
        writer.writerows(queries)


def generate_one_query(npred, depth, max_duration, nvars, predicate_list, attr_predicate_list, ratio_lower_bound, ratio_upper_bound, dataset_name, nattr_pred, port):
    """
    Generate one query with the specific complexity (npred, depth), using predicates from predicate_list.
    """
    def print_scene_graph(predicates):
        # Conj(Conj(p13, p12), p11)
        if len(predicates) == 1:
            if predicates[0]["parameters"]:
                theta = random.choice(predicates[0]["parameters"])
                vars_str = ", ".join(predicates[0]["args"])
                return "{}_{}({})".format(predicates[0]["name"], theta, vars_str)
            else:
                vars_str = ", ".join(predicates[0]["args"])
                return "{}({})".format(predicates[0]["name"], vars_str)
        else:
            if predicates[-1]["parameters"]:
                theta = random.choice(predicates[-1]["parameters"])
                vars_str = ", ".join(predicates[-1]["args"])
                return "Conjunction({}, {})".format(print_scene_graph(predicates[:-1]), "{}_{}({})".format(predicates[-1]["name"], theta, vars_str))
            else:
                vars_str = ", ".join(predicates[-1]["args"])
                return "Conjunction({}, {})".format(print_scene_graph(predicates[:-1]), "{}({})".format(predicates[-1]["name"], vars_str))

    assert(npred >= depth)
    assert(npred <= len(predicate_list) * depth)
    npred_per_scene_graph = [1] * depth
    nattr_pred_per_scene_graph = [0] * depth
    for _ in range(nattr_pred):
        nattr_pred_per_scene_graph[random.choice(range(depth))] += 1
    for _ in range(npred - depth):
        candidates = [i for i in range(depth) if npred_per_scene_graph[i] < len(predicate_list)]
        npred_per_scene_graph[random.choice(candidates)] += 1

    # duration_unit = 5
    # x = np.arange(1, max_duration // duration_unit + 2)
    # weights = x ** (-1.6)
    # weights /= weights.sum()
    # duration_values = [1]
    # for i in range(1, max_duration // duration_unit + 1):
    #     duration_values.append(i * duration_unit)
    # bounded_zipf = stats.rv_discrete(name='bounded_zipf', values=(duration_values, weights))
    # duration_per_scene_graph = bounded_zipf.rvs(size=depth)

    duration_unit = 5
    duration_values = [1]
    for i in range(1, max_duration // duration_unit + 1):
        duration_values.append(i * duration_unit)
    duration_per_scene_graph = [1 for _ in range(depth)]
    while sum(duration_per_scene_graph) == depth and max_duration > 1:
        duration_per_scene_graph = [random.choice(duration_values) for _ in range(depth)]

    scene_graphs = []
    for i in range(depth):
        sampled_predicates = random.sample(predicate_list, npred_per_scene_graph[i])
        for pred in sampled_predicates:
            pred["args"] = ["o_{}".format(i) for i in random.sample(list(range(nvars)), pred["nargs"])]
        if nattr_pred_per_scene_graph[i] > 0:
            sampled_attr_predicates = random.sample(attr_predicate_list, nattr_pred_per_scene_graph[i])
            for pred in sampled_attr_predicates:
                pred["args"] = ["o_{}".format(i) for i in random.sample(list(range(nvars)), pred["nargs"])]
            sampled_predicates += sampled_attr_predicates
        scene_graph_str = print_scene_graph(sampled_predicates)
        if duration_per_scene_graph[i] > 1:
            scene_graph_str = "Duration({}, {})".format(scene_graph_str, duration_per_scene_graph[i])
        scene_graphs.append(scene_graph_str)
    query = "; ".join(scene_graphs)
    program = str_to_program_postgres(query)
    query = rewrite_program_postgres(program)
    print(query)
    inputs_table_name = "Obj_clevrer"
    return prepare_data_given_target_query(query, ratio_lower_bound, ratio_upper_bound, dataset_name, inputs_table_name, port)


def prepare_data_given_target_query(program_str, ratio_lower_bound, ratio_upper_bound, dataset_name, inputs_table_name, port=5432, sampling_rate=None):
    """
    Given the target query (in string format), generate inputs.json and labels.json files containing
    inputs.json.

    ratio: minimum ratio of positive examples to negative examples
    """
    program = str_to_program_postgres(program_str)
    dsn = "dbname=myinner_db user=enhaoz host=localhost port={}".format(port)
    conn = psycopg.connect(dsn)
    if inputs_table_name == "Obj_clevrer":
        is_trajectory = False
        input_vids = 10000
    elif inputs_table_name == "Obj_trajectories":
        is_trajectory = True
        input_vids = 10080
    elif inputs_table_name == "Obj_shibuya":
        is_trajectory = False
        input_vids = 1801
    elif inputs_table_name == "Obj_warsaw":
        is_trajectory = True
        input_vids = 72159
    else:
        raise ValueError("Unknown inputs_table_name: {}".format(inputs_table_name))
    _start = time.time()
    result, new_memo = postgres_execute_cache_sequence(conn, program, memo, inputs_table_name, input_vids, is_trajectory=is_trajectory, sampling_rate=sampling_rate)
    print("Time to execute query: {}".format(time.time() - _start))

    lock.acquire()
    for i, memo_dict in enumerate(new_memo):
        for k, v in memo_dict.items():
            memo[i][k] = v
    lock.release()
    labels = []
    for i in range(input_vids):
        if i in result:
            labels.append(1)
        else:
            labels.append(0)

    print("Generated {} positive inputs and {} negative inputs".format(len(result), input_vids - len(result)))
    if len(result) / input_vids < ratio_lower_bound:
        print("Query {} doesn't have enough positive examples".format(program_str))
        return None
    if len(result) / input_vids > ratio_upper_bound:
        print("Query {} doesn't have enough negative examples".format(program_str))
        return None

    if not os.path.exists("/gscratch/balazinska/enhaoz/complex_event_video/inputs/{}".format(dataset_name)):
        os.makedirs("/gscratch/balazinska/enhaoz/complex_event_video/inputs/{}".format(dataset_name), exist_ok=True)
    with open("/gscratch/balazinska/enhaoz/complex_event_video/inputs/{}/{}_labels.json".format(dataset_name, program_str), 'w') as f:
        f.write(json.dumps(labels))
    return program_str, sum(labels), len(labels) - sum(labels), sum(labels) / (len(labels) - sum(labels))

def prepare_noisy_data(fn_error_rate, fp_error_rate, dataset_name):
    source_folder_name = os.path.join("/gscratch/balazinska/enhaoz/complex_event_video/inputs", dataset_name)
    target_folder_name = "/gscratch/balazinska/enhaoz/complex_event_video/inputs/{}-fn_error_rate_{}-fp_error_rate_{}".format(dataset_name, fn_error_rate, fp_error_rate)
    if not os.path.exists(target_folder_name):
        os.makedirs(target_folder_name, exist_ok=True)

    for filename in os.listdir(source_folder_name):
        fn_count = 0
        fp_count = 0
        if filename.endswith("_labels.json"):
            with open(os.path.join(source_folder_name, filename), 'r') as f:
                labels = json.load(f)
            # flip the label with probability error_rate
            for i in range(len(labels)):
                if labels[i] and random.random() < fn_error_rate:
                    labels[i] = 0
                    fn_count += 1
                elif not labels[i] and random.random() < fp_error_rate:
                    labels[i] = 1
                    fp_count += 1
            with open(os.path.join(target_folder_name, filename), 'w') as f:
                f.write(json.dumps(labels))
            print("fp_count: {}, fn_count: {}".format(fp_count, fn_count))
            # copy file
            # shutil.copy(os.path.join(source_folder_name, filename.replace("_labels", "_inputs")), os.path.join(target_folder_name, filename.replace("_labels", "_inputs")))

def construct_train_test(dir_name, n_train, n_test=None):
    for filename in os.listdir(dir_name):
        if filename.endswith("_labels.json"):
            query_str = filename[:-12]
            if not os.path.exists(os.path.join(dir_name, "test/{}_labels.json".format(query_str))):
                construct_train_test_per_query(dir_name, query_str, n_train, n_test)

def construct_train_test_per_query(dir_name, query_str, n_train, n_test):
    labels_filename = query_str + "_labels"
    inputs_filename = query_str + "_inputs"

    # read from json file
    with open(os.path.join(dir_name, "{}.json".format(labels_filename)), 'r') as f:
        labels = json.load(f)

    labels = np.asarray(labels, dtype=object)
    inputs = np.arange(len(labels))

    inputs_train, inputs_test, labels_train, labels_test = train_test_split(inputs, labels, train_size=n_train, random_state=42, stratify=labels)
    if n_test:
        inputs_test, _, labels_test, _ = train_test_split(inputs_test, labels_test, train_size=n_test, random_state=42, stratify=labels_test)
    # if folder doesn't exist, create it
    if not os.path.exists(os.path.join(dir_name, "train/")):
        os.makedirs(os.path.join(dir_name, "train/"))
    if not os.path.exists(os.path.join(dir_name, "test/")):
        os.makedirs(os.path.join(dir_name, "test/"))

    with open(os.path.join(dir_name, "train/{}.json".format(inputs_filename)), 'w') as f:
        json.dump(inputs_train.tolist(), f)
    with open(os.path.join(dir_name, "train/{}.json".format(labels_filename)), 'w') as f:
        json.dump(labels_train.tolist(), f)
    with open(os.path.join(dir_name, "test/{}.json".format(inputs_filename)), 'w') as f:
        json.dump(inputs_test.tolist(), f)
    with open(os.path.join(dir_name, "test/{}.json".format(labels_filename)), 'w') as f:
        json.dump(labels_test.tolist(), f)

    print("inputs_train", len(inputs_train))
    print("labels_train", len(labels_train), sum(labels_train))
    print("inputs_test", len(inputs_test))
    print("labels_test", len(labels_test), sum(labels_test))

def prepare_data_postgres(port):
    predicate_list = [
        {"name": "Near", "parameters": [1], "nargs": 2},
        {"name": "Far", "parameters": [3], "nargs": 2},
        {"name": "LeftOf", "parameters": None, "nargs": 2},
        {"name": "Behind", "parameters": None, "nargs": 2},
        {"name": "RightOf", "parameters": None, "nargs": 2},
        {"name": "FrontOf", "parameters": None, "nargs": 2},
        {"name": "RightQuadrant", "parameters": None, "nargs": 1},
        {"name": "LeftQuadrant", "parameters": None, "nargs": 1},
        {"name": "TopQuadrant", "parameters": None, "nargs": 1},
        {"name": "BottomQuadrant", "parameters": None, "nargs": 1}
        ]
    # attr_predicate_list = [{"name": "Gray", "parameters": None, "nargs": 1}, {"name": "Red", "parameters": None, "nargs": 1}, {"name": "Blue", "parameters": None, "nargs": 1}, {"name": "Green", "parameters": None, "nargs": 1}, {"name": "Brown", "parameters": None, "nargs": 1}, {"name": "Cyan", "parameters": None, "nargs": 1}, {"name": "Purple", "parameters": None, "nargs": 1}, {"name": "Yellow", "parameters": None, "nargs": 1}, {"name": "Cube", "parameters": None, "nargs": 1}, {"name": "Sphere", "parameters": None, "nargs": 1}, {"name": "Cylinder", "parameters": None, "nargs": 1}, {"name": "Metal", "parameters": None, "nargs": 1}, {"name": "Rubber", "parameters": None, "nargs": 1}]
    attr_predicate_list = [
        {"name": "Color", "parameters": ["gray", "red", "blue", "green", "brown", "cyan", "purple", "yellow"], "nargs": 1},
        {"name": "Shape", "parameters": ["cube", "sphere", "cylinder"], "nargs": 1},
        {"name": "Material", "parameters": ["metal", "rubber"], "nargs": 1}
    ]
    generate_queries(n_queries=40, ratio_lower_bound=0.05, ratio_upper_bound=0.1, npred=5, depth=3, max_duration=1, nvars=3, predicate_list=predicate_list, attr_predicate_list=attr_predicate_list, max_workers=4, dataset_name="synthetic_scene_graph_without_duration-npred_5-nattr_pred_2-40", nattr_pred=2, port=port)
    construct_train_test("/gscratch/balazinska/enhaoz/complex_event_video/inputs/synthetic_scene_graph_without_duration-npred_5-nattr_pred_2-40", n_train=500)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5432)
    args = ap.parse_args()
    port = args.port

    # prepare_data_trajectories(port, sampling_rate)
    prepare_data_postgres(port)
    # prepare_data_given_target_query(query, 0, 1, "", "Obj_clevrer", None)
