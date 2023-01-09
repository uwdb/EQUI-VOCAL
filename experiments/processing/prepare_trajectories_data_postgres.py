import json
import random
import itertools
import shutil
import numpy as np
import os
from utils import rewrite_program_postgres, str_to_program_postgres, postgres_execute, postgres_execute_cache_sequence
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
memoize_scene_graph = [LRU(10000) for _ in range(10080)]
memoize_sequence = [LRU(10000) for _ in range(10080)]

def generate_queries(n_queries, ratio_lower_bound, ratio_upper_bound, npred, depth, max_duration, predicate_list, max_workers, dataset_name, port, sampling_rate):
    """
    Generate (n_queries) queries with the same complexity (npred, depth), removing those that don't have enough positive data (i.e., highly imbalanced)
    """
    queries = []
    while len(queries) < n_queries:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for res in executor.map(generate_one_query, repeat(npred, max_workers), repeat(depth, max_workers), repeat(max_duration, max_workers), repeat(predicate_list, max_workers), repeat(ratio_lower_bound, max_workers), repeat(ratio_upper_bound, max_workers), repeat(dataset_name, max_workers), repeat(port, max_workers), repeat(sampling_rate, max_workers)):
                if res:
                    queries.append(res)
                    print("Generated {} queries".format(len(queries)))
    # write queries to csv file
    with open("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/{}/queries.csv".format(dataset_name), "w") as csvfile:
        writer = csv.writer(csvfile)
        # write header
        writer.writerow(["query", "npos", "nneg", "ratio"])
        writer.writerows(queries)


def generate_one_query(npred, depth, max_duration, predicate_list, ratio_lower_bound, ratio_upper_bound, dataset_name, port, sampling_rate):
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
    for _ in range(npred - depth):
        candidates = [i for i in range(depth) if npred_per_scene_graph[i] < len(predicate_list)]
        npred_per_scene_graph[random.choice(candidates)] += 1

    x = np.arange(1, max_duration + 1)
    weights = x ** (-1.6)
    weights /= weights.sum()
    bounded_zipf = stats.rv_discrete(name='bounded_zipf', values=(x, weights))
    duration_per_scene_graph = bounded_zipf.rvs(size=depth)

    scene_graphs = []
    for i in range(depth):
        sampled_predicates = random.sample(predicate_list, npred_per_scene_graph[i])
        for pred in sampled_predicates:
            if pred["nargs"] == 2:
                pred["args"] = ["o0", "o1"]
            elif pred["nargs"] == 1:
                pred["args"] = ["o0"]
            else:
                raise ValueError("Invalid number of arguments")
        scene_graph_str = print_scene_graph(sampled_predicates)
        if duration_per_scene_graph[i] > 1:
            scene_graph_str = "Duration({}, {})".format(scene_graph_str, duration_per_scene_graph[i])
        scene_graphs.append(scene_graph_str)
    query = "; ".join(scene_graphs)
    program = str_to_program_postgres(query)
    query = rewrite_program_postgres(program)
    print(query)
    return prepare_data_given_target_query(query, ratio_lower_bound, ratio_upper_bound, dataset_name, port, sampling_rate)

def prepare_data_given_target_query(program_str, ratio_lower_bound, ratio_upper_bound, dataset_name, port, sampling_rate=None):
    """
    Given the target query (in string format), generate inputs.json and labels.json files containing
    inputs.json.

    ratio: minimum ratio of positive examples to negative examples
    """
    program = str_to_program_postgres(program_str)
    dsn = "dbname=myinner_db user=enhaoz host=localhost port={}".format(port)

    inputs_table_name = "Obj_trajectories"
    input_vids = 10080
    _start = time.time()
    outputs, new_memoize_scene_graph, new_memoize_sequence = postgres_execute_cache_sequence(dsn, program, memoize_scene_graph, memoize_sequence, inputs_table_name, input_vids, is_trajectory=True, sampling_rate=sampling_rate)

    print("Time to execute query: {}".format(time.time() - _start))

    lock.acquire()
    for i, memo_dict in enumerate(new_memoize_scene_graph):
        for k, v in memo_dict.items():
            memoize_scene_graph[i][k] = v
    for i, memo_dict in enumerate(new_memoize_sequence):
        for k, v in memo_dict.items():
            memoize_sequence[i][k] = v
    lock.release()
    labels = []
    for i in range(input_vids):
        if i in outputs:
            labels.append(1)
        else:
            labels.append(0)

    print("Generated {} positive inputs and {} negative inputs".format(len(outputs), input_vids - len(outputs)))

    if len(outputs) / input_vids < ratio_lower_bound:
        print("Query {} doesn't have enough positive examples".format(program_str))
        return None
    if len(outputs) / input_vids > ratio_upper_bound:
        print("Query {} doesn't have enough negative examples".format(program_str))
        return None

    if not os.path.exists("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/{}".format(dataset_name)):
        os.makedirs("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/{}".format(dataset_name), exist_ok=True)
    with open("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/{}/{}_labels.json".format(dataset_name, program_str), 'w') as f:
        f.write(json.dumps(labels))
    return program_str, sum(labels), len(labels) - sum(labels), sum(labels) / len(labels)

def prepare_noisy_data(fn_error_rate, fp_error_rate, dataset_name):
    source_folder_name = os.path.join("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs", dataset_name)
    target_folder_name = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/{}-fn_error_rate_{}-fp_error_rate_{}".format(dataset_name, fn_error_rate, fp_error_rate)
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

def prepare_data_trajectories(port, sampling_rate):
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
    max_duration = 1
    generate_queries(n_queries=50, ratio_lower_bound=0.05, ratio_upper_bound=0.1, npred=5, depth=3, max_duration=max_duration, predicate_list=predicate_list, max_workers=8, dataset_name="synthetic_trajectories_rare-max_d_{}-sampling_rate_{}".format(max_duration, sampling_rate), port=port, sampling_rate=sampling_rate)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5432)
    args = ap.parse_args()
    port = args.port

    sampling_rate = 4

    prepare_data_trajectories(port, sampling_rate)

