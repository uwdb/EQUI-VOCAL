import json
import random
import itertools
import shutil
import numpy as np
import os
from quivr.utils import rewrite_program_postgres, str_to_program_postgres, postgres_execute
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

# random.seed(1234)
# np.random.seed(10)

m = multiprocessing.Manager()
lock = m.Lock()
memoize_all_inputs = [LRU(10000) for _ in range(10080)]

def generate_queries(n_queries, ratio_lower_bound, ratio_upper_bound, npred, depth, max_duration, nvars, predicate_list, attr_predicate_list, max_workers, dataset_name, nattr_pred):
    """
    Generate (n_queries) queries with the same complexity (npred, depth), removing those that don't have enough positive data (i.e., highly imbalanced)
    """
    queries = []
    while len(queries) < n_queries:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for res in executor.map(generate_one_query, repeat(npred, max_workers), repeat(depth, max_workers), repeat(max_duration, max_workers), repeat(nvars, max_workers), repeat(predicate_list, max_workers), repeat(attr_predicate_list, max_workers), repeat(ratio_lower_bound, max_workers), repeat(ratio_upper_bound, max_workers), repeat(dataset_name, max_workers), repeat(nattr_pred, max_workers)):
                if res:
                    queries.append(res)
                    print("Generated {} queries".format(len(queries)))
    # write queries to csv file
    with open("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/{}/queries.csv".format(dataset_name), "w") as csvfile:
        writer = csv.writer(csvfile)
        # write header
        writer.writerow(["query", "npos", "nneg", "ratio"])
        writer.writerows(queries)


def generate_one_query(npred, depth, max_duration, nvars, predicate_list, attr_predicate_list, ratio_lower_bound, ratio_upper_bound, dataset_name, nattr_pred):
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

    x = np.arange(1, max_duration + 1)
    weights = x ** (-1.6)
    weights /= weights.sum()
    bounded_zipf = stats.rv_discrete(name='bounded_zipf', values=(x, weights))
    duration_per_scene_graph = bounded_zipf.rvs(size=depth)

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
    return prepare_data_given_target_query(query, ratio_lower_bound, ratio_upper_bound, dataset_name, inputs_table_name)


def prepare_data_given_target_query(program_str, ratio_lower_bound, ratio_upper_bound, dataset_name, inputs_table_name, sampling_rate):
    """
    Given the target query (in string format), generate inputs.json and labels.json files containing
    inputs.json.

    ratio: minimum ratio of positive examples to negative examples
    """
    program = str_to_program_postgres(program_str)
    dsn = "dbname=myinner_db user=enhaoz host=localhost"

    if inputs_table_name == "Obj_clevrer":
        is_trajectory = False
        input_vids = 10000
    elif inputs_table_name == "Obj_trajectories":
        is_trajectory = True
        input_vids = 10080
    else:
        raise ValueError("Unknown inputs_table_name: {}".format(inputs_table_name))
    _start = time.time()
    result, new_memoize = postgres_execute(dsn, program, memoize_all_inputs, inputs_table_name, input_vids, is_trajectory=is_trajectory, sampling_rate=sampling_rate)
    print("Time to execute query: {}".format(time.time() - _start))

    lock.acquire()
    for i, v in enumerate(new_memoize):
        memoize_all_inputs[i].update(v)
    lock.release()
    labels = []
    for i in range(input_vids):
        if i in result:
            labels.append(1)
        else:
            labels.append(0)

    print("Generated {} positive inputs and {} negative inputs".format(len(result), input_vids - len(result)))

    if len(result) / (input_vids - len(result)) < ratio_lower_bound:
        print("Query {} doesn't have enough positive examples".format(program_str))
        return None
    if len(result) / (input_vids - len(result)) > ratio_upper_bound:
        print("Query {} doesn't have enough negative examples".format(program_str))
        return None

    if not os.path.exists("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/{}".format(dataset_name)):
        os.makedirs("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/{}".format(dataset_name), exist_ok=True)
    with open("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/{}/{}_labels.json".format(dataset_name, program_str), 'w') as f:
        f.write(json.dumps(labels))
    return program_str, sum(labels), len(labels) - sum(labels), sum(labels) / (len(labels) - sum(labels))

def prepare_noisy_data(fn_error_rate, fp_error_rate, dataset_name):
    source_folder_name = os.path.join("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs", dataset_name)
    target_folder_name = "/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/{}-fn_error_rate_{}-fp_error_rate_{}".format(dataset_name, fn_error_rate, fp_error_rate)
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

def prepare_data_postgres():
    predicate_list = [{"name": "Near", "parameters": [1.05], "nargs": 2}, {"name": "Far", "parameters": [0.9], "nargs": 2}, {"name": "LeftOf", "parameters": None, "nargs": 2}, {"name": "Behind", "parameters": None, "nargs": 2}, {"name": "RightOf", "parameters": None, "nargs": 2}, {"name": "FrontOf", "parameters": None, "nargs": 2}, {"name": "RightQuadrant", "parameters": None, "nargs": 1}, {"name": "LeftQuadrant", "parameters": None, "nargs": 1}, {"name": "TopQuadrant", "parameters": None, "nargs": 1}, {"name": "BottomQuadrant", "parameters": None, "nargs": 1}]
    attr_predicate_list = [{"name": "Gray", "parameters": None, "nargs": 1}, {"name": "Red", "parameters": None, "nargs": 1}, {"name": "Blue", "parameters": None, "nargs": 1}, {"name": "Green", "parameters": None, "nargs": 1}, {"name": "Brown", "parameters": None, "nargs": 1}, {"name": "Cyan", "parameters": None, "nargs": 1}, {"name": "Purple", "parameters": None, "nargs": 1}, {"name": "Yellow", "parameters": None, "nargs": 1}, {"name": "Cube", "parameters": None, "nargs": 1}, {"name": "Sphere", "parameters": None, "nargs": 1}, {"name": "Cylinder", "parameters": None, "nargs": 1}, {"name": "Metal", "parameters": None, "nargs": 1}, {"name": "Rubber", "parameters": None, "nargs": 1}]
    with psycopg.connect("dbname=myinner_db user=enhaoz host=localhost") as conn:
        with conn.cursor() as cur:
            # Create predicate functions (if not exists)
            for predicate in itertools.chain(predicate_list, attr_predicate_list):
                args = ", ".join(["text, text, text, double precision, double precision, double precision, double precision"] * predicate["nargs"])
                if predicate["parameters"]:
                    args = "double precision, " + args
                cur.execute("CREATE OR REPLACE FUNCTION {name}({args}) RETURNS boolean AS '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', '{name}' LANGUAGE C STRICT;".format(name=predicate["name"], args=args))
            conn.commit()
    generate_queries(n_queries=50, ratio_lower_bound=0.05, ratio_upper_bound=0.1, npred=5, depth=3, max_duration=5, nvars=3, predicate_list=predicate_list, attr_predicate_list=attr_predicate_list, max_workers=2, dataset_name="synthetic_scene_graph_rare", nattr_pred=2)

if __name__ == '__main__':
    # construct_train_test("inputs/synthetic_scene_graph_rare", 300)
    # prepare_postgres_data_test_trajectories()
    prepare_data_postgres()
