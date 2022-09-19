import csv
import os
from quivr.methods.exhaustive_search import ExhaustiveSearch
from quivr.utils import print_program, rewrite_program_postgres, str_to_program
from quivr.methods.quivr_exact import QUIVR
from quivr.methods.vocal import VOCAL
from quivr.methods.quivr_soft import QUIVRSoft
from quivr.methods.random import Random
from quivr.methods.vocal_postgres import VOCALPostgres
import json
import random
import math
import numpy as np
from sklearn.metrics import f1_score
import argparse
import sys
import quivr.dsl as dsl
import pandas
import time
from sklearn.model_selection import train_test_split

# random.seed(10)
random.seed(time.time())


def test_quivr_exact(dataset_name, n_labeled, npred, depth, max_duration, multithread, query_str, predicate_dict):
    if query_str == "collision":
        # read from json file
        with open("inputs/collision_inputs_train.json", 'r') as f:
            inputs = json.load(f)
        with open("inputs/collision_labels_train.json", 'r') as f:
            labels = json.load(f)
        inputs = np.asarray(inputs, dtype=object)
        labels = np.asarray(labels, dtype=object)
    else:
        with open("inputs/{}/train/{}_inputs.json".format(dataset_name, query_str), 'r') as f:
            inputs = json.load(f)
        with open("inputs/{}/train/{}_labels.json".format(dataset_name, query_str), 'r') as f:
            labels = json.load(f)
        inputs = np.asarray(inputs, dtype=object)
        labels = np.asarray(labels, dtype=object)

    inputs, _, labels, _ = train_test_split(inputs, labels, train_size=n_labeled, stratify=labels)
    print("labels", sum(labels), len(labels))

    algorithm = QUIVR(inputs, labels, predicate_dict, max_npred=npred, max_depth=depth, max_duration=max_duration, multithread=multithread)
    answers, total_time = algorithm.run()

    answers = [[print_program(query), 1.0] for query in answers]
    return answers, total_time

def test_algorithm(method, dataset_name, n_init_pos, n_init_neg, npred, depth, max_duration, beam_width, k, samples_per_iter, budget, multithread, query_str, predicate_dict, strategy, max_vars):
    if query_str == "collision":
        # read from json file
        with open("inputs/collision_inputs_train.json", 'r') as f:
            inputs = json.load(f)
        with open("inputs/collision_labels_train.json", 'r') as f:
            labels = json.load(f)
        inputs = np.asarray(inputs, dtype=object)
        labels = np.asarray(labels, dtype=object)
    else:
        with open("inputs/{}/train/{}_inputs.json".format(dataset_name, query_str), 'r') as f:
            inputs = json.load(f)
        with open("inputs/{}/train/{}_labels.json".format(dataset_name, query_str), 'r') as f:
            labels = json.load(f)
        inputs = np.asarray(inputs, dtype=object)
        labels = np.asarray(labels, dtype=object)
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        print("pos_idx", len(pos_idx), pos_idx, "neg_idx", len(neg_idx), neg_idx)

    init_labeled_index = random.sample(pos_idx.tolist(), n_init_pos) + random.sample(neg_idx.tolist(), n_init_neg)
    print(init_labeled_index)
    if method == "vocal":
        algorithm = VOCAL(inputs, labels, predicate_dict, max_npred=npred, max_depth=depth, max_duration=max_duration, beam_width=beam_width, k=k, samples_per_iter=samples_per_iter, budget=budget, multithread=multithread, strategy=strategy)
    elif method == "quivr_soft":
        algorithm = QUIVRSoft(inputs, labels, predicate_dict, max_npred=npred, max_depth=depth, max_duration=max_duration, beam_width=beam_width, k=k, samples_per_iter=samples_per_iter, budget=budget, multithread=multithread, strategy=strategy)
    elif method == "random":
        algorithm = Random(inputs, labels, predicate_dict, max_npred=npred, max_depth=depth, max_duration=max_duration, beam_width=beam_width, k=k, samples_per_iter=samples_per_iter, budget=budget, multithread=multithread)
    elif method == "vocal_postgres":
        with open("/mmfs1/gscratch/balazinska/enhaoz/postgres_server.info", 'r') as f:
            host = f.readlines()[0].strip().split(" ")[0]
        algorithm = VOCALPostgres(inputs, labels, predicate_dict, max_npred=npred, max_depth=depth, max_duration=max_duration, beam_width=beam_width, k=k, samples_per_iter=samples_per_iter, budget=budget, multithread=multithread, strategy=strategy, max_vars=max_vars, host=host)

    answers, total_time = algorithm.run(init_labeled_index)
    if method == "vocal_postgres":
        answers = [[rewrite_program_postgres(query_graph.program), score] for query_graph, score in answers]
    elif method in ["vocal", "quivr_soft", "random"]:
        answers = [[print_program(query_graph.program), score] for query_graph, score in answers]
    return answers, total_time


def test_exhaustive(n_labeled_pos, n_labeled_neg, npred, depth, max_duration, multithread, predicate_dict):
    # read from json file
    with open("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/collision_inputs_test.json", 'r') as f:
        inputs = json.load(f)
    with open("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/collision_labels_test.json", 'r') as f:
        labels = json.load(f)

    inputs = np.asarray(inputs, dtype=object)
    labels = np.asarray(labels, dtype=object)

    n_pos = sum(labels)
    sampled_labeled_index = random.sample(list(range(n_pos)), n_labeled_pos) + random.sample(list(range(n_pos, len(labels))), n_labeled_neg)
    inputs = inputs[sampled_labeled_index]
    labels = labels[sampled_labeled_index]

    algorithm = ExhaustiveSearch(inputs, labels, predicate_dict, max_npred=npred, max_depth=depth, k=1000, max_duration=max_duration, multithread=multithread)
    algorithm.run()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', type=str)
    ap.add_argument('--n_labeled_pos', type=int, default=50)
    ap.add_argument('--n_labeled_neg', type=int, default=250)
    ap.add_argument('--n_init_pos', type=int, default=10)
    ap.add_argument('--n_init_neg', type=int, default=50)
    ap.add_argument('--dataset_name', type=str)
    ap.add_argument('--npred', type=int, default=5)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--max_duration', type=int, default=2)
    ap.add_argument('--beam_width', type=int, default=32)
    ap.add_argument('--k', type=int, default=100)
    ap.add_argument('--samples_per_iter', type=int, default=1)
    ap.add_argument('--budget', type=int, default=100)
    ap.add_argument('--multithread', type=int, default=1)
    ap.add_argument('--strategy', type=str, default="sampling")
    ap.add_argument('--max_vars', type=int, default=2)
    ap.add_argument('--query_str', type=str, default="collision")
    ap.add_argument('--run_id', type=int)
    ap.add_argument('--output_to_file', action="store_true")

    args = ap.parse_args()
    method_str = args.method
    n_labeled_pos = args.n_labeled_pos
    n_labeled_neg = args.n_labeled_neg
    n_init_pos = args.n_init_pos
    n_init_neg = args.n_init_neg
    dataset_name = args.dataset_name
    npred = args.npred
    depth = args.depth
    max_duration = args.max_duration
    beam_width = args.beam_width
    k = args.k
    samples_per_iter = args.samples_per_iter
    budget = args.budget
    multithread = args.multithread
    strategy = args.strategy
    max_vars = args.max_vars
    query_str = args.query_str
    run_id = args.run_id
    output_to_file = args.output_to_file

    # samples_per_iter should be >= (budget - n_init_pos - n_init_neg) / (npred + max_duration * depth), to ensure the search algorithm can reach to the leaf node.

    if method_str == "vocal":
        method_name = "{}-{}-unrestricted_v4".format(method_str, strategy)
        config_name = "nip_{}-nin_{}-npred_{}-depth_{}-max_d_{}-bw_{}-k_{}-per_iter_{}-budget_{}-thread_{}".format(n_init_pos, n_init_neg, npred, depth, max_duration, beam_width, k, samples_per_iter, budget, multithread)
    elif method_str == "quivr":
        method_name = method_str
        config_name = "nlabeled_{}-npred_{}-depth_{}-max_d_{}-thread_{}".format(budget, npred, depth, max_duration, multithread)
    elif method_str == "quivr_soft":
        method_name = "{}-{}".format(method_str, strategy)
        config_name = "nip_{}-nin_{}-npred_{}-depth_{}-max_d_{}-bw_{}-k_{}-per_iter_{}-budget_{}-thread_{}".format(n_init_pos, n_init_neg, npred, depth, max_duration, beam_width, k, samples_per_iter, budget, multithread)
    elif method_str == "random":
        method_name = "{}-unrestricted_v4".format(method_str)
        config_name = "nip_{}-nin_{}-npred_{}-depth_{}-max_d_{}-bw_{}-k_{}-per_iter_{}-budget_{}-thread_{}".format(n_init_pos, n_init_neg, npred, depth, max_duration, beam_width, k, samples_per_iter, budget, multithread)
    elif method_str == "vocal_postgres":
        method_name = "{}-{}-unrestricted_v4".format(method_str, strategy)
        config_name = "nip_{}-nin_{}-npred_{}-depth_{}-max_d_{}-nvars_{}-bw_{}-k_{}-per_iter_{}-budget_{}-thread_{}".format(n_init_pos, n_init_neg, npred, depth, max_duration, max_vars, beam_width, k, samples_per_iter, budget, multithread)

    if dataset_name == "collision":
        predicate_dict = {dsl.Near: [-1.05], dsl.Far: [1.1], dsl.LeftOf: None, dsl.RightOf: None}
    elif dataset_name.startswith("synthetic-") or dataset_name == "synthetic":
        predicate_dict = {dsl.Near: [-1.05], dsl.Far: [0.9], dsl.LeftOf: None, dsl.RightOf: None, dsl.BackOf: None, dsl.FrontOf: None}
    elif dataset_name.startswith("synthetic_rare"):
        if method_str == "vocal_postgres":
            predicate_dict = [{"name": "Near", "parameters": [1.05], "nargs": 2}, {"name": "Far", "parameters": [0.9], "nargs": 2}, {"name": "LeftOf", "parameters": None, "nargs": 2}, {"name": "Behind", "parameters": None, "nargs": 2}, {"name": "RightQuadrant", "parameters": None, "nargs": 1}, {"name": "TopQuadrant", "parameters": None, "nargs": 1}]
        else:
            predicate_dict = {dsl.Near: [-1.05], dsl.Far: [0.9], dsl.LeftOf: None, dsl.BackOf: None, dsl.RightQuadrant: None, dsl.TopQuadrant: None}
    print("predicate_dict", predicate_dict)

    log_name = "{}-{}".format(query_str, run_id)
    # if dir not exist, create it
    if output_to_file:
        if not os.path.exists("outputs/{}/{}/{}/verbose".format(method_name, dataset_name, config_name)):
            os.makedirs("outputs/{}/{}/{}/verbose".format(method_name, dataset_name, config_name), exist_ok=True)
        f = open("outputs/{}/{}/{}/verbose/{}.log".format(method_name, dataset_name, config_name, log_name), 'w')
        sys.stdout = f

    print(args)

    if method_str == 'quivr':
        answers, total_time = test_quivr_exact(dataset_name, budget, npred, depth, max_duration, multithread, query_str, predicate_dict)
    elif method_str == "exhaustive":
        test_exhaustive(n_labeled_pos, n_labeled_neg, npred, depth, max_duration, multithread, predicate_dict)
    elif method_str in ['vocal', 'quivr_soft', 'random', 'vocal_postgres']:
        answers, total_time = test_algorithm(method_str, dataset_name, n_init_pos, n_init_neg, npred, depth, max_duration, beam_width, k, samples_per_iter, budget, multithread, query_str, predicate_dict, strategy, max_vars)

    if output_to_file:
        with open("outputs/{}/{}/{}/{}.log".format(method_name, dataset_name, config_name, log_name), 'w') as f:
            for query_str, score in answers:
                f.write("{} {}\n".format(query_str, score))
            f.write("Total Time: {}\n".format(total_time))

        f.close()
        sys.stdout = sys.__stdout__
