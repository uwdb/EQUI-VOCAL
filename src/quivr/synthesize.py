import csv
import os
from quivr.methods.exhaustive_search import ExhaustiveSearch
from quivr.utils import print_program, str_to_program
from quivr.methods.quivr_exact import QUIVR
from quivr.methods.vocal import VOCAL
from quivr.methods.quivr_soft import QUIVRSoft
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

# random.seed(10)
random.seed(time.time())

def test_quivr_exact(n_labeled_pos, n_labeled_neg, npred, depth, max_duration, k, budget, multithread, query_str):
    if query_str == "collision":
        # read from json file
        with open("inputs/collision_inputs_train.json", 'r') as f:
            inputs = json.load(f)
        with open("inputs/collision_labels_train.json", 'r') as f:
            labels = json.load(f)
        inputs = np.asarray(inputs, dtype=object)
        labels = np.asarray(labels, dtype=object)
        predicate_dict = {dsl.Near: [-1.05], dsl.Far: [1.1], dsl.Left: None, dsl.Right: None}
    else:
        with open("inputs/synthetic/train/{}_inputs.json".format(query_str), 'r') as f:
            inputs = json.load(f)
        with open("inputs/synthetic/train/{}_labels.json".format(query_str), 'r') as f:
            labels = json.load(f)
        inputs = np.asarray(inputs, dtype=object)
        labels = np.asarray(labels, dtype=object)
        # argsort in descending order
        sort_idx = np.argsort(labels)
        inputs = inputs[sort_idx][::-1]
        labels = labels[sort_idx][::-1]
        print("labels", labels, sum(labels), len(labels))
        predicate_dict = {dsl.Near: [-1.05], dsl.Far: [0.9], dsl.Left: None, dsl.Right: None, dsl.Back: None, dsl.Front: None}

    algorithm = QUIVR(max_depth=3)
    answer = algorithm.run(inputs, labels)

    answers = [[print_program(query_graph.program), score] for query_graph, score in answers]
    return answers, total_time


def test_quivr_soft(n_labeled_pos, n_labeled_neg, max_num_atomic_predicates, max_depth, k, log_name):
    # read from json file
    with open("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/collision_inputs.json", 'r') as f:
        inputs = json.load(f)
    with open("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/collision_labels.json", 'r') as f:
        labels = json.load(f)
    inputs = np.asarray(inputs, dtype=object)
    labels = np.asarray(labels, dtype=object)

    n_pos = sum(labels)
    sampled_labeled_index = random.sample(list(range(n_pos)), n_labeled_pos) + random.sample(list(range(n_pos, len(labels))), n_labeled_neg)
    inputs = inputs[sampled_labeled_index]
    labels = labels[sampled_labeled_index]

    algorithm = QUIVRSoft(max_num_atomic_predicates=max_num_atomic_predicates, max_depth=max_depth, k=k, log_name=log_name)
    answer = algorithm.run(inputs, labels)
    print("answer", len(answer))
    for q, s in answer:
        print(print_program(q), s)

def test_vocal(n_init_pos, n_init_neg, npred, depth, max_duration, beam_width, k, samples_per_iter, budget, multithread, strategy, query_str="collision"):
    if query_str == "collision":
        # read from json file
        with open("inputs/collision_inputs_train.json", 'r') as f:
            inputs = json.load(f)
        with open("inputs/collision_labels_train.json", 'r') as f:
            labels = json.load(f)
        inputs = np.asarray(inputs, dtype=object)
        labels = np.asarray(labels, dtype=object)
        predicate_dict = {dsl.Near: [-1.05], dsl.Far: [1.1], dsl.Left: None, dsl.Right: None}
    else:
        with open("inputs/synthetic/train/{}_inputs.json".format(query_str), 'r') as f:
            inputs = json.load(f)
        with open("inputs/synthetic/train/{}_labels.json".format(query_str), 'r') as f:
            labels = json.load(f)
        inputs = np.asarray(inputs, dtype=object)
        labels = np.asarray(labels, dtype=object)
        # argsort in descending order
        sort_idx = np.argsort(labels)
        inputs = inputs[sort_idx][::-1]
        labels = labels[sort_idx][::-1]
        print("labels", labels, sum(labels), len(labels))
        predicate_dict = {dsl.Near: [-1.05], dsl.Far: [0.9], dsl.Left: None, dsl.Right: None, dsl.Back: None, dsl.Front: None}

    init_labeled_index = random.sample(list(range(sum(labels))), n_init_pos) + random.sample(list(range(sum(labels), len(labels))), n_init_neg)
    algorithm = VOCAL(inputs, labels, predicate_dict, max_npred=npred, max_depth=depth, max_duration=max_duration, beam_width=beam_width, k=k, samples_per_iter=samples_per_iter, budget=budget, multithread=multithread, strategy=strategy)
    answers, total_time = algorithm.run(init_labeled_index)

    answers = [[print_program(query_graph.program), score] for query_graph, score in answers]
    return answers, total_time

def test_exhaustive(n_labeled_pos, n_labeled_neg, npred, depth, max_duration, multithread):
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

    predicate_dict = {dsl.Near: [-1.05], dsl.Far: [1.1], dsl.Left: None, dsl.Right: None}
    algorithm = ExhaustiveSearch(inputs, labels, predicate_dict, max_num_atomic_predicates=npred, max_depth=depth, k=1000, max_duration=max_duration, multithread=multithread)
    algorithm.run()

def evaluate(method, n_init_pos, n_init_neg, npred, depth, max_duration, beam_width, k, budget, multithread, query_str="collision"):
    if query_str == "collision":
        # read from json file
        with open("inputs/collision_inputs_train.json", 'r') as f:
            inputs = json.load(f)
        with open("inputs/collision_labels_train.json", 'r') as f:
            labels = json.load(f)
        inputs = np.asarray(inputs, dtype=object)
        labels = np.asarray(labels, dtype=object)
        predicate_dict = {dsl.Near: [-1.05], dsl.Far: [1.1], dsl.Left: None, dsl.Right: None}
    else:
        with open("inputs/synthetic/train/{}_inputs.json".format(query_str), 'r') as f:
            inputs = json.load(f)
        with open("inputs/synthetic/train/{}_labels.json".format(query_str), 'r') as f:
            labels = json.load(f)
        inputs = np.asarray(inputs, dtype=object)
        labels = np.asarray(labels, dtype=object)
        # argsort in descending order
        sort_idx = np.argsort(labels)
        inputs = inputs[sort_idx][::-1]
        labels = labels[sort_idx][::-1]
        print("labels", labels, sum(labels), len(labels))
        predicate_dict = {dsl.Near: [-1.05], dsl.Far: [0.9], dsl.Left: None, dsl.Right: None, dsl.Back: None, dsl.Front: None}

    init_labeled_index = random.sample(list(range(sum(labels))), n_init_pos) + random.sample(list(range(sum(labels), len(labels))), n_init_neg)

    if method == "vocal":
        algorithm = VOCAL(inputs, labels, predicate_dict, max_npred=npred, max_depth=depth, max_duration=max_duration, beam_width=beam_width, k=k, budget=budget, multithread=multithread)
        answers, total_time = algorithm.run(init_labeled_index)
    elif method == "quivr":
        algorithm = QUIVR(inputs, labels, predicate_dict, max_num_atomic_predicates=npred, max_depth=depth, max_duration=max_duration, multithread=multithread)
        answers, total_time = algorithm.run()

    answers = [[print_program(query_graph.program), score] for query_graph, score in answers]
    return answers, total_time


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', type=str)
    ap.add_argument('--n_labeled_pos', type=int, default=50)
    ap.add_argument('--n_labeled_neg', type=int, default=250)
    ap.add_argument('--n_init_pos', type=int, default=10)
    ap.add_argument('--n_init_neg', type=int, default=50)
    ap.add_argument('--npred', type=int, default=5)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--max_duration', type=int, default=2)
    ap.add_argument('--beam_width', type=int, default=32)
    ap.add_argument('--k', type=int, default=100)
    ap.add_argument('--samples_per_iter', type=int, default=1)
    ap.add_argument('--budget', type=int, default=100)
    ap.add_argument('--multithread', type=int, default=1)
    ap.add_argument('--strategy', type=str, default="sampling")
    ap.add_argument('--query_str', type=str, default="collision")
    ap.add_argument('--run_id', type=int)
    ap.add_argument('--output_to_file', action="store_true")

    args = ap.parse_args()
    method_str = args.method
    n_labeled_pos = args.n_labeled_pos
    n_labeled_neg = args.n_labeled_neg
    n_init_pos = args.n_init_pos
    n_init_neg = args.n_init_neg
    npred = args.npred
    depth = args.depth
    max_duration = args.max_duration
    beam_width = args.beam_width
    k = args.k
    samples_per_iter = args.samples_per_iter
    budget = args.budget
    multithread = args.multithread
    strategy = args.strategy
    query_str = args.query_str
    run_id = args.run_id
    output_to_file = args.output_to_file

    # samples_per_iter should be >= (budget - n_init_pos - n_init_neg) / (npred + max_duration * depth), to ensure the search algorithm can reach to the leaf node.

    # log_name = "{}-npos_{}-nneg_{}-npred_{}-depth_{}-k_{}-model_picker_include_answers".format(method_str, n_labeled_pos, n_labeled_neg, npred, depth, k)
    # log_name = "{}-npred_{}-depth_{}-k_{}-initpos_{}-initneg_{}-max_duration_{}".format(method_str, npred, depth, k, n_init_pos, n_init_neg, max_duration)
    log_name = "{}-{}".format(query_str, run_id)
    config_name = "nip_{}-nin_{}-npred_{}-depth_{}-max_d_{}-bw_{}-k_{}-per_iter_{}-budget_{}-thread_{}".format(n_init_pos, n_init_neg, npred, depth, max_duration, beam_width, k, samples_per_iter, budget, multithread)
    # if dir not exist, create it
    if not os.path.exists("outputs/{}-{}/synthetic/{}/verbose".format(method_str, strategy, config_name)):
        os.makedirs("outputs/{}-{}/synthetic/{}/verbose".format(method_str, strategy, config_name), exist_ok=True)
    if output_to_file:
        f = open("outputs/{}-{}/synthetic/{}/verbose/{}.log".format(method_str, strategy, config_name, log_name), 'w')
        sys.stdout = f

    print(args)

    if method_str == 'quivr':
        test_quivr_exact(n_labeled_pos, n_labeled_neg, npred, depth, max_duration, k, budget, multithread, query_str)
    elif method_str == 'vocal':
        answers, total_time = test_vocal(n_init_pos, n_init_neg, npred, depth, max_duration, beam_width, k, samples_per_iter, budget, multithread, strategy, query_str)
    elif method_str == "exhaustive":
        test_exhaustive(n_labeled_pos, n_labeled_neg, npred, depth, max_duration, multithread)
    elif method_str == 'quivr_soft':
        test_quivr_soft(npred, depth, k, log_name)
        # test_quivr_soft(n_labeled_pos, n_labeled_neg, npred, depth, k, log_name)


    with open("outputs/{}-{}/synthetic/{}/{}.log".format(method_str, strategy, config_name, log_name), 'w') as f:
        for query_str, score in answers:
            f.write("{} {}\n".format(query_str, score))
        f.write("Total Time: {}\n".format(total_time))

    if output_to_file:
        f.close()
        sys.stdout = sys.__stdout__
