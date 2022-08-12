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

random.seed(10)

def test_quivr_exact(n_labeled_pos, n_labeled_neg):
    # read from json file
    with open("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/collision_inputs.json", 'r') as f:
        inputs = json.load(f)
    with open("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/collision_labels.json", 'r') as f:
        labels = json.load(f)
    inputs = np.asarray(inputs, dtype=object)
    labels = np.asarray(labels, dtype=object)

    sampled_pos_index = [2340, 133, 1756, 1976, 2367, 60, 844, 1894, 2012, 1136, 656, 140, 2132, 2007, 1342, 311, 1023, 1479, 182, 1721, 568, 1454, 1563, 1725, 1161, 1074, 1871, 715, 1241, 1485, 544, 980, 1800, 1536, 181, 2387, 16, 965, 548, 798, 1240, 2196, 1499, 983, 1287, 2248, 1845, 1785, 1925, 266]
    sampled_neg_index = [13155, 12034, 7776, 10680, 5019, 6131, 9221, 6362, 8912, 5132, 10593, 7391, 12396, 13235, 3637, 11197, 3783, 4909, 8755, 11750, 8587, 12308, 4307, 4039, 9691, 5182, 5585, 8169, 9555, 9241, 9757, 6478, 13611, 6957, 4808, 12570, 11007, 5380, 4414, 6831, 9923, 7414, 5159, 13277, 13085, 5312, 5342, 10323, 8151, 6542]
    inputs = inputs[sampled_pos_index[:n_labeled_pos]].tolist() + inputs[sampled_neg_index[:n_labeled_neg]].tolist()
    labels = labels[sampled_pos_index[:n_labeled_pos]].tolist() + labels[sampled_neg_index[:n_labeled_neg]].tolist()

    algorithm = QUIVR(max_depth=3)
    answer = algorithm.run(inputs, labels)
    print("answer", len(answer))
    for q in answer:
        print(print_program(q))

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

def test_vocal(n_init_pos, n_init_neg, npred, depth, max_duration, beam_width, k, budget, multithread, query_str="collision"):
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
    algorithm = VOCAL(inputs, labels, predicate_dict, max_npred=npred, max_depth=depth, max_duration=max_duration, beam_width=beam_width, k=k, budget=budget, multithread=multithread)
    algorithm.run(init_labeled_index)

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

    answers = [[print_program(query_graph), score] for query_graph, score in answers]
    return answers, total_time


def test_synthetic_queries(method, num_runs, n_init_pos, n_init_neg, npred, depth, max_duration, beam_width, k, budget, multithread):
    # read synthetic queries from csv file
    colnames = ["query", "npos", "nneg", "ratio"]
    data = pandas.read_csv('inputs/synthetic/queries.csv', names=colnames)
    synthetic_queries = data.query.tolist()

    rank_log = []
    true_positive, total_count = 0, 0
    total_time_log = []

    for query in synthetic_queries:
        for _ in range(num_runs):
            answers, total_time = evaluate(method, n_init_pos, n_init_neg, npred, depth, max_duration, beam_width, k, budget, multithread, query)
            rank = k
            for i, (query_str, _) in enumerate(answers, start=1):
                # TODO: make sure that predicates in each scene graph are ordered.
                if query_str == query:
                    true_positive += 1
                    rank = i
                    break
            total_count += 1
            rank_log.append(rank)
            total_time_log.append(total_time)

    # Median Rank
    median_rank = np.median(rank_log)
    # Mean Reciprocal Rank
    mean_reciprocal_rank = np.mean(1.0 / np.array(rank_log))
    # Recall at k
    recall_at_k = true_positive / total_count
    # Mean Time
    mean_time = np.mean(total_time_log)
    # Standard Deviation of Time
    std_time = np.std(total_time_log)

    print("rank_log: {}; total_time_log: {}".format(rank_log, total_time_log))
    print("Median Rank: {}".format(median_rank))
    print("Mean Reciprocal Rank: {}".format(mean_reciprocal_rank))
    print("Recall at k: {}".format(recall_at_k))
    print("Mean Time: {}".format(mean_time))
    print("Standard Deviation of Time: {}".format(std_time))
    return median_rank, mean_reciprocal_rank, recall_at_k, mean_time, std_time


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
    ap.add_argument('--budget', type=int, default=100)
    ap.add_argument('--multithread', type=int, default=1)
    ap.add_argument('--query_str', type=str, default="collision")
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
    budget = args.budget
    multithread = args.multithread
    query_str = args.query_str
    output_to_file = args.output_to_file

    # log_name = "{}-npos_{}-nneg_{}-npred_{}-depth_{}-k_{}-model_picker_include_answers".format(method_str, n_labeled_pos, n_labeled_neg, npred, depth, k)
    # log_name = "{}-npred_{}-depth_{}-k_{}-initpos_{}-initneg_{}-max_duration_{}".format(method_str, npred, depth, k, n_init_pos, n_init_neg, max_duration)
    log_name = "{}-query_{}-npred_{}-depth_{}-k_{}-max_duration_{}-multithread_{}".format(method_str, query_str, npred, depth, k, max_duration, multithread)
    # if dir not exist, create it
    if not os.path.exists("outputs/{}".format(method_str)):
        os.makedirs("outputs/{}".format(method_str))
    if output_to_file:
        f = open("outputs/{}/{}.log".format(method_str, log_name), 'w')
        sys.stdout = f

    print(args)

    if method_str == 'quivr':
        test_quivr_exact()
        # test_quivr_exact(n_labeled_pos, n_labeled_neg)
    elif method_str == 'vocal':
        test_vocal(n_init_pos, n_init_neg, npred, depth, max_duration, beam_width, k, budget, multithread, query_str)
    elif method_str == "exhaustive":
        test_exhaustive(n_labeled_pos, n_labeled_neg, npred, depth, max_duration, multithread)
    elif method_str == 'quivr_soft':
        test_quivr_soft(npred, depth, k, log_name)
        # test_quivr_soft(n_labeled_pos, n_labeled_neg, npred, depth, k, log_name)

    if output_to_file:
        f.close()
        sys.stdout = sys.__stdout__
