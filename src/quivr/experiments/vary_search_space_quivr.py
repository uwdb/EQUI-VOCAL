import csv
import os
from quivr.methods.exhaustive_search import ExhaustiveSearch
from quivr.utils import print_program, rewrite_program_postgres, str_to_program
from quivr.methods.quivr_exact import QUIVR
from quivr.methods.quivr_original import QUIVROriginal
from quivr.methods.quivr_original_no_kleene import QUIVROriginalNoKleene
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

random.seed(10)
# random.seed(time.time())

def test_quivr_original(dataset_name, n_init_pos, n_init_neg, npred, n_nontrivial, n_trivial, depth, max_duration, budget, multithread, query_str, predicate_dict, lru_capacity, with_kleene):

    with open("inputs/trajectory_pairs.json", 'r') as f:
        trajectories = json.load(f)
    with open("inputs/{}/train/{}_inputs.json".format(dataset_name, query_str), 'r') as f:
        inputs = json.load(f)
    with open("inputs/{}/train/{}_labels.json".format(dataset_name, query_str), 'r') as f:
        labels = json.load(f)
    trajectories = np.asarray(trajectories, dtype=object)
    labels = np.asarray(labels, dtype=object)
    inputs = trajectories[inputs]
    if dataset_name.startswith("trajectories_handwritten-sampling_rate"):
        sampling_rate = int(dataset_name.replace("trajectories_handwritten-sampling_rate_", ""))
        print("sampling_rate", sampling_rate)
        # Down-sample the trajectory once every sampling_rate frames
        inputs_downsampled = []
        for input in inputs:
            inputs_downsampled.append([input[0][::sampling_rate], input[1][::sampling_rate]])
        inputs = inputs_downsampled
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    print("pos_idx", len(pos_idx), pos_idx, "neg_idx", len(neg_idx), neg_idx)

    init_labeled_index = random.sample(pos_idx.tolist(), n_init_pos) + random.sample(neg_idx.tolist(), n_init_neg)
    print(init_labeled_index)

    if with_kleene:
        method = QUIVROriginal
    else:
        method = QUIVROriginalNoKleene
    algorithm = method(inputs, labels, predicate_dict, npred, n_nontrivial, n_trivial, depth, max_duration, budget, multithread, lru_capacity)
    output_log = algorithm.run(init_labeled_index)

    return output_log

def test_algorithm(method, dataset_name, n_init_pos, n_init_neg, npred, depth, max_duration, beam_width, pool_size, k, samples_per_iter, budget, multithread, query_str, predicate_dict, strategy, max_vars, port):
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
        inputs = np.asarray(inputs) # input video ids
        labels = np.asarray(labels)
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        print("pos_idx", len(pos_idx), pos_idx, "neg_idx", len(neg_idx), neg_idx)

    init_labeled_index = random.sample(pos_idx.tolist(), n_init_pos) + random.sample(neg_idx.tolist(), n_init_neg)
    print(init_labeled_index)
    if method == "vocal":
        algorithm = VOCAL(inputs, labels, predicate_dict, max_npred=npred, max_depth=depth, max_duration=max_duration, beam_width=beam_width, pool_size=pool_size, k=k, samples_per_iter=samples_per_iter, budget=budget, multithread=multithread, strategy=strategy)
    elif method == "quivr_soft":
        algorithm = QUIVRSoft(inputs, labels, predicate_dict, max_npred=npred, max_depth=depth, max_duration=max_duration, beam_width=beam_width, pool_size=pool_size, k=k, samples_per_iter=samples_per_iter, budget=budget, multithread=multithread, strategy=strategy)
    elif method == "random":
        algorithm = Random(inputs, labels, predicate_dict, max_npred=npred, max_depth=depth, max_duration=max_duration, beam_width=beam_width, pool_size=pool_size, k=k, samples_per_iter=samples_per_iter, budget=budget, multithread=multithread)
    elif method == "vocal_postgres":
        sampling_rate = 4 if dataset_name == "without_duration-sampling_rate_4" else None
        algorithm = VOCALPostgres(dataset_name, inputs, labels, predicate_dict, max_npred=npred, max_depth=depth, max_duration=max_duration, beam_width=beam_width, pool_size=pool_size, k=k, samples_per_iter=samples_per_iter, budget=budget, multithread=multithread, strategy=strategy, max_vars=max_vars, port=port, sampling_rate=sampling_rate)

    output_log = algorithm.run(init_labeled_index)
    return output_log

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', type=str)
    ap.add_argument('--n_labeled_pos', type=int, default=50)
    ap.add_argument('--n_labeled_neg', type=int, default=250)
    ap.add_argument('--n_init_pos', type=int, default=10)
    ap.add_argument('--n_init_neg', type=int, default=50)
    ap.add_argument('--dataset_name', type=str)
    ap.add_argument('--npred', type=int, default=5)
    ap.add_argument('--n_nontrivial', type=int)
    ap.add_argument('--n_trivial', type=int)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--max_duration', type=int, default=2)
    ap.add_argument('--beam_width', type=int, default=32)
    ap.add_argument('--pool_size', type=int, default=100)
    ap.add_argument('--k', type=int, default=100)
    # ap.add_argument('--samples_per_iter', type=int, default=1)
    ap.add_argument('--budget', type=int, default=100)
    ap.add_argument('--multithread', type=int, default=1)
    ap.add_argument('--strategy', type=str, default="sampling")
    ap.add_argument('--max_vars', type=int, default=2)
    ap.add_argument('--query_str', type=str, default="collision")
    ap.add_argument('--run_id', type=int)
    ap.add_argument('--output_to_file', action="store_true")
    ap.add_argument('--port', type=int, default=5432)
    ap.add_argument('--n_candidate_pred', type=int)
    ap.add_argument('--lru_capacity', type=int)

    args = ap.parse_args()
    method_str = args.method
    n_labeled_pos = args.n_labeled_pos
    n_labeled_neg = args.n_labeled_neg
    n_init_pos = args.n_init_pos
    n_init_neg = args.n_init_neg
    dataset_name = args.dataset_name
    npred = args.npred
    n_nontrivial = args.n_nontrivial
    n_trivial = args.n_trivial
    depth = args.depth
    max_duration = args.max_duration
    beam_width = args.beam_width
    pool_size = args.pool_size
    k = args.k
    # samples_per_iter = args.samples_per_iter
    budget = args.budget
    multithread = args.multithread
    strategy = args.strategy
    max_vars = args.max_vars
    query_str = args.query_str
    run_id = args.run_id
    output_to_file = args.output_to_file
    port = args.port
    n_candidate_pred = args.n_candidate_pred
    lru_capacity = args.lru_capacity
    # samples_per_iter should be >= (budget - n_init_pos - n_init_neg) / (npred + max_duration * depth), to ensure the search algorithm can reach to the leaf node.
    samples_per_iter = math.ceil((budget - n_init_pos - n_init_neg) * 1.0 / (npred + max_duration * depth))
    print(samples_per_iter)

    # Define file directory and name
    if method_str in ["quivr_original", "quivr_original_no_kleene"]:
        method_name = method_str
        config_name = "nip_{}-nin_{}-npred_{}-depth_{}-max_d_{}-thread_{}".format(n_init_pos, n_init_neg, npred, depth, max_duration, multithread)
    elif method_str == "vocal_postgres":
        method_name = "{}-{}".format(method_str, strategy)
        config_name = "nip_{}-nin_{}-npred_{}-depth_{}-max_d_{}-nvars_{}-bw_{}-pool_size_{}-k_{}-per_iter_{}-budget_{}-thread_{}".format(n_init_pos, n_init_neg, npred, depth, max_duration, max_vars, beam_width, pool_size, k, samples_per_iter, budget, multithread)

    if dataset_name.startswith("trajectories_handwritten") and query_str == "Far_3(o0, o1); Conjunction(Conjunction(Near_1(o0, o1), LeftQuadrant(o0)), Behind(o0, o1))":
        # if method_str == "vocal_postgres":
        #     predicate_dict = [{"name": "Near", "parameters": [1], "nargs": 2}, {"name": "Far", "parameters": [3], "nargs": 2}, {"name": "LeftOf", "parameters": None, "nargs": 2}, {"name": "Behind", "parameters": None, "nargs": 2}, {"name": "RightOf", "parameters": None, "nargs": 2}, {"name": "FrontOf", "parameters": None, "nargs": 2}, {"name": "RightQuadrant", "parameters": None, "nargs": 1}, {"name": "LeftQuadrant", "parameters": None, "nargs": 1}, {"name": "TopQuadrant", "parameters": None, "nargs": 1}, {"name": "BottomQuadrant", "parameters": None, "nargs": 1}]
        # elif method_str == "quivr_original":
        #     predicate_dict = {dsl.Near: [-1], dsl.Far: [3], dsl.MinLength: None, dsl.LeftOf: None, dsl.RightOf: None, dsl.Behind: None, dsl.FrontOf: None, dsl.LeftQuadrant: None, dsl.RightQuadrant: None, dsl.TopQuadrant: None, dsl.BottomQuadrant: None}
        if method_str == "quivr_original_no_kleene":
            if n_candidate_pred == 4:
                predicate_dict = {dsl.Near: [-1], dsl.Far: [3], dsl.Behind: None, dsl.LeftQuadrant: None}
            elif n_candidate_pred == 5:
                predicate_dict = {dsl.Near: [-1], dsl.Far: [3], dsl.Behind: None, dsl.LeftQuadrant: None, dsl.FrontOf: None}
            elif n_candidate_pred == 6:
                predicate_dict = {dsl.Near: [-1], dsl.Far: [3], dsl.Behind: None, dsl.LeftQuadrant: None, dsl.FrontOf: None, dsl.TopQuadrant: None}
            elif n_candidate_pred == 7:
                predicate_dict = {dsl.Near: [-1], dsl.Far: [3], dsl.Behind: None, dsl.LeftQuadrant: None, dsl.FrontOf: None, dsl.TopQuadrant: None, dsl.BottomQuadrant: None}
            elif n_candidate_pred == 8:
                predicate_dict = {dsl.Near: [-1], dsl.Far: [3], dsl.Behind: None, dsl.LeftQuadrant: None, dsl.FrontOf: None, dsl.TopQuadrant: None, dsl.BottomQuadrant: None, dsl.LeftOf: None}
            elif n_candidate_pred == 9:
                predicate_dict = {dsl.Near: [-1], dsl.Far: [3], dsl.Behind: None, dsl.LeftQuadrant: None, dsl.FrontOf: None, dsl.TopQuadrant: None, dsl.BottomQuadrant: None, dsl.LeftOf: None, dsl.RightOf: None}
            elif n_candidate_pred == 10:
                predicate_dict = {dsl.Near: [-1], dsl.Far: [3], dsl.Behind: None, dsl.LeftQuadrant: None, dsl.FrontOf: None, dsl.TopQuadrant: None, dsl.BottomQuadrant: None, dsl.LeftOf: None, dsl.RightOf: None, dsl.RightQuadrant: None}

    print("predicate_dict", predicate_dict)
    print("n_candidate_pred", n_candidate_pred)
    print("lru_capacity", lru_capacity)

    log_dirname = os.path.join("outputs", "vary_search_space", dataset_name, method_name, config_name)
    log_filename = "{}-lru_{}-n_candidate_pred_{}".format(query_str, lru_capacity, n_candidate_pred) if lru_capacity else "{}-lru_none-n_candidate_pred_{}".format(query_str, n_candidate_pred)

    # if dir not exist, create it
    if output_to_file:
        if not os.path.exists(os.path.join(log_dirname, "verbose")):
            os.makedirs(os.path.join(log_dirname, "verbose"), exist_ok=True)
        verbose_f = open(os.path.join(log_dirname, "verbose", "{}.log".format(log_filename)), 'w')
        sys.stdout = verbose_f

    print(args)

    if method_str == 'quivr_original':
        output_log = test_quivr_original(dataset_name, n_init_pos, n_init_neg, npred, n_nontrivial, n_trivial, depth, max_duration, budget, multithread, query_str, predicate_dict, lru_capacity, with_kleene=True)
    elif method_str == 'quivr_original_no_kleene':
        output_log = test_quivr_original(dataset_name, n_init_pos, n_init_neg, npred, n_nontrivial, n_trivial, depth, max_duration, budget, multithread, query_str, predicate_dict, lru_capacity, with_kleene=False)
    elif method_str in ['vocal', 'quivr_soft', 'random', 'vocal_postgres']:
        output_log = test_algorithm(method_str, dataset_name, n_init_pos, n_init_neg, npred, depth, max_duration, beam_width, pool_size, k, samples_per_iter, budget, multithread, query_str, predicate_dict, strategy, max_vars, port)

    if output_to_file:
        with open(os.path.join(log_dirname, "{}.log".format(log_filename)), 'w') as f:
            for line in output_log:
                f.write("{}\n".format(line))

        verbose_f.close()
        sys.stdout = sys.__stdout__
