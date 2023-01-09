import csv
import os
from methods.exhaustive_search import ExhaustiveSearch
from utils import print_program, rewrite_program_postgres, str_to_program
from methods.quivr_exact import QUIVR
from methods.quivr_original import QUIVROriginal
from methods.quivr_original_no_kleene import QUIVROriginalNoKleene
from methods.vocal import VOCAL
from methods.quivr_soft import QUIVRSoft
from methods.random import Random
from methods.vocal_postgres import VOCALPostgres
import json
import random
import math
import numpy as np
from sklearn.metrics import f1_score
import argparse
import sys
import dsl
import pandas
import time
from sklearn.model_selection import train_test_split

random.seed(10)
# random.seed(time.time())

def test_quivr_original(dataset_name, n_init_pos, n_init_neg, npred, n_nontrivial, n_trivial, depth, max_duration, budget, multithread, query_str, predicate_dict, lru_capacity, sampling_rate, with_kleene):

    with open("inputs/trajectory_pairs.json", 'r') as f:
        trajectories = json.load(f)
    with open("inputs/{}/train/{}_inputs.json".format(dataset_name, query_str), 'r') as f:
        inputs = json.load(f)
    with open("inputs/{}/train/{}_labels.json".format(dataset_name, query_str), 'r') as f:
        labels = json.load(f)
    trajectories = np.asarray(trajectories, dtype=object)
    labels = np.asarray(labels, dtype=object)
    inputs = trajectories[inputs]

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

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', type=str)
    ap.add_argument('--n_init_pos', type=int, default=10)
    ap.add_argument('--n_init_neg', type=int, default=50)
    ap.add_argument('--dataset_name', type=str)
    ap.add_argument('--npred', type=int, default=5)
    ap.add_argument('--n_nontrivial', type=int)
    ap.add_argument('--n_trivial', type=int)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--max_duration', type=int, default=2)
    ap.add_argument('--budget', type=int, default=100)
    ap.add_argument('--multithread', type=int, default=1)
    ap.add_argument('--query_str', type=str, default="collision")
    ap.add_argument('--output_to_file', action="store_true")
    ap.add_argument('--n_candidate_pred', type=int)
    ap.add_argument('--lru_capacity', type=int)
    ap.add_argument('--sampling_rate', type=int)

    args = ap.parse_args()
    method_str = args.method
    n_init_pos = args.n_init_pos
    n_init_neg = args.n_init_neg
    dataset_name = args.dataset_name
    npred = args.npred
    n_nontrivial = args.n_nontrivial
    n_trivial = args.n_trivial
    depth = args.depth
    max_duration = args.max_duration
    budget = args.budget
    multithread = args.multithread
    query_str = args.query_str
    output_to_file = args.output_to_file
    n_candidate_pred = args.n_candidate_pred
    lru_capacity = args.lru_capacity
    sampling_rate = args.sampling_rate

    dataset_name = dataset_name + "-sampling_rate_{}".format(sampling_rate)

    # Define file directory and name
    if method_str in ["quivr_original", "quivr_original_no_kleene"]:
        method_name = method_str
        config_name = "nip_{}-nin_{}-npred_{}-depth_{}-max_d_{}-thread_{}".format(n_init_pos, n_init_neg, npred, depth, max_duration, multithread)

    if dataset_name.startswith("trajectories_handwritten") and query_str == "Far_3(o0, o1); Conjunction(Conjunction(Near_1(o0, o1), LeftQuadrant(o0)), Behind(o0, o1))":
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
    pring("sampling_rate", sampling_rate)

    log_dirname = os.path.join("outputs", "vary_sampling_rate", dataset_name, method_name, config_name)
    log_filename = "{}-lru_{}-n_candidate_pred_{}".format(query_str, lru_capacity, n_candidate_pred) if lru_capacity else "{}-lru_none-n_candidate_pred_{}".format(query_str, n_candidate_pred)

    # if dir not exist, create it
    if output_to_file:
        if not os.path.exists(os.path.join(log_dirname, "verbose")):
            os.makedirs(os.path.join(log_dirname, "verbose"), exist_ok=True)
        verbose_f = open(os.path.join(log_dirname, "verbose", "{}.log".format(log_filename)), 'w')
        sys.stdout = verbose_f

    print(args)

    if method_str == 'quivr_original':
        output_log = test_quivr_original(dataset_name, n_init_pos, n_init_neg, npred, n_nontrivial, n_trivial, depth, max_duration, budget, multithread, query_str, predicate_dict, lru_capacity, sampling_rate, with_kleene=True)
    elif method_str == 'quivr_original_no_kleene':
        output_log = test_quivr_original(dataset_name, n_init_pos, n_init_neg, npred, n_nontrivial, n_trivial, depth, max_duration, budget, multithread, query_str, predicate_dict, lru_capacity, sampling_rate, with_kleene=False)

    if output_to_file:
        with open(os.path.join(log_dirname, "{}.log".format(log_filename)), 'w') as f:
            for line in output_log:
                f.write("{}\n".format(line))

        verbose_f.close()
        sys.stdout = sys.__stdout__
