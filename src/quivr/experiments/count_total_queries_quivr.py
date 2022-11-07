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

def test_quivr_original(npred, n_nontrivial, n_trivial, depth, max_duration, multithread, predicate_dict, with_kleene):
    inputs = np.array([1])
    labels = np.array([1])

    if with_kleene:
        method = QUIVROriginal
    else:
        method = QUIVROriginalNoKleene
    algorithm = method(inputs, labels, predicate_dict, npred, n_nontrivial, n_trivial, depth, max_duration, 0, multithread, None)
    query_count = algorithm.enumerate_search_space()
    return query_count

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', type=str)
    ap.add_argument('--dataset_name', type=str)
    ap.add_argument('--npred', type=int, default=5)
    ap.add_argument('--n_nontrivial', type=int)
    ap.add_argument('--n_trivial', type=int)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--max_duration', type=int, default=2)
    ap.add_argument('--multithread', type=int, default=1)
    ap.add_argument('--output_to_file', action="store_true")
    ap.add_argument('--n_candidate_pred', type=int)

    args = ap.parse_args()
    method_str = args.method
    dataset_name = args.dataset_name
    npred = args.npred
    n_nontrivial = args.n_nontrivial
    n_trivial = args.n_trivial
    depth = args.depth
    max_duration = args.max_duration
    multithread = args.multithread
    output_to_file = args.output_to_file
    n_candidate_pred = args.n_candidate_pred

    if dataset_name.startswith("trajectories_handwritten"):
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
    elif dataset_name.startswith("synthetic_trajectories_rare"):
        predicate_dict = {dsl.Near: [-1], dsl.Far: [3], dsl.LeftOf: None, dsl.RightOf: None, dsl.Behind: None, dsl.FrontOf: None, dsl.LeftQuadrant: None, dsl.RightQuadrant: None, dsl.TopQuadrant: None, dsl.BottomQuadrant: None}

    # if method_str == "quivr_original":
    #     predicate_dict[dsl.MinLength] = None

    print("predicate_dict", predicate_dict)
    print("n_candidate_pred", n_candidate_pred)

    log_dirname = os.path.join("outputs", "count_search_space")
    log_filename = "dataset_{}-n_candidate_pred_{}-method_{}-npred_{}-n_nontrivial_{}-n_trivial_{}-depth_{}-max_d_{}".format(dataset_name, n_candidate_pred, method_str, npred, n_nontrivial, n_trivial, depth, max_duration)

    print(args)

    if method_str == 'quivr_original':
        with_kleene = True
    elif method_str == 'quivr_original_no_kleene':
        with_kleene = False

    output_log = test_quivr_original(npred, n_nontrivial, n_trivial, depth, max_duration, multithread, predicate_dict, with_kleene=with_kleene)

    if output_to_file:
        if not os.path.exists(log_dirname):
            os.makedirs(log_dirname, exist_ok=True)
        with open(os.path.join(log_dirname, "{}.log".format(log_filename)), 'w') as f:
            for line in output_log:
                f.write("{}\n".format(line))
