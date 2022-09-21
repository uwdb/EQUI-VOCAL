import argparse
from doctest import testfile
from functools import partial
from posixpath import dirname
import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from quivr.utils import print_program, rewrite_program_postgres, str_to_program, quivr_str_to_postgres_program, postgres_execute, str_to_program_postgres
from sklearn.metrics import f1_score
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from time import time
from lru import LRU
import sys
import uuid
import psycopg

def test_synthetic_queries(method, dataset_name, n_init_pos, n_init_neg, npred, depth, max_duration, nvars, beam_width, k, samples_per_iter, budget, multithread):
    dir_name = "outputs/{}/{}/nip_{}-nin_{}-npred_{}-depth_{}-max_d_{}-nvars_{}-bw_{}-k_{}-per_iter_{}-budget_{}-thread_{}".format(method, dataset_name, n_init_pos, n_init_neg, npred, depth, max_duration, nvars, beam_width, k, samples_per_iter, budget, multithread)
    print("dir_name", dir_name)

    rank_log = []
    true_positive_at_10, true_positive_at_100, total_count = 0, 0, 0
    total_time_log = []
    test_f1_log = []

    args = []
    # for each file in folder
    for file in os.listdir(dir_name):
        if file.endswith(".log"):
            args.append([file, method])

    with ThreadPoolExecutor(max_workers=1) as executor:
        for result in executor.map(partial(_helper, dir_name=dir_name, dataset_name=dataset_name, k=k), args):
            f1_score, rank, total_time, is_positive_at_10, is_positive_at_100 = result
            rank_log.append(rank)
            total_time_log.append(total_time)
            test_f1_log.append(f1_score)
            total_count += 1
            if is_positive_at_10:
                true_positive_at_10 += 1
            if is_positive_at_100:
                true_positive_at_100 += 1


    # Mean F1 score
    mean_f1_score = np.mean(test_f1_log)
    # Median Rank
    median_rank = np.median(rank_log)
    # Mean Reciprocal Rank
    mean_reciprocal_rank = np.mean(1.0 / np.array(rank_log))
    # Recall at k
    recall_at_10 = true_positive_at_10 / total_count
    recall_at_100 = true_positive_at_100 / total_count
    # Mean Time
    mean_time = np.mean(total_time_log)
    # Standard Deviation of Time
    std_time = np.std(total_time_log)

    print("test_f1_log: {}".format(test_f1_log))
    print("rank_log: {}; total_time_log: {}".format(rank_log, total_time_log))
    print("Mean F1 score: {}".format(mean_f1_score))
    print("Median Rank: {}".format(median_rank))
    print("Mean Reciprocal Rank: {}".format(mean_reciprocal_rank))
    print("Recall@10: {}".format(recall_at_10))
    print("Recall@100: {}".format(recall_at_100))
    print("Mean Time: {}".format(mean_time))
    print("Standard Deviation of Time: {}".format(std_time))
    return mean_f1_score, median_rank, mean_reciprocal_rank, recall_at_10, recall_at_100, mean_time, std_time

def _helper(args, dir_name, dataset_name, k):
    file, method = args
    is_positive_at_100 = 0
    is_positive_at_10 = 0
    target_query = file.split("-")[0]
    # read file line by line
    with open(os.path.join(dir_name, file), "r") as f:
        lines = f.read().splitlines()
    answers = lines[:-1]
    answers = [answer.rsplit(" ", 1) for answer in answers]
    total_time = float(lines[-1].rsplit(" ", 1)[-1])
    rank = k
    for i, (query_str, _) in enumerate(answers, start=1):
        postgres_program = quivr_str_to_postgres_program(target_query)
        target_query_postgres_str = rewrite_program_postgres(postgres_program)
        if query_str == target_query_postgres_str:
            if i <= 10:
                is_positive_at_10 = 1
            if i <= 100:
                is_positive_at_100 = 1
            rank = i
            break
    if method.startswith("vocal_postgres"):
        f1_score = get_f1_score_postgres(answers[:10], target_query, dataset_name)
    else:
        f1_score = get_f1_score(answers[:10], target_query, dataset_name)
    # f1_score = -1
    return f1_score, rank, total_time, is_positive_at_10, is_positive_at_100

def get_f1_score(test_queries, target_query, dataset_name):
    _start = time()
    with open("inputs/{}/test/{}_inputs.json".format(dataset_name, target_query), 'r') as f:
        inputs = json.load(f)
    with open("inputs/{}/test/{}_labels.json".format(dataset_name, target_query), 'r') as f:
        labels = json.load(f)
    inputs = np.asarray(inputs, dtype=object)
    labels = np.asarray(labels, dtype=object)
    inputs, _, labels, _ = train_test_split(inputs, labels, train_size=300, random_state=42, stratify=labels)

    # Top-10 queries, majority vote
    y_pred = []
    for i in range(len(inputs)):
        input = inputs[i]
        label = labels[i]
        memoize = LRU(10000)
        y_pred_per_query = []
        weights = []
        for test_query, train_score in test_queries:
            weights.append(float(train_score))
            program = str_to_program(test_query)
            result, new_memoize = program.execute(input, label, memoize, {})
            y_pred_per_query.append(int(result[0, len(input[0])] > 0))
            memoize.update(new_memoize)
        # majority vote
        # print(y_pred_per_query, weights)
        y_pred.append(np.average(y_pred_per_query, weights=weights) >= 0.5)
    score = f1_score(list(labels), y_pred)
    print("score: {}".format(score))
    print("time: {}".format(time() - _start))
    return score

def get_f1_score_postgres(test_queries, target_query, dataset_name):
    print("target_query", target_query)
    _start = time()
    with open("inputs/{}/test/{}_inputs.json".format(dataset_name, target_query), 'r') as f:
        inputs = json.load(f)
    with open("inputs/{}/test/{}_labels.json".format(dataset_name, target_query), 'r') as f:
        labels = json.load(f)
    inputs = np.asarray(inputs, dtype=object)
    labels = np.asarray(labels, dtype=object)
    inputs, _, labels, _ = train_test_split(inputs, labels, train_size=300, random_state=42, stratify=labels)
    print("labels", labels)

    inputs_table_name = "Obj_trajectories_{}".format(uuid.uuid4().hex)
    dsn = "dbname=myinner_db user=enhaoz host=localhost"
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            # Create temporary table for inputs
            cur.execute("""
            CREATE TABLE {} (
                oid INT,
                vid INT,
                fid INT,
                shape varchar,
                color varchar,
                material varchar,
                x1 float,
                y1 float,
                x2 float,
                y2 float
            );
            """.format(inputs_table_name))

            # Load inputs into temporary table
            csv_data = []
            for vid, pair in enumerate(inputs):
                t1 = pair[0]
                t2 = pair[1]
                assert(len(t1) == len(t2))
                for fid, (bbox1, bbox2) in enumerate(zip(t1, t2)):
                    csv_data.append((0, vid, fid, "cube", "red", "metal", bbox1[0], bbox1[1], bbox1[2], bbox1[3]))
                    csv_data.append((1, vid, fid, "cube", "red", "metal", bbox2[0], bbox2[1], bbox2[2], bbox2[3]))
            with cur.copy("COPY {} FROM STDIN".format(inputs_table_name)) as cur_copy:
                for row in csv_data:
                    cur_copy.write_row(row)
            conn.commit()

    y_pred = []
    weights = []
    for test_query, train_score in test_queries:
        print("test_query", test_query)
        y_pred_per_query = []
        weights.append(float(train_score))
        program = str_to_program_postgres(test_query)
        memoize = [{} for _ in range(len(inputs))]
        result, new_memoize = postgres_execute(dsn, program, list(range(len(inputs))), memoize, inputs_table_name)
        for i in range(len(inputs)):
            if i in result:
                y_pred_per_query.append(1)
            else:
                y_pred_per_query.append(0)
        y_pred.append(y_pred_per_query)
    y_pred = np.asarray(y_pred)
    y_pred = np.average(y_pred, axis=0, weights=weights) >= 0.5
    print("weights", weights)
    print("y_pred", y_pred)
    score = f1_score(list(labels), y_pred)
    print("score: {}".format(score))
    print("time: {}".format(time() - _start))
    return score

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', type=str)
    ap.add_argument('--n_init_pos', type=int)
    ap.add_argument('--n_init_neg', type=int)
    ap.add_argument('--dataset_name', type=str)
    ap.add_argument('--npred', type=int)
    ap.add_argument('--depth', type=int)
    ap.add_argument('--max_duration', type=int)
    ap.add_argument('--beam_width', type=int)
    ap.add_argument('--k', type=int)
    ap.add_argument('--samples_per_iter', type=int)
    ap.add_argument('--budget', type=int)
    ap.add_argument('--multithread', type=int)
    ap.add_argument('--nvars', type=int)

    args = ap.parse_args()
    method_str = args.method
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
    nvars = args.nvars
    # args = ["random-unrestricted_v2", "synthetic_rare", 10, 10, 5, 3, 5, 2, 100, 5, 100, 8]
    with open("outputs/{}/{}/eval-nip_{}-nin_{}-npred_{}-depth_{}-max_d_{}-nvars_{}-bw_{}-k_{}-per_iter_{}-budget_{}-thread_{}.txt".format(method_str, dataset_name, n_init_pos, n_init_neg, npred, depth, max_duration, nvars, beam_width, k, samples_per_iter, budget, multithread), 'w') as f:
        sys.stdout = f
        _start = time()
        test_synthetic_queries(method_str, dataset_name, n_init_pos, n_init_neg, npred, depth, max_duration, nvars, beam_width, k, samples_per_iter, budget, multithread)
        print("Time: {}".format(time() - _start))