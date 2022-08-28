from doctest import testfile
from functools import partial
from posixpath import dirname
import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from quivr.utils import print_program, str_to_program
from sklearn.metrics import f1_score
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from time import time
from lru import LRU
import sys

def test_synthetic_queries(method, dataset_name, n_init_pos, n_init_neg, npred, depth, max_duration, beam_width, k, samples_per_iter, budget, multithread):
    dir_name = "outputs/{}/{}/nip_{}-nin_{}-npred_{}-depth_{}-max_d_{}-bw_{}-k_{}-per_iter_{}-budget_{}-thread_{}".format(method, dataset_name, n_init_pos, n_init_neg, npred, depth, max_duration, beam_width, k, samples_per_iter, budget, multithread)
    print("dir_name", dir_name)

    rank_log = []
    true_positive_at_10, true_positive_at_100, total_count = 0, 0, 0
    total_time_log = []
    test_f1_log = []

    args = []
    # for each file in folder
    for file in os.listdir(dir_name):
        if file.endswith(".log"):
            args.append(file)

    with ThreadPoolExecutor(max_workers=20) as executor:
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

def _helper(file, dir_name, dataset_name, k):
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
        if query_str == target_query:
            if i <= 10:
                is_positive_at_10 = 1
            if i <= 100:
                is_positive_at_100 = 1
            rank = i
            break
    f1_score = get_f1_score(answers[:10], target_query, dataset_name)
    return f1_score, rank, total_time, is_positive_at_10, is_positive_at_100

def get_f1_score(test_queries, target_query, dataset_name):
    with open("inputs/{}/test/{}_inputs.json".format(dataset_name, target_query), 'r') as f:
        inputs = json.load(f)
    with open("inputs/{}/test/{}_labels.json".format(dataset_name, target_query), 'r') as f:
        labels = json.load(f)
    inputs = np.asarray(inputs, dtype=object)
    labels = np.asarray(labels, dtype=object)
    inputs, _, labels, _ = train_test_split(inputs, labels, train_size=500, random_state=42, stratify=labels)

    # Top-10 queries, majority vote
    y_pred = []
    for i in range(len(inputs)):
        input = inputs[i]
        label = labels[i]
        memoize = LRU(5000)
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
    return score


if __name__ == '__main__':
    # with open("outputs/{}/synthetic/eval-nip_{}-nin_{}-npred_{}-depth_{}-max_d_{}-bw_{}-k_{}-per_iter_{}-budget_{}-thread_{}.txt".format("quivr_soft-sampling", 10, 10, 5, 3, 5, 128, 100, 5, 100, 8), 'w') as f:
    args = ["random", "synthetic_rare", 10, 10, 5, 3, 5, 32, 100, 5, 100, 8]
    with open("outputs/{}/{}/eval-nip_{}-nin_{}-npred_{}-depth_{}-max_d_{}-bw_{}-k_{}-per_iter_{}-budget_{}-thread_{}.txt".format(*args), 'w') as f:
        sys.stdout = f
        _start = time()
        test_synthetic_queries(*args)
        print("Time: {}".format(time() - _start))