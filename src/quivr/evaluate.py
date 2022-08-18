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

def test_synthetic_queries(method, n_init_pos, n_init_neg, npred, depth, max_duration, beam_width, k, samples_per_iter, budget, multithread):
    dir_name = "outputs/{}/synthetic/nip_{}-nin_{}-npred_{}-depth_{}-max_d_{}-bw_{}-k_{}-per_iter_{}-budget_{}-thread_{}".format(method, n_init_pos, n_init_neg, npred, depth, max_duration, beam_width, k, samples_per_iter, budget, multithread)
    print("dir_name", dir_name)

    rank_log = []
    true_positive, total_count = 0, 0
    total_time_log = []
    test_f1_log = []

    args = []
    # for each file in folder
    for file in os.listdir(dir_name):
        if file.endswith(".log"):
            args.append(file)

    with ThreadPoolExecutor(max_workers=20) as executor:
        for result in executor.map(partial(_helper, dir_name=dir_name, k=k), args):
            average_f1_score, rank, total_time, is_positive = result
            rank_log.append(rank)
            total_time_log.append(total_time)
            test_f1_log.append(average_f1_score)
            total_count += 1
            if is_positive:
                true_positive += 1


    # Mean F1 score
    mean_f1_score = np.mean(test_f1_log)
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

    print("test_f1_log: {}".format(test_f1_log))
    print("rank_log: {}; total_time_log: {}".format(rank_log, total_time_log))
    print("Mean F1 score: {}".format(mean_f1_score))
    print("Median Rank: {}".format(median_rank))
    print("Mean Reciprocal Rank: {}".format(mean_reciprocal_rank))
    print("Recall at k: {}".format(recall_at_k))
    print("Mean Time: {}".format(mean_time))
    print("Standard Deviation of Time: {}".format(std_time))
    return mean_f1_score, median_rank, mean_reciprocal_rank, recall_at_k, mean_time, std_time

def _helper(file, dir_name, k):
    is_positive = 0
    target_query = file.split("-")[0]
    # read file line by line
    with open(os.path.join(dir_name, file), "r") as f:
        lines = f.readlines()
        answers = lines[:-1]
        answers = [answer.rsplit(" ", 1) for answer in answers]
        total_time = float(lines[-1].rsplit(" ", 1)[-1])
        rank = k
        for i, (query_str, _) in enumerate(answers, start=1):
            if query_str == target_query:
                is_positive = 1
                rank = i
                break
        average_f1_score = get_f1_score(answers[:10], target_query)
    return average_f1_score, rank, total_time, is_positive

def get_f1_score(test_queries, target_query):
    with open("inputs/synthetic/test/{}_inputs.json".format(target_query), 'r') as f:
        inputs = json.load(f)
    with open("inputs/synthetic/test/{}_labels.json".format(target_query), 'r') as f:
        labels = json.load(f)
    inputs = np.asarray(inputs, dtype=object)
    labels = np.asarray(labels, dtype=object)
    inputs, _, labels, _ = train_test_split(inputs, labels, train_size=500, random_state=42, stratify=labels)
    print("labels", labels, sum(labels), len(labels))

    memoize_all_inputs = [LRU(5000) for _ in range(len(inputs))]

    results = []
    for test_query, _ in test_queries:
        program = str_to_program(test_query)
        print_program(program)
        # compute score
        score, memoize_all_inputs = compute_query_score(program, inputs, labels, memoize_all_inputs)
        results.append(score)
    print("scores: ", results)
    return np.mean(results)


def compute_query_score(current_query, inputs, labels, memoize_all_inputs):
    y_pred = []
    for i in range(len(inputs)):
        input = inputs[i]
        label = labels[i]
        memoize = memoize_all_inputs[i]
        result, new_memoize = current_query.execute(input, label, memoize, {})
        y_pred.append(int(result[0, len(input[0])] > 0))
        memoize_all_inputs[i].update(new_memoize)
    score = f1_score(list(labels), y_pred)
    return score, memoize_all_inputs


if __name__ == '__main__':
    _start = time()
    test_synthetic_queries("vocal-sampling", 10, 10, 5, 3, 5, 32, 100, 5, 100, 8)
    print("Time: {}".format(time() - _start))