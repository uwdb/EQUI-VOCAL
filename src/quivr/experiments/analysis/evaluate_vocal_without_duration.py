import psycopg2 as psycopg
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from lru import LRU
import time
from quivr.utils import str_to_program_postgres, postgres_execute, postgres_execute_cache_sequence, postgres_execute_no_caching
import math
import argparse

def evaluate_vocal(dataset_name, method, query_str, sampling_rate, port, beam_width):
    budgets = list(range(12, 21)) + list(range(25, 51, 5))

    memoize_scene_graph = [LRU(10000) for _ in range(10080)]
    memoize_sequence = [LRU(10000) for _ in range(10080)]
    # Read the test data
    test_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/{}/test".format(dataset_name)
    inputs_filename = query_str + "_inputs.json"
    labels_filename = query_str + "_labels.json"
    # n_examples = int(filename[11:-12])
    with open(os.path.join(test_dir, inputs_filename), 'r') as f:
        input_vids = json.load(f)
    with open(os.path.join(test_dir, labels_filename), 'r') as f:
        labels = json.load(f)

    score_log = []
    runtime_log = []
    for run in range(20):
        score_log_per_run = []
        runtime_log_per_run = []
        for budget in budgets:
            # output_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/{}/{}/nip_2-nin_10-npred_5-depth_3-max_d_5-nvars_2-bw_10-pool_size_100-k_100-budget_{}-thread_1-lru_None".format(dataset_name, method, budget)
            output_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/{}/{}/nip_2-nin_10-npred_4-depth_3-max_d_1-nvars_2-bw_{}-pool_size_100-k_100-budget_{}-thread_1-lru_None".format(dataset_name, method, beam_width, budget)
            # Read the log file
            with open(os.path.join(output_dir, "{}-{}.log".format(query_str, run)), 'r') as f:
                lines = f.readlines()
                lines = [line.rstrip() for line in lines]
            step = 0
            for line in lines:
                if line == "[Step {}]".format(step):
                    returned_queries = []
                    continue
                elif line.startswith("[Runtime so far]"):
                    runtime = float(line.replace("[Runtime so far] ", ""))
                    step += 1
                elif line.startswith("[Final answers]"):
                    break
                else:
                    line = line.split("'")[1]
                    returned_queries.append(line)
            print("returned_queries", len(returned_queries))
            print("run: {}, budget: {}".format(run, budget))

            f1_scores = []
            for test_query in returned_queries:
                test_program = str_to_program_postgres(test_query)

                dsn = "dbname=myinner_db user=enhaoz host=localhost port={}".format(port)

                inputs_table_name = "Obj_trajectories"
                _start = time.time()
                outputs, new_memoize_scene_graph, new_memoize_sequence = postgres_execute_cache_sequence(dsn, test_program, memoize_scene_graph, memoize_sequence, inputs_table_name, input_vids, is_trajectory=True, sampling_rate=sampling_rate)

                for i, memo_dict in enumerate(new_memoize_scene_graph):
                    for k, v in memo_dict.items():
                        memoize_scene_graph[i][k] = v
                for i, memo_dict in enumerate(new_memoize_sequence):
                    for k, v in memo_dict.items():
                        memoize_sequence[i][k] = v

                preds = []
                for input in input_vids:
                    if input in outputs:
                        preds.append(1)
                    else:
                        preds.append(0)
                score = f1_score(labels, preds)
                print(score)
                f1_scores.append(score)
            median_f1 = np.median(f1_scores)
            print("Median F1: ", median_f1)
            score_log_per_run.append(median_f1)
            runtime_log_per_run.append(runtime)
        score_log.append(score_log_per_run)
        runtime_log.append(runtime_log_per_run)
    out_dict = {"score": score_log, "runtime": runtime_log}
    exp_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/{}/".format(dataset_name)
    if not os.path.exists(os.path.join(exp_dir, "stats", "{}-bw_{}".format(method, beam_width))):
        os.makedirs(os.path.join(exp_dir, "stats", "{}-bw_{}".format(method, beam_width)))
    with open(os.path.join(exp_dir, "stats", "{}-bw_{}".format(method, beam_width), "{}.json".format(query_str)), "w") as f:
        json.dump(out_dict, f)


def evaluate_vocal_simplest_queries(method, query_str, sampling_rate, port):
    n_init_pos = 2
    n_init_neg = 10
    config_npred = 4
    config_depth = 3
    max_duration = 1
    budgets = list(range(12, 21)) + list(range(25, 51, 5))

    memoize_scene_graph = [LRU(10000) for _ in range(10080)]
    memoize_sequence = [LRU(10000) for _ in range(10080)]
    # Read the test data
    test_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/without_duration-sampling_rate_{}/test".format(sampling_rate)
    inputs_filename = query_str + "_inputs.json"
    labels_filename = query_str + "_labels.json"
    # n_examples = int(filename[11:-12])
    with open(os.path.join(test_dir, inputs_filename), 'r') as f:
        input_vids = json.load(f)
    with open(os.path.join(test_dir, labels_filename), 'r') as f:
        labels = json.load(f)

    score_log = []
    runtime_log = []
    for run in range(20):
        score_log_per_run = []
        runtime_log_per_run = []
        for budget in budgets:
            samples_per_iter = math.ceil((budget - n_init_pos - n_init_neg) * 1.0 / (config_npred + max_duration * config_depth))
            # print("samples_per_iter", samples_per_iter, "budget", budget, "n_init_pos", n_init_pos, "n_init_neg", n_init_neg, "npred", npred, "depth", depth, "max_duration", max_duration)
            output_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/without_duration-sampling_rate_{}/{}/nip_2-nin_10-npred_4-depth_3-max_d_1-nvars_2-bw_5-pool_size_100-k_100-per_iter_{}-budget_{}-thread_1".format(sampling_rate, method, samples_per_iter, budget)
            # Read the log file
            with open(os.path.join(output_dir, "{}-{}.log".format(query_str, run)), 'r') as f:
                lines = f.readlines()
                lines = [line.rstrip() for line in lines]
            step = 0
            min_npred = 9999
            min_depth = 9999
            for line in lines:
                if line == "[Step {}]".format(step):
                    returned_queries = []
                    continue
                elif line.startswith("[Runtime so far]"):
                    runtime = float(line.replace("[Runtime so far] ", ""))
                    step += 1
                    min_npred = 9999
                    min_depth = 9999
                elif line.startswith("[Final answers]"):
                    break
                else:
                    line = line.split("'")[1]
                    program = str_to_program_postgres(line)
                    depth = len(program)
                    npred = sum([len(dict["scene_graph"]) for dict in program])
                    if npred < min_npred:
                        min_npred = npred
                        min_depth = depth
                        returned_queries = [line]
                    elif npred == min_npred:
                        if depth < min_depth:
                            min_depth = depth
                            returned_queries = [line]
                        elif depth == min_depth:
                            returned_queries.append(line)
            print("returned_queries", len(returned_queries))
            print("run: {}, budget: {}".format(run, budget))

            f1_scores = []
            for test_query in returned_queries:
                test_program = str_to_program_postgres(test_query)

                dsn = "dbname=myinner_db user=enhaoz host=localhost port={}".format(port)


                inputs_table_name = "Obj_trajectories"
                _start = time.time()
                outputs, new_memoize_scene_graph, new_memoize_sequence = postgres_execute_cache_sequence(dsn, test_program, memoize_scene_graph, memoize_sequence, inputs_table_name, input_vids, is_trajectory=True, sampling_rate=sampling_rate)

                for i, memo_dict in enumerate(new_memoize_scene_graph):
                    for k, v in memo_dict.items():
                        memoize_scene_graph[i][k] = v
                for i, memo_dict in enumerate(new_memoize_sequence):
                    for k, v in memo_dict.items():
                        memoize_sequence[i][k] = v

                preds = []
                for input in input_vids:
                    if input in outputs:
                        preds.append(1)
                    else:
                        preds.append(0)
                score = f1_score(labels, preds)
                print(score)
                f1_scores.append(score)
            median_f1 = np.median(f1_scores)
            print("Median F1: ", median_f1)
            score_log_per_run.append(median_f1)
            runtime_log_per_run.append(runtime)
        score_log.append(score_log_per_run)
        runtime_log.append(runtime_log_per_run)
    out_dict = {"score": score_log, "runtime": runtime_log}
    exp_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/without_duration-sampling_rate_{}/".format(sampling_rate)
    if not os.path.exists(os.path.join(exp_dir, "stats", method, "simplest_queries")):
        os.makedirs(os.path.join(exp_dir, "stats", method, "simplest_queries"))
    with open(os.path.join(exp_dir, "stats", method, "simplest_queries", "{}.json".format(query_str)), "w") as f:
        json.dump(out_dict, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", type=str)
    ap.add_argument("--query_str", type=str)
    ap.add_argument("--method", type=str)
    ap.add_argument("--beam_width", type=int)
    ap.add_argument("--port", type=int, default=5432)

    args = ap.parse_args()
    dataset_name = args.dataset_name
    query_str = args.query_str
    method = args.method
    beam_width = args.beam_width
    port = args.port

    if "sampling_rate" in dataset_name:
        splits = dataset_name.split("-")
        for split in splits:
            if split.startswith("sampling_rate_"):
                sampling_rate = int(split.replace("sampling_rate_", ""))
                break
    else:
        sampling_rate = None
    print("sampling_rate: ", sampling_rate)

    evaluate_vocal(dataset_name, method, query_str, sampling_rate, port, beam_width)
    # evaluate_vocal_simplest_queries(method, query_str, sampling_rate, port)