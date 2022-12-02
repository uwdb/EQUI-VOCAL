import psycopg2 as psycopg
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from lru import LRU
import time
from quivr.utils import str_to_program_postgres, postgres_execute, postgres_execute_cache_sequence, postgres_execute_no_caching, complexity_cost
import math
import argparse

def evaluate_vocal(dataset_name, method, query_str, sampling_rate, port, init):
    # budgets = list(range(12, 21)) + list(range(25, 51, 5))
    budgets = [50]
    # budgets = [12, 15, 20, 25, 30, 50]
    if dataset_name.startswith("collision"):
        list_size = 12747
    else:
        list_size = 10080
    memoize_scene_graph = [{} for _ in range(list_size)]
    memoize_sequence = [{} for _ in range(list_size)]
    # Read the test data
    test_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/{}/test".format(dataset_name)
    inputs_filename = query_str + "_inputs.json"
    labels_filename = query_str + "_labels.json"
    # n_examples = int(filename[11:-12])
    with open(os.path.join(test_dir, inputs_filename), 'r') as f:
        input_vids = json.load(f)
    with open(os.path.join(test_dir, labels_filename), 'r') as f:
        labels = json.load(f)

    score_median_log = []
    score_random_log = []
    runtime_log = []
    for run in range(2):
    # for run in [10, 11]:
        score_median_log_per_run = []
        score_random_log_per_run = []
        runtime_log_per_run = []
        for budget in budgets:
        # for init_example in init_examples:
            # output_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/{}/{}/nip_2-nin_10-npred_5-depth_3-max_d_15-nvars_2-bw_10-pool_size_100-k_100-budget_{}-thread_1-lru_None-lambda_0.01".format(dataset_name, method, budget) # trajectory pairs dataset
            # output_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/{}/{}/nip_10-nin_10-npred_7-depth_3-max_d_15-nvars_3-bw_10-pool_size_100-k_100-budget_{}-thread_1-lru_None-lambda_0.01".format(dataset_name, method, budget) # scene graphs dataset
            output_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/{}/{}/nip_{}-nin_{}-npred_7-depth_3-max_d_15-nvars_3-bw_10-pool_size_100-k_100-budget_100-thread_4-lru_None-lambda_0.01".format(dataset_name, method, init, init) # scene graphs dataset
            # output_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/{}/{}/nip_{}-nin_{}-npred_7-depth_3-max_d_15-nvars_3-bw_10-pool_size_100-k_100-budget_{}-thread_1-lru_None-lambda_0.01".format(dataset_name, method, init, init, budget+init+init) # scene graphs dataset

            try:
                # Read the log file
                with open(os.path.join(output_dir, "{}-{}.log".format(query_str, run)), 'r') as f:
                    lines = f.readlines()
                    lines = [line.rstrip() for line in lines]
                step = 0
                min_cost = 9999
                max_score = 0
                for line in lines:
                    if line == "[Step {}]".format(step):
                        returned_queries = []
                        continue
                    elif line.startswith("[Runtime so far]"):
                        runtime = float(line.replace("[Runtime so far] ", ""))
                        step += 1
                        min_cost = 9999
                        max_score = 0
                    elif line.startswith("[Final answers]"):
                        break
                    else:
                        print(line)
                        test_query_str = line.split("'")[1]
                        score = float(line.split("'")[2][1:-1])
                        if score > max_score:
                            max_score = score
                            returned_queries = [test_query_str]
                        elif score == max_score:
                            returned_queries.append(test_query_str)
                        # program = str_to_program_postgres(test_query_str)
                        # c = complexity_cost(program)
                        # if c < min_cost:
                        #     min_cost = c
                        #     returned_queries = [test_query_str]
                        # elif c == min_cost:
                        #     returned_queries.append(test_query_str)
                print("returned_queries", len(returned_queries))
                print("run: {}, budget: {}".format(run, budget))

                f1_scores = []
                for test_query in returned_queries:
                    test_program = str_to_program_postgres(test_query)

                    dsn = "dbname=myinner_db user=enhaoz host=localhost port={}".format(port)
                    if dataset_name.startswith("collision"):
                        inputs_table_name = "Obj_collision"
                        is_trajectory = True
                    elif "scene_graph" in dataset_name:
                        inputs_table_name = "Obj_clevrer"
                        is_trajectory = False
                    else:
                        inputs_table_name = "Obj_trajectories"
                        is_trajectory = True
                    _start = time.time()
                    outputs, new_memoize_scene_graph, new_memoize_sequence = postgres_execute_cache_sequence(dsn, test_program, memoize_scene_graph, memoize_sequence, inputs_table_name, input_vids, is_trajectory=is_trajectory, sampling_rate=sampling_rate)

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
                score_random = np.random.choice(f1_scores)
                print("Median F1: ", median_f1)
                print("Random F1: ", score_random)
                score_median_log_per_run.append(median_f1)
                score_random_log_per_run.append(score_random)
                runtime_log_per_run.append(runtime)
            except Exception as err:
                print(err)
                score_median_log_per_run.append(-1)
                score_random_log_per_run.append(-1)
                runtime_log_per_run.append(-1)
        score_median_log.append(score_median_log_per_run)
        score_random_log.append(score_random_log_per_run)
        runtime_log.append(runtime_log_per_run)
    out_dict = {"score_median": score_median_log, "score_random": score_random_log, "runtime": runtime_log}
    exp_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/{}/".format(dataset_name)
    if not os.path.exists(os.path.join(exp_dir, "stats", method+"-init_{}".format(init))):
        os.makedirs(os.path.join(exp_dir, "stats", method+"-init_{}".format(init)), exist_ok=True)
    with open(os.path.join(exp_dir, "stats", method+"-init_{}".format(init), "{}.json".format(query_str)), "w") as f:
        json.dump(out_dict, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", type=str)
    ap.add_argument("--query_str", type=str)
    ap.add_argument("--method", type=str)
    ap.add_argument("--port", type=int, default=5432)

    args = ap.parse_args()
    dataset_name = args.dataset_name
    query_str = args.query_str
    method = args.method
    port = args.port
    # reg_lambda = args.reg_lambda

    if "sampling_rate" in dataset_name:
        splits = dataset_name.split("-")
        for split in splits:
            if split.startswith("sampling_rate_"):
                sampling_rate = int(split.replace("sampling_rate_", ""))
                break
    else:
        sampling_rate = None
    print("sampling_rate: ", sampling_rate)
    # bw_values = [5, 10, 15, 20]
    # for bw in bw_values:
        # evaluate_vocal(dataset_name, method, query_str, sampling_rate, port, bw)
    # k_values = [10]
    # for k in k_values:
    #     evaluate_vocal(dataset_name, method, query_str, sampling_rate, port, k)
    init_examples = [5, 10, 15, 20, 25]
    for init in init_examples:
        evaluate_vocal(dataset_name, method, query_str, sampling_rate, port, init)
    # cpu_values = [1, 2, 4, 8]
    # for cpu_value in cpu_values:
    #     evaluate_vocal(dataset_name, method, query_str, sampling_rate, port, cpu_value)