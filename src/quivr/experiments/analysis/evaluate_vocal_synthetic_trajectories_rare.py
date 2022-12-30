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
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

def evaluate_vocal(dataset_name, method, query_str, sampling_rate, port, multithread, cpu_value):
    def compute_f1_score(test_query):
        test_program = str_to_program_postgres(test_query)
        outputs, new_memoize_scene_graph, new_memoize_sequence = postgres_execute_cache_sequence(dsn, test_program, memoize_scene_graph, memoize_sequence, inputs_table_name, input_vids, is_trajectory=is_trajectory, sampling_rate=sampling_rate)

        if lock:
            lock.acquire()
        for i, memo_dict in enumerate(new_memoize_scene_graph):
            for k, v in memo_dict.items():
                memoize_scene_graph[i][k] = v
        for i, memo_dict in enumerate(new_memoize_sequence):
            for k, v in memo_dict.items():
                memoize_sequence[i][k] = v
        if lock:
            lock.release()
        preds = []
        for input in input_vids:
            if input in outputs:
                preds.append(1)
            else:
                preds.append(0)
        score = f1_score(labels, preds)
        print(score)
        return score

    if multithread > 1:
        executor = ThreadPoolExecutor(max_workers=multithread)
        m = multiprocessing.Manager()
        lock = m.Lock()
    elif multithread == 1:
        lock = None

    # budgets = list(range(12, 21)) + list(range(25, 51, 5))
    budgets = [50]
    # budgets = [30, 50, 100]
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
    for run in range(5):
    # for run in range(20):
    # for run in run_id:
    # for run in [10, 11]:
        score_median_log_per_run = []
        score_random_log_per_run = []
        runtime_log_per_run = []
        for budget in budgets:
        # for init_example in init_examples:
            # output_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/{}/{}/nip_2-nin_10-npred_5-depth_3-max_d_1-nvars_2-bw_10-pool_size_100-k_100-budget_{}-thread_1-lru_None-lambda_0.01".format(dataset_name, method, budget) # trajectory pairs dataset
            # output_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/{}/{}/nip_15-nin_15-npred_7-depth_3-max_d_15-nvars_3-bw_10-pool_size_100-k_1000-budget_{}-thread_4-lru_None-lambda_0.001".format(dataset_name, method, budget) # vary budget
            # output_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/{}/{}/nip_{}-nin_{}-npred_7-depth_3-max_d_15-nvars_3-bw_10-pool_size_100-k_1000-budget_{}-thread_4-lru_None-lambda_0.001".format(dataset_name, method, init, init, budget) # vary init
            # output_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/{}/{}/nip_15-nin_15-npred_7-depth_3-max_d_15-nvars_3-bw_{}-pool_size_100-k_1000-budget_100-thread_4-lru_None-lambda_0.001".format(dataset_name, method, bw) # vary bw
            # output_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/{}/{}/nip_15-nin_15-npred_7-depth_3-max_d_15-nvars_3-bw_10-pool_size_100-k_{}-budget_100-thread_4-lru_None-lambda_0.001".format(dataset_name, method, k_value) # vary k_value
            # output_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/{}/{}/nip_15-nin_15-npred_7-depth_3-max_d_15-nvars_3-bw_10-pool_size_100-k_1000-budget_{}-thread_4-lru_None-lambda_{}".format(dataset_name, method, budget, reg_lambda) # vary lambda (scene graphs)
            # output_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/{}/{}/nip_2-nin_10-npred_5-depth_3-max_d_15-nvars_2-bw_10-pool_size_100-k_100-budget_{}-thread_4-lru_None-lambda_{}".format(dataset_name, method, budget, reg_lambda) # vary lambda (trajectory)
            output_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/{}/{}/nip_15-nin_15-npred_7-depth_3-max_d_15-nvars_3-bw_10-pool_size_100-k_1000-budget_50-thread_{}-lru_None-lambda_0.001".format(dataset_name, method, cpu_value) # vary cpu

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

                f1_scores = []
                if multithread > 1:
                    f1_scores = []
                    for result in executor.map(compute_f1_score, returned_queries):
                        f1_scores.append(result)
                else:
                    for test_query in returned_queries:
                        score = compute_f1_score(test_query)
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
    #### vary init ####
    # if not os.path.exists(os.path.join(exp_dir, "stats", method+"-init_{}-budget_{}".format(init, budget))):
    #     os.makedirs(os.path.join(exp_dir, "stats", method+"-init_{}-budget_{}".format(init, budget)), exist_ok=True)
    # with open(os.path.join(exp_dir, "stats", method+"-init_{}-budget_{}".format(init, budget), "{}.json".format(query_str)), "w") as f:
    #     json.dump(out_dict, f)

    #### vary bw ####
    # if not os.path.exists(os.path.join(exp_dir, "stats", method+"-bw_{}".format(bw))):
    #     os.makedirs(os.path.join(exp_dir, "stats", method+"-bw_{}".format(bw)), exist_ok=True)
    # with open(os.path.join(exp_dir, "stats", method+"-bw_{}".format(bw), "{}.json".format(query_str)), "w") as f:
    #     json.dump(out_dict, f)

    ### vary cpu ####
    if not os.path.exists(os.path.join(exp_dir, "stats", method+"-cpu_{}-budget_50".format(cpu_value))):
        os.makedirs(os.path.join(exp_dir, "stats", method+"-cpu_{}-budget_50".format(cpu_value)), exist_ok=True)
    with open(os.path.join(exp_dir, "stats", method+"-cpu_{}-budget_50".format(cpu_value), "{}.json".format(query_str)), "w") as f:
        json.dump(out_dict, f)

    #### vary k_value ####
    # if not os.path.exists(os.path.join(exp_dir, "stats", method+"-k_{}".format(k_value))):
    #     os.makedirs(os.path.join(exp_dir, "stats", method+"-k_{}".format(k_value)), exist_ok=True)
    # with open(os.path.join(exp_dir, "stats", method+"-k_{}".format(k_value), "{}.json".format(query_str)), "w") as f:
    #     json.dump(out_dict, f)

    #### vary lambda (scene graphs) ####
    # if not os.path.exists(os.path.join(exp_dir, "stats", method+"-lambda_{}-budget_{}".format(reg_lambda, budgets[0]))):
    #     os.makedirs(os.path.join(exp_dir, "stats", method+"-lambda_{}-budget_{}".format(reg_lambda, budgets[0])), exist_ok=True)
    # with open(os.path.join(exp_dir, "stats", method+"-lambda_{}-budget_{}".format(reg_lambda, budgets[0]), "{}.json".format(query_str)), "w") as f:
    #     json.dump(out_dict, f)

    #### vary lambda (trajectories) ####
    # if not os.path.exists(os.path.join(exp_dir, "stats", method+"-lambda_{}-budget_{}".format(reg_lambda, budgets[0]))):
    #     os.makedirs(os.path.join(exp_dir, "stats", method+"-lambda_{}-budget_{}".format(reg_lambda, budgets[0])), exist_ok=True)
    # with open(os.path.join(exp_dir, "stats", method+"-lambda_{}-budget_{}".format(reg_lambda, budgets[0]), "{}-{}.json".format(query_str, run_id[0])), "w") as f:
    #     json.dump(out_dict, f)

    #### vary budget ####
    # if not os.path.exists(os.path.join(exp_dir, "stats", method)):
    #     os.makedirs(os.path.join(exp_dir, "stats", method), exist_ok=True)
    # with open(os.path.join(exp_dir, "stats", method, "{}.json".format(query_str)), "w") as f:
    #     json.dump(out_dict, f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", type=str)
    ap.add_argument("--query_str", type=str)
    ap.add_argument("--method", type=str)
    ap.add_argument("--port", type=int, default=5432)
    ap.add_argument("--multithread", type=int, default=1)
    ap.add_argument("--budget", type=int)
    ap.add_argument("--run_id", type=int)

    args = ap.parse_args()
    dataset_name = args.dataset_name
    query_str = args.query_str
    method = args.method
    port = args.port
    multithread = args.multithread
    budget = args.budget
    run_id = args.run_id

    if "sampling_rate" in dataset_name:
        splits = dataset_name.split("-")
        for split in splits:
            if split.startswith("sampling_rate_"):
                sampling_rate = int(split.replace("sampling_rate_", ""))
                break
    else:
        sampling_rate = None
    print("sampling_rate: ", sampling_rate)
    #### vary budget ####
    # evaluate_vocal(dataset_name, method, query_str, sampling_rate, port, multithread)

    #### vary bw ####
    # bw_values = [1, 5, 10, 15, 20]
    # for bw in bw_values:
    #     evaluate_vocal(dataset_name, method, query_str, sampling_rate, port, multithread, bw)

    #### vary k ####
    # k_values = [1, 10, 100, 1000]
    # for k_value in k_values:
    #     evaluate_vocal(dataset_name, method, query_str, sampling_rate, port, multithread, k_value)

    #### vary init ####
    # init_examples = [5, 10, 15, 20, 25]
    # for init in init_examples:
    #     evaluate_vocal(dataset_name, method, query_str, sampling_rate, port, multithread, init)

    #### vary cpu ####
    # cpu_values = [1, 2, 4, 8]
    cpu_values = [1]
    for cpu_value in cpu_values:
        evaluate_vocal(dataset_name, method, query_str, sampling_rate, port, multithread, cpu_value)

    #### vary reg_lambda ####
    # reg_lambdas = [0.0]
    # reg_lambdas = [0.0, 0.001, 0.005, 0.01, 0.05]
    # for reg_lambda in reg_lambdas:
    #     evaluate_vocal(dataset_name, method, query_str, sampling_rate, port, multithread, reg_lambda)

    #### vary reg_lambda ####
    # reg_lambda = 0.01
    # evaluate_vocal(dataset_name, method, query_str, sampling_rate, port, multithread, reg_lambda, [budget], [run_id])
