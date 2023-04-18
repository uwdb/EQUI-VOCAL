import psycopg2 as psycopg
from psycopg2 import pool
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from lru import LRU
import time
from src.utils import str_to_program_postgres, postgres_execute, postgres_execute_cache_sequence, postgres_execute_no_caching
import math
import argparse
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

def evaluate_vocal(dataset_name, input_dir, output_dir, method, query_str, sampling_rate, port, multithread, task_name, value, **kwargs):
    def compute_f1_score(test_query):
        test_program = str_to_program_postgres(test_query)
        conn = connections.getconn()
        outputs, new_memo = postgres_execute_cache_sequence(conn, test_program, memo, inputs_table_name, input_vids, is_trajectory=is_trajectory, sampling_rate=sampling_rate)
        connections.putconn(conn)
        if lock:
            lock.acquire()
        for i, memo_dict in enumerate(new_memo):
            for k, v in memo_dict.items():
                memo[i][k] = v
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

    dsn = "dbname=myinner_db user=enhaoz host=localhost port={}".format(port)
    connections = psycopg.pool.ThreadedConnectionPool(1, multithread, dsn)
    if multithread > 1:
        executor = ThreadPoolExecutor(max_workers=multithread)
        m = multiprocessing.Manager()
        lock = m.Lock()
    elif multithread == 1:
        lock = None

    # Configure budget values
    if "budget" in kwargs:
        budgets = [kwargs["budget"]]
    else:
        if task_name in ["trajectory"]:
            budgets = list(range(12, 21)) + list(range(25, 51, 5))
        elif task_name == "warsaw":
            budgets = [15, 20, 25, 30, 40, 50]
        elif task_name in ["lambda_trajectory"]:
            budgets = [20]
        elif task_name in ["lambda_scene_graph", "cpu"]:
            budgets = [50]
        elif task_name in ["num_init", "bw", "k", "cpu"]:
            budgets = [100]
        elif task_name in ["budget"]:
            budgets = [30, 50, 100]
        else:
            raise ValueError("Unknown task")

    # Configure run_id
    if "run_id" in kwargs:
        run_id_list = [kwargs["run_id"]]
    else:
        if task_name in ["trajectory", "warsaw"]:
            run_id_list = list(range(20))
        elif task_name in ["num_init", "bw", "k", "lambda_scene_graph", "lambda_trajectory", "cpu", "budget"]:
            run_id_list = list(range(5))
        else:
            raise ValueError("Unknown task")

    list_size = 72159
    memo = [{} for _ in range(list_size)]
    # Read the test data
    test_dir = os.path.join(input_dir, dataset_name, "test")
    inputs_filename = query_str + "_inputs.json"
    labels_filename = query_str + "_labels.json"
    with open(os.path.join(test_dir, inputs_filename), 'r') as f:
        input_vids = json.load(f)
    with open(os.path.join(test_dir, labels_filename), 'r') as f:
        labels = json.load(f)

    score_median_log = []
    score_random_log = []
    runtime_log = []
    for run in run_id_list:
        score_median_log_per_run = []
        score_random_log_per_run = []
        runtime_log_per_run = []
        for budget in budgets:
            if task_name == "trajectory":
                log_dir = os.path.join(output_dir, dataset_name, method, "nip_2-nin_10-npred_5-depth_3-max_d_1-nvars_2-bw_10-pool_size_100-k_100-budget_{}-thread_1-lru_None-lambda_0.01".format(budget))
            elif task_name == "warsaw":
                log_dir = os.path.join(output_dir, dataset_name, method, "nip_2-nin_10-npred_5-depth_3-max_d_1-nvars_2-bw_10-pool_size_100-n_sampled_videos_500-k_100-budget_{}-thread_1-lru_None-lambda_0.01".format(budget))
            elif task_name == "budget":
                log_dir = os.path.join(output_dir, dataset_name, method, "nip_15-nin_15-npred_7-depth_3-max_d_15-nvars_3-bw_10-pool_size_100-k_1000-budget_{}-thread_4-lru_None-lambda_0.001".format(budget))
            elif task_name == "num_init":
                log_dir = os.path.join(output_dir, dataset_name, method, "nip_{}-nin_{}-npred_7-depth_3-max_d_15-nvars_3-bw_10-pool_size_100-k_1000-budget_{}-thread_4-lru_None-lambda_0.001".format(init, init, budget))
            elif task_name == "bw":
                log_dir = os.path.join(output_dir, dataset_name, method, "nip_15-nin_15-npred_7-depth_3-max_d_15-nvars_3-bw_{}-pool_size_100-k_1000-budget_100-thread_4-lru_None-lambda_0.001".format(bw))
            elif task_name == "k":
                log_dir = os.path.join(output_dir, dataset_name, method, "nip_15-nin_15-npred_7-depth_3-max_d_15-nvars_3-bw_10-pool_size_100-k_{}-budget_100-thread_4-lru_None-lambda_0.001".format(k_value))
            elif task_name == "reg_lambda" and "scene_graph" in dataset_name:
                log_dir = os.path.join(output_dir, dataset_name, method, "nip_15-nin_15-npred_7-depth_3-max_d_15-nvars_3-bw_10-pool_size_100-k_1000-budget_{}-thread_4-lru_None-lambda_{}".format(budget, reg_lambda))
            elif task_name == "reg_lambda" and not "scene_graph" in dataset_name:
                log_dir = os.path.join(output_dir, dataset_name, method, "nip_2-nin_10-npred_5-depth_3-max_d_15-nvars_2-bw_10-pool_size_100-k_100-budget_{}-thread_4-lru_None-lambda_{}".format(budget, reg_lambda))
            elif task_name == "cpu":
                log_dir = os.path.join(output_dir, dataset_name, method, "nip_15-nin_15-npred_7-depth_3-max_d_15-nvars_3-bw_10-pool_size_100-k_1000-budget_50-thread_{}-lru_None-lambda_0.001".format(value))

            # output_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/{}/{}/nip_2-nin_10-npred_5-depth_3-max_d_1-nvars_2-bw_10-pool_size_100-k_100-budget_{}-thread_1-lru_None-lambda_0.01".format(dataset_name, method, budget) # trajectory pairs dataset
            # output_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/{}/{}/nip_15-nin_15-npred_7-depth_3-max_d_15-nvars_3-bw_10-pool_size_100-k_1000-budget_{}-thread_4-lru_None-lambda_0.001".format(dataset_name, method, budget) # vary budget
            # output_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/{}/{}/nip_{}-nin_{}-npred_7-depth_3-max_d_15-nvars_3-bw_10-pool_size_100-k_1000-budget_{}-thread_4-lru_None-lambda_0.001".format(dataset_name, method, init, init, budget) # vary init
            # output_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/{}/{}/nip_15-nin_15-npred_7-depth_3-max_d_15-nvars_3-bw_{}-pool_size_100-k_1000-budget_100-thread_4-lru_None-lambda_0.001".format(dataset_name, method, bw) # vary bw
            # output_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/{}/{}/nip_15-nin_15-npred_7-depth_3-max_d_15-nvars_3-bw_10-pool_size_100-k_{}-budget_100-thread_4-lru_None-lambda_0.001".format(dataset_name, method, k_value) # vary k_value
            # output_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/{}/{}/nip_15-nin_15-npred_7-depth_3-max_d_15-nvars_3-bw_10-pool_size_100-k_1000-budget_{}-thread_4-lru_None-lambda_{}".format(dataset_name, method, budget, reg_lambda) # vary lambda (scene graphs)
            # output_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/{}/{}/nip_2-nin_10-npred_5-depth_3-max_d_15-nvars_2-bw_10-pool_size_100-k_100-budget_{}-thread_4-lru_None-lambda_{}".format(dataset_name, method, budget, reg_lambda) # vary lambda (trajectory)
            # "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/{}/{}/nip_15-nin_15-npred_7-depth_3-max_d_15-nvars_3-bw_10-pool_size_100-k_1000-budget_50-thread_{}-lru_None-lambda_0.001".format(dataset_name, method, cpu_value) # vary cpu

            try:
                # Read the log file
                with open(os.path.join(log_dir, "{}-{}.log".format(query_str, run)), 'r') as f:
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
                print("returned_queries", len(returned_queries))
                print("run: {}, budget: {}".format(run, budget))

                if dataset_name.startswith("collision"):
                    inputs_table_name = "Obj_collision"
                    is_trajectory = True
                elif "scene_graph" in dataset_name:
                    inputs_table_name = "Obj_clevrer"
                    is_trajectory = False
                elif dataset_name == "warsaw":
                    inputs_table_name = "Obj_warsaw"
                    is_trajectory = True
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
    exp_dir = os.path.join(output_dir, dataset_name)

    if task_name == "num_init":
        if not os.path.exists(os.path.join(exp_dir, "stats", method+"-init_{}-budget_{}".format(init, budget))):
            os.makedirs(os.path.join(exp_dir, "stats", method+"-init_{}-budget_{}".format(init, budget)), exist_ok=True)
        with open(os.path.join(exp_dir, "stats", method+"-init_{}-budget_{}".format(init, budget), "{}.json".format(query_str)), "w") as f:
            json.dump(out_dict, f)
    elif task_name == "bw":
        if not os.path.exists(os.path.join(exp_dir, "stats", method+"-bw_{}".format(bw))):
            os.makedirs(os.path.join(exp_dir, "stats", method+"-bw_{}".format(bw)), exist_ok=True)
        with open(os.path.join(exp_dir, "stats", method+"-bw_{}".format(bw), "{}.json".format(query_str)), "w") as f:
            json.dump(out_dict, f)
    elif task_name == "cpu":
        if not os.path.exists(os.path.join(exp_dir, "stats", method+"-cpu_{}-budget_50".format(value))):
            os.makedirs(os.path.join(exp_dir, "stats", method+"-cpu_{}-budget_50".format(value)), exist_ok=True)
        with open(os.path.join(exp_dir, "stats", method+"-cpu_{}-budget_50".format(value), "{}.json".format(query_str)), "w") as f:
            json.dump(out_dict, f)
    elif task_name == "k":
        if not os.path.exists(os.path.join(exp_dir, "stats", method+"-k_{}".format(k_value))):
            os.makedirs(os.path.join(exp_dir, "stats", method+"-k_{}".format(k_value)), exist_ok=True)
        with open(os.path.join(exp_dir, "stats", method+"-k_{}".format(k_value), "{}.json".format(query_str)), "w") as f:
            json.dump(out_dict, f)
    elif task_name == "lambda_scene_graph":
        if not os.path.exists(os.path.join(exp_dir, "stats", method+"-lambda_{}-budget_{}".format(reg_lambda, budgets[0]))):
            os.makedirs(os.path.join(exp_dir, "stats", method+"-lambda_{}-budget_{}".format(reg_lambda, budgets[0])), exist_ok=True)
        with open(os.path.join(exp_dir, "stats", method+"-lambda_{}-budget_{}".format(reg_lambda, budgets[0]), "{}.json".format(query_str)), "w") as f:
            json.dump(out_dict, f)
    elif task_name == "lambda_trajectory":
        if not os.path.exists(os.path.join(exp_dir, "stats", method+"-lambda_{}-budget_{}".format(reg_lambda, budgets[0]))):
            os.makedirs(os.path.join(exp_dir, "stats", method+"-lambda_{}-budget_{}".format(reg_lambda, budgets[0])), exist_ok=True)
        with open(os.path.join(exp_dir, "stats", method+"-lambda_{}-budget_{}".format(reg_lambda, budgets[0]), "{}-{}.json".format(query_str, run_id[0])), "w") as f:
            json.dump(out_dict, f)
    elif task_name in ["budget", "trajectory"]:
        if not os.path.exists(os.path.join(exp_dir, "stats", method)):
            os.makedirs(os.path.join(exp_dir, "stats", method), exist_ok=True)
        with open(os.path.join(exp_dir, "stats", method, "{}.json".format(query_str)), "w") as f:
            json.dump(out_dict, f)
    elif task_name == "warsaw":
        if not os.path.exists(os.path.join(exp_dir, "stats", method + "-max_d_1-n_sampled_videos_500")):
            os.makedirs(os.path.join(exp_dir, "stats", method + "-max_d_1-n_sampled_videos_500"), exist_ok=True)
        with open(os.path.join(exp_dir, "stats", method + "-max_d_1-n_sampled_videos_500", "{}.json".format(query_str)), "w") as f:
            json.dump(out_dict, f)

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
    # if not os.path.exists(os.path.join(exp_dir, "stats", method+"-cpu_{}-budget_50".format(value))):
    #     os.makedirs(os.path.join(exp_dir, "stats", method+"-cpu_{}-budget_50".format(value)), exist_ok=True)
    # with open(os.path.join(exp_dir, "stats", method+"-cpu_{}-budget_50".format(value), "{}.json".format(query_str)), "w") as f:
    #     json.dump(out_dict, f)

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


def evaluate_vocal_no_params(dataset_name, input_dir, output_dir, config_name, method, query_str, sampling_rate, port, multithread):
    def compute_f1_score(test_query):
        test_program = str_to_program_postgres(test_query)
        conn = connections.getconn()
        outputs, new_memo = postgres_execute_cache_sequence(conn, test_program, memo, inputs_table_name, input_vids, is_trajectory=is_trajectory, sampling_rate=sampling_rate)
        connections.putconn(conn)

        if lock:
            lock.acquire()
        for i, memo_dict in enumerate(new_memo):
            for k, v in memo_dict.items():
                memo[i][k] = v
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

    dsn = "dbname=myinner_db user=enhaoz host=localhost port={}".format(port)
    connections = psycopg.pool.ThreadedConnectionPool(1, multithread, dsn)

    if multithread > 1:
        executor = ThreadPoolExecutor(max_workers=multithread)
        m = multiprocessing.Manager()
        lock = m.Lock()
    elif multithread == 1:
        lock = None

    list_size = 72159
    memo = [{} for _ in range(list_size)]
    # Read the test data
    test_dir = os.path.join(input_dir, dataset_name, "test")
    inputs_filename = query_str + "_inputs.json"
    labels_filename = query_str + "_labels.json"
    with open(os.path.join(test_dir, inputs_filename), 'r') as f:
        input_vids = json.load(f)
    with open(os.path.join(test_dir, labels_filename), 'r') as f:
        labels = json.load(f)

    log_dir = os.path.join(output_dir, dataset_name, method, config_name)
    score_median_log = []
    score_random_log = []
    runtime_log = []
    for run in range(20):
        try:
            with open(os.path.join(log_dir, "{}-{}.log".format(query_str, run)), 'r') as f:
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
            print("returned_queries", len(returned_queries))

            if dataset_name.startswith("collision"):
                inputs_table_name = "Obj_collision"
                is_trajectory = True
            elif "scene_graph" in dataset_name:
                inputs_table_name = "Obj_clevrer"
                is_trajectory = False
            elif dataset_name == "warsaw":
                inputs_table_name = "Obj_warsaw"
                is_trajectory = True
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
            score_median_log.append(median_f1)
            score_random_log.append(score_random)
            runtime_log.append(runtime)
        except Exception as e:
            print(e)
            score_median_log.append(-1)
            score_random_log.append(-1)
            runtime_log.append(-1)
    out_dict = {"score_median": score_median_log, "score_random": score_random_log, "runtime": runtime_log}
    if not os.path.exists(os.path.join(output_dir, dataset_name, "stats", method, config_name)):
        os.makedirs(os.path.join(output_dir, dataset_name, "stats", method, config_name), exist_ok=True)
    with open(os.path.join(output_dir, dataset_name, "stats", method, config_name, "{}.json".format(query_str)), "w") as f:
        json.dump(out_dict, f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", type=str, help='Dataset to evaluate.', choices=['synthetic_scene_graph_easy', 'synthetic_scene_graph_medium', 'synthetic_scene_graph_hard', 'without_duration-sampling_rate_4', 'trajectories_duration', 'trajectories_handwritten', 'user_study_queries_scene_graph', 'warsaw', 'synthetic_scene_graph_hard_v2'])
    ap.add_argument("--config_name", type=str, default="", help='Config directory name.')
    ap.add_argument("--query_str", type=str, help='Target query to evalaute, written in the compact notation.')
    ap.add_argument("--method", type=str, help='Query synthesis method.')
    ap.add_argument("--port", type=int, default=5432, help='Port on which Postgres is to listen.')
    ap.add_argument("--multithread", type=int, default=1, help='Number of CPUs to use.')
    ap.add_argument("--budget", type=int, help='Labeling budget.')
    ap.add_argument("--task_name", type=str, default='no_params', help='Task name, e.g., the name of the tested hyperparameter.', choices=['trajectory', 'budget', 'bw', 'k', 'num_init', 'cpu', 'reg_lambda', 'no_params', 'warsaw'])
    ap.add_argument("--value", type=int, help='Value of the tested hyperparameter. If specified, evaluate on the single value; otherwise, evaluate on all values tested in our experiment.')
    ap.add_argument("--run_id", type=int, help='Run ID.')
    ap.add_argument('--input_dir', type=str, default="/gscratch/balazinska/enhaoz/complex_event_video/inputs", help='Input directory.')
    ap.add_argument('--output_dir', type=str, default="/gscratch/balazinska/enhaoz/complex_event_video/outputs", help='Output directory.')

    args = ap.parse_args()
    dataset_name = args.dataset_name
    query_str = args.query_str
    method = args.method
    port = args.port
    multithread = args.multithread
    value = args.value
    budget = args.budget
    run_id = args.run_id
    task_name = args.task_name
    input_dir = args.input_dir
    output_dir = args.output_dir
    config_name = args.config_name

    if "sampling_rate" in dataset_name:
        splits = dataset_name.split("-")
        for split in splits:
            if split.startswith("sampling_rate_"):
                sampling_rate = int(split.replace("sampling_rate_", ""))
                break
    else:
        sampling_rate = None
    print("sampling_rate: ", sampling_rate)

    #### trajectory ####
    if task_name == "trajectory":
        evaluate_vocal(dataset_name, input_dir, output_dir, method, query_str, sampling_rate, port, multithread, "trajectory", None)

    #### warsaw ####
    if task_name == "warsaw":
        evaluate_vocal(dataset_name, input_dir, output_dir, method, query_str, sampling_rate, port, multithread, "warsaw", None)

    #### vary budget ####
    elif task_name == "budget":
        evaluate_vocal(dataset_name, input_dir, output_dir, method, query_str, sampling_rate, port, multithread, "budget", None)

    #### vary bw ####
    elif task_name == "bw":
        if value:
            bw_values = [value]
        else:
            bw_values = [1, 5, 10, 15, 20]
        for bw in bw_values:
            evaluate_vocal(dataset_name, input_dir, output_dir, method, query_str, sampling_rate, port, multithread, "bw", bw)

    #### vary k ####
    elif task_name == "k":
        if value:
            k_values = [value]
        else:
            k_values = [1, 10, 100, 1000]
        for k_value in k_values:
            evaluate_vocal(dataset_name, input_dir, output_dir, method, query_str, sampling_rate, port, multithread, "k", k_value)

    #### vary num_init ####
    elif task_name == "num_init":
        if value:
            init_examples = [value]
        else:
            init_examples = [5, 10, 15, 20, 25]
        for init in init_examples:
            evaluate_vocal(dataset_name, input_dir, output_dir, method, query_str, sampling_rate, port, multithread, "num_init", init)

    #### vary cpu ####
    elif task_name == "cpu":
        if value:
            cpu_values = [value]
        else:
            cpu_values = [1, 2, 4, 8]
        for cpu_value in cpu_values:
            evaluate_vocal(dataset_name, input_dir, output_dir, method, query_str, sampling_rate, port, multithread, "cpu", cpu_value)

    #### vary reg_lambda_scene_graph ####
    elif task_name == "reg_lambda" and "scene_graph" in dataset_name:
        if value:
            reg_lambdas = [value]
        else:
            reg_lambdas = [0.0, 0.001, 0.01]
        for reg_lambda in reg_lambdas:
            evaluate_vocal(dataset_name, input_dir, output_dir, method, query_str, sampling_rate, port, multithread, "lambda_scene_graph", reg_lambda)

    ### vary reg_lambda_trajectory ####
    elif task_name == "reg_lambda" and not "scene_graph" in dataset_name:
        if value:
            reg_lambdas = [value]
        else:
            reg_lambdas = [0.0, 0.001, 0.01]
        for reg_lambda in reg_lambdas:
            evaluate_vocal(dataset_name, input_dir, output_dir, method, query_str, sampling_rate, port, multithread, "lambda_trajectory", reg_lambda, budget=budget, run_id=run_id)

    ### None of the above. Evaluate every trial separately ####
    else:
        evaluate_vocal_no_params(dataset_name, input_dir, output_dir, config_name, method, query_str, sampling_rate, port, multithread)
