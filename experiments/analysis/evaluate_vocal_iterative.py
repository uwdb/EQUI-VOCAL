import psycopg2 as psycopg
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from lru import LRU
import time
from utils import str_to_program_postgres, postgres_execute
import math

def evaluate_vocal(query_str, sampling_rate):
    n_init_pos = 2
    n_init_neg = 10
    npred = 4
    depth = 3
    max_duration = 1
    budgets = list(range(12, 21)) + list(range(25, 51, 5))

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
            samples_per_iter = math.ceil((budget - n_init_pos - n_init_neg) * 1.0 / (npred + max_duration * depth))
            output_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/without_duration-sampling_rate_{}/vocal_postgres-topk-test/nip_2-nin_10-npred_4-depth_3-max_d_1-nvars_2-bw_5-pool_size_100-k_100-per_iter_{}-budget_{}-thread_1".format(sampling_rate, samples_per_iter, budget)
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


            f1_scores = []
            for test_query in returned_queries:
                test_program = str_to_program_postgres(test_query)

                dsn = "dbname=myinner_db user=enhaoz host=localhost"

                memoize = [LRU(10000) for _ in range(10080)]
                inputs_table_name = "Obj_trajectories"
                _start = time.time()
                outputs, new_memoize = postgres_execute(dsn, test_program, memoize, inputs_table_name, input_vids, is_trajectory=True, sampling_rate=sampling_rate)

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
    if not os.path.exists(os.path.join(exp_dir, "stats", "vocal_postgres-topk-test")):
        os.makedirs(os.path.join(exp_dir, "stats", "vocal_postgres-topk-test"))
    with open(os.path.join(exp_dir, "stats", "vocal_postgres-topk-test", "{}.json".format(query_str)), "w") as f:
        json.dump(out_dict, f)

if __name__ == "__main__":
    sampling_rate = 4
    query_strs = [
        # "Conjunction(Near_1(o0, o1), BottomQuadrant(o0))",
        # "Conjunction(FrontOf(o0, o1), TopQuadrant(o0))",
        # "Near_1(o0, o1); Far_3(o0, o1)",
        "Conjunction(Conjunction(Near_1(o0, o1), LeftQuadrant(o0)), Behind(o0, o1))",
        # "Far_3(o0, o1); Near_1(o0, o1); Far_3(o0, o1)",
        # "Conjunction(Far_3(o0, o1), BottomQuadrant(o0)); Near_1(o0, o1)",
        # "Far_3(o0, o1); Conjunction(Near_1(o0, o1), Behind(o0, o1))",
        # "Conjunction(Far_3(o0, o1), LeftQuadrant(o0)); Conjunction(Near_1(o0, o1), LeftQuadrant(o0))",
        # "Far_3(o0, o1); Conjunction(Conjunction(Near_1(o0, o1), LeftQuadrant(o0)), Behind(o0, o1))"
        ]
    for query_str in query_strs:
        evaluate_vocal(query_str, sampling_rate)