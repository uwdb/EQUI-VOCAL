import psycopg2 as psycopg
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from src.utils import dsl_to_program_quivr, get_depth_and_npred
from lru import LRU
import argparse

def evaluate_quivr(dataset_name, method, query_str, run, sampling_rate):
    output_dir = "/gscratch/balazinska/enhaoz/complex_event_video/outputs/{}/{}".format(dataset_name, method)

    test_dir = "/gscratch/balazinska/enhaoz/complex_event_video/inputs/{}/test".format(dataset_name)
    inputs_filename = query_str + "_inputs.json"
    labels_filename = query_str + "_labels.json"
    # n_examples = int(filename[11:-12])
    with open(os.path.join(test_dir, inputs_filename), 'r') as f:
        inputs = json.load(f)
    with open(os.path.join(test_dir, labels_filename), 'r') as f:
        labels = json.load(f)
    if dataset_name.startswith("collision"):
        with open("/gscratch/balazinska/enhaoz/complex_event_video/inputs/collision.json", 'r') as f:
            trajectories = json.load(f)
    elif dataset_name == "warsaw":
        with open("/gscratch/balazinska/enhaoz/complex_event_video/inputs/warsaw_trajectory_pairs.json", 'r') as f:
            trajectories = json.load(f)
    else:
        with open("/gscratch/balazinska/enhaoz/complex_event_video/inputs/trajectory_pairs.json", 'r') as f:
            trajectories = json.load(f)
    trajectories = np.asarray(trajectories, dtype=object)
    labels = np.asarray(labels, dtype=object)
    inputs = trajectories[inputs]

    # Down-sample the trajectory once every sampling_rate frames
    if sampling_rate:
        inputs_downsampled = []
        for input in inputs:
            inputs_downsampled.append([input[0][::sampling_rate], input[1][::sampling_rate]])
        inputs = inputs_downsampled
        inputs = np.asarray(inputs, dtype=object)

    # If more than 5000 trajectories, randomly sample 5000 trajectories for evaluation
    if len(inputs) > 5000:
        sampled_idx = np.random.choice(len(inputs), 5000, replace=False)
        inputs = inputs[sampled_idx]
        labels = labels[sampled_idx]

    memoize_all_inputs = [{} for _ in range(len(inputs))]

    score_median_log_per_run = []
    score_random_log_per_run = []
    runtime_log_per_run = []
    with open(os.path.join(output_dir, "{}-{}.log".format(query_str, run)), 'r') as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    returned_queries_all_steps = []
    runtime_all_steps = []
    memory_all_steps = []
    step = 0
    min_npred = 9999
    min_depth = 9999
    for line in lines:
        if line == "[Step {}]".format(step):
            returned_queries = []
            continue
        elif line.startswith("[# queries]") or line.startswith("[Count candidate queries]") or line.startswith("[Count predictions]"):
            continue
        elif line.startswith("[Runtime so far]"):
            runtime = float(line.replace("[Runtime so far] ", ""))
            runtime_all_steps.append(runtime)
        elif line.startswith("No more uncertainty") or line.startswith("[Runtime]"):
            break
        elif line.startswith("[Memory footprint]"):
            memory = float(line.replace("[Memory footprint] profile: mem=", "").replace(" MB", ""))
            memory_all_steps.append(memory)
            returned_queries_all_steps.append(returned_queries)
            print("number of queries: ", len(returned_queries))
            print(returned_queries)
            step += 1
            min_npred = 9999
            min_depth = 9999
            # If more than 1000 queries are returned, we sample 1000 queries (for clevrer trajectories)
            # If more than 10 queries are returned, we sample 10 queries (for warsaw trajectories)
            if len(returned_queries) > 10:
                returned_queries = np.random.choice(returned_queries, 10, replace=False)
        else:
            program = dsl_to_program_quivr(line)
            depth, num_nontrivial_predicates, num_trivial_predicates = get_depth_and_npred(program)
            npred = num_nontrivial_predicates + num_trivial_predicates
            # print(line, depth, npred)
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

    for returned_queries, runtime in zip(returned_queries_all_steps, runtime_all_steps):
        if len(returned_queries) == 0:
            median_f1 = 0
            score_random = 0
        else:
            prediction_matrix = []
            for i in range(len(inputs)):
                input = inputs[i]
                pred_per_input = []
                for query in returned_queries:
                    program = dsl_to_program_quivr(query)
                    memoize = memoize_all_inputs[i]
                    result, new_memoize = program.execute(input, -1, memoize, {})
                    memoize_all_inputs[i].update(new_memoize)
                    pred_per_input.append(int(result[0, len(input[0])] > 0))
                prediction_matrix.append(pred_per_input)
            prediction_matrix = np.asarray(prediction_matrix)
            f1_scores = []
            for i in range(len(returned_queries)):
                f1 = f1_score(labels, prediction_matrix[:, i])
                f1_scores.append(f1)
            median_f1 = np.median(f1_scores)
            score_random = np.random.choice(f1_scores)
        print("Median F1: ", median_f1)
        print("Random F1: ", score_random)
        score_median_log_per_run.append(median_f1)
        score_random_log_per_run.append(score_random)
        runtime_log_per_run.append(runtime)
    for _ in range(39-len(returned_queries_all_steps)): # labeling budget between 12 to 50
        score_median_log_per_run.append(score_median_log_per_run[-1])
        score_random_log_per_run.append(score_random_log_per_run[-1])
        runtime_log_per_run.append(runtime_log_per_run[-1])

    out_dict = {"score_median": score_median_log_per_run, "score_random": score_random_log_per_run, "runtime": runtime_log_per_run}
    exp_dir = "/gscratch/balazinska/enhaoz/complex_event_video/outputs/{}/".format(dataset_name)
    if not os.path.exists(os.path.join(exp_dir, "stats", method)):
        os.makedirs(os.path.join(exp_dir, "stats", method), exist_ok=True)
    with open(os.path.join(exp_dir, "stats", method, "{}-{}.json".format(query_str, run)), "w") as f:
        json.dump(out_dict, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", type=str)
    ap.add_argument("--method", type=str)
    ap.add_argument("--query_str", type=str)
    ap.add_argument('--run_id', type=int)
    args = ap.parse_args()
    dataset_name = args.dataset_name
    method = args.method
    query_str = args.query_str
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
    evaluate_quivr(dataset_name, method, query_str, run_id, sampling_rate)
    # evaluate_quivr_simplest_queries(dataset_name, method, query_str, run_id, sampling_rate)

# python evaluate_quivr_without_duration.py --dataset_name "synthetic_trajectories_rare-sampling_rate_4" --method "quivr_original_no_kleene/nip_2-nin_10-npred_5-n_nontrivial_None-n_trivial_None-depth_3-max_d_1-thread_1-lru_None" --query_str "Behind(o0, o1); Duration(RightOf(o0, o1), 3); Duration(Conjunction(Conjunction(FrontOf(o0, o1), RightOf(o0, o1)), TopQuadrant(o0)), 2)" --run_id 0