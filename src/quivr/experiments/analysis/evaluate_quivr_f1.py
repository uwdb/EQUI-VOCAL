import psycopg2 as psycopg
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from quivr.utils import str_to_program
from lru import LRU

def evaluate_quivr_f1(target_query, out_file):
    with open(out_file, 'r') as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    returned_queries_all_steps = []
    returned_queries = []
    for line in lines:
        if line.startswith("answer Start"):
            returned_queries.append(line[7:])
        elif line.startswith("step"):
            returned_queries_all_steps.append(returned_queries)
            print("number of returned queries: ", len(returned_queries))
            returned_queries = []
    test_dir = "/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/vary_num_examples-sampling_rate_4/test"
    inputs_filename = target_query + "_inputs.json"
    labels_filename = target_query + "_labels.json"
    # n_examples = int(filename[11:-12])
    with open(os.path.join(test_dir, inputs_filename), 'r') as f:
        inputs = json.load(f)
    with open(os.path.join(test_dir, labels_filename), 'r') as f:
        labels = json.load(f)
    with open("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/trajectory_pairs.json", 'r') as f:
        trajectories = json.load(f)
    trajectories = np.asarray(trajectories, dtype=object)
    labels = np.asarray(labels)
    inputs = trajectories[inputs]

    # Down-sample the trajectory once every 4 frames
    inputs_downsampled = []
    for input in inputs:
        inputs_downsampled.append([input[0][::4], input[1][::4]])
    inputs = inputs_downsampled

    memoize_all_inputs = [LRU(10000) for _ in range(len(inputs))]
    for returned_queries in returned_queries_all_steps:
        prediction_matrix = []
        for i in range(len(inputs)):
            input = inputs[i]
            pred_per_input = []
            for query in returned_queries:
                program = str_to_program(query)
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
        mean_f1 = np.mean(f1_scores)
        print("Mean F1: ", mean_f1)


if __name__ == "__main__":
    query_strs = [
        "Conjunction(Near_1.05(o0, o1), RightQuadrant(o0))",
        "Near_1.05(o0, o1); RightQuadrant(o0)",
        "Duration(Near_1.05(o0, o1), 5)",
        "Conjunction(Near_1.05(o0, o1), RightQuadrant(o0)); Near_1.05(o0, o1)",
        "Duration(Conjunction(Near_1.05(o0, o1), RightQuadrant(o0)), 5)",
        "Duration(Near_1.05(o0, o1), 5); Duration(RightQuadrant(o0), 5)",
        "Duration(Conjunction(Near_1.05(o0, o1), RightQuadrant(o0)), 5); Duration(Near_1.05(o0, o1), 5)",
        ]
    target_query = "Near_1.05(o0, o1); Far_0.9(o0, o1); Near_1.05(o0, o1)"

    out_file = "/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/quivr_original/vary_num_examples-sampling_rate_4/nip_2-nin_10-npred_5-depth_3-max_d_1-thread_1/verbose/Near_1.05(o0, o1); Far_0.9(o0, o1); Near_1.05(o0, o1)-1.log"
    evaluate_quivr_f1(target_query, out_file)