import psycopg2 as psycopg
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from lru import LRU
import time
from quivr.utils import str_to_program_postgres, postgres_execute

def evaluate_vocal_f1(target_query, test_queries):
    test_dir = "/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/vary_num_examples-sampling_rate_4/test"
    inputs_filename = target_query + "_inputs.json"
    labels_filename = target_query + "_labels.json"
    # n_examples = int(filename[11:-12])
    with open(os.path.join(test_dir, inputs_filename), 'r') as f:
        input_vids = json.load(f)
    with open(os.path.join(test_dir, labels_filename), 'r') as f:
        labels = json.load(f)

    f1_scores = []
    for test_query in test_queries:
        test_query = str_to_program_postgres(test_query)

        dsn = "dbname=myinner_db user=enhaoz host=localhost"

        memoize = [LRU(10000) for _ in range(10080)]
        inputs_table_name = "Obj_trajectories"
        _start = time.time()
        outputs, new_memoize = postgres_execute(dsn, test_query, memoize, inputs_table_name, input_vids, is_trajectory=True, sampling_rate=4)
        print(len(outputs))
        print(outputs)
        print("time", time.time() - _start)

        preds = []
        for input in input_vids:
            if input in outputs:
                preds.append(1)
            else:
                preds.append(0)
        score = f1_score(labels, preds)
        print(score)
        f1_scores.append(score)
    mean_f1 = np.mean(f1_scores)
    print("mean f1 score: ", mean_f1)

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
    test_queries = ["Near_1.05(o0, o1); Far_0.9(o0, o1); Near_1.05(o0, o1)",
"Near_1.05(o0, o1); Conjunction(Far_0.9(o0, o1), Near_1.05(o0, o1)); Near_1.05(o0, o1)"]

    evaluate_vocal_f1(target_query, test_queries)