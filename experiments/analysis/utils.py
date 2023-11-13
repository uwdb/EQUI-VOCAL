import psycopg2 as psycopg
from psycopg2 import pool
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from lru import LRU
import time
from src.utils import dsl_to_program, postgres_execute, postgres_execute_cache_sequence, postgres_execute_no_caching, get_inputs_table_name_and_is_trajectory
import math
import argparse
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import yaml

current_dir = os.path.dirname(os.path.realpath(__file__))
base_dir = os.path.dirname(os.path.dirname(current_dir))

def compute_f1_score_single_query(test_query, gt_query, dataset_name, multithread, input_dir, ):
    # read username from yaml file
    config = yaml.safe_load(open(os.path.join(base_dir, "configs/config.yaml")))
    username = config["postgres"]["username"]
    port = config["postgres"]["port"]
    dsn = "dbname=myinner_db user={} host=localhost port={}".format(username, port)
    connections = psycopg.pool.ThreadedConnectionPool(1, multithread, dsn)

    list_size = 72159
    memo = [{} for _ in range(list_size)]
    # Read the test data
    test_dir = os.path.join(input_dir, dataset_name, "test")
    inputs_filename = gt_query + "_inputs.json"
    labels_filename = gt_query + "_labels.json"
    with open(os.path.join(test_dir, inputs_filename), 'r') as f:
        input_vids = json.load(f)
    with open(os.path.join(test_dir, labels_filename), 'r') as f:
        labels = json.load(f)

    inputs_table_name, is_trajectory = get_inputs_table_name_and_is_trajectory(dataset_name)

    test_program = dsl_to_program(test_query)
    conn = connections.getconn()
    outputs, new_memo = postgres_execute_cache_sequence(conn, test_program, memo, inputs_table_name, input_vids, is_trajectory=is_trajectory)
    connections.putconn(conn)

    preds = []
    for input in input_vids:
        if input in outputs:
            preds.append(1)
        else:
            preds.append(0)
    score = f1_score(labels, preds)
    print(score)
    return score

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", type=str, help='Dataset to evaluate.')
    ap.add_argument("--test_query", type=str, help='Target query to evalaute, written in the compact notation.')
    ap.add_argument("--gt_query", type=str, help='ground-truth query to evalaute, written in the compact notation.')
    ap.add_argument("--multithread", type=int, default=1, help='Number of CPUs to use.')
    ap.add_argument('--input_dir', type=str, default="/gscratch/balazinska/enhaoz/complex_event_video/inputs", help='Input directory.')

    args = ap.parse_args()
    dataset_name = args.dataset_name
    test_query = args.test_query
    gt_query = args.gt_query
    multithread = args.multithread
    input_dir = args.input_dir

    compute_f1_score_single_query(test_query, gt_query, dataset_name, multithread, input_dir)