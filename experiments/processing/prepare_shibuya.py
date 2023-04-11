import json
import random
import itertools
import shutil
import numpy as np
import os
from src.utils import rewrite_program_postgres, str_to_program_postgres, postgres_execute, postgres_execute_cache_sequence
import csv
from itertools import repeat
from concurrent.futures import ThreadPoolExecutor
import scipy.stats as stats
import time
import psycopg2 as psycopg
import multiprocessing
from lru import LRU
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
from experiments.processing.prepare_data_postgres import prepare_data_given_target_query, construct_train_test

# 31000 frames; 3100 seconds
# 10 frames per second


def json_to_csv():
    segment_length = 100 # frames
    step_size = 100 # frames

    count = 0
    vid = 0
    # write output to csv
    with open('/gscratch/balazinska/enhaoz/complex_event_video/postgres/shibuya.csv', 'w') as out_f:
        for i in range(6):
            with open('/gscratch/balazinska/enhaoz/miris/data/shibuya/json/{}-baseline.json'.format(i)) as f:
                data = json.load(f)
            length = len(data)
            idx = 0
            while idx < length:
                if count == segment_length:
                    vid += 1
                    count = 0
                    idx = idx - segment_length + step_size
                frame_list = data[idx]
                if frame_list:
                    for obj in frame_list:
                        fid = count
                        oid = obj['track_id']
                        x1 = obj['left'] // 2
                        y1 = obj['top'] // 2
                        x2 = obj['right'] // 2
                        y2 = obj['bottom'] // 2
                        out_f.write('{},{},{},{},{},{},{}\n'.format(oid, vid, fid, x1, y1, x2, y2))
                count += 1
                idx += 1
            vid += 1
            count = 0


def prepare_shibuya_queries():
    query_programs = [
        [
            {'scene_graph': [{'predicate': 'BottomPoly', 'parameter': None, 'variables': ['o0']}, {'predicate': 'TopPoly', 'parameter': None, 'variables': ['o1']}], 'duration_constraint': 1},
            {'scene_graph': [{'predicate': 'TopPoly', 'parameter': None, 'variables': ['o0']}, {'predicate': 'BottomPoly', 'parameter': None, 'variables': ['o1']}], 'duration_constraint': 1},
        ],
        [
            {'scene_graph': [{'predicate': 'BottomPoly', 'parameter': None, 'variables': ['o0']}, {'predicate': 'TopPoly', 'parameter': None, 'variables': ['o1']}], 'duration_constraint': 1},
            {'scene_graph': [{'predicate': 'LeftPoly', 'parameter': None, 'variables': ['o0']}, {'predicate': 'RightPoly', 'parameter': None, 'variables': ['o1']}], 'duration_constraint': 1},
        ],
    ]

    dataset_name = "shibuya"
    for query_program in query_programs:
        query_str = rewrite_program_postgres(query_program)
        print(query_str)
        prepare_data_given_target_query(query_str, 0, 1, dataset_name, "Obj_shibuya", sampling_rate=None)
        construct_train_test("/gscratch/balazinska/enhaoz/complex_event_video/inputs/{}".format(dataset_name), n_train=500)

if __name__ == '__main__':
    # json_to_csv()
    prepare_shibuya_queries()