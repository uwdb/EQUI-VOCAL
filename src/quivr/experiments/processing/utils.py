import json
import random
import itertools
import shutil
import numpy as np
import os
from quivr.utils import rewrite_program_postgres, str_to_program_postgres, postgres_execute, postgres_execute_cache_sequence
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

# random.seed(1234)
# np.random.seed(10)

def construct_train_test(dir_name, n_train, n_test=None):
    for filename in os.listdir(dir_name):
        if filename.endswith("_labels.json"):
            query_str = filename[:-12]
            if not os.path.exists(os.path.join(dir_name, "test/{}_labels.json".format(query_str))):
                construct_train_test_per_query(dir_name, query_str, n_train, n_test)

def construct_train_test_per_query(dir_name, query_str, n_train, n_test):
    labels_filename = query_str + "_labels"
    inputs_filename = query_str + "_inputs"

    # read from json file
    with open(os.path.join(dir_name, "{}.json".format(labels_filename)), 'r') as f:
        labels = json.load(f)

    labels = np.asarray(labels)
    inputs = np.arange(len(labels))

    inputs_train, inputs_test, labels_train, labels_test = train_test_split(inputs, labels, train_size=n_train, random_state=42, stratify=labels)
    if n_test:
        inputs_test, _, labels_test, _ = train_test_split(inputs_test, labels_test, train_size=n_test, random_state=42, stratify=labels_test)
    # if folder doesn't exist, create it
    if not os.path.exists(os.path.join(dir_name, "train/")):
        os.makedirs(os.path.join(dir_name, "train/"))
    if not os.path.exists(os.path.join(dir_name, "test/")):
        os.makedirs(os.path.join(dir_name, "test/"))

    with open(os.path.join(dir_name, "train/{}.json".format(inputs_filename)), 'w') as f:
        json.dump(inputs_train.tolist(), f)
    with open(os.path.join(dir_name, "train/{}.json".format(labels_filename)), 'w') as f:
        json.dump(labels_train.tolist(), f)
    with open(os.path.join(dir_name, "test/{}.json".format(inputs_filename)), 'w') as f:
        json.dump(inputs_test.tolist(), f)
    with open(os.path.join(dir_name, "test/{}.json".format(labels_filename)), 'w') as f:
        json.dump(labels_test.tolist(), f)

    print("inputs_train", len(inputs_train))
    print("labels_train", len(labels_train), sum(labels_train))
    print("inputs_test", len(inputs_test))
    print("labels_test", len(labels_test), sum(labels_test))

if __name__ == '__main__':
    construct_train_test("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/collision/", n_train=500)