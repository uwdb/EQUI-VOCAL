import json
import random
import itertools
import shutil
import numpy as np
import os
from src.utils import program_to_dsl, dsl_to_program, postgres_execute, postgres_execute_cache_sequence
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

def clean_data():
    dir_name = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/collision"
    # read from json file
    with open(os.path.join(dir_name, "collision.json"), 'r') as f:
        inputs = json.load(f)
    with open(os.path.join(dir_name, "collision_labels.json"), 'r') as f:
        labels = json.load(f)

    cleaned_inputs = []
    cleaned_labels = []
    for input, label in zip(inputs, labels):
        assert(len(input[0])==len(input[1]))
        if len(input[0]) == 0:
            continue
        cleaned_inputs.append(input)
        cleaned_labels.append(label)

    with open(os.path.join(dir_name, "cleaned_collision.json"), 'w') as f:
        json.dump(cleaned_inputs, f)
    with open(os.path.join(dir_name, "cleaned_collision_labels.json"), 'w') as f:
        json.dump(cleaned_labels, f)

if __name__ == '__main__':
    clean_data()