from symbol import parameters
import pytest
from quivr.utils import rewrite_program_postgres, postgres_execute, str_to_program, quivr_str_to_postgres_program
import csv
import pandas as pd
import psycopg
import time
import json
import numpy as np
from lru import LRU

def test_rare_event_queries():
    df = pd.read_csv('/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/synthetic_rare/queries.csv')

    for quivr_str in df["query"]:
        print(quivr_str)
        postgres_program = quivr_str_to_postgres_program(quivr_str)

        memoize = [{} for _ in range(300)]
        _start = time.time()
        postgres_outputs, _ = postgres_execute(None, postgres_program, list(range(0, 300)), memoize)
        print("postgres time", time.time() - _start)

        with open("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/synthetic_rare/train/Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Near_1.05), True*), Conjunction(LeftOf, BackOf)), True*), Duration(Conjunction(TopQuadrant, Far_0.9), 5)), True*)_inputs.json", 'r') as f:
            inputs = json.load(f)
        inputs = np.asarray(inputs, dtype=object)

        _start = time.time()
        quivr_outputs = []
        for i in range(len(inputs)):
            input = inputs[i]
            quivr_program = str_to_program(quivr_str)
            result, _ = quivr_program.execute(input, -1, {}, {})
            if result[0, len(input[0])] > 0:
                quivr_outputs.append(i)
        print("quivr time", time.time() - _start)

        assert set(postgres_outputs) == set(quivr_outputs)

def test_rare_event_queries_with_cache():
    df = pd.read_csv('/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/synthetic_rare/queries.csv')

    memoize_postgres = [LRU(1000) for _ in range(300)]
    memoize_quivr = [LRU(1000) for _ in range(300)]
    for quivr_str in df["query"]:
        print(quivr_str)
        postgres_program = quivr_str_to_postgres_program(quivr_str)
        _start = time.time()
        postgres_outputs, new_memoize_postgres = postgres_execute(None, postgres_program, list(range(0, 300)), memoize_postgres)
        print("postgres time", time.time() - _start)
        for i, v in enumerate(new_memoize_postgres):
            memoize_postgres[i].update(v)
        with open("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/synthetic_rare/train/Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Near_1.05), True*), Conjunction(LeftOf, BackOf)), True*), Duration(Conjunction(TopQuadrant, Far_0.9), 5)), True*)_inputs.json", 'r') as f:
            inputs = json.load(f)
        inputs = np.asarray(inputs, dtype=object)

        _start = time.time()
        quivr_outputs = []
        for i in range(len(inputs)):
            input = inputs[i]
            memo = memoize_quivr[i]
            quivr_program = str_to_program(quivr_str)
            result, new_memoize_quivr = quivr_program.execute(input, -1, memo, {})
            memoize_quivr[i].update(new_memoize_quivr)
            if result[0, len(input[0])] > 0:
                quivr_outputs.append(i)
        print("quivr time", time.time() - _start)

        assert set(postgres_outputs) == set(quivr_outputs)

if __name__ == '__main__':
    # test_rare_event_queries()
    test_rare_event_queries_with_cache()