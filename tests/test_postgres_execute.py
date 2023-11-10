from symbol import parameters
import pytest
from quivr.utils import program_to_dsl, postgres_execute, dsl_to_program_quivr, quivr_str_to_postgres_program, dsl_to_program, postgres_execute_no_caching, postgres_execute_cache_sequence, using
import csv
import pandas as pd
import psycopg
import time
import json
import numpy as np
from lru import LRU
import argparse

def test_rare_event_queries():
    df = pd.read_csv('/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/synthetic_rare/queries.csv')

    for quivr_str in df["query"]:
        print(quivr_str)
        postgres_program = quivr_str_to_postgres_program(quivr_str)

        memoize = [{} for _ in range(300)]
        _start = time.time()
        postgres_outputs, _ = postgres_execute(None, postgres_program, list(range(0, 300)), memoize)
        print("postgres time", time.time() - _start)

        with open("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/synthetic_rare/train/Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Near_1.05), True*), Conjunction(LeftOf, BackOf)), True*), Duration(Conjunction(TopQuadrant, Far_0.9), 5)), True*)_inputs.json", 'r') as f:
            inputs = json.load(f)
        inputs = np.asarray(inputs, dtype=object)

        _start = time.time()
        quivr_outputs = []
        for i in range(len(inputs)):
            input = inputs[i]
            quivr_program = dsl_to_program_quivr(quivr_str)
            result, _ = quivr_program.execute(input, -1, {}, {})
            if result[0, len(input[0])] > 0:
                quivr_outputs.append(i)
        print("quivr time", time.time() - _start)

        assert set(postgres_outputs) == set(quivr_outputs)

def test_rare_event_queries_with_cache():
    df = pd.read_csv('/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/synthetic_rare/queries.csv')

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
        with open("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/synthetic_rare/train/Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Near_1.05), True*), Conjunction(LeftOf, BackOf)), True*), Duration(Conjunction(TopQuadrant, Far_0.9), 5)), True*)_inputs.json", 'r') as f:
            inputs = json.load(f)
        inputs = np.asarray(inputs, dtype=object)

        _start = time.time()
        quivr_outputs = []
        for i in range(len(inputs)):
            input = inputs[i]
            memo = memoize_quivr[i]
            quivr_program = dsl_to_program_quivr(quivr_str)
            result, new_memoize_quivr = quivr_program.execute(input, -1, memo, {})
            memoize_quivr[i].update(new_memoize_quivr)
            if result[0, len(input[0])] > 0:
                quivr_outputs.append(i)
        print("quivr time", time.time() - _start)

        assert set(postgres_outputs) == set(quivr_outputs)

def test_scalability(size, port):
    dsn = "dbname=myinner_db user=enhaoz host=localhost port={}".format(port)
    query_str = "Duration(LeftOf(o0, o1), 5); Conjunction(Conjunction(Conjunction(Conjunction(Behind(o0, o2), Cyan(o2)), FrontOf(o0, o1)), RightQuadrant(o2)), Sphere(o2)); Duration(RightQuadrant(o2), 3)"
    current_query = dsl_to_program(query_str)
    memoize_scene_graph = [LRU(10000) for _ in range(10000)]
    memoize_sequence = [LRU(10000) for _ in range(10000)]
    inputs_table_name = "Obj_clevrer"
    execute_funct = postgres_execute_cache_sequence
    # execute_funct = postgres_execute_no_caching
    _start = time.time()
    outputs, new_memoize_scene_graph, new_memoize_sequence = execute_funct(dsn, current_query, memoize_scene_graph, memoize_sequence, inputs_table_name, list(range(size)), is_trajectory=False)
    print("time", time.time() - _start)
    print(len(outputs))
    print(outputs)
    print(using("profile"))
    for i, memo_dict in enumerate(new_memoize_scene_graph):
            for k, v in memo_dict.items():
                memoize_scene_graph[i][k] = v
    for i, memo_dict in enumerate(new_memoize_sequence):
            for k, v in memo_dict.items():
                memoize_sequence[i][k] = v
    _start = time.time()
    outputs, new_memoize_scene_graph, new_memoize_sequence = execute_funct(dsn, current_query, memoize_scene_graph, memoize_sequence, inputs_table_name, list(range(size)), is_trajectory=False)
    print("time", time.time() - _start)
    print(len(outputs))
    print(outputs)
    print(using("profile"))
    for i, memo_dict in enumerate(new_memoize_scene_graph):
            for k, v in memo_dict.items():
                memoize_scene_graph[i][k] = v
    for i, memo_dict in enumerate(new_memoize_sequence):
            for k, v in memo_dict.items():
                memoize_sequence[i][k] = v
    _start = time.time()
    outputs, new_memoize_scene_graph, new_memoize_sequence = execute_funct(dsn, current_query, memoize_scene_graph, memoize_sequence, inputs_table_name, list(range(size+1)), is_trajectory=False)
    print("time", time.time() - _start)
    print(len(outputs))
    print(outputs)
    print(using("profile"))
    for i, memo_dict in enumerate(new_memoize_scene_graph):
            for k, v in memo_dict.items():
                memoize_scene_graph[i][k] = v
    for i, memo_dict in enumerate(new_memoize_sequence):
            for k, v in memo_dict.items():
                memoize_sequence[i][k] = v
    _start = time.time()
    outputs, new_memoize_scene_graph, new_memoize_sequence = execute_funct(dsn, current_query, memoize_scene_graph, memoize_sequence, inputs_table_name, list(range(size+2)), is_trajectory=False)
    print("time", time.time() - _start)
    print(len(outputs))
    print(outputs)
    print(using("profile"))

    # outputs, new_memoize = postgres_execute_no_caching(dsn, current_query, memoize, inputs_table_name, size, is_trajectory=False)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int)
    ap.add_argument("--port", type=int, default=5432)
    args = ap.parse_args()

    size = args.size
    port = args.port
    test_scalability(size, port)

    # test_rare_event_queries()
    # test_rare_event_queries_with_cache()
