from symbol import parameters
import pytest
from quivr.utils import rewrite_program_postgres, postgres_execute, str_to_program
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


def quivr_str_to_postgres_program(quivr_str):
    if quivr_str.startswith("Near"):
        predicate = {
                        "predicate": "Near",
                        "parameter": float(quivr_str.split("(")[0].split("_")[1]),
                        "variables": ["o0", "o1"]
                    }
        scene_graph = {
                "scene_graph": [predicate],
                "duration_constraint": 1

            }
        return [scene_graph]
    elif quivr_str.startswith("Far"):
        predicate = {
                        "predicate": "Far",
                        "parameter": float(quivr_str.split("(")[0].split("_")[1]),
                        "variables": ["o0", "o1"]
                    }
        scene_graph = {
                "scene_graph": [predicate],
                "duration_constraint": 1

            }
        return [scene_graph]
    elif quivr_str.startswith("True*"):
        return None
    elif quivr_str.startswith("LeftOf"):
        predicate = {
                        "predicate": "LeftOf",
                        "parameter": None,
                        "variables": ["o0", "o1"]
                    }
        scene_graph = {
                "scene_graph": [predicate],
                "duration_constraint": 1

            }
        return [scene_graph]
    elif quivr_str.startswith("BackOf"):
        predicate = {
                        "predicate": "Behind",
                        "parameter": None,
                        "variables": ["o0", "o1"]
                    }
        scene_graph = {
                "scene_graph": [predicate],
                "duration_constraint": 1

            }
        return [scene_graph]
    elif quivr_str.startswith("TopQuadrant"):
        predicate = {
                        "predicate": "TopQuadrant",
                        "parameter": None,
                        "variables": ["o0"]
                    }
        scene_graph = {
                "scene_graph": [predicate],
                "duration_constraint": 1

            }
        return [scene_graph]
    elif quivr_str.startswith("RightQuadrant"):
        predicate = {
                        "predicate": "RightQuadrant",
                        "parameter": None,
                        "variables": ["o0"]
                    }
        scene_graph = {
                "scene_graph": [predicate],
                "duration_constraint": 1

            }
        return [scene_graph]
    else:
        idx = quivr_str.find("(")
        idx_r = quivr_str.rfind(")")
        # True*, Sequencing(Near_1.0, MinLength_10.0)
        functionclass = quivr_str[:idx]
        submodules = quivr_str[idx+1:idx_r]
        counter = 0
        submodule_list = []
        submodule_start = 0
        for i, char in enumerate(submodules):
            if char == "," and counter == 0:
                submodule_list.append(submodules[submodule_start:i])
                submodule_start = i+2
            elif char == "(":
                counter += 1
            elif char == ")":
                counter -= 1
        submodule_list.append(submodules[submodule_start:])
        if functionclass == "Sequencing":
            g1 = quivr_str_to_postgres_program(submodule_list[0])
            g2 = quivr_str_to_postgres_program(submodule_list[1])
            if not g1:
                return g2
            if not g2:
                return g1
            else:
                return [*g1, *g2]
        elif functionclass == "Conjunction":
            scene_graph = quivr_str_to_postgres_program(submodule_list[0])
            scene_graph[0]["scene_graph"].append(quivr_str_to_postgres_program(submodule_list[1])[0]["scene_graph"][0])
            return scene_graph
        elif functionclass == "Duration":
            scene_graph = quivr_str_to_postgres_program(submodule_list[0])
            scene_graph[0]["duration_constraint"] = int(submodule_list[1])
            return scene_graph

if __name__ == '__main__':
    # test_rare_event_queries()
    test_rare_event_queries_with_cache()