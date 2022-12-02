"""
Generate data for end-to-end experiments to compare against PATSQL, Quivr, and VOCAL.
Trajectory data model.
Each vieo segment is down-sampled every 4 frames.
500 for searching, 500 for testing.
Vary the number of labeling budget b.
The predicates used to construct a query are exactly those appeared in the target query.
"""


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
from quivr.experiments.processing.prepare_data_postgres import prepare_data_given_target_query, construct_train_test


def prepare_patsql_examples(dir_name, query_str, nruns):
    n_init_pos = 2
    n_init_neg = 10
    inputs_filename = query_str + "_inputs.json"
    labels_filename = query_str + "_labels.json"
    if not os.path.exists(os.path.join(dir_name, "PATSQL", query_str)):
        for i in range(nruns):
            write_dir = os.path.join(dir_name, "PATSQL", query_str, str(i))
            os.makedirs(write_dir)
            with open(os.path.join(dir_name, "train", labels_filename), 'r') as f:
                labels = json.load(f)
            with open(os.path.join(dir_name, "train", inputs_filename), 'r') as f:
                inputs = json.load(f)

            labels = np.asarray(labels)
            inputs = np.asarray(inputs)

            pos_idx = np.where(labels == 1)[0]
            neg_idx = np.where(labels == 0)[0]
            # print("pos_idx", len(pos_idx), pos_idx, "neg_idx", len(neg_idx), neg_idx)

            labeled_index = random.sample(pos_idx.tolist(), n_init_pos) + random.sample(neg_idx.tolist(), n_init_neg)
            with open(os.path.join(write_dir, "n_examples_{}_inputs.json".format(len(labeled_index))), 'w') as f:
                saved_inputs = inputs[labeled_index].tolist()
                json.dump(saved_inputs, f)
            with open(os.path.join(write_dir, "n_examples_{}_labels.json".format(len(labeled_index))), 'w') as f:
                saved_labels = labels[labeled_index].tolist()
                json.dump(saved_labels, f)

            budgets = list(range(13, 21)) + list(range(25, 51, 5))
            for budget in budgets:
                unlabeled_index = np.setdiff1d(np.arange(len(labels)), labeled_index)
                sampled_index = random.sample(unlabeled_index.tolist(), budget-len(labeled_index))
                labeled_index += sampled_index
                with open(os.path.join(write_dir, "n_examples_{}_inputs.json".format(len(labeled_index))), 'w') as f:
                    saved_inputs = inputs[labeled_index].tolist()
                    json.dump(saved_inputs, f)
                with open(os.path.join(write_dir, "n_examples_{}_labels.json".format(len(labeled_index))), 'w') as f:
                    saved_labels = labels[labeled_index].tolist()
                    json.dump(saved_labels, f)


def prepare_patsql_inputs_outputs(dir_name, query_str, predicate_list, nruns, sampling_rate, str_or_int="str"):
    dsn = "dbname=myinner_db user=enhaoz host=localhost"
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            for i in range(nruns):
                for filename in os.listdir(os.path.join(dir_name, query_str, str(i))):
                    if filename.endswith("_inputs.json"):
                        inputs_filename = filename
                        labels_filename = filename[:-12] + "_labels.json"
                        # n_examples = int(filename[11:-12])
                        with open(os.path.join(dir_name, query_str, str(i), inputs_filename), 'r') as f:
                            inputs = json.load(f)
                        with open(os.path.join(dir_name, query_str, str(i), labels_filename), 'r') as f:
                            labels = json.load(f)
                        inputs = np.asarray(inputs)
                        labels = np.asarray(labels)

                        for predicate in predicate_list:
                            if predicate == "RightQuadrant":
                                cur.execute("""
                                SELECT a.oid as oid1, a.vid, a.fid / {v} as fid
                                FROM Obj_trajectories a
                                where RightQuadrant(a.shape, a.color, a.material, a.x1, a.y1, a.x2, a.y2)=true
                                and a.vid = ANY(%s) and a.fid %% {v} = 0 and a.oid = 0;
                                """.format(v=sampling_rate), [inputs.tolist()])
                                df = pd.DataFrame(cur.fetchall())
                                if str_or_int == "str":
                                    df.columns = ["oid1:Str", "vid:Str", "fid:Int"] # Rename columns
                                    df['oid1:Str'] = 'o' + df['oid1:Str'].astype(str)
                                    df['vid:Str'] = 'v' + df['vid:Str'].astype(str)
                                else:
                                    df.columns = ["oid1:Int", "vid:Int", "fid:Int"]
                                df.to_csv(os.path.join(dir_name, query_str, str(i), "rightquadrant-n_examples_{}.csv".format(len(inputs))), index=False)
                            elif predicate == "LeftQuadrant":
                                cur.execute("""
                                SELECT a.oid as oid1, a.vid, a.fid / {v} as fid
                                FROM Obj_trajectories a
                                where LeftQuadrant(a.shape, a.color, a.material, a.x1, a.y1, a.x2, a.y2)=true
                                and a.vid = ANY(%s) and a.fid %% {v} = 0 and a.oid = 0;
                                """.format(v=sampling_rate), [inputs.tolist()])
                                df = pd.DataFrame(cur.fetchall())
                                if str_or_int == "str":
                                    df.columns = ["oid1:Str", "vid:Str", "fid:Int"] # Rename columns
                                    df['oid1:Str'] = 'o' + df['oid1:Str'].astype(str)
                                    df['vid:Str'] = 'v' + df['vid:Str'].astype(str)
                                else:
                                    df.columns = ["oid1:Int", "vid:Int", "fid:Int"]
                                df.to_csv(os.path.join(dir_name, query_str, str(i), "leftquadrant-n_examples_{}.csv".format(len(inputs))), index=False)
                            elif predicate == "BottomQuadrant":
                                cur.execute("""
                                SELECT a.oid as oid1, a.vid, a.fid / {v} as fid
                                FROM Obj_trajectories a
                                where BottomQuadrant(a.shape, a.color, a.material, a.x1, a.y1, a.x2, a.y2)=true
                                and a.vid = ANY(%s) and a.fid %% {v} = 0 and a.oid = 0;
                                """.format(v=sampling_rate), [inputs.tolist()])
                                df = pd.DataFrame(cur.fetchall())
                                if str_or_int == "str":
                                    df.columns = ["oid1:Str", "vid:Str", "fid:Int"] # Rename columns
                                    df['oid1:Str'] = 'o' + df['oid1:Str'].astype(str)
                                    df['vid:Str'] = 'v' + df['vid:Str'].astype(str)
                                else:
                                    df.columns = ["oid1:Int", "vid:Int", "fid:Int"]
                                df.to_csv(os.path.join(dir_name, query_str, str(i), "bottomquadrant-n_examples_{}.csv".format(len(inputs))), index=False)
                            elif predicate == "TopQuadrant":
                                cur.execute("""
                                SELECT a.oid as oid1, a.vid, a.fid / {v} as fid
                                FROM Obj_trajectories a
                                where TopQuadrant(a.shape, a.color, a.material, a.x1, a.y1, a.x2, a.y2)=true
                                and a.vid = ANY(%s) and a.fid %% {v} = 0 and a.oid = 0;
                                """.format(v=sampling_rate), [inputs.tolist()])
                                df = pd.DataFrame(cur.fetchall())
                                if str_or_int == "str":
                                    df.columns = ["oid1:Str", "vid:Str", "fid:Int"] # Rename columns
                                    df['oid1:Str'] = 'o' + df['oid1:Str'].astype(str)
                                    df['vid:Str'] = 'v' + df['vid:Str'].astype(str)
                                else:
                                    df.columns = ["oid1:Int", "vid:Int", "fid:Int"]
                                df.to_csv(os.path.join(dir_name, query_str, str(i), "topquadrant-n_examples_{}.csv".format(len(inputs))), index=False)
                            elif predicate == "Near_1":
                                cur.execute("""
                                SELECT a.oid as oid1, b.oid as oid2, a.vid, a.fid / {v} as fid
                                FROM Obj_trajectories a, Obj_trajectories b
                                where a.vid = b.vid and a.fid = b.fid
                                and Near(1, a.shape, a.color, a.material, a.x1, a.y1, a.x2, a.y2, b.shape, b.color, b.material, b.x1, b.y1, b.x2, b.y2)=true
                                and a.vid = ANY(%s) and a.fid %% {v} = 0 and a.oid = 0 and b.oid = 1;
                                """.format(v=sampling_rate), [inputs.tolist()])
                                df = pd.DataFrame(cur.fetchall())
                                if str_or_int == "str":
                                    df.columns = ["oid1:Str", "oid2:Str", "vid:Str", "fid:Int"] # Rename columns
                                    df['oid1:Str'] = 'o' + df['oid1:Str'].astype(str)
                                    df['oid2:Str'] = 'o' + df['oid2:Str'].astype(str)
                                    df['vid:Str'] = 'v' + df['vid:Str'].astype(str)
                                else:
                                    df.columns = ["oid1:Int", "oid2:Int", "vid:Int", "fid:Int"]
                                df.to_csv(os.path.join(dir_name, query_str, str(i), "near-n_examples_{}.csv".format(len(inputs))), index=False)
                            elif predicate == "Far_3":
                                cur.execute("""
                                SELECT a.oid as oid1, b.oid as oid2, a.vid, a.fid / {v} as fid
                                FROM Obj_trajectories a, Obj_trajectories b
                                where a.vid = b.vid and a.fid = b.fid
                                and Far(3, a.shape, a.color, a.material, a.x1, a.y1, a.x2, a.y2, b.shape, b.color, b.material, b.x1, b.y1, b.x2, b.y2)=true
                                and a.vid = ANY(%s) and a.fid %% {v} = 0 and a.oid = 0 and b.oid = 1;
                                """.format(v=sampling_rate), [inputs.tolist()])
                                df = pd.DataFrame(cur.fetchall())
                                if str_or_int == "str":
                                    df.columns = ["oid1:Str", "oid2:Str", "vid:Str", "fid:Int"] # Rename columns
                                    df['oid1:Str'] = 'o' + df['oid1:Str'].astype(str)
                                    df['oid2:Str'] = 'o' + df['oid2:Str'].astype(str)
                                    df['vid:Str'] = 'v' + df['vid:Str'].astype(str)
                                else:
                                    df.columns = ["oid1:Int", "oid2:Int", "vid:Int", "fid:Int"]
                                df.to_csv(os.path.join(dir_name, query_str, str(i), "far-n_examples_{}.csv".format(len(inputs))), index=False)
                            elif predicate == "Behind":
                                cur.execute("""
                                SELECT a.oid as oid1, b.oid as oid2, a.vid, a.fid / {v} as fid
                                FROM Obj_trajectories a, Obj_trajectories b
                                where a.vid = b.vid and a.fid = b.fid
                                and Behind(a.shape, a.color, a.material, a.x1, a.y1, a.x2, a.y2, b.shape, b.color, b.material, b.x1, b.y1, b.x2, b.y2)=true
                                and a.vid = ANY(%s) and a.fid %% {v} = 0 and a.oid = 0 and b.oid = 1;
                                """.format(v=sampling_rate), [inputs.tolist()])
                                df = pd.DataFrame(cur.fetchall())
                                if str_or_int == "str":
                                    df.columns = ["oid1:Str", "oid2:Str", "vid:Str", "fid:Int"] # Rename columns
                                    df['oid1:Str'] = 'o' + df['oid1:Str'].astype(str)
                                    df['oid2:Str'] = 'o' + df['oid2:Str'].astype(str)
                                    df['vid:Str'] = 'v' + df['vid:Str'].astype(str)
                                else:
                                    df.columns = ["oid1:Int", "oid2:Int", "vid:Int", "fid:Int"]
                                df.to_csv(os.path.join(dir_name, query_str, str(i), "behind-n_examples_{}.csv".format(len(inputs))), index=False)
                            elif predicate == "FrontOf":
                                cur.execute("""
                                SELECT a.oid as oid1, b.oid as oid2, a.vid, a.fid / {v} as fid
                                FROM Obj_trajectories a, Obj_trajectories b
                                where a.vid = b.vid and a.fid = b.fid
                                and FrontOf(a.shape, a.color, a.material, a.x1, a.y1, a.x2, a.y2, b.shape, b.color, b.material, b.x1, b.y1, b.x2, b.y2)=true
                                and a.vid = ANY(%s) and a.fid %% {v} = 0 and a.oid = 0 and b.oid = 1;
                                """.format(v=sampling_rate), [inputs.tolist()])
                                df = pd.DataFrame(cur.fetchall())
                                if str_or_int == "str":
                                    df.columns = ["oid1:Str", "oid2:Str", "vid:Str", "fid:Int"] # Rename columns
                                    df['oid1:Str'] = 'o' + df['oid1:Str'].astype(str)
                                    df['oid2:Str'] = 'o' + df['oid2:Str'].astype(str)
                                    df['vid:Str'] = 'v' + df['vid:Str'].astype(str)
                                else:
                                    df.columns = ["oid1:Int", "oid2:Int", "vid:Int", "fid:Int"]
                                df.to_csv(os.path.join(dir_name, query_str, str(i), "frontof-n_examples_{}.csv".format(len(inputs))), index=False)
                            else:
                                raise ValueError("Unsupported predicate: {}".format(predicate))

                        # Output: inputs where labels are 1
                        pos_idx = np.where(labels == 1)[0]
                        output_vids = inputs[pos_idx]
                        if str_or_int == "str":
                            df = pd.DataFrame(data={"vid:Str": output_vids})
                            df['vid:Str'] = 'v' + df['vid:Str'].astype(str)
                        else:
                            df = pd.DataFrame(data={"vid:Int": output_vids})
                        df.to_csv(os.path.join(dir_name, query_str, str(i), "output-n_examples_{}.csv".format(len(inputs))), index=False)

def prepare_data_for_various_types_of_queries(sampling_rate):
    if (not sampling_rate) or sampling_rate == 1:
        dataset_name = "without_duration"
    else:
        dataset_name = "without_duration-sampling_rate_{}".format(sampling_rate)
    query_strs = [
        "Conjunction(Near_1(o0, o1), BottomQuadrant(o0))",
        "Conjunction(FrontOf(o0, o1), TopQuadrant(o0))",
        "Near_1(o0, o1); Far_3(o0, o1)",
        "Conjunction(Conjunction(Near_1(o0, o1), LeftQuadrant(o0)), Behind(o0, o1))",
        "Far_3(o0, o1); Near_1(o0, o1); Far_3(o0, o1)",
        "Conjunction(Far_3(o0, o1), BottomQuadrant(o0)); Near_1(o0, o1)",
        "Far_3(o0, o1); Conjunction(Near_1(o0, o1), Behind(o0, o1))",
        "Conjunction(Far_3(o0, o1), LeftQuadrant(o0)); Conjunction(Near_1(o0, o1), LeftQuadrant(o0))",
        "Far_3(o0, o1); Conjunction(Conjunction(Near_1(o0, o1), LeftQuadrant(o0)), Behind(o0, o1))"
        ]
    predicates = [
        ("Near_1", "BottomQuadrant"),
        ("FrontOf", "TopQuadrant"),
        ("Near_1", "Far_3"),
        ("Near_1", "LeftQuadrant", "Behind"),
        ("Near_1", "Far_3"),
        ("Near_1", "BottomQuadrant", "Far_3"),
        ("Far_3", "Near_1", "Behind"),
        ("Near_1", "LeftQuadrant", "Far_3"),
        ("Near_1", "LeftQuadrant", "Behind", "Far_3")
    ]
    for query_str in query_strs:
        prepare_data_given_target_query(query_str, 0, 1, dataset_name, "Obj_trajectories", sampling_rate=sampling_rate)
    # construct_train_test("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/{}".format(dataset_name), n_train=500, n_test=500)
    # for query_str, predicate_list in zip(query_strs, predicates):
    #     # prepare_patsql_examples("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/{}".format(dataset_name), query_str, nruns=20)
    #     prepare_patsql_inputs_outputs("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/{}/PATSQL".format(dataset_name), query_str, predicate_list, nruns=20, sampling_rate=sampling_rate, str_or_int="str")

if __name__ == '__main__':
    prepare_data_for_various_types_of_queries(sampling_rate=None)