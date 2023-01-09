import psycopg2 as psycopg
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import argparse

def evaluate_patsql(query_str, predicate_list, port, sampling_rate):
    """
    For each budget, evaluate the f1 score of the solution query, and the runtime of finding the solution query
    """
    test_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/without_duration-sampling_rate_4/test"
    dsn = "dbname=myinner_db user=enhaoz host=localhost port={}".format(port)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            inputs_filename = query_str + "_inputs.json"
            labels_filename = query_str + "_labels.json"
            # n_examples = int(filename[11:-12])
            with open(os.path.join(test_dir, inputs_filename), 'r') as f:
                inputs = json.load(f)
            with open(os.path.join(test_dir, labels_filename), 'r') as f:
                labels = json.load(f)
            inputs = np.asarray(inputs)
            labels = np.asarray(labels)

            for predicate in predicate_list:
                if predicate == "RightQuadrant":
                    cur.execute("""
                    CREATE TEMPORARY TABLE rightquadrant AS
                    SELECT a.oid as oid1, a.vid, a.fid / {v} as fid
                    FROM Obj_trajectories a
                    where RightQuadrant(a.shape, a.color, a.material, a.x1, a.y1, a.x2, a.y2)=true
                    and a.vid = ANY(%s) and a.fid %% {v} = 0 and a.oid = 0;
                    """.format(v=sampling_rate), [inputs.tolist()])
                    cur.execute("CREATE INDEX idx_rightquadrant ON rightquadrant (vid, fid, oid1);")
                elif predicate == "LeftQuadrant":
                    cur.execute("""
                    CREATE TEMPORARY TABLE leftquadrant AS
                    SELECT a.oid as oid1, a.vid, a.fid / {v} as fid
                    FROM Obj_trajectories a
                    where LeftQuadrant(a.shape, a.color, a.material, a.x1, a.y1, a.x2, a.y2)=true
                    and a.vid = ANY(%s) and a.fid %% {v} = 0 and a.oid = 0;
                    """.format(v=sampling_rate), [inputs.tolist()])
                    cur.execute("CREATE INDEX idx_leftquadrant ON leftquadrant (vid, fid, oid1);")
                elif predicate == "BottomQuadrant":
                    cur.execute("""
                    CREATE TEMPORARY TABLE bottomquadrant AS
                    SELECT a.oid as oid1, a.vid, a.fid / {v} as fid
                    FROM Obj_trajectories a
                    where BottomQuadrant(a.shape, a.color, a.material, a.x1, a.y1, a.x2, a.y2)=true
                    and a.vid = ANY(%s) and a.fid %% {v} = 0 and a.oid = 0;
                    """.format(v=sampling_rate), [inputs.tolist()])
                    cur.execute("CREATE INDEX idx_bottomquadrant ON bottomquadrant (vid, fid, oid1);")
                elif predicate == "TopQuadrant":
                    cur.execute("""
                    CREATE TEMPORARY TABLE topquadrant AS
                    SELECT a.oid as oid1, a.vid, a.fid / {v} as fid
                    FROM Obj_trajectories a
                    where TopQuadrant(a.shape, a.color, a.material, a.x1, a.y1, a.x2, a.y2)=true
                    and a.vid = ANY(%s) and a.fid %% {v} = 0 and a.oid = 0;
                    """.format(v=sampling_rate), [inputs.tolist()])
                    cur.execute("CREATE INDEX idx_topquadrant ON topquadrant (vid, fid, oid1);")
                elif predicate == "Near_1":
                    cur.execute("""
                    CREATE TEMPORARY TABLE near AS
                    SELECT a.oid as oid1, b.oid as oid2, a.vid, a.fid / {v} as fid
                    FROM Obj_trajectories a, Obj_trajectories b
                    where a.vid = b.vid and a.fid = b.fid
                    and Near(1, a.shape, a.color, a.material, a.x1, a.y1, a.x2, a.y2, b.shape, b.color, b.material, b.x1, b.y1, b.x2, b.y2)=true
                    and a.vid = ANY(%s) and a.fid %% {v} = 0 and a.oid = 0 and b.oid = 1;
                    """.format(v=sampling_rate), [inputs.tolist()])
                    cur.execute("CREATE INDEX idx_near ON near (vid, fid, oid1, oid2);")
                elif predicate == "Far_3":
                    cur.execute("""
                    CREATE TEMPORARY TABLE far AS
                    SELECT a.oid as oid1, b.oid as oid2, a.vid, a.fid / {v} as fid
                    FROM Obj_trajectories a, Obj_trajectories b
                    where a.vid = b.vid and a.fid = b.fid
                    and Far(3, a.shape, a.color, a.material, a.x1, a.y1, a.x2, a.y2, b.shape, b.color, b.material, b.x1, b.y1, b.x2, b.y2)=true
                    and a.vid = ANY(%s) and a.fid %% {v} = 0 and a.oid = 0 and b.oid = 1;
                    """.format(v=sampling_rate), [inputs.tolist()])
                    cur.execute("CREATE INDEX idx_far ON far (vid, fid, oid1, oid2);")
                elif predicate == "Behind":
                    cur.execute("""
                    CREATE TEMPORARY TABLE behind AS
                    SELECT a.oid as oid1, b.oid as oid2, a.vid, a.fid / {v} as fid
                    FROM Obj_trajectories a, Obj_trajectories b
                    where a.vid = b.vid and a.fid = b.fid
                    and Behind(a.shape, a.color, a.material, a.x1, a.y1, a.x2, a.y2, b.shape, b.color, b.material, b.x1, b.y1, b.x2, b.y2)=true
                    and a.vid = ANY(%s) and a.fid %% {v} = 0 and a.oid = 0 and b.oid = 1;
                    """.format(v=sampling_rate), [inputs.tolist()])
                    cur.execute("CREATE INDEX idx_behind ON behind (vid, fid, oid1, oid2);")
                elif predicate == "FrontOf":
                    cur.execute("""
                    CREATE TEMPORARY TABLE frontof AS
                    SELECT a.oid as oid1, b.oid as oid2, a.vid, a.fid / {v} as fid
                    FROM Obj_trajectories a, Obj_trajectories b
                    where a.vid = b.vid and a.fid = b.fid
                    and FrontOf(a.shape, a.color, a.material, a.x1, a.y1, a.x2, a.y2, b.shape, b.color, b.material, b.x1, b.y1, b.x2, b.y2)=true
                    and a.vid = ANY(%s) and a.fid %% {v} = 0 and a.oid = 0 and b.oid = 1;
                    """.format(v=sampling_rate), [inputs.tolist()])
                    cur.execute("CREATE INDEX idx_frontof ON frontof (vid, fid, oid1, oid2);")
                else:
                    raise ValueError("Unsupported predicate: {}".format(predicate))

            budgets = list(range(12, 21)) + list(range(25, 51, 5))
            # Read the output file
            output_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/without_duration-sampling_rate_4"
            score_log = []
            runtime_log = []
            for run in range(20):
                score_log_per_run = []
                runtime_log_per_run = []
                for budget in budgets:
                    with open(os.path.join(output_dir, "PATSQL", query_str, str(run), "n_examples_{}.txt".format(budget)), 'r') as f:
                        solution_query = []
                        runtime = []
                        for line in f:
                            line = line.rstrip('\n')
                            if line != "":
                                solution_query.append(line)
                                if line == "Not found":
                                    runtime = f.readline().rstrip('\n')
                                    break
                            else:
                                runtime = f.readline().rstrip('\n')
                                break
                    solution_query = " ".join(solution_query)
                    runtime = float(runtime.replace('That took ','').replace(' milliseconds','')) / 1000
                    if solution_query == "Not found":
                        score = -1
                    else:
                        # Run solution query
                        cur.execute(solution_query)
                        out_vids = cur.fetchall()
                        out_vids = [row[0] for row in out_vids]
                        preds = []
                        for input in inputs:
                            if input in out_vids:
                                preds.append(1)
                            else:
                                preds.append(0)
                        score = f1_score(labels, preds)
                    print("score: {}, runtime: {}".format(score, runtime))
                    score_log_per_run.append(score)
                    runtime_log_per_run.append(runtime)
                score_log.append(score_log_per_run)
                runtime_log.append(runtime_log_per_run)
    out_dict = {"score": score_log, "runtime": runtime_log}
    if not os.path.exists(os.path.join(output_dir, "stats", "PATSQL")):
        os.makedirs(os.path.join(output_dir, "stats", "PATSQL"))
    with open(os.path.join(output_dir, "stats", "PATSQL", "{}.json".format(query_str)), "w") as f:
        json.dump(out_dict, f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_str", type=str)
    ap.add_argument("--port", type=int, default=5432)
    args = ap.parse_args()
    query_str = args.query_str
    port = args.port

    sampling_rate = 4
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
    query_idx = query_strs.index(query_str)
    predicate_list = predicates[query_idx]

    # for i, (query_str, predicate_list) in enumerate(zip(query_strs, predicates)):
    evaluate_patsql(query_str, predicate_list, port, sampling_rate)