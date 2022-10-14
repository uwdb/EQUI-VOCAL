import psycopg2 as psycopg
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

def evaluate_patsql_f1(query_str, solution_query):
    test_dir = "/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/vary_num_examples-sampling_rate_4/test"
    dsn = "dbname=myinner_db user=enhaoz host=localhost"
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

            # # Near
            # cur.execute("""
            # CREATE TEMPORARY TABLE near AS
            # SELECT a.oid as oid1, b.oid as oid2, a.vid, a.fid
            # FROM Obj_trajectories a, Obj_trajectories b
            # where a.vid = b.vid and a.fid = b.fid
            # and Near(1.05, a.shape, a.color, a.material, a.x1, a.y1, a.x2, a.y2, b.shape, b.color, b.material, b.x1, b.y1, b.x2, b.y2)=true
            # and a.vid = ANY(%s) and a.oid = 0 and b.oid = 1;
            # """, [inputs.tolist()])

            # Near
            cur.execute("""
            CREATE TEMPORARY TABLE near AS
            SELECT a.oid as oid1, b.oid as oid2, a.vid, a.fid / 4 as fid
            FROM Obj_trajectories a, Obj_trajectories b
            where a.vid = b.vid and a.fid = b.fid
            and Near(1.05, a.shape, a.color, a.material, a.x1, a.y1, a.x2, a.y2, b.shape, b.color, b.material, b.x1, b.y1, b.x2, b.y2)=true
            and a.vid = ANY(%s) and a.fid %% {} = 0 and a.oid = 0 and b.oid = 1;
            """.format(4), [inputs.tolist()])

            # Far
            cur.execute("""
            CREATE TEMPORARY TABLE far AS
            SELECT a.oid as oid1, b.oid as oid2, a.vid, a.fid / 4 as fid
            FROM Obj_trajectories a, Obj_trajectories b
            where a.vid = b.vid and a.fid = b.fid
            and Far(0.9, a.shape, a.color, a.material, a.x1, a.y1, a.x2, a.y2, b.shape, b.color, b.material, b.x1, b.y1, b.x2, b.y2)=true
            and a.vid = ANY(%s) and a.fid %% {} = 0 and a.oid = 0 and b.oid = 1;
            """.format(4), [inputs.tolist()])

            # # RightQuadrant
            # cur.execute("""
            # CREATE TEMPORARY TABLE rightquadrant AS
            # SELECT a.oid as oid1, a.vid, a.fid
            # FROM Obj_trajectories a
            # where RightQuadrant(a.shape, a.color, a.material, a.x1, a.y1, a.x2, a.y2)=true
            # and a.vid = ANY(%s) and a.oid = 0;
            # """, [inputs.tolist()])

            # Run solution query
            cur.execute(solution_query)
            out_vids = cur.fetchall()
            preds = []
            for input in inputs:
                if input in out_vids:
                    preds.append(1)
                else:
                    preds.append(0)
            score = f1_score(labels, preds)
            print(score)


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
    query_str = "Near_1.05(o0, o1); Far_0.9(o0, o1); Near_1.05(o0, o1)"
    solution_query = """
        SELECT
    DISTINCT T0.vid
FROM
    near AS T0
JOIN
    near AS T1
        ON T0.oid1 = T1.oid1
JOIN
    far AS T2
        ON T0.vid = T2.vid
        AND T0.fid < T2.fid
        AND T1.vid = T2.vid
        AND T1.fid > T2.fid
    """
    evaluate_patsql_f1(query_str, solution_query)