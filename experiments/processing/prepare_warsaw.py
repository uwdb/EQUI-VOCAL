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
from experiments.processing.prepare_data_postgres import prepare_data_given_target_query, construct_train_test

# 31000 frames; 3100 seconds
# 10 frames per second


def json_to_relation():
    segment_length = 100 # frames
    step_size = 20 # frames

    # write output to csv
    with open('/gscratch/balazinska/enhaoz/complex_event_video/postgres/warsaw_relation.csv', 'w') as out_f:
        for i in range(6):
            with open('/gscratch/balazinska/enhaoz/miris/data/warsaw/json/{}-baseline.json'.format(i)) as f:
                data = json.load(f)
            length = len(data)
            idx = 0
            while idx < length:
                frame_list = data[idx]
                if frame_list:
                    for obj in frame_list:
                        assert(obj['frame_idx'] == idx)
                        fid = idx
                        oid = obj['track_id']
                        x1 = obj['left'] // 2
                        y1 = obj['top'] // 2
                        x2 = obj['right'] // 2
                        y2 = obj['bottom'] // 2
                        out_f.write('{},{},{},{},{},{},{}\n'.format(i, fid, oid, x1, y1, x2, y2))
                idx += 1

def compute_velocity_and_acceleration():
    df = pd.read_csv('/gscratch/balazinska/enhaoz/complex_event_video/postgres/warsaw_relation.csv', header=None, names=['vid', 'fid', 'oid', 'x1', 'y1', 'x2', 'y2'])
    print(len(df))
    # select *
    # from trajectory t1
    # left join trajectory t2
    # on t1.vid = t2.vid and t1.oid = t2.oid and t1.fid = t2.fid - 1
    df['fid_next'] = df['fid'] + 1
    result = pd.merge(df, df, left_on=['vid', 'oid', 'fid_next'], right_on=['vid', 'oid', 'fid'], how='left', suffixes=('_x', '_y'))
    # print(result.head(10))
    result = result[['vid', 'fid_x', 'oid', 'x1_x', 'y1_x', 'x2_x', 'y2_x', 'x1_y', 'y1_y', 'x2_y', 'y2_y']].rename(columns={'fid_x': 'fid'})
    print(len(result))

    result['v_x'] = (result['x1_y'] + result['x2_y']) / 2.0 - (result['x1_x'] + result['x2_x']) / 2.0
    result['v_y'] = (result['y1_y'] + result['y2_y']) / 2.0 - (result['y1_x'] + result['y2_x']) / 2.0
    # result[['v_x', 'v_y']] = result.apply(compute_velocity, axis=1)
    result = result[['vid', 'fid', 'oid', 'x1_x', 'y1_x', 'x2_x', 'y2_x', 'v_x', 'v_y']].rename(columns={'x1_x': 'x1', 'y1_x': 'y1', 'x2_x': 'x2', 'y2_x': 'y2'})
    print(len(result))

    # Do it again for acceleration
    result['fid_next'] = result['fid'] + 1
    result = pd.merge(result, result, left_on=['vid', 'oid', 'fid_next'], right_on=['vid', 'oid', 'fid'], how='left', suffixes=('_x', '_y'))

    result = result[['vid', 'fid_x', 'oid', 'x1_x', 'y1_x', 'x2_x', 'y2_x', 'v_x_x', 'v_y_x', 'x1_y', 'y1_y', 'x2_y', 'y2_y', 'v_x_y', 'v_y_y']].rename(columns={'fid_x': 'fid'})
    result['a_x'] = result['v_x_y'] - result['v_x_x']
    result['a_y'] = result['v_y_y'] - result['v_y_x']
    result = result[['vid', 'fid', 'oid', 'x1_x', 'y1_x', 'x2_x', 'y2_x', 'v_x_x', 'v_y_x', 'a_x', 'a_y']].rename(columns={'v_x_x': 'v_x', 'v_y_x': 'v_y', 'x1_x': 'x1', 'y1_x': 'y1', 'x2_x': 'x2', 'y2_x': 'y2'})

    result.to_csv('/gscratch/balazinska/enhaoz/complex_event_video/postgres/warsaw_relation_velocity.csv', index=False, header=False)



def relation_to_trajectories():
    # create view trajectory as
    #     select vid, oid, min(fid) as fs, max(fid) as fe
    #     from t
    #     group by vid, oid;

    # select distinct t1.vid, t1.oid, t2.oid, t2.fs as fs,
    # from trajectory t1, trajectory t2
    # where t1.vid = t2.vid and t1.oid != t2.oid and t1.fs <= t2.fs and t1.fe >= t2.fs
    df = pd.read_csv('/gscratch/balazinska/enhaoz/complex_event_video/postgres/warsaw_relation_velocity.csv', header=None, names=['vid', 'fid', 'oid', 'x1', 'y1', 'x2', 'y2', 'v_x', 'v_y', 'a_x', 'a_y'])
    filtered = df[df['vid'] == 2]
    grouped = filtered.groupby(['vid', 'oid']).agg({'fid': [min, max]})
    grouped.columns = ['fs', 'fe']
    # grouped = grouped[grouped['fe'] - grouped['fs'] >= 20]
    trajectory = grouped.reset_index()
    # trajectory = trajectory[trajectory['fe'] - trajectory['fs'] <= 300]
    print("Trajectory length: {}".format(len(trajectory)))
    # compute the average of (fe - fs)
    print("Average length: {}".format(np.mean(trajectory['fe'] - trajectory['fs'])))
    # join the table with itself on the 'vid' column
    joined = pd.merge(trajectory, trajectory, on='vid')

    # filter the joined table based on the specified conditions
    filtered = joined[(joined['oid_x'] != joined['oid_y']) & (joined['fs_x'] <= joined['fs_y']) & (joined['fe_x'] >= joined['fs_y'])]
    filtered['fe'] = filtered[['fe_x', 'fe_y']].min(axis=1)
    # select the desired columns and rename them
    result = filtered[['vid', 'oid_x', 'oid_y', 'fs_y', 'fe']].rename(columns={'oid_x': 'oid1', 'oid_y': 'oid2', 'fs_y': 'fs'})

    # drop duplicate rows
    result.drop_duplicates(inplace=True)
    # Add a new column to a table with incremental values starting at 0
    result['new_vid'] = range(len(result))
    print(result)
    # result.to_csv('/gscratch/balazinska/enhaoz/complex_event_video/postgres/warsaw_trajectories.csv', index=False)
    # select r.new_vid as vid, t.fid - r.fs as fid, 0 as oid, t.x1, t.y1, t.x2, t.y2
    # from result r, t
    # where r.vid = t.vid and r.oid1 = t.oid and r.fs <= t.fid and r.fe >= t.fid

    # join the 'result' and 't' DataFrames based on common columns
    output1 = pd.merge(result, df, left_on=['vid', 'oid1'], right_on=['vid', 'oid'], how='inner')
    output1 = output1[(output1['fs'] <= output1['fid']) & (output1['fe'] >= output1['fid'])]

    # create a new DataFrame with the selected columns and modified values
    output1 = pd.DataFrame({
        'oid': 0,
        # 'oid': output1['oid'],
        'vid': output1['new_vid'],
        'fid': output1['fid'] - output1['fs'],
        'x1': output1['x1'],
        'y1': output1['y1'],
        'x2': output1['x2'],
        'y2': output1['y2'],
        'v_x': output1['v_x'],
        'v_y': output1['v_y'],
        'a_x': output1['a_x'],
        'a_y': output1['a_y']
    })
    print("output1 length", len(output1))
    output2 = pd.merge(result, df, left_on=['vid', 'oid2'], right_on=['vid', 'oid'], how='inner')
    output2 = output2[(output2['fs'] <= output2['fid']) & (output2['fe'] >= output2['fid'])]
    output2 = pd.DataFrame({
        'oid': 1,
        # 'oid': output2['oid'],
        'vid': output2['new_vid'],
        'fid': output2['fid'] - output2['fs'],
        'x1': output2['x1'],
        'y1': output2['y1'],
        'x2': output2['x2'],
        'y2': output2['y2'],
        'v_x': output2['v_x'],
        'v_y': output2['v_y'],
        'a_x': output2['a_x'],
        'a_y': output2['a_y']
    })
    print("output2 length", len(output2))
    # concatenate the two DataFrames
    output = pd.concat([output1, output2])
    output.to_csv('/gscratch/balazinska/enhaoz/complex_event_video/postgres/warsaw_trajectories.csv', index=False, header=False)

def postgres_trajectories_to_quivr_trajectories():
    df = pd.read_csv('/gscratch/balazinska/enhaoz/complex_event_video/postgres/warsaw_trajectories.csv', header=None, names=['oid', 'vid', 'fid', 'x1', 'y1', 'x2', 'y2', 'v_x', 'v_y', 'a_x', 'a_y'])
    output = []
    # sort the DataFrame by 'vid', 'fid', 'oid'
    df.sort_values(by=['vid', 'fid', 'oid'], inplace=True)

    trajectory_pair = [[],[]]
    current_vid = 0
    current_fid = 0
    current_oid = 0
    # iterate over the rows of the DataFrame
    for index, row in df.iterrows():
        if row['vid'] != current_vid:
            if current_oid == 1:
                trajectory_pair[1].append([np.nan] * 8)
            output.append(trajectory_pair)
            trajectory_pair = [[],[]]
            current_vid += 1
            current_fid = 0
            current_oid = 0
        while not (row['fid'] == current_fid and row['oid'] == current_oid):
            trajectory_pair[current_oid].append([np.nan] * 8)
            current_oid = (current_oid + 1) % 2
            if current_oid == 0:
                current_fid += 1
        trajectory_pair[current_oid].append([row['x1'], row['y1'], row['x2'], row['y2'], row['v_x'], row['v_y'], row['a_x'], row['a_y']])
        current_oid = (current_oid + 1) % 2
        if current_oid == 0:
            current_fid += 1
    if current_oid == 1:
        trajectory_pair[1].append([np.nan] * 8)
    output.append(trajectory_pair)
    # write the output to a json file
    with open('/gscratch/balazinska/enhaoz/complex_event_video/inputs/warsaw_trajectory_pairs.json', 'w') as f:
        json.dump(output, f)


def prepare_warsaw_queries():
    query_programs = [
        [
            {'scene_graph': [{'predicate': 'Southward1Upper', 'parameter': None, 'variables': ['o0']}, {'predicate': 'Westward2', 'parameter': None, 'variables': ['o1']}], 'duration_constraint': 1},
            {'scene_graph': [{'predicate': 'Westward2', 'parameter': None, 'variables': ['o0']}, {'predicate': 'Westward2', 'parameter': None, 'variables': ['o1']}], 'duration_constraint': 1},
        ],
        [
            {'scene_graph': [{'predicate': 'Eastward4', 'parameter': None, 'variables': ['o0']}, {'predicate': 'Eastward3', 'parameter': None, 'variables': ['o1']}, {'predicate': 'Faster', 'parameter': 1.5, 'variables': ['o0', 'o1']}], 'duration_constraint': 5},
        ],
        [
            {'scene_graph': [{'predicate': 'Eastward3', 'parameter': None, 'variables': ['o0']}, {'predicate': 'Eastward2', 'parameter': None, 'variables': ['o1']}, {'predicate': 'Faster', 'parameter': 1.5, 'variables': ['o0', 'o1']}], 'duration_constraint': 5},
        ],
        [
            {'scene_graph': [{'predicate': 'Eastward4', 'parameter': None, 'variables': ['o0']}, {'predicate': 'Eastward3', 'parameter': None, 'variables': ['o1']}, {'predicate': 'DistanceSmall', 'parameter': 100, 'variables': ['o0', 'o1']}], 'duration_constraint': 5},
        ],
        [
            {'scene_graph': [{'predicate': 'Eastward3', 'parameter': None, 'variables': ['o0']}, {'predicate': 'Eastward2', 'parameter': None, 'variables': ['o1']}, {'predicate': 'DistanceSmall', 'parameter': 100, 'variables': ['o0', 'o1']}], 'duration_constraint': 5},
        ],
        [
            {'scene_graph': [{'predicate': 'Eastward3', 'parameter': None, 'variables': ['o0']}, {'predicate': 'Eastward3', 'parameter': None, 'variables': ['o1']}, {'predicate': 'DistanceSmall', 'parameter': 100, 'variables': ['o0', 'o1']}], 'duration_constraint': 1},
        ],
        [
            {'scene_graph': [{'predicate': 'Eastward4', 'parameter': None, 'variables': ['o0']}, {'predicate': 'Eastward4', 'parameter': None, 'variables': ['o1']}, {'predicate': 'DistanceSmall', 'parameter': 100, 'variables': ['o0', 'o1']}], 'duration_constraint': 1},
        ],
        [
            {'scene_graph': [{'predicate': 'Eastward2', 'parameter': None, 'variables': ['o0']}, {'predicate': 'Eastward2', 'parameter': None, 'variables': ['o1']}, {'predicate': 'DistanceSmall', 'parameter': 100, 'variables': ['o0', 'o1']}], 'duration_constraint': 1},
        ],
        [
            {'scene_graph': [{'predicate': 'Eastward4', 'parameter': None, 'variables': ['o0']}, {'predicate': 'Eastward2', 'parameter': None, 'variables': ['o1']}], 'duration_constraint': 1},
            {'scene_graph': [{'predicate': 'Eastward3', 'parameter': None, 'variables': ['o0']}, {'predicate': 'Eastward2', 'parameter': None, 'variables': ['o1']}], 'duration_constraint': 1},
        ],
        [
            {'scene_graph': [{'predicate': 'Eastward4', 'parameter': None, 'variables': ['o1']}, {'predicate': 'Eastward2', 'parameter': None, 'variables': ['o0']}], 'duration_constraint': 1},
            {'scene_graph': [{'predicate': 'Eastward3', 'parameter': None, 'variables': ['o1']}, {'predicate': 'Eastward2', 'parameter': None, 'variables': ['o0']}], 'duration_constraint': 1},
        ],
        [
            {'scene_graph': [{'predicate': 'Eastward2', 'parameter': None, 'variables': ['o0']}, {'predicate': 'HighAccel', 'parameter': 2, 'variables': ['o0']}], 'duration_constraint': 1},
            {'scene_graph': [{'predicate': 'Eastward2', 'parameter': None, 'variables': ['o1']}, {'predicate': 'HighAccel', 'parameter': 2, 'variables': ['o1']}], 'duration_constraint': 1},
        ],
        [
            {'scene_graph': [{'predicate': 'Eastward3', 'parameter': None, 'variables': ['o0']}, {'predicate': 'HighAccel', 'parameter': 2, 'variables': ['o0']}], 'duration_constraint': 1},
            {'scene_graph': [{'predicate': 'Eastward3', 'parameter': None, 'variables': ['o1']}, {'predicate': 'HighAccel', 'parameter': 2, 'variables': ['o1']}], 'duration_constraint': 1},
        ],
        [
            {'scene_graph': [{'predicate': 'Eastward4', 'parameter': None, 'variables': ['o0']}, {'predicate': 'HighAccel', 'parameter': 2, 'variables': ['o0']}], 'duration_constraint': 1},
            {'scene_graph': [{'predicate': 'Eastward4', 'parameter': None, 'variables': ['o1']}, {'predicate': 'HighAccel', 'parameter': 2, 'variables': ['o1']}], 'duration_constraint': 1},
        ],
    ]

    dataset_name = "warsaw"
    for query_program in query_programs:
        query_str = program_to_dsl(query_program, False)
        print(query_str)
        prepare_data_given_target_query(query_str, 0, 1, dataset_name, "Obj_warsaw", sampling_rate=None)
        construct_train_test("/gscratch/balazinska/enhaoz/complex_event_video/inputs/{}".format(dataset_name), n_train=37500)

if __name__ == '__main__':
    # json_to_relation()
    # compute_velocity_and_acceleration()
    # relation_to_trajectories()
    prepare_warsaw_queries()
    # postgres_trajectories_to_quivr_trajectories()