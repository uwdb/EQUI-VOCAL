"""Query initialization
Input: raw frames Fr
Output: n_0 materialized frames Fm_0, initial proxy model m_0
"""
# TODO: discrepancy between object detector and user, e.g. object detector annnotates the frame as not having any objects while the user labels it as positive frame that contains a turning vehicle.

from time import time
from shutil import copyfile
import json
from typing import List, Tuple
import torch
import math
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import csv
from tqdm import tqdm
from sklearn import tree, metrics
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import graphviz
import random
import more_itertools as mit
import matplotlib.pyplot as plt
import sys
from utils.utils import isInsideIntersection, isOverlapping
from utils import tools

np.set_printoptions(threshold=sys.maxsize)


class BaseQueryInitialization:
    def __init__(self, pos_frames, candidates):
        self.pos_frames = pos_frames
        self.candidates = candidates

    def run(self):
        pass


class RandomInitialization(BaseQueryInitialization):
    def run(self, materialized_frames: np.ndarray, positive_frames_seen: List[int], negative_frames_seen: List[int], pos_frames_per_instance, num_positive_instances_found, plot_data_y_annotated, plot_data_y_materialized, stats_per_chunk) -> Tuple[np.ndarray, np.ndarray, np.ndarray, RandomForestClassifier]:
        """Input: raw frames Fr
        Output: n_0 materialized frames Fm_0, updated raw frames Fr_0, initial proxy model m_0
        """
        while not (positive_frames_seen and negative_frames_seen):
            arr = materialized_frames * self.candidates
            frame_id = np.random.choice(arr.nonzero()[0])

            materialized_frames[frame_id] = False
            # NOTE: the user will label a frame as positive only if:
            # 1. it is positive,
            # 2. the object detector says it contains objects of interest in the region of interest.
            chunk_idx = int(frame_id / (1.0 * materialized_frames.size / len(stats_per_chunk)))
            stats_per_chunk[chunk_idx][1] += 1
            if frame_id in self.pos_frames:
                for key, (start_frame, end_frame, flag) in pos_frames_per_instance.items():
                    if start_frame <= frame_id and frame_id < end_frame:
                        if flag == 0:
                            num_positive_instances_found += 1
                            print("num_positive_instances_found:", num_positive_instances_found)
                            plot_data_y_materialized = np.append(plot_data_y_materialized, materialized_frames.nonzero()[0].size)
                            pos_frames_per_instance[key] = (start_frame, end_frame, 1)
                            stats_per_chunk[chunk_idx][0] += 1
                        else:
                            del pos_frames_per_instance[key]
                            # TODO: when the same object found once in two chunks, N^1 can go negative
                            if stats_per_chunk[chunk_idx][0] > 0:
                                stats_per_chunk[chunk_idx][0] -= 1
                        break
                positive_frames_seen.append(frame_id)
                plot_data_y_annotated = np.append(plot_data_y_annotated, num_positive_instances_found)
            else:
                negative_frames_seen.append(frame_id)
                plot_data_y_annotated = np.append(plot_data_y_annotated, num_positive_instances_found)

        return materialized_frames, positive_frames_seen, negative_frames_seen, pos_frames_per_instance, num_positive_instances_found, plot_data_y_annotated, plot_data_y_materialized, stats_per_chunk
