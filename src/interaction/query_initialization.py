"""Query initialization
Input: raw frames Fr
Output: n_0 materialized frames Fm_0, initial proxy model m_0
"""
# TODO: discrepancy between object detector and user, e.g. object detector annnotates the frame as not having any objects while the user labels it as positive frame that contains a turning vehicle.

import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)


class QueryInitializationMixin:
    def random_initialization(self):
        """Input: raw frames Fr
        Output: n_0 materialized frames Fm_0, updated raw frames Fr_0, initial proxy model m_0
        """
        while not (self.positive_frames_seen and self.negative_frames_seen):
            arr = self.materialized_frames * self.candidates
            frame_id = np.random.choice(arr.nonzero()[0])
            self.materialized_frames[frame_id] = False
            # NOTE: the user will label a frame as positive only if:
            # 1. it is positive,
            # 2. the object detector says it contains objects of interest in the region of interest.
            chunk_idx = int(frame_id / (1.0 * self.materialized_frames.size / len(self.stats_per_chunk)))
            self.stats_per_chunk[chunk_idx][1] += 1
            if frame_id in self.pos_frames:
                for key in list(self.pos_frames_per_instance):
                    start_frame, end_frame, flag = self.pos_frames_per_instance[key]
                    if start_frame <= frame_id and frame_id < end_frame:
                        if flag == 0:
                            self.num_positive_instances_found += 1
                            print("num_positive_instances_found:", self.num_positive_instances_found)
                            self.plot_data_y_materialized = np.append(self.plot_data_y_materialized, self.materialized_frames.nonzero()[0].size)
                            self.pos_frames_per_instance[key] = (start_frame, end_frame, 1)
                            self.stats_per_chunk[chunk_idx][0] += 1
                        else:
                            del self.pos_frames_per_instance[key]
                            # TODO: when the same object found once in two chunks, N^1 can go negative
                            if self.stats_per_chunk[chunk_idx][0] > 0:
                                self.stats_per_chunk[chunk_idx][0] -= 1
                self.positive_frames_seen.append(frame_id)
                self.plot_data_y_annotated = np.append(self.plot_data_y_annotated, self.num_positive_instances_found)
            else:
                self.negative_frames_seen.append(frame_id)
                self.plot_data_y_annotated = np.append(self.plot_data_y_annotated, self.num_positive_instances_found)
