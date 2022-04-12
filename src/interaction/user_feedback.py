import numpy as np

class UserFeedbackMixin:
    def base_user_feedback(self):
        """Input: n materialized frames
        Output: updated annotated frames Fa_i+1
        Method: primary object with spatial feature
        """
        num_materialized_frames = self.materialized_frames.nonzero()[0].size
        for frame_id in self.frame_id_arr_to_annotate:
            # ExSample: update stats
            # Difference from the original paper: frame sampling happens when model annotating frames, while stats are updated only after human annotating frames.
            chunk_idx = int(frame_id / (1.0 * self.n_frames / len(self.stats_per_chunk)))
            self.stats_per_chunk[chunk_idx][1] += 1
            if frame_id in self.pos_frames:
                for key in list(self.pos_frames_per_instance):
                    start_frame, end_frame, flag = self.pos_frames_per_instance[key]
                    if start_frame <= frame_id and frame_id < end_frame:
                        if flag == 0:
                            self.num_positive_instances_found += 1
                            print("num_positive_instances_found:", self.num_positive_instances_found)
                            self.plot_data_y_materialized = np.append(self.plot_data_y_materialized, num_materialized_frames)
                            self.pos_frames_per_instance[key] = (start_frame, end_frame, 1)
                            self.stats_per_chunk[chunk_idx][0] += 1
                        else:
                            del self.pos_frames_per_instance[key]
                            # TODO
                            if self.stats_per_chunk[chunk_idx][0] > 0:
                                self.stats_per_chunk[chunk_idx][0] -= 1
                self.positive_frames_seen.append(frame_id)
                self.plot_data_y_annotated = np.append(self.plot_data_y_annotated, self.num_positive_instances_found)
                # self.get_frames_stats()
            else:
                self.negative_frames_seen.append(frame_id)
                self.plot_data_y_annotated = np.append(self.plot_data_y_annotated, self.num_positive_instances_found)