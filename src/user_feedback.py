import numpy as np


class UserFeedback:
    def __init__(self, pos_frames, candidates):
        self.pos_frames = pos_frames
        self.candidates = candidates


    def run(self, frame_id_arr_to_annotate, positive_frames_seen, negative_frames_seen, pos_frames_per_instance, num_positive_instances_found, plot_data_y_annotated, plot_data_y_materialized, num_materialized_frames, num_all_frames, stats_per_chunk):
        """Input: n materialized frames
        Output: updated annotated frames Fa_i+1
        Method: primary object with spatial feature
        """
        for frame_id in frame_id_arr_to_annotate:
            # ExSample: update stats
            # Difference from the original paper: frame sampling happens when model annotating frames, while stats are updated only after human annotating frames.
            chunk_idx = int(frame_id / (1.0 * num_all_frames / len(stats_per_chunk)))
            stats_per_chunk[chunk_idx][1] += 1
            if frame_id in self.pos_frames and self.candidates[frame_id]:
                for key, (start_frame, end_frame, flag) in pos_frames_per_instance.items():
                    if start_frame <= frame_id and frame_id < end_frame:
                        if flag == 0:
                            num_positive_instances_found += 1
                            print("num_positive_instances_found:", num_positive_instances_found)
                            plot_data_y_materialized = np.append(plot_data_y_materialized, num_materialized_frames)
                            pos_frames_per_instance[key] = (start_frame, end_frame, 1)
                            stats_per_chunk[chunk_idx][0] += 1
                        else:
                            del pos_frames_per_instance[key]
                            # TODO
                            if stats_per_chunk[chunk_idx][0] > 0:
                                stats_per_chunk[chunk_idx][0] -= 1
                        break
                positive_frames_seen.append(frame_id)
                plot_data_y_annotated = np.append(plot_data_y_annotated, num_positive_instances_found)
                # self.get_frames_stats()
            else:
                negative_frames_seen.append(frame_id)
                plot_data_y_annotated = np.append(plot_data_y_annotated, num_positive_instances_found)
        return positive_frames_seen, negative_frames_seen, pos_frames_per_instance, num_positive_instances_found, plot_data_y_annotated, plot_data_y_materialized, stats_per_chunk