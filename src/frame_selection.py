import numpy as np
from numpy.core.fromnumeric import argsort

# Average duration of the target event (60.8 frames per event)
AVG_DURATION = 61

class BaseFrameSelection:
    def __init__(self, spatial_features, Y):
        self.spatial_features = spatial_features
        self.Y = Y

    def run(self):
        pass

class RandomFrameSelection(BaseFrameSelection):
    def __init__(self, spatial_features, Y, materialized_batch_size, annotated_batch_size) -> None:
        self.materialized_batch_size = materialized_batch_size
        self.annotated_batch_size = annotated_batch_size
        super().__init__(spatial_features, Y)

    def run(self, clf, raw_frames, materialized_frames, positive_frames_seen, negative_frames_seen, stats_per_chunk):
        """Input: current model m_i, current materialized frames Fm_i, annotated frames Fa_i (rows from event tables)
        Output: annotated_batch_size frames for user to label, updated materialized frames Fm_i+1, raw frames Fr_i+1
        Method: materialize materialized_batch_size new frames (randomly with heuristic/exsample), and select annotated_batch_size frames with greatest confidence score for user to label
        """
        p = self.update_random_choice_p(positive_frames_seen)

        # materialize materialized_batch_size new frames (randomly with heuristic)
        if raw_frames.nonzero()[0].size:
            # ExSample: choice of chunk and frame
            scores = []
            for i in range(len(stats_per_chunk)):
                # print(stats_per_chunk[i][0] + 0.1)
                scores.append(np.random.gamma(stats_per_chunk[i][0] + 0.1, scale=1/(stats_per_chunk[i][1] + 1), size=1)[0])
            # chunk_idx = max(enumerate(scores), key=lambda x: x[1])[0]
            chunk_idx_rank = 0
            while True:
                chunk_idx = np.argsort(-np.asarray(scores))[chunk_idx_rank]
                # chunk_idx = argsort(scores)[chunk_idx_rank]
                # max(enumerate(scores), key=lambda x: x[1])[chunk_idx_rank]
                chunk_idx_rank += 1
                # print("chunk_idx", chunk_idx, chunk_idx_rank)
                frames_in_selected_chunk = np.array_split(range(raw_frames.size), len(stats_per_chunk))[chunk_idx]
                # list(self.chunks(range(raw_frames.size), len(stats_per_chunk)))[chunk_idx]
                frame_start = frames_in_selected_chunk[0]
                frame_end = frames_in_selected_chunk[-1]
                frames_in_selected_chunk = raw_frames[frame_start:(frame_end+1)]
                p_in_selected_chunk = p[frame_start:(frame_end+1)][raw_frames[frame_start:(frame_end+1)]]
                if frames_in_selected_chunk.nonzero()[0].size == 0:
                    continue
                normalized_p = p_in_selected_chunk / p_in_selected_chunk.sum()
                frame_id_arr = np.random.choice(frames_in_selected_chunk.nonzero()[0], size=min(self.materialized_batch_size, frames_in_selected_chunk.nonzero()[0].size), replace=False, p=normalized_p)
                raw_frames[frame_start + frame_id_arr] = False
                materialized_frames[frame_start + frame_id_arr] = True
                break
            # frame_id_arr = np.random.choice(raw_frames.nonzero()[0], size=min(self.materialized_batch_size, raw_frames.nonzero()[0].size), replace=False, p=normalized_p)
            # raw_frames[frame_id_arr] = False
            # materialized_frames[frame_id_arr] = True

        # select annotated_batch_size frames with greatest confidence score for user to label
        preds = clf.predict_proba(self.spatial_features)[:, 1]
        # print("Get next batch before:", np.argsort(-preds)[:5], preds[np.argsort(-preds)][:5])
        # self.update_random_choice_p()
        preds = preds * p
        scores = preds[materialized_frames]
        frames = materialized_frames.nonzero()[0]
        ind = np.argsort(-scores)
        # print("Get next batch:", frames[ind][:5], scores[ind][:5])
        frame_id_arr_to_annotate = frames[ind][:self.annotated_batch_size]
        materialized_frames[frame_id_arr_to_annotate] = False
        return frame_id_arr_to_annotate, raw_frames, materialized_frames, stats_per_chunk

    def update_random_choice_p(self, positive_frames_seen):
        """Given positive frames seen, compute the probabilities associated with each frame for random choice.
        Frames that are close to a positive frame that have been seen are more likely to be positive as well, thus should have a smaller probability of returning to user for annotations.
        Heuristic: For each observed positive frame, the probability function is 0 at that observed frame, grows ``linearly'' as the distance from the observed frame increases, and becomes constantly 1 after AVG_DURATION distance on each side.
        TODO: considering cases when two observed positive frames are close enough that their probability functions overlap.
        Return: Numpy_Array[proba]
        """
        scale = 2
        func = lambda x : (x ** 2) / (int(AVG_DURATION * scale) ** 2)
        p = np.ones(len(self.Y))
        for frame_id in positive_frames_seen:
           # Right half of the probability function
            for i in range(int(AVG_DURATION * scale) + 1):
                if frame_id + i < len(self.Y):
                    p[frame_id + i] = min(func(i), p[frame_id + i])
            # Left half of the probability function
            for i in range(int(AVG_DURATION * scale) + 1):
                if frame_id - i >= 0:
                    p[frame_id - i] = min(func(i), p[frame_id - i])
        return p

    @staticmethod
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # @staticmethod
    # def argsort(seq):
    #     return [x for x,y in sorted(enumerate(seq), key = lambda x: x[1], reverse=True)]