import numpy as np
from numpy.core.fromnumeric import argsort
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import random

class BaseFrameSelection:
    def __init__(self, spatial_features, Y, materialized_batch_size, annotated_batch_size, avg_duration, candidates):
        self.spatial_features = spatial_features
        self.Y = Y
        self.materialized_batch_size = materialized_batch_size
        self.annotated_batch_size = annotated_batch_size
        self.avg_duration = avg_duration
        self.candidates = candidates
    def run(self):
        pass

class GreatestConfidenceFrameSelection(BaseFrameSelection):
    def run(self, p, clf, raw_frames, materialized_frames, positive_frames_seen, stats_per_chunk):
        """Input: current model m_i, current materialized frames Fm_i, annotated frames Fa_i (rows from event tables)
        Output: annotated_batch_size frames for user to label, updated materialized frames Fm_i+1, raw frames Fr_i+1
        Method: materialize materialized_batch_size new frames (randomly with heuristic/exsample), and select annotated_batch_size frames with greatest confidence score for user to label
        """
        # Step 1 (optional): materialize materialized_batch_size new frames (randomly with heuristic)
        if raw_frames.nonzero()[0].size:
            self.materialized_new_frames(raw_frames, materialized_frames, p, stats_per_chunk)

        # Step 2: select annotated_batch_size frames with greatest confidence score for user to label
        # self.eval_metric(clf)
        preds = clf.predict_proba(self.spatial_features)[:, 1]
        # print("Get next batch before:", np.argsort(-preds)[:5], preds[np.argsort(-preds)][:5])
        frames = self.candidates * materialized_frames
        # scores_before_heuristic = preds[frames]
        preds = preds * p
        scores = preds[frames]
        frames = frames.nonzero()[0]
        ind = np.argsort(-scores)
        # print("Get next batch:", frames[ind][:5], scores[ind][:5], scores_before_heuristic[ind][:5], self.Y[frames[ind][:5]])
        frame_id_arr_to_annotate = frames[ind][:self.annotated_batch_size]
        materialized_frames[frame_id_arr_to_annotate] = False
        return frame_id_arr_to_annotate, raw_frames, materialized_frames, stats_per_chunk

    def materialize_new_frames(self, raw_frames, materialized_frames, p, stats_per_chunk):
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

    @staticmethod
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def eval_metric(self, clf):
        """
        Evaluate model performance on all (filtered/candidate) data
        """
        y_pred = clf.predict(self.spatial_features)
        tn, fp, fn, tp = confusion_matrix(self.Y, y_pred).ravel()
        print("accuracy:", accuracy_score(self.Y, y_pred), "; f1_score:", f1_score(self.Y, y_pred), "; tn, fp, fn, tp:", tn, fp, fn, tp)

    # @staticmethod
    # def argsort(seq):
    #     return [x for x,y in sorted(enumerate(seq), key = lambda x: x[1], reverse=True)]


class LeastConfidenceFrameSelection(GreatestConfidenceFrameSelection):
    def run(self, p, clf, raw_frames, materialized_frames, positive_frames_seen, stats_per_chunk):
        """Input: current model m_i, current materialized frames Fm_i, annotated frames Fa_i (rows from event tables)
        Output: annotated_batch_size frames for user to label, updated materialized frames Fm_i+1, raw frames Fr_i+1
        Method: materialize materialized_batch_size new frames (randomly with heuristic/exsample), and select annotated_batch_size frames with greatest confidence score for user to label
        """
        # Step 1 (optional): materialize materialized_batch_size new frames (randomly with heuristic)
        if raw_frames.nonzero()[0].size:
            self.materialized_new_frames(raw_frames, materialized_frames, p, stats_per_chunk)

        # Step 2: select annotated_batch_size frames with greatest confidence score for user to label
        # self.eval_metric(clf)
        preds = clf.predict_proba(self.spatial_features)[:, 1]
        # print("Get next batch before:", np.argsort(-preds)[:5], preds[np.argsort(-preds)][:5])
        frames = self.candidates * materialized_frames
        # scores_before_heuristic = preds[frames]
        preds = preds * p
        scores = preds[frames]
        scores[scores < 0.5] = 1 - scores[scores < 0.5]
        frames = frames.nonzero()[0]
        ind = np.argsort(scores)
        # print("Get next batch:", frames[ind][:5], scores[ind][:5], scores_before_heuristic[ind][:5], self.Y[frames[ind][:5]])
        frame_id_arr_to_annotate = frames[ind][:self.annotated_batch_size]
        materialized_frames[frame_id_arr_to_annotate] = False
        return frame_id_arr_to_annotate, raw_frames, materialized_frames, stats_per_chunk


class RandomFrameSelection(GreatestConfidenceFrameSelection):
    def run(self, p, clf, raw_frames, materialized_frames, positive_frames_seen, stats_per_chunk):
        """Input: current model m_i, current materialized frames Fm_i, annotated frames Fa_i (rows from event tables)
        Output: annotated_batch_size frames for user to label, updated materialized frames Fm_i+1, raw frames Fr_i+1
        Method: materialize materialized_batch_size new frames (randomly with heuristic/exsample), and select annotated_batch_size frames with greatest confidence score for user to label
        """
        # Step 1 (optional): materialize materialized_batch_size new frames (randomly with heuristic)
        if raw_frames.nonzero()[0].size:
            self.materialized_new_frames(raw_frames, materialized_frames, p, stats_per_chunk)

        # Step 2: select annotated_batch_size frames with greatest confidence score for user to label
        frames = self.candidates * materialized_frames
        frames = frames.nonzero()[0]
        random.shuffle(frames)
        frame_id_arr_to_annotate = frames[:self.annotated_batch_size]
        materialized_frames[frame_id_arr_to_annotate] = False
        return frame_id_arr_to_annotate, raw_frames, materialized_frames, stats_per_chunk