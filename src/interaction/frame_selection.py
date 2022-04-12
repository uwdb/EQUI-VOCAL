import numpy as np
from numpy.core.fromnumeric import argsort
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import random

class FrameSelectionMixin:
    def greatest_confidence_frame_selection(self):
        """Input: current model m_i, current materialized frames Fm_i, annotated frames Fa_i (rows from event tables)
        Output: annotated_batch_size frames for user to label, updated materialized frames Fm_i+1, raw frames Fr_i+1
        Method: materialize materialized_batch_size new frames (randomly with heuristic/exsample), and select annotated_batch_size frames with greatest confidence score for user to label
        """
        # Step 1 (optional): materialize materialized_batch_size new frames (randomly with heuristic)
        if self.raw_frames.nonzero()[0].size:
            self.materialized_new_frames()

        # Step 2: select annotated_batch_size frames with greatest confidence score for user to label
        # self.eval_metric(clf)
        preds = self.clf.predict_proba(self.spatial_features)[:, 1]
        # print("Get next batch before:", np.argsort(-preds)[:5], preds[np.argsort(-preds)][:5])
        frames = self.candidates * self.materialized_frames
        # scores_before_heuristic = preds[frames]
        preds = preds * self.p
        scores = preds[frames]
        frames = frames.nonzero()[0]
        ind = np.argsort(-scores)
        # print("Get next batch:", frames[ind][:5], scores[ind][:5], scores_before_heuristic[ind][:5], self.Y[frames[ind][:5]])
        self.frame_id_arr_to_annotate = frames[ind][:self.annotated_batch_size]
        self.materialized_frames[self.frame_id_arr_to_annotate] = False

    def materialized_new_frames(self):
        # ExSample: choice of chunk and frame
        scores = []
        for i in range(len(self.stats_per_chunk)):
            # print(stats_per_chunk[i][0] + 0.1)
            scores.append(np.random.gamma(self.stats_per_chunk[i][0] + 0.1, scale=1/(self.stats_per_chunk[i][1] + 1), size=1)[0])
        # chunk_idx = max(enumerate(scores), key=lambda x: x[1])[0]
        chunk_idx_rank = 0
        while True:
            chunk_idx = np.argsort(-np.asarray(scores))[chunk_idx_rank]
            # chunk_idx = argsort(scores)[chunk_idx_rank]
            # max(enumerate(scores), key=lambda x: x[1])[chunk_idx_rank]
            chunk_idx_rank += 1
            # print("chunk_idx", chunk_idx, chunk_idx_rank)
            frames_in_selected_chunk = np.array_split(range(self.raw_frames.size), len(self.stats_per_chunk))[chunk_idx]
            # list(self.chunks(range(raw_frames.size), len(stats_per_chunk)))[chunk_idx]
            frame_start = frames_in_selected_chunk[0]
            frame_end = frames_in_selected_chunk[-1]
            frames_in_selected_chunk = self.raw_frames[frame_start:(frame_end+1)]
            p_in_selected_chunk = self.p[frame_start:(frame_end+1)][self.raw_frames[frame_start:(frame_end+1)]]
            if frames_in_selected_chunk.nonzero()[0].size == 0:
                continue
            normalized_p = p_in_selected_chunk / p_in_selected_chunk.sum()
            frame_id_arr = np.random.choice(frames_in_selected_chunk.nonzero()[0], size=min(self.materialized_batch_size, frames_in_selected_chunk.nonzero()[0].size), replace=False, p=normalized_p)
            self.raw_frames[frame_start + frame_id_arr] = False
            self.materialized_frames[frame_start + frame_id_arr] = True
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

    def least_confidence_frame_selection(self):
        """Input: current model m_i, current materialized frames Fm_i, annotated frames Fa_i (rows from event tables)
        Output: annotated_batch_size frames for user to label, updated materialized frames Fm_i+1, raw frames Fr_i+1
        Method: materialize materialized_batch_size new frames (randomly with heuristic/exsample), and select annotated_batch_size frames with greatest confidence score for user to label
        """
        # Step 1 (optional): materialize materialized_batch_size new frames (randomly with heuristic)
        if self.raw_frames.nonzero()[0].size:
            self.materialized_new_frames()

        # Step 2: select annotated_batch_size frames with greatest confidence score for user to label
        # self.eval_metric(clf)
        preds = self.clf.predict_proba(self.spatial_features)[:, 1]
        # print("Get next batch before:", np.argsort(-preds)[:5], preds[np.argsort(-preds)][:5])
        frames = self.candidates * self.materialized_frames
        # scores_before_heuristic = preds[frames]
        preds = preds * self.p
        scores = preds[frames]
        scores[scores < 0.5] = 1 - scores[scores < 0.5]
        frames = frames.nonzero()[0]
        ind = np.argsort(scores)
        # print("Get next batch:", frames[ind][:5], scores[ind][:5], scores_before_heuristic[ind][:5], self.Y[frames[ind][:5]])
        self.frame_id_arr_to_annotate = frames[ind][:self.annotated_batch_size]
        self.materialized_frames[self.frame_id_arr_to_annotate] = False


    def random_frame_selection(self):
        """Input: current model m_i, current materialized frames Fm_i, annotated frames Fa_i (rows from event tables)
        Output: annotated_batch_size frames for user to label, updated materialized frames Fm_i+1, raw frames Fr_i+1
        Method: materialize materialized_batch_size new frames (randomly with heuristic/exsample), and select annotated_batch_size frames with greatest confidence score for user to label
        """
        # Step 1 (optional): materialize materialized_batch_size new frames (randomly with heuristic)
        if self.raw_frames.nonzero()[0].size:
            self.materialized_new_frames()

        # Step 2: select annotated_batch_size frames with greatest confidence score for user to label
        frames = self.candidates * self.materialized_frames
        frames = frames.nonzero()[0]
        random.shuffle(frames)
        self.frame_id_arr_to_annotate = frames[:self.annotated_batch_size]
        self.materialized_frames[self.frame_id_arr_to_annotate] = False