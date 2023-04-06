from collections import deque
from sklearn.metrics import f1_score
import resource
import numpy as np
import time
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
from src.utils import print_program, rewrite_program, str_to_program, postgres_execute, postgres_execute_cache_sequence, postgres_execute_no_caching, rewrite_program_postgres, str_to_program_postgres, complexity_cost
import itertools
from sklearn.utils import resample
import pandas as pd

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return '''%s: mem=%s MB
           '''%(point, usage/1024.0 )

# np.random.seed(0)
class BaseMethod:
    def compute_query_score_postgres(self, current_query):
        # NOTE: sufficinet to lock only when writing to the memoize_all_inputs? Updating dict/list in python is atomic operation, so no conflicts for write, but reading might get old values (which is fine for us).
        input_vids = self.inputs[self.labeled_index].tolist()
        y_pred = []
        result, new_memoize_scene_graph, new_memoize_sequence = postgres_execute_cache_sequence(self.dsn, current_query, self.memoize_scene_graph_all_inputs, self.memoize_sequence_all_inputs, self.inputs_table_name, input_vids, is_trajectory=self.is_trajectory, sampling_rate=self.sampling_rate)
        if self.lock:
            self.lock.acquire()
        for i, memo_dict in enumerate(new_memoize_scene_graph):
            for k, v in memo_dict.items():
                self.memoize_scene_graph_all_inputs[i][k] = v
        for i, memo_dict in enumerate(new_memoize_sequence):
            for k, v in memo_dict.items():
                self.memoize_sequence_all_inputs[i][k] = v
        if self.lock:
            self.lock.release()
        for i in input_vids:
            if i in result:
                y_pred.append(1)
            else:
                y_pred.append(0)

        f1 = f1_score(list(self.labels[self.labeled_index]), y_pred)
        score = f1 - self.reg_lambda * complexity_cost(current_query)
        return score

    def _compute_u_t(self, posterior_t, predictions_c):

        # Initialize possible u_t's
        u_t_list = np.zeros(2)

        # Repeat for each class
        for c in [0, 1]:
            # Compute the loss of models if the label of the streamed data is "c"
            loss_c = np.array(predictions_c != c)*1
            #
            # Compute the respective u_t value (conditioned on class c)
            term1 = np.inner(posterior_t, loss_c)
            u_t_list[c] = term1*(1-term1)

        # Return the final u_t
        u_t = np.max(u_t_list)

        return u_t

    def pick_next_segment_model_picker_postgres(self):
        """
        Pick the next segment to be labeled, using the Model Picker algorithm.
        """
        true_labels = np.array(self.labels)
        n_instances = len(true_labels)
        prediction_matrix = []
        _start = time.time()

        query_list = [query_graph.program for query_graph, _ in self.candidate_queries[:self.pool_size]]
        print("query pool", [rewrite_program_postgres(query) for query in query_list])
        unlabeled_index = np.setdiff1d(np.arange(n_instances), self.labeled_index, assume_unique=True)

        # If more than self.n_sampled_videos videos, sample self.n_sampled_videos videos
        if len(unlabeled_index) > self.n_sampled_videos:
            self.sampled_index = np.random.choice(unlabeled_index, self.n_sampled_videos, replace=False)
        else:
            self.sampled_index = unlabeled_index

        self.n_prediction_count += len(query_list) * len(self.sampled_index)
        if self.multithread > 1:

            for pred_per_query in self.executor.map(self.execute_over_all_inputs_postgres, query_list):
                prediction_matrix.append(pred_per_query)
        else:
            for query in query_list:
                pred_per_query = self.execute_over_all_inputs_postgres(query)
                prediction_matrix.append(pred_per_query)
        prediction_matrix = np.array(prediction_matrix).transpose()
        print("constructing prediction matrix", time.time()-_start)
        print("prediction_matrix size", prediction_matrix.shape)

        # Use F1-scores as weights
        posterior_t = [score for _, score in self.candidate_queries[:self.pool_size]]
        posterior_t  /= np.sum(posterior_t)  # normalized weight

        print("query weights", posterior_t)
        entropy_list = np.zeros(len(self.sampled_index))
        for i in range(len(self.sampled_index)):
            entropy_list[i] = self._compute_u_t(posterior_t, prediction_matrix[i, :])
        ind = np.argsort(-entropy_list)
        print("entropy list", entropy_list[ind])
        print("sampled index", self.sampled_index[ind])
        # find argmax of entropy (top k)
        max_entropy_index = self.sampled_index[np.argmax(entropy_list)]
        return [max_entropy_index]
        # video_segment_ids = np.argpartition(entropy_list, -self.samples_per_iter)[-self.samples_per_iter:]
        # return video_segment_ids.tolist()


    def pick_next_segment_randomly_postgres(self):
        """
        Pick the next segment to be labeled, randomly.
        """
        true_labels = np.array(self.labels)
        n_instances = len(true_labels)
        unlabeled_index = np.setdiff1d(np.arange(n_instances), self.labeled_index, assume_unique=True)

        random_index = np.random.choice(unlabeled_index, 1)[0]

        return [random_index]

    def execute_over_all_inputs_postgres(self, query, is_test=False):
        if is_test:
            input_vids = self.test_inputs.tolist()
        else:
            input_vids = self.inputs[self.sampled_index].tolist()
        pred_per_query = []
        result, new_memoize_scene_graph, new_memoize_sequence = postgres_execute_cache_sequence(self.dsn, query, self.memoize_scene_graph_all_inputs, self.memoize_sequence_all_inputs, self.inputs_table_name, input_vids, is_trajectory=self.is_trajectory, sampling_rate=self.sampling_rate)
        if self.lock:
            self.lock.acquire()
        for i, memo_dict in enumerate(new_memoize_scene_graph):
            for k, v in memo_dict.items():
                self.memoize_scene_graph_all_inputs[i][k] = v
        for i, memo_dict in enumerate(new_memoize_sequence):
            for k, v in memo_dict.items():
                self.memoize_sequence_all_inputs[i][k] = v
        if self.lock:
            self.lock.release()
        for i in input_vids:
            if i in result:
                pred_per_query.append(1)
            else:
                pred_per_query.append(0)
        return pred_per_query