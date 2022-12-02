from collections import deque
from sklearn.metrics import f1_score
import resource
import numpy as np
import time
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
from quivr.utils import print_program, rewrite_program, str_to_program, postgres_execute, postgres_execute_cache_sequence, postgres_execute_no_caching, rewrite_program_postgres, str_to_program_postgres, complexity_cost
import itertools
from sklearn.utils import resample
import pandas as pd

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return '''%s: mem=%s MB
           '''%(point, usage/1024.0 )

# np.random.seed(0)
class BaseMethod:
    # def compute_query_score(self, current_query):
    #     y_pred = []
    #     # for i, (input, label) in enumerate(zip(self.inputs, self.labels)):
    #     for i in self.labeled_index:
    #         input = self.inputs[i]
    #         label = self.labels[i]
    #         if self.lock:
    #             self.lock.acquire()
    #         memoize = self.memoize_all_inputs[i]
    #         if self.lock:
    #             self.lock.release()
    #         result, new_memoize = current_query.execute(input, label, memoize, {})
    #         y_pred.append(int(result[0, len(input[0])] > 0))
    #         if self.lock:
    #             self.lock.acquire()
    #         for k, v in new_memoize.items():
    #             self.memoize_all_inputs[i][k] = v
    #         # self.memoize_all_inputs[i].update(new_memoize)
    #         if self.lock:
    #             self.lock.release()
    #     # print(self.labels[self.labeled_index], y_pred)
    #     # print("cache", len(self.memoize_all_inputs[0]))
    #     # print(using("profile"))
    #     score = f1_score(list(self.labels[self.labeled_index]), y_pred)
    #     return score

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

    def compute_query_score_bootstrapping_postgres(self, current_query):
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

        y_true = list(self.labels[self.labeled_index])
        print("score before bootstrapping", f1_score(y_true, y_pred))
        f1_bootstrapping = []
        for _ in range(100):
            y_true_sampled, y_pred_sampled = resample(y_true, y_pred)  # NOTE: stratify?
            f1 = f1_score(y_true_sampled, y_pred_sampled, zero_division=1)
            f1_bootstrapping.append(f1)
        score = np.mean(f1_bootstrapping) - self.reg_lambda * complexity_cost(current_query)
        print("bootstrapping score: mean {}, std {}".format(score, np.std(f1_bootstrapping)))
        return score

    # def pick_next_segment(self):
    #     """
    #     Pick the next segment to be labeled, based on disagreement among candidate queries (not their weights yet).
    #     """
    #     tuning_par = 1
    #     true_labels = np.array(self.labels)
    #     prediction_matrix = []
    #     _start = time.time()
    #     for i in range(len(self.inputs)):
    #         # print("prediction matrix", i)
    #         input = self.inputs[i]
    #         label = self.labels[i]
    #         if self.lock:
    #             self.lock.acquire()
    #         memoize = self.memoize_all_inputs[i]
    #         if self.lock:
    #             self.lock.release()
    #         pred_per_input = []
    #         for query_graph, _ in self.candidate_queries:
    #             query = query_graph.program
    #             result, new_memoize = query.execute(input, label, memoize, {})
    #             pred_per_input.append(int(result[0, len(input[0])] > 0))
    #             if self.lock:
    #                 self.lock.acquire()
    #             self.memoize_all_inputs[i].update(new_memoize)
    #             if self.lock:
    #                 self.lock.release()
    #         prediction_matrix.append(pred_per_input)
    #     prediction_matrix = np.array(prediction_matrix)
    #     print("constructing prediction matrix", time.time()-_start)
    #     n_instances = len(true_labels)

    #     # # Initialize
    #     # loss_t = np.zeros(self.beam_width)
    #     # for i in range(n_instances):
    #     #     if i not in self.labeled_index:
    #     #         continue
    #     #     if true_labels[i]:
    #     #         loss_t += (np.array((prediction_matrix[i, :] != 1) * 1) * 5)
    #     #         # loss_t += (np.array((prediction_matrix_pair_level[t, :] != 1) * 1) / u_t * (n_pos + n_neg) / (2 * n_pos))
    #     #         # n_pos += 1
    #     #     else:
    #     #         loss_t += (np.array((prediction_matrix[i, :] != 0) * 1))

    #     # eta = np.sqrt(np.log(self.beam_width)/(2*(len(self.labeled_index)+1)))

    #     # posterior_t = np.exp(-eta * (loss_t-np.min(loss_t)))
    #     # # Note that above equation is equivalent to np.exp(-eta * loss_t).
    #     # # `-np.min(loss_t)` is applied only to avoid entries being near zero for large eta*loss_t values before the normalization
    #     # posterior_t  /= np.sum(posterior_t)  # normalized weight

    #     entropy_list = np.zeros(n_instances)
    #     for i in range(n_instances):
    #         if i in self.labeled_index:
    #             entropy_list[i] = -1
    #         else:
    #             # Measure the normalized entropy of the incoming data
    #             hist, bin_edges = np.histogram(prediction_matrix[i, :], bins=2)
    #             prob_i = hist/np.sum(hist)
    #             entropy_i = stats.entropy(prob_i, base=2) * tuning_par
    #             # Check if the normalized entropy is greater than 1
    #             if entropy_i > 1:
    #                 entropy_i = 1
    #             if entropy_i < 0:
    #                 entropy_i = 0
    #             entropy_list[i] = entropy_i
    #     # find argmax of entropy (top k)
    #     n_to_label = 1
    #     video_segment_ids = np.argpartition(entropy_list, -n_to_label)[-n_to_label:]
    #     # max_entropy_index = np.argmax(entropy_list)
    #     return video_segment_ids.tolist()

    # def execute_over_all_inputs(self, query):
    #     pred_per_query = []
    #     for i in range(len(self.inputs)):
    #         input = self.inputs[i]
    #         label = self.labels[i]
    #         if self.lock:
    #             self.lock.acquire()
    #         memoize = self.memoize_all_inputs[i]
    #         if self.lock:
    #             self.lock.release()
    #         result, new_memoize = query.execute(input, label, memoize, {})
    #         pred_per_query.append(int(result[0, len(input[0])] > 0))
    #         if self.lock:
    #             self.lock.acquire()
    #         # self.memoize_all_inputs[i].update(new_memoize)
    #         for k, v in new_memoize.items():
    #             self.memoize_all_inputs[i][k] = v
    #         if self.lock:
    #             self.lock.release()
    #     return pred_per_query

    # def pick_next_segment_model_picker(self):
    #     """
    #     Pick the next segment to be labeled, using the Model Picker algorithm.
    #     """
    #     true_labels = np.array(self.labels)
    #     prediction_matrix = []
    #     _start = time.time()

    #     # query_list = [query_graph.program for query_graph, _ in itertools.chain(self.candidate_queries, self.answers[:max(self.beam_width, 10)])]
    #     query_list = [query_graph.program for query_graph, _ in self.candidate_queries]
    #     query_list_removing_duplicates = []
    #     signatures = set()
    #     for program in query_list:
    #         signature = rewrite_program(program)
    #         if signature not in signatures:
    #             new_program = str_to_program(signature)
    #             query_list_removing_duplicates.append(new_program)
    #             signatures.add(signature)
    #     query_list = query_list_removing_duplicates
    #     print("query pool", [print_program(query) for query in query_list])
    #     if self.multithread > 1:
    #       for pred_per_query in self.executor.map(self.execute_over_all_inputs, query_list):
    #           prediction_matrix.append(pred_per_query)
    #     else:
    #         for query in query_list:
    #             pred_per_query = self.execute_over_all_inputs(query)
    #             prediction_matrix.append(pred_per_query)
    #         prediction_matrix.append(pred_per_query)
    #     prediction_matrix = np.array(prediction_matrix).transpose()
    #     print("constructing prediction matrix", time.time()-_start)
    #     # print(prediction_matrix.shape)
    #     n_instances = len(true_labels)

    #     # Initialize
    #     loss_t = np.zeros(prediction_matrix.shape[1])
    #     for i in range(n_instances):
    #         if i not in self.labeled_index:
    #             continue
    #         if true_labels[i]:
    #             loss_t += (np.array((prediction_matrix[i, :] != 1) * 1) * 5)
    #         else:
    #             loss_t += (np.array((prediction_matrix[i, :] != 0) * 1))

    #     eta = np.sqrt(np.log(prediction_matrix.shape[1])/(2*(len(self.labeled_index)-self.init_nlabels+1)))
    #     posterior_t = np.exp(-eta * (loss_t-np.min(loss_t)))
    #     # Note that above equation is equivalent to np.exp(-eta * loss_t).
    #     # `-np.min(loss_t)` is applied only to avoid entries being near zero for large eta*loss_t values before the normalization
    #     posterior_t  /= np.sum(posterior_t)  # normalized weight
    #     print("query weights", posterior_t)
    #     entropy_list = np.zeros(n_instances)
    #     for i in range(n_instances):
    #         if i in self.labeled_index:
    #             entropy_list[i] = -1
    #         else:
    #             entropy_list[i] = self._compute_u_t(posterior_t, prediction_matrix[i, :])
    #     # find argmax of entropy (top k)
    #     new_labeled_index = np.argpartition(entropy_list, -self.samples_per_iter)[-self.samples_per_iter:]
    #     # max_entropy_index = np.argmax(entropy_list)
    #     return new_labeled_index.tolist()

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

        # query_list = [query_graph.program for query_graph, _ in itertools.chain(self.candidate_queries, self.answers[:max(self.beam_width, 10)])]
        query_list = [query_graph.program for query_graph, _ in self.candidate_queries[:self.pool_size]]
        # query_list_removing_duplicates = []
        # signatures = set()
        # for program in query_list:
        #     signature = rewrite_program_postgres(program)
        #     if signature not in signatures:
        #         new_program = str_to_program_postgres(signature)
        #         query_list_removing_duplicates.append(new_program)
        #         signatures.add(signature)
        # query_list = query_list_removing_duplicates
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


        # # Initialize
        # # TODO: should we use loss or f1-score to compute the weights?
        # loss_t = np.zeros(prediction_matrix.shape[1])
        # for i in range(n_instances):
        #     if i not in self.labeled_index:
        #         continue
        #     if true_labels[i]:
        #         loss_t += (np.array((prediction_matrix[i, :] != 1) * 1) * 10) # TODO: change the weight to 10?
        #     else:
        #         loss_t += (np.array((prediction_matrix[i, :] != 0) * 1))
        # # eta = np.sqrt(np.log(prediction_matrix.shape[1])/(2*(len(self.labeled_index)-self.init_nlabels+1))) # TODO: do we need a decaying learning rate?
        # eta = 1
        # posterior_t = np.exp(-eta * (loss_t-np.min(loss_t)))
        # # Note that above equation is equivalent to np.exp(-eta * loss_t).
        # # `-np.min(loss_t)` is applied only to avoid entries being near zero for large eta*loss_t values before the normalization
        # posterior_t  /= np.sum(posterior_t)  # normalized weight

        # Alternative: use F1-scores as weights
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

    def pick_next_segment_most_likely_positive_postgres(self):
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

        # Alternative: use F1-scores as weights
        posterior_t = [score for _, score in self.candidate_queries[:self.pool_size]]
        posterior_t  /= np.sum(posterior_t)  # normalized weight

        print("query weights", posterior_t)
        scores = np.zeros(len(self.sampled_index))
        for i in range(len(self.sampled_index)):
            scores[i] = np.inner(posterior_t, prediction_matrix[i, :])
        ind = np.argsort(-scores)
        print("entropy list", scores[ind])
        print("sampled index", self.sampled_index[ind])
        # find argmax of entropy (top k)
        most_likely_positive_index = self.sampled_index[np.argmax(scores)]
        return [most_likely_positive_index]
        # video_segment_ids = np.argpartition(entropy_list, -self.samples_per_iter)[-self.samples_per_iter:]
        # return video_segment_ids.tolist()


    def get_disagreement_score(self, candidate_queries):
        """
        Pick the next segment to be labeled, using the Model Picker algorithm.
        """
        true_labels = np.array(self.labels)
        n_instances = len(true_labels)
        prediction_matrix = []
        _start = time.time()

        query_list = [query_graph.program for query_graph, _ in candidate_queries[:self.pool_size]]
        print("query pool", [rewrite_program_postgres(query) for query in query_list])
        unlabeled_index = np.setdiff1d(np.arange(n_instances), self.labeled_index, assume_unique=True)

        # If more than self.n_sampled_videos videos, sample self.n_sampled_videos videos
        if len(unlabeled_index) > self.n_sampled_videos:
            self.sampled_index = np.random.choice(unlabeled_index, self.n_sampled_videos, replace=False)
        else:
            self.sampled_index = unlabeled_index

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


        # # Initialize
        # # TODO: should we use loss or f1-score to compute the weights?
        # loss_t = np.zeros(prediction_matrix.shape[1])
        # for i in range(n_instances):
        #     if i not in self.labeled_index:
        #         continue
        #     if true_labels[i]:
        #         loss_t += (np.array((prediction_matrix[i, :] != 1) * 1) * 10) # TODO: change the weight to 10?
        #     else:
        #         loss_t += (np.array((prediction_matrix[i, :] != 0) * 1))
        # # eta = np.sqrt(np.log(prediction_matrix.shape[1])/(2*(len(self.labeled_index)-self.init_nlabels+1))) # TODO: do we need a decaying learning rate?
        # eta = 1
        # posterior_t = np.exp(-eta * (loss_t-np.min(loss_t)))
        # # Note that above equation is equivalent to np.exp(-eta * loss_t).
        # # `-np.min(loss_t)` is applied only to avoid entries being near zero for large eta*loss_t values before the normalization
        # posterior_t  /= np.sum(posterior_t)  # normalized weight

        # Alternative: use F1-scores as weights
        posterior_t = [score for _, score in candidate_queries[:self.pool_size]]
        posterior_t  /= np.sum(posterior_t)  # normalized weight

        print("query weights", posterior_t)
        entropy_list = np.zeros(len(self.sampled_index))
        for i in range(len(self.sampled_index)):
            entropy_list[i] = self._compute_u_t(posterior_t, prediction_matrix[i, :])
        max_entropy_index = self.sampled_index[np.argmax(entropy_list)]
        ind = np.argsort(-entropy_list)
        print("entropy list", entropy_list[ind])
        print("sampled index", self.sampled_index[ind])
        return np.max(entropy_list), [max_entropy_index]


    def execute_over_all_inputs_postgres(self, query):
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



    # def compute_query_score_duckdb(self, current_query):
    #     # NOTE: sufficinet to lock only when writing to the memoize_all_inputs? Updating dict/list in python is atomic operation, so no conflicts for write, but reading might get old values (which is fine for us).
    #     self.input_vids = self.inputs[self.labeled_index].tolist()
    #     y_pred = []
    #     result, new_memoize_scene_graph, new_memoize_sequence = self.duckdb_execute_cache_sequence(current_query, self.input_vids)
    #     if self.lock:
    #         self.lock.acquire()
    #     for i, memo_dict in enumerate(new_memoize_scene_graph):
    #         for k, v in memo_dict.items():
    #             self.memoize_scene_graph_all_inputs[i][k] = v
    #     for i, memo_dict in enumerate(new_memoize_sequence):
    #         for k, v in memo_dict.items():
    #             self.memoize_sequence_all_inputs[i][k] = v
    #     if self.lock:
    #         self.lock.release()
    #     for i in self.input_vids:
    #         if i in result:
    #             y_pred.append(1)
    #         else:
    #             y_pred.append(0)

    #     f1 = f1_score(list(self.labels[self.labeled_index]), y_pred)
    #     score = f1 - self.reg_lambda * complexity_cost(current_query)
    #     return score

    # def duckdb_execute_cache_sequence(self, current_query):
    #     """
    #     input_vids: list of video segment ids
    #     Example query:
    #         Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Near_1.05), True*), Conjunction(LeftOf, Behind)), True*), Duration(Conjunction(TopQuadrant, Far_0.9), 5)), True*)
    #     two types of caches:
    #         1. scene graph cache (without duration constraints): cache[graph] = vid, fid, oids (where fid is every frame that satisfies the graph)
    #         2. sequence cache: cache[sequence] = vid, fid, oids (where fid is the minimum frame that satisfies the sequence)
    #         Example: g1, (g1, d1), g2, (g1, d1); (g2, d2), g3, (g1, d1); (g2, d2); (g3, d3)
    #     Output:
    #     new_memoize: new cached results from this query, which will be added to the global cache (for multi-threading)
    #     """
    #     with self.con.cursor() as cur:
    #         """
    #         Caching implementation:
    #         cached_df_deque = []
    #         cached_vids_deque = []
    #         remaining_vids = input_vids
    #         for each query (from the most top-level to the most bottom-level):
    #             cached_df_per_query = []
    #             cached_vids_per_query = []
    #             next_remaining_vids = []
    #             for each video segment in remaining_vids:
    #                 if the result is cached:
    #                     add the result to cached_df_per_query
    #                     add the video segment id to cached_vids_per_query
    #                 else:
    #                     add the video segment id to next_remaining_vids
    #             push cached_df_per_query to cached_df_deque
    #             push cached_vids_per_query to cached_vids_deque
    #             remaining_vids = next_remaining_vids
    #         """
    #         new_memoize_scene_graph = [{} for _ in range(len(self.memoize_scene_graph))]
    #         new_memoize_sequence = [{} for _ in range(len(self.memoize_sequence))]

    #         # Prepare cache result
    #         filtered_vids = []
    #         cached_df_seq_deque = deque()
    #         cached_vids_deque = deque()
    #         if isinstance(self.input_vids, int):
    #             remaining_vids = set(range(self.input_vids))
    #         else:
    #             remaining_vids = set(self.input_vids)
    #         signatures = deque()
    #         for i in range(len(current_query)):
    #             cached_df_seq_per_query = [pd.DataFrame()]

    #             # sequence cache
    #             seq_signature = rewrite_program_postgres(current_query[:len(current_query)-i])
    #             cached_vids_per_query = set()
    #             next_remaining_vids = set()
    #             for vid in remaining_vids:
    #                 if seq_signature in self.memoize_sequence[vid]:
    #                     cached_df_seq_per_query.append(self.memoize_sequence[vid][seq_signature])
    #                     cached_vids_per_query.add(vid)
    #                 else:
    #                     next_remaining_vids.add(vid)
    #             cached_df_seq_per_query = pd.concat(cached_df_seq_per_query, ignore_index=True)
    #             cached_df_seq_deque.append(cached_df_seq_per_query)
    #             cached_vids_deque.append(cached_vids_per_query)
    #             if i == 0:
    #                 filtered_vids = list(next_remaining_vids)
    #             remaining_vids = next_remaining_vids

    #             signatures.append(seq_signature)
    #         cached_vids_deque.append(remaining_vids)
    #         # print("filtered_vids", filtered_vids)
    #         # select input videos
    #         _start = time.time()
    #         if isinstance(self.input_vids, int):
    #             if self.sampling_rate:
    #                 cur.execute("CREATE TEMPORARY TABLE Obj_filtered AS SELECT oid, vid, fid / {} as fid, shape, color, material, x1, y1, x2, y2 FROM Obj WHERE vid < {} AND fid % {} = 0;".format(self.sampling_rate, self.input_vids, self.sampling_rate))
    #             else:
    #                 cur.execute("CREATE TEMPORARY TABLE Obj_filtered AS SELECT * FROM Obj WHERE vid < {};".format(self.input_vids))
    #         else:
    #             if self.sampling_rate:
    #                 cur.execute("CREATE TEMPORARY TABLE Obj_filtered AS SELECT oid, vid, fid / {} as fid, shape, color, material, x1, y1, x2, y2 FROM Obj WHERE vid = ANY(%s) AND fid %% {} = 0;".format(self.sampling_rate, self.sampling_rate), [filtered_vids])
    #             else:
    #                 cur.execute("CREATE TEMPORARY TABLE Obj_filtered AS SELECT * FROM Obj WHERE vid = ANY(%s);", [filtered_vids])
    #         cur.execute("CREATE INDEX IF NOT EXISTS idx_obj_filtered ON Obj_filtered (vid, fid, oid);")
    #         # print("select input videos: ", time.time() - _start)

    #         encountered_variables_prev_graphs = []
    #         encountered_variables_current_graph = []
    #         delta_input_vids = []
    #         for graph_idx, dict in enumerate(current_query):
    #             _start = time.time()
    #             # Generate scene graph:
    #             scene_graph = dict["scene_graph"]
    #             duration_constraint = dict["duration_constraint"]
    #             for p in scene_graph:
    #                 for v in p["variables"]:
    #                     if v not in encountered_variables_current_graph:
    #                         encountered_variables_current_graph.append(v)

    #             delta_input_vids.extend(cached_vids_deque.pop())
    #             # Execute for unseen videos
    #             _start_execute = time.time()
    #             encountered_variables_current_graph = sorted(encountered_variables_current_graph, key=lambda x: int(x[1:]))
    #             tables = ", ".join(["Obj_filtered as {}".format(v) for v in encountered_variables_current_graph])
    #             where_clauses = []
    #             where_clauses.append("{}.vid = ANY(%s)".format(encountered_variables_current_graph[0]))
    #             for i in range(len(encountered_variables_current_graph)-1):
    #                 where_clauses.append("{v1}.vid = {v2}.vid and {v1}.fid = {v2}.fid".format(v1=encountered_variables_current_graph[i], v2=encountered_variables_current_graph[i+1])) # join variables
    #             for p in scene_graph:
    #                 predicate = p["predicate"]
    #                 parameter = p["parameter"]
    #                 variables = p["variables"]
    #                 args = []
    #                 for v in variables:
    #                     args.append("{v}.shape, {v}.color, {v}.material, {v}.x1, {v}.y1, {v}.x2, {v}.y2".format(v=v))
    #                 args = ", ".join(args)
    #                 if parameter:
    #                     if isinstance(parameter, str):
    #                         args = "'{}', {}".format(parameter, args)
    #                     else:
    #                         args = "{}, {}".format(parameter, args)
    #                 where_clauses.append("{}({}) = true".format(predicate, args))
    #             if is_trajectory:
    #                 # only for trajectory example
    #                 for v in encountered_variables_current_graph:
    #                     where_clauses.append("{}.oid = {}".format(v, v[1:]))
    #             else:
    #                 # For general case
    #                 for var_pair in itertools.combinations(encountered_variables_current_graph, 2):
    #                     where_clauses.append("{}.oid <> {}.oid".format(var_pair[0], var_pair[1]))
    #             where_clauses = " and ".join(where_clauses)
    #             fields = "{v}.vid as vid, {v}.fid as fid, ".format(v=encountered_variables_current_graph[0])
    #             fields += ", ".join(["{v}.oid as {v}_oid".format(v=v) for v in encountered_variables_current_graph])
    #             oid_list = ["{}_oid".format(v) for v in encountered_variables_current_graph]
    #             oids = ", ".join(oid_list)
    #             sql_sring = """
    #             CREATE TEMPORARY TABLE g{} AS
    #             SELECT {}
    #             FROM {}
    #             WHERE {};
    #             """.format(graph_idx, fields, tables, where_clauses)
    #             # print(sql_sring)
    #             cur.execute(sql_sring, [delta_input_vids])
    #             # cur.execute("CREATE INDEX IF NOT EXISTS idx_g{} ON g{} (vid);".format(graph_idx, graph_idx))
    #             # print("execute for unseen videos: ", time.time() - _start_execute)
    #             # print("Time for graph {}: {}".format(graph_idx, time.time() - _start))

    #             # Read cached results
    #             seq_signature = signatures.pop()
    #             cached_results = cached_df_seq_deque.pop()

    #             _start_filtered = time.time()
    #             if graph_idx > 0:
    #                 obj_union = copy.deepcopy(encountered_variables_prev_graphs)
    #                 obj_intersection = []
    #                 obj_union_fields = []
    #                 obj_intersection_fields = []
    #                 for v in encountered_variables_prev_graphs:
    #                     obj_union_fields.append("t0.{}_oid".format(v))
    #                 for v in encountered_variables_current_graph:
    #                     if v in encountered_variables_prev_graphs:
    #                         obj_intersection.append(v)
    #                         obj_intersection_fields.append("t0.{v}_oid = t1.{v}_oid".format(v=v))
    #                     else:
    #                         obj_union.append(v)
    #                         obj_union_fields.append("t1.{}_oid".format(v))
    #                 obj_union_fields = ", ".join(obj_union_fields)
    #                 obj_intersection_fields = " and ".join(obj_intersection_fields)
    #                 # where_clauses = "t0.vid = ANY(%s)"
    #                 # if current_seq == "g0_seq_view":
    #                 #     where_clauses += " and t0.vid = t1.vid and t0.fid2 < t1.fid1"
    #                 # else:
    #                 #     where_clauses += " and t0.vid = t1.vid and t0.fid < t1.fid1"
    #                 sql_string = """
    #                 CREATE TEMPORARY TABLE g{graph_idx}_filtered AS (
    #                     SELECT t0.vid, t1.fid, {obj_union_fields}
    #                     FROM g{graph_idx_prev}_contiguous t0, g{graph_idx} t1
    #                     WHERE t0.vid = t1.vid AND {obj_intersection_fields} AND t0.fid < t1.fid
    #                 );
    #                 """.format(graph_idx=graph_idx, graph_idx_prev=graph_idx-1, obj_union_fields=obj_union_fields, obj_intersection_fields=obj_intersection_fields)
    #                 # print(sql_string)
    #                 cur.execute(sql_string)
    #             else:
    #                 obj_union = encountered_variables_current_graph
    #             # print("filtered: ", time.time() - _start_filtered)

    #             # Generate scene graph sequence:
    #             _start_windowed = time.time()
    #             table_name = "g{}_filtered".format(graph_idx) if graph_idx > 0 else "g{}".format(graph_idx)
    #             obj_union_fields = ", ".join(["{}_oid".format(v) for v in obj_union])
    #             sql_string = """
    #                 CREATE TEMPORARY TABLE g{graph_idx}_windowed AS (
    #                 SELECT vid, fid, {obj_union_fields},
    #                 lead(fid, {duration_constraint} - 1, 0) OVER (PARTITION BY vid, {obj_union_fields} ORDER BY fid) as fid_offset
    #                 FROM {table_name}
    #             );
    #             """.format(graph_idx=graph_idx, duration_constraint=duration_constraint, obj_union_fields=obj_union_fields, table_name=table_name)
    #             # print(sql_string)
    #             cur.execute(sql_string)
    #             # print("windowed: ", time.time() - _start_windowed)

    #             _start_contiguous = time.time()
    #             sql_string = """
    #                 CREATE TEMPORARY TABLE g{graph_idx}_contiguous AS (
    #                 SELECT vid, {obj_union_fields}, min(fid_offset) AS fid
    #                 FROM g{graph_idx}_windowed
    #                 WHERE fid_offset = fid + ({duration_constraint} - 1)
    #                 GROUP BY vid, {obj_union_fields}
    #             );
    #             """.format(graph_idx=graph_idx, obj_union_fields=obj_union_fields, duration_constraint=duration_constraint)
    #             # print(sql_string)
    #             cur.execute(sql_string)
    #             # print("contiguous: ", time.time() - _start_contiguous)
    #             # Store new cached results
    #             for input_vid in delta_input_vids:
    #                 new_memoize_sequence[input_vid][seq_signature] = pd.DataFrame()
    #             _start_execute = time.time()
    #             cur.execute("SELECT * FROM g{}_contiguous".format(graph_idx))
    #             df = pd.DataFrame(cur.fetchall())
    #             # print("[store cache]: fetchall", time.time() - _start_execute)
    #             _start_store = time.time()
    #             if df.shape[0]: # if results not empty
    #                 df.columns = [x.name for x in cur.description]
    #                 for vid, group in df.groupby("vid"):
    #                     cached_df = group.reset_index(drop=True)
    #                     new_memoize_sequence[vid][seq_signature] = cached_df
    #             # print("[store cache]: store", time.time() - _start_store)
    #             # Appending cached results of seen videos:
    #             _start_append = time.time()
    #             if cached_results.shape[0]:
    #                 # save dataframe to an in memory buffer
    #                 buffer = StringIO()
    #                 cached_results.to_csv(buffer, header=False, index = False)
    #                 buffer.seek(0)
    #                 cur.copy_from(buffer, "g{}_contiguous".format(graph_idx), sep=",")
    #             # print("append: ", time.time() - _start_append)
    #             encountered_variables_prev_graphs = obj_union
    #             encountered_variables_current_graph = []

    #         cur.execute("SELECT DISTINCT vid FROM g{}_contiguous".format(len(current_query) - 1))
    #         # print("SELECT DISTINCT vid FROM g{}_contiguous".format(len(current_query) - 1))
    #         output_vids = cur.fetchall()
    #         output_vids = [row[0] for row in output_vids]
    #     return output_vids, new_memoize_scene_graph, new_memoize_sequence