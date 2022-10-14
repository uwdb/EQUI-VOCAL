from sklearn.metrics import f1_score
import resource
import random
import numpy as np
import time
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
from quivr.utils import print_program, rewrite_program, str_to_program, postgres_execute, rewrite_program_postgres, str_to_program_postgres
import itertools

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return '''%s: mem=%s MB
           '''%(point, usage/1024.0 )

random.seed(time.time())
# random.seed(10)
class BaseMethod:
    def compute_query_score(self, current_query):
        y_pred = []
        # for i, (input, label) in enumerate(zip(self.inputs, self.labels)):
        for i in self.labeled_index:
            input = self.inputs[i]
            label = self.labels[i]
            if self.lock:
                self.lock.acquire()
            memoize = self.memoize_all_inputs[i]
            if self.lock:
                self.lock.release()
            result, new_memoize = current_query.execute(input, label, memoize, {})
            y_pred.append(int(result[0, len(input[0])] > 0))
            if self.lock:
                self.lock.acquire()
            for k, v in new_memoize.items():
                self.memoize_all_inputs[i][k] = v
            # self.memoize_all_inputs[i].update(new_memoize)
            if self.lock:
                self.lock.release()
        # print(self.labels[self.labeled_index], y_pred)
        # print("cache", len(self.memoize_all_inputs[0]))
        # print(using("profile"))
        score = f1_score(list(self.labels[self.labeled_index]), y_pred)
        return score

    def compute_query_score_postgres(self, current_query):
        y_pred = []
        result, new_memoize = postgres_execute(self.dsn, current_query, self.labeled_index, self.memoize_all_inputs, self.inputs_table_name)
        if self.lock:
            self.lock.acquire()
        for i, memo_dict in enumerate(new_memoize):
            for k, v in memo_dict.items():
                self.memoize_all_inputs[i][k] = v
        if self.lock:
            self.lock.release()
        for i in self.labeled_index:
            if i in result:
                y_pred.append(1)
            else:
                y_pred.append(0)

        score = f1_score(list(self.labels[self.labeled_index]), y_pred)
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

    def execute_over_all_inputs(self, query):
        pred_per_query = []
        for i in range(len(self.inputs)):
            input = self.inputs[i]
            label = self.labels[i]
            if self.lock:
                self.lock.acquire()
            memoize = self.memoize_all_inputs[i]
            if self.lock:
                self.lock.release()
            result, new_memoize = query.execute(input, label, memoize, {})
            pred_per_query.append(int(result[0, len(input[0])] > 0))
            if self.lock:
                self.lock.acquire()
            # self.memoize_all_inputs[i].update(new_memoize)
            for k, v in new_memoize.items():
                self.memoize_all_inputs[i][k] = v
            if self.lock:
                self.lock.release()
        return pred_per_query

    def pick_next_segment_model_picker(self):
        """
        Pick the next segment to be labeled, using the Model Picker algorithm.
        """
        true_labels = np.array(self.labels)
        prediction_matrix = []
        _start = time.time()

        # query_list = [query_graph.program for query_graph, _ in itertools.chain(self.candidate_queries, self.answers[:max(self.beam_width, 10)])]
        query_list = [query_graph.program for query_graph, _ in self.candidate_queries]
        query_list_removing_duplicates = []
        signatures = set()
        for program in query_list:
            signature = rewrite_program(program)
            if signature not in signatures:
                new_program = str_to_program(signature)
                query_list_removing_duplicates.append(new_program)
                signatures.add(signature)
        query_list = query_list_removing_duplicates
        print("query pool", [print_program(query) for query in query_list])
        if self.multithread > 1:
            with ThreadPoolExecutor(max_workers=self.multithread) as executor:
                for pred_per_query in executor.map(self.execute_over_all_inputs, query_list):
                    prediction_matrix.append(pred_per_query)
        else:
            for query in query_list:
                pred_per_query = self.execute_over_all_inputs(query)
                prediction_matrix.append(pred_per_query)
            prediction_matrix.append(pred_per_query)
        prediction_matrix = np.array(prediction_matrix).transpose()
        print("constructing prediction matrix", time.time()-_start)
        # print(prediction_matrix.shape)
        n_instances = len(true_labels)

        # Initialize
        loss_t = np.zeros(prediction_matrix.shape[1])
        for i in range(n_instances):
            if i not in self.labeled_index:
                continue
            if true_labels[i]:
                loss_t += (np.array((prediction_matrix[i, :] != 1) * 1) * 5)
            else:
                loss_t += (np.array((prediction_matrix[i, :] != 0) * 1))

        eta = np.sqrt(np.log(prediction_matrix.shape[1])/(2*(len(self.labeled_index)-self.init_nlabels+1)))
        posterior_t = np.exp(-eta * (loss_t-np.min(loss_t)))
        # Note that above equation is equivalent to np.exp(-eta * loss_t).
        # `-np.min(loss_t)` is applied only to avoid entries being near zero for large eta*loss_t values before the normalization
        posterior_t  /= np.sum(posterior_t)  # normalized weight
        print("query weights", posterior_t)
        entropy_list = np.zeros(n_instances)
        for i in range(n_instances):
            if i in self.labeled_index:
                entropy_list[i] = -1
            else:
                entropy_list[i] = self._compute_u_t(posterior_t, prediction_matrix[i, :])
        # find argmax of entropy (top k)
        video_segment_ids = np.argpartition(entropy_list, -self.samples_per_iter)[-self.samples_per_iter:]
        # max_entropy_index = np.argmax(entropy_list)
        return video_segment_ids.tolist()

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
        prediction_matrix = []
        _start = time.time()

        # query_list = [query_graph.program for query_graph, _ in itertools.chain(self.candidate_queries, self.answers[:max(self.beam_width, 10)])]
        query_list = [query_graph.program for query_graph, _ in self.candidate_queries]
        query_list_removing_duplicates = []
        signatures = set()
        for program in query_list:
            signature = rewrite_program_postgres(program)
            if signature not in signatures:
                new_program = str_to_program_postgres(signature)
                query_list_removing_duplicates.append(new_program)
                signatures.add(signature)
        query_list = query_list_removing_duplicates
        print("query pool", [rewrite_program_postgres(query) for query in query_list])
        if self.multithread > 1:
            with ThreadPoolExecutor(max_workers=self.multithread) as executor:
                for pred_per_query in executor.map(self.execute_over_all_inputs_postgres, query_list):
                    prediction_matrix.append(pred_per_query)
        else:
            for query in query_list:
                pred_per_query = self.execute_over_all_inputs_postgres(query)
                prediction_matrix.append(pred_per_query)
            prediction_matrix.append(pred_per_query)
        prediction_matrix = np.array(prediction_matrix).transpose()
        print("constructing prediction matrix", time.time()-_start)
        # print(prediction_matrix.shape)
        n_instances = len(true_labels)

        # Initialize
        loss_t = np.zeros(prediction_matrix.shape[1])
        for i in range(n_instances):
            if i not in self.labeled_index:
                continue
            if true_labels[i]:
                loss_t += (np.array((prediction_matrix[i, :] != 1) * 1) * 5)
            else:
                loss_t += (np.array((prediction_matrix[i, :] != 0) * 1))

        eta = np.sqrt(np.log(prediction_matrix.shape[1])/(2*(len(self.labeled_index)-self.init_nlabels+1)))
        posterior_t = np.exp(-eta * (loss_t-np.min(loss_t)))
        # Note that above equation is equivalent to np.exp(-eta * loss_t).
        # `-np.min(loss_t)` is applied only to avoid entries being near zero for large eta*loss_t values before the normalization
        posterior_t  /= np.sum(posterior_t)  # normalized weight
        print("query weights", posterior_t)
        entropy_list = np.zeros(n_instances)
        for i in range(n_instances):
            if i in self.labeled_index:
                entropy_list[i] = -1
            else:
                entropy_list[i] = self._compute_u_t(posterior_t, prediction_matrix[i, :])
        # find argmax of entropy (top k)
        video_segment_ids = np.argpartition(entropy_list, -self.samples_per_iter)[-self.samples_per_iter:]
        # max_entropy_index = np.argmax(entropy_list)
        return video_segment_ids.tolist()

    def execute_over_all_inputs_postgres(self, query):
        pred_per_query = []
        result, new_memoize = postgres_execute(self.dsn, query, list(range(len(self.inputs))), self.memoize_all_inputs, self.inputs_table_name)
        if self.lock:
            self.lock.acquire()
        for i, memo_dict in enumerate(new_memoize):
            for k, v in memo_dict.items():
                self.memoize_all_inputs[i][k] = v
        if self.lock:
            self.lock.release()
        for i in range(len(self.inputs)):
            if i in result:
                pred_per_query.append(1)
            else:
                pred_per_query.append(0)
        return pred_per_query