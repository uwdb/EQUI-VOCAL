from quivr.methods.base_method import BaseMethod
from quivr.utils import print_program
from quivr.query_graph import QueryGraph
import copy
import numpy as np
import time
from scipy import stats
import itertools
from lru import LRU
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import resource
import random
import quivr.dsl as dsl

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return '''%s: mem=%s MB
           '''%(point, usage/1024.0 )

random.seed(10)

class VOCAL(BaseMethod):

    def __init__(self, inputs, labels, predicate_dict, max_num_atomic_predicates=5, max_depth=2, k1=10, k2=20, budget=200, thresh=0.5, max_duration=2, multithread=1):
        self.inputs = np.array(inputs, dtype=object)
        self.labels = np.array(labels, dtype=object)
        self.predicate_dict = predicate_dict
        self.max_num_atomic_predicates = max_num_atomic_predicates
        self.max_depth = max_depth
        self.k1 = k1
        self.k2 = k2
        self.budget = budget
        self.thresh = thresh
        self.max_duration = max_duration
        self.multithread = multithread

        self.query_expansion_time = 0
        self.segment_selection_time = 0
        self.retain_top_k2_queries_time = 0
        self.answers = []

        if self.multithread > 1:
            self.m = multiprocessing.Manager()
            self.lock = self.m.Lock()
        elif self.multithread == 1:
            self.lock = None
        else:
            raise ValueError("multithread must be 1 or greater")

    def run(self, init_labeled_index):
        _start_total_time = time.time()
        self.labeled_index = init_labeled_index
        self.memoize_all_inputs = [LRU(10000) for _ in range(len(self.inputs))] # For each (input, label) pair, memoize the results of all sub-queries encountered during synthesis.

        # Initialize candidate queries: [query_graph, score]
        self.candidate_queries = []

        for pred in self.predicate_dict:
            pred_instances = []
            if self.predicate_dict[pred]:
                for param in self.predicate_dict[pred]:
                    pred_instances.append(pred(param))
            else:
                pred_instances.append(pred())
            for pred_instance in pred_instances:
                query_graph = QueryGraph(self.max_num_atomic_predicates, self.max_depth)
                query_graph.program = dsl.SequencingOperator(dsl.SequencingOperator(dsl.TrueStar(), pred_instance), dsl.TrueStar())
                query_graph.num_atomic_predicates = 1
                query_graph.depth = 1
                score = self.compute_query_score(query_graph.program)
                print("initialization", print_program(query_graph.program), score)
                self.candidate_queries.append([query_graph, score])
                self.answers.append([query_graph, score])

        while len(self.candidate_queries) and len(self.labeled_index) < self.budget:
            # Generate more queries
            _start_query_expansion_time = time.time()
            new_candidate_queries = []

            if self.multithread > 1:
                with ThreadPoolExecutor(max_workers=self.multithread) as executor:
                    for result in executor.map(self.expand_query_and_compute_score, self.candidate_queries):
                        new_candidate_queries.extend(result)
            else:
                for query_graph, _ in self.candidate_queries:
                    new_candidate_queries.extend(self.expand_query_and_compute_score(query_graph))
            self.answers.extend(new_candidate_queries)
            self.query_expansion_time += time.time() - _start_query_expansion_time

            # Sample k1 queries
            if len(new_candidate_queries):
                new_candidate_queries = np.asarray(new_candidate_queries, dtype=object)
                weight = new_candidate_queries[:, 1].astype(np.float)
                weight = weight / np.sum(weight)
                candidate_idx = np.random.choice(np.arange(new_candidate_queries.shape[0]), size=min(self.k1, new_candidate_queries.shape[0]), replace=False, p=weight)
                print("candidate_idx", candidate_idx)
                self.candidate_queries = new_candidate_queries[candidate_idx].tolist()
                print("k1 queries", [(print_program(query.program), score) for query, score in self.candidate_queries])
            else:
                self.candidate_queries = []

            # Baseline: keep all queries
            # self.candidate_queries = new_candidate_queries

            # Retain the top k2 queries with the highest scores as answers
            _start_retain_top_k2_queries_time = time.time()
            for i in range(len(self.answers)):
                self.answers[i][1] = self.compute_query_score(self.answers[i][0].program)
            self.answers = sorted(self.answers, key=lambda x: x[1], reverse=True)
            self.answers = self.answers[:self.k2]
            print("top k2 queries", [(print_program(query.program), score) for query, score in self.answers])
            self.retain_top_k2_queries_time += time.time() - _start_retain_top_k2_queries_time

            # Select new video segments to label
            _start_segmnet_selection_time = time.time()
            # video_segment_ids = self.pick_next_segment()
            video_segment_ids = self.pick_next_segment_model_picker()
            print("pick next segments", video_segment_ids)
            self.labeled_index += video_segment_ids
            print("# labeled segments", len(self.labeled_index))
            # assert labeled_index does not contain duplicates
            assert(len(self.labeled_index) == len(set(self.labeled_index)))
            self.segment_selection_time += time.time() - _start_segmnet_selection_time

        # RETURN: the list.
        print("final_answers")
        for query_graph, score in self.answers:
            print("answer", print_program(query_graph.program), score)

        print("[Runtime] query expansion time: {}, segment selection time: {}, retain top k2 queries time: {}, total time: {}".format(self.query_expansion_time, self.segment_selection_time, self.retain_top_k2_queries_time, time.time() - _start_total_time))

        return self.answers

    def expand_query_and_compute_score(self, current_query):
        current_query_graph, _ = current_query
        current_query = current_query_graph.program
        print("expand search space", print_program(current_query))
        all_children = current_query_graph.get_all_children_bu(self.predicate_dict, self.max_duration)

        new_candidate_queries = []

        # Compute F1 score of each candidate query
        for child in all_children:
            score = self.compute_query_score(child.program)
            if score > 0:
                new_candidate_queries.append([child, score])
                print(print_program(child.program), score)
        return new_candidate_queries


    def prune_parameter_values(self, current_query, inputs, labels):
        # current_query is sketch (containing only parameter holes)
        # 1. Find all parameter holes
        # 2. For each parameter hole, fix it and fill in other holes with -inf or inf.
        if QueryGraph.is_complete(current_query):
            return
            # self.consistent_queries.append(current_query)
        queue = [current_query]
        while len(queue) != 0:
            current = queue.pop(0)
            for submod, functionclass in current.submodules.items():
                if isinstance(functionclass, dsl.ParameterHole):
                    orig_fclass = copy.deepcopy(current.submodules[submod])
                    value_range = functionclass.get_value_range()
                    step = functionclass.get_step()
                    # Update the current parameter hole to parameter class (but theta is still unset)
                    predicate = copy.deepcopy(functionclass.get_predicate())
                    predicate.with_hole = True
                    current.submodules[submod] = predicate
                    new_query = copy.deepcopy(current_query)
                    value_interval = self.fill_other_parameter_holes(new_query, inputs, labels, value_range)
                    print(value_interval)
                    if value_interval[0] > value_interval[1]:
                        return
                    orig_fclass.value_range = value_interval
                    current.submodules[submod] = orig_fclass
                    continue
                if issubclass(type(functionclass), dsl.Predicate):
                    continue
                else:
                    #add submodules
                    queue.append(functionclass)
        # 3. Construct all (candidate) consistent queries
        self.fill_all_parameter_holes(current_query)

    def fill_other_parameter_holes(self, new_query, inputs, labels, value_range):
        theta_lb = value_range[0]
        theta_ub = value_range[1]
        for i, (input, label) in enumerate(zip(inputs, labels)):
            if self.lock:
                self.lock.acquire()
            memoize = self.memoize_all_inputs[i]
            if self.lock:
                self.lock.release()
            result, new_memoize = new_query.execute(input, label, memoize, {})
            if self.lock:
                self.lock.acquire()
            self.memoize_all_inputs[i].update(new_memoize)
            if self.lock:
                self.lock.release()
            if label == 1:
                theta_ub = min(result[0, len(input[0])], theta_ub)
            if label == 0:
                theta_lb = max(result[0, len(input[0])], theta_lb)
            if theta_lb > theta_ub:
                break
        return [theta_lb, theta_ub]

    def fill_all_parameter_holes(self, current_query):
        if QueryGraph.is_complete(current_query):
            score = self.compute_query_score(current_query)
            if score > self.thresh:
                # TODO: should be query graph instead of query
                query_graph = QueryGraph(self.max_num_atomic_predicates, self.max_depth)
                query_graph.program = current_query
                print("add complete query", print_program(current_query))
                self.candidate_list.append([query_graph, score])
            return
        queue = [current_query]
        while len(queue) != 0:
            current = queue.pop(0)
            for submod, functionclass in current.submodules.items():
                if isinstance(functionclass, dsl.PredicateHole):
                    raise ValueError
                if issubclass(type(functionclass), dsl.Predicate):
                    continue
                if isinstance(functionclass, dsl.ParameterHole):
                    assert(isinstance(functionclass.predicate, dsl.MinLength))
                    orig_fclass = copy.deepcopy(current.submodules[submod])
                    value_range = functionclass.get_value_range()
                    step = functionclass.get_step()
                    for theta in np.arange(value_range[0], value_range[1] + 1e-6, step):
                        predicate = functionclass.fill_hole(theta)
                        # print("fill hole", predicate.name, theta)
                        # replace the parameter hole with a predicate
                        current.submodules[submod] = predicate
                        # create the correct child node
                        new_query = copy.deepcopy(current_query)
                        if not QueryGraph.is_sketch(new_query):
                            raise ValueError
                            # print("here1")
                        self.fill_all_parameter_holes(new_query)
                    current.submodules[submod] = orig_fclass
                    return
                else:
                    #add submodules
                    queue.append(functionclass)

    def pick_next_segment(self):
        """
        Pick the next segment to be labeled, based on disagreement among candidate queries (not their weights yet).
        """
        tuning_par = 1
        true_labels = np.array(self.labels)
        prediction_matrix = []
        _start = time.time()
        for i in range(len(self.inputs)):
            # print("prediction matrix", i)
            input = self.inputs[i]
            label = self.labels[i]
            if self.lock:
                self.lock.acquire()
            memoize = self.memoize_all_inputs[i]
            if self.lock:
                self.lock.release()
            pred_per_input = []
            for query_graph, _ in self.candidate_queries:
                query = query_graph.program
                result, new_memoize = query.execute(input, label, memoize, {})
                pred_per_input.append(int(result[0, len(input[0])] > 0))
                if self.lock:
                    self.lock.acquire()
                self.memoize_all_inputs[i].update(new_memoize)
                if self.lock:
                    self.lock.release()
            prediction_matrix.append(pred_per_input)
        prediction_matrix = np.array(prediction_matrix)
        print("constructing prediction matrix", time.time()-_start)
        n_instances = len(true_labels)

        # # Initialize
        # loss_t = np.zeros(self.k1)
        # for i in range(n_instances):
        #     if i not in self.labeled_index:
        #         continue
        #     if true_labels[i]:
        #         loss_t += (np.array((prediction_matrix[i, :] != 1) * 1) * 5)
        #         # loss_t += (np.array((prediction_matrix_pair_level[t, :] != 1) * 1) / u_t * (n_pos + n_neg) / (2 * n_pos))
        #         # n_pos += 1
        #     else:
        #         loss_t += (np.array((prediction_matrix[i, :] != 0) * 1))

        # eta = np.sqrt(np.log(self.k1)/(2*(len(self.labeled_index)+1)))

        # posterior_t = np.exp(-eta * (loss_t-np.min(loss_t)))
        # # Note that above equation is equivalent to np.exp(-eta * loss_t).
        # # `-np.min(loss_t)` is applied only to avoid entries being near zero for large eta*loss_t values before the normalization
        # posterior_t  /= np.sum(posterior_t)  # normalized weight

        entropy_list = np.zeros(n_instances)
        for i in range(n_instances):
            if i in self.labeled_index:
                entropy_list[i] = -1
            else:
                # Measure the normalized entropy of the incoming data
                hist, bin_edges = np.histogram(prediction_matrix[i, :], bins=2)
                prob_i = hist/np.sum(hist)
                entropy_i = stats.entropy(prob_i, base=2) * tuning_par
                # Check if the normalized entropy is greater than 1
                if entropy_i > 1:
                    entropy_i = 1
                if entropy_i < 0:
                    entropy_i = 0
                entropy_list[i] = entropy_i
        # find argmax of entropy (top k)
        n_to_label = 1
        video_segment_ids = np.argpartition(entropy_list, -n_to_label)[-n_to_label:]
        # max_entropy_index = np.argmax(entropy_list)
        return video_segment_ids.tolist()

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
            self.memoize_all_inputs[i].update(new_memoize)
            if self.lock:
                self.lock.release()
        return pred_per_query

    def pick_next_segment_model_picker(self):
        """
        Pick the next segment to be labeled, using the Model Picker algorithm.
        """
        # tuning_par = 1
        true_labels = np.array(self.labels)
        prediction_matrix = []
        _start = time.time()
        # for i in range(len(self.inputs)):
        #     # print("prediction matrix", i)
        #     input = self.inputs[i]
        #     label = self.labels[i]
        #     if self.lock:
        #         self.lock.acquire()
        #     memoize = self.memoize_all_inputs[i]
        #     if self.lock:
        #         self.lock.release()
        #     pred_per_input = []
        #     for query_graph in itertools.chain(self.candidate_queries, [item[0] for item in self.answers]):
        #         query = query_graph.program
        #         # print("test", print_program(query))
        #         result, new_memoize = query.execute(input, label, memoize, {})
        #         pred_per_input.append(int(result[0, len(input[0])] > 0))
        #         if self.lock:
        #             self.lock.acquire()
        #         self.memoize_all_inputs[i].update(new_memoize)
        #         if self.lock:
        #             self.lock.release()
        #     prediction_matrix.append(pred_per_input)
        # prediction_matrix = np.array(prediction_matrix)


        if self.multithread > 1:
            with ThreadPoolExecutor(max_workers=self.multithread) as executor:
                query_list = [query_graph.program for query_graph, _ in itertools.chain(self.candidate_queries, self.answers)]
                for pred_per_query in executor.map(self.execute_over_all_inputs, query_list):
                    prediction_matrix.append(pred_per_query)
        else:
            for query_graph, _ in itertools.chain(self.candidate_queries, self.answers):
                query = query_graph.program
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
                # loss_t += (np.array((prediction_matrix_pair_level[t, :] != 1) * 1) / u_t * (n_pos + n_neg) / (2 * n_pos))
                # n_pos += 1
            else:
                loss_t += (np.array((prediction_matrix[i, :] != 0) * 1))

        eta = np.sqrt(np.log(self.k1)/(2*(len(self.labeled_index)+1)))

        posterior_t = np.exp(-eta * (loss_t-np.min(loss_t)))
        # Note that above equation is equivalent to np.exp(-eta * loss_t).
        # `-np.min(loss_t)` is applied only to avoid entries being near zero for large eta*loss_t values before the normalization
        posterior_t  /= np.sum(posterior_t)  # normalized weight

        entropy_list = np.zeros(n_instances)
        for i in range(n_instances):
            if i in self.labeled_index:
                entropy_list[i] = -1
            else:
                entropy_list[i] = self._compute_u_t(posterior_t, prediction_matrix[i, :], tuning_par=10)
        # find argmax of entropy (top k)
        n_to_label = 1
        video_segment_ids = np.argpartition(entropy_list, -n_to_label)[-n_to_label:]
        # max_entropy_index = np.argmax(entropy_list)
        return video_segment_ids.tolist()

    def _compute_u_t(self, posterior_t, predictions_c, tuning_par=1):

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
        u_t = tuning_par * np.max(u_t_list)

        return u_t