from logging import print_program
from query_graph import QueryGraph
import dsl
import copy
import numpy as np
import json
import random
import math

class QUIVR:

    def __init__(self, max_num_programs=100, max_num_atomic_predicates=5,
        max_depth=2):
        self.max_num_programs = max_num_programs
        self.max_num_atomic_predicates = max_num_atomic_predicates
        self.max_depth = max_depth

    def run(self, inputs, labels):
        # StartFunction
        # Initialize program graph
        query_graph = QueryGraph(self.max_num_atomic_predicates, self.max_depth)

        # root = query_graph.root_node

        queue = [query_graph]
        self.consistent_queries = []
        while len(queue) != 0:
            # Pop the last element
            current_query_graph = queue.pop(0)
            current_query = current_query_graph.program
            # print("current_query", print_program(current_query))
            # IF: overapproximation doesn't match, prune Q
            overapproximation_for_all_inputs = True
            for input, label in zip(inputs, labels):
                if not (current_query.execute(input, label)[0, len(input[0])] > 0) == label:
                    overapproximation_for_all_inputs = False
                    break
            if not overapproximation_for_all_inputs:
                continue
            # ELSE IF: Q is sketch, computes all parameters for Q that are consistent with inputs, and adds them to the final list.
            if current_query_graph.is_sketch(current_query):
                # print("is sketch")
                print("current_query", print_program(current_query))
                self.fill_all_parameter_holes(current_query)
                # self.prune_parameter_values(current_query, inputs, labels)
            # ELSE: adds all the children of Q to queue.
            else:
                all_children = current_query_graph.get_all_children()
                queue.extend(all_children)
            # If the number of atomic predicates >= max_num_atomic_predicates, or the dapth >= max_depth, stop.

        # RETURN: the list. Need to enumerate over all possible queries to omit ones that are inconsistenet with W.
        answer = []
        print("candidate queries size", len(self.consistent_queries))
        for candidate_query in self.consistent_queries:
            print("candidate query", print_program(candidate_query))
            matched = True
            for input, label in zip(inputs, labels):
                if not (candidate_query.execute(input, label)[0, len(input[0])] > 0) == label:
                    matched = False
                    break
            if matched:
                answer.append(candidate_query)
        return answer

    # def get_theta_lower_bound(self, input, current, parameter_hole, value_range, step):
    #     # value range [ , )
    #     if value_range[0] + step >= value_range[1]:
    #         return
    #     mid_val = value_range[0] + int((value_range[1] - value_range[0]) / 2 / step) * step
    #     predicate = parameter_hole.fill_hole(mid_val)
    #     current.submodules[submd] = predicate
    #     self.get_theta_lower_bound(input, current, parameter_hole, [value_range[0], mid_val], step)
    #     parameter_hole.fill_hole()

    def prune_parameter_values(self, current_query, inputs, labels):
        # current_query is sketch (containing only parameter holes)
        # 1. Find all parameter holes
        # 2. For each parameter hole, fix it and fill in other holes with -inf or inf.
        if QueryGraph.is_complete(current_query):
            pass
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
        for input, label in zip(inputs, labels):
            if label == 1:
                theta_ub = min(new_query.execute(input, label)[0, len(input[0])], theta_ub)
            if label == 0:
                theta_lb = max(new_query.execute(input, label)[0, len(input[0])], theta_lb)
        return [theta_lb, theta_ub]

    def fill_all_parameter_holes(self, current_query):
        # print("current_query2", print_program(current_query))
        if QueryGraph.is_complete(current_query):
            # print("here3")
            self.consistent_queries.append(current_query)
        queue = [current_query]
        while len(queue) != 0:
            current = queue.pop(0)
            for submod, functionclass in current.submodules.items():
                # if isinstance(functionclass, dsl.PredicateHole):
                #     continue
                if issubclass(type(functionclass), dsl.Predicate):
                    continue
                if isinstance(functionclass, dsl.ParameterHole):
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
                        if QueryGraph.is_sketch(new_query):
                            # print("here1")
                            self.fill_all_parameter_holes(new_query)
                        # elif QueryGraph.is_complete(new_query):
                        #     print("here2")
                        #     self.consistent_queries.append(new_query)
                    current.submodules[submod] = orig_fclass
                    return
                else:
                    #add submodules
                    queue.append(functionclass)


if __name__ == '__main__':

    # read from json file
    with open("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/sample_inputs.json", 'r') as f:
        inputs = json.load(f)
    with open("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/sample_labels.json", 'r') as f:
        labels = json.load(f)
    inputs = np.asarray(inputs, dtype=object)
    labels = np.asarray(labels, dtype=object)
    num_pos = 10
    num_neg = 10
    # sample_idx = random.sample(list(range(20)), num_pos) + random.sample(list(range(20, 40)), num_neg)
    sample_idx = list(range(20))[:num_pos] + list(range(20, 40))[:num_neg]
    print(sample_idx)
    inputs = inputs[sample_idx].tolist()
    labels = labels[sample_idx].tolist()

    # inputs = [
    #     [
    #         [[0, 0, 50, 50], [2, 2, 52, 52], [5, 5, 55, 55]],
    #         [[100, 100, 150, 150], [100, 100, 150, 150], [100, 100, 150, 150]]
    #     ],
    #     [
    #         [[0, 0, 50, 50], [2, 2, 52, 52], [5, 5, 55, 55]],
    #         [[0, 0, 50, 50], [2, 2, 52, 52], [5, 5, 55, 55]]
    #     ],
    #     [
    #         [[0, 0, 50, 50], [2, 2, 52, 52], [5, 5, 55, 55]],
    #         [[0.2, 0.2, 50.2, 50.2], [100, 100, 150, 150], [5.2, 5.2, 55.2, 55.2]]
    #     ]
    # ]
    # labels = [0, 1, 0]
    algorithm = QUIVR(max_depth=2)
    answer = algorithm.run(inputs, labels)
    print("answer", len(answer))
    for q in answer:
        print(print_program(q))
