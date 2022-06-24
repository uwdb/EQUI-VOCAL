import copy
import dsl


# class QueryNode(object):

#     def __init__(self, program, score, parent, depth, cost, order):
#         self.score = score
#         self.program = program
#         self.children = []
#         self.parent = parent
#         self.depth = depth
#         self.cost = cost
#         self.order = order


class QueryGraph(object):

    def __init__(self, max_num_atomic_predicates, max_depth):
        self.max_num_atomic_predicates = max_num_atomic_predicates
        self.max_depth = max_depth
        self.max_num_atomic_predicates = 0

        start = dsl.StartOperator()
        # self.root_node = QueryNode(start, 0, None, 0, 0, 0)
        # self.score = score
        self.program = start
        # self.children = []
        # self.parent = parent
        self.depth = 0
        # self.cost = cost
        # self.order = order

    def get_parameter_holes_and_value_space(self):
        parameter_holes = []
        value_space = []
        steps = []
        queue = [self.program]
        while len(queue) != 0:
            current = queue.pop()
            # (key, value) pair
            for submod, functionclass in current.submodules.items():
                if isinstance(functionclass, dsl.PredicateHole):
                    continue
                if issubclass(type(functionclass), dsl.Predicate):
                    continue
                if isinstance(functionclass, dsl.ParameterHole):
                    parameter_holes.append(functionclass)
                    value_space.append(parameter_holes.get_value_range())
                    steps.append(parameter_holes.get_step())
                else:
                    #add submodules
                    queue.append(functionclass)
        return parameter_holes, value_space, steps

    @classmethod
    def is_sketch(cls, candidate_query):
        # print("candidate_query", candidate_query)
        # A partial query Q is a sketch if it has no predicate holes
        queue = [candidate_query]
        while len(queue) != 0:
            current_function = queue.pop()
            if isinstance(current_function, dsl.PredicateHole):
                return False
            if isinstance(current_function, dsl.ParameterHole):
                continue
            if issubclass(type(current_function), dsl.Predicate):
                continue
            else:
                for submodule_name in current_function.submodules:
                    queue.append(current_function.submodules[submodule_name])
        return True

    @classmethod
    def is_complete(cls, candidate_query):
        # A partial query Q is a sketch if it has no holes
        queue = [candidate_query]
        while len(queue) != 0:
            current_function = queue.pop()
            if issubclass(type(current_function), dsl.Hole):
                return False
            if issubclass(type(current_function), dsl.Predicate):
                continue
            else:
                for submodule_name in current_function.submodules:
                    queue.append(current_function.submodules[submodule_name])
        return True


    def get_all_children(self):
        # Fill the predicate hole that is found first (BFS).
        all_children = []
        # child_depth = self.depth + 1
        # child_num_units = self.num_units_at_depth(child_depth)
        queue = [self.program]
        while len(queue) != 0:
            current = queue.pop(0)
            for submod, functionclass in current.submodules.items():
                if isinstance(functionclass, dsl.ParameterHole):
                    continue
                if issubclass(type(functionclass), dsl.Predicate):
                    continue
                if isinstance(functionclass, dsl.PredicateHole):
                    replacement_candidates = self.construct_candidates()
                    orig_fclass = copy.deepcopy(current.submodules[submod])
                    for child_candidate, inc_depth in replacement_candidates:
                        # replace the neural function with a candidate
                        current.submodules[submod] = child_candidate
                        # create the correct child node
                        new_query_graph = copy.deepcopy(self)
                        new_query_graph.depth = self.depth + inc_depth
                        # check if child program can be completed within max_depth
                        if new_query_graph.depth > self.max_depth:
                            continue
                        # if yes, compute costs and add to list of children
                        all_children.append(new_query_graph)
                    # once we've copied it, set current back to the original current
                    current.submodules[submod] = orig_fclass
                    return all_children
                else:
                    #add submodules
                    queue.append(functionclass)
        return all_children

    def construct_candidates(self):
        candidates = []
        replacement_candidates = [dsl.ConjunctionOperator, dsl.SequencingOperator, dsl.KleeneOperator, dsl.Near, dsl.Far, dsl.MinLength, dsl.TrueStar]
        for functionclass in replacement_candidates:
            if issubclass(functionclass, dsl.Predicate):
                if functionclass.has_theta:
                    candidate = dsl.ParameterHole(functionclass())
                else:
                    candidate = functionclass()
                candidates.append([candidate, 0])
            else:
                candidate = functionclass()
                candidates.append([candidate, 1])
        return candidates