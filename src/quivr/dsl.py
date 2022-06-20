# InLaneK, MinLength_theta, Dist_theta(A, B)
from dataclasses import dataclass, field
from email.mime import base
from typing import Dict, List
from pprint import pprint
import math
import copy
import numpy as np

# (predicate), conjunction, sequencing, Kleene star, finite k iteration
# Omit nested Kleene star operators, as well as Kleene star around sequencing.
# Use Boolean semantics

class BaseOperator:

    def __init__(self, submodules, name="", has_params=False):
        self.submodules = submodules
        self.name = name
        self.has_params = has_params

        # if self.has_params:
        #     assert "init_params" in dir(self)
        #     self.init_params()

    def get_submodules(self):
        return self.submodules

    def set_submodules(self, new_submodules):
        self.submodules = new_submodules

    def get_typesignature(self):
        return self.input_type, self.output_type

    def execute(self, input, label):
        # Boolean semantics: Z (trajectories) --> B (0 or 1)
        raise NotImplementedError

class StartOperator(BaseOperator):
    def __init__(self):
        self.program = PredicateHole()
        submodules = { 'program' : self.program }
        super().__init__(submodules, name="Start")

    def execute(self, input, label):
        return self.submodules["program"].execute(input, label)

class ConjunctionOperator(BaseOperator):
    def __init__(self, function1=None, function2=None):
        if function1 is None:
            function1 = PredicateHole()
        if function2 is None:
            function2 = PredicateHole()
        submodules = { "function1": function1, "function2": function2 }
        super().__init__(submodules, name="Conjunction")

    def execute(self, input, label):
        return np.minimum(self.submodules["function1"].execute(input, label), self.submodules["function2"].execute(input, label))
        # predicted_function1 = self.submodules["function1"].execute(input, label)
        # if not predicted_function1:
        #     return False
        # predicted_function2 = self.submodules["function2"].execute(input, label)
        # return predicted_function1 and predicted_function2

class SequencingOperator(BaseOperator):
    def __init__(self, function1=None, function2=None):
        if function1 is None:
            function1 = PredicateHole()
        if function2 is None:
            function2 = PredicateHole()
        submodules = { "function1": function1, "function2": function2 }
        super().__init__(submodules, name="Sequencing")

    def execute(self, input, label):
        # TODO: Make sure when len(input) > 2, all trajectories have the same length
        return np.amax(np.minimum(self.submodules["function1"].execute(input, label)[..., np.newaxis], self.submodules["function2"].execute(input, label)[np.newaxis, ...]), axis=1)
        # for k in range(len(input[0]) + 1):
        #     split1 = []
        #     split2 = []
        #     for trajectory in input:
        #         split1.append(trajectory[:k])
        #         split2.append(trajectory[k:])
        #     predicted_function1 = self.submodules["function1"].execute(split1, label)
        #     predicted_function2 = self.submodules["function2"].execute(split2, label)
        #     if predicted_function1 and predicted_function2:
        #         return True
        # return False
    # Matrix semantics
    # np.max(np.minimum(self.submodules["function1"].execute(split1, label)[..., np.newaxis], self.submodules["function2"].execute(split2, label)[np.newaxis, ...]), axis=1)

class IterationOperator(BaseOperator):
    def __init__(self, k, function1=None):
        self.k = k
        if function1 is None:
            function1 = PredicateHole()
        submodules = { "iteration": function1}
        super().__init__(submodules, name="Iteration")

    def execute(self, input, label):
        return self.execute_helper(self.k, input, label)

    def execute_helper(self, k, input, label):
        # TODO: this is not optimized (refer to the paper)
        if k == 0:
            # A matrix with inf on the diagonal and -inf elsewhere
            return np.fill_diagonal(np.full((len(input[0]) + 1, len(input[0]) + 1), -float("inf")), float("inf"))

        pow = self.execute_helper(self, k // 2, input, label)

        if k % 2 == 0:
            # pow * pow
            return np.max(np.minimum(pow[..., np.newaxis], pow[np.newaxis, ...]), axis=1)
        else:
            # M * pow * pow
            return np.max(np.minimum(self.submodules["iteration"].execute(input, label)[..., np.newaxis], np.max(np.minimum(pow[..., np.newaxis], pow[np.newaxis, ...]), axis=1)), axis=1)

    # def execute(self, input, label):
    #     return self.execute_helper(self.k, input, label)

    # def execute_helper(self, k, input, label):
    #     if k == 0:
    #         return len(input[0]) == 0
    #     for i in range(len(input[0]) + 1):
    #         split1 = []
    #         split2 = []
    #         for trajectory in input:
    #             split1.append(trajectory[:i])
    #             split2.append(trajectory[i:])
    #         predicted_function1 = self.execute_helper(self.k - 1, split1, label)
    #         predicted_function2 = self.submodules["iteration"].execute(split2, label)
    #         if predicted_function1 and predicted_function2:
    #             return True
    #     return False

class KleeneOperator(BaseOperator):
    def __init__(self, function1=None):
        if function1 is None:
            function1 = PredicateHole()
        submodules = { "kleene": function1}
        super().__init__(submodules, name="Kleene")

    def execute(self, input, label):
        identity_mtx = np.full((len(input[0]) + 1, len(input[0]) + 1), -float("inf"))
        np.fill_diagonal(identity_mtx, float("inf"))

        base_arr = [identity_mtx]

        if len(input[0]) > 0:
            Q_mtx = self.submodules["kleene"].execute(input, label)
            base_arr.append(Q_mtx)

        for _ in range(2, len(input[0]) + 1):
            Q_pow_k = np.amax(np.minimum(base_arr[-1][..., np.newaxis], Q_mtx[np.newaxis, ...]), axis=1)
            base_arr.append(Q_pow_k)

        return np.amax(np.stack(base_arr, axis=0), axis=0)

    # def execute_iteration(self, k, input, label):
    #     # if isinstance(self.submodules["kleene"], PredicateHole):
    #     #     if label == 0:
    #     #         return False
    #     #     else:
    #     #         return True
    #     if k == 0:
    #         return len(input[0]) == 0
    #     for i in range(len(input[0]) + 1):
    #         # print("k", k, "i", i)
    #         split1 = []
    #         split2 = []
    #         for trajectory in input:
    #             split1.append(trajectory[:i])
    #             split2.append(trajectory[i:])
    #         predicted_function1 = self.execute_iteration(k - 1, split1, label)
    #         predicted_function2 = self.submodules["kleene"].execute(split2, label)
    #         if predicted_function1 and predicted_function2:
    #             return True
    #     return False

############### Predicate ################
class Predicate:
    def __init__(self, name):
        self.name = name

    def set_theta(self, theta):
        self.theta = theta

    def execute(self, input, label):
        raise NotImplementedError

class Near(Predicate):
    has_theta=True

    def __init__(self, theta=-1.05, step=1, with_hole=False):
        self.theta = theta
        self.step = step
        self.with_hole = with_hole
        super().__init__("Near")

    def init_value_range(self):
        return [-2, 0]

    def execute_with_hole(self, input, label):
        assert len(input) == 2
        memo = np.full((len(input[0]) + 1, len(input[0]) + 1), -float("inf"))
        for i in range(len(input[0])):
            memo[i, i + 1] = -1 * obj_distance(input[0][i], input[1][i])
        return memo

    def execute(self, input, label):
        if self.with_hole:
            return self.execute_with_hole(input, label)

        assert len(input) == 2
        memo = np.full((len(input[0]) + 1, len(input[0]) + 1), -float("inf"))
        for i in range(len(input[0])):
            if -1 * obj_distance(input[0][i], input[1][i]) >= self.theta:
                memo[i, i + 1] = float("inf")
        return memo

class Far(Predicate):
    has_theta=True

    def __init__(self, theta=1.1, step=1, with_hole=False):
        self.theta = theta
        self.step = step
        self.with_hole = with_hole
        super().__init__("Far")

    def init_value_range(self):
        return [0, 3]

    def execute_with_hole(self, input, label):
        assert len(input) == 2
        memo = np.full((len(input[0]) + 1, len(input[0]) + 1), -float("inf"))
        for i in range(len(input[0])):
            memo[i, i + 1] = obj_distance(input[0][i], input[1][i])
        return memo

    def execute(self, input, label):
        if self.with_hole:
            return self.execute_with_hole(input, label)

        assert len(input) == 2
        memo = np.full((len(input[0]) + 1, len(input[0]) + 1), -float("inf"))
        for i in range(len(input[0])):
            if obj_distance(input[0][i], input[1][i]) >= self.theta:
                memo[i, i + 1] = float("inf")
        return memo

class TrueStar(Predicate):
    has_theta=False

    def __init__(self):
        super().__init__("True*")

    def execute(self, input, label):
        memo = np.full((len(input[0]) + 1, len(input[0]) + 1), -float("inf"))
        for i in range(len(input[0]) + 1):
            for j in range(i, len(input[0]) + 1):
                memo[i, j] = float("inf")
        return memo

class MinLength(Predicate):
    has_theta=True

    def __init__(self, theta=1, step=1, with_hole=False):
        self.theta = theta
        self.step = step
        self.with_hole = with_hole
        super().__init__("MinLength")

    def init_value_range(self):
        return [1, 10]

    def execute_with_hole(self, input, label):
        memo = np.full((len(input[0]) + 1, len(input[0]) + 1), -float("inf"))
        for i in range(len(input[0])):
            for j in range(i, len(input[0]) + 1):
                memo[i, j] = j - i
        return memo

    def execute(self, input, label):
        if self.with_hole:
            return self.execute_with_hole(input, label)

        memo = np.full((len(input[0]) + 1, len(input[0]) + 1), -float("inf"))
        for i in range(len(input[0])):
            for j in range(i, len(input[0]) + 1):
                if j - i >= self.theta:
                    memo[i, j] = float("inf")
        return memo

def obj_distance(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    cx1 = (x1 + x2) / 2
    cy1 = (y1 + y2) / 2
    cx2 = (x3 + x4) / 2
    cy2 = (y3 + y4) / 2
    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) / ((x2 - x1 + x4 - x3) / 2)


######## Hole ###########
class Hole:
    def __init__(self, name):
        self.name = name

class PredicateHole(Hole):
    def __init__(self):
        super().__init__("PredicateHole")

    def execute(self, input, label):
        if label == 1:
            # over-approximation
            memo = np.full((len(input[0]) + 1, len(input[0]) + 1), -float("inf"))
            for i in range(len(input[0]) + 1):
                for j in range(i, len(input[0]) + 1):
                    memo[i, j] = float("inf")
            return memo
        else:
            # under-approximation
            return np.full((len(input[0]) + 1, len(input[0]) + 1), -float("inf"))

class ParameterHole(Hole):
    def __init__(self, predicate):
        super().__init__("ParameterHole*" + predicate.name)
        self.predicate = predicate
        self.value_range = predicate.init_value_range()
        self.step = predicate.step

    def get_value_range(self):
        return self.value_range

    def get_step(self):
        return self.step

    def get_predicate(self):
        return self.predicate

    def fill_hole(self, theta):
        self.predicate.set_theta(theta)
        return copy.deepcopy(self.predicate)

    def execute(self, input, label):
        # TODO: fix it for matrix semantics
        if label == 1:
            self.predicate.set_theta(-float("inf"))
        else:
            self.predicate.set_theta(float("inf"))
        return self.predicate.execute(input, label)
