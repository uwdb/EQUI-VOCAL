# InLaneK, MinLength_theta, Dist_theta(A, B)
import math
import copy
import numpy as np
import utils_bu as utils
import functools

# (predicate), conjunction, sequencing, Kleene star, finite k iteration
# Omit nested Kleene star operators, as well as Kleene star around sequencing.
# Use Boolean semantics

class BaseOperator:

    def __init__(self, submodules, name=""):
        self.submodules = submodules
        self.name = name

    def get_submodules(self):
        return self.submodules

    def set_submodules(self, new_submodules):
        self.submodules = new_submodules

    def get_typesignature(self):
        return self.input_type, self.output_type

    def execute(self):
        # Boolean semantics: Z (trajectories) --> B (0 or 1)
        raise NotImplementedError

class StartOperator(BaseOperator):
    def __init__(self, function1=None):
        if function1 is None:
            self.program = PredicateHole()
        else:
            self.program = function1
        submodules = { 'program' : self.program }
        super().__init__(submodules, name="Start")

    def execute(self, input, label, memoize, cache=True):
        return self.submodules["program"].execute(input, label, memoize, cache)

class ConjunctionOperator(BaseOperator):
    def __init__(self, function1=None, function2=None):
        if function1 is None:
            function1 = PredicateHole()
        if function2 is None:
            function2 = PredicateHole()
        submodules = { "function1": function1, "function2": function2 }
        super().__init__(submodules, name="Conjunction")

    def execute(self, input, label, memoize, cache=True):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], memoize
        res1, memoize = self.submodules["function1"].execute(input, label, memoize)
        res2, memoize = self.submodules["function2"].execute(input, label, memoize)
        result = np.minimum(res1, res2)

        if cache:
            memoize[subquery_str] = result

        return result, memoize

class SequencingOperator(BaseOperator):
    def __init__(self, function1=None, function2=None):
        if function1 is None:
            function1 = PredicateHole()
        if function2 is None:
            function2 = PredicateHole()
        submodules = { "function1": function1, "function2": function2 }
        super().__init__(submodules, name="Sequencing")

    def execute(self, input, label, memoize, cache=True):
        # TODO: Make sure when len(input) > 2, all trajectories have the same length
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], memoize
        result = np.amax(np.minimum(self.submodules["function1"].execute(input, label, memoize)[0][..., np.newaxis], self.submodules["function2"].execute(input, label, memoize)[0][np.newaxis, ...]), axis=1)

        if cache:
            memoize[subquery_str] = result

        return result, memoize

class IterationOperator(BaseOperator):
    def __init__(self, k, function1=None):
        self.k = k
        if function1 is None:
            function1 = PredicateHole()
        submodules = { "iteration": function1}
        super().__init__(submodules, name="Iteration")

    def execute(self, input, label, memoize):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], memoize
        result = self.execute_helper(self.k, input, label, memoize)
        memoize[subquery_str] = result
        return result, memoize

    def execute_helper(self, k, input, label, memoize):
        # TODO: this is not optimized (refer to the paper)
        if k == 0:
            # A matrix with inf on the diagonal and -inf elsewhere
            return np.fill_diagonal(np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf), np.inf)

        pow = self.execute_helper(self, k // 2, input, label)

        if k % 2 == 0:
            # pow * pow
            return np.max(np.minimum(pow[..., np.newaxis], pow[np.newaxis, ...]), axis=1)
        else:
            # M * pow * pow
            return np.max(np.minimum(self.submodules["iteration"].execute(input, label, memoize)[0][..., np.newaxis], np.max(np.minimum(pow[..., np.newaxis], pow[np.newaxis, ...]), axis=1)), axis=1)

class KleeneOperator(BaseOperator):
    def __init__(self, function1=None):
        if function1 is None:
            function1 = PredicateHole()
        submodules = { "kleene": function1}
        super().__init__(submodules, name="Kleene")

    def execute(self, input, label, memoize, cache=True):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], memoize
        identity_mtx = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        np.fill_diagonal(identity_mtx, np.inf)

        base_arr = [identity_mtx]

        if len(input[0]) > 0:
            Q_mtx, memoize = self.submodules["kleene"].execute(input, label, memoize)
            base_arr.append(Q_mtx)

        for _ in range(2, len(input[0]) + 1):
            Q_pow_k = np.amax(np.minimum(base_arr[-1][..., np.newaxis], Q_mtx[np.newaxis, ...]), axis=1)
            base_arr.append(Q_pow_k)

        result = np.amax(np.stack(base_arr, axis=0), axis=0)

        if cache:
            memoize[subquery_str] = result

        return result, memoize


class DurationOperator(BaseOperator):
    def __init__(self, function1, theta):
        # theta >= 2 and is an integer
        self.theta = int(theta)
        submodules = { "duration": function1 }
        super().__init__(submodules, name="Duration")

    def execute(self, input, label, memoize, cache=True):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], memoize
        if len(input[0]) < self.theta:
            return np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf), memoize

        Q_mtx, memoize = self.submodules["duration"].execute(input, label, memoize)
        base_arr = [Q_mtx]

        for _ in range(2, len(input[0]) + 1):
            Q_pow_k = np.amax(np.minimum(base_arr[-1][..., np.newaxis], Q_mtx[np.newaxis, ...]), axis=1)
            base_arr.append(Q_pow_k)

        result = np.amax(np.stack(base_arr[(self.theta-1):], axis=0), axis=0)

        if cache:
            memoize[subquery_str] = result

        return result, memoize


############### Predicate ################
@functools.total_ordering
class Predicate:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return self.name == other.name

    def __lt__(self, other):
        return self.name < other.name

    def set_theta(self, theta):
        self.theta = theta

    def execute(self):
        raise NotImplementedError

class Near(Predicate):
    has_theta=True

    def __init__(self, theta=-1.05, step=0.2, with_hole=False):
        self.theta = theta
        self.step = step
        self.with_hole = with_hole
        super().__init__("Near")

    def init_value_range(self):
        return [-2, 0]

    def execute_with_hole(self, input, label, memoize):
        assert len(input) == 2
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        for i in range(len(input[0])):
            result[i, i + 1] = -1 * obj_distance(input[0][i], input[1][i])
        memoize[subquery_str] = result
        return result, memoize

    def execute(self, input, label, memoize, cache=True):
        if self.with_hole:
            return self.execute_with_hole(input, label, memoize)

        assert len(input) == 2

        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        for i in range(len(input[0])):
            if -1 * obj_distance(input[0][i], input[1][i]) >= self.theta:
                result[i, i + 1] = np.inf

        if cache:
            memoize[subquery_str] = result

        return result, memoize

class Far(Predicate):
    has_theta=True

    def __init__(self, theta=1.1, step=0.2, with_hole=False):
        self.theta = theta
        self.step = step
        self.with_hole = with_hole
        super().__init__("Far")

    def init_value_range(self):
        return [1, 3]

    def execute_with_hole(self, input, label, memoize):
        assert len(input) == 2
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        for i in range(len(input[0])):
            result[i, i + 1] = obj_distance(input[0][i], input[1][i])
        memoize[subquery_str] = result
        return result, memoize

    def execute(self, input, label, memoize, cache=True):
        if self.with_hole:
            return self.execute_with_hole(input, label, memoize)

        assert len(input) == 2
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], memoize
        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        for i in range(len(input[0])):
            if obj_distance(input[0][i], input[1][i]) >= self.theta:
                result[i, i + 1] = np.inf

        if cache:
            memoize[subquery_str] = result

        return result, memoize

class TrueStar(Predicate):
    has_theta=False

    def __init__(self):
        super().__init__("True*")

    def execute(self, input, label, memoize, cache=True):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], memoize

        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        for i in range(len(input[0]) + 1):
            for j in range(i, len(input[0]) + 1):
                result[i, j] = np.inf

        if cache:
            memoize[subquery_str] = result

        return result, memoize

class MinLength(Predicate):
    has_theta=True

    def __init__(self, theta=1, step=1, with_hole=False):
        self.theta = theta
        self.step = step
        self.with_hole = with_hole
        super().__init__("MinLength")

    def init_value_range(self):
        return [1, 10]

    def execute_with_hole(self, input, label, memoize):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], memoize

        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        for i in range(len(input[0])):
            for j in range(i, len(input[0]) + 1):
                result[i, j] = j - i
        memoize[subquery_str] = result
        return result, memoize

    def execute(self, input, label, memoize, cache=True):
        if self.with_hole:
            return self.execute_with_hole(input, label, memoize)

        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], memoize

        result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
        for i in range(len(input[0])):
            for j in range(i, len(input[0]) + 1):
                if j - i >= self.theta:
                    result[i, j] = np.inf

        if cache:
            memoize[subquery_str] = result

        return result, memoize

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

    def execute(self, input, label, memoize):
        subquery_str = utils.print_program(self)
        if subquery_str in memoize:
            return memoize[subquery_str], memoize
        if label == 1:
            # over-approximation
            result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
            for i in range(len(input[0]) + 1):
                for j in range(i, len(input[0]) + 1):
                    result[i, j] = np.inf
            memoize[subquery_str] = result
            return result, memoize
        else:
            # under-approximation
            result = np.full((len(input[0]) + 1, len(input[0]) + 1), -np.inf)
            memoize[subquery_str] = result
            return result, memoize

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
        # self.predicate.with_hole = False
        return copy.deepcopy(self.predicate)

    def execute(self, input, label, memoize):
        if label == 1:
            self.predicate.set_theta(-np.inf)
        else:
            self.predicate.set_theta(np.inf)
        return self.predicate.execute(input, label, memoize)