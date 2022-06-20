# # InLaneK, MinLength_theta, Dist_theta(A, B)
# from dataclasses import dataclass, field
# from typing import Dict, List
# from pprint import pprint
# import math

# class Predicate:
#     def __init__(self, name):
#         self.name = name

#     def execute(self, input):
#         raise NotImplementedError

# class PredicateHole:
#     def __init__(self):
#         self.name = "PredicateHole"

#     def execute(self, input):
#         pass

# class ParameterHole:
#     def __init__(self):
#         self.name = "ParameterHole"

#     def execute(self, input):
#         pass

# class Near(Predicate):
#     def __init__(self, theta=-1.05):
#         self.theta = theta
#         super().__init__("Near")

#     def execute(self, input):
#         assert len(input) == 2 and len(input[0]) == 1 and len(input[1]) == 1
#         return -1 * obj_distance(input[0], input[1]) >= self.theta

# class Far(Predicate):
#     def __init__(self, theta=1.1):
#         self.theta = theta
#         super().__init__("Far")

#     def execute(self, input):
#         assert len(input) == 2 and len(input[0]) == 1 and len(input[1]) == 1
#         return obj_distance(input[0], input[1]) >= self.theta

# class MinLength(Predicate):
#     def __init__(self, theta=1):
#         self.theta = theta
#         super().__init__("MinLength")

#     def execute(self, input):
#         return len(input[0]) >= self.theta

# def obj_distance(bbox1, bbox2):
#     x1, y1, x2, y2 = bbox1
#     x3, y3, x4, y4 = bbox2
#     cx1 = (x1 + x2) / 2
#     cy1 = (y1 + y2) / 2
#     cx2 = (x3 + x4) / 2
#     cy2 = (y3 + y4) / 2
#     return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) / ((x2 - x1 + x4 - x3) / 2)