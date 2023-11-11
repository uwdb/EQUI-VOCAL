import json
from src.synthesize import test_algorithm_demo_precompute
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

if __name__=="__main__":
    dataset_name = "demo_queries_scene_graph"

    query_str = "(Color_purple(o0), Color_cyan(o1), Far_3.0(o0, o1)); Near_1.0(o0, o1)"
    n_init_pos = n_init_neg = 5

    # query_str = "(Color_red(o0), LeftOf(o0, o1), Shape_cube(o1), TopQuadrant(o2)); BottomQuadrant(o2)"
    # n_init_pos = n_init_neg = 5

    # query_str = "Duration((Color_yellow(o0), LeftOf(o0, o1), Shape_cylinder(o1)), 10); Duration(RightOf(o0, o1), 10)"
    # n_init_pos = n_init_neg = 10

    predicate_dict = [{"name": "Near", "parameters": [1], "nargs": 2}, {"name": "Far", "parameters": [3], "nargs": 2}, {"name": "LeftOf", "parameters": None, "nargs": 2}, {"name": "Behind", "parameters": None, "nargs": 2}, {"name": "RightOf", "parameters": None, "nargs": 2}, {"name": "FrontOf", "parameters": None, "nargs": 2}, {"name": "RightQuadrant", "parameters": None, "nargs": 1}, {"name": "LeftQuadrant", "parameters": None, "nargs": 1}, {"name": "TopQuadrant", "parameters": None, "nargs": 1}, {"name": "BottomQuadrant", "parameters": None, "nargs": 1}, {"name": "Color", "parameters": ["gray", "red", "blue", "green", "brown", "cyan", "purple", "yellow"], "nargs": 1}, {"name": "Shape", "parameters": ["cube", "sphere", "cylinder"], "nargs": 1}, {"name": "Material", "parameters": ["metal", "rubber"], "nargs": 1}]

    input_dir = "/Users/zhangenhao/Desktop/UW/Research/equi-vocal-demo/EQUI-VOCAL/inputs"

    init_vids, log = test_algorithm_demo_precompute(method="vocal_postgres", dataset_name=dataset_name, n_init_pos=n_init_pos, n_init_neg=n_init_neg, npred=7, depth=3, max_duration=15, beam_width=10, pool_size=100, n_sampled_videos=100, k=100, budget=50, multithread=30, query_str=query_str, predicate_dict=predicate_dict, lru_capacity=None, reg_lambda=0.001, strategy='topk', max_vars=3, port=5432, input_dir=input_dir)

    result = {"vids": init_vids, "query_str": query_str, "log": log}
    with open("{}.json".format(query_str), "w") as f:
        json.dump(result, f, cls=NpEncoder)
