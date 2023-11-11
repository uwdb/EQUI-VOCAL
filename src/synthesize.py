import os
import json
import random
import numpy as np
import argparse
import sys
from src.methods.exhaustive_search import ExhaustiveSearch
from src.methods.quivr_original import QUIVROriginal
from src.methods.quivr_original_no_kleene import QUIVROriginalNoKleene
from src.methods.vocal_postgres import VOCALPostgres
from src.methods.vocal_postgres_no_active_learning import VOCALPostgresNoActiveLearning
import src.dsl as dsl
import time

def test_quivr_original(dataset_name, n_init_pos, n_init_neg, npred, n_nontrivial, n_trivial, depth, max_duration, budget, multithread, query_str, predicate_dict, lru_capacity, input_dir, with_kleene):
    if dataset_name.startswith("collision"):
        with open(os.path.join(input_dir, "collision.json"), 'r') as f:
            trajectories = json.load(f)
    elif dataset_name == "warsaw":
        with open(os.path.join(input_dir, "warsaw_trajectory_pairs.json"), 'r') as f:
            trajectories = json.load(f)
    else:
        with open(os.path.join(input_dir, "trajectory_pairs.json"), 'r') as f:
            trajectories = json.load(f)
    with open(os.path.join(input_dir, "{}/train/{}_inputs.json".format(dataset_name, query_str)), 'r') as f:
        inputs = json.load(f)
    with open(os.path.join(input_dir, "{}/train/{}_labels.json".format(dataset_name, query_str)), 'r') as f:
        labels = json.load(f)
    trajectories = np.asarray(trajectories, dtype=object)
    labels = np.asarray(labels, dtype=object)
    inputs = trajectories[inputs]
    if "sampling_rate" in dataset_name:
        splits = dataset_name.split("-")
        for split in splits:
            if split.startswith("sampling_rate_"):
                sampling_rate = int(split.replace("sampling_rate_", ""))
                break
        print("sampling_rate", sampling_rate)
        # Down-sample the trajectory once every sampling_rate frames
        inputs_downsampled = []
        for input in inputs:
            inputs_downsampled.append([input[0][::sampling_rate], input[1][::sampling_rate]])
        inputs = inputs_downsampled
        inputs = np.asarray(inputs, dtype=object)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    print("pos_idx", len(pos_idx), pos_idx, "neg_idx", len(neg_idx), neg_idx)

    init_labeled_index = random.sample(pos_idx.tolist(), n_init_pos) + random.sample(neg_idx.tolist(), n_init_neg)
    print(init_labeled_index)

    if with_kleene:
        method = QUIVROriginal
    else:
        method = QUIVROriginalNoKleene
    algorithm = method(dataset_name, inputs, labels, predicate_dict, npred, n_nontrivial, n_trivial, depth, max_duration, budget, multithread, lru_capacity)
    output_log = algorithm.run(init_labeled_index)

    return output_log

def test_quivr_original_active_learning(dataset_name, n_init_pos, n_init_neg, npred, n_nontrivial, n_trivial, depth, max_duration, budget, multithread, query_str, predicate_dict, lru_capacity, input_dir, log_dirname, log_filename):
    if dataset_name.startswith("collision"):
        with open(os.path.join(input_dir, "collision.json"), 'r') as f:
            trajectories = json.load(f)
    elif dataset_name == "warsaw":
        with open(os.path.join(input_dir, "warsaw_trajectory_pairs.json"), 'r') as f:
            trajectories = json.load(f)
    else:
        with open(os.path.join(input_dir, "trajectory_pairs.json"), 'r') as f:
            trajectories = json.load(f)
    with open(os.path.join(input_dir, "{}/train/{}_inputs.json".format(dataset_name, query_str)), 'r') as f:
        inputs = json.load(f)
    with open(os.path.join(input_dir, "{}/train/{}_labels.json".format(dataset_name, query_str)), 'r') as f:
        labels = json.load(f)
    trajectories = np.asarray(trajectories, dtype=object)
    labels = np.asarray(labels, dtype=object)
    inputs = trajectories[inputs]
    if "sampling_rate" in dataset_name:
        splits = dataset_name.split("-")
        for split in splits:
            if split.startswith("sampling_rate_"):
                sampling_rate = int(split.replace("sampling_rate_", ""))
                break
        print("sampling_rate", sampling_rate)
        # Down-sample the trajectory once every sampling_rate frames
        inputs_downsampled = []
        for input in inputs:
            inputs_downsampled.append([input[0][::sampling_rate], input[1][::sampling_rate]])
        inputs = inputs_downsampled
        inputs = np.asarray(inputs, dtype=object)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    print("pos_idx", len(pos_idx), pos_idx, "neg_idx", len(neg_idx), neg_idx)

    init_labeled_index = random.sample(pos_idx.tolist(), n_init_pos) + random.sample(neg_idx.tolist(), n_init_neg)
    print(init_labeled_index)

    method = QUIVROriginal
    algorithm = method(dataset_name, inputs, labels, predicate_dict, npred, n_nontrivial, n_trivial, depth, max_duration, budget, multithread, lru_capacity)
    output_log = algorithm.active_learning_stage(init_labeled_index, log_dirname, log_filename)

    return output_log

def test_algorithm(method, dataset_name, n_init_pos, n_init_neg, npred, depth, max_duration, beam_width, pool_size, n_sampled_videos, k, budget, multithread, query_str, predicate_dict, lru_capacity, reg_lambda, strategy, max_vars, port, input_dir):

    if dataset_name.startswith("user_study_queries_scene_graph"):
        if dataset_name == "user_study_queries_scene_graph-simulated_user_error":
            fn_error_rates = [0.061, 0.091, 0.286, 0.123, 0.133, 0.714]
            fp_error_rates = [0.25, 0.169, 0.308, 0.065, 0.028, 0.029]
            query_strs = [
                "Conjunction(Conjunction(Color_red(o0), Far_3(o0, o1)), Shape_cylinder(o1)); Near_1(o0, o1)",
                "Conjunction(Conjunction(Color_red(o0), Far_3(o0, o1)), Shape_cylinder(o1)); Conjunction(Conjunction(Near_1(o0, o1), RightQuadrant(o2)), TopQuadrant(o2))",
                "Duration(Conjunction(Conjunction(Color_red(o0), Far_3(o0, o1)), Shape_cylinder(o1)), 25); Conjunction(Conjunction(Near_1(o0, o1), RightQuadrant(o2)), TopQuadrant(o2))",
                "Conjunction(Conjunction(Conjunction(Behind(o0, o1), BottomQuadrant(o1)), Color_purple(o0)), material_metal(o0))",
                "Conjunction(Conjunction(Conjunction(Behind(o0, o1), BottomQuadrant(o1)), Color_purple(o0)), material_metal(o0)); TopQuadrant(o1)",
                "Conjunction(Conjunction(Conjunction(Behind(o0, o1), BottomQuadrant(o1)), Color_purple(o0)), material_metal(o0)); TopQuadrant(o1); Duration(Conjunction(BottomQuadrant(o2), RightQuadrant(o2)), 25)"
            ]
            query_idx = query_strs.index(query_str)
            fn_error_rate = fn_error_rates[query_idx]
            fp_error_rate = fp_error_rates[query_idx]
            print("fn_error_rate", fn_error_rate, "fp_error_rate", fp_error_rate)
        else: # 'user_study_queries_scene_graph-fn_error_rate_0.3-fp_error_rate_0.03',
            fn_error_rate = None
            fp_error_rate = None
            splits = dataset_name.split("-")
            for split in splits:
                if split.startswith("fn_error_rate"):
                    fn_error_rate = float(split.replace("fn_error_rate_", ""))
                elif split.startswith("fp_error_rate"):
                    fp_error_rate = float(split.replace("fp_error_rate_", ""))
        dataset_name = "user_study_queries_scene_graph"

    with open(os.path.join(input_dir, "{}/train/{}_inputs.json".format(dataset_name, query_str)), 'r') as f:
        inputs = json.load(f)
    with open(os.path.join(input_dir, "{}/train/{}_labels.json".format(dataset_name, query_str)), 'r') as f:
        labels = json.load(f)
    inputs = np.asarray(inputs) # input video ids
    labels = np.asarray(labels)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    print("pos_idx", len(pos_idx), pos_idx, "neg_idx", len(neg_idx), neg_idx)

    if "sampling_rate" in dataset_name:
        splits = dataset_name.split("-")
        for split in splits:
            if split.startswith("sampling_rate_"):
                sampling_rate = int(split.replace("sampling_rate_", ""))
                break
    else:
        sampling_rate = None
    print("sampling_rate", sampling_rate)
    init_labeled_index = random.sample(pos_idx.tolist(), n_init_pos) + random.sample(neg_idx.tolist(), n_init_neg)
    print(init_labeled_index)

    if dataset_name == "user_study_queries_scene_graph" and fn_error_rate:
        fn_count = 0
        fp_count = 0
        # flip the label with probability error_rate
        for i in range(len(labels)):
            if i not in init_labeled_index and labels[i] and random.random() < fn_error_rate:
                labels[i] = 0
                fn_count += 1
            elif i not in init_labeled_index and not labels[i] and random.random() < fp_error_rate:
                labels[i] = 1
                fp_count += 1
        print("fn_count", fn_count, "fp_count", fp_count)

    if method == "vocal_postgres":
        algorithm = VOCALPostgres(dataset_name, inputs, labels, predicate_dict, max_npred=npred, max_depth=depth, max_duration=max_duration, beam_width=beam_width, pool_size=pool_size, n_sampled_videos=n_sampled_videos, k=k, budget=budget, multithread=multithread, strategy=strategy, max_vars=max_vars, port=port, sampling_rate=sampling_rate, lru_capacity=lru_capacity, reg_lambda=reg_lambda, n_init_pos=n_init_pos, n_init_neg=n_init_neg)
    elif method == "vocal_postgres_no_active_learning":
        algorithm = VOCALPostgresNoActiveLearning(dataset_name, inputs, labels, predicate_dict, max_npred=npred, max_depth=depth, max_duration=max_duration, beam_width=beam_width, pool_size=pool_size, n_sampled_videos=n_sampled_videos, k=k, budget=budget, multithread=multithread, strategy=strategy, max_vars=max_vars, port=port, sampling_rate=sampling_rate, lru_capacity=lru_capacity, reg_lambda=reg_lambda, n_init_pos=n_init_pos, n_init_neg=n_init_neg)
    output_log = algorithm.run(init_labeled_index)
    return output_log

def test_algorithm_demo_precompute(method, dataset_name, n_init_pos, n_init_neg, npred, depth, max_duration, beam_width, pool_size, n_sampled_videos, k, budget, multithread, query_str, predicate_dict, lru_capacity, reg_lambda, strategy, max_vars, port, input_dir):
    with open(os.path.join(input_dir, "{}/train/{}_inputs.json".format(dataset_name, query_str)), 'r') as f:
        inputs = json.load(f)
    with open(os.path.join(input_dir, "{}/train/{}_labels.json".format(dataset_name, query_str)), 'r') as f:
        labels = json.load(f)
    inputs = np.asarray(inputs) # input video ids
    labels = np.asarray(labels)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    print("pos_idx", len(pos_idx), pos_idx, "neg_idx", len(neg_idx), neg_idx)

    # Test dataset for interactive demo
    with open(os.path.join(input_dir, "{}/test/{}_inputs.json".format(dataset_name, query_str)), 'r') as f:
        test_inputs = json.load(f)
    with open(os.path.join(input_dir, "{}/test/{}_labels.json".format(dataset_name, query_str)), 'r') as f:
        test_labels = json.load(f)
    # Sample 100 test videos for interactive demo
    test_inputs = np.asarray(test_inputs)[:100]
    test_labels = np.asarray(test_labels)[:100]

    if "sampling_rate" in dataset_name:
        splits = dataset_name.split("-")
        for split in splits:
            if split.startswith("sampling_rate_"):
                sampling_rate = int(split.replace("sampling_rate_", ""))
                break
    else:
        sampling_rate = None
    print("sampling_rate", sampling_rate)

    # use a random seed
    random.seed(time.time())
    init_labeled_index = random.sample(pos_idx.tolist(), n_init_pos) + random.sample(neg_idx.tolist(), n_init_neg)

    init_vids = inputs[init_labeled_index]
    print("init_vids", init_vids)
    if method == "vocal_postgres":
        algorithm = VOCALPostgres(dataset_name, inputs, labels, predicate_dict, max_npred=npred, max_depth=depth, max_duration=max_duration, beam_width=beam_width, pool_size=pool_size, n_sampled_videos=n_sampled_videos, k=k, budget=budget, multithread=multithread, strategy=strategy, max_vars=max_vars, port=port, sampling_rate=sampling_rate, lru_capacity=lru_capacity, reg_lambda=reg_lambda, n_init_pos=n_init_pos, n_init_neg=n_init_neg, test_inputs=test_inputs, test_labels=test_labels)
    elif method == "vocal_postgres_no_active_learning":
        algorithm = VOCALPostgresNoActiveLearning(dataset_name, inputs, labels, predicate_dict, max_npred=npred, max_depth=depth, max_duration=max_duration, beam_width=beam_width, pool_size=pool_size, n_sampled_videos=n_sampled_videos, k=k, budget=budget, multithread=multithread, strategy=strategy, max_vars=max_vars, port=port, sampling_rate=sampling_rate, lru_capacity=lru_capacity, reg_lambda=reg_lambda, n_init_pos=n_init_pos, n_init_neg=n_init_neg)

    algorithm.run_init(init_labeled_index)

    log = algorithm.demo_main()
    return init_vids, log

def test_algorithm_interactive(init_labeled_index, user_labels, method, dataset_name, n_init_pos, n_init_neg, npred, depth, max_duration, beam_width, pool_size, n_sampled_videos, k, budget, multithread, query_str, predicate_dict, lru_capacity, reg_lambda, strategy, max_vars, port, input_dir):

    with open(os.path.join(input_dir, "{}/train/{}_inputs.json".format(dataset_name, query_str)), 'r') as f:
        inputs = json.load(f)
    with open(os.path.join(input_dir, "{}/train/{}_labels.json".format(dataset_name, query_str)), 'r') as f:
        labels = json.load(f)
    inputs = np.asarray(inputs) # input video ids
    labels = np.asarray(labels)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    print("pos_idx", len(pos_idx), pos_idx, "neg_idx", len(neg_idx), neg_idx)

    # Test dataset for interactive demo
    with open(os.path.join(input_dir, "{}/test/{}_inputs.json".format(dataset_name, query_str)), 'r') as f:
        test_inputs = json.load(f)
    with open(os.path.join(input_dir, "{}/test/{}_labels.json".format(dataset_name, query_str)), 'r') as f:
        test_labels = json.load(f)
    # Sample 100 test videos for interactive demo
    test_inputs = np.asarray(test_inputs)[:100]
    test_labels = np.asarray(test_labels)[:100]

    if "sampling_rate" in dataset_name:
        splits = dataset_name.split("-")
        for split in splits:
            if split.startswith("sampling_rate_"):
                sampling_rate = int(split.replace("sampling_rate_", ""))
                break
    else:
        sampling_rate = None
    print("sampling_rate", sampling_rate)
    # init_labeled_index = random.sample(pos_idx.tolist(), n_init_pos) + random.sample(neg_idx.tolist(), n_init_neg)
    print(init_labeled_index)
    if method == "vocal_postgres":
        algorithm = VOCALPostgres(dataset_name, inputs, labels, predicate_dict, max_npred=npred, max_depth=depth, max_duration=max_duration, beam_width=beam_width, pool_size=pool_size, n_sampled_videos=n_sampled_videos, k=k, budget=budget, multithread=multithread, strategy=strategy, max_vars=max_vars, port=port, sampling_rate=sampling_rate, lru_capacity=lru_capacity, reg_lambda=reg_lambda, n_init_pos=n_init_pos, n_init_neg=n_init_neg, test_inputs=test_inputs, test_labels=test_labels)
    elif method == "vocal_postgres_no_active_learning":
        algorithm = VOCALPostgresNoActiveLearning(dataset_name, inputs, labels, predicate_dict, max_npred=npred, max_depth=depth, max_duration=max_duration, beam_width=beam_width, pool_size=pool_size, n_sampled_videos=n_sampled_videos, k=k, budget=budget, multithread=multithread, strategy=strategy, max_vars=max_vars, port=port, sampling_rate=sampling_rate, lru_capacity=lru_capacity, reg_lambda=reg_lambda, n_init_pos=n_init_pos, n_init_neg=n_init_neg)
    algorithm.run_init(init_labeled_index, user_labels)
    return algorithm

def test_exhaustive(n_labeled_pos, n_labeled_neg, npred, depth, max_duration, multithread, predicate_dict, input_dir):
    # read from json file
    with open("/gscratch/balazinska/enhaoz/complex_event_video/src/inputs/collision_inputs_test.json", 'r') as f:
        inputs = json.load(f)
    with open("/gscratch/balazinska/enhaoz/complex_event_video/src/inputs/collision_labels_test.json", 'r') as f:
        labels = json.load(f)

    inputs = np.asarray(inputs, dtype=object)
    labels = np.asarray(labels, dtype=object)

    n_pos = sum(labels)
    sampled_labeled_index = random.sample(list(range(n_pos)), n_labeled_pos) + random.sample(list(range(n_pos, len(labels))), n_labeled_neg)
    inputs = inputs[sampled_labeled_index]
    labels = labels[sampled_labeled_index]

    algorithm = ExhaustiveSearch(inputs, labels, predicate_dict, max_npred=npred, max_depth=depth, k=1000, max_duration=max_duration, multithread=multithread)
    algorithm.run()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', type=str, help='Query synthesis method.', choices=['vocal_postgres', 'vocal_postgres_no_active_learning', 'quivr_original', 'quivr_original_no_kleene', "quivr_original_active_learning", "quivr_original_no_kleene_active_learning"])
    ap.add_argument('--n_init_pos', type=int, default=2, help='Number of initial positive examples provided by the user.')
    ap.add_argument('--n_init_neg', type=int, default=10, help='Number of initial negative examples provided by the user.')
    ap.add_argument('--dataset_name', type=str, help='Name of the dataset.', choices=['synthetic_scene_graph_easy', 'synthetic_scene_graph_medium', 'synthetic_scene_graph_hard', 'without_duration-sampling_rate_4', 'trajectories_duration', 'trajectories_handwritten', 'without_duration-sampling_rate_4-fn_error_rate_0.1-fp_error_rate_0.01', 'without_duration-sampling_rate_4-fn_error_rate_0.3-fp_error_rate_0.03', 'demo_queries_scene_graph', 'shibuya', 'warsaw', 'user_study_queries_scene_graph-fn_error_rate_0.3-fp_error_rate_0.03', 'user_study_queries_scene_graph-fn_error_rate_0.5-fp_error_rate_0.05',
    'user_study_queries_scene_graph-simulated_user_error',
    'user_study_queries_scene_graph', 'synthetic_scene_graph_hard_v2'])
    ap.add_argument('--npred', type=int, default=5, help='Maximum number of predicates that the synthesized queries can have.')
    ap.add_argument('--n_nontrivial', type=int, help='Maximum number of non-trivial predicates that the synthesized queries can have. Used by Quivr.')
    ap.add_argument('--n_trivial', type=int, help='Maximum number of trivial predicates (i.e., <True>* predicate) that the synthesized queries can have. Used by Quivr.')
    ap.add_argument('--depth', type=int, default=3, help='For EQUI-VOCAL: Maximum number of region graphs that the synthesized queries can have. For Quivr: Maximum depth of the nested constructs that the synthesized queries can have.')
    ap.add_argument('--max_duration', type=int, default=1, help='Maximum number of the duration constraint.')
    ap.add_argument('--beam_width', type=int, default=32, help='Beam width.')
    ap.add_argument('--pool_size', type=int, default=100, help='Number of queries sampled during example selection.')
    ap.add_argument('--n_sampled_videos', type=int, default=100, help='Number of videos sampled during example selection.')
    ap.add_argument('--k', type=int, default=100, help='Number of queries in the final answer.')
    ap.add_argument('--budget', type=int, default=100, help='Labeling budget.')
    ap.add_argument('--multithread', type=int, default=1, help='Number of CPUs to use.')
    ap.add_argument('--strategy', type=str, default="topk", help='Strategy for query sampling.')
    ap.add_argument('--max_vars', type=int, default=2, help='Maximum number of variables that the synthesized queries can have.')
    ap.add_argument('--query_str', type=str, help='Target query written in the compact notation.')
    ap.add_argument('--run_id', type=int, help='Run ID. This sets the random seed.')
    ap.add_argument('--output_to_file', action="store_true", help='Whether write the output to file or print the output on the terminal console.')
    ap.add_argument('--port', type=int, default=5432, help='Port on which Postgres is to listen.')
    ap.add_argument('--lru_capacity', type=int, help='LRU cache capacity. Only used for Quivr due to its large memory footprint.')
    ap.add_argument('--reg_lambda', type=float, default=0.01, help='Regularization parameter.')
    ap.add_argument('--input_dir', type=str, default="../inputs", help='Input directory.')
    ap.add_argument('--output_dir', type=str, default="../outputs", help='Output directory.')

    args = ap.parse_args()
    method_str = args.method
    n_init_pos = args.n_init_pos
    n_init_neg = args.n_init_neg
    dataset_name = args.dataset_name
    npred = args.npred
    n_nontrivial = args.n_nontrivial
    n_trivial = args.n_trivial
    depth = args.depth
    max_duration = args.max_duration
    beam_width = args.beam_width
    pool_size = args.pool_size
    n_sampled_videos = args.n_sampled_videos
    k = args.k
    # samples_per_iter = args.samples_per_iter
    budget = args.budget
    multithread = args.multithread
    strategy = args.strategy
    max_vars = args.max_vars
    query_str = args.query_str
    run_id = args.run_id
    output_to_file = args.output_to_file
    port = args.port
    lru_capacity = args.lru_capacity
    reg_lambda = args.reg_lambda
    input_dir = args.input_dir
    output_dir = args.output_dir
    random.seed(run_id)
    np.random.seed(run_id)
    # random.seed(time.time())

    # Define file directory and name
    if method_str in ["quivr_original", "quivr_original_no_kleene", "quivr_original_active_learning", "quivr_original_no_kleene_active_learning"]:
        method_name = method_str
        config_name = "nip_{}-nin_{}-npred_{}-n_nontrivial_{}-n_trivial_{}-depth_{}-max_d_{}-thread_{}-lru_{}".format(n_init_pos, n_init_neg, npred, n_nontrivial, n_trivial, depth, max_duration, multithread, lru_capacity)
    elif method_str.startswith("vocal_postgres"):
        method_name = "{}-{}".format(method_str, strategy)
        config_name = "nip_{}-nin_{}-npred_{}-depth_{}-max_d_{}-nvars_{}-bw_{}-pool_size_{}-n_sampled_videos_{}-k_{}-budget_{}-thread_{}-lru_{}-lambda_{}".format(n_init_pos, n_init_neg, npred, depth, max_duration, max_vars, beam_width, pool_size, n_sampled_videos, k, budget, multithread, lru_capacity, reg_lambda)

    log_dirname = os.path.join(output_dir, dataset_name, method_name, config_name)
    log_filename = "{}-{}".format(query_str, run_id)
    # if dir not exist, create it
    if output_to_file:
        if not os.path.exists(os.path.join(log_dirname, "verbose")):
            os.makedirs(os.path.join(log_dirname, "verbose"), exist_ok=True)
        verbose_f = open(os.path.join(log_dirname, "verbose", "{}.log".format(log_filename)), 'w')
        sys.stdout = verbose_f


    # Define candidate predicates
    if dataset_name.startswith("trajectories_handwritten") or dataset_name.startswith("trajectories_duration"):
        if method_str.startswith("vocal_postgres"):
            predicate_dict = [{"name": "Near", "parameters": [1], "nargs": 2}, {"name": "Far", "parameters": [3], "nargs": 2}, {"name": "LeftOf", "parameters": None, "nargs": 2}, {"name": "Behind", "parameters": None, "nargs": 2}, {"name": "RightOf", "parameters": None, "nargs": 2}, {"name": "FrontOf", "parameters": None, "nargs": 2}, {"name": "RightQuadrant", "parameters": None, "nargs": 1}, {"name": "LeftQuadrant", "parameters": None, "nargs": 1}, {"name": "TopQuadrant", "parameters": None, "nargs": 1}, {"name": "BottomQuadrant", "parameters": None, "nargs": 1}]
        elif method_str in ["quivr_original", "quivr_original_active_learning"]:
            predicate_dict = {dsl.Near: [-1], dsl.Far: [3], dsl.MinLength: None, dsl.LeftOf: None, dsl.RightOf: None, dsl.Behind: None, dsl.FrontOf: None, dsl.LeftQuadrant: None, dsl.RightQuadrant: None, dsl.TopQuadrant: None, dsl.BottomQuadrant: None}
        elif method_str in ["quivr_original_no_kleene", "quivr_original_no_kleene_active_learning"]:
            predicate_dict = {dsl.Near: [-1], dsl.Far: [3], dsl.LeftOf: None, dsl.RightOf: None, dsl.Behind: None, dsl.FrontOf: None, dsl.LeftQuadrant: None, dsl.RightQuadrant: None, dsl.TopQuadrant: None, dsl.BottomQuadrant: None}
    elif dataset_name == "warsaw":
        if method_str.startswith("vocal_postgres"):
            predicate_dict = [{"name": "Eastward4", "parameters": None, "nargs": 1}, {"name": "Eastward3", "parameters": None, "nargs": 1}, {"name": "Eastward2", "parameters": None, "nargs": 1}, {"name": "Westward2", "parameters": None, "nargs": 1}, {"name": "Southward1Upper", "parameters": None, "nargs": 1}, {"name": "Stopped", "parameters": [2], "nargs": 1}, {"name": "HighAccel", "parameters": [2], "nargs": 1}, {"name": "DistanceSmall", "parameters": [100], "nargs": 2}, {"name": "Faster", "parameters": [1.5], "nargs": 2}]
        elif method_str in ["quivr_original", "quivr_original_active_learning"]:
            predicate_dict = {dsl.MinLength: None, dsl.AEastward4: None, dsl.AEastward3: None, dsl.AEastward2: None, dsl.AWestward2: None, dsl.ASouthward1Upper: None, dsl.AStopped: [-2], dsl.AHighAccel: [2], dsl.BEastward4: None, dsl.BEastward3: None, dsl.BEastward2: None, dsl.BWestward2: None, dsl.BSouthward1Upper: None, dsl.BStopped: [-2], dsl.BHighAccel: [2], dsl.DistanceSmall: [-100], dsl.Faster: [1.5]}
        elif method_str in ["quivr_original_no_kleene", "quivr_original_no_kleene_active_learning"]:
            predicate_dict = {dsl.AEastward4: None, dsl.AEastward3: None, dsl.AEastward2: None, dsl.AWestward2: None, dsl.ASouthward1Upper: None, dsl.AStopped: [-2], dsl.AHighAccel: [2], dsl.BEastward4: None, dsl.BEastward3: None, dsl.BEastward2: None, dsl.BWestward2: None, dsl.BSouthward1Upper: None, dsl.BStopped: [-2], dsl.BHighAccel: [2], dsl.DistanceSmall: [-100], dsl.Faster: [1.5]}
        else:
            raise NotImplementedError
    elif "scene_graph" in dataset_name:
        if method_str.startswith("vocal_postgres"):
            predicate_dict = [{"name": "Near", "parameters": [1], "nargs": 2}, {"name": "Far", "parameters": [3], "nargs": 2}, {"name": "LeftOf", "parameters": None, "nargs": 2}, {"name": "Behind", "parameters": None, "nargs": 2}, {"name": "RightOf", "parameters": None, "nargs": 2}, {"name": "FrontOf", "parameters": None, "nargs": 2}, {"name": "RightQuadrant", "parameters": None, "nargs": 1}, {"name": "LeftQuadrant", "parameters": None, "nargs": 1}, {"name": "TopQuadrant", "parameters": None, "nargs": 1}, {"name": "BottomQuadrant", "parameters": None, "nargs": 1}, {"name": "Color", "parameters": ["gray", "red", "blue", "green", "brown", "cyan", "purple", "yellow"], "nargs": 1}, {"name": "Shape", "parameters": ["cube", "sphere", "cylinder"], "nargs": 1}, {"name": "Material", "parameters": ["metal", "rubber"], "nargs": 1}]
        else:
            raise NotImplementedError
    elif dataset_name.startswith("without_duration"):
        query_strs = [
            "(BottomQuadrant(o0), Near_1.0(o0, o1))",
            "Conjunction(FrontOf(o0, o1), TopQuadrant(o0))",
            "Near_1(o0, o1); Far_3(o0, o1)",
            "Conjunction(Conjunction(Near_1(o0, o1), LeftQuadrant(o0)), Behind(o0, o1))",
            "Far_3(o0, o1); Near_1(o0, o1); Far_3(o0, o1)",
            "Conjunction(Far_3(o0, o1), BottomQuadrant(o0)); Near_1(o0, o1)",
            "Far_3(o0, o1); Conjunction(Near_1(o0, o1), Behind(o0, o1))",
            "Conjunction(Far_3(o0, o1), LeftQuadrant(o0)); Conjunction(Near_1(o0, o1), LeftQuadrant(o0))",
            "Far_3(o0, o1); Conjunction(Conjunction(Near_1(o0, o1), LeftQuadrant(o0)), Behind(o0, o1))"
            ]
        if method_str in ["quivr_original", "quivr_original_no_kleene", "quivr_original_active_learning", "quivr_original_no_kleene_active_learning"]:
            predicate_dicts = [
                {dsl.Near: [-1], dsl.BottomQuadrant: None},
                {dsl.FrontOf: None, dsl.TopQuadrant: None},
                {dsl.Near: [-1], dsl.Far: [3]},
                {dsl.Near: [-1], dsl.LeftQuadrant: None, dsl.Behind: None},
                {dsl.Near: [-1], dsl.Far: [3]},
                {dsl.Near: [-1], dsl.BottomQuadrant: None, dsl.Far: [3]},
                {dsl.Far: [3], dsl.Near: [-1], dsl.Behind: None},
                {dsl.Near: [-1], dsl.LeftQuadrant: None, dsl.Far: [3]},
                {dsl.Near: [-1], dsl.LeftQuadrant: None, dsl.Behind: None, dsl.Far: [3]}
            ]
        elif method_str.startswith("vocal_postgres"):
            predicate_dicts = [
                [{"name": "Near", "parameters": [1], "nargs": 2}, {"name": "BottomQuadrant", "parameters": None, "nargs": 1}],
                [{"name": "FrontOf", "parameters": None, "nargs": 2}, {"name": "TopQuadrant", "parameters": None, "nargs": 1}],
                [{"name": "Near", "parameters": [1], "nargs": 2}, {"name": "Far", "parameters": [3], "nargs": 2}],
                [{"name": "Near", "parameters": [1], "nargs": 2}, {"name": "LeftQuadrant", "parameters": None, "nargs": 1}, {"name": "Behind", "parameters": None, "nargs": 2}],
                [{"name": "Near", "parameters": [1], "nargs": 2}, {"name": "Far", "parameters": [3], "nargs": 2}],
                [{"name": "Near", "parameters": [1], "nargs": 2}, {"name": "BottomQuadrant", "parameters": None, "nargs": 1}, {"name": "Far", "parameters": [3], "nargs": 2}],
                [{"name": "Far", "parameters": [3], "nargs": 2}, {"name": "Near", "parameters": [1], "nargs": 2}, {"name": "Behind", "parameters": None, "nargs": 2}],
                [{"name": "Near", "parameters": [1], "nargs": 2}, {"name": "LeftQuadrant", "parameters": None, "nargs": 1}, {"name": "Far", "parameters": [3], "nargs": 2}],
                [{"name": "Near", "parameters": [1], "nargs": 2}, {"name": "LeftQuadrant", "parameters": None, "nargs": 1}, {"name": "Behind", "parameters": None, "nargs": 2}, {"name": "Far", "parameters": [3], "nargs": 2}]
            ]
        print("query string: ", query_str)
        query_idx = query_strs.index(query_str)
        predicate_dict = predicate_dicts[query_idx]
    print("predicate_dict", predicate_dict)

    print(args)

    if method_str == 'quivr_original':
        output_log = test_quivr_original(dataset_name, n_init_pos, n_init_neg, npred, n_nontrivial, n_trivial, depth, max_duration, budget, multithread, query_str, predicate_dict, lru_capacity, input_dir, with_kleene=True)
    elif method_str == 'quivr_original_no_kleene':
        output_log = test_quivr_original(dataset_name, n_init_pos, n_init_neg, npred, n_nontrivial, n_trivial, depth, max_duration, budget, multithread, query_str, predicate_dict, lru_capacity, input_dir, with_kleene=False)
    elif method_str == "exhaustive":
        test_exhaustive(n_init_pos, n_init_neg, npred, depth, max_duration, multithread, predicate_dict, input_dir)
    elif method_str in ['vocal_postgres', 'vocal_postgres_no_active_learning']:
        output_log = test_algorithm(method_str, dataset_name, n_init_pos, n_init_neg, npred, depth, max_duration, beam_width, pool_size, n_sampled_videos, k, budget, multithread, query_str, predicate_dict, lru_capacity, reg_lambda, strategy, max_vars, port, input_dir)
    elif method_str in ["quivr_original_active_learning", "quivr_original_no_kleene_active_learning"]:
        output_log = test_quivr_original_active_learning(dataset_name, n_init_pos, n_init_neg, npred, n_nontrivial, n_trivial, depth, max_duration, budget, multithread, query_str, predicate_dict, lru_capacity, input_dir, os.path.join(output_dir, dataset_name, method_name.replace("_active_learning", "_zero_step"), config_name), log_filename)
    if output_to_file:
        with open(os.path.join(log_dirname, "{}.log".format(log_filename)), 'w') as f:
            for line in output_log:
                f.write("{}\n".format(line))

        verbose_f.close()
        sys.stdout = sys.__stdout__