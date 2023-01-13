
from experiments.processing.prepare_trajectories_data_postgres import prepare_data_given_target_query

if __name__ == "__main__":
    sampling_rate = 2
    query_strs = [
        # "Conjunction(Near_1(o0, o1), BottomQuadrant(o0))",
        # "Conjunction(FrontOf(o0, o1), TopQuadrant(o0))",
        # "Near_1(o0, o1); Far_3(o0, o1)",
        # "Conjunction(Conjunction(Near_1(o0, o1), LeftQuadrant(o0)), Behind(o0, o1))",
        # "Far_3(o0, o1); Near_1(o0, o1); Far_3(o0, o1)",
        # "Conjunction(Far_3(o0, o1), BottomQuadrant(o0)); Near_1(o0, o1)",
        # "Far_3(o0, o1); Conjunction(Near_1(o0, o1), Behind(o0, o1))",
        # "Conjunction(Far_3(o0, o1), LeftQuadrant(o0)); Conjunction(Near_1(o0, o1), LeftQuadrant(o0))",
        "Far_3(o0, o1); Conjunction(Conjunction(Near_1(o0, o1), LeftQuadrant(o0)), Behind(o0, o1))"
        ]
    for query in query_strs:
        prepare_data_given_target_query(query, ratio_lower_bound=0.0, ratio_upper_bound=1.0, dataset_name="trajectories_handwritten-sampling_rate_{}".format(sampling_rate), port=5432, sampling_rate=sampling_rate)