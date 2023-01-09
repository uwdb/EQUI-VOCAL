from experiments.processing.prepare_data_postgres import prepare_data_given_target_query, construct_train_test

if __name__ == '__main__':
    query_strs = [
        "Duration(Conjunction(Far_3(o0, o1), RightQuadrant(o1)), 15); Conjunction(FrontOf(o0, o1), Near_1(o0, o1)); Duration(Conjunction(Far_3(o0, o1), RightQuadrant(o1)), 5)",
        "Conjunction(Conjunction(Color_cyan(o0), Far_3(o1, o2)), Shape_sphere(o1)); Conjunction(FrontOf(o1, o2), Near_1(o1, o2)); Conjunction(Far_3(o1, o2), TopQuadrant(o2))",
        "Duration(Conjunction(LeftOf(o0, o1), Shape_sphere(o1)), 15); Conjunction(FrontOf(o0, o2), Near_1(o0, o2)); Duration(Conjunction(Behind(o0, o2), Far_3(o0, o2)), 5)"
        ]
    dataset_name = "scene_graphs"
    for query_str in query_strs:
        prepare_data_given_target_query(query_str, 0, 1, dataset_name, "Obj_clevrer", sampling_rate=None)
        construct_train_test("/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/inputs/{}".format(dataset_name), n_train=500)