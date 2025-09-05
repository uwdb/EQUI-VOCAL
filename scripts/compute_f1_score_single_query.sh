python \
    ../experiments/analysis/utils.py \
    --dataset_name "user_study_queries_scene_graph" \
    --test_query "(Behind(o1, o2), BottomQuadrant(o1), Color(o1, 'purple'), material(o1, 'metal')); TopQuadrant(o2)" \
    --gt_query "(Behind(o0, o1), BottomQuadrant(o1), Color(o0, 'purple'), material(o0, 'metal')); TopQuadrant(o1)" \
    --input_dir "/home/enhao/EQUI-VOCAL/inputs"