python \
    ../experiments/analysis/utils.py \
    --dataset_name "user_study_queries_scene_graph" \
    --test_query "(Behind(o1, o2), BottomQuadrant(o1), Color_purple(o1), material_metal(o1)); TopQuadrant(o2)" \
    --gt_query "(Behind(o0, o1), BottomQuadrant(o1), Color_purple(o0), material_metal(o0)); TopQuadrant(o1)" \
    --input_dir "/home/enhao/EQUI-VOCAL/inputs"