python \
    ../src/synthesize.py \
    --method vocal_postgres `# options: vocal_postgres, vocal_postgres_no_active_learning`\
    --n_init_pos 10 \
    --n_init_neg 10 \
    --dataset_name "synthetic_scene_graph_hard" `# options: synthetic_scene_graph_easy, synthetic_scene_graph_medium, synthetic_scene_graph_hard`\
    --query_str "(Behind(o0, o1), Material(o2, 'metal')); (FrontOf(o0, o1), LeftOf(o0, o1)); Duration((Near(o0, o2, 1.0), Shape(o0, 'cube'), TopQuadrant(o1)), 10)" `# A hard query` \
    --npred 7 \
    --depth 3 \
    --max_duration 15 \
    --max_vars 3 \
    --beam_width 10 \
    --pool_size 100 \
    --k 100 \
    --budget 50 \
    --multithread 4 \
    --strategy "topk" \
    --run_id 0 \
    --output_dir "../outputs_test" \
    --reg_lambda 0.001