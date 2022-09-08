# python synthesize.py --method vocal --npred 5 --depth 3 --k 32 --max_duration 2


# Query: collision
# python synthesize.py --method vocal --npred 5 --depth 3 --k 32 --max_duration 2 --output_to_file

# Exhaustive search. Query: collision
# python synthesize.py --method vocal_exhaustive --n_labeled_pos 1 --n_labeled_neg 5 --npred 5 --depth 3 --k 32 --max_duration 2 --multithread 4 --output_to_file

# Query 1: "Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Front), True*), Duration(Left, 2)), True*), Conjunction(Far_0.9, Left)), True*)"
# Query 2: "Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Conjunction(Front, Left)), True*), Duration(Left, 2)), True*), Conjunction(Far_0.9, Left)), True*)"
# python synthesize.py --method vocal --n_init_pos 10 --n_init_neg 10 --npred 5 --depth 3 --max_duration 5 --beam_width 32 --k 100 --samples_per_iter 5 --multithread 4 --query_str "Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Back), True*), Left), True*), Conjunction(Conjunction(Back, Left), Far_0.9)), True*)" --run_id 0

python synthesize.py --method vocal --n_init_pos 10 --n_init_neg 10 --dataset_name "synthetic_rare" --npred 5 --depth 3 --max_duration 5 --beam_width 5 --k 100 --samples_per_iter 5 --multithread 4 --strategy "topk" --query_str "Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, BackOf), True*), Duration(Conjunction(Near_1.05, LeftOf), 4)), True*), Conjunction(TopQuadrant, Near_1.05)), True*)" --run_id 0 --output_to_file