# python synthesize.py --method vocal --npred 5 --depth 3 --k 32 --max_duration 2


# Query: collision
# python synthesize.py --method vocal --npred 5 --depth 3 --k 32 --max_duration 2 --output_to_file

# Exhaustive search. Query: collision
python synthesize.py --method vocal_exhaustive --n_labeled_pos 1 --n_labeled_neg 5 --npred 5 --depth 3 --k 32 --max_duration 2 --multithread 4 --output_to_file

# Query 1: "Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Front), True*), Duration(Left, 2)), True*), Conjunction(Far_0.9, Left)), True*)"
# Query 2: "Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Conjunction(Front, Left)), True*), Duration(Left, 2)), True*), Conjunction(Far_0.9, Left)), True*)"
# python synthesize.py --method vocal --npred 5 --depth 3 --k 32 --max_duration 2 --query_str "Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Conjunction(Front, Left)), True*), Duration(Left, 2)), True*), Conjunction(Far_0.9, Left)), True*)" --output_to_file
