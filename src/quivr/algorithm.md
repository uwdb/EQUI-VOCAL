User initializes hard constraints (e.g., must have a red cube)

labeled_data = [10 positive and negative examples]
candidates_list = [("red cube" and ??, score)]

// Generate K1 candidates from the initial query
while len(candidates_list) < K1:
    Compute all the children of Q.
    For each child Q':
        compute the score of Q' based on FN lower bound (using overapproximation), FP lower bound (using underapproximation), and structure (or depth) of query. Add (Q', score) to candidates_list

while len(labeled_data) < N:
    // Pick next data to label
    For each top K1 query in candidates_list:
        compute the weight of each query, and select the next video segment to label based on (weighted) disagreement among the top K1 queries.
        add the selected video segment to labeled_data.
    // Update scores
    For each top K2 query Q' in candidates_list:
        Update the score of Q'.
    // Expand search space
    Find the query Q with the highest score in candidates_list that is incomplete.
    Find all the children of Q.
    For each child Q':
        if Q' is a sketch:
            compute all parameters for Q' that are at least x% consistent with W (for each parameter in Q', compute its largest/smallest possible value when overapproximating all other parameters).
            add them along with their scores to candidates_list
        else:
            add (Q', score) to candidates_list
