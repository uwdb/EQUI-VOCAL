from methods.vocal_postgres import VOCALPostgres
from utils import rewrite_program_postgres

class VOCALPostgresDurationRefinementLast(VOCALPostgres):
    def expand_query_and_compute_score(self, current_query):
        current_query_graph, _ = current_query
        current_query = current_query_graph.program
        print("expand search space", rewrite_program_postgres(current_query))
        all_children = current_query_graph.get_all_children_duration_refinement_last_postgres()

        new_candidate_queries = []

        # Compute F1 score of each candidate query
        for child in all_children:
            score = self.compute_query_score_postgres(child.program)
            if score > 0:
                new_candidate_queries.append([child, score])
                print(rewrite_program_postgres(child.program), score)
        return new_candidate_queries