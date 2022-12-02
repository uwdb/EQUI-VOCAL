from quivr.methods.vocal_postgres import VOCALPostgres
from sklearn.utils import resample
from sklearn.metrics import f1_score
import random
import time
from lru import LRU
import psycopg2 as psycopg
from quivr.utils import rewrite_program_postgres, str_to_program_postgres, complexity_cost
import resource
from functools import cmp_to_key
import numpy as np

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return '''%s: mem=%s MB'''%(point, usage/1024.0 )

class VOCALPostgresAdaptiveSearchSpaceBootstrap(VOCALPostgres):
    def run(self, init_labeled_index):
        self._start_total_time = time.time()
        self.init_nlabels = len(init_labeled_index)
        self.labeled_index = init_labeled_index

        bootstrap_times = 10
        labeled_index_bootstrapping_list = []
        for _ in range(bootstrap_times):
            labeled_index_bootstrapping = []
            labeled_index_bootstrapping.append(random.choice(self.labeled_index[:2]))
            labeled_index_bootstrapping.append(random.choice(self.labeled_index[2:]))
            labeled_index_bootstrapping += resample(self.labeled_index, n_samples=10)
            labeled_index_bootstrapping_list.append(labeled_index_bootstrapping)
        print("labeled_index_bootstrapping_list", labeled_index_bootstrapping_list)

        self.answers_bootstrapping = []
        for i in range(bootstrap_times):
            self.best_query_after_each_iter = []
            self.iteration = 0
            self.query_expansion_time = 0
            self.segment_selection_time = 0
            self.retain_top_k_queries_time = 0
            self.answers = []
            print("Run ", i)

            self.labeled_index = labeled_index_bootstrapping_list[i]
            if self.is_trajectory:
                if self.dataset_name.startswith("collision"):
                    list_size = 12747
                else:
                    list_size = 10080
            else:
                list_size = 10000

            if self.lru_capacity:
                self.memoize_scene_graph_all_inputs = [LRU(self.lru_capacity) for _ in range(list_size)]
                self.memoize_sequence_all_inputs = [LRU(self.lru_capacity) for _ in range(list_size)]
            else:
                self.memoize_scene_graph_all_inputs = [{} for _ in range(list_size)]
                self.memoize_sequence_all_inputs = [{} for _ in range(list_size)]
            self.candidate_queries = []

            self.main()

            # RETURN: the list.
            print("final_answers")
            self.answers_bootstrapping += self.answers
            for query_graph, score in self.answers:
                print("answer", rewrite_program_postgres(query_graph.program), score)
            total_time = time.time() - self._start_total_time
            print("[Runtime] query expansion time: {}, segment selection time: {}, retain top k queries time: {}, total time: {}".format(self.query_expansion_time, self.segment_selection_time, self.retain_top_k_queries_time, total_time))
            print("n_queries_explored", self.n_queries_explored)
            print("n_prediction_count", self.n_prediction_count)
            print(using("profile"))


        # Remove duplicates for self.answers_bootstrapping
        answers_removing_duplicates = []
        print("[self.answers_bootstrapping] before removing duplicates:", len(self.answers_bootstrapping))
        for query, score in self.answers_bootstrapping:
            print(rewrite_program_postgres(query.program), score)
        signatures = set()
        for query, score in self.answers_bootstrapping:
            signature = rewrite_program_postgres(query.program)
            if signature not in signatures:
                query.program = str_to_program_postgres(signature)
                answers_removing_duplicates.append([query, score])
                signatures.add(signature)
        print("[self.answers_bootstrapping] after removing duplicates:", len(answers_removing_duplicates))
        for query, score in answers_removing_duplicates:
            print(rewrite_program_postgres(query.program), score)
        self.answers_bootstrapping = answers_removing_duplicates


        # for query, _ in self.answers_bootstrapping:
        for i in range(len(self.answers_bootstrapping)):
            avg_score = []
            print(rewrite_program_postgres(self.answers_bootstrapping[i][0].program))
            for run in range(bootstrap_times):
                self.labeled_index = labeled_index_bootstrapping_list[run]
                query_score = self.get_query_score(self.answers_bootstrapping[i][0].program)
                avg_score.append(query_score)
            print(avg_score)
            avg_score = np.mean(avg_score)
            self.answers_bootstrapping[i][1] = avg_score

        self.answers = sorted(self.answers_bootstrapping, key=lambda x: cmp_to_key(self.compare_with_ties)(x[1]), reverse=True)
        utility_bound = self.answers[:self.k][-1][1]
        self.answers = [e for e in self.answers if e[1] >= utility_bound]
        print("top k queries", [(rewrite_program_postgres(query.program), score) for query, score in self.answers])

        self.output_log.append("[Step 0]")
        for query, score in self.answers:
            self.output_log.append((rewrite_program_postgres(query.program), score))
        self.output_log.append("[Runtime so far] {}".format(time.time() - self._start_total_time))
        self.output_log.append("[Final answers]")

        # Drop input table
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE {};".format(self.inputs_table_name))

        return self.output_log