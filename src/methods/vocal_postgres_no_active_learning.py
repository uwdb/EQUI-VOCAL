from src.methods.vocal_postgres import VOCALPostgres

class VOCALPostgresNoActiveLearning(VOCALPostgres):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_learning = self.pick_next_segment_randomly_postgres