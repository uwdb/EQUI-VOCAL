from quivr.methods.vocal_postgres import VOCALPostgres

class VOCALPostgresMostLikelyPositive(VOCALPostgres):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_learning = self.pick_next_segment_most_likely_positive_postgres
