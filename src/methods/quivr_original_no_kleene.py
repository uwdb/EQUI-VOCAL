from src.methods.quivr_original import QUIVROriginal

class QUIVROriginalNoKleene(QUIVROriginal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_kleene = False