from sklearn.ensemble import RandomForestClassifier
import numpy as np

class ProxyModelTraining:
    def __init__(self, spatial_features, Y):
        self.spatial_features = spatial_features
        self.Y = Y

    def run(self, raw_frames, materialized_frames):
        """Input: annotated frames Fa_i
        Output: updated model (or model set) m_i+1
        """
        clf = RandomForestClassifier(
            # criterion="entropy",
            # max_depth=10,
            n_estimators=10,
            # min_samples_split=32,
            class_weight="balanced"
        )
        clf = clf.fit(self.spatial_features[~(raw_frames | materialized_frames)], self.Y[~(raw_frames | materialized_frames)])
        return clf