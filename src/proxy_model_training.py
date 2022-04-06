from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
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
            n_estimators=50,
            # min_samples_split=32,
            class_weight="balanced",
            # min_impurity_decrease=0.1
        )
        # clf = BalancedRandomForestClassifier(n_estimators=50, random_state=42)
        train_x = self.spatial_features[~(raw_frames | materialized_frames)]
        train_y = self.Y[~(raw_frames | materialized_frames)]
        clf = clf.fit(train_x, train_y)
        return clf