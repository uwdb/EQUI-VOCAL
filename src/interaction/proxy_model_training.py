from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
import numpy as np

class ProxyModelTrainingMixin:
    def random_forest_training(self):
        """Input: annotated frames Fa_i
        Output: updated model (or model set) m_i+1
        """
        self.clf = RandomForestClassifier(
            # criterion="entropy",
            # max_depth=10,
            n_estimators=50,
            # min_samples_split=32,
            class_weight="balanced",
            # min_impurity_decrease=0.1
        )
        # clf = BalancedRandomForestClassifier(n_estimators=50, random_state=42)
        train_x = self.spatial_features[~(self.raw_frames | self.materialized_frames)]
        train_y = self.Y[~(self.raw_frames | self.materialized_frames)]
        self.clf = self.clf.fit(train_x, train_y)