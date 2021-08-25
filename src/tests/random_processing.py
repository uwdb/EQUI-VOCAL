import json
import random
from matplotlib import pyplot as plt

import numpy as np
from iterative_processing import IterativeProcessing

BATCH_SIZE = 1

class RandomProcessing(IterativeProcessing):
    def __init__(self) -> None:
        super().__init__()

    def random_sampling(self):
        while self.num_positive_instances_found < 15:
            normalized_p = self.p[self.frames_unseen] / self.p[self.frames_unseen].sum()
            frame_id = np.random.choice(self.frames_unseen.nonzero()[0], p=normalized_p)
            self.simulate_user_annotation(frame_id)


class RandomProcessingFiltered(IterativeProcessing):
    def __init__(self) -> None:
        super().__init__()

    def random_sampling(self):
        while self.num_positive_instances_found < 15:
            arr = self.frames_unseen * self.candidates
            normalized_p = self.p[arr] / self.p[arr].sum()
            frame_id = np.random.choice(arr.nonzero()[0], p=normalized_p)
            self.simulate_user_annotation(frame_id)


class RandomProcessingWithoutHeuristic(RandomProcessing):
    def __init__(self) -> None:
        super().__init__()

    def update_random_choice_p(self):
        p = np.ones(len(self.maskrcnn_bboxes))
        self.p = p


class RandomProcessingFilteredWithoutHeuristic(RandomProcessingFiltered):
    def __init__(self) -> None:
        super().__init__()

    def update_random_choice_p(self):
        p = np.ones(len(self.maskrcnn_bboxes))
        self.p = p


if __name__ == '__main__':
    plot_data_y_list = []
    for _ in range(100):
        ip = RandomProcessing()
        # ip = RandomProcessingFiltered()
        ip.random_sampling()
        plot_data_y_list.append(ip.get_plot_data_y())
    ip.save_data(plot_data_y_list, "random_without_heuristic")