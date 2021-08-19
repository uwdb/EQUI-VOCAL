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

class RandomProcessingWithROI(IterativeProcessing):
    def __init__(self) -> None:
        super().__init__()

    def random_sampling(self):
        while self.num_positive_instances_found < 15:
            arr = self.frames_unseen * self.candidates
            normalized_p = self.p[arr] / self.p[arr].sum()
            frame_id = np.random.choice(arr.nonzero()[0], p=normalized_p)
            self.simulate_user_annotation(frame_id)

def plot_data(plot_data_y_list, method):
    with open("{}.json".format(method), 'w') as f:
        f.write(json.dumps([arr.tolist() for arr in plot_data_y_list]))
    fig, ax = plt.subplots(1)
    for plot_data_y in plot_data_y_list:
        x_values = range(plot_data_y.size)
        ax.plot(x_values, plot_data_y, color='tab:blue')
    ax.set_ylabel('number of positive instances the user finds')
    ax.set_xlabel('number of frames that user has seen')
    ax.grid()
    plt.savefig("{}".format(method))

if __name__ == '__main__':
    plot_data_y_list = []
    for _ in range(100):
        # ip = RandomProcessing()
        ip = RandomProcessingWithROI()
        ip.random_sampling()
        plot_data_y_list.append(ip.get_plot_data_y())
    # plot_data(plot_data_y_list, "random")
    plot_data(plot_data_y_list, "random_roi")