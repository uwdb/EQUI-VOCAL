import random
from matplotlib import pyplot as plt

import numpy as np
from iterative_processing import IterativeProcessing

BATCH_SIZE = 1

class RandomProcessing(IterativeProcessing):
    def __init__(self) -> None:
        super().__init__()

    def random_sampling(self):
        while len(self.positive_frames_seen) < 912:
            frame_idx = random.choice(self.frames_unseen)
            self.simulate_user_annotation(frame_idx)


def plot_data(plot_data_y_list):
    np.savetxt('random_processing.csv', plot_data_y_list, fmt='%d', delimiter=',')
    y_upper = np.max(plot_data_y_list, axis=0)
    y_lower = np.min(plot_data_y_list, axis=0)
    y_mean = np.mean(plot_data_y_list, axis=0)
    x_values = range(plot_data_y_list.shape[1])
    plt.fill_between(x_values, y_lower, y_upper, alpha=0.2)
    plt.plot(x_values, y_mean)
    plt.savefig("random_processing_plot")


if __name__ == '__main__':
    plot_data_y_list = np.empty((0, 913))
    for _ in range(20):
        ip = RandomProcessing()
        ip.random_sampling()
        if ip.get_plot_data_y().shape[0] < 913:
            print("Warning: failed to find all positive frames. Ignore this iteration.")
        else:
            plot_data_y_list = np.vstack((plot_data_y_list, ip.get_plot_data_y()))
    plot_data(plot_data_y_list)