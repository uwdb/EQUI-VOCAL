import json
import random
from matplotlib import pyplot as plt
import os
import numpy as np
from iterative_processing import IterativeProcessing
from matplotlib import rcParams
import itertools
clist = rcParams['axes.prop_cycle']
cgen = itertools.cycle(clist)

AVG_DURATION = 61
BATCH_SIZE = 1
root_dir = "temporal_selection"
if not os.path.exists(root_dir):
    os.makedirs(root_dir)


class TemporalSelection(IterativeProcessing):
    def __init__(self, scale=1, func="linear") -> None:
        super().__init__()
        self.scale = scale
        if func == "linear":
            self.func = lambda x : x / int(AVG_DURATION * scale)
        elif func == "quadratic":
            self.func = lambda x : (x ** 2) / (int(AVG_DURATION * scale) ** 2)
    def update_random_choice_p(self):
        p = np.ones(len(self.maskrcnn_bboxes))
        for frame_id in self.positive_frames_seen:
            # Right half of the probability function
            for i in range(int(AVG_DURATION * self.scale) + 1):
                if frame_id + i < len(self.maskrcnn_bboxes):
                    p[frame_id + i] = min(self.func(i), p[frame_id + i])
            # Left half of the probability function
            for i in range(int(AVG_DURATION * self.scale) + 1):
                if frame_id - i >= 0:
                    p[frame_id - i] = min(self.func(i), p[frame_id - i])
        self.p = p

def save_data(plot_data_y_list, method):
    with open(os.path.join(root_dir, "{}.json".format(method)), 'w') as f:
        f.write(json.dumps([arr.tolist() for arr in plot_data_y_list]))

def plot_data():
    best_method = [None, 99999]
    fig, ax = plt.subplots(1)
    legend_items = []
    for filename in os.listdir(root_dir):
        if not filename.endswith(".json"):
            continue
        parsed = filename.strip(".json")
        with open(os.path.join(root_dir, filename), 'r') as f:
            iterative_y_list = json.loads(f.read())
        ip_y = []
        for plot_data_y in iterative_y_list:
            current = 0
            num_frames_as_y = []
            for i, y in enumerate(plot_data_y):

                if y > current:
                    num_frames_as_y.append(i)
                    current = y
            ip_y.append(num_frames_as_y)
        ip_y = np.asarray(ip_y)
        ip_y_upper = np.percentile(ip_y, 75, axis=0)
        ip_y_lower = np.percentile(ip_y, 25, axis=0)
        ip_y_mean = np.percentile(ip_y, 50, axis=0)
        if ip_y_mean[-1] < best_method[1]:
            best_method = [parsed, ip_y_mean[-1]]
        # ip_y_mean = np.mean(ip_y, axis=0)
        ip_x_values = range(ip_y.shape[1])
        color = next(cgen)["color"]
        line, = ax.plot(ip_y_mean, ip_x_values, label=parsed, color=color)
        legend_items.append((parsed, line))
        # ax.fill_betweenx(ip_x_values, ip_y_lower, ip_y_upper, facecolor=color, alpha=0.2)
    legend_items.sort()
    ax.legend([x[1] for x in legend_items], [x[0] for x in legend_items], loc='lower right')
    ax.set_xlabel('number of frames that user has seen')
    ax.set_ylabel('number of positive frames the user finds')
    ax.grid()
    # ['quadratic-2', 99.5]
    print("Best method: ", best_method)
    plt.savefig(os.path.join(root_dir, "plot"))

if __name__ == '__main__':
    scale_list = [0.5, 1, 2, 3, 4, 8, 16]
    func_list = ["linear", "quadratic"]
    for scale in scale_list:
        for func in func_list:
            plot_data_y_list = []
            for _ in range(100):
                ip = TemporalSelection(scale, func)
                # Cold start
                print("Cold start:")
                ip.random_sampling()
                print("Cold start done.")
                # Iterative processing
                batched_frames = np.empty(BATCH_SIZE)  # Construct a pseudo list of length BATCH_SIZE to pass the While condition.
                while ip.get_num_positive_instances_found() < 15 and batched_frames.size >= BATCH_SIZE:
                    batched_frames = ip.get_next_batch()
                    for frame_id in batched_frames:
                        ip.simulate_user_annotation(frame_id)
                plot_data_y_list.append(ip.get_plot_data_y())
            save_data(plot_data_y_list, func + "-" + str(scale))
    plot_data()