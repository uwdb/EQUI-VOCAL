from matplotlib import pyplot as plt
import numpy as np
import json

if __name__ == '__main__':
    fig, ax = plt.subplots(1)

    with open('iterative_processing.json', 'r') as f:
        iterative_y_list = json.loads(f.read())
    for plot_data_y in iterative_y_list:
        x_values = range(len(plot_data_y))
        ax.plot(x_values, plot_data_y, color='tab:blue', alpha=0.3)

    with open('random_processing.json', 'r') as f:
        random_y_list = json.loads(f.read())
    for plot_data_y in random_y_list:
        x_values = range(len(plot_data_y))
        ax.plot(x_values, plot_data_y, color='tab:orange', alpha=0.3)

    with open('random_processing_with_roi.json', 'r') as f:
        random_roi_y_list = json.loads(f.read())
    for plot_data_y in random_roi_y_list:
        x_values = range(len(plot_data_y))
        ax.plot(x_values, plot_data_y, color='tab:green', alpha=0.3)

    ax.set_ylabel('number of positive instances the user finds')
    ax.set_xlabel('number of frames that user has seen')
    ax.grid()
    plt.savefig("aggregate_plot")