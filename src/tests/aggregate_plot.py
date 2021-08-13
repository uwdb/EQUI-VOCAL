from matplotlib import pyplot as plt
import numpy as np


if __name__ == '__main__':
    ip_y = np.genfromtxt("iterative_processing.csv", delimiter=',')
    ip_y_upper = np.max(ip_y, axis=0)
    ip_y_lower = np.min(ip_y, axis=0)
    ip_y_mean = np.mean(ip_y, axis=0)
    ip_x_values = range(ip_y.shape[1])
    rp_y = np.genfromtxt("random_processing.csv", delimiter=',')[:,:822]
    rp_y_upper = np.max(rp_y, axis=0)
    rp_y_lower = np.min(rp_y, axis=0)
    rp_y_mean = np.mean(rp_y, axis=0)
    rp_x_values = range(rp_y.shape[1])
    # plot it!
    fig, ax = plt.subplots(1)
    ax.plot(ip_x_values, ip_y_mean, label='iterative', color='tab:blue')
    ax.plot(rp_x_values, rp_y_mean, label='random', color='tab:orange')
    ax.fill_between(ip_x_values, ip_y_lower, ip_y_upper, facecolor='tab:blue', alpha=0.2)
    ax.fill_between(rp_x_values, rp_y_lower, rp_y_upper, facecolor='tab:orange', alpha=0.2)
    ax.legend(loc='upper left')
    ax.set_ylabel('number of frames that user has seen')
    ax.set_xlabel('number of positive frames the user finds')
    ax.grid()
    plt.savefig("aggregate_plot")