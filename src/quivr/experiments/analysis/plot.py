import matplotlib.pyplot as plt
import numpy as np
import os
import json

def plot_query_median(query_str):
    exp_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/without_duration-sampling_rate_4"

    fig, axs = plt.subplots(1, 2, figsize = (12, 3))

    # PATSQL
    try:
        with open(os.path.join(exp_dir, "stats", "PATSQL", "{}.json".format(query_str)), "r") as f:
            patsql_stats = json.load(f)
        patsql_runtime = patsql_stats["runtime"]
        patsql_f1 = patsql_stats["score"]
        patsql_x = list(range(12, 21)) + list(range(25, 51, 5))
        patsql_f1 = np.array(patsql_f1)
        if np.all(patsql_f1 == -1):
            patsql_f1 = np.zeros_like(patsql_f1)
        else:
            patsql_f1[patsql_f1 == -1] = np.nan # replace -1 with nan
            patsql_f1[:, np.all(np.isnan(patsql_f1), axis=0)] = 0 # If all values are nan, replace with 0
    #     patsql_f1[patsql_f1 == -1] = 0 # replace -1 with nan
        patsql_f1_25 = np.nanpercentile(patsql_f1, 25, axis=0)
        patsql_f1_50 = np.nanpercentile(patsql_f1, 50, axis=0)
        patsql_f1_75 = np.nanpercentile(patsql_f1, 75, axis=0)
        patsql_runtime = np.array(patsql_runtime)
        patsql_runtime_25 = np.percentile(patsql_runtime, 25, axis=0)
        patsql_runtime_50 = np.percentile(patsql_runtime, 50, axis=0)
        patsql_runtime_75 = np.percentile(patsql_runtime, 75, axis=0)
        axs[0].plot(patsql_x, patsql_f1_50, marker='o', markersize=1, label="patsql", color='tab:orange')
        # axs[0].fill_between(patsql_x, patsql_f1_25, patsql_f1_75, facecolor='tab:orange', alpha=0.3)
        axs[1].plot(patsql_x, patsql_runtime_50, marker='o', markersize=1, label="patsql", color='tab:orange')
        # axs[1].fill_between(patsql_x, patsql_runtime_25, patsql_runtime_75, facecolor='tab:orange', alpha=0.3)
    except:
        pass


    # Quivr without kleene
    try:
        quivr_f1 = []
        quivr_runtime = []
        for run in range(20):
            with open(os.path.join(exp_dir, "stats", "quivr_original_no_kleene", "{}-{}.json".format(query_str, run)), "r") as f:
                quivr_stats = json.load(f)
            quivr_runtime_per_run = quivr_stats["runtime"]
            quivr_f1_per_run = quivr_stats["score"]
            quivr_f1.append(quivr_f1_per_run)
            quivr_runtime.append(quivr_runtime_per_run)

        quivr_x = list(range(12, 51))
        quivr_f1 = np.array(quivr_f1)
        quivr_f1_25 = np.percentile(quivr_f1, 25, axis=0)
        quivr_f1_50 = np.percentile(quivr_f1, 50, axis=0)
        quivr_f1_75 = np.percentile(quivr_f1, 75, axis=0)
        quivr_runtime = np.array(quivr_runtime)
        quivr_runtime_25 = np.percentile(quivr_runtime, 25, axis=0)
        quivr_runtime_50 = np.percentile(quivr_runtime, 50, axis=0)
        quivr_runtime_75 = np.percentile(quivr_runtime, 75, axis=0)
        axs[0].plot(quivr_x, quivr_f1_50, marker='s', markersize=1, label="quivr (no kleene)", color='tab:blue')
        # axs[0].fill_between(quivr_x, quivr_f1_25, quivr_f1_75, facecolor='tab:blue', alpha=0.3)
        axs[1].plot(quivr_x, quivr_runtime_50, marker='s', markersize=1, label="quivr (no kleene)", color='tab:blue')
        # axs[1].fill_between(quivr_x, quivr_runtime_25, quivr_runtime_75, facecolor='tab:blue', alpha=0.3)
    except:
        pass

    # Quivr with kleene
    try:
        quivr_with_kleene_f1 = []
        quivr_with_kleene_runtime = []
        for run in range(20):
            with open(os.path.join(exp_dir, "stats", "quivr_original", "{}-{}.json".format(query_str, run)), "r") as f:
                quivr_with_kleene_stats = json.load(f)
            quivr_with_kleene_runtime_per_run = quivr_with_kleene_stats["runtime"]
            quivr_with_kleene_f1_per_run = quivr_with_kleene_stats["score"]
            quivr_with_kleene_f1.append(quivr_with_kleene_f1_per_run)
            quivr_with_kleene_runtime.append(quivr_with_kleene_runtime_per_run)

        quivr_with_kleene_x = list(range(12, 51))
        quivr_with_kleene_f1 = np.array(quivr_with_kleene_f1)
        quivr_with_kleene_f1_25 = np.percentile(quivr_with_kleene_f1, 25, axis=0)
        quivr_with_kleene_f1_50 = np.percentile(quivr_with_kleene_f1, 50, axis=0)
        quivr_with_kleene_f1_75 = np.percentile(quivr_with_kleene_f1, 75, axis=0)
        quivr_with_kleene_runtime = np.array(quivr_with_kleene_runtime)
        quivr_with_kleene_runtime_25 = np.percentile(quivr_with_kleene_runtime, 25, axis=0)
        quivr_with_kleene_runtime_50 = np.percentile(quivr_with_kleene_runtime, 50, axis=0)
        quivr_with_kleene_runtime_75 = np.percentile(quivr_with_kleene_runtime, 75, axis=0)
        axs[0].plot(quivr_with_kleene_x, quivr_with_kleene_f1_50, marker='s', markersize=1, label="quivr (kleene)", color='tab:red')
        # axs[0].fill_between(quivr_with_kleene_x, quivr_with_kleene_f1_25, quivr_with_kleene_f1_75, facecolor='tab:red', alpha=0.3)
        axs[1].plot(quivr_with_kleene_x, quivr_with_kleene_runtime_50, marker='s', markersize=1, label="quivr (kleene)", color='tab:red')
        # axs[1].fill_between(quivr_with_kleene_x, quivr_with_kleene_runtime_25, quivr_with_kleene_runtime_75, facecolor='tab:red', alpha=0.3)
    except:
        pass

    # Quivr with kleene; simplest queries
    try:
        quivr_with_kleene_f1 = []
        quivr_with_kleene_runtime = []
        for run in range(20):
            with open(os.path.join(exp_dir, "stats", "quivr_original", "simplest_queries", "{}-{}.json".format(query_str, run)), "r") as f:
                quivr_with_kleene_stats = json.load(f)
            quivr_with_kleene_runtime_per_run = quivr_with_kleene_stats["runtime"]
            quivr_with_kleene_f1_per_run = quivr_with_kleene_stats["score"]
            quivr_with_kleene_f1.append(quivr_with_kleene_f1_per_run)
            quivr_with_kleene_runtime.append(quivr_with_kleene_runtime_per_run)

        quivr_with_kleene_x = list(range(12, 51))
        quivr_with_kleene_f1 = np.array(quivr_with_kleene_f1)
        quivr_with_kleene_f1_25 = np.percentile(quivr_with_kleene_f1, 25, axis=0)
        quivr_with_kleene_f1_50 = np.percentile(quivr_with_kleene_f1, 50, axis=0)
        quivr_with_kleene_f1_75 = np.percentile(quivr_with_kleene_f1, 75, axis=0)
        quivr_with_kleene_runtime = np.array(quivr_with_kleene_runtime)
        quivr_with_kleene_runtime_25 = np.percentile(quivr_with_kleene_runtime, 25, axis=0)
        quivr_with_kleene_runtime_50 = np.percentile(quivr_with_kleene_runtime, 50, axis=0)
        quivr_with_kleene_runtime_75 = np.percentile(quivr_with_kleene_runtime, 75, axis=0)
        axs[0].plot(quivr_with_kleene_x, quivr_with_kleene_f1_50, linestyle='--', marker='s', markersize=1, label="quivr (kleene; simplest)", color='tab:pink')
        # axs[0].fill_between(quivr_with_kleene_x, quivr_with_kleene_f1_25, quivr_with_kleene_f1_75, facecolor='tab:pink', alpha=0.3)
        axs[1].plot(quivr_with_kleene_x, quivr_with_kleene_runtime_50, linestyle='--', marker='s', markersize=1, label="quivr (kleene; simplest)", color='tab:pink')
        # axs[1].fill_between(quivr_with_kleene_x, quivr_with_kleene_runtime_25, quivr_with_kleene_runtime_75, facecolor='tab:pink', alpha=0.3)
    except:
        pass


    # Quivr without kleene; simplest queries
    try:
        quivr_with_kleene_f1 = []
        quivr_with_kleene_runtime = []
        for run in range(20):
            with open(os.path.join(exp_dir, "stats", "quivr_original_no_kleene", "simplest_queries", "{}-{}.json".format(query_str, run)), "r") as f:
                quivr_with_kleene_stats = json.load(f)
            quivr_with_kleene_runtime_per_run = quivr_with_kleene_stats["runtime"]
            quivr_with_kleene_f1_per_run = quivr_with_kleene_stats["score"]
            quivr_with_kleene_f1.append(quivr_with_kleene_f1_per_run)
            quivr_with_kleene_runtime.append(quivr_with_kleene_runtime_per_run)

        quivr_with_kleene_x = list(range(12, 51))
        quivr_with_kleene_f1 = np.array(quivr_with_kleene_f1)
        quivr_with_kleene_f1_25 = np.percentile(quivr_with_kleene_f1, 25, axis=0)
        quivr_with_kleene_f1_50 = np.percentile(quivr_with_kleene_f1, 50, axis=0)
        quivr_with_kleene_f1_75 = np.percentile(quivr_with_kleene_f1, 75, axis=0)
        quivr_with_kleene_runtime = np.array(quivr_with_kleene_runtime)
        quivr_with_kleene_runtime_25 = np.percentile(quivr_with_kleene_runtime, 25, axis=0)
        quivr_with_kleene_runtime_50 = np.percentile(quivr_with_kleene_runtime, 50, axis=0)
        quivr_with_kleene_runtime_75 = np.percentile(quivr_with_kleene_runtime, 75, axis=0)
        axs[0].plot(quivr_with_kleene_x, quivr_with_kleene_f1_50, linestyle='--', marker='s', markersize=1, label="quivr (no kleene; simplest)", color='tab:cyan')
        # axs[0].fill_between(quivr_with_kleene_x, quivr_with_kleene_f1_25, quivr_with_kleene_f1_75, facecolor='tab:cyan', alpha=0.3)
        axs[1].plot(quivr_with_kleene_x, quivr_with_kleene_runtime_50, linestyle='--', marker='s', markersize=1, label="quivr (no kleene; simplest)", color='tab:cyan')
        # axs[1].fill_between(quivr_with_kleene_x, quivr_with_kleene_runtime_25, quivr_with_kleene_runtime_75, facecolor='tab:cyan', alpha=0.3)
    except:
        pass


    # VOCAL
    try:
        with open(os.path.join(exp_dir, "stats", "vocal_postgres-topk-test", "{}.json".format(query_str)), "r") as f:
            vocal_stats = json.load(f)
        vocal_runtime = vocal_stats["runtime"]
        vocal_f1 = vocal_stats["score"]
        vocal_x = list(range(12, 21)) + list(range(25, 51, 5))
        vocal_f1 = np.array(vocal_f1)
        vocal_f1_25 = np.percentile(vocal_f1, 25, axis=0)
        vocal_f1_50 = np.percentile(vocal_f1, 50, axis=0)
        vocal_f1_75 = np.percentile(vocal_f1, 75, axis=0)
        vocal_runtime = np.array(vocal_runtime)
        vocal_runtime_25 = np.percentile(vocal_runtime, 25, axis=0)
        vocal_runtime_50 = np.percentile(vocal_runtime, 50, axis=0)
        vocal_runtime_75 = np.percentile(vocal_runtime, 75, axis=0)
        axs[0].plot(vocal_x, vocal_f1_50, marker='^', markersize=1, label="vocal", color='tab:green')
        # axs[0].fill_between(vocal_x, vocal_f1_25, vocal_f1_75, facecolor='tab:green', alpha=0.3)
        axs[1].plot(vocal_x, vocal_runtime_50, marker='^', markersize=1, label="vocal", color='tab:green')
        # axs[1].fill_between(vocal_x, vocal_runtime_25, vocal_runtime_75, facecolor='tab:green', alpha=0.3)
    except:
        pass

    #     # VOCAL (sample one segment at a time)
    #     with open(os.path.join(exp_dir, "stats", "vocal_postgres-topk-finegrained", "{}.json".format(query_str)), "r") as f:
    #         vocal_finegrained_stats = json.load(f)
    #     vocal_finegrained_runtime = vocal_finegrained_stats["runtime"]
    #     vocal_finegrained_f1 = vocal_finegrained_stats["score"]
    #     vocal_finegrained_x = list(range(12, 21)) + list(range(25, 51, 5))
    #     vocal_finegrained_f1 = np.array(vocal_finegrained_f1)
    #     vocal_finegrained_f1_25 = np.percentile(vocal_finegrained_f1, 25, axis=0)
    #     vocal_finegrained_f1_50 = np.percentile(vocal_finegrained_f1, 50, axis=0)
    #     vocal_finegrained_f1_75 = np.percentile(vocal_finegrained_f1, 75, axis=0)
    #     vocal_finegrained_runtime = np.array(vocal_finegrained_runtime)
    #     vocal_finegrained_runtime_25 = np.percentile(vocal_finegrained_runtime, 25, axis=0)
    #     vocal_finegrained_runtime_50 = np.percentile(vocal_finegrained_runtime, 50, axis=0)
    #     vocal_finegrained_runtime_75 = np.percentile(vocal_finegrained_runtime, 75, axis=0)
    #     axs[0].plot(vocal_finegrained_x, vocal_finegrained_f1_50, marker='^', markersize=1, label="vocal (finegrained)", color='tab:purple')
    #     axs[0].fill_between(vocal_finegrained_x, vocal_finegrained_f1_25, vocal_finegrained_f1_75, facecolor='tab:purple', alpha=0.3)
    #     axs[1].plot(vocal_finegrained_x, vocal_finegrained_runtime_50, marker='^', markersize=1, label="vocal (finegrained)", color='tab:purple')
    #     axs[1].fill_between(vocal_finegrained_x, vocal_finegrained_runtime_25, vocal_finegrained_runtime_75, facecolor='tab:purple', alpha=0.3)


    # VOCAL (sample one segment at a time; with cache)
    try:
        with open(os.path.join(exp_dir, "stats", "vocal_postgres-topk", "{}.json".format(query_str)), "r") as f:
            vocal_finegrained_cache_stats = json.load(f)
        vocal_finegrained_cache_runtime = vocal_finegrained_cache_stats["runtime"]
        vocal_finegrained_cache_f1 = vocal_finegrained_cache_stats["score"]
        vocal_finegrained_cache_x = list(range(12, 21)) + list(range(25, 51, 5))
        vocal_finegrained_cache_f1 = np.array(vocal_finegrained_cache_f1)
        vocal_finegrained_cache_f1_25 = np.percentile(vocal_finegrained_cache_f1, 25, axis=0)
        vocal_finegrained_cache_f1_50 = np.percentile(vocal_finegrained_cache_f1, 50, axis=0)
        vocal_finegrained_cache_f1_75 = np.percentile(vocal_finegrained_cache_f1, 75, axis=0)
        vocal_finegrained_cache_runtime = np.array(vocal_finegrained_cache_runtime)
        vocal_finegrained_cache_runtime_25 = np.percentile(vocal_finegrained_cache_runtime, 25, axis=0)
        vocal_finegrained_cache_runtime_50 = np.percentile(vocal_finegrained_cache_runtime, 50, axis=0)
        vocal_finegrained_cache_runtime_75 = np.percentile(vocal_finegrained_cache_runtime, 75, axis=0)
        axs[0].plot(vocal_finegrained_cache_x, vocal_finegrained_cache_f1_50, marker='^', markersize=1, label="vocal (finegrained; cache)", color='tab:brown')
        # axs[0].fill_between(vocal_finegrained_cache_x, vocal_finegrained_cache_f1_25, vocal_finegrained_cache_f1_75, facecolor='tab:brown', alpha=0.3)
        axs[1].plot(vocal_finegrained_cache_x, vocal_finegrained_cache_runtime_50, marker='^', markersize=1, label="vocal (finegrained; cache)", color='tab:brown')
        # axs[1].fill_between(vocal_finegrained_cache_x, vocal_finegrained_cache_runtime_25, vocal_finegrained_cache_runtime_75, facecolor='tab:brown', alpha=0.3)
    except:
        pass


    # VOCAL (sample one segment at a time; with cache; simplest queries)
    try:
        with open(os.path.join(exp_dir, "stats", "vocal_postgres-topk", "simplest_queries", "{}.json".format(query_str)), "r") as f:
            vocal_finegrained_cache_stats = json.load(f)
        vocal_finegrained_cache_runtime = vocal_finegrained_cache_stats["runtime"]
        vocal_finegrained_cache_f1 = vocal_finegrained_cache_stats["score"]
        vocal_finegrained_cache_x = list(range(12, 21)) + list(range(25, 51, 5))
        vocal_finegrained_cache_f1 = np.array(vocal_finegrained_cache_f1)
        vocal_finegrained_cache_f1_25 = np.percentile(vocal_finegrained_cache_f1, 25, axis=0)
        vocal_finegrained_cache_f1_50 = np.percentile(vocal_finegrained_cache_f1, 50, axis=0)
        vocal_finegrained_cache_f1_75 = np.percentile(vocal_finegrained_cache_f1, 75, axis=0)
        vocal_finegrained_cache_runtime = np.array(vocal_finegrained_cache_runtime)
        vocal_finegrained_cache_runtime_25 = np.percentile(vocal_finegrained_cache_runtime, 25, axis=0)
        vocal_finegrained_cache_runtime_50 = np.percentile(vocal_finegrained_cache_runtime, 50, axis=0)
        vocal_finegrained_cache_runtime_75 = np.percentile(vocal_finegrained_cache_runtime, 75, axis=0)
        axs[0].plot(vocal_finegrained_cache_x, vocal_finegrained_cache_f1_50, linestyle='--', marker='^', markersize=1, label="vocal (finegrained; cache; simplest)", color='tab:olive')
        # axs[0].fill_between(vocal_finegrained_cache_x, vocal_finegrained_cache_f1_25, vocal_finegrained_cache_f1_75, facecolor='tab:olive', alpha=0.3)
        axs[1].plot(vocal_finegrained_cache_x, vocal_finegrained_cache_runtime_50, linestyle='--', marker='^', markersize=1, label="vocal (finegrained; cache; simplest)", color='tab:olive')
        # axs[1].fill_between(vocal_finegrained_cache_x, vocal_finegrained_cache_runtime_25, vocal_finegrained_cache_runtime_75, facecolor='tab:olive', alpha=0.3)
    except:
        pass


    axs[0].set(xlabel="# examples", ylabel="Test F1 score")
    axs[0].legend(prop={"size":10}, loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=2)
    axs[0].set_ylim(bottom=0)
    axs[1].set(xlabel="# examples", ylabel="Runtime (s)")
    axs[1].legend(prop={"size":10}, loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=2)
    # axs[1].set_ylim(bottom=0)
    axs[1].set_ylim(bottom=1, top=4000)
    axs[1].set_yscale('log')

    plt.suptitle(query_str)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("figures/main/{}_median.png".format(query_str), bbox_inches='tight')


def plot_query_median_simplest_queries(query_str):
    exp_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/without_duration-sampling_rate_4"

    fig, axs = plt.subplots(1, 2, figsize = (12, 3))

    # Quivr with kleene; simplest queries
    try:
        quivr_with_kleene_f1 = []
        quivr_with_kleene_runtime = []
        for run in range(20):
            with open(os.path.join(exp_dir, "stats", "quivr_original", "simplest_queries", "{}-{}.json".format(query_str, run)), "r") as f:
                quivr_with_kleene_stats = json.load(f)
            quivr_with_kleene_runtime_per_run = quivr_with_kleene_stats["runtime"]
            quivr_with_kleene_f1_per_run = quivr_with_kleene_stats["score"]
            quivr_with_kleene_f1.append(quivr_with_kleene_f1_per_run)
            quivr_with_kleene_runtime.append(quivr_with_kleene_runtime_per_run)

        quivr_with_kleene_x = list(range(12, 51))
        quivr_with_kleene_f1 = np.array(quivr_with_kleene_f1)
        quivr_with_kleene_f1_25 = np.percentile(quivr_with_kleene_f1, 25, axis=0)
        quivr_with_kleene_f1_50 = np.percentile(quivr_with_kleene_f1, 50, axis=0)
        quivr_with_kleene_f1_75 = np.percentile(quivr_with_kleene_f1, 75, axis=0)
        quivr_with_kleene_runtime = np.array(quivr_with_kleene_runtime)
        quivr_with_kleene_runtime_25 = np.percentile(quivr_with_kleene_runtime, 25, axis=0)
        quivr_with_kleene_runtime_50 = np.percentile(quivr_with_kleene_runtime, 50, axis=0)
        quivr_with_kleene_runtime_75 = np.percentile(quivr_with_kleene_runtime, 75, axis=0)
        axs[0].plot(quivr_with_kleene_x, quivr_with_kleene_f1_50, linestyle='--', marker='s', markersize=1, label="quivr (kleene; simplest)", color='tab:pink')
        axs[0].fill_between(quivr_with_kleene_x, quivr_with_kleene_f1_25, quivr_with_kleene_f1_75, facecolor='tab:pink', alpha=0.3)
        axs[1].plot(quivr_with_kleene_x, quivr_with_kleene_runtime_50, linestyle='--', marker='s', markersize=1, label="quivr (kleene; simplest)", color='tab:pink')
        axs[1].fill_between(quivr_with_kleene_x, quivr_with_kleene_runtime_25, quivr_with_kleene_runtime_75, facecolor='tab:pink', alpha=0.3)
    except:
        pass


    # Quivr without kleene; simplest queries
    try:
        quivr_with_kleene_f1 = []
        quivr_with_kleene_runtime = []
        for run in range(20):
            with open(os.path.join(exp_dir, "stats", "quivr_original_no_kleene", "simplest_queries", "{}-{}.json".format(query_str, run)), "r") as f:
                quivr_with_kleene_stats = json.load(f)
            quivr_with_kleene_runtime_per_run = quivr_with_kleene_stats["runtime"]
            quivr_with_kleene_f1_per_run = quivr_with_kleene_stats["score"]
            quivr_with_kleene_f1.append(quivr_with_kleene_f1_per_run)
            quivr_with_kleene_runtime.append(quivr_with_kleene_runtime_per_run)

        quivr_with_kleene_x = list(range(12, 51))
        quivr_with_kleene_f1 = np.array(quivr_with_kleene_f1)
        quivr_with_kleene_f1_25 = np.percentile(quivr_with_kleene_f1, 25, axis=0)
        quivr_with_kleene_f1_50 = np.percentile(quivr_with_kleene_f1, 50, axis=0)
        quivr_with_kleene_f1_75 = np.percentile(quivr_with_kleene_f1, 75, axis=0)
        quivr_with_kleene_runtime = np.array(quivr_with_kleene_runtime)
        quivr_with_kleene_runtime_25 = np.percentile(quivr_with_kleene_runtime, 25, axis=0)
        quivr_with_kleene_runtime_50 = np.percentile(quivr_with_kleene_runtime, 50, axis=0)
        quivr_with_kleene_runtime_75 = np.percentile(quivr_with_kleene_runtime, 75, axis=0)
        axs[0].plot(quivr_with_kleene_x, quivr_with_kleene_f1_50, linestyle='--', marker='s', markersize=1, label="quivr (no kleene; simplest)", color='tab:cyan')
        axs[0].fill_between(quivr_with_kleene_x, quivr_with_kleene_f1_25, quivr_with_kleene_f1_75, facecolor='tab:cyan', alpha=0.3)
        axs[1].plot(quivr_with_kleene_x, quivr_with_kleene_runtime_50, linestyle='--', marker='s', markersize=1, label="quivr (no kleene; simplest)", color='tab:cyan')
        axs[1].fill_between(quivr_with_kleene_x, quivr_with_kleene_runtime_25, quivr_with_kleene_runtime_75, facecolor='tab:cyan', alpha=0.3)
    except:
        pass


    # VOCAL (sample one segment at a time; with cache; simplest queries)
    try:
        with open(os.path.join(exp_dir, "stats", "vocal_postgres-topk", "simplest_queries", "{}.json".format(query_str)), "r") as f:
            vocal_finegrained_cache_stats = json.load(f)
        vocal_finegrained_cache_runtime = vocal_finegrained_cache_stats["runtime"]
        vocal_finegrained_cache_f1 = vocal_finegrained_cache_stats["score"]
        vocal_finegrained_cache_x = list(range(12, 21)) + list(range(25, 51, 5))
        vocal_finegrained_cache_f1 = np.array(vocal_finegrained_cache_f1)
        vocal_finegrained_cache_f1_25 = np.percentile(vocal_finegrained_cache_f1, 25, axis=0)
        vocal_finegrained_cache_f1_50 = np.percentile(vocal_finegrained_cache_f1, 50, axis=0)
        vocal_finegrained_cache_f1_75 = np.percentile(vocal_finegrained_cache_f1, 75, axis=0)
        vocal_finegrained_cache_runtime = np.array(vocal_finegrained_cache_runtime)
        vocal_finegrained_cache_runtime_25 = np.percentile(vocal_finegrained_cache_runtime, 25, axis=0)
        vocal_finegrained_cache_runtime_50 = np.percentile(vocal_finegrained_cache_runtime, 50, axis=0)
        vocal_finegrained_cache_runtime_75 = np.percentile(vocal_finegrained_cache_runtime, 75, axis=0)
        axs[0].plot(vocal_finegrained_cache_x, vocal_finegrained_cache_f1_50, linestyle='--', marker='^', markersize=1, label="vocal (finegrained; cache; simplest)", color='tab:olive')
        axs[0].fill_between(vocal_finegrained_cache_x, vocal_finegrained_cache_f1_25, vocal_finegrained_cache_f1_75, facecolor='tab:olive', alpha=0.3)
        axs[1].plot(vocal_finegrained_cache_x, vocal_finegrained_cache_runtime_50, linestyle='--', marker='^', markersize=1, label="vocal (finegrained; cache; simplest)", color='tab:olive')
        axs[1].fill_between(vocal_finegrained_cache_x, vocal_finegrained_cache_runtime_25, vocal_finegrained_cache_runtime_75, facecolor='tab:olive', alpha=0.3)
    except:
        pass


    axs[0].set(xlabel="# examples", ylabel="Test F1 score")
    axs[0].legend(prop={"size":10}, loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=2)
    # axs[0].set_ylim(bottom=0)
    axs[1].set(xlabel="# examples", ylabel="Runtime (s)")
    axs[1].legend(prop={"size":10}, loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=2)
    # axs[1].set_ylim(bottom=0)
    axs[1].set_ylim(bottom=1, top=4000)
    axs[1].set_yscale('log')

    plt.suptitle(query_str)
    plt.subplots_adjust(bottom=0.15)
    dir_name = "simplest_queries"
    if not os.path.exists(os.path.join("figures", dir_name)):
        os.makedirs(os.path.join("figures", dir_name))
    plt.savefig(os.path.join("figures", dir_name, "{}_median_simplest_queries.png".format(query_str)), bbox_inches='tight')


def plot_query_median_zoomed_in(query_str):
    exp_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/without_duration-sampling_rate_4"

    fig, axs = plt.subplots(1, 2, figsize = (12, 3))

    # Quivr without kleene
    try:
        quivr_f1 = []
        quivr_runtime = []
        for run in range(20):
            with open(os.path.join(exp_dir, "stats", "quivr_original_no_kleene", "{}-{}.json".format(query_str, run)), "r") as f:
                quivr_stats = json.load(f)
            quivr_runtime_per_run = quivr_stats["runtime"]
            quivr_f1_per_run = quivr_stats["score"]
            quivr_f1.append(quivr_f1_per_run)
            quivr_runtime.append(quivr_runtime_per_run)

        quivr_x = list(range(12, 51))
        quivr_f1 = np.array(quivr_f1)
        quivr_f1_25 = np.percentile(quivr_f1, 25, axis=0)
        quivr_f1_50 = np.percentile(quivr_f1, 50, axis=0)
        quivr_f1_75 = np.percentile(quivr_f1, 75, axis=0)
        quivr_runtime = np.array(quivr_runtime)
        quivr_runtime_25 = np.percentile(quivr_runtime, 25, axis=0)
        quivr_runtime_50 = np.percentile(quivr_runtime, 50, axis=0)
        quivr_runtime_75 = np.percentile(quivr_runtime, 75, axis=0)
        axs[0].plot(quivr_x, quivr_f1_50, marker='s', markersize=1, label="quivr (no kleene)", color='tab:blue')
        axs[0].fill_between(quivr_x, quivr_f1_25, quivr_f1_75, facecolor='tab:blue', alpha=0.3)
        axs[1].plot(quivr_x, quivr_runtime_50, marker='s', markersize=1, label="quivr (no kleene)", color='tab:blue')
        axs[1].fill_between(quivr_x, quivr_runtime_25, quivr_runtime_75, facecolor='tab:blue', alpha=0.3)
    except:
        pass

    # Quivr with kleene
    try:
        quivr_with_kleene_f1 = []
        quivr_with_kleene_runtime = []
        for run in range(20):
            with open(os.path.join(exp_dir, "stats", "quivr_original", "{}-{}.json".format(query_str, run)), "r") as f:
                quivr_with_kleene_stats = json.load(f)
            quivr_with_kleene_runtime_per_run = quivr_with_kleene_stats["runtime"]
            quivr_with_kleene_f1_per_run = quivr_with_kleene_stats["score"]
            quivr_with_kleene_f1.append(quivr_with_kleene_f1_per_run)
            quivr_with_kleene_runtime.append(quivr_with_kleene_runtime_per_run)

        quivr_with_kleene_x = list(range(12, 51))
        quivr_with_kleene_f1 = np.array(quivr_with_kleene_f1)
        quivr_with_kleene_f1_25 = np.percentile(quivr_with_kleene_f1, 25, axis=0)
        quivr_with_kleene_f1_50 = np.percentile(quivr_with_kleene_f1, 50, axis=0)
        quivr_with_kleene_f1_75 = np.percentile(quivr_with_kleene_f1, 75, axis=0)
        quivr_with_kleene_runtime = np.array(quivr_with_kleene_runtime)
        quivr_with_kleene_runtime_25 = np.percentile(quivr_with_kleene_runtime, 25, axis=0)
        quivr_with_kleene_runtime_50 = np.percentile(quivr_with_kleene_runtime, 50, axis=0)
        quivr_with_kleene_runtime_75 = np.percentile(quivr_with_kleene_runtime, 75, axis=0)
        axs[0].plot(quivr_with_kleene_x, quivr_with_kleene_f1_50, marker='s', markersize=1, label="quivr (kleene)", color='tab:red')
        axs[0].fill_between(quivr_with_kleene_x, quivr_with_kleene_f1_25, quivr_with_kleene_f1_75, facecolor='tab:red', alpha=0.3)
        axs[1].plot(quivr_with_kleene_x, quivr_with_kleene_runtime_50, marker='s', markersize=1, label="quivr (kleene)", color='tab:red')
        axs[1].fill_between(quivr_with_kleene_x, quivr_with_kleene_runtime_25, quivr_with_kleene_runtime_75, facecolor='tab:red', alpha=0.3)
    except:
        pass

    # Quivr with kleene; simplest queries
    try:
        quivr_with_kleene_f1 = []
        quivr_with_kleene_runtime = []
        for run in range(20):
            with open(os.path.join(exp_dir, "stats", "quivr_original", "simplest_queries", "{}-{}.json".format(query_str, run)), "r") as f:
                quivr_with_kleene_stats = json.load(f)
            quivr_with_kleene_runtime_per_run = quivr_with_kleene_stats["runtime"]
            quivr_with_kleene_f1_per_run = quivr_with_kleene_stats["score"]
            quivr_with_kleene_f1.append(quivr_with_kleene_f1_per_run)
            quivr_with_kleene_runtime.append(quivr_with_kleene_runtime_per_run)

        quivr_with_kleene_x = list(range(12, 51))
        quivr_with_kleene_f1 = np.array(quivr_with_kleene_f1)
        quivr_with_kleene_f1_25 = np.percentile(quivr_with_kleene_f1, 25, axis=0)
        quivr_with_kleene_f1_50 = np.percentile(quivr_with_kleene_f1, 50, axis=0)
        quivr_with_kleene_f1_75 = np.percentile(quivr_with_kleene_f1, 75, axis=0)
        quivr_with_kleene_runtime = np.array(quivr_with_kleene_runtime)
        quivr_with_kleene_runtime_25 = np.percentile(quivr_with_kleene_runtime, 25, axis=0)
        quivr_with_kleene_runtime_50 = np.percentile(quivr_with_kleene_runtime, 50, axis=0)
        quivr_with_kleene_runtime_75 = np.percentile(quivr_with_kleene_runtime, 75, axis=0)
        axs[0].plot(quivr_with_kleene_x, quivr_with_kleene_f1_50, linestyle='--', marker='s', markersize=1, label="quivr (kleene; simplest)", color='tab:pink')
        axs[0].fill_between(quivr_with_kleene_x, quivr_with_kleene_f1_25, quivr_with_kleene_f1_75, facecolor='tab:pink', alpha=0.3)
        axs[1].plot(quivr_with_kleene_x, quivr_with_kleene_runtime_50, linestyle='--', marker='s', markersize=1, label="quivr (kleene; simplest)", color='tab:pink')
        axs[1].fill_between(quivr_with_kleene_x, quivr_with_kleene_runtime_25, quivr_with_kleene_runtime_75, facecolor='tab:pink', alpha=0.3)
    except:
        pass


    # Quivr without kleene; simplest queries
    try:
        quivr_with_kleene_f1 = []
        quivr_with_kleene_runtime = []
        for run in range(20):
            with open(os.path.join(exp_dir, "stats", "quivr_original_no_kleene", "simplest_queries", "{}-{}.json".format(query_str, run)), "r") as f:
                quivr_with_kleene_stats = json.load(f)
            quivr_with_kleene_runtime_per_run = quivr_with_kleene_stats["runtime"]
            quivr_with_kleene_f1_per_run = quivr_with_kleene_stats["score"]
            quivr_with_kleene_f1.append(quivr_with_kleene_f1_per_run)
            quivr_with_kleene_runtime.append(quivr_with_kleene_runtime_per_run)

        quivr_with_kleene_x = list(range(12, 51))
        quivr_with_kleene_f1 = np.array(quivr_with_kleene_f1)
        quivr_with_kleene_f1_25 = np.percentile(quivr_with_kleene_f1, 25, axis=0)
        quivr_with_kleene_f1_50 = np.percentile(quivr_with_kleene_f1, 50, axis=0)
        quivr_with_kleene_f1_75 = np.percentile(quivr_with_kleene_f1, 75, axis=0)
        quivr_with_kleene_runtime = np.array(quivr_with_kleene_runtime)
        quivr_with_kleene_runtime_25 = np.percentile(quivr_with_kleene_runtime, 25, axis=0)
        quivr_with_kleene_runtime_50 = np.percentile(quivr_with_kleene_runtime, 50, axis=0)
        quivr_with_kleene_runtime_75 = np.percentile(quivr_with_kleene_runtime, 75, axis=0)
        axs[0].plot(quivr_with_kleene_x, quivr_with_kleene_f1_50, linestyle='--', marker='s', markersize=1, label="quivr (no kleene; simplest)", color='tab:cyan')
        axs[0].fill_between(quivr_with_kleene_x, quivr_with_kleene_f1_25, quivr_with_kleene_f1_75, facecolor='tab:cyan', alpha=0.3)
        axs[1].plot(quivr_with_kleene_x, quivr_with_kleene_runtime_50, linestyle='--', marker='s', markersize=1, label="quivr (no kleene; simplest)", color='tab:cyan')
        axs[1].fill_between(quivr_with_kleene_x, quivr_with_kleene_runtime_25, quivr_with_kleene_runtime_75, facecolor='tab:cyan', alpha=0.3)
    except:
        pass


    # # VOCAL
    # try:
    #     with open(os.path.join(exp_dir, "stats", "vocal_postgres-topk-test", "{}.json".format(query_str)), "r") as f:
    #         vocal_stats = json.load(f)
    #     vocal_runtime = vocal_stats["runtime"]
    #     vocal_f1 = vocal_stats["score"]
    #     vocal_x = list(range(12, 21)) + list(range(25, 51, 5))
    #     vocal_f1 = np.array(vocal_f1)
    #     vocal_f1_25 = np.percentile(vocal_f1, 25, axis=0)
    #     vocal_f1_50 = np.percentile(vocal_f1, 50, axis=0)
    #     vocal_f1_75 = np.percentile(vocal_f1, 75, axis=0)
    #     vocal_runtime = np.array(vocal_runtime)
    #     vocal_runtime_25 = np.percentile(vocal_runtime, 25, axis=0)
    #     vocal_runtime_50 = np.percentile(vocal_runtime, 50, axis=0)
    #     vocal_runtime_75 = np.percentile(vocal_runtime, 75, axis=0)
    #     axs[0].plot(vocal_x, vocal_f1_50, marker='^', markersize=1, label="vocal", color='tab:green')
    #     # axs[0].fill_between(vocal_x, vocal_f1_25, vocal_f1_75, facecolor='tab:green', alpha=0.3)
    #     axs[1].plot(vocal_x, vocal_runtime_50, marker='^', markersize=1, label="vocal", color='tab:green')
    #     # axs[1].fill_between(vocal_x, vocal_runtime_25, vocal_runtime_75, facecolor='tab:green', alpha=0.3)
    # except:
    #     pass

        # VOCAL (sample one segment at a time)
        with open(os.path.join(exp_dir, "stats", "vocal_postgres-topk-finegrained", "{}.json".format(query_str)), "r") as f:
            vocal_finegrained_stats = json.load(f)
        vocal_finegrained_runtime = vocal_finegrained_stats["runtime"]
        vocal_finegrained_f1 = vocal_finegrained_stats["score"]
        vocal_finegrained_x = list(range(12, 21)) + list(range(25, 51, 5))
        vocal_finegrained_f1 = np.array(vocal_finegrained_f1)
        vocal_finegrained_f1_25 = np.percentile(vocal_finegrained_f1, 25, axis=0)
        vocal_finegrained_f1_50 = np.percentile(vocal_finegrained_f1, 50, axis=0)
        vocal_finegrained_f1_75 = np.percentile(vocal_finegrained_f1, 75, axis=0)
        vocal_finegrained_runtime = np.array(vocal_finegrained_runtime)
        vocal_finegrained_runtime_25 = np.percentile(vocal_finegrained_runtime, 25, axis=0)
        vocal_finegrained_runtime_50 = np.percentile(vocal_finegrained_runtime, 50, axis=0)
        vocal_finegrained_runtime_75 = np.percentile(vocal_finegrained_runtime, 75, axis=0)
        axs[0].plot(vocal_finegrained_x, vocal_finegrained_f1_50, marker='^', markersize=1, label="vocal (finegrained)", color='tab:purple')
        axs[0].fill_between(vocal_finegrained_x, vocal_finegrained_f1_25, vocal_finegrained_f1_75, facecolor='tab:purple', alpha=0.3)
        axs[1].plot(vocal_finegrained_x, vocal_finegrained_runtime_50, marker='^', markersize=1, label="vocal (finegrained)", color='tab:purple')
        axs[1].fill_between(vocal_finegrained_x, vocal_finegrained_runtime_25, vocal_finegrained_runtime_75, facecolor='tab:purple', alpha=0.3)


    # VOCAL (sample one segment at a time; with cache)
    try:
        with open(os.path.join(exp_dir, "stats", "vocal_postgres-topk", "{}.json".format(query_str)), "r") as f:
            vocal_finegrained_cache_stats = json.load(f)
        vocal_finegrained_cache_runtime = vocal_finegrained_cache_stats["runtime"]
        vocal_finegrained_cache_f1 = vocal_finegrained_cache_stats["score"]
        vocal_finegrained_cache_x = list(range(12, 21)) + list(range(25, 51, 5))
        vocal_finegrained_cache_f1 = np.array(vocal_finegrained_cache_f1)
        vocal_finegrained_cache_f1_25 = np.percentile(vocal_finegrained_cache_f1, 25, axis=0)
        vocal_finegrained_cache_f1_50 = np.percentile(vocal_finegrained_cache_f1, 50, axis=0)
        vocal_finegrained_cache_f1_75 = np.percentile(vocal_finegrained_cache_f1, 75, axis=0)
        vocal_finegrained_cache_runtime = np.array(vocal_finegrained_cache_runtime)
        vocal_finegrained_cache_runtime_25 = np.percentile(vocal_finegrained_cache_runtime, 25, axis=0)
        vocal_finegrained_cache_runtime_50 = np.percentile(vocal_finegrained_cache_runtime, 50, axis=0)
        vocal_finegrained_cache_runtime_75 = np.percentile(vocal_finegrained_cache_runtime, 75, axis=0)
        axs[0].plot(vocal_finegrained_cache_x, vocal_finegrained_cache_f1_50, marker='^', markersize=1, label="vocal (finegrained; cache)", color='tab:brown')
        axs[0].fill_between(vocal_finegrained_cache_x, vocal_finegrained_cache_f1_25, vocal_finegrained_cache_f1_75, facecolor='tab:brown', alpha=0.3)
        axs[1].plot(vocal_finegrained_cache_x, vocal_finegrained_cache_runtime_50, marker='^', markersize=1, label="vocal (finegrained; cache)", color='tab:brown')
        axs[1].fill_between(vocal_finegrained_cache_x, vocal_finegrained_cache_runtime_25, vocal_finegrained_cache_runtime_75, facecolor='tab:brown', alpha=0.3)
    except:
        pass


    # VOCAL (sample one segment at a time; with cache; simplest queries)
    try:
        with open(os.path.join(exp_dir, "stats", "vocal_postgres-topk", "simplest_queries", "{}.json".format(query_str)), "r") as f:
            vocal_finegrained_cache_stats = json.load(f)
        vocal_finegrained_cache_runtime = vocal_finegrained_cache_stats["runtime"]
        vocal_finegrained_cache_f1 = vocal_finegrained_cache_stats["score"]
        vocal_finegrained_cache_x = list(range(12, 21)) + list(range(25, 51, 5))
        vocal_finegrained_cache_f1 = np.array(vocal_finegrained_cache_f1)
        vocal_finegrained_cache_f1_25 = np.percentile(vocal_finegrained_cache_f1, 25, axis=0)
        vocal_finegrained_cache_f1_50 = np.percentile(vocal_finegrained_cache_f1, 50, axis=0)
        vocal_finegrained_cache_f1_75 = np.percentile(vocal_finegrained_cache_f1, 75, axis=0)
        vocal_finegrained_cache_runtime = np.array(vocal_finegrained_cache_runtime)
        vocal_finegrained_cache_runtime_25 = np.percentile(vocal_finegrained_cache_runtime, 25, axis=0)
        vocal_finegrained_cache_runtime_50 = np.percentile(vocal_finegrained_cache_runtime, 50, axis=0)
        vocal_finegrained_cache_runtime_75 = np.percentile(vocal_finegrained_cache_runtime, 75, axis=0)
        axs[0].plot(vocal_finegrained_cache_x, vocal_finegrained_cache_f1_50, linestyle='--', marker='^', markersize=1, label="vocal (finegrained; cache; simplest)", color='tab:olive')
        axs[0].fill_between(vocal_finegrained_cache_x, vocal_finegrained_cache_f1_25, vocal_finegrained_cache_f1_75, facecolor='tab:olive', alpha=0.3)
        axs[1].plot(vocal_finegrained_cache_x, vocal_finegrained_cache_runtime_50, linestyle='--', marker='^', markersize=1, label="vocal (finegrained; cache; simplest)", color='tab:olive')
        axs[1].fill_between(vocal_finegrained_cache_x, vocal_finegrained_cache_runtime_25, vocal_finegrained_cache_runtime_75, facecolor='tab:olive', alpha=0.3)
    except:
        pass


    axs[0].set(xlabel="# examples", ylabel="Test F1 score")
    axs[0].legend(prop={"size":10}, loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=2)
    # axs[0].set_ylim(bottom=0)
    axs[1].set(xlabel="# examples", ylabel="Runtime (s)")
    axs[1].legend(prop={"size":10}, loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=2)
    # axs[1].set_ylim(bottom=0)
    axs[1].set_ylim(bottom=1, top=4000)
    axs[1].set_yscale('log')

    plt.suptitle(query_str)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("figures/zoomed_in/{}_median_zoomed_in.png".format(query_str), bbox_inches='tight')


if __name__ == "__main__":

    query_strs = [
        "Conjunction(Near_1(o0, o1), BottomQuadrant(o0))",
        "Conjunction(FrontOf(o0, o1), TopQuadrant(o0))",
        "Near_1(o0, o1); Far_3(o0, o1)",
        "Conjunction(Conjunction(Near_1(o0, o1), LeftQuadrant(o0)), Behind(o0, o1))",
        "Far_3(o0, o1); Near_1(o0, o1); Far_3(o0, o1)",
        "Conjunction(Far_3(o0, o1), BottomQuadrant(o0)); Near_1(o0, o1)",
        "Far_3(o0, o1); Conjunction(Near_1(o0, o1), Behind(o0, o1))",
        "Conjunction(Far_3(o0, o1), LeftQuadrant(o0)); Conjunction(Near_1(o0, o1), LeftQuadrant(o0))",
        "Far_3(o0, o1); Conjunction(Conjunction(Near_1(o0, o1), LeftQuadrant(o0)), Behind(o0, o1))"
        ]
    for query_str in query_strs:
        plot_query_median(query_str)
        # plot_query_median_zoomed_in(query_str)
        # plot_query_median_simplest_queries(query_str)
    # plot_query_median("Conjunction(Conjunction(Near_1(o0, o1), LeftQuadrant(o0)), Behind(o0, o1))")