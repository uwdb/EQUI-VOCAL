"""
This is a copy of topk_queries.py and is used for searching disjunctive queries.
"""

from calendar import c
import json
import itertools
from ntpath import join
from turtle import left
import joblib
from filter import construct_spatial_feature_spatial_relationship
from itertools import groupby, count
from multiprocessing import Pool
import multiprocessing as mp
import random
import argparse
from functools import partial
from matplotlib.pyplot import subplots_adjust
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import f1_score
from utils import tools
from time import time
import os
import math
from scipy import stats

def model_picker():
    with open("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/clevrer_query_exploration/object_collision_perfect-1.05-1.1/true_labels-pair_level-1000.json", 'r') as f:
        true_labels_pair_level = json.load(f)
    true_labels_pair_level = np.array(true_labels_pair_level)

    with open("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/clevrer_query_exploration/object_collision_perfect-1.05-1.1/prediction-pair_level-1000.json", 'r') as f:
        prediction_matrix_pair_level = json.load(f)
    prediction_matrix_pair_level = np.array(prediction_matrix_pair_level)

    n_instances = len(true_labels_pair_level)
    budget = 200
    print("n_instances: {}".format(n_instances))
    queries_str = ["0_1", "0_2", "0_3", "0_4", "0_1;1_1", "0_2;1_2", "0_3;1_3", "0_4;1_4", "1_1;0_1", "1_2;0_2", "1_3;0_3", "1_4;0_4", "1_1;0_1;1_1", "1_1;0_2;1_1", "1_1;0_3;1_1"]
    k = len(queries_str)

    # Shuffle the streaming data; Edit the input data accordingly with the indices of streaming data
    segment_ids = list(range(n_instances))
    random.shuffle(segment_ids)
    true_labels_pair_level = true_labels_pair_level[segment_ids]
    prediction_matrix_pair_level = prediction_matrix_pair_level[segment_ids]

    # Initialize
    loss_t = np.zeros(k)
    z_t_log = np.zeros(n_instances, dtype=int) # z_t = Q_t
    z_t_budget = np.zeros(n_instances, dtype=int) # binary query decision (bounded by budget)
    posterior_t_log = np.zeros((n_instances, k)) # posterior log
    mp_oracle = np.zeros(n_instances)
    hidden_loss_log = np.zeros(n_instances, dtype=int)
    It_log = np.zeros(n_instances, dtype=int)
    recommended_model_log = np.full((budget, k), np.nan, dtype=int)
    f1_score_log = np.full((budget, k), np.nan, dtype=float)
    class_weight_log = np.full(budget, np.nan, dtype=float)
    # f1_gap_log = np.zeros(n_instances)
    posterior_t = np.ones(k)/k # weights
    n_pos, n_neg = 0, 0
    n_observed = 0

    for t in range(n_instances):
        eta = np.sqrt(np.log(k)/(2*(t+1)))

        posterior_t = np.exp(-eta * (loss_t-np.min(loss_t)))
        # Note that above equation is equivalent to np.exp(-eta * loss_t).
        # `-np.min(loss_t)` is applied only to avoid entries being near zero for large eta*loss_t values before the normalization
        posterior_t  /= np.sum(posterior_t)  # normalize

         # Log posterior_t
        posterior_t_log[t, :] = posterior_t

        # Compute v(p_t, w_t)
        u_t = _compute_u_t(posterior_t, prediction_matrix_pair_level[t, :], tuning_par=10)
        # Sanity checks for sampling probability
        if u_t > 1:
            u_t = 1
        if np.logical_and(u_t>=0, u_t<=1):
            u_t = u_t
        else:
            u_t = 0

        # Is x_t in the region of disagreement? yes if dis_t>1, no otherwise
        dist_t = len(np.unique(prediction_matrix_pair_level[t, :]))

        # If u_t is in the region of agreement, don't query anything (all models give the same predication)
        if dist_t == 1:
            u_t = 0
            z_t = 0
            z_t_log[t] = z_t
        else:
            #Else, make a random query decision
            if u_t>0:
                u_t = np.maximum(u_t, eta)
            if u_t>1:
                u_t=1
            z_t = np.random.binomial(size=1, n=1, p=u_t)
            z_t_log[t] = z_t

        if z_t == 1:
            # loss_t += (np.array((prediction_matrix_pair_level[t, :] != true_labels_pair_level[t]) * 1) / u_t)
            if true_labels_pair_level[t]:
                loss_t += (np.array((prediction_matrix_pair_level[t, :] != 1) * 1) * 5 / u_t)
                # loss_t += (np.array((prediction_matrix_pair_level[t, :] != 1) * 1) / u_t * (n_pos + n_neg) / (2 * n_pos))
                # n_pos += 1
            else:
                loss_t += (np.array((prediction_matrix_pair_level[t, :] != 0) * 1) / u_t)
                # loss_t += (np.array((prediction_matrix_pair_level[t, :] != 0) * 1) / u_t * (n_pos + n_neg) / (2 * n_neg))
                # n_neg += 1
            # loss_t = loss_t.reshape(k, 1)
            # loss_t = np.squeeze(np.asarray(loss_t))
            # Get recommended model
            recommended_model = np.argsort(posterior_t)[::-1]
            # print("Round {}: n_observed: {} Recommended model: {}".format(t, n_observed + 1, recommended_model))
            if n_observed < budget:
                recommended_model_log[n_observed] = recommended_model
                Y_true = true_labels_pair_level[np.where(z_t_log==1)[0]]
                Y_pred = prediction_matrix_pair_level[np.where(z_t_log==1)[0]]
                for j in range(k):
                    f1_score_log[n_observed, j] = f1_score(Y_true, Y_pred[:, j], zero_division=0)
                if true_labels_pair_level[t]:
                    n_pos += 1
                else:
                    n_neg += 1
                class_weight_log[n_observed] = n_pos / (n_pos + n_neg)
                n_observed += 1
        m_star = np.random.choice(list(range(k)), p=posterior_t) # I_t
        It_log[t] = m_star
        # Incur hidden loss; regret = hidden loss - loss by best model
        hidden_loss_log[t] = (prediction_matrix_pair_level[t, m_star] != true_labels_pair_level[t]) * 1
        # print(z_t)
        # print(loss_t)
        # f1_gap_log[t] = f1_score(true_labels_pair_level[:(t+1)], prediction_matrix_pair_level[:(t+1), 8], zero_division=0) - f1_score(true_labels_pair_level[:(t+1)], prediction_matrix_pair_level[:(t+1), np.argmax(posterior_t)], zero_division=0)
        # Terminate if it exceeds the budget
        if np.sum(z_t_log) < budget:
            z_t_budget[t] = z_t_log[t]
        # else:
        #     break

    # Labelling decisions as 0's and 1's
    labelled_instances = z_t_log
    ct_log = np.ones(n_instances, dtype=int)

    # if np.sum(z_t_log) < budget:
    #     f1_score_log[n_observed:, :] = f1_score_log[(n_observed-1), :]

    return (labelled_instances, ct_log, z_t_budget, hidden_loss_log, posterior_t_log, recommended_model_log, f1_score_log, class_weight_log)

def _compute_u_t(posterior_t, predictions_c, tuning_par=1):

    # Initialize possible u_t's
    u_t_list = np.zeros(2)

    # Repeat for each class
    for c in [0, 1]:
        # Compute the loss of models if the label of the streamed data is "c"
        loss_c = np.array(predictions_c != c)*1
        #
        # Compute the respective u_t value (conditioned on class c)
        term1 = np.inner(posterior_t, loss_c)
        u_t_list[c] = term1*(1-term1)

    # Return the final u_t
    u_t = tuning_par * np.max(u_t_list)

    return u_t

def query_by_committee_online():
    tuning_par = 1
    with open("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/clevrer_query_exploration/object_collision_perfect-1.05-1.1/true_labels-pair_level-1000.json", 'r') as f:
        true_labels_pair_level = json.load(f)
    true_labels_pair_level = np.array(true_labels_pair_level)

    with open("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/clevrer_query_exploration/object_collision_perfect-1.05-1.1/prediction-pair_level-1000.json", 'r') as f:
        prediction_matrix_pair_level = json.load(f)
    prediction_matrix_pair_level = np.array(prediction_matrix_pair_level)

    n_instances = len(true_labels_pair_level)
    budget = 200
    print("n_instances: {}".format(n_instances))
    queries_str = ["0_1", "0_2", "0_3", "0_4", "0_1;1_1", "0_2;1_2", "0_3;1_3", "0_4;1_4", "1_1;0_1", "1_2;0_2", "1_3;0_3", "1_4;0_4", "1_1;0_1;1_1", "1_1;0_2;1_1", "1_1;0_3;1_1"]
    k = len(queries_str)

    # Shuffle the streaming data; Edit the input data accordingly with the indices of streaming data
    segment_ids = list(range(n_instances))
    random.shuffle(segment_ids)
    true_labels_pair_level = true_labels_pair_level[segment_ids]
    prediction_matrix_pair_level = prediction_matrix_pair_level[segment_ids]

    # Initialize
    loss_t = np.zeros(k)
    posterior_t = np.ones(k) / k # Weights
    posterior_t_log = np.zeros((n_instances, k)) # posterior log
    z_t_log = np.zeros(n_instances, dtype=int)
    z_t_budget = np.zeros(n_instances, dtype=int)
    recommended_model_log = np.full((budget, k), np.nan, dtype=int)
    f1_score_log = np.full((budget, k), np.nan, dtype=float)
    class_weight_log = np.full(budget, np.nan, dtype=float)
    n_pos, n_neg = 0, 0
    n_observed = 0
    # If the strategy is adaptive,
    for i in range(n_instances):
        eta = np.sqrt(np.log(k)/(2*(i+1)))

        posterior_t = np.exp(-eta * (loss_t-np.min(loss_t)))
        # Note that above equation is equivalent to np.exp(-eta * loss_t).
        # `-np.min(loss_t)` is applied only to avoid entries being near zero for large eta*loss_t values before the normalization
        posterior_t  /= np.sum(posterior_t)  # normalize

        posterior_t_log[i, :] = posterior_t

        # Measure the normalized entropy of the incoming data
        hist, bin_edges = np.histogram(prediction_matrix_pair_level[i, :], bins=2)
        prob_i = hist/np.sum(hist)
        entropy_i = stats.entropy(prob_i, base=2) * tuning_par

        # Check if the normalized entropy is greater than 1
        if entropy_i > 1:
            entropy_i = 1
        if entropy_i < 0:
            entropy_i = 0
        # Randomly decide whether to query z_i or not
        z_i = np.random.binomial(size=1, n=1, p=entropy_i)
        # Log the value
        z_t_log[i] = z_i
        if z_i == 1:
            if true_labels_pair_level[i]:
                loss_t += (np.array((prediction_matrix_pair_level[i, :] != 1) * 1) * 5)
                # loss_t += (np.array((prediction_matrix_pair_level[t, :] != 1) * 1) / u_t * (n_pos + n_neg) / (2 * n_pos))
                # n_pos += 1
            else:
                loss_t += (np.array((prediction_matrix_pair_level[i, :] != 0) * 1))
            if n_observed < budget:
                Y_true = true_labels_pair_level[np.where(z_t_log==1)[0]]
                Y_pred = prediction_matrix_pair_level[np.where(z_t_log==1)[0]]
                for j in range(k):
                    f1_score_log[n_observed, j] = f1_score(Y_true, Y_pred[:, j], zero_division=0)
                recommended_model_log[n_observed] = np.argsort(f1_score_log[n_observed, :])[::-1]
                if true_labels_pair_level[i]:
                    n_pos += 1
                else:
                    n_neg += 1
                class_weight_log[n_observed] = n_pos / (n_pos + n_neg)
                n_observed += 1
        # Terminate if budget is exceeded
        if np.sum(z_t_log) <= budget:
            z_t_budget[i] = z_t_log[i]


    return (z_t_log, z_t_budget, recommended_model_log, f1_score_log, class_weight_log, posterior_t_log)


def ts():
    with open("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/clevrer_query_exploration/object_collision_perfect-1.05-1.1/true_labels-pair_level-1000.json", 'r') as f:
        true_labels_pair_level = json.load(f)
    true_labels_pair_level = np.array(true_labels_pair_level)

    with open("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/clevrer_query_exploration/object_collision_perfect-1.05-1.1/prediction-pair_level-1000.json", 'r') as f:
        prediction_matrix_pair_level = json.load(f)
    prediction_matrix_pair_level = np.array(prediction_matrix_pair_level)

    budget = len(true_labels_pair_level)
    print("budget: {}".format(budget))
    observed = []
    queries_str = ["0_1", "0_2", "0_3", "0_4", "0_1;1_1", "0_2;1_2", "0_3;1_3", "0_4;1_4", "1_1;0_1", "1_2;0_2", "1_3;0_3", "1_4;0_4", "1_1;0_1;1_1", "1_1;0_2;1_1", "1_1;0_3;1_1"]

    topk_queries = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    # randomly pick 3 queries
    random.shuffle(topk_queries)
    topk_queries = topk_queries[:3]
    # topk_queries = [3, 11, 14]
    n_actions = 2 ** len(topk_queries)

    n_observed_pos = 0
    n_observed_neg = 0
    while len(observed) < budget or n_observed_pos == np.sum(true_labels_pair_level):
        # 0. Construct clusters
        preds = prediction_matrix_pair_level[:, topk_queries]
        clusters = [[] for _ in range(n_actions)]
        # 0.1 Update beta_params
        beta_params = [[1, 1] for _ in range(n_actions)]
        for seg_id in observed:
            pred = preds[seg_id]
            cluster_id = 4 * pred[0] + 2 * pred[1] + pred[2]
            if true_labels_pair_level[seg_id] == 1:
                beta_params[cluster_id][0] += 1
            else:
                beta_params[cluster_id][1] += 1
        # 0.2 Update clusters
        for seg_id, pred in enumerate(preds):
            if seg_id not in observed:
                cluster_id = 4 * pred[0] + 2 * pred[1] + pred[2]
                clusters[cluster_id].append(seg_id)
        for cluster_id, cluster in enumerate(clusters):
            print("cluster {}: {}, beta_params ({})".format(cluster_id, len(cluster), beta_params[cluster_id]))

        # 1. Choose action
        # 1.1 Compute reward
        rewards = np.zeros(n_actions)
        for i in range(n_actions):
            rewards[i] = np.random.beta(beta_params[i][0], beta_params[i][1])
        # 1.2 Choose action
        ranked_actions = np.argsort(rewards)[::-1]
        picked_action = -1
        for ranked_action in ranked_actions:
            if len(clusters[ranked_action]):
                picked_action = ranked_action
                break

        # 1.3 Randomly pick a segment from the selected cluster
        picked_segment = random.choice(clusters[picked_action])
        assert(picked_segment not in observed)
        observed.append(picked_segment)
        # 2. Update state
        if true_labels_pair_level[picked_segment] == 1:
            # beta_params[picked_action][0] += 1
            n_observed_pos += 1
        else:
            # beta_params[picked_action][1] += 1
            n_observed_neg += 1
        print("observed: {}, # pos: {}, # neg: {}".format(len(observed), n_observed_pos, n_observed_neg))
        # 3. Recompute topk queries
        y_true = true_labels_pair_level[observed]
        scores = np.zeros(len(queries_str))
        for i in range(len(queries_str)):
            y_pred = prediction_matrix_pair_level[observed, i]
            # f1 score
            scores[i] = f1_score(y_true, y_pred)
        topk_queries = np.argsort(scores)[-3:]
        print("topk queries: {}".format(topk_queries))
        # check observed doesn't have duplicates
        assert(len(observed) == len(set(observed)))

def user_init_query():
    with open("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/clevrer_query_exploration/object_collision_perfect-1.05-1.1/true_labels-pair_level-1000.json", 'r') as f:
        true_labels_pair_level = json.load(f)
    true_labels_pair_level = np.array(true_labels_pair_level)

    with open("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/clevrer_query_exploration/object_collision_perfect-1.05-1.1/prediction-pair_level-1000.json", 'r') as f:
        prediction_matrix_pair_level = json.load(f)
    prediction_matrix_pair_level = np.array(prediction_matrix_pair_level)

    budget = len(true_labels_pair_level)
    print("budget: {}".format(budget))
    observed = []
    queries_str = ["0_1", "0_2", "0_3", "0_4", "0_1;1_1", "0_2;1_2", "0_3;1_3", "0_4;1_4", "1_1;0_1", "1_2;0_2", "1_3;0_3", "1_4;0_4", "1_1;0_1;1_1", "1_1;0_2;1_1", "1_1;0_3;1_1"]

    query_id = 8

    n_observed_pos = 0
    n_observed_neg = 0
    while len(observed) < budget or n_observed_pos == np.sum(true_labels_pair_level):
        unobserved = [i for i in range(len(true_labels_pair_level)) if i not in observed]
        preds = prediction_matrix_pair_level[:, query_id]
        # get indices where preds == 1
        indices = np.where(preds == 1)[0]
        # remove indices where index is in observed
        preds_true_indices = [i for i in indices if i not in observed]
        # randomly pick one from preds_true
        if len(preds_true_indices) > 0:
            picked_segment = random.choice(preds_true_indices)
        else:
            picked_segment = random.choice(unobserved)
        assert(picked_segment not in observed)
        observed.append(picked_segment)
        if true_labels_pair_level[picked_segment] == 1:
            n_observed_pos += 1
        else:
            n_observed_neg += 1
        print("observed: {}, # pos: {}, # neg: {}".format(len(observed), n_observed_pos, n_observed_neg))


def random_selection():
    with open("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/clevrer_query_exploration/object_collision_perfect-1.05-1.1/true_labels-pair_level-1000.json", 'r') as f:
        true_labels_pair_level = json.load(f)
    true_labels_pair_level = np.array(true_labels_pair_level)

    with open("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/clevrer_query_exploration/object_collision_perfect-1.05-1.1/prediction-pair_level-1000.json", 'r') as f:
        prediction_matrix_pair_level = json.load(f)
    prediction_matrix_pair_level = np.array(prediction_matrix_pair_level)

    budget = 200
    print("budget: {}".format(budget))
    observed = []
    queries_str = ["0_1", "0_2", "0_3", "0_4", "0_1;1_1", "0_2;1_2", "0_3;1_3", "0_4;1_4", "1_1;0_1", "1_2;0_2", "1_3;0_3", "1_4;0_4", "1_1;0_1;1_1", "1_1;0_2;1_1", "1_1;0_3;1_1"]

    segment_ids = list(range(len(true_labels_pair_level)))
    # randomly pick 3 queries
    random.shuffle(segment_ids)
    true_labels_pair_level = true_labels_pair_level[segment_ids]
    prediction_matrix_pair_level = prediction_matrix_pair_level[segment_ids]
    recommended_model_log = np.zeros(budget, dtype=int)
    for i in range(budget):
    # while len(observed) < budget or n_observed_pos == np.sum(true_labels_pair_level):
        Y_true = true_labels_pair_level[:(i+1)]
        Y_pred = prediction_matrix_pair_level[:(i+1)]
        f1_scores = []
        for j in range(len(queries_str)):
            f1_scores.append(f1_score(Y_true, Y_pred[:, j], zero_division=0))
        f1_scores = np.array(f1_scores)
        query_id = np.argmax(f1_scores)
        recommended_model_log[i] = query_id
    return recommended_model_log


if __name__ == '__main__':
    # method = ["weight_ut", "QBC_online"]
    method = "weight_ut"
    # user_init_query()
    # ts()
    posterior_t_log_realizations = []
    recommended_model_log_realizations = []
    f1_score_log_realizations = []
    z_t_budget_realizations = []
    class_weight_log_realizations = []
    for i in range(100):
        print(i)
        # recommended_model_log = random_selection()
        _, _, z_t_budget, _, posterior_t_log, recommended_model_log, f1_score_log, class_weight_log = model_picker()
        # z_t_log, z_t_budget, recommended_model_log, f1_score_log, class_weight_log, posterior_t_log = query_by_committee_online()
        z_t_budget_realizations.append(z_t_budget)
        posterior_t_log_realizations.append(posterior_t_log)
        recommended_model_log_realizations.append(recommended_model_log)
        f1_score_log_realizations.append(f1_score_log)
        class_weight_log_realizations.append(class_weight_log)
    z_t_budget_realizations = np.array(z_t_budget_realizations)
    posterior_t_log_realizations = np.array(posterior_t_log_realizations)
    recommended_model_log_realizations = np.array(recommended_model_log_realizations)
    f1_score_log_realizations = np.array(f1_score_log_realizations)
    class_weight_log_realizations = np.array(class_weight_log_realizations)
    # recommended_model_log_realizations = stats.mode(recommended_model_log_realizations)[0]
    # save to file
    with open("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/clevrer_query_exploration/object_collision_perfect-1.05-1.1/posterior_t_{}.json".format(method), 'w') as f:
        json.dump(posterior_t_log_realizations.tolist(), f)
    with open("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/clevrer_query_exploration/object_collision_perfect-1.05-1.1/recommended_model_{}.json".format(method), 'w') as f:
        json.dump(recommended_model_log_realizations.tolist(), f)
    with open("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/clevrer_query_exploration/object_collision_perfect-1.05-1.1/f1_score_{}.json".format(method), 'w') as f:
        json.dump(f1_score_log_realizations.tolist(), f)
    with open("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/clevrer_query_exploration/object_collision_perfect-1.05-1.1/z_t_budget_{}.json".format(method), 'w') as f:
        json.dump(z_t_budget_realizations.tolist(), f)
    with open("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/clevrer_query_exploration/object_collision_perfect-1.05-1.1/class_weight_{}.json".format(method), 'w') as f:
        json.dump(class_weight_log_realizations.tolist(), f)