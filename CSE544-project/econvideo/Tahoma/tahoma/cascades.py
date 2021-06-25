import matplotlib.pyplot as plt
import numpy as np
import shutil
import os
import random
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from utils import config

def accuracy(predictions, actuals):
    if len(predictions) == 0:
        return 0.0

    p = np.array(predictions).reshape(actuals.shape)
    p[p >= 0.5] = 1.0
    p[p < 0.5] = 0.0
    return np.sum(1.0 - np.abs(p - actuals)) / p.shape[0]

# def sum_fps(fps_vals):
#     return (1./sum(1./fps_vals))

def get_resnet_stats(resnet_data, ground_truth):
    p = np.array(resnet_data).reshape(ground_truth.shape)
    p[p >= 0.5] = 1.0
    p[p < 0.5] = 0.0

    precision = precision_score(ground_truth, p)
    recall = recall_score(ground_truth, p)
    precision_neg = precision_score(ground_truth, p, pos_label=0)
    recall_neg = recall_score(ground_truth, p, pos_label=0) # recall for negative labels
    acc = accuracy(p, ground_truth)

    return precision, recall,precision_neg, recall_neg,  acc

def find_threshholds(predictions, actuals, target_pos_precision, target_neg_precision, step = 0.05):
    n_steps = int(1./step)  # step is used for thresh (from 0 to 1)
    thresh = 0.0
    high_thresh = 2.0
    low_thresh = -1.0
    recall_pos_max = 0.0
    recall_neg_max = 0.0

    preds = np.array(predictions).reshape(actuals.shape) # need to reshape it

    for i in range(n_steps):
        thresh = thresh + step
        p = np.zeros(actuals.shape)
        p[preds >= thresh] = 1.0
        print('thresh', thresh)
        if thresh >= 0.5:
            precision_pos = precision_score(actuals, p)
            print('precision_pos', precision_pos)
            recall_pos = recall_score(actuals, p)
            print('recall_pos', recall_pos)
            if precision_pos >= target_pos_precision and recall_pos > recall_pos_max:
                high_thresh = thresh
                recall_pos_max = recall_pos
        else:
            precision_neg = precision_score(actuals, p, pos_label=0)
            print('precision_neg', precision_neg)
            recall_neg = recall_score(actuals, p, pos_label=0)
            print('recall_neg', recall_neg)
            if precision_neg >= target_neg_precision and recall_neg > recall_neg_max:
                low_thresh = thresh
                recall_neg_max = recall_neg

    return low_thresh, high_thresh


# NOTE: the method below allows for cascade like : ResNet -> simple -> ResNet
# where we trust the positive classes in ResNet, pass all negatives to the simple classifer
# and then pass the uncertain ones back to ResNet.

def compute_casades(user_thresholds, load_times, infer_times, threshold_set, test_set, threshold_gt, test_gt, models):
    # total time to process an image by a given model (for each image)
    total_times = infer_times + load_times
    num_models = threshold_set.shape[0] # should be 360 + 1
    final_model_id = num_models - 1 # it's the last one, which is ResNet

    certain = []
    uncertain = []

    item_ids = np.arange(len(test_gt))
    num_items = len(item_ids) # number of images
    for i, r in enumerate(threshold_set):
        # find the thresholds based on the results without the test set
        low, high = find_threshholds(r, threshold_gt, user_thresholds[0], user_thresholds[1])

        # use the test set to evaluate timing and accuracy
        # the model is 'certain' if its prediction value is below the low or above the high thresholds
        # NOTE: certain[i] will give you a set of image indecies that is certain given model i
        certain.append(set(item_ids[np.logical_or(test_set[i] <= low, test_set[i] >= high).squeeze()]))
        uncertain.append(set(item_ids[np.logical_and(test_set[i] > low, test_set[i] < high).squeeze()]))

    time_matrix = np.zeros([num_models, num_models])
    acc_matrix = np.zeros([num_models, num_models])

    for i in range(num_models): # iterating the first model

        # first stage of the cascade requires loading and inferring all images
        stage1_cost = total_times[i] * num_items

        for j in range(num_models): # iterating the second model
            # overlap contains indices classified by 2nd classifier (1-index)
            overlap = uncertain[i].intersection(certain[j]) # classified by 1st (?)
            # different from set of indcies needed to be classified by the last classifier
            sent_to_final = uncertain[j] - certain[i] # set difference operation

            # if the 2nd model is ResNet, we'll stop here (3rd model is duplicated)
            if j == final_model_id:
                if i == final_model_id: # degrade from 2 cascades to 1
                    stage2_cost = 0.0 # don't double count ResNet
                else:
                    stage2_cost = (total_times[j]) * len(uncertain[i])
                final_cost = 0.0
            else:
                # if we've already loaded this data (same input_type and shape),
                # we don't incur the loading cost again.
                if (models[j]['preprocessor'] == models[i]['preprocessor'] and
                     models[j]['input_shape'] == models[i]['input_shape']):
                    data_load_time = 0.0
                else:
                    data_load_time = load_times[j]

                stage2_cost = (infer_times[j] + data_load_time) * len(uncertain[i])

                # if we've already loaded the resnet-style image, don't count resnet again
                # (if ResNet appears as 1st or 2nd model, we degrade the cascades by 1 level)
                if ((models[final_model_id]['preprocessor'] == models[i]['preprocessor'] and
                     models[final_model_id]['input_shape'] == models[i]['input_shape']) or
                    (models[final_model_id]['preprocessor'] == models[j]['preprocessor'] and
                     models[final_model_id]['input_shape'] == models[j]['input_shape'])):
                        final_cost = 0.0
                else:
                    final_cost = total_times[final_model_id] * (len(uncertain[i]) - len(overlap))

            cascade_cost = stage1_cost + stage2_cost + final_cost
            time_matrix[i,j] = cascade_cost

            # now compute the accuracy for everything
            # first find which items get sent to which classifier
            second_idx = list(uncertain[i])
            
            final_idx = list(sent_to_final)
            # QUESTION: what is the meaning of each dimension for test_set; seems like 2D for cascade_preds?
            
            cascade_preds = np.array(test_set)[i,:] # first classifier results
            # update indices that are classified after second classifier
            cascade_preds[second_idx] = np.array(test_set)[j, second_idx] # second_classifier
            # update indices that are classified after third classifier
            cascade_preds[final_idx] = np.array(test_set)[final_model_id, final_idx] # final classifier

            acc_matrix[i,j] = accuracy(cascade_preds, test_gt)

    return time_matrix, acc_matrix

def get_image_size_name(model):
    n_inputs = model['input_shape'][0] * model['input_shape'][1]
    if model['preprocessor'] == 'ColorImage':
        n_inputs *= 3
    # Return: C * H * W
    return str(n_inputs)

def get_arch_name(model):
    # Return: name for a model (a model is uniquely defined by layers, image size and color representation; dropout is unused; what is dense size?)
    return str(model['cnn_layer']) + '_' + str(model['dropout_rate']) + '_' + str(model['fc_layer']) + '_' + str(get_image_size_name(model))

def get_resize_time(model):
    resize_times = {'[30, 30]_color': 0.00155,
                '[30, 30]_bw': 0.00065,
                '[30, 30]_red': 0.00182,
                '[30, 30]_green': 0.00183,
                '[30, 30]_blue': 0.00184,
                '[60, 60]_color': 0.0017,
                '[60, 60]_bw': 0.0007,
                '[60, 60]_red': 0.00187,
                '[60, 60]_green': 0.00188,
                '[60, 60]_blue': 0.00187,
                '[120, 120]_color': 0.00221,
                '[120, 120]_bw': 0.00089,
                '[120, 120]_red': 0.00206,
                '[120, 120]_green': 0.00206,
                '[120, 120]_blue': 0.00206,
                '[224, 224]_color': 0.0033,
                '[224, 224]_bw': 0.00132,
                '[224, 224]_red': 0.00249,
                '[224, 224]_green': 0.00249,
                '[224, 224]_blue': 0.00249,
                # Values for 240 * 240 are unused
                '[240, 240]_color': 0.00351,
                '[240, 240]_bw': 0.00144,
                '[240, 240]_red': 0.0026,
                '[240, 240]_green': 0.00261,
                '[240, 240]_blue': 0.00261,}
    # Return: one of the time listed above 
    # return resize_times[str(model['input_shape']) + '_' + str(model['preprocessor'])]
    return 0

def load_cascade_data(prefix, models_numpy, result_directory):
    # models_numpy = np.load(os.path.join(result_directory, prefix+'_models.npy'))
    ground_truth = np.load(os.path.join(result_directory, prefix+'_actual.npy'), allow_pickle=True)
    results = np.load(os.path.join(result_directory, prefix+'_results.npy'), allow_pickle=True)
    cascade_ground_truth = np.load(os.path.join(result_directory, prefix+'_cascade_actual.npy'), allow_pickle=True)
    cascade_results = np.load(os.path.join(result_directory, prefix+'_cascade_results.npy'), allow_pickle=True)
    # resnet = np.load(result_directory + prefix+'_resnet.npy')
    # cascade_resnet = np.load(result_directory + prefix+'_cascade_resnet.npy')
    infer_fps_vals = np.load(os.path.join(result_directory, prefix+'_infer_fps.npy'), allow_pickle=True)
    # load_fps_vals = np.load(os.path.join(result_directory, prefix+'_load_fps.npy'))

    models = list(models_numpy)
    
    ground_truth = np.concatenate(ground_truth).ravel()
    results = [np.concatenate(result).ravel() for result in results]
    cascade_ground_truth = np.concatenate(cascade_ground_truth).ravel()
    cascade_results = [np.concatenate(result).ravel() for result in cascade_results]
    # Compute base accuracy results
    acc_vals = [accuracy(result, ground_truth) for result in results]
    cascade_acc_vals = [accuracy(result, cascade_ground_truth) for result in cascade_results]

    # here are load and image resize times for different input types
    avg_raw_img_load = 0.005 #seconds

    # models:
    return (models, ground_truth, results, cascade_ground_truth, cascade_results,
            infer_fps_vals, acc_vals, cascade_acc_vals, avg_raw_img_load)


# def compute_pareto_frontier(fps_vals, acc_vals):
#     # make a new np array from the tuples, so we can
#     # break the ties properly. (Might be a better way to do this).
#     pd = []
#     for i in range(len(fps_vals)):
#         pd.append((fps_vals[i], acc_vals[i]))

#     pareto_sort_data = np.array(pd,
#                            dtype=[('fps', fps_vals.dtype), ('acc', acc_vals.dtype)])

#     sort_idx = np.argsort(pareto_sort_data, order=('acc', 'fps'))[::-1]
#     fps_sort = fps_vals[sort_idx]
#     acc_sort = acc_vals[sort_idx]

#     dominant = []
#     highest = 0
#     for i in range(len(acc_sort)):
#         if fps_sort[i] > highest:
#             dominant.append(i)
#             highest = fps_sort[i]

#     return dominant, fps_sort, acc_sort, sort_idx

def compute_pareto_frontier(all_cascades): #(id, fps, accuracy, precision, threshold)
    # Sort the cascades in descending order, first by accuracy, then by fps
    all_cascades.sort(key=lambda tup: (-tup[2], -tup[1]))
    dominant = []
    highest = 0

    for i in range(len(all_cascades)):
        if all_cascades[i][1] > highest:
            dominant.append(all_cascades[i])
            highest = all_cascades[i][1]

        # Reset for knowledge distillation.
        # if len(all_cascades[i][0])==1 and all_cascades[i][0][0] == 18:
        #     user_accuracy_cascade = all_cascades[i]
        if len(all_cascades[i][0]) == 1 and all_cascades[i][0][0] == 360:
            user_accuracy_cascade = all_cascades[i]
    
    # Sort the cascades in descending order, first by precision, then by fps
    all_cascades.sort(key=lambda tup: (-tup[3], -tup[1]))    
    dominant_precision = []
    highest = 0

    for i in range(len(all_cascades)):
        if all_cascades[i][1] > highest:
            dominant_precision.append(all_cascades[i])
            highest = all_cascades[i][1]

        # Reset for knowledge distillation.
        # if len(all_cascades[i][0])==1 and all_cascades[i][0][0]==18:
        #     user_precision_cascade = all_cascades[i]
        if len(all_cascades[i][0]) == 1 and all_cascades[i][0][0] == 360:
            user_precision_cascade = all_cascades[i]

    return dominant, dominant_precision, user_accuracy_cascade, user_precision_cascade