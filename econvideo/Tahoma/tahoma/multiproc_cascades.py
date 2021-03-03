import multiprocessing as mp
import numpy as np
import os
import pickle
import random
import shutil
import time

from sklearn.metrics import confusion_matrix, precision_score, recall_score
from cascades import *

def eval_cascades(cascades, ids, queue, model_results, ground_truth, infer_times, data_handling_time, certain, uncertain):
    num_items = len(ground_truth)
    return_val = []

    for i, cascade in enumerate(cascades):
        cascade_id = ids[i]
        num_levels = len(cascade)
        final_level = num_levels - 1
        remain_items = set(range(num_items))
        data_handling_times = []
        level_times = []
        level_items = []
        indexes = {}
        predictions = np.zeros(num_items) - 1
        ns = []

        for n, model_id in enumerate(cascade):
            level_times.append(len(remain_items) * infer_times[model_id])
            #data_handling_times.append(num_items * data_handling_time[model_id])
            data_handling_times.append(len(remain_items) * data_handling_time[model_id])
            ns.append(len(remain_items))

            if n == final_level:
                #level_items.append(remain_items)
                predictions[list(remain_items)] = model_results[model_id, list(remain_items)]
                # print('final_level remain_items:', len(remain_items))
                remain_items = set([])
            else:
                if n == 0:
                    if model_id not in indexes:
                        indexes[model_id] = list(certain[model_id].intersection(remain_items))
                    idx = indexes[model_id]
                else:
                    idx = list(certain[model_id].intersection(remain_items))

                predictions[idx] = model_results[model_id, idx]
                remain_items = remain_items - certain[model_id]
                         
        fps = num_items / (sum(level_times) + sum(data_handling_times))
        # fps = num_items / sum(level_times)
        # print('predictions:', predictions)
        acc = accuracy(predictions, ground_truth)
        p = np.zeros(ground_truth.shape)
        p[predictions >= 0.5] = 1.0
        precision = precision_score(ground_truth, p)
        # print('acc:', acc)
        return_val.append((cascade_id, cascade, (fps, acc, precision)))

    queue.put(return_val)

def compute_all_cascades(classifier_name, thresh, models, load_time, standardize_time, user_infer_time, result_directory):
    # NOTE: load_time is different from later constructed load_times    
    print("Starting", classifier_name, thresh)

    (models, ground_truth, results, extra_ground_truth, extra_results,
     infer_fps_vals, acc_vals,
     extra_acc_vals, _) = load_cascade_data(classifier_name, models, result_directory)

    # we'll use time instead of throughput for our evaluations below,
    # so let's switch it around here first to make things easier
    measured_infer_times = 1. / infer_fps_vals
    # measured_load_times = 1. / load_fps_vals
    infer_times = np.array(measured_infer_times)
    # load_times = np.array(measured_load_times)
    load_times = np.zeros(len(models))
    # resize_times = np.zeros(load_times.shape)

    infers = {}

    for i, m in enumerate(models):
        if 'name' in m and m['name'] == 'user':
            continue

        arch_name = get_arch_name(m) 

        if arch_name not in infers:
            infers[arch_name] = []

        infers[arch_name].append(measured_infer_times[i])

    for i, m in enumerate(models):
        if 'name' in m and m['name'] == 'user':
            # The model is user_classifier
            continue

        arch_name = get_arch_name(m)

        preprocessor = m['preprocessor'] if m['preprocessor'] == "ColorImage" or m['preprocessor'] == "BWImage" else "SingleChannelImage"
        input_shape = m['input_shape']
        key = 'x'.join([str(a) for a in input_shape])
        infer_times[i] = np.mean(infers[arch_name])
        load_times[i] = load_time[key] + standardize_time[key][preprocessor]

    # add on the average times for the ResNet model
    # these were computed by averaging over several dozen runs on ec2 box
    input_shape = models[-1]['input_shape']
    infer_times = np.append(infer_times, user_infer_time)
    print('infer_times', infer_times)
    key = 'x'.join([str(a) for a in input_shape])
    load_times = np.append(load_times, load_time[key] + standardize_time[key]['ColorImage'])
    print('load_times', load_times)
    # resize_times = np.append(resize_times, 0.0033)

    data_handling_time = load_times                         # Load

    threshold_gt = ground_truth.reshape(results[0].shape)

    brute_results = np.array(results)
    # brute_resnet = np.array(resnet[np.newaxis,...])
    threshold_set = np.vstack([brute_results, ground_truth[np.newaxis,...]]) # add the ResNet results as the last model in the list

    # the 'extra' data is an image set scraped from Google Images,
    # we'll use it as a test set to evaluate model accuracy and load times
    extra_brute_results = np.array(extra_results)
    # extra_brute_resnet = np.array(extra_resnet[np.newaxis,...])
    extra_res = np.vstack([extra_brute_results, extra_ground_truth[np.newaxis,...]])
    extra_gt = extra_ground_truth.reshape(extra_results[0].shape)

    test_set = extra_res
    test_gt = extra_gt

    time_results = {}
    acc_results = {}
    user_thresholds = [thresh, thresh]

    # total time to process an image by a given model
    #total_times = infer_times + data_handling_time
    num_models = threshold_set.shape[0]
    final_model_id = num_models - 1 # it's the last one

    item_ids = np.arange(len(test_gt))
    num_items = len(item_ids)
    all_accs = []

    ############################################################################   
    # now compute trust thresholds
    ############################################################################
    if True or 'certain_ids' not in vars() and 'certain_ids' not in globals():
        certain = []
        uncertain = []
        certain_ids = []
        uncertain_ids = []
        cert_ratios = []

        start = time.time()
        for i, r in enumerate(threshold_set):    
            # find the thresholds based on the results without the test set
            low, high = find_threshholds(r, threshold_gt, user_thresholds[0], user_thresholds[1])
            print('model #:', i, 'low:', low, 'high:', high)
            models[i][thresh] = [low, high]
            # use the test set to evaluate timing and accuracy
            # the model is 'certain' if its prediction value is below the low or above the high thresholds
            certain_ids.append(item_ids[np.logical_or(test_set[i] <= low, test_set[i] >= high).squeeze()])
            # print('certain_ids', certain_ids[-1])
            uncertain_ids.append(item_ids[np.logical_and(test_set[i] > low, test_set[i] < high).squeeze()])
            # print('uncertain_ids', uncertain_ids[-1])
            certain.append(set(certain_ids[-1]))
            # print('certain', certain[-1])
            uncertain.append(set(uncertain_ids[-1]))
            # print('uncertain', uncertain)
            cert_ratios.append(float(len(certain_ids[-1]))/len(test_set[i]))
            # print('cert_ratios', cert_ratios[-1])
            all_accs.append(accuracy(test_set[i], test_gt))
            # print('all_accs', all_accs[-1])
        end = time.time()

        print("Calibrated trust thresholds: %.3f s" % (end - start))

    ############################################################################   
    # build list of cascades
    ############################################################################
    start = time.time()
    num_models = threshold_set.shape[0]
    oracle_id = num_models - 1

    cascade_list = []

    for i in range(num_models):
        cascade_list.append([i])
        for j in range(num_models):
            if j == i:
                continue
            cascade_list.append([i,j]) 
            cascade_list.append([i,j,oracle_id])

#             for k in range(num_models):
#                 if k == i or k == j or k == oracle_id:
#                     continue
#                 cascade_list.append([i,j,k])
#                 cascade_list.append([i,j,k, oracle_id])

    end = time.time()

    cascade_ids = np.arange(len(cascade_list))
    
    print("Enumerated %d cascades: %.3f s" % (len(cascade_list), (end-start)))

    num_models = test_set.shape[0]
    oracle_id = num_models - 1
    num_items = len(extra_ground_truth)

    cascade_times = []
    cascade_accs = []
    indexes = {}

    n_processes = 15
    def split_list(inlist, chunksize):
        return [inlist[x:x+chunksize] for x in range(0, len(inlist), chunksize)]
    cascade_jobs = split_list(cascade_list, len(cascade_list) // n_processes)
    cascade_job_ids = split_list(cascade_list, len(cascade_list) // n_processes)

    print("Split into %d jobs: %s" % (len(cascade_jobs), [len(x) for x in cascade_jobs]))

    t = time.time()
    q = mp.Queue()

    for i, job in enumerate(cascade_jobs):
        p = mp.Process(target=eval_cascades, args=(job, cascade_job_ids[i], q, test_set, test_gt, infer_times, data_handling_time, certain, uncertain))
        p.Daemon = True
        p.start()

    eval_results = []
    for job in cascade_jobs:
        eval_results += q.get()

    for job in cascade_jobs:
        p.join()
    print("Cascade time: %.3f" % (time.time()-t))
    print("Num. results:", len(eval_results))

    cascade_ids = np.array([x[0] for x in eval_results])
    cascade_vals = np.array([x[1] for x in eval_results])
    fps_vals = np.array([x[2][0] for x in eval_results])
    acc_vals = np.array([x[2][1] for x in eval_results])    
    precision_vals = np.array([x[2][2] for x in eval_results])    
    sort_idx = np.argsort(cascade_ids)
    
    # print('fps_vals[sort_idx]', fps_vals)
    # print('acc_vals[sort_idx]', acc_vals)

    # change the structure of the lists: instead of three lists,
    # return a list of tuples of structure
    # (cascade_structure, fps, accuracy, threshold)
    res = []
    for cascade_val, fps_val, acc_val, precision_val in zip(cascade_vals[sort_idx], fps_vals[sort_idx], acc_vals[sort_idx], precision_vals[sort_idx]):
        res.append((cascade_val, fps_val, acc_val, precision_val, thresh))
    return res
    
#     fname = "/z/analytics/VideoDB/cascades/cascade_defs_"+imagenet_name+"_"+str(thresh)+"_"+cost_name+".pkl"
    
#     if not os.path.exists("/z/analytics/VideoDB/cascades/"):
#         os.makedirs("/z/analytics/VideoDB/cascades/")

#     start = time.time()
#     with open(fname, 'wb') as f:
#         pickle.dump(cascade_vals[sort_idx], f, protocol=pickle.HIGHEST_PROTOCOL)
#     end = time.time()

#     print("Save cascades time: %.3f s" % (end - start))
    
#     np.save('/z/analytics/VideoDB/cascades/cascade_fps_'+imagenet_name+"_"+str(thresh)+'_'+cost_name, fps_vals[sort_idx])
#     np.save('/z/analytics/VideoDB/cascades/cascade_acc_'+imagenet_name+"_"+str(thresh)+'_'+cost_name, acc_vals[sort_idx])
    
# #    np.savez('/z/mrander/cascades_resize_' + imagenet_name , cascades=cascade_vals, fps=fps_vals, accuracy=acc_vals)
#     end = time.time()
#     print("Saved: %.3f" % ((end - start)))
#     print('-'*70)
    
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Compute cascades')
    # parser.add_argument('imagenet_name')
    # parser.add_argument('thresh')
    # parser.add_argument('cost_name')
    # args = parser.parse_args()
    names = 'is_fall'
    threshes = [0.91, 0.93, 0.95, 0.97, 0.99]
    for thresh in threshes:
        compute_all_cascades(names, float(thresh))
