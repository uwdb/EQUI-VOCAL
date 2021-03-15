import argparse
import logging
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys
from blazeit.data.generate_fnames import get_csv_fname, get_video_fname
from blazeit.specializers.specializers import BinarySpecializer
from blazeit.specializers import tiny_models

np.set_printoptions(threshold=sys.maxsize)

class PytorchBinary(BinarySpecializer):
    def train(self, write_out=False, **kwargs):
        super().train(**kwargs, epochs=[1, 0], lrs=[0.01, 0.001])
        self.write_out = write_out

    def eval(self, X):
        Y_prob = super().eval(X)
        Y_prob = torch.autograd.Variable(torch.from_numpy(Y_prob))
        # print(Y_prob)
        Y_prob = torch.nn.functional.softmax(Y_prob, dim=1).data.numpy()
        # print(Y_prob)
        return Y_prob

    def get_max_count(self):
        max_count = 0
        Y = self.getY()
        counts = defaultdict(int)
        for y in Y:
            counts[int(y)] += 1
        for num in counts:
            frac = float(counts[num]) / len(Y)
            if frac > 0.01:
                max_count = max(max_count, num)
        return max_count


    def get_pred(self, X, Y_true, lo_thresh):
        pred_neg = 0
        correct = 0
        incorrect = 0
        acc_correct = 0
        Y_prob = self.eval(X)
        Y_pred = np.argmax(Y_prob, axis=1)
        print(np.sum(Y_true))
        if len(Y_pred) < len(Y_true):
            Y_pred = np.pad(Y_pred, (0, len(Y_true) - len(Y_pred)), 'constant')
        if len(Y_true) < len(Y_pred):
            Y_true = np.pad(Y_true, (0, len(Y_pred) - len(Y_true)), 'constant')
        for i, prob in enumerate(Y_prob):
            if Y_pred[i] == Y_true[i]:
                acc_correct += 1
            if prob[1] <= lo_thresh:
                pred_neg += 1
                if (Y_true[i] == 0):
                    correct += 1
                else:
                    incorrect += 1
        print("fnr: ", incorrect / np.sum(Y_true))
        return len(Y_prob), pred_neg, correct, incorrect, acc_correct / len(Y_prob)


def get_nyquist(csv_fname):
    df = pd.read_csv(csv_fname)
    ind_counts = df['ind'].value_counts()
    ind_counts = ind_counts[ind_counts > 3]
    sample_freq = ind_counts.values[int(len(ind_counts) * 0.99)] - 1
    print('Selected', sample_freq, 'as sample frequency')
    return sample_freq

def bootstrap(Y, conf=0.95, nb_bootstrap=10000, nb_samples=50000):
    samples = [np.mean(np.random.choice(Y, nb_samples)) for i in range(nb_bootstrap)]
    low_ind = int(len(samples) * (1 - conf) / 2)
    hi_ind = int(len(samples) - len(samples) * (1 - conf) / 2)
    samples.sort()
    return samples[low_ind], samples[hi_ind]

# def get_thresh(Y_prob, Y_true, err=0.05):
#     # 1% target false positive and false negative rates
#     Y_max = np.max(Y_prob, axis=1)
#     Y_pred = np.argmax(Y_prob, axis=1)
#     tmp = [(ymax, ind) for ind, ymax in enumerate(Y_max)]
#     Y_ord = sorted(tmp)

#     pred_count = np.sum(Y_pred)
#     true_count = float(np.sum(Y_true))
#     pred_err = (pred_count - true_count) / true_count * 100.
#     print('Thresh pred, true, err: {} {} {}'.
#           format(pred_count, true_count, pred_err))
#     if len(Y_pred) < len(Y_true):
#         Y_pred = np.pad(Y_pred, (0, len(Y_true) - len(Y_pred)), 'constant')
#     if len(Y_true) < len(Y_pred):
#         Y_true = np.pad(Y_true, (0, len(Y_pred) - len(Y_true)), 'constant')
#     print('Thresh bootstrap: {}'.
#           format(bootstrap(Y_pred - Y_true)))
#     threshold = 0
#     ind = 0
#     # return 0. # FIXME FIXME
#     while abs(pred_count - true_count) / true_count > err:
#         threshold = Y_ord[ind][0]
#         i = Y_ord[ind][1]
#         if i >= len(Y_pred) or i >= len(Y_true):
#             ind += 1
#             continue
#         pred_count -= Y_pred[i]
#         pred_count += Y_true[i]
#         ind += 1
#     print(threshold, ind)
#     # return 0. # FIXME FIXME FIXME
#     return threshold

def train_and_test(DATA_PATH, TRAIN_DATE, THRESH_DATE, TEST_DATE,
                   base_name, objects,
                   tiny_name='trn10', normalize=True, nb_classes=-1,
                   load_video=False):

    nb_classes = 2

   # setup
    base_model = tiny_models.create_tiny_model(tiny_name, nb_classes, weights='imagenet')
    model_dump_fname = os.path.join(
            DATA_PATH, 'models', base_name, '%s-%s-%s.t7' % (base_name, TRAIN_DATE, tiny_name))


    load_times = []
    total_time = time.time()
    # train
    csv_fname = "/home/ubuntu/CSE544-project/data/bdd100k/bdd100k_" + TRAIN_DATE + ".csv"
    sample_freq = 3
    video_fname = "/home/ubuntu/CSE544-project/blazeit/data/resol-65/bdd100k/bdd100k-" + TRAIN_DATE + ".npy"
    spec = PytorchBinary(base_name, video_fname, csv_fname, objects, normalize,
                        base_model, model_dump_fname)

    start = time.time()
    ###### Computing stats ######
    # spec.load_data(selection='balanced', nb_train=50000)
    print("total count: ", len(spec.getY()))
    print("true count: ", np.sum(spec.getY()))
    
    csv_fname = "/home/ubuntu/CSE544-project/data/bdd100k/bdd100k_" + THRESH_DATE + ".csv"
    video_fname = "/home/ubuntu/CSE544-project/blazeit/data/resol-65/bdd100k/bdd100k-" + THRESH_DATE + ".npy"
    spec = PytorchBinary(base_name, video_fname, csv_fname, objects, normalize,
                        spec.model, model_dump_fname)
    print("total count: ", len(spec.getY()))
    print("true count: ", np.sum(spec.getY()))

    csv_fname = "/home/ubuntu/CSE544-project/data/bdd100k/bdd100k_" + TEST_DATE + ".csv"
    video_fname = "/home/ubuntu/CSE544-project/blazeit/data/resol-65/bdd100k/bdd100k-" + TEST_DATE + ".npy"
    spec = PytorchBinary(base_name, video_fname, csv_fname, objects, normalize,
                        spec.model, model_dump_fname)
    print("total count: ", len(spec.getY()))
    print("true count: ", np.sum(spec.getY()))
    exit(1)
    ###### End #########
    spec.load_data(selection='balanced', nb_train=50000)
    
    end = time.time()
    load_times.append(end - start)
    train_time = time.time()
    spec.train(silent=False)
    train_time = time.time() - train_time

    # thresh
    csv_fname = "/home/ubuntu/CSE544-project/data/bdd100k/bdd100k_" + THRESH_DATE + ".csv"
    video_fname = "/home/ubuntu/CSE544-project/blazeit/data/resol-65/bdd100k/bdd100k-" + THRESH_DATE + ".npy"
    spec = PytorchBinary(base_name, video_fname, csv_fname, objects, normalize,
                        spec.model, model_dump_fname)
    start = time.time()
    # Thresholding just requires an estimate of the answer
    X = spec.getX()
    Y_true = spec.getY()
    end = time.time()
    load_times.append(end - start)
    thresh_time = time.time()
    Y_prob = spec.eval(X)
    
    lo_thresh, nb_lo = spec.find_two_sided_thresh(Y_prob, Y_true, fnr=0.005, fpr=0.02)
    # thresh = get_thresh(Y_prob, Y_true)
    thresh_time = time.time() - thresh_time
    del X, Y_prob, Y_true
    print("(lo_thresh, nb_lo): ", lo_thresh, nb_lo)
    # test
    csv_fname = "/home/ubuntu/CSE544-project/data/bdd100k/bdd100k_" + TEST_DATE + ".csv"
    video_fname = "/home/ubuntu/CSE544-project/blazeit/data/resol-65/bdd100k/bdd100k-" + TEST_DATE + ".npy"
    spec = PytorchBinary(base_name, video_fname, csv_fname, objects, normalize,
                        spec.model, model_dump_fname,
                        write_out=True)
    start = time.time()
    X = spec.getX()
    Y_true = spec.getY()
    end = time.time()
    load_times.append(end - start)
    # Y_prob = spec.eval(X)
    eval_time = time.time()
    # Y_pred = spec.get_pred(X, Y_true, thresh)
    total_pred, pred_neg, correct, incorrect, acc = spec.get_pred(X, Y_true, lo_thresh)
    print(total_pred, pred_neg, correct, incorrect, acc)
    eval_time = time.time() - eval_time
    # pred_count = float(np.sum(Y_pred)) * sample_freq
    # true_count = float(np.sum(Y_true))
    # print(pred_count, true_count, 100 * (pred_count - true_count) / true_count)

    total_time = time.time() - total_time
    print('Train, thresh, eval time: %.2f, %.2f, %.2f' % (train_time, thresh_time, eval_time))
    print('Times:', total_time - sum(load_times), load_times, total_time)
    
    # return pred_count, sample_freq, Y_pred
    return [], sample_freq, []


# Stats
"""
object: truck, epochs: 10, fnr = 0.02
loss: 0.5465, acc: 70.3: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2487/2487 [00:11<00:00, 217.26it/s]
loss: 0.5156, acc: 73.5: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2487/2487 [00:11<00:00, 219.29it/s]
loss: 0.5077, acc: 74.6: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2487/2487 [00:11<00:00, 216.64it/s]
loss: 0.5042, acc: 74.7: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2487/2487 [00:10<00:00, 227.18it/s]
loss: 0.5006, acc: 75.0: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2487/2487 [00:10<00:00, 229.26it/s]
loss: 0.4971, acc: 75.1: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2487/2487 [00:11<00:00, 224.75it/s]
loss: 0.4964, acc: 75.5: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2487/2487 [00:11<00:00, 218.26it/s]
loss: 0.4933, acc: 75.5: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2487/2487 [00:10<00:00, 227.32it/s]
loss: 0.4970, acc: 75.3: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2487/2487 [00:11<00:00, 221.69it/s]
loss: 0.4963, acc: 75.6: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2487/2487 [00:11<00:00, 216.23it/s]
val loss: 0.4637, val acc: 76.1: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [01:59<00:00, 11.98s/it]

(lo_thresh, nb_lo):  0.02505381 369
test video: 56763 frames
skip: 4104 frames (unlikely to contain object of interest)
Train, thresh, eval time: 119.85, 1.67, 6.55
Times: 128.11562371253967 [2.509387493133545, 1.1744229793548584, 1.6423602104187012] 133.44179439544678


object: car, epochs: 10, fnr = 0.02
loss: 0.4201, acc: 86.9: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1763/1763 [00:08<00:00, 214.37it/s]
loss: 0.3463, acc: 89.4: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1763/1763 [00:07<00:00, 232.74it/s]
loss: 0.3240, acc: 90.1: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1763/1763 [00:08<00:00, 218.93it/s]
loss: 0.3081, acc: 90.1: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1763/1763 [00:07<00:00, 228.28it/s]
loss: 0.2991, acc: 90.4: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1763/1763 [00:08<00:00, 216.06it/s]
loss: 0.2943, acc: 90.5: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1763/1763 [00:07<00:00, 239.79it/s]
loss: 0.2921, acc: 90.4: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1763/1763 [00:07<00:00, 220.67it/s]
loss: 0.2860, acc: 90.6: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1763/1763 [00:07<00:00, 227.30it/s]
val loss: 0.3159, val acc: 87.4:  70%|██████████████████████████████████████████████████████████████████████████▏                               | 7/10 [01:08<00:29,  9.84s/it]

14547 290.94
(lo_thresh, nb_lo):  0.06795085 321
test video: 56763 frames
skip: 3990 frames (unlikely to contain object of interest)
Train, thresh, eval time: 68.86, 1.74, 6.65
Times: 77.29147124290466 [2.8018064498901367, 3.054844379425049, 4.45451807975769] 87.60264015197754


object: truck, epochs: 10, fnr = 0.01
(lo_thresh, nb_lo):  0.01453239 665
test video: 56763 frames
skip: 2286 frames (unlikely to contain object of interest)
"""