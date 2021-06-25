import os
import random
import sys

import numpy as np
from sklearn.metrics import precision_score
import torch

sys.path.insert(0, "./tahoma/")
from VideoDB import VideoDB


torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)

def get_emptiest_gpu():
	# Command line to find memory usage of GPUs. Return the one with most mem available.
	output = os.popen('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free').read().strip()
	mem_avail = [int(x.split()[2]) for x in output.split('\n')]
	return mem_avail.index(max(mem_avail))

# Set emptiest GPU
torch.cuda.set_device(get_emptiest_gpu())
# torch.cuda.set_device(2)
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# my_classifier = torch.load("/z/analytics/VideoDB/resnet_night/checkpoints/epoch=20.model.pth.tar").to(device)
# my_classifier = torch.load("/z/analytics/VideoDB/resnet_spring/checkpoints/epoch=20.model.pth.tar").to(device)

# my_classifier = torch.load("/z/analytics/VideoDB/resnet_summer/checkpoints/epoch=40.model.pth.tar").to(device)

# my_classifier = torch.load("/z/analytics/VideoDB/resnet_fall/checkpoints/epoch=40.model.pth.tar").to(device)

# my_classifier = torch.load("/z/analytics/VideoDB/resnet_winter/checkpoints/epoch=20.model.pth.tar").to(device)

# my_classifier = torch.load("./user_model/ImageNet/checkpoints/icecream/epoch=40.model.pth.tar")
# my_classifier.to(device)

video_db = VideoDB(device)
video_db.query(user_classifier="faster_rcnn", classifier_name="car", mode="FAST1", topK=10000, accuracy=0.9, interval=1)

# video_db = VideoDB(device)
# video_db.query(user_classifier=my_classifier, classifier_name="is_spring_resnet_1111", mode="FAST1", topK=10000, accuracy=0.9, info="/z/analytics/VideoDB/resnet_spring/info_spring_resnet.json", interval=60)

# video_db.query(user_classifier=my_classifier, classifier_name="is_winter_resnet_0930_resize_video_more_samples_for_calculating_loading_time", mode="FAST1", topK=10000, accuracy=0.95, info="/z/analytics/VideoDB/resnet_winter/info_winter_resnet.json", interval=1)

#video_db.query(user_classifier=my_classifier, classifier_name="is_night_resnet_0813", mode="FAST1", topK=10000, accuracy=0.9, info="/z/analytics/VideoDB/resnet_night/info_night_resnet.json", interval=60)

# if not os.path.exists('is_winter_resnet_1024_change_init/result_naive.npy'):
# 	result_naive = video_db.query(user_classifier=my_classifier, classifier_name="is_winter_resnet_1024_change_init", mode="NAIVE", topK=100000, accuracy=0.90, info="/z/analytics/VideoDB/resnet_winter/info_winter_resnet.json", interval=1)
# 	np.save('is_winter_resnet_1024_change_init/result_naive', result_naive)

# else: 
# 	result_naive = np.load('is_winter_resnet_1024_change_init/result_naive.npy')
# result_fast = video_db.query(user_classifier=my_classifier, classifier_name="is_winter_resnet_1024_change_init", mode="FAST1", topK=100000, accuracy=0.95, info="/z/analytics/VideoDB/resnet_winter/info_winter_resnet.json", interval=1)
# if len(result_naive) < len(result_fast):
# 	result_fast = result_fast[:len(result_naive)]
# else:
# 	result_naive = result_naive[:len(result_fast)]
# fidelity = (result_fast == result_naive).sum() / len(result_naive)
# precision = precision_score(result_naive, result_fast)
# print('fidelity:', fidelity)
# print('precision:', precision)

# result_fast = video_db.query(user_classifier=my_classifier, classifier_name="is_night_resnet", mode="FAST1", topK=10000, accuracy=0.9, info="/z/analytics/VideoDB/resnet_night/info_night_resnet.json", interval=1)
# result_fast = video_db.query(user_classifier=my_classifier, classifier_name="is_spring_resnet", mode="FAST1", topK=5000, accuracy=0.93, info="/z/analytics/VideoDB/resnet_spring/info_spring_resnet.json", interval=120)
# result_fast = video_db.query(user_classifier=my_classifier, classifier_name="is_fall_resnet", mode="FAST1", topK=5000, accuracy=0.97, info="/z/analytics/VideoDB/resnet/info_fall_resnet.json", interval=120)
# video_db.query(user_classifier=my_classifier, classifier_name="is_fall", mode="NAIVE", topK=100000, accuracy=0.97, info="/z/analytics/VideoDB/binary_classifiers/info/info_fall.json", interval=1)