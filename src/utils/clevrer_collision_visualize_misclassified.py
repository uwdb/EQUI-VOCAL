import os
import sys
import cv2
import numpy as np
import json
import random

sampling_method = "least_confidence"
thresh = 2.0
output_dir = "/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/clevrer_far"

# if misclassified/least_confidence doesn't exist, create it
if not os.path.exists(os.path.join(output_dir, "misclassified", sampling_method)):
    os.makedirs(os.path.join(output_dir, "misclassified", sampling_method))

with open(os.path.join(output_dir, "{}-{}-original-random_forest-sampled_fn_fp.json".format(thresh, sampling_method)), "r") as f:
    sampled_fn_fp = json.loads(f.read())

for num_train in sampled_fn_fp:
    sampled_fn_list = sampled_fn_fp[num_train]["fn"]
    sampled_fp_list = sampled_fn_fp[num_train]["fp"]
    for fn_or_fp, lst in [("fn", sampled_fn_list), ("fp", sampled_fp_list)]:
        if len(lst) > 10:
            lst = random.sample(lst, 10)
        for misclassified_sample in lst:
            # ["10696", 41, [298.0, 63.0, 346.0, 109.0, "metal", "cyan", "cube"], [323.0, 51.0, 356.0, 89.0, "metal", "green", "cylinder"]]
            vid, fid, obj1, obj2 = misclassified_sample
            print(vid, fid, obj1, obj2)
            vid = int(vid)
            cap = cv2.VideoCapture("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/video_validation/video_{}-{}/video_{}.mp4".format(str((vid // 1000) * 1000).zfill(5), str((vid // 1000 + 1) * 1000).zfill(5), str(vid).zfill(5)))

            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, image = cap.read()

            res = cv2.rectangle(image, (int(obj1[0]), int(obj1[1])), (int(obj1[2]), int(obj1[3])), (0,255,0), 1)
            res = cv2.putText(res, "{}-{}-{}".format(obj1[4], obj1[5], obj1[6]), (int(obj1[0]), int(obj1[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            res = cv2.rectangle(res, (int(obj2[0]), int(obj2[1])), (int(obj2[2]), int(obj2[3])), (0,0,255), 1)
            res = cv2.putText(res, "{}-{}-{}".format(obj2[4], obj2[5], obj2[6]), (int(obj2[0]), int(obj2[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite(os.path.join(output_dir, "misclassified/{}/{}_{}_{}_{}.jpg".format(sampling_method, num_train, fn_or_fp, vid, fid)), res)


