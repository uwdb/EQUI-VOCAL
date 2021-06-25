#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 31st December 2017 - new year eve :)
# ----------------------------------------------

from utils.color_recognition_module import color_histogram_feature_extraction
from utils.color_recognition_module import knn_classifier
import os, cv2
from utils.image_utils import crop_image

# current_path = os.getcwd()
current_path = "/home/ubuntu/CSE544-project/vehicle_counting_tensorflow/"


def color_recognition(crop_img):

    (height, width, channels) = crop_img.shape
#     crop_img = crop_image.crop_center(crop_img, 50, 50)  # crop the detected vehicle image and get a image piece from center of it both for debugging and sending that image piece to color recognition module

    # for debugging
#     cv2.imwrite(current_path + "/debug_utility"+".png",crop_img) # save image piece for debugging
    open(current_path + '/utils/color_recognition_module/' + 'test.data', 'w')
    color_histogram_feature_extraction.color_histogram_of_test_image(crop_img)  # send image piece to regonize vehicle color
    prediction = knn_classifier.main(current_path
            + '/utils/color_recognition_module/' + 'training.data',
            current_path + '/utils/color_recognition_module/'
            + 'test.data')

    return prediction
