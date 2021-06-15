import torch, torchvision

# import some common libraries
import numpy as np
import os, json, cv2, random, time, csv
from tqdm import tqdm
import pickle
from PIL import Image
from torchvision import transforms as T
import sys

import mysql.connector

from construct_input_streams import *
from pattern_matching import *


def isOverlapping(box1, box2):
    # box: x1, y1, x2, y2
    x1min, y1min, x1max, y1max = box1
    x2min, y2min, x2max, y2max = box2
    return x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max

def construct_pos_samples(idx, dataset, connection):
    video_clip_count = 0
    car_stream = construct_input_streams_motorbike_crossing(connection, idx)
    
    video = cv2.VideoCapture("/home/ubuntu/CSE544-project/data/visual_road/traffic-{0}.mp4".format(idx))
    # Get video configurations
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    for i, car_frame_id in enumerate(car_stream):
        filename = "motorbike_crossing/{0}/pos/traffic{1}-{2}.jpg".format(dataset, idx, i)
        if os.path.exists(filename):
            start_frame = car_frame_id - 15
            end_frame = car_frame_id + 15
            # Check whether the examined clip contains pedestrain as well. 
            # has_pedestrain = False
            # intersection = (384, 355, 515, 400)
            # cursor = connection.cursor(buffered=True)
            # cursor.execute("SELECT v.x1, v.x2, v.y1, v.y2, v.frame_id FROM Person p, Event e, VisibleAt v WHERE e.event_id = p.event_id AND e.event_id = v.event_id AND v.filename = 'traffic-%s.mp4' AND v.frame_id >= %s AND v.frame_id <= %s", [idx, start_frame, end_frame])
            # for row in cursor:
            #     x1, x2, y1, y2, frame_id = row
            #     person_box = (x1, y1, x2, y2)
            #     if isOverlapping(intersection, person_box):
            #         has_pedestrain = True
            #         break

            # connection.commit()
            # cursor.close()
            
            # if has_pedestrain == False:
            #     continue

            # Writing video clips
            output_dir = 'video_clips_motorbike_crossing/{0}/pos'.format(dataset)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Define the codec and create VideoWriter object. The output is stored in 'outpy.avi' file.
            out = cv2.VideoWriter(os.path.join(output_dir, 'traffic' + str(idx) + '_' + str(video_clip_count) + '_' + str(start_frame) + '_' + str(end_frame) + '.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width,frame_height))

            video_clip_count += 1
            
            for frame_count in range(end_frame - start_frame + 1):
                ret, frame = video.read()
                out.write(frame)
            out.release()


# def construct_neg_samples(idx, dataset, connection):
#     video_clip_count = 0
#     car_stream1 = construct_input_streams_car_turning_right(connection, idx)
#     car_stream2 = construct_input_streams_car_turning_right_neg(connection, idx)
#     car_stream = []
#     for i, car_frame_id in enumerate(car_stream1):
#         car_stream.append(("pass", i, car_frame_id))
#     for i, car_frame_id in enumerate(car_stream2):
#         car_stream.append(("no", i, car_frame_id))
#     video = cv2.VideoCapture("/home/ubuntu/CSE544-project/data/visual_road/traffic-{0}.mp4".format(idx))
#     # Get video configurations
#     num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = video.get(cv2.CAP_PROP_FPS)
#     frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
#     for sample_class, i, car_frame_id in car_stream:
#         if sample_class == "pass":
#             filename = "car_turning_right_test/{0}/neg/pass_traffic{1}-{2}.jpg".format(dataset, idx, i)
#         else:
#             filename = "car_turning_right_test/{0}/neg/traffic{1}-{2}.jpg".format(dataset, idx, i)
#         if os.path.exists(filename):
#             start_frame = car_frame_id - 15
#             end_frame = car_frame_id + 15
#             # Check whether the examined clip contains pedestrain as well. 
#             # has_pedestrain = False
#             # intersection = (384, 355, 515, 400)
#             # cursor = connection.cursor(buffered=True)
#             # cursor.execute("SELECT v.x1, v.x2, v.y1, v.y2, v.frame_id FROM Person p, Event e, VisibleAt v WHERE e.event_id = p.event_id AND e.event_id = v.event_id AND v.filename = 'traffic-%s.mp4' AND v.frame_id >= %s AND v.frame_id <= %s", [idx, start_frame, end_frame])
#             # for row in cursor:
#             #     x1, x2, y1, y2, frame_id = row
#             #     person_box = (x1, y1, x2, y2)
#             #     if isOverlapping(intersection, person_box):
#             #         has_pedestrain = True
#             #         break

#             # connection.commit()
#             # cursor.close()
            
#             # if has_pedestrain == False:
#             #     continue

#             # Writing video clips
#             output_dir = 'video_clips_test3/{0}/neg'.format(dataset)
#             if not os.path.exists(output_dir):
#                 os.makedirs(output_dir)

#             video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
#             # Define the codec and create VideoWriter object. The output is stored in 'outpy.avi' file.
#             out = cv2.VideoWriter(os.path.join(output_dir, 'traffic' + str(idx) + '_' + str(video_clip_count) + '_' + str(start_frame) + '_' + str(end_frame) + '.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width,frame_height))

#             video_clip_count += 1
            
#             for frame_count in range(end_frame - start_frame + 1):
#                 ret, frame = video.read()
#                 out.write(frame)
#             out.release()


def construct_neg_samples(idx, dataset, connection):
    video_clip_count = 0
    car_stream = construct_input_streams_motorbike_crossing_neg(connection, idx)
    
    video = cv2.VideoCapture("/home/ubuntu/CSE544-project/data/visual_road/traffic-{0}.mp4".format(idx))
    # Get video configurations
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    for i, car_frame_id in enumerate(car_stream):
        filename = "motorbike_crossing/{0}/neg/traffic{1}-{2}.jpg".format(dataset, idx, i)
        if os.path.exists(filename):
            start_frame = car_frame_id - 15
            end_frame = car_frame_id + 15
            # Check whether the examined clip contains pedestrain as well. 
            # has_pedestrain = False
            # intersection = (384, 355, 515, 400)
            # cursor = connection.cursor(buffered=True)
            # cursor.execute("SELECT v.x1, v.x2, v.y1, v.y2, v.frame_id FROM Person p, Event e, VisibleAt v WHERE e.event_id = p.event_id AND e.event_id = v.event_id AND v.filename = 'traffic-%s.mp4' AND v.frame_id >= %s AND v.frame_id <= %s", [idx, start_frame, end_frame])
            # for row in cursor:
            #     x1, x2, y1, y2, frame_id = row
            #     person_box = (x1, y1, x2, y2)
            #     if isOverlapping(intersection, person_box):
            #         has_pedestrain = True
            #         break

            # connection.commit()
            # cursor.close()
            
            # if has_pedestrain == False:
            #     continue

            # Writing video clips
            output_dir = 'video_clips_motorbike_crossing/{0}/neg'.format(dataset)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Define the codec and create VideoWriter object. The output is stored in 'outpy.avi' file.
            out = cv2.VideoWriter(os.path.join(output_dir, 'traffic' + str(idx) + '_' + str(video_clip_count) + '_' + str(start_frame) + '_' + str(end_frame) + '.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width,frame_height))

            video_clip_count += 1
            
            for frame_count in range(end_frame - start_frame + 1):
                ret, frame = video.read()
                out.write(frame)
            out.release()


if __name__ == '__main__':
    connection = mysql.connector.connect(user='admin', password='123456abcABC',
                              host='database-1.cld3cb8o2zkf.us-east-1.rds.amazonaws.com',
                              database='complex_event')

    # for idx in range(1, 16):
    #     construct_pos_samples(idx, "train", connection)
    # for idx in range(16, 21):
    #     construct_pos_samples(idx, "val", connection)
    for idx in range(1, 16):
        construct_neg_samples(idx, "train", connection)
    # for idx in range(16, 21):
    #     construct_neg_samples(idx, "val", connection)

    connection.close()