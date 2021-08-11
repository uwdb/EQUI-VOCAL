import cv2
from utils import frame_from_video
import csv, os


def video_to_images():
    vidcap = cv2.VideoCapture('/home/ubuntu/complex_event_video/data/visual_road/traffic-2.mp4')
    frame_gen = frame_from_video(vidcap)
    for i, frame in enumerate(frame_gen):
        cv2.imwrite("{0}/frame_{1}.jpg".format("/home/ubuntu/complex_event_video/data/traffic2", i), frame)

"""
Target event: a car turning at the intersection while a pedestrian also appears at the intersection.

Annotation for traffic-2.mp4 is stored in the file:
/home/ubuntu/complex_event_video/data/annotation.csv

Output:
A directory containing positive and negative video frames.
car_turning_traffic2
├── neg
│   ├── [frame_number].jpg
│   ├── ...
├── pos
│   ├── [frame_number].jpg
│   ├── ...
"""
def annotate_traffic2():
    pos_frames = []
    with open("/home/ubuntu/complex_event_video/data/annotation.csv", "r") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            start_frame, end_frame = int(row[0]), int(row[1])
            pos_frames += list(range(start_frame, end_frame+1))

    # Create directory and write video frames
    out_dir = "/home/ubuntu/complex_event_video/data/car_turning_traffic2"
    pos_dir = os.path.join(out_dir, "pos")
    neg_dir = os.path.join(out_dir, "neg")
    if not os.path.exists(pos_dir):
        os.makedirs(pos_dir)
    if not os.path.exists(neg_dir):
        os.makedirs(neg_dir)

    vidcap = cv2.VideoCapture('/home/ubuntu/complex_event_video/data/visual_road/traffic-2.mp4')
    frame_gen = frame_from_video(vidcap)
    for i, frame in enumerate(frame_gen):
        cv2.imwrite("{0}/frame_{1}.jpg".format(pos_dir if i in pos_frames else neg_dir, i), frame)


if __name__ == '__main__':
    annotate_traffic2()