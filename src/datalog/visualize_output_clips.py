"""
Given input file (e.g. q1_bdd.ans), write to disk a set of video files,
with each video containing the target event and the bounding boxes.
"""
import json, csv
import cv2
import os

def one_graph():
    bdd_filenames = {}
    with open("/home/ubuntu/complex_event_video/src/datalog/data/bdd_filenames.json") as f:
        bdd_filenames = json.loads(f.read())

    query_name = "q2_bdd"
    ans_file_path = "/home/ubuntu/complex_event_video/src/datalog/data/postgres_output/{}.csv".format(query_name)
    with open(ans_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in reader:
            vid, fid1, fid2, x11, y11, x21, y21, x12, y12, x22, y22 = row
            vid = int(vid)
            fid1 = int(fid1)
            fid2 = int(fid2)
            x11, y11, x21, y21, x12, y12, x22, y22 = float(x11), float(y11), float(x21), float(y21), float(x12), float(y12), float(x22), float(y22)
            start_fid = fid1
            end_fid = fid2
            video_basename = os.path.splitext(bdd_filenames[str(vid)])[0]
            video = cv2.VideoCapture(os.path.join("/home/ubuntu/complex_event_video/data/bdd100k/videos/test", video_basename + ".mov"))
            width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
            height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
            fps = video.get(cv2.CAP_PROP_FPS)

            directory = os.path.join("/home/ubuntu/complex_event_video/data/", query_name)
            if not os.path.exists(directory):
                os.makedirs(directory)

            out = cv2.VideoWriter(os.path.join(directory, '{}_{}.mp4'.format(video_basename, i)), cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(width), int(height)))

            video.set(cv2.CAP_PROP_POS_FRAMES, start_fid)

            current_frame = start_fid
            while(video.isOpened()):
                if current_frame > end_fid:
                    break
                ret, frame = video.read()
                if ret==True:
                    if current_frame == fid1:
                        frame = cv2.rectangle(frame, (int(x11), int(y11)), (int(x21), int(y21)), (36,255,12), 3)
                        for _ in range(20):
                            out.write(frame)
                    if current_frame == fid2:
                        frame = cv2.rectangle(frame, (int(x12), int(y12)), (int(x22), int(y22)), (36,255,12), 3)
                        for _ in range(20):
                            out.write(frame)

                    out.write(frame)
                    current_frame += 1
                else:
                    break

            # Release everything if job is finished
            video.release()
            out.release()
            cv2.destroyAllWindows()

            i += 1


def two_graphs():
    bdd_filenames = {}
    with open("/home/ubuntu/complex_event_video/src/datalog/data/bdd_filenames.json") as f:
        bdd_filenames = json.loads(f.read())

    query_name = "q1_bdd"
    ans_file_path = "/home/ubuntu/complex_event_video/src/datalog/data/postgres_output/{}.csv".format(query_name)
    with open(ans_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in reader:
            vid, fid1, fid2, fid3, fid4, x11, y11, x21, y21, x12, y12, x22, y22, x13, y13, x23, y23, x14, y14, x24, y24 = row
            vid = int(vid)
            fid1 = int(fid1)
            fid2 = int(fid2)
            fid3 = int(fid3)
            fid4 = int(fid4)
            x11, y11, x21, y21, x12, y12, x22, y22, x13, y13, x23, y23, x14, y14, x24, y24 = float(x11), float(y11), float(x21), float(y21), float(x12), float(y12), float(x22), float(y22), float(x13), float(y13), float(x23), float(y23), float(x14), float(y14), float(x24), float(y24)
            start_fid = min(fid1, fid3)
            end_fid = max(fid2, fid4)
            video_basename = os.path.splitext(bdd_filenames[str(vid)])[0]
            video = cv2.VideoCapture(os.path.join("/home/ubuntu/complex_event_video/data/bdd100k/videos/test", video_basename + ".mov"))
            # num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
            height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
            fps = video.get(cv2.CAP_PROP_FPS)

            directory = os.path.join("/home/ubuntu/complex_event_video/data/", query_name)
            if not os.path.exists(directory):
                os.makedirs(directory)

            out = cv2.VideoWriter(os.path.join(directory, '{}_{}.mp4'.format(video_basename, i)), cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(width), int(height)))

            video.set(cv2.CAP_PROP_POS_FRAMES, start_fid)

            current_frame = start_fid
            while(video.isOpened()):
                if current_frame > end_fid:
                    break
                ret, frame = video.read()
                if ret==True:
                    if current_frame == fid1:
                        frame = cv2.rectangle(frame, (int(x11), int(y11)), (int(x21), int(y21)), (36,255,12), 3)
                        for _ in range(20):
                            out.write(frame)
                    if current_frame == fid2:
                        frame = cv2.rectangle(frame, (int(x12), int(y12)), (int(x22), int(y22)), (36,255,12), 3)
                        for _ in range(20):
                            out.write(frame)
                    if current_frame == fid3:
                        frame = cv2.rectangle(frame, (int(x13), int(y13)), (int(x23), int(y23)), (0, 255, 255), 3)
                        for _ in range(20):
                            out.write(frame)
                    if current_frame == fid4:
                        frame = cv2.rectangle(frame, (int(x14), int(y14)), (int(x24), int(y24)), (0, 255, 255), 3)
                        for _ in range(20):
                            out.write(frame)

                    out.write(frame)
                    current_frame += 1
                else:
                    break

            # Release everything if job is finished
            video.release()
            out.release()
            cv2.destroyAllWindows()

            i += 1

if __name__ == '__main__':
    two_graphs()