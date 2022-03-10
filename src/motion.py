# Turning car
# pos_ids = [1, 18, 24, 91, 100, 117, 126, 154, 162, 214, 217, 232, 245, 291, 294, 354, 328, 335, 359, 344, 400, 419]
# neg_ids = [279, 280, 391, 431, 459, 10, 97, 143, 144, 137, 150, 170, 241, 244, 248, 252]

import json
from sort import *
import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
from scipy import signal
import matplotlib.pyplot as plt
from skimage.transform import resize
from numpy.random import default_rng

def prepare_track():
    #create instance of SORT
    mot_tracker = Sort(max_age=5)

    with open("../data/car_turning_traffic2/bbox.json", 'r') as f:
        maskrcnn_bboxes = json.loads(f.read())

    with open("/gscratch/balazinska/enhaoz/complex_event_video/data/car_turning_traffic2/track.json",'w') as out_file:
        n_frames =len(maskrcnn_bboxes)
        for frame_id in range(n_frames):
            detections = []
            # get detections
            res_per_frame = maskrcnn_bboxes["frame_{}.jpg".format(frame_id)]
            for x1, y1, x2, y2, class_name, score in res_per_frame:
                if class_name in ["car", "truck"]:
                    detections.append([max(x1, 0.0), y1, x2, y2, score])
            if len(detections):
                detections = np.asarray(detections)
            else:
                detections = np.empty((0, 5))
            # update SORT
            track_bbs_ids = mot_tracker.update(detections)

            # track_bbs_ids is a np array where each row contains a valid bounding box and track_id (last column)
            for d in track_bbs_ids:
                print('%d,%d,%.2f,%.2f,%.2f,%.2f'%(frame_id,d[4],d[0],d[1],d[2],d[3]),file=out_file)

def visualize_track():
    track_bbs_ids = np.loadtxt("/gscratch/balazinska/enhaoz/complex_event_video/data/car_turning_traffic2/track.json", delimiter=',')

    # Write to video file
    video = cv2.VideoCapture("/gscratch/balazinska/enhaoz/complex_event_video/data/visual_road/traffic-2.mp4")
    width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    fps = video.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter("/gscratch/balazinska/enhaoz/complex_event_video/tmp/traffic-2-annotated.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(width), int(height)))

    current_frame = 0
    while(video.isOpened()):
        ret, frame = video.read()
        if ret==True:
            track_bbs_ids_per_frame = track_bbs_ids[track_bbs_ids[:, 0]==current_frame]
            for obj in track_bbs_ids_per_frame:
                frame = cv2.rectangle(frame, (int(obj[2]), int(obj[3])), (int(obj[4]), int(obj[5])), (36,255,12), 3)
                frame = cv2.putText(
                    img = frame,
                    text = "id: " + str(obj[1]),
                    org = (int(obj[2]), int(obj[3])),
                    fontFace = cv2.FONT_HERSHEY_DUPLEX,
                    fontScale = 1.0,
                    color = (36,255,12),
                    thickness = 3
                    )

            out.write(frame)
            current_frame += 1
        else:
            break

    # Release everything if job is finished
    video.release()
    out.release()
    cv2.destroyAllWindows()

def match_track(tracks):
    track_id_list = np.unique(tracks[:, 1])

    # Turning from east to north
    # train_pos = [18, 154, 328] # List of track_id [18, 154, 162, 328]
    # train_neg = [1, 100, 126, 150, 241, 232]

    # Turning car
    pos_ids = [1, 18, 24, 91, 100, 117, 126, 154, 162, 214, 217, 232, 245, 291, 294, 354, 328, 335, 359, 344, 400, 419]
    neg_ids = []
    for track_id in track_id_list:
        if track_id not in pos_ids:
            neg_ids.append(track_id)

    train_pos, test_pos = train_test_split(pos_ids, train_size=0.75)
    train_neg, test_neg = train_test_split(neg_ids, train_size=len(train_pos)/len(neg_ids))
    print("train_pos:", train_pos)
    print("train_neg:", train_neg)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    test_track_id = []
    for track_id in track_id_list:
        one_track = tracks[tracks[:, 1]==track_id]
        one_track = one_track[np.argsort(one_track[:, 0])]
        bbox_start = one_track[0, 2:]
        bbox_start = list(map(lambda x: max(x,0), bbox_start))
        start_optical_flow = optical_flow(one_track[0, 0], bbox_start)
        # bbox_start = np.array([(bbox_start[0] + bbox_start[2]) / 2, (bbox_start[1] + bbox_start[3]) / 2])
        # bbox_middle = one_track[len(one_track)//2, 2:]
        bbox_end = one_track[-1, 2:]
        bbox_end = list(map(lambda x: max(x,0), bbox_end))
        end_optical_flow = optical_flow(one_track[-1, 0], bbox_end)
        # bbox_end = np.array([(bbox_end[0] + bbox_end[2]) / 2, (bbox_end[1] + bbox_end[3]) / 2])
        motion_features = np.concatenate((bbox_start, bbox_end, start_optical_flow.flatten(), end_optical_flow.flatten()))
        # motion_features = np.concatenate((start_optical_flow.flatten(), end_optical_flow.flatten()))
        if track_id in train_pos:
            train_x.append(motion_features)
            train_y.append(1)
        elif track_id in train_neg:
            train_x.append(motion_features)
            train_y.append(0)
        else:
            test_x.append(motion_features)
            if track_id in test_pos:
                test_y.append(1)
            else:
                test_y.append(0)
            test_track_id.append(track_id)

    # Train the model
    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    pca = PCA(n_components=16)
    pca = pca.fit(train_x[:, 8:])
    train_transformed = pca.transform(train_x[:, 8:])
    train_transformed = np.concatenate((train_x[:, :8], train_transformed), axis=1)
    print("train_transformed.shape", train_transformed.shape)
    clf = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
        )
    clf = clf.fit(train_transformed, train_y)

    # Predict similar tracks
    test_track_id = np.asarray(test_track_id)
    test_x = np.asarray(test_x)
    test_transformed = pca.transform(test_x[:, 8:])
    test_transformed = np.concatenate((test_x[:, :8], test_transformed), axis=1)
    print("test_transformed.shape", test_transformed.shape)

    preds = clf.predict_proba(test_transformed)[:, 1]
    pred_y = clf.predict(test_transformed)
    print("[Metrics] Accuracy: {}, Balanced Accuracy: {}, F1 Score: {}, Precision: {}, Recall: {}".format(metrics.accuracy_score(test_y, pred_y), metrics.balanced_accuracy_score(test_y, pred_y), metrics.f1_score(test_y, pred_y), metrics.precision_score(test_y, pred_y), metrics.recall_score(test_y, pred_y)))

    ind = np.argsort(-preds)
    ranked = test_track_id[ind]
    ranked_scores = preds[ind]
    # print("pred:", ranked)
    rank = 0
    for idx in test_pos:
        print(np.where(ranked == idx)[0])
        rank += np.where(ranked == idx)[0][0]
    print("avg rank of test_pos:", rank/len(test_pos), "total number:", len(ranked))
    return rank/len(test_pos)

def dense_optical_flow(frame_no, bbox):
    cap = cv2.VideoCapture("/gscratch/balazinska/enhaoz/complex_event_video/data/visual_road/traffic-2.mp4")
    frame_no = min(frame_no, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame1 = cap.read()

    # Crop the bounding box
    frame1 = frame1[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    # Resize image
    frame1 = cv2.resize(frame1, (16, 16))
    # cv2.imwrite('/gscratch/balazinska/enhaoz/complex_event_video/tmp/resized_frame1.png', frame1)
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        return None
    # Crop the bounding box
    frame2 = frame2[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    # Resize image
    frame2 = cv2.resize(frame2, (16, 16))
    # cv2.imwrite('/gscratch/balazinska/enhaoz/complex_event_video/tmp/resized_frame2.png', frame2)
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    # move_sense = ang[mag > 0]
    # move_mode = mode(move_sense)[0]
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # cv2.imwrite('/gscratch/balazinska/enhaoz/complex_event_video/tmp/opticalfb.png', frame2)
    # cv2.imwrite('/gscratch/balazinska/enhaoz/complex_event_video/tmp/opticalhsv.png', bgr)
    return bgr

def optical_flow(frame_no, bbox, window_size=15, tau=1e-2):
    cap = cv2.VideoCapture("/gscratch/balazinska/enhaoz/complex_event_video/data/visual_road/traffic-2.mp4")
    frame_no = min(frame_no, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame1 = cap.read()

    # Crop the bounding box
    frame1 = frame1[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    # Resize image
    # frame1 = cv2.resize(frame1, (16, 16))
    I1g = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('/gscratch/balazinska/enhaoz/complex_event_video/tmp/resized_frame1.png', frame1)

    ret, frame2 = cap.read()
    cap.release()
    # Crop the bounding box
    frame2 = frame2[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    # Resize image
    # frame2 = cv2.resize(frame2, (16, 16))
    I2g = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])#*.25
    w = int(window_size/2) # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    I1g = I1g / 255. # normalize pixels
    I2g = I2g / 255. # normalize pixels
    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)
    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)
    # within window window_size * window_size
    for i in range(w, I1g.shape[0]-w):
        for j in range(w, I1g.shape[1]-w):
            Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            It = ft[i-w:i+w+1, j-w:j+w+1].flatten()
            b = np.reshape(It, (It.shape[0],1)) # get b here
            A = np.vstack((Ix, Iy)).T # get A here
            # if threshold τ is larger than the smallest eigenvalue of A'A:
            if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
                nu = np.matmul(np.linalg.pinv(A), b)
                u[i,j]=nu[0]
                v[i,j]=nu[1]

    u = resize(u, (16, 16), anti_aliasing=True)
    v = resize(v, (16, 16), anti_aliasing=True)
    # u = resize(u, I1g.shape, anti_aliasing=True)
    # v = resize(v, I1g.shape, anti_aliasing=True)
    # plt.imshow(frame1)
    # plt.quiver(u, -v)
    # plt.show()
    # plt.savefig("/gscratch/balazinska/enhaoz/complex_event_video/tmp/optical_flow.png")
    res = np.concatenate((u,v))

    return res

def trail_based_match(tracks):
    track_id_list = np.unique(tracks[:, 1])

    # Turning car
    pos_ids = [1, 18, 24, 91, 100, 117, 126, 154, 162, 214, 217, 232, 245, 291, 294, 354, 328, 335, 359, 344, 400, 419]
    neg_ids = []
    for track_id in track_id_list:
        if track_id not in pos_ids:
            neg_ids.append(track_id)

    train_pos, test_pos = train_test_split(pos_ids, train_size=0.75)
    train_neg, test_neg = train_test_split(neg_ids, train_size=len(train_pos)/len(neg_ids))
    print("train_pos:", train_pos)
    print("train_neg:", train_neg)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    test_track_id = []
    for track_id in track_id_list:
        one_track = tracks[tracks[:, 1]==track_id]
        one_track = one_track[np.argsort(one_track[:, 0])]
        mask = np.zeros((540, 960))
        for row in one_track:
            x1 = max(row[2], 0)
            y1 = max(row[3], 0)
            x2 = max(row[4], 0)
            y2 = max(row[5], 0)
            mask[int(y1):int(y2), int(x1):int(x2)] = 1
        if track_id in train_pos:
            train_x.append(mask)
            train_y.append(1)
        else:
            test_x.append(mask)
            if track_id in test_pos:
                test_y.append(1)
            else:
                test_y.append(0)
            test_track_id.append(track_id)

    # Predict similar tracks
    # train_x = np.asarray(train_x)
    # test_x = np.asarray(test_x)
    # print("train_x shape:", train_x.shape)
    test_track_id = np.asarray(test_track_id)
    scores = []
    for track_id, test_mask in zip(test_track_id, test_x):
        dist = -1
        for train_mask in train_x:
            similarity = np.sum(test_mask*train_mask) * (np.sum(train_mask) + np.sum(test_mask)) / 0.5
            dist = max(similarity, dist)
        # print("test_track_id:", track_id, "score: ", dist)
        scores.append(dist)
    scores = np.asarray(scores)
    ind = np.argsort(-scores)
    ranked = test_track_id[ind]
    print("ranked", ranked)
    ranked_scores = scores[ind]

    rank = 0
    for idx in test_pos:
        print(np.where(ranked == idx)[0])
        rank += np.where(ranked == idx)[0][0]
    print("avg rank of test_pos:", rank/len(test_pos), "total number:", len(ranked))
    return rank/len(test_pos)

"""
MOT20:
gt.txt: Each line contains 10 values
<frame>, <track_id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
"""
def ingest_mot():
    print("[ingest_mot]")
    tracks = np.loadtxt("/gscratch/balazinska/enhaoz/complex_event_video/data/MOT20/train/MOT20-05/gt/gt.txt", delimiter=',')
    tracks = tracks[:, :6]
    tracks[:, 4] += tracks[:, 2]
    tracks[:, 5] += tracks[:, 3]
    return tracks

def trail_based_match_inference(tracks, pos_ids, seq_name="MOT20-05", topk=10):
    print("[trail_based_match_inference]")
    img = os.path.join("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/data/MOT20/train/", seq_name, "img1/000001.jpg")
    im =  cv2.imread(img,1)
    height, width, c = im.shape

    track_id_list = np.unique(tracks[:, 1])

    train_x = []
    test_x = []
    test_track_id = []
    for track_id in track_id_list:
        one_track = tracks[tracks[:, 1]==track_id]
        one_track = one_track[np.argsort(one_track[:, 0])]
        mask = np.zeros((height, width))
        for row in one_track:
            x1 = max(row[2], 0)
            y1 = max(row[3], 0)
            x2 = max(row[4], 0)
            y2 = max(row[5], 0)
            mask[int(y1):int(y2), int(x1):int(x2)] = 1
        if track_id in pos_ids:
            train_x.append(mask)
        else:
            test_x.append(mask)
            test_track_id.append(track_id)

    # Predict similar tracks
    test_track_id = np.asarray(test_track_id)
    scores = []
    for track_id, test_mask in zip(test_track_id, test_x):
        dist = -1
        for train_mask in train_x:
            # similarity = np.sum(test_mask*train_mask) * (np.sum(train_mask) + np.sum(test_mask)) / 0.5
            similarity = np.sum(test_mask*train_mask) / np.sum(np.maximum(train_mask, test_mask))
            dist = max(similarity, dist)
        # print("test_track_id:", track_id, "score: ", dist)
        scores.append(dist)
    scores = np.asarray(scores)
    ind = np.argsort(-scores)
    ranked = test_track_id[ind]
    ranked_scores = scores[ind]
    topk_track_ids = ranked[:topk]
    topk_scores = ranked_scores[:topk]
    print("topk_track_ids", topk_track_ids)
    return topk_track_ids, topk_scores

def visualize_track_mot(tracks, pos_ids, topk_ids, topk_scores, seq_name="MOT20-05"):
    fps=25
    img = os.path.join("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/data/MOT20/train/", seq_name, "img1/000001.jpg")
    ref_im =  cv2.imread(img,1)
    height, width, _ = ref_im.shape

    filepaths = []
    for idx, track_ids in enumerate([pos_ids, topk_ids]):
        query_name = "_".join([str(int(x)) for x in pos_ids])
        for rank, track_id in enumerate(track_ids):
            track_id = int(track_id)
            one_track = tracks[tracks[:, 1]==track_id]
            one_track = one_track[np.argsort(one_track[:, 0])]

            output_dir = "/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/tmp/track_mot_new_metric_aggregate/"

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Define the codec and create VideoWriter object. The output is stored in 'outpy.avi' file.
            sub_dir = "topk_preds" if idx else "pos_samples"
            if idx:
                out_filename = os.path.join(output_dir, query_name, sub_dir, '{}_{}_{}.mp4'.format(rank+1, str(track_id).zfill(6), topk_scores[rank]))
            else:
                out_filename = os.path.join(output_dir, query_name, sub_dir, '{}.mp4'.format(str(track_id).zfill(6)))
            filepaths.append(out_filename)
            if os.path.isfile(out_filename):
                continue
            if not os.path.exists(os.path.join(output_dir, query_name, sub_dir)):
                os.makedirs(os.path.join(output_dir, query_name, sub_dir))
            print("[Writing video file]", out_filename)
            out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

            tracking_path = []
            for row in one_track:
                fid = int(row[0])
                img = os.path.join("/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/data/MOT20/train/", seq_name, "img1", "{}.jpg".format(str(fid).zfill(6)))
                frame =  cv2.imread(img,1)
                frame = cv2.rectangle(frame, (int(row[2]), int(row[3])), (int(row[4]), int(row[5])), (36,255,12), 3)
                tracking_path.append([(row[2] + row[4]) / 2, (row[3] + row[5]) / 2])
                for i in range(1, len(tracking_path)):
                    frame = cv2.line(frame, (int(tracking_path[i - 1][0]), int(tracking_path[i - 1][1])), (int(tracking_path[i][0]), int(tracking_path[i][1])), (0, 255, 0), 3)
                out.write(frame)
            out.release()
            frame = ref_im.copy()
            # Write a preview image
            for i in range(1, len(tracking_path)):
                frame = cv2.line(frame, (int(tracking_path[i - 1][0]), int(tracking_path[i - 1][1])), (int(tracking_path[i][0]), int(tracking_path[i][1])), (0, 255, 0), 3)
            if idx:
                cv2.imwrite(os.path.join(output_dir, query_name, sub_dir, '{}_{}_{}.jpg'.format(rank+1, str(track_id).zfill(6), topk_scores[rank])), frame)
            else:
                cv2.imwrite(os.path.join(output_dir, query_name, sub_dir, '{}.jpg'.format(str(track_id).zfill(6))), frame)


if __name__ == '__main__':
    tracks = ingest_mot()
    track_id_list = np.unique(tracks[:, 1])
    # rng = default_rng()
    # pos_ids = rng.choice(track_id_list, 10, replace=False)
    # pos_ids = [181]
    """Manually pick some trajectories.
    Long: 181, 981, 1032, 582, 931
    Short: 199, 360, 252, 957, 844, 838
    Bad: 742, 402, 799, 424
    """
    # Single tracjectory query
    # for pos_ids in [[181], [742], [199], [402], [981], [1032], [582], [931],  [360], [252], [957], [844], [838], [799], [424]]:

    # Aggregate trajectories query
    for pos_ids in [[181, 981], [199, 360], [742, 402], [1032, 252], [957, 799], [424, 582]]:
        topk_ids, topk_scores = trail_based_match_inference(tracks, pos_ids)
        visualize_track_mot(tracks, pos_ids, topk_ids, topk_scores)

    # prepare_track()
    # visualize_track()
    # tracks = np.loadtxt("/gscratch/balazinska/enhaoz/complex_event_video/data/car_turning_traffic2/track.json", delimiter=',')

    # mean_avg_rank = 0
    # for _ in range(100):
    #     # mean_avg_rank += match_track(tracks)
    #     mean_avg_rank += trail_based_match(tracks)
    # mean_avg_rank /= 100
    # print("mean avg rank:", mean_avg_rank)

    # bgr = dense_optical_flow(0, tracks[0, 2:])
    # print(bgr)
    # optical_flow(118, [117.96,371.88,211.56,448.66], 15)
    # optical_flow(0, tracks[0, 2:], 15)
