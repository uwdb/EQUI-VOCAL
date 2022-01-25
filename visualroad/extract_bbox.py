import cv2
import os
import json
import numpy as np

if __name__ == '__main__':
    path = "/home/ubuntu/complex_event_video/visualroad/my-dataset"
    segment_colors = {'pedestrian': (60, 20, 220), 'vehicle': (142, 0, 0)}  # BGR
    objects = ['pedestrian', 'vehicle']
    threshold = 50

    for filename in os.listdir(path):
        if not filename.startswith("semantic"):
            continue
        print(filename)
        bbox_dict = {}
        output_filename = filename[9:-4]
        semantic_video = cv2.VideoCapture(os.path.join(path, filename))
        segmented_result = True
        index = 0

        while segmented_result:
            segmented_result, segmented_frame = semantic_video.read()

            if segmented_result:
                res_per_frame = []
                truth_frame= np.full_like(segmented_frame, 0)
                # Generate truth frame
                for object in objects:
                    color = segment_colors[object]
                    thresholded = cv2.inRange(segmented_frame, tuple(t - threshold for t in color), tuple(t + threshold for t in color))
                    contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(segmented_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        # if index == 0 and filename == "semantic-traffic-000.mp4":
                        #     cv2.imwrite("test.jpg", segmented_frame)
                        res_per_frame.append([x, y, x + w, y + h, object, 1.0])
                bbox_dict["frame_{}.jpg".format(index)] = res_per_frame
                index += 1

        # Write bbox json to file
        with open(os.path.join(path, '{}.json'.format(output_filename)), 'w') as f:
            f.write(json.dumps(bbox_dict))