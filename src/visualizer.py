import os, json, cv2, random, time, csv

class Visualizer:
    def __init__(self, event_name, input_video_dir="/home/ubuntu/complex_event_video/data/visual_road/"):
        self.event_name = event_name
        self.input_video_dir = input_video_dir
        self.out_csv_fn = event_name + ".csv"
        self.output_dir = "/home/ubuntu/complex_event_video/data/" + event_name

    def write_csv(self):
        # Write start_time, end_time to csv file
        with open(self.out_csv_fn, 'w') as f:
            writer = csv.writer(f)
            csv_line = "start_time, end_time"
            writer.writerows([csv_line.split(',')])
            
            for row in out_stream:
                writer.writerow([row[0], row[1]])
                
    def visualize_results(self, out_stream, input_video_fn, event_name, train_or_val, pos_or_neg, write_csv=False):
        if write_csv:
            write_csv()
        video = cv2.VideoCapture(os.path.join(self.input_video_dir, input_video_fn))
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        
        directory = "{0}/{1}/{2}/".format(self.output_dir, train_or_val, pos_or_neg)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Visualize bboxes
        for i, frame_id in enumerate(out_stream):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = video.read()
            cv2.imwrite("{0}/{1}_{2}.jpg".format(directory, input_video_fn[:-4], i), frame)