import csv 
import pickle
import os 

fields = ["file_name", "frame", "object_name", "confidence", "xmin", "ymin", "xmax", "ymax"]
bbox_file_dir = "/home/ubuntu/CSE544-project/data/bdd100k/bbox_files/"

bbox_list = os.listdir(bbox_file_dir)
bbox_list.sort()

total_num = len(bbox_list)
num_per_set = int(total_num / 3) + 1

train_list = bbox_list[0:num_per_set]
thresh_list = bbox_list[num_per_set:(2*num_per_set)]
test_list = bbox_list[(2*num_per_set):total_num]

suffix = ["train", "thresh", "test"]


for idx, sub_list in enumerate([train_list, thresh_list, test_list]):
    total_frames = 0
    rows = []
    for file in sub_list:
        if os.path.splitext(file)[1] != '.pkl':
            continue
        with open(os.path.join(bbox_file_dir, file), 'rb') as f:
            maskrcnn_bboxes = pickle.load(f)
        frame_cnt_per_video = 0
        for frame_id, res_per_frame in enumerate(maskrcnn_bboxes):
            frame_cnt_per_video += 1
            for bbox in res_per_frame:
                # each bbox has format: [638.4600219726562, 421.3493957519531, 741.8064575195312, 507.1121520996094, "car", 0.9954990744590759, "0000000001.jpg"]
                rows.append([os.path.splitext(file)[0], frame_id + total_frames, bbox[4], bbox[5], bbox[0], bbox[2], bbox[1], bbox[3]])
        total_frames += frame_cnt_per_video
    filename = "bdd100k_" + suffix[idx] + ".csv"

    with open("bdd100k_video_list_" + suffix[idx] + ".txt", "w") as f:
        for file in sub_list:
            f.write("%s\n" % (os.path.splitext(file)[0] + ".mov"))

    with open(filename, 'w') as csvfile:
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  
            
        # writing the fields  
        csvwriter.writerow(fields)  
            
        # writing the data rows  
        csvwriter.writerows(rows) 