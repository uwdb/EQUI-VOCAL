import json
import itertools
from sklearn.metrics import f1_score

segment_length = 128
n_chunks = int(128 / segment_length)

with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_bboxes_evaluation_1000.json", 'r') as f:
    maskrcnn_bboxes_evaluation = json.loads(f.read())
with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_video_list_evaluation_1000.json", 'r') as f:
    video_list_evaluation = json.loads(f.read())

Y_true_all = []
Y_pred_all = []
for video_i, (video_basename, _, _) in enumerate(video_list_evaluation):
    print(video_i, video_basename)
    prediction_matrix_pair_level_per_video = []
    true_labels_pair_level_per_video = []
    # Construct object list
    obj_set = set()
    for frame_id in range(128):
        res_per_frame = maskrcnn_bboxes_evaluation["{}_{}".format(video_basename, frame_id)]
        # [81.0, 109.0, 128.0, 164.0, "metal", "blue", "cylinder"]
        for obj in res_per_frame:
            obj_set.add("{}_{}_{}".format(obj[4], obj[5], obj[6]))

    file = "/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_proposals/sim_{}.json".format(video_basename)
    propnet_preds_file = "/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/propnet_preds/with_edge_supervision_old/sim_{}.json".format(video_basename)
    # Read in bbox info
    with open(file, 'r') as f:
        data = json.load(f)
    collisions = data["ground_truth"]["collisions"]
    objects = data["ground_truth"]["objects"]

    with open(propnet_preds_file, 'r') as f:
        pred_data = json.load(f)
    collisions_pred = pred_data["predictions"][0]["collisions"]
    assert(pred_data["predictions"][0]["what_if"] == -1)

    # Start querying
    for obj1_str_id, obj2_str_id in itertools.combinations(obj_set, 2):
        obj1_id = obj1_str_id.split("_")
        obj2_id = obj2_str_id.split("_")

        Y_true = 0
        Y_pred = 0
        for collision in collisions:
            pos_obj1_id, pos_obj2_id = collision["object"]
            for obj in objects:
                if obj["id"] == pos_obj1_id:
                    pos_obj1 = obj
                if obj["id"] == pos_obj2_id:
                    pos_obj2 = obj
            if (obj1_id[0] == pos_obj1["material"] and obj1_id[1] == pos_obj1["color"] and obj1_id[2] == pos_obj1["shape"] and obj2_id[0] == pos_obj2["material"] and obj2_id[1] == pos_obj2["color"] and obj2_id[2] == pos_obj2["shape"]) or (obj1_id[0] == pos_obj2["material"] and obj1_id[1] == pos_obj2["color"] and obj1_id[2] == pos_obj2["shape"] and obj2_id[0] == pos_obj1["material"] and obj2_id[1] == pos_obj1["color"] and obj2_id[2] == pos_obj1["shape"]):
                # Positive example
                Y_true = 1
                break

        for collision in collisions_pred:
            if collision["frame"] > 128:
                continue
            pos_obj1, pos_obj2 = collision["objects"]
            if (obj1_id[0] == pos_obj1["material"] and obj1_id[1] == pos_obj1["color"] and obj1_id[2] == pos_obj1["shape"] and obj2_id[0] == pos_obj2["material"] and obj2_id[1] == pos_obj2["color"] and obj2_id[2] == pos_obj2["shape"]) or (obj1_id[0] == pos_obj2["material"] and obj1_id[1] == pos_obj2["color"] and obj1_id[2] == pos_obj2["shape"] and obj2_id[0] == pos_obj1["material"] and obj2_id[1] == pos_obj1["color"] and obj2_id[2] == pos_obj1["shape"]):
                # Positive example
                Y_pred = 1
                break

        Y_true_all.append(Y_true)
        Y_pred_all.append(Y_pred)

print("f1 score:", f1_score(Y_true_all, Y_pred_all))
# f1 score: 0.9575177009579342