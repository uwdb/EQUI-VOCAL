import os
from glob import glob
import ujson as json
import pycocotools._mask as _mask
from torchvision.ops import masks_to_boxes
from .vocal import Vocal
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, balanced_accuracy_score
import joblib

def decode(rleObjs):
    if type(rleObjs) == list:
        return _mask.decode(rleObjs)
    else:
        return _mask.decode([rleObjs])[:,:,0]


class TrainAndEvalProxyModel(Vocal):
    def __init__(self, dataset="visualroad_traffic2", query="test_b", temporal_heuristic=True, method="VOCAL", budget=800, frame_selection_method="least_confidence", thresh=1.0):

        super().__init__(dataset, query, temporal_heuristic, method)
        self.thresh = thresh
        self.frame_selection_method = frame_selection_method
        self.budget = budget
        if frame_selection_method == "least_confidence":
            self.frame_selection = self.least_confidence_frame_selection
        elif frame_selection_method == "greatest_confidence":
            self.frame_selection = self.greatest_confidence_frame_selection
        elif frame_selection_method == "random":
            self.frame_selection = self.random_frame_selection
        if dataset == "clevrer":
            self.ingest_bbox_info_clevrer_evaluation()
        self.ingest_gt_labels_evaluation()
        self.model_performance_output = []
        self.sampled_fn_fp = {}

    def ingest_gt_labels(self):
        if self.query == "clevrer_collision":
            self.pos_frames, self.pos_frames_per_instance = self.clevrer_collision()
        elif self.query in ["clevrer_far", "clevrer_near"]:
            outfile_name = os.path.join("/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/intermediate_results", "{}-{}.npz".format(self.query, self.thresh))
            pos_frames_per_instance_outfile_name = os.path.join("/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/intermediate_results", "{}-{}-pos_frames_per_instance.json".format(self.query, self.thresh))
            if os.path.exists(outfile_name) and os.path.exists(pos_frames_per_instance_outfile_name):
                npzfile = np.load(outfile_name)
                self.spatial_features = npzfile["spatial_features"]
                self.candidates = npzfile["candidates"]
                self.Y = npzfile["Y"]
                self.pos_frames = npzfile["pos_frames"]
                self.spatial_feature_dim = 8
                self.feature_names = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
                with open(pos_frames_per_instance_outfile_name, 'r') as f:
                    self.pos_frames_per_instance = json.loads(f.read())
            else:
                if self.query == "clevrer_far":
                    self.clevrer_far()
                elif self.query == "clevrer_near":
                    self.clevrer_near()
                np.savez(outfile_name, spatial_features=self.spatial_features, candidates=self.candidates, Y=self.Y, pos_frames=self.pos_frames)
                # save pos_frames_per_instance to file
                with open(pos_frames_per_instance_outfile_name, 'w') as f:
                    f.write(json.dumps(self.pos_frames_per_instance))
        self.n_positive_instances = len(self.pos_frames_per_instance)
        self.avg_duration = 1.0 * len(self.pos_frames) / self.n_positive_instances
        print("# positive frames: ", len(self.pos_frames), "; # distinct instances: ", self.n_positive_instances, "; average duration: ", self.avg_duration)
        print("Durations of all instances: ")
        print(" ".join([str(v[1]- v[0]) for _, v in self.pos_frames_per_instance.items()]))
        temp = np.array([(v[1]- v[0]) for _, v in self.pos_frames_per_instance.items()])
        print("duration std: ", temp.std())


    def ingest_bbox_info_clevrer_evaluation(self):
        # check if file exists
        if os.path.isfile("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_bboxes_evaluation_1000.json") and os.path.isfile("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_video_list_evaluation_1000.json"):
            with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_bboxes_evaluation_1000.json", 'r') as f:
                self.maskrcnn_bboxes_evaluation = json.loads(f.read())
            with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_video_list_evaluation_1000.json", 'r') as f:
                self.video_list_evaluation = json.loads(f.read())
            self.n_frames_evaluation = len(self.maskrcnn_bboxes_evaluation)
            print("# all frames in evaluation: ", self.n_frames_evaluation, "; # video files in evaluation: ", len(self.video_list_evaluation))
        else:
            self.maskrcnn_bboxes_evaluation = {} # {video_id_frame_id: [x1, y1, x2, y2, material, color, shape]}
            self.video_list_evaluation = [] # (video_basename, frame_offset, n_frames)
            # iterate all files in the folder
            for file in glob("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_proposals/*.json"):
                video_basename = os.path.basename(file).replace(".json", "").replace("sim_", "")
                if int(video_basename) >= 11000 or int(video_basename) < 10000:
                    continue
                # read in the json file
                with open(file, 'r') as f:
                    data = json.loads(f.read())
                frame_offset = len(self.maskrcnn_bboxes_evaluation)
                # iterate all videos in the json file
                current_fid = 0
                for frame in data['frames']:
                    local_frame_id = frame['frame_index']
                    while local_frame_id > current_fid:
                        self.maskrcnn_bboxes_evaluation[video_basename + "_" + str(current_fid)] = []
                        current_fid += 1
                    objects = frame['objects']
                    box_list = []
                    for i in range(len(objects)):
                        # print(objects[i]['material'], objects[i]['color'], objects[i]['shape'])
                        mask = decode(objects[i]['mask'])
                        # O represents black, 1 represents white.
                        box = masks_to_boxes(torch.from_numpy(mask[np.newaxis, :]))
                        box = np.squeeze(box.numpy(), axis=0).tolist()
                        box.extend([objects[i]['material'], objects[i]['color'], objects[i]['shape']])
                        box_list.append(box)
                    self.maskrcnn_bboxes_evaluation[video_basename + "_" + str(local_frame_id)] = box_list
                    current_fid += 1
                while current_fid < 128:
                    self.maskrcnn_bboxes_evaluation[video_basename + "_" + str(current_fid)] = []
                    current_fid += 1
                self.video_list_evaluation.append((video_basename, frame_offset, 128))
            self.n_frames_evaluation = len(self.maskrcnn_bboxes_evaluation)
            print("# all frames in evaluation: ", self.n_frames_evaluation, "; # video files: ", len(self.video_list_evaluation))
            # write dict to file
            with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_bboxes_evaluation_1000.json", 'w') as f:
                f.write(json.dumps(self.maskrcnn_bboxes_evaluation))
            # write video_list to file
            with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_video_list_evaluation_1000.json", 'w') as f:
                f.write(json.dumps(self.video_list_evaluation))

    def ingest_gt_labels_evaluation(self):
        if self.query == "clevrer_collision":
            outfile_name = os.path.join("/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/intermediate_results", "{}-original-evaluation_1000.npz".format(self.query))
            raw_data_pair_level_evaluation_outfile_name = "/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/intermediate_results/{}-original-raw_data_pair_level_evaluation.json".format(self.query)
            if os.path.exists(outfile_name) and os.path.exists(raw_data_pair_level_evaluation_outfile_name):
                npzfile = np.load(outfile_name)
                self.spatial_features_evaluation = npzfile["spatial_features_evaluation"]
                self.Y_evaluation = npzfile["Y_evaluation"]
                self.Y_pair_level_evaluation = npzfile["Y_pair_level_evaluation"]
                self.feature_index = npzfile["feature_index"]
                with open(raw_data_pair_level_evaluation_outfile_name, 'r') as f:
                    self.raw_data_pair_level_evaluation = json.loads(f.read())
            else:
                self.spatial_features_evaluation, self.Y_evaluation, self.Y_pair_level_evaluation, self.feature_index, self.raw_data_pair_level_evaluation = self.clevrer_collision_evaluation()
                np.savez(outfile_name, spatial_features_evaluation=self.spatial_features_evaluation, Y_evaluation=self.Y_evaluation, Y_pair_level_evaluation=self.Y_pair_level_evaluation, feature_index=self.feature_index)
                # save raw_data_pair_level_evaluation to file
                with open(raw_data_pair_level_evaluation_outfile_name, 'w') as f:
                    f.write(json.dumps(self.raw_data_pair_level_evaluation))
        elif self.query in ["clevrer_far", "clevrer_near"]:
            outfile_name = os.path.join("/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/intermediate_results", "{}-{}-evaluation.npz".format(self.query, self.thresh))
            raw_data_pair_level_evaluation_outfile_name = "/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/intermediate_results/{}-{}-raw_data_pair_level_evaluation.json".format(self.query, self.thresh)
            if os.path.exists(outfile_name) and os.path.exists(raw_data_pair_level_evaluation_outfile_name):
                npzfile = np.load(outfile_name)
                self.spatial_features_evaluation = npzfile["spatial_features_evaluation"]
                self.Y_evaluation = npzfile["Y_evaluation"]
                self.Y_pair_level_evaluation = npzfile["Y_pair_level_evaluation"]
                self.feature_index = npzfile["feature_index"]
                with open(raw_data_pair_level_evaluation_outfile_name, 'r') as f:
                    self.raw_data_pair_level_evaluation = json.loads(f.read())
            else:
                if self.query == "clevrer_near":
                    self.clevrer_near_evaluation()
                elif self.query == "clevrer_far":
                    self.clevrer_far_evaluation()
                np.savez(outfile_name, spatial_features_evaluation=self.spatial_features_evaluation, Y_evaluation=self.Y_evaluation, Y_pair_level_evaluation=self.Y_pair_level_evaluation, feature_index=self.feature_index)
                # save raw_data_pair_level_evaluation to file
                with open(raw_data_pair_level_evaluation_outfile_name, 'w') as f:
                    f.write(json.dumps(self.raw_data_pair_level_evaluation))

    def run(self):
        self.query_initialization = self.random_initialization()

        self.proxy_model_training()

        self.frame_id_arr_to_annotate = np.empty(self.annotated_batch_size)  # Construct a pseudo list of length BATCH_SIZE to pass the While condition.
        while self.num_positive_instances_found < self.n_positive_instances and self.frame_id_arr_to_annotate.size >= self.annotated_batch_size and (len(self.negative_frames_seen) + len(self.positive_frames_seen)) <= self.budget:
            self.update_random_choice_p()
            self.frame_selection()
            self.user_feedback()
            self.proxy_model_training()
            self.get_frames_stats()

            num_trained = len(self.negative_frames_seen) + len(self.positive_frames_seen)
            if (num_trained) % 50 == 0:
                iteration_str = "training with {} data: ".format(num_trained)
                print(iteration_str)
                self.model_performance_output.append(iteration_str)
                self.eval_and_save_proxy_model()
                # Save the random forest model
                joblib.dump(self.clf, "/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/{}/models/random_forest-{}-{}-{}-original{}.joblib".format(self.query, self.thresh, num_trained, self.frame_selection_method, "-with_heuristic" if self.temporal_heuristic else ""))
                # loaded_rf = joblib.load("./random_forest.joblib")
            self.iteration += 1

        print("stats_per_chunk", self.stats_per_chunk)
        if self.query == "clevrer_collision":
            # save model_performance_output to txt file
            with open("/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/clevrer_collision/{}-original-random_forest{}.txt".format(self.frame_selection_method, "-with_heuristic" if self.temporal_heuristic else ""), 'w') as f:
                f.write(json.dumps(self.model_performance_output))
            # save self.sampled_fn_fp to json file
            with open("/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/clevrer_collision/{}-original-random_forest-sampled_fn_fp{}.json".format(self.frame_selection_method, "-with_heuristic" if self.temporal_heuristic else ""), 'w') as f:
                f.write(json.dumps(self.sampled_fn_fp))
        elif self.query in ["clevrer_far", "clevrer_near"]:
            # save model_performance_output to txt file
            with open("/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/{}/{}-{}-original-random_forest{}.txt".format(self.query, self.thresh, self.frame_selection_method, "-with_heuristic" if self.temporal_heuristic else ""), 'w') as f:
                f.write(json.dumps(self.model_performance_output))
            # save self.sampled_fn_fp to json file
            with open("/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/{}/{}-{}-original-random_forest-sampled_fn_fp{}.json".format(self.query, self.thresh, self.frame_selection_method, "-with_heuristic" if self.temporal_heuristic else ""), 'w') as f:
                f.write(json.dumps(self.sampled_fn_fp))
        return self.plot_data_y_annotated, self.plot_data_y_materialized

    def eval_and_save_proxy_model(self):
        # Evaluate metrics on evaluation set
        y_pred_pair_level = self.clf.predict(self.spatial_features_evaluation)
        tn, fp, fn, tp = confusion_matrix(self.Y_pair_level_evaluation, y_pred_pair_level).ravel()
        evaluation_per_pair_str = "[evaluation pair level] balanced_accuracy: {}; f1_score: {}; tn, fp, fn, tp: {}, {}, {}, {}".format(balanced_accuracy_score(self.Y_pair_level_evaluation, y_pred_pair_level), f1_score(self.Y_pair_level_evaluation, y_pred_pair_level), tn, fp, fn, tp)
        self.model_performance_output.append(evaluation_per_pair_str)
        print(evaluation_per_pair_str)

        df = pd.DataFrame({"y_pred_pair_level": y_pred_pair_level, "feature_index": self.feature_index})
        df = df.groupby("feature_index", as_index=False).max()
        pos_index = df[df.y_pred_pair_level == 1]["feature_index"].values.tolist()
        Y_pred_frame_level = np.zeros(self.n_frames_evaluation, dtype=int)
        Y_pred_frame_level[pos_index] = 1

        tn, fp, fn, tp = confusion_matrix(self.Y_evaluation, Y_pred_frame_level).ravel()
        evaluation_str = "[evaluation frame level] balanced_accuracy: {}; f1_score: {}; tn, fp, fn, tp: {}, {}, {}, {}".format(balanced_accuracy_score(self.Y_evaluation, Y_pred_frame_level), f1_score(self.Y_evaluation, Y_pred_frame_level), tn, fp, fn, tp)
        self.model_performance_output.append(evaluation_str)
        print(evaluation_str)

        # Get indexes for tp, tn, fp, fn
        unq = np.array([x + 2*y for x, y in zip(y_pred_pair_level, self.Y_pair_level_evaluation)])
        fp_ind = np.array(np.where(unq == 1)).tolist()[0]
        fn_ind = np.array(np.where(unq == 2)).tolist()[0]

        # Randomly sample 10 fp and fn to visualize
        if len(fp_ind) > 10:
            fp_ind = np.random.choice(fp_ind, size=10, replace=False)
        if len(fn_ind) > 10:
            fn_ind = np.random.choice(fn_ind, size=10, replace=False)

        # each item in self.raw_data_pair_level_evaluation[i] is of form: [video_basename, frame_id, obj1, obj2]
        sampled_fp = [self.raw_data_pair_level_evaluation[i] for i in fp_ind]
        sampled_fn = [self.raw_data_pair_level_evaluation[i] for i in fn_ind]
        self.sampled_fn_fp[len(self.negative_frames_seen) + len(self.positive_frames_seen)] = {"fp": sampled_fp, "fn": sampled_fn}

    def save_data(self, plot_data_y):
        if self.query == "clevrer_collision":
            method =  "{}-original-random_forest{}".format(self.frame_selection_method, "-with_heuristic" if self.temporal_heuristic else "")
        elif self.query in ["clevrer_far", "clevrer_near"]:
            method =  "{}-{}-original-random_forest{}".format(self.thresh, self.frame_selection_method, "-with_heuristic" if self.temporal_heuristic else "")

        with open("/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/{}/instances_found_speed/{}.json".format(self.query, method), 'w') as f:
            f.write(json.dumps(plot_data_y.tolist()))


class ClevrerEdge(TrainAndEvalProxyModel):
    def __init__(self, dataset="clevrer", query="clevrer_edge", temporal_heuristic=False, method="VOCAL", budget=800, frame_selection_method="least_confidence", thresh=1.0):
        super().__init__(dataset, query, temporal_heuristic, method, thresh)
        self.thresh = thresh
        self.frame_selection_method = frame_selection_method
        self.budget = budget
        if frame_selection_method == "least_confidence":
            self.frame_selection = self.least_confidence_frame_selection
        elif frame_selection_method == "greatest_confidence":
            self.frame_selection = self.greatest_confidence_frame_selection
        elif frame_selection_method == "random":
            self.frame_selection = self.random_frame_selection
        if dataset == "clevrer":
            self.ingest_bbox_info_clevrer_evaluation()
        self.ingest_gt_labels_evaluation()
        self.model_performance_output = []
        self.sampled_fn_fp = {}

    def ingest_gt_labels(self):
        outfile_name = os.path.join("/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/intermediate_results", "{}-{}.npz".format(self.query, self.thresh))
        pos_frames_per_instance_outfile_name = os.path.join("/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/intermediate_results", "{}-{}-pos_frames_per_instance.json".format(self.query, self.thresh))
        if os.path.exists(outfile_name) and os.path.exists(pos_frames_per_instance_outfile_name):
            npzfile = np.load(outfile_name)
            self.spatial_features = npzfile["spatial_features"]
            self.candidates = npzfile["candidates"]
            self.Y = npzfile["Y"]
            self.pos_frames = npzfile["pos_frames"]
            self.spatial_feature_dim = 5
            self.feature_names = ["x", "y", "w", "h", "r"]
            with open(pos_frames_per_instance_outfile_name, 'r') as f:
                self.pos_frames_per_instance = json.loads(f.read())
        else:
            if self.query == "clevrer_edge":
                self.clevrer_edge()
            elif self.query == "clevrer_center":
                self.clevrer_center()
            np.savez(outfile_name, spatial_features=self.spatial_features, candidates=self.candidates, Y=self.Y, pos_frames=self.pos_frames)
            # save pos_frames_per_instance to file
            with open(pos_frames_per_instance_outfile_name, 'w') as f:
                f.write(json.dumps(self.pos_frames_per_instance))

        self.n_positive_instances = len(self.pos_frames_per_instance)
        self.avg_duration = 1.0 * len(self.pos_frames) / self.n_positive_instances
        print("# positive frames: ", len(self.pos_frames), "; # distinct instances: ", self.n_positive_instances, "; average duration: ", self.avg_duration)
        print("Durations of all instances: ")
        print(" ".join([str(v[1]- v[0]) for _, v in self.pos_frames_per_instance.items()]))
        temp = np.array([(v[1]- v[0]) for _, v in self.pos_frames_per_instance.items()])
        print("duration std: ", temp.std())


    def ingest_bbox_info_clevrer_evaluation(self):
        # check if file exists
        if os.path.isfile("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_bboxes_evaluation_1000.json") and os.path.isfile("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_video_list_evaluation_1000.json"):
            with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_bboxes_evaluation_1000.json", 'r') as f:
                self.maskrcnn_bboxes_evaluation = json.loads(f.read())
            with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_video_list_evaluation_1000.json", 'r') as f:
                self.video_list_evaluation = json.loads(f.read())
            self.n_frames_evaluation = len(self.maskrcnn_bboxes_evaluation)
            print("# all frames in evaluation: ", self.n_frames_evaluation, "; # video files in evaluation: ", len(self.video_list_evaluation))
        else:
            self.maskrcnn_bboxes_evaluation = {} # {video_id_frame_id: [x1, y1, x2, y2, material, color, shape]}
            self.video_list_evaluation = [] # (video_basename, frame_offset, n_frames)
            # iterate all files in the folder
            for file in glob("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_proposals/*.json"):
                video_basename = os.path.basename(file).replace(".json", "").replace("sim_", "")
                if int(video_basename) >= 11000 or int(video_basename) < 10000:
                    continue
                # read in the json file
                with open(file, 'r') as f:
                    data = json.loads(f.read())
                frame_offset = len(self.maskrcnn_bboxes_evaluation)
                # iterate all videos in the json file
                current_fid = 0
                for frame in data['frames']:
                    local_frame_id = frame['frame_index']
                    while local_frame_id > current_fid:
                        self.maskrcnn_bboxes_evaluation[video_basename + "_" + str(current_fid)] = []
                        current_fid += 1
                    objects = frame['objects']
                    box_list = []
                    for i in range(len(objects)):
                        # print(objects[i]['material'], objects[i]['color'], objects[i]['shape'])
                        mask = decode(objects[i]['mask'])
                        # O represents black, 1 represents white.
                        box = masks_to_boxes(torch.from_numpy(mask[np.newaxis, :]))
                        box = np.squeeze(box.numpy(), axis=0).tolist()
                        box.extend([objects[i]['material'], objects[i]['color'], objects[i]['shape']])
                        box_list.append(box)
                    self.maskrcnn_bboxes_evaluation[video_basename + "_" + str(local_frame_id)] = box_list
                    current_fid += 1
                while current_fid < 128:
                    self.maskrcnn_bboxes_evaluation[video_basename + "_" + str(current_fid)] = []
                    current_fid += 1
                self.video_list_evaluation.append((video_basename, frame_offset, 128))
            self.n_frames_evaluation = len(self.maskrcnn_bboxes_evaluation)
            print("# all frames in evaluation: ", self.n_frames_evaluation, "; # video files: ", len(self.video_list_evaluation))
            # write dict to file
            with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_bboxes_evaluation_1000.json", 'w') as f:
                f.write(json.dumps(self.maskrcnn_bboxes_evaluation))
            # write video_list to file
            with open("/gscratch/balazinska/enhaoz/complex_event_video/data/clevrer/processed_video_list_evaluation_1000.json", 'w') as f:
                f.write(json.dumps(self.video_list_evaluation))

    def ingest_gt_labels_evaluation(self):
        outfile_name = os.path.join("/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/intermediate_results", "{}-{}-evaluation.npz".format(self.query, self.thresh))
        raw_data_pair_level_evaluation_outfile_name = "/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/intermediate_results/{}-{}-raw_data_pair_level_evaluation.json".format(self.query, self.thresh)
        if os.path.exists(outfile_name) and os.path.exists(raw_data_pair_level_evaluation_outfile_name):
            npzfile = np.load(outfile_name)
            self.spatial_features_evaluation = npzfile["spatial_features_evaluation"]
            self.Y_evaluation = npzfile["Y_evaluation"]
            self.Y_pair_level_evaluation = npzfile["Y_pair_level_evaluation"]
            self.feature_index = npzfile["feature_index"]
            with open(raw_data_pair_level_evaluation_outfile_name, 'r') as f:
                self.raw_data_pair_level_evaluation = json.loads(f.read())
        else:
            if self.query == "clevrer_edge":
                self.clevrer_edge_evaluation()
            elif self.query == "clevrer_center":
                self.clevrer_center_evaluation()
            np.savez(outfile_name, spatial_features_evaluation=self.spatial_features_evaluation, Y_evaluation=self.Y_evaluation, Y_pair_level_evaluation=self.Y_pair_level_evaluation, feature_index=self.feature_index)
            # save raw_data_pair_level_evaluation to file
            with open(raw_data_pair_level_evaluation_outfile_name, 'w') as f:
                f.write(json.dumps(self.raw_data_pair_level_evaluation))

    def run(self):
        self.query_initialization = self.random_initialization()

        self.proxy_model_training()

        self.frame_id_arr_to_annotate = np.empty(self.annotated_batch_size)  # Construct a pseudo list of length BATCH_SIZE to pass the While condition.
        while self.num_positive_instances_found < self.n_positive_instances and self.frame_id_arr_to_annotate.size >= self.annotated_batch_size and (len(self.negative_frames_seen) + len(self.positive_frames_seen)) <= self.budget:
            self.update_random_choice_p()
            self.frame_selection()
            self.user_feedback()
            self.proxy_model_training()
            self.get_frames_stats()

            num_trained = len(self.negative_frames_seen) + len(self.positive_frames_seen)
            if (num_trained) % 50 == 0:
                iteration_str = "training with {} data: ".format(num_trained)
                print(iteration_str)
                self.model_performance_output.append(iteration_str)
                self.eval_and_save_proxy_model()
                # Save the random forest model
                joblib.dump(self.clf, "/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/{}/models/random_forest-{}-{}-{}-original{}.joblib".format(self.query, self.thresh, num_trained, self.frame_selection_method, "-with_heuristic" if self.temporal_heuristic else ""))
                # loaded_rf = joblib.load("./random_forest.joblib")
            self.iteration += 1

        print("stats_per_chunk", self.stats_per_chunk)

        # save model_performance_output to txt file
        with open("/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/{}/{}-{}-original-random_forest{}.txt".format(self.query, self.thresh, self.frame_selection_method, "-with_heuristic" if self.temporal_heuristic else ""), 'w') as f:
            f.write(json.dumps(self.model_performance_output))
        # save self.sampled_fn_fp to json file
        with open("/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/{}/{}-{}-original-random_forest-sampled_fn_fp{}.json".format(self.query, self.thresh, self.frame_selection_method, "-with_heuristic" if self.temporal_heuristic else ""), 'w') as f:
            f.write(json.dumps(self.sampled_fn_fp))
        return self.plot_data_y_annotated, self.plot_data_y_materialized


    def save_data(self, plot_data_y):
        method =  "{}-{}-original-random_forest{}".format(self.thresh, self.frame_selection_method, "-with_heuristic" if self.temporal_heuristic else "")
        with open("/gscratch/balazinska/enhaoz/complex_event_video/src/outputs/{}/instances_found_speed/{}.json".format(self.query, method), 'w') as f:
            f.write(json.dumps(plot_data_y.tolist()))