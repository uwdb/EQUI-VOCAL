"""
In this experiment, the target query is: a car turning at the intersection while a pedestrian also appears at the intersection.

Different query specifications with various complexity are tried:
1. Random sampling
2. Only car
3. Only pedestrian
4. Car + pedestrian
5. Objects with spatial configurations
6. Scene graph: edges are created based on bbox information
7. negation in scene graph
8. Temporal scene graph: refer to the paper
"""

from shutil import copyfile
import json
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os, csv
from tqdm import tqdm
from sklearn import tree, metrics
import numpy as np
import graphviz
from sklearn.ensemble import RandomForestClassifier

from utils.utils import isInsideIntersection, isOverlapping
from utils import tools
from faster_r_cnn import FasterRCNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Acc: 0.5017 F1: 0.0663
# time: 173.445s
class RandomSampling:
    def __call__(self, x):
        return torch.randint(0, 2, (x.size()[0],)).to(device)


# Acc: 0.8349 F1: 0.2599
# time: 1736.709s
class OnlyCar(FasterRCNN):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        inputs = [{"image": img_tensor} for img_tensor in x]
        outputs = self.model(inputs)
        preds = torch.zeros(len(inputs), dtype=torch.int)
        for i, output in enumerate(outputs):
            instances = output["instances"]
            pred_boxes = instances.pred_boxes
            scores = instances.scores
            pred_classes = instances.pred_classes
            for pred_box, score, pred_class in zip(pred_boxes, scores, pred_classes):
                if (pred_class.item() in [2, 7] and isInsideIntersection(pred_box)):
                    preds[i] = 1
                    break
        return preds.to(device)

# Acc: 0.3529 F1: 0.0949
class OnlyPedestrian(FasterRCNN):
    def __init__(self):
        super().__init__()
        self.edge_corner_bbox = (367, 345, 540, 418)

    def __call__(self, x):
        inputs = [{"image": img_tensor} for img_tensor in x]
        outputs = self.model(inputs)
        preds = torch.zeros(len(inputs), dtype=torch.int)
        for i, output in enumerate(outputs):
            instances = output["instances"]
            pred_boxes = instances.pred_boxes
            scores = instances.scores
            pred_classes = instances.pred_classes
            for pred_box, score, pred_class in zip(pred_boxes, scores, pred_classes):
                if (pred_class.item() == 0 and isOverlapping(self.edge_corner_bbox, pred_box)):
                    preds[i] = 1
                    break
        return preds.to(device)

# Standard method:
# Acc: 0.8985 F1: 0.3584 Precision: 0.2317 Recall: 0.7916
# time: 1882.309s
# Higher_precision method (distinct intersection definition):
# Acc: 0.9750 F1: 0.6694 Precision: 0.6742 Recall: 0.6647
# time: 1788.699s
class PedestrianAndCar(FasterRCNN):
    def __init__(self):
        super().__init__()
        self.edge_corner_bbox = (367, 345, 540, 418)

    def __call__(self, x):
        inputs = [{"image": img_tensor} for img_tensor in x]
        outputs = self.model(inputs)
        preds = torch.zeros(len(inputs), dtype=torch.int)
        for i, output in enumerate(outputs):
            instances = output["instances"]
            pred_boxes = instances.pred_boxes
            has_car = 0
            has_pedestrian = 0
            pred_classes = instances.pred_classes
            for pred_box, pred_class in zip(pred_boxes, pred_classes):
                if (pred_class.item() == 0 and isOverlapping(self.edge_corner_bbox, pred_box)):
                    has_pedestrian = 1
                elif (pred_class.item() in [2, 7] and isInsideIntersection(pred_box)):
                    has_car = 1
                if has_car and has_pedestrian:
                    preds[i] = 1
                    break
        return preds.to(device)

# Smaller intersection (inner line)
# training data stats: # pos: 590; # neg: 15059; pos percent: 0.03770208959038916
# test data stats: # pos: 254; # neg: 6598; pos percent: 0.037069468768242846
# training data performance:
# Accuracy: 0.9845996549300274
# F1: 0.7644183773216031
# Precision: 1.0
# Recall: 0.6186708860759493
# test data performance:
# Accuracy: 0.9896380618797431
# F1: 0.8255528255528255
# Precision: 0.9180327868852459
# Recall: 0.75

# Larger intersection (outer line)
# training data stats: # pos: 1749; # neg: 13900; pos percent: 0.11176432998913668
# test data stats: # pos: 1005; # neg: 5847; pos percent: 0.14667250437828372
# training data performance:
# Accuracy: 0.9904147229854943
# F1: 0.8653500897666068
# Precision: 1.0
# Recall: 0.7626582278481012
# test data performance:
# Accuracy: 0.989492119089317
# F1: 0.8277511961722488
# Precision: 0.8917525773195877
# Recall: 0.772321428571428
class PedestrianAndCarWithSpatialFeatures(FasterRCNN):
    def __init__(self):
        super().__init__()
        self.spatial_feature_dim = 5
        self.edge_corner_bbox = (367, 345, 540, 418)
        # Read in bbox info
        bbox_file = "/home/ubuntu/complex_event_video/data/car_turning_traffic2/bbox.json"
        with open(bbox_file, 'r') as f:
            self.maskrcnn_bboxes = json.loads(f.read())
        # Read labels
        self.pos_frames = []
        with open("/home/ubuntu/complex_event_video/data/annotation.csv", "r") as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)
            for row in csvreader:
                start_frame, end_frame = int(row[0]), int(row[1])
                self.pos_frames += list(range(start_frame, end_frame+1))
        # Training: [0, 15648]
        # Testing: [15649, 22500]
        self.X = np.zeros((len(self.maskrcnn_bboxes), self.spatial_feature_dim), dtype=np.float64)
        self.Y = np.zeros(len(self.maskrcnn_bboxes), dtype=np.int)
        for i in range(len(self.maskrcnn_bboxes)):
            if i in self.pos_frames:
                self.Y[i] = 1
        self.pred = np.zeros(len(self.maskrcnn_bboxes), dtype=np.int)
        # self.clf = tree.DecisionTreeClassifier(
        #     # criterion="entropy",
        #     max_depth=None,
        #     min_samples_split=2,
        #     # class_weight="balanced"
        #     )
        self.clf = RandomForestClassifier(
            # criterion="entropy",
            # max_depth=10,
            n_estimators=10,
            # min_samples_split=32,
            class_weight="balanced"
        )
        # self.clf = SVC(gamma='scale', C=0.0025, class_weight="balanced")

    # First pass
    def eval_faster_r_cnn(self):
        for image_name, res_per_frame in self.maskrcnn_bboxes.items():
            frame_idx = int(image_name.strip("frame_.jpg"))
            has_car = 0
            has_pedestrian = 0
            for x1, y1, x2, y2, class_name, score in res_per_frame:
                if (class_name == "person" and isOverlapping(self.edge_corner_bbox, (x1, y1, x2, y2))):
                    has_pedestrian = 1
                elif (class_name in ["car", "truck"] and isInsideIntersection((x1, y1, x2, y2))):
                    has_car = 1
                if has_car and has_pedestrian:
                    self.pred[frame_idx] = 1
                    break

    def construct_features(self):
        for image_name, res_per_frame in self.maskrcnn_bboxes.items():
            frame_idx = int(image_name.strip("frame_.jpg"))
            if self.pred[frame_idx] == 0:
                continue
            for x1, y1, x2, y2, class_name, score in res_per_frame:
                if (class_name in ["car", "truck"] and isInsideIntersection((x1, y1, x2, y2))):
                    centroid_x = (x1 + x2) / 2
                    centroid_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    wh_ratio = width / height
                    self.X[frame_idx] = np.array([centroid_x, centroid_y, width, height, wh_ratio])

    def decision_tree(self):
        self.pred_train = self.pred[:15649]
        self.pred_test = self.pred[15649:]
        self.X_train = self.X[:15649]
        self.X_test = self.X[15649:]
        self.Y_train = self.Y[:15649]
        self.Y_test = self.Y[15649:]
        ind_train = self.pred_train == 1
        ind_test = self.pred_test == 1
        print("training data stats: # pos: {}; # neg: {}; pos percent: {}".format(ind_train.sum(), len(self.pred_train) - ind_train.sum(), ind_train.sum() / len(self.pred_train)))
        print("test data stats: # pos: {}; # neg: {}; pos percent: {}".format(ind_test.sum(), len(self.pred_test) - ind_test.sum(), ind_test.sum() / len(self.pred_test)))

        # print(self.X_train[ind_train], self.Y_train[ind_train], self.pred_train[ind_train])
        self.clf = self.clf.fit(self.X_train[ind_train], self.Y_train[ind_train])
        # Predict on training data
        self.pred_train[ind_train] = self.clf.predict(self.X_train[ind_train])
        print("training data performance:")
        print("Accuracy:", metrics.accuracy_score(self.Y_train, self.pred_train))
        print("F1:", metrics.f1_score(self.Y_train, self.pred_train))
        print("Precision:", metrics.precision_score(self.Y_train, self.pred_train))
        print("Recall:", metrics.recall_score(self.Y_train, self.pred_train))
        # Predict on test data
        self.pred_test[ind_test] = self.clf.predict(self.X_test[ind_test])
        print("test data performance:")
        print("Accuracy:", metrics.accuracy_score(self.Y_test, self.pred_test))
        print("F1:", metrics.f1_score(self.Y_test, self.pred_test))
        print("Precision:", metrics.precision_score(self.Y_test, self.pred_test))
        print("Recall:", metrics.recall_score(self.Y_test, self.pred_test))

    def visualize_decision_tree(self):
        # Visualize decision tree
        dot_data = tree.export_graphviz(self.clf, out_file=None,
                            feature_names=["centroid_x", "centroid_y", "width", "height", "wh_ratio"],
                            class_names=["0", "1"],
                            filled=True, rounded=True,
                            special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.format = "png"
        graph.render("decision_tree")

    def write_pred(self, out_dir):
        # Create directories if not exists
        false_pos_dir = os.path.join(out_dir, "false_pos")
        false_neg_dir = os.path.join(out_dir, "false_neg")
        true_pos_dir = os.path.join(out_dir, "true_pos")
        if not os.path.exists(false_pos_dir):
            os.makedirs(false_pos_dir)
        if not os.path.exists(false_neg_dir):
            os.makedirs(false_neg_dir)
        if not os.path.exists(true_pos_dir):
            os.makedirs(true_pos_dir)
        preds = np.zeros(len(self.maskrcnn_bboxes), dtype=np.int)
        # Write out false positive and false negative results
        for image_name, res_per_frame in self.maskrcnn_bboxes.items():
            frame_idx = int(image_name.strip("frame_.jpg"))
            has_car = 0
            has_pedestrian = 0
            pred_label = 0
            true_label = 1 if frame_idx in self.pos_frames else 0
            spatial_feature = np.zeros((1, self.spatial_feature_dim), dtype=np.float64)
            for x1, y1, x2, y2, class_name, score in res_per_frame:
                if (class_name == "person" and isOverlapping(self.edge_corner_bbox, (x1, y1, x2, y2))):
                    has_pedestrian = 1
                elif (class_name in ["car", "truck"] and isInsideIntersection((x1, y1, x2, y2))):
                    has_car = 1
                    centroid_x = (x1 + x2) / 2
                    centroid_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    wh_ratio = width / height
                    spatial_feature = np.array([[centroid_x, centroid_y, width, height, wh_ratio]])
                if has_car and has_pedestrian:
                    pred_label = self.clf.predict(spatial_feature)[0]

            preds[frame_idx] = pred_label
            if (pred_label == 0 and true_label == 1):
                copyfile(os.path.join("/home/ubuntu/complex_event_video/data/car_turning_traffic2/pos", image_name), os.path.join(false_neg_dir, image_name))
            elif (pred_label == 1 and true_label == 0):
                copyfile(os.path.join("/home/ubuntu/complex_event_video/data/car_turning_traffic2/neg", image_name), os.path.join(false_pos_dir, image_name))
            elif (pred_label == 1 and true_label == 1):
                copyfile(os.path.join("/home/ubuntu/complex_event_video/data/car_turning_traffic2/pos", image_name), os.path.join(true_pos_dir, image_name))


# inter spatial relations only
# Large region
# training data performance:
# Accuracy: 0.9677295673844974
# F1: 0.5354185832566697
# Precision: 0.6395604395604395
# Recall: 0.46044303797468356
# test data performance:
# Accuracy: 0.9699357851722125
# F1: 0.485
# Precision: 0.5511363636363636
# Recall: 0.4330357142857143
# Small region
# training data performance:
# Accuracy: 0.9718831874241166
# F1: 0.6399345335515547
# Precision: 0.6627118644067796
# Recall: 0.6186708860759493
# test data performance:
# Accuracy: 0.9821949795680094
# F1: 0.7447698744769874
# Precision: 0.7007874015748031
# Recall: 0.7946428571428571

# inter and intra spatial relations
# training data performance:
# Accuracy: 0.9904147229854943
# F1: 0.8653500897666068
# Precision: 1.0
# Recall: 0.7626582278481012
# test data performance:
# Accuracy: 0.9887624051371863
# F1: 0.8126520681265206
# Precision: 0.893048128342246
# Recall: 0.7455357142857143
class SceneGraph(PedestrianAndCarWithSpatialFeatures):
    def __init__(self):
        super().__init__()
        self.spatial_feature_dim = 10
        self.X = np.zeros((len(self.maskrcnn_bboxes), self.spatial_feature_dim), dtype=np.float64)
        self.clf = tree.DecisionTreeClassifier(
            criterion="entropy",
            # class_weight="balanced",
            # min_samples_split=16
            )

    def construct_features(self):
        for image_name, res_per_frame in self.maskrcnn_bboxes.items():
            frame_idx = int(image_name.strip("frame_.jpg"))
            if self.pred[frame_idx] == 0:
                continue
            for x1, y1, x2, y2, class_name, score in res_per_frame:
                if (class_name in ["car", "truck"] and isInsideIntersection((x1, y1, x2, y2))):
                    x1_car = x1
                    y1_car = y1
                    x2_car = x2
                    y2_car = y2
                elif (class_name == "person" and isOverlapping(self.edge_corner_bbox, (x1, y1, x2, y2))):
                    x1_person = x1
                    y1_person = y1
                    x2_person = x2
                    y2_person = y2

            self.X[frame_idx] = np.array([int(x1_person > x2_car), int(x1_car > x2_person), int(y1_car > y2_person), int(y1_person > y2_car), int(x1_car < x2_person and x1_person < x2_car and y1_car < y2_person and y1_person < y2_car), (x1_car+x2_car)/2, (y1_car+y2_car)/2, x2_car - x1_car, y2_car - y1_car, (x2_car-x1_car)/(y2_car-y1_car)])

    def visualize_decision_tree(self):
        # Visualize decision tree
        dot_data = tree.export_graphviz(self.clf, out_file=None,
                            feature_names=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"],
                            class_names=["0", "1"],
                            filled=True, rounded=True,
                            special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.format = "png"
        graph.render("decision_tree_scene_graph")


@tools.tik_tok
def evaluate_model(model, save_results=False, out_dir=None):
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.0] * 3, [1 / 255.0] * 3)])
    inv_transforms = transforms.Normalize(mean = [0.] * 3, std = [255.0] * 3)
    data_dir = '/home/ubuntu/complex_event_video/data/car_turning_traffic2'
    image_dataset = datasets.ImageFolder(data_dir, data_transform)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=8, shuffle=False, num_workers=4)
    dataset_size = len(image_dataset)
    # class_names = image_dataset.classes

    running_corrects = 0
    running_tp = 0
    running_tn = 0
    running_fp = 0
    running_fn = 0

    # Iterate over data.
    for i, (inputs, labels) in enumerate(tqdm(dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        preds = model(inputs)

        # statistics
        running_corrects += torch.sum(preds == labels.data)
        running_tp += (labels.data * preds).sum()
        running_tn += ((1 - labels.data) * (1 - preds)).sum()
        running_fp += ((1 - labels.data) * preds).sum()
        running_fn += (labels.data * (1 - preds)).sum()

        if save_results:
            # Create directories if not exists
            false_pos_dir = os.path.join(out_dir, "false_pos")
            false_neg_dir = os.path.join(out_dir, "false_neg")
            true_pos_dir = os.path.join(out_dir, "true_pos")
            if not os.path.exists(false_pos_dir):
                os.makedirs(false_pos_dir)
            if not os.path.exists(false_neg_dir):
                os.makedirs(false_neg_dir)
            if not os.path.exists(true_pos_dir):
                os.makedirs(true_pos_dir)
            # Write out false positive and false negative results
            for j in range(inputs.size()[0]):
                pred_label = preds[j].item()
                true_label = labels.data[j].item()
                file_name = os.path.split(dataloader.dataset.samples[8 * i + j][0])[1]
                if (pred_label == 0 and true_label == 1):
                    save_image(inv_transforms(inputs.cpu().data[j]), os.path.join(false_neg_dir, file_name))
                elif (pred_label == 1 and true_label == 0):
                    save_image(inv_transforms(inputs.cpu().data[j]), os.path.join(false_pos_dir, file_name))
                elif (pred_label == 1 and true_label == 1):
                    save_image(inv_transforms(inputs.cpu().data[j]), os.path.join(true_pos_dir, file_name))

    epsilon = 1e-7

    precision = running_tp / (running_tp + running_fp + epsilon)
    recall = running_tp / (running_tp + running_fn + epsilon)

    f1 = 2 * (precision*recall) / (precision + recall + epsilon)

    accuracy = running_corrects.double() / dataset_size

    print('Acc: {:.4f} F1: {:.4f} Precision: {:.4f} Recall: {:.4f}'.format(accuracy, f1, precision, recall))


if __name__ == '__main__':
    # model = RandomSampling()
    # model = OnlyCar()
    # model = OnlyPedestrian()
    # model = PedestrianAndCar()
    # out_dir = "/home/ubuntu/complex_event_video/src/tests/pedestrian_and_car_preds"
    # evaluate_model(model, save_results=False)

    model = PedestrianAndCarWithSpatialFeatures()
    # model = SceneGraph()
    # out_dir = "/home/ubuntu/complex_event_video/src/tests/pedestrian_and_car_with_spatial_features_larger_intersection_preds"
    print("evaluating faster r cnn.")
    model.eval_faster_r_cnn()
    print("constructing features.")
    model.construct_features()
    print("fitting decision tree.")
    model.decision_tree()
    # model.visualize_decision_tree()
    # model.write_pred(out_dir)