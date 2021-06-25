import os
import shutil

def main():
    data_root = "/z/wenjiah/ImageNet/train/"

    classes = ['n01818515', 'n02123045', 'n02981792', 'n03100240', 'n03594945', 'n03642806', 'n03769881', 'n03770679', 'n03791053', 'n03792782', 'n03977966', 'n03868863', 'n03902125', 'n03930630', 'n04399382', 'n04456115', 'n06794110', 'n06874185', 'n07583066', 'n07614500', 'n07753592']

    sementic_labels = ['macaw', 'tabby', 'catamaran', 'convertible', 'jeep', 'laptop', 'minibus', 'minivan', 'scooter', 'mountain_bike', 'police_van', 'oxygen_mask', 'pay-phone', 'pickup', 'teddy', 'torch', 'street_sign', 'traffic_light', 'guacamole', 'icecream', 'banana']

    for i, pos_class in enumerate(classes):
        pos_img_src = os.path.join(data_root, pos_class)
        class_name = sementic_labels[i]
        
        tr_img_pos_dst = f"/z/analytics/ImageNet/data/{class_name}/train/pos"
        val_img_pos_dst = f"/z/analytics/ImageNet/data/{class_name}/val/pos"
        if not os.path.exists(tr_img_pos_dst):
            os.makedirs(tr_img_pos_dst)
        if not os.path.exists(val_img_pos_dst):
            os.makedirs(val_img_pos_dst)

        tr_count = 0

        # Copy positive examples for training.
        for img_name in os.listdir(pos_img_src):
            if img_name.endswith('.JPEG'):
                img_path = os.path.join(pos_img_src, img_name)
                if tr_count < 1000:
                    shutil.copy(img_path, tr_img_pos_dst)
                    tr_count += 1
                else:
                    shutil.copy(img_path, val_img_pos_dst)
                    tr_count += 1
                #shutil.copy(img_path, tr_img_pos_dst)
                # tr_count += 1
        
        print("-------------------------------------------------------")
        print(f"Class: {pos_class}")
        print(f"    Positive training examples count: {tr_count}")

        tr_img_neg_dst = f"/z/analytics/ImageNet/data/{class_name}/train/neg"
        val_img_neg_dst = f"/z/analytics/ImageNet/data/{class_name}/val/neg"
        if not os.path.exists(tr_img_neg_dst):
            os.makedirs(tr_img_neg_dst)
        if not os.path.exists(val_img_neg_dst):
            os.makedirs(val_img_neg_dst)
        
        # Copy negative examples for training.
        neg_train_count = 0

        for neg_class in classes:
            if neg_class != pos_class:
                # print(neg_class)
                neg_count_per_class = 0
                neg_img_src = os.path.join(data_root, neg_class)

                for img_name in os.listdir(neg_img_src):
                    if img_name.endswith('.JPEG'):
                        img_path = os.path.join(neg_img_src, img_name)
                        if neg_count_per_class < 50:
                            shutil.copy(img_path, tr_img_neg_dst)
                            neg_count_per_class += 1
                        else:
                            shutil.copy(img_path, val_img_neg_dst)
                            neg_count_per_class += 1
                        # shutil.copy(img_path, tr_img_neg_dst)
                        # neg_count_per_class += 1
                        neg_train_count += 1
                
                    if neg_count_per_class >= 65:
                        break

        print(f"    Negative training examples count: {neg_train_count}")
        print("-------------------------------------------------------")


if __name__ == '__main__':
    main()