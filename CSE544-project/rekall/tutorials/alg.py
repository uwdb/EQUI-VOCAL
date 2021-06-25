Input: min_duration: minimal duration constraint for event A, frame_list: list of all video frames
Output: bboxes

frame_detection_list = [-1 for _ in range(len(frame_list))] # -1: hasn't invoked model; 0: doesn't contain object; 1: contains object of interest; 2: doesn't contain object of interest (Skip)

# Phase 1
while min_duration > 0:
    for i in range(0, len(frame_list), min_duration):
        if frame_detection_list[i] == -1:
            outputs = predictor(frame_list[i])
            if has_event_a(outputs):
                frame_detection_list[i] = 1
            else:
                frame_detection_list[i] = 0

            bboxes[i] = outputs
    
    # If two frames within 30 frames apart don't contain object of interest, then we can mark any frames in between as 0 as well. 
    for i in range(0, len(frame_detection_list) - min_duration, min_duration):
        if frame_detection_list[i] % 2 == 0 and frame_detection_list[i+min_duration] % 2 == 0 and frame_detection_list[i+1] % 2 != 0:
            for j in range(i+1, i+min_duration):
                frame_detection_list[j] = 2
    min_duration = int(min_duration / 2)

# Done for truck. Next, we still need to run model on frames that haven't been detected but may contain cars, but only those within the window constraint.
# At this point, elements in frame_detection_list can be either 0, 1, or 2 (cannot be -1).
print("Phase 2...")
for i in range(len(frame_detection_list)):
    if frame_detection_list[i] == 1:
        for j in range(i+1, min(i+301, len(frame_detection_list))): 
            if frame_detection_list[j] == 2:
                outputs = predictor(frame_list[j])
                frame_detection_list[j] = 0
                bboxes[i] = outputs



def binary_search(start_frame, end_frame, ML_model, min_duration, event_a=True):
    if min_duration == 0: 
        return 
    for i in range(start_frame, end_frame, min_duration):
        if (event_a and frame_detection_list[i][0] != -1) or (event_a==False and frame_detection_list2[i][0] != -1):
            continue
        prediction, pred_boxes, scores, pred_classes = ML_model(frame_list[i])
        
        # Update F
        if event_a:
            frame_detection_list[i] = [int(prediction == True), pred_classes]
        else:
            frame_detection_list2[i] = [int(prediction == True), pred_classes]
    
        # Write results to bboxes list 
        res_per_frame = []
        for pred_box, score, pred_class in zip(pred_boxes, scores, pred_classes):
            res_per_frame.append([pred_box[0].item(), pred_box[1].item(), pred_box[2].item(), pred_box[3].item(), coco_names[pred_class.item()], score.item(), str(i + 1).zfill(10) + '.jpg'])
        res_per_video[i] = res_per_frame
    
    # This pass is done.
    # Update F by examining if there are any frames that can be skipped.
    for i in range(start_frame, end_frame - min_duration, min_duration):
        if event_a:
            if frame_detection_list[i][0] % 2 == 0 and frame_detection_list[i+min_duration][0] % 2 == 0 and frame_detection_list[i+1][0] % 2 != 0:
                for j in range(i+1, i+min_duration):
                    if frame_detection_list[j][0] == -1:
                        frame_detection_list[j] = [2, []]
        else:
            if frame_detection_list2[i][0] % 2 == 0 and frame_detection_list2[i+min_duration][0] % 2 == 0 and frame_detection_list2[i+1][0] % 2 != 0:
                for j in range(i+1, i+min_duration):
                    if frame_detection_list2[j][0] == -1:
                        frame_detection_list2[j] = [2, []]
    # Go to the next level of recursion.
    binary_search(start_frame, end_frame, ML_model, int(min_duration / 2), event_a)

# Phase 1
binary_search(0, N, model_a, min_duration_a)

for i in range(N):
    if frame_detection_list[i][0] != 2:
        has_car = False
        pred_classes = frame_detection_list[i][1]
        for pred_class in pred_classes:
            if coco_names[pred_class.item()] == "car":
                has_car = True
                break
        frame_detection_list2[i] = [int(has_car == True), pred_classes]

print("Phase 2...")
idx = 0
while idx < N:
    if frame_detection_list[idx][0] == 1:
        event_a_count = 1
        event_a_start = idx 
        while idx+event_a_count < N and frame_detection_list[idx+event_a_count][0] == 1:
            event_a_count += 1
        if event_a_count >= 30:
            binary_search(idx, min(idx+301, N), model_b, min_duration_b, False)
        idx += event_a_count
    idx += 1