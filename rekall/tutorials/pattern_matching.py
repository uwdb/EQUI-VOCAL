import pandas as pd 


# Temporal predicates
def before(intrvl1, intrvl2, min_dist=0, max_dist="infty"):
    return intrvl1[0] + min_dist <= intrvl2[0] and (intrvl1[0] + max_dist >= intrvl2[0] or max_dist == "infty")


def isOverlapping(box1, box2):
    # box: x1, y1, x2, y2
    x1min, y1min, x1max, y1max = box1
    x2min, y2min, x2max, y2max = box2
    return x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max


def pattern_matching_before_within_5s(person_stream, car_stream):
    out_stream = []
    # Person, followed by car, within 5 seconds
    for intrvl1 in person_stream:
        for intrvl2 in car_stream:
            if before(intrvl1, intrvl2, max_dist=5):
                out_stream.append((intrvl1[0], intrvl2[1], intrvl1[2], intrvl1[3], intrvl1[4], intrvl1[5], intrvl1[6], intrvl2[2], intrvl2[3], intrvl2[4], intrvl2[5], intrvl2[6]))
    return out_stream

def pattern_matching_three_objects_overlap(motorbike_stream):
    out_stream = []
    df = pd.DataFrame(motorbike_stream, columns=['start_time', 'end_time', 'x1', 'y1', 'x2', 'y2', 'frame_id'])
    
    grouped = df.groupby(by=["start_time"])

    for group_name, df_group in grouped:
    # for start_time, grouped_list in d.items():
        length = df_group.shape[0]
        for i in range(length):
            for j in range(i + 1, length):
                for k in range(j + 1, length):
                    e1 = df_group.iloc[[i], :]
                    e2 = df_group.iloc[[j], :]
                    e3 = df_group.iloc[[k], :]
                    box1 = (e1.iloc[0]['x1'], e1.iloc[0]['y1'], e1.iloc[0]['x2'], e1.iloc[0]['y2'])
                    box2 = (e2.iloc[0]['x1'], e2.iloc[0]['y1'], e2.iloc[0]['x2'], e2.iloc[0]['y2'])
                    box3 = (e3.iloc[0]['x1'], e3.iloc[0]['y1'], e3.iloc[0]['x2'], e3.iloc[0]['y2'])
                    if (isOverlapping(box1, box2) and isOverlapping(box1, box3)) \
                    or (isOverlapping(box2, box1) and isOverlapping(box2, box3)) \
                    or (isOverlapping(box3, box1) and isOverlapping(box3, box2)):
                        out_stream.append((e1.iloc[0]['start_time'], e1.iloc[0]['end_time'], e1.iloc[0]['x1'], e1.iloc[0]['y1'], e1.iloc[0]['x2'], e1.iloc[0]['y2'], e2.iloc[0]['x1'], e2.iloc[0]['y1'], e2.iloc[0]['x2'], e2.iloc[0]['y2'], e3.iloc[0]['x1'], e3.iloc[0]['y1'], e3.iloc[0]['x2'], e3.iloc[0]['y2'], e1.iloc[0]['frame_id']))
    return out_stream
