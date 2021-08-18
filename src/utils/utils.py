def isInsideIntersection(box):
    def point_is_inside_intersection(x, y):
        x0, y0, x1, y1, x2, y2 = 0, 480, 450, 394, 782, 492 # Large region
        # x0, y0, x1, y1, x2, y2 = 0, 516, 445, 414, 744, 532 # Small region
        return y > (y0 - y1) * x / (x0 - x1) + (x0 * y1 - x1 * y0) / (x0 - x1) and y > (y1 - y2) * x / (x1 - x2) + (x1 * y2 - x2 * y1) / (x1 - x2)

    # box: x1, y1, x2, y2
    x1, y1, x2, y2 = box
    return point_is_inside_intersection(x1, y1) or point_is_inside_intersection(x1, y2) or point_is_inside_intersection(x2, y1) or point_is_inside_intersection(x2, y2)


def isOverlapping(box1, box2):
    # box: x1, y1, x2, y2
    x1min, y1min, x1max, y1max = box1
    x2min, y2min, x2max, y2max = box2
    return x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max


def frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break