import os, cv2
from tqdm import tqdm
import mysql.connector


def frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break

def isOverlapping(box1, box2):
    # box: x1, y1, x2, y2
    x1min, y1min, x1max, y1max = box1
    x2min, y2min, x2max, y2max = box2
    return x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max

def construct_input_streams_person_female_upwhite_then_car_white_size_ge_30000(connection):
    person_stream = []
    car_stream = []
    
    frames = []
    display_video_list = ["cabc30fc-e7726578"]
    input_video_dir = "/home/ubuntu/CSE544-project/data/bdd100k/videos/test/"

    for file in os.listdir(input_video_dir):
        if os.path.splitext(file)[1] != '.mp4' and os.path.splitext(file)[1] != '.mov':
            continue
        if os.path.splitext(file)[0] not in display_video_list:
            continue

        video = cv2.VideoCapture(os.path.join(input_video_dir, file))
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)

        if not video.isOpened():
            print("Error opening video stream or file: ", file)
        else:
            frame_gen = frame_from_video(video)
            for frame in tqdm(frame_gen, total=num_frames):
                frames.append(frame)
    
    # person_stream: (start_time, end_time, x1, y1, x2, y2, frame)
    # car_stream: (start_time, end_time, x1, y1, x2, y2, frame)
    cursor = connection.cursor()
    
    # Construct person stream 
    # Female and upwhite
    cursor.execute("SELECT e.start_time, e.end_time, v.x1, v.x2, v.y1, v.y2, v.frame_id FROM Person p, Event e, VisibleAt v WHERE e.event_id = p.event_id AND e.event_id = v.event_id AND p.female = true AND p.upwhite = true")
    for row in cursor:
        start_time, end_time, x1, x2, y1, y2, frame_id = row
        frame = frames[frame_id]
        person_stream.append((start_time, end_time, x1, y1, x2, y2, frame))

    # Construct car stream
    # White and size >= 30000
    cursor.execute("SELECT e.start_time, e.end_time, v.x1, v.x2, v.y1, v.y2, v.frame_id FROM Car c, Event e, VisibleAt v WHERE e.event_id = c.event_id AND e.event_id = v.event_id AND c.color = 'white' AND c.size >= 30000")
    for row in cursor:
        start_time, end_time, x1, x2, y1, y2, frame_id = row
        frame = frames[frame_id]
        car_stream.append((start_time, end_time, x1, y1, x2, y2, frame))

    connection.commit()
    cursor.close()
    return person_stream, car_stream


def construct_input_streams_watch_out_person_cross_road_when_car_turn_left(connection):
    # Query: a person is detected at the pavement edge corner (who might cross the road), then within 5 seconds, a car is turning left in the intersection.
    
    edge_corner = (2145, 1365, 2146, 1366)
    intersection = (1680, 1500, 1681, 1501)

    person_stream = []
    car_stream = []
    
    # person_stream: (start_time, end_time, x1, y1, x2, y2, frame)
    # car_stream: (start_time, end_time, x1, y1, x2, y2, frame)
    cursor = connection.cursor()
    
    # Construct person stream 
    cursor.execute("SELECT e.start_time, e.end_time, v.x1, v.x2, v.y1, v.y2, v.frame_id FROM Person p, Event e, VisibleAt v WHERE e.event_id = p.event_id AND e.event_id = v.event_id AND v.filename = 'traffic-4k-002.mp4'")
    for row in cursor:
        start_time, end_time, x1, x2, y1, y2, frame_id = row
        person_box = (x1, y1, x2, y2)
        if isOverlapping(edge_corner, person_box):
            person_stream.append((start_time, end_time, x1, y1, x2, y2, frame_id))

    # Construct car stream
    cursor.execute("SELECT e.start_time, e.end_time, v.x1, v.x2, v.y1, v.y2, v.frame_id FROM Car c, Event e, VisibleAt v WHERE e.event_id = c.event_id AND e.event_id = v.event_id AND v.filename = 'traffic-4k-002.mp4'")
    for row in cursor:
        start_time, end_time, x1, x2, y1, y2, frame_id = row
        car_box = (x1, y1, x2, y2)
        if isOverlapping(intersection, car_box):
            car_stream.append((start_time, end_time, x1, y1, x2, y2, frame_id))

    connection.commit()
    cursor.close()
    return person_stream, car_stream


def construct_input_streams_three_motorbikes_in_a_row(connection):
    # Query: three motorbikes in a row

    motorbike_stream = []
    
    # motorbike_stream: (start_time, end_time, x1, y1, x2, y2, frame_id)
    cursor = connection.cursor()
    
    # Construct motorbike stream 
    cursor.execute("SELECT e.start_time, e.end_time, v.x1, v.x2, v.y1, v.y2, v.frame_id FROM Event e, VisibleAt v WHERE e.event_id = v.event_id AND v.filename = 'traffic-4k-002.mp4' AND e.event_type = 'motorbike' ORDER BY e.start_time")
    for row in cursor:
        start_time, end_time, x1, x2, y1, y2, frame_id = row
        motorbike_stream.append((start_time, end_time, x1, y1, x2, y2, frame_id))

    connection.commit()
    cursor.close()
    return motorbike_stream


def construct_input_streams_same_car_reappears(connection):
    # Query: Same car (orange) reappears in the video

    car_stream = []
    
    # car_stream: (start_time, end_time, x1, y1, x2, y2, frame_id)
    cursor = connection.cursor()
    
    # Construct car stream 
    cursor.execute("SELECT e.start_time, e.end_time, v.x1, v.x2, v.y1, v.y2, v.frame_id FROM Event e, VisibleAt v, Car c WHERE c.event_id = e.event_id AND e.event_id = v.event_id AND v.filename = 'traffic-4k-002.mp4' AND c.color = 'orange' AND c.size >= 40000 ORDER BY e.start_time")

    for row in cursor:
        start_time, end_time, x1, x2, y1, y2, frame_id = row
        car_stream.append((start_time, end_time, x1, y1, x2, y2, frame_id))

    connection.commit()
    cursor.close()
    return car_stream


def construct_input_streams_car_turning_right(connection):
    # Query: Car turning right. 
    # Heuristic: object detection for car, and bounding box overlaps a specific region. 
    
    # intersection = (900, 1650, 2153, 1832)
    intersection = (140, 470, 720, 500)
    count = 0
    car_stream = []
    
    # car_stream: (start_time, end_time, x1, y1, x2, y2, frame)
    cursor = connection.cursor()
    
    # Construct car stream
    cursor.execute("SELECT e.start_time, e.end_time, v.x1, v.x2, v.y1, v.y2, v.frame_id FROM Car c, Event e, VisibleAt v WHERE e.event_id = c.event_id AND e.event_id = v.event_id AND v.filename = 'traffic-1.mp4'")
    for row in cursor:
        start_time, end_time, x1, x2, y1, y2, frame_id = row
        car_box = (x1, y1, x2, y2)
        if isOverlapping(intersection, car_box):
            car_stream.append((start_time, end_time, x1, y1, x2, y2, frame_id))

        # if not isOverlapping(intersection, car_box):
        #     count += 1
        #     if count % 60 == 0:
        #         car_stream.append((start_time, end_time, x1, y1, x2, y2, frame_id))

    connection.commit()
    cursor.close()
    return car_stream


def construct_input_streams_person_edge_corner(connection):
    # Query: Person at edge corner.
    # Heuristic: object detection for person, and bounding box overlaps a specific region. 
    
    # intersection = (1554, 1299, 2400, 1583)
    intersection = (384, 355, 515, 400)
    count = 0
    count_neg = 0
    person_stream = []
    
    # car_stream: (start_time, end_time, x1, y1, x2, y2, frame)
    cursor = connection.cursor()
    
    # Construct person stream
    # cursor.execute("SELECT e.start_time, e.end_time, v.x1, v.x2, v.y1, v.y2, v.frame_id FROM Person p, Event e, VisibleAt v WHERE e.event_id = p.event_id AND e.event_id = v.event_id AND v.filename = 'traffic-4k-002.mp4' AND e.start_time > 600")
    cursor.execute("SELECT e.start_time, e.end_time, v.x1, v.x2, v.y1, v.y2, v.frame_id FROM Person p, Event e, VisibleAt v WHERE e.event_id = p.event_id AND e.event_id = v.event_id AND v.filename = 'traffic-20.mp4' AND e.start_time < 600")
    for row in cursor:
        start_time, end_time, x1, x2, y1, y2, frame_id = row
        person_box = (x1, y1, x2, y2)
        # if isOverlapping(intersection, person_box):
        #     # count += 1
        #     # if count % 5 == 0:
        #     person_stream.append((start_time, end_time, x1, y1, x2, y2, frame_id))
        if not isOverlapping(intersection, person_box):
            count += 1
            if count % 60 == 0:
                person_stream.append((start_time, end_time, x1, y1, x2, y2, frame_id))

    connection.commit()
    cursor.close()
    return person_stream

    # select count(*) from Person p, Event e, VisibleAt v WHERE e.event_id = p.event_id AND e.event_id = v.event_id AND v.filename = 'traffic-5.mp4' AND e.start_time < 600;
    # delete from VisibleAt v where v.filename = 'traffic-3.mp4';  