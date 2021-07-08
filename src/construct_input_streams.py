import os, cv2
from tqdm import tqdm
import mysql.connector
import random 
from random import sample

random.seed(10)

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

def isInsideIntersection(box):
    # box: x1, y1, x2, y2
    xmin, ymin, xmax, ymax = box
    centroid_x = (xmin + xmax) / 2
    centroid_y = (ymin + ymax) / 2 

    x0, y0, x1, y1, x2, y2 = 0, 480, 450, 394, 782, 492

    return centroid_y > (y0 - y1) * centroid_x / (x0 - x1) + (x0 * y1 - x1 * y0) / (x0 - x1) and centroid_y > (y1 - y2) * centroid_x / (x1 - x2) + (x1 * y2 - x2 * y1) / (x1 - x2)


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


def construct_input_streams_car_turning(connection, video_fn):
    # Query: Car turning right. 
    # Heuristic: object detection for car, and bounding box overlaps a specific region. 
    
    # intersection = (900, 1650, 2153, 1832)
    intersection = (140, 470, 720, 500)
    count = 0
    car_stream = []
    
    # car_stream: (start_time, end_time, x1, y1, x2, y2, frame)
    cursor = connection.cursor()
    
    # Construct car stream: some car turning in the intersection
    cursor.execute("SELECT e.start_time, e.end_time, v.x1, v.x2, v.y1, v.y2, v.frame_id FROM Car c, Event e, VisibleAt v WHERE e.event_id = c.event_id AND e.event_id = v.event_id AND v.filename = %s", [video_fn])
    for row in cursor:
        start_time, end_time, x1, x2, y1, y2, frame_id = row
        car_box = (x1, y1, x2, y2)
        if isOverlapping(intersection, car_box):
            car_stream.append(frame_id)
    
    connection.commit()
    cursor.close()
    return car_stream


def construct_input_streams_car_turning_neg(connection, video_fn):
    # Query: Car turning right. 
    # Heuristic: object detection for car, and bounding box overlaps a specific region. 
    
    # intersection = (900, 1650, 2153, 1832)
    intersection = (140, 470, 720, 500)
    count = 0
    car_stream = []
    
    # car_stream: (start_time, end_time, x1, y1, x2, y2, frame)
    cursor = connection.cursor()

    # Construct car stream: no car turning in the intersection
    cursor.execute("SELECT e.start_time, e.end_time, v.x1, v.x2, v.y1, v.y2, v.frame_id FROM Car c, Event e, VisibleAt v WHERE e.event_id = c.event_id AND e.event_id = v.event_id AND v.filename = %s ORDER BY v.frame_id", [video_fn])
    current_frame_id = 0
    results = cursor.fetchall()
    for row in results:
        if row[6] >= current_frame_id:
            if row[6] > current_frame_id:
                # Start a new frame
                for i in range(current_frame_id, row[6], 10):
                    car_stream.append(i)
                current_frame_id = row[6]
            start_time, end_time, x1, x2, y1, y2, frame_id = row
            car_box = (x1, y1, x2, y2)
            if isOverlapping(intersection, car_box):
                # Current frame contains turning car, skip it. 
                current_frame_id += 10
        # Else, the examined frame already contains turning car; skip it
    car_stream = sample(car_stream, 25)
    
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


def construct_input_streams_motorbike_crossing(connection, idx):
    # Query: Motorbike crossing in the intersection. 
    # Heuristic: object detection for bike, and bounding box overlaps a specific region. 
    
    # intersection = (103, 416, 750, 540)
    count = 0
    motorbike_stream = []
    
    # motorbike_stream: (start_time, end_time, x1, y1, x2, y2, frame)
    cursor = connection.cursor()
    
    # Construct bike stream
    cursor.execute("SELECT e.start_time, e.end_time, v.x1, v.x2, v.y1, v.y2, v.frame_id FROM Event e, VisibleAt v WHERE e.event_type in ('motorbike', 'bicycle') AND e.event_id = v.event_id AND v.filename = 'traffic-%s.mp4'", [idx])
    for row in cursor:
        start_time, end_time, x1, x2, y1, y2, frame_id = row
        motorbike_box = (x1, y1, x2, y2)
        if isInsideIntersection(motorbike_box):
            motorbike_stream.append(frame_id)
    
    connection.commit()
    cursor.close()
    return motorbike_stream


def construct_input_streams_motorbike_crossing_neg(connection, idx):
    # Query: Motorbike crossing in the intersection. 
    # Heuristic: object detection for bike, and bounding box overlaps a specific region. 
    
    count = 0
    motorbike_stream = []
    
    # motorbike_stream: (start_time, end_time, x1, y1, x2, y2, frame)
    cursor = connection.cursor()

    # Construct bike stream
    cursor.execute("SELECT e.start_time, e.end_time, v.x1, v.x2, v.y1, v.y2, v.frame_id FROM Event e, VisibleAt v WHERE e.event_type in ('motorbike', 'bicycle') AND e.event_id = v.event_id AND v.filename = 'traffic-%s.mp4' ORDER BY v.frame_id", [idx])
    current_frame_id = 0
    results = cursor.fetchall()
    for row in results:
        if row[6] >= current_frame_id:
            if row[6] > current_frame_id:
                # Start a new frame
                for i in range(current_frame_id, row[6], 10):
                    motorbike_stream.append(i)
                current_frame_id = row[6]
            start_time, end_time, x1, x2, y1, y2, frame_id = row
            motorbike_box = (x1, y1, x2, y2)
            if isInsideIntersection(motorbike_box):
                # Current frame contains crossing bike, skip it. 
                current_frame_id += 10
        # Else, the examined frame already contains crossing bike; skip it
    
    cursor.execute("SELECT MAX(v.frame_id) FROM VisibleAt v WHERE v.filename = 'traffic-%s.mp4'", [idx])
    row = cursor.fetchone()
    motorbike_stream += [*range(current_frame_id, row[0], 10)]
    motorbike_stream = sample(motorbike_stream, 8)
    
    connection.commit()
    cursor.close()
    return motorbike_stream


def construct_input_streams_avg_cars(connection, idx):
    # Query: Average number of cars in window of videos
    
    count = 0
    input_stream = []
    
    # input_stream: (start_time, end_time, x1, y1, x2, y2, frame)
    cursor = connection.cursor()

    # Construct input stream
    cursor.execute("""
        SELECT v.frame_id 
        FROM VisibleAt v 
        LEFT JOIN Event e 
        ON e.event_id = v.event_id 
        WHERE e.event_type = 'car' AND v.filename = 'traffic-%s.mp4' 
        GROUP BY v.frame_id
        HAVING COUNT(*) <= 4
        ORDER BY v.frame_id
        """, [idx])
    for row in cursor:
        input_stream.append(row[0])
    
    if len(input_stream) > 50:
        input_stream = sample(input_stream, 50)
    
    connection.commit()
    cursor.close()
    return input_stream