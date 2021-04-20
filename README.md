# CSE544-project

## Detectron2
- Model Output Format: https://detectron2.readthedocs.io/en/latest/tutorials/models.html

    1. pred_boxes
    2. scores
    3. pred_classes
    4. pred_masks
    5. pred_keypoints

## ReKall

## Connecting to MySQL
mysql -h database-1.cld3cb8o2zkf.us-east-1.rds.amazonaws.com -P 3306 -u admin -p
# Drop tables
source /home/ubuntu/CSE544-project/rekall/tutorials/dropTable.sql;
# Create tables
source /home/ubuntu/CSE544-project/rekall/tutorials/createTable.sql;

## Visual Road: interesting events
# car-pov-2k-000-shortened.mp4
1. 3 consecutive red lights: 3:21 - 4:46
2. Changing lanes in the intersection: 4:48 
3. Tunnel: 4:50 - 5:20 
    2x3 partitioned. first row and second column is dark (normally, it's sky).
4. Blocking intersection: 6:24, 8:28
    many cars right below the red lights. 

# traffic-4k-002.mp4
1. three motorcycles in a row: 6:00 
2. three people overlap: start_time: 386, 228.5, 53