# Compositional Query Processing

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
pass: 123456abcABC
# Use database
use complex_event;
# Drop tables
source /home/ubuntu/CSE544-project/rekall/tutorials/dropTable.sql;
# Create tables
source /home/ubuntu/CSE544-project/rekall/tutorials/createTable.sql;

## Visual Road: interesting events
# traffic-4k-002.mp4
1. person at the pavement edge corner, then car turning right in the intersection
2. Same car reappears in the video:
    orange, size >= 40000, min_dist = 20: 9:00 - 9:37
    red, size >= 20000, min_dist = 20: 8:28 - 9:22, 9:24 - 10:09, 10:09 - 10:57, 10:58 - 12:04, 12:07 - 13:13
3. three motorcycles in a row: 6:00
4. three people overlap: start_time: 386, 228.5, 53

# car-pov-2k-000-shortened.mp4
1. 3 consecutive red lights: 3:21 - 4:46
2. Changing lanes in the intersection: 4:48
3. Tunnel: 4:50 - 5:20
    2x3 partitioned. first row and second column is dark (normally, it's sky).
4. Blocking intersection: 6:24, 8:28
    many cars right below the red lights.

# Visual Road
build failed:
- Fix: https://github.com/carla-simulator/carla/issues/2664
open carla/Util/BuildTools/Setup.sh and replace

wget "https://dl.bintray.com/boostorg/release/${BOOST_VERSION}/source/${BOOST_PACKAGE_BASENAME}.tar.gz"
with
wget "https://sourceforge.net/projects/boost/files/boost/1.72.0/boost_1_72_0.tar.gz"

it is a temporary fix. it worked for me.

# Souffle
- install-dir: /home/ubuntu/complex_event_video/src/datalog/souffle/install/bin/souffle

- User-defined functors:
```
g++ functors.cpp -c -fPIC -o functors.o
g++ -shared -o libfunctors.so functors.o
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}`pwd`
```

# Postgresql
- sudo -u postgres createuser -s $(whoami);
- To run queries in postgres, type:
    * psql complex_event
- Run .sql file:
    * psql -f xxx.sql complex_event
- User-defined functors:
```
cc -I /usr/include/postgresql/10/server -fpic -c functors.c
cc -shared -o functors.so functors.o
```

# Frame Number Overlay With FFmpeg
ffmpeg -i /home/ubuntu/complex_event_video/data/bdd100k/videos/test/cafae648-c3d4e13a.mov -vf "drawtext=fontfile=Arial.ttf: text='%{frame_num}': start_number=0: x=5: y=5: fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5" -c:a copy output.mov

# Downloading MEVA files
- Downloading pattern matched entries from S3 bucket:
```
aws s3 sync s3://mevadata-public-01/drops-123-r13 . --no-sign-request --exclude "*" --include "*school.G336*"
```
- Potential interesting events: person stands up, person sits down, person rides bicycle, person embraces person.