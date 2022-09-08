/*
person trajectory (> 30 frames), followed by car trajectory (> 30), window size < 100.
*/
\timing

DROP FUNCTION IF EXISTS get_iou;
DROP INDEX IF EXISTS idx_obj_bdd, idx_person_bdd, idx_car_bdd, idx_car_seq_filtered, idx_car_seq_b, idx_car_seq_view, idx_person_seq_filtered;

CREATE INDEX idx_obj_bdd
ON Obj_bdd (vid, fid);

-- CREATE INDEX idx_obj_spatial
-- ON Obj USING btree_gist (x1, y1, x2, y2);

CREATE FUNCTION get_iou(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS double precision
    AS '/home/ubuntu/complex_event_video/src/datalog/postgres-recursion/functors', 'get_iou'
    LANGUAGE C STRICT;

CREATE TEMPORARY TABLE person AS
SELECT oid, vid, fid, x1, y1, x2, y2
FROM Obj_bdd
WHERE cid = 0;

CREATE TEMPORARY TABLE car AS
SELECT oid, vid, fid, x1, y1, x2, y2
FROM Obj_bdd
WHERE cid = 2;

CREATE INDEX idx_person_bdd
ON person (vid, fid);

CREATE INDEX idx_car_bdd
ON car (vid, fid);

-- EXPLAIN ANALYZE

-- BEGIN;
-- SET LOCAL enable_mergejoin TO false;
CREATE TEMPORARY TABLE person_seq_view AS
WITH RECURSIVE person_seq (vid, fid1, fid2, oid1, oid2, x1, y1, x2, y2) AS (
    -- base case
    SELECT vid, fid, fid, oid, oid, x1, y1, x2, y2 FROM person
        UNION ALL
    -- step case
    SELECT s.vid, s.fid1, p.fid, s.oid1, p.oid, p.x1, p.y1, p.x2, p.y2
    FROM person_seq s, person p
    WHERE s.vid = p.vid AND p.fid = s.fid2 + 1 AND get_iou(s.x1, s.y1, s.x2, s.y2, p.x1, p.y1, p.x2, p.y2) > 0.8
    -- WHERE p.fid = s.fid2 + 1
)
SELECT DISTINCT vid, fid1, fid2, oid1, oid2 FROM person_seq WHERE fid2 - fid1 > 30;
-- COMMIT;

CREATE TEMPORARY TABLE person_seq_a AS
SELECT vid, fid1, max(fid2) AS fid2
FROM person_seq_view
GROUP BY vid, fid1;

CREATE TEMPORARY TABLE person_seq_b AS
SELECT vid, min(fid1) AS fid1, fid2
FROM person_seq_a
GROUP BY vid, fid2;

CREATE TEMPORARY TABLE person_seq_filtered AS
SELECT p.vid, p.fid1, p.fid2, p.oid1, p.oid2
FROM person_seq_view p, person_seq_b pb
WHERE p.vid = pb.vid AND p.fid1 = pb.fid1 AND p.fid2 = pb.fid2;

CREATE INDEX idx_person_seq_filtered
ON person_seq_filtered (vid, fid1);

CREATE TEMPORARY TABLE car_seq_view AS
WITH RECURSIVE car_seq (vid, fid1, fid2, oid1, oid2, x1, y1, x2, y2) AS (
    -- base case
    SELECT vid, fid, fid, oid, oid, x1, y1, x2, y2 FROM car
        UNION ALL
    -- step case
    SELECT s.vid, s.fid1, p.fid, s.oid1, p.oid, p.x1, p.y1, p.x2, p.y2
    FROM car_seq s, car p
    WHERE s.vid = p.vid AND p.fid = s.fid2 + 1 AND get_iou(s.x1, s.y1, s.x2, s.y2, p.x1, p.y1, p.x2, p.y2) > 0.8
    -- WHERE p.fid = s.fid2 + 1
)
SELECT DISTINCT vid, fid1, fid2, oid1, oid2 FROM car_seq WHERE fid2 - fid1 > 30;
-- COMMIT;

CREATE INDEX idx_car_seq_view
ON car_seq_view (vid, fid1, fid2);

CREATE TEMPORARY TABLE car_seq_a AS
SELECT vid, fid1, max(fid2) AS fid2
FROM car_seq_view
GROUP BY vid, fid1;

CREATE TEMPORARY TABLE car_seq_b AS
SELECT vid, min(fid1) AS fid1, fid2
FROM car_seq_a
GROUP BY vid, fid2;

CREATE INDEX idx_car_seq_b
ON car_seq_b (vid, fid1, fid2);

CREATE TEMPORARY TABLE car_seq_filtered AS
SELECT p.vid, p.fid1, p.fid2, p.oid1, p.oid2
FROM car_seq_view p, car_seq_b pb
WHERE p.vid = pb.vid AND p.fid1 = pb.fid1 AND p.fid2 = pb.fid2;

CREATE INDEX idx_car_seq_filtered
ON car_seq_filtered (vid, fid1);

CREATE TEMPORARY TABLE q AS
SELECT p.vid, p.fid1 AS fid1, p.fid2 AS fid2, c.fid1 AS fid3, c.fid2 AS fid4, o1.x1 AS x11, o1.y1 AS y11, o1.x2 AS x21, o1.y2 AS y21, o2.x1 AS x12, o2.y1 AS y12, o2.x2 AS x22, o2.y2 AS y22, o3.x1 AS x13, o3.y1 AS y13, o3.x2 AS x23, o3.y2 AS y23, o4.x1 AS x14, o4.y1 AS y14, o4.x2 AS x24, o4.y2 AS y24
FROM person_seq_filtered p, car_seq_filtered c, Obj_bdd o1, Obj_bdd o2, Obj_bdd o3, Obj_bdd o4
WHERE p.vid = c.vid AND p.fid1 < c.fid1 AND c.fid1 - p.fid1 < 100 AND p.oid1 = o1.oid AND p.oid2 = o2.oid AND c.oid1 = o3.oid AND c.oid2 = o4.oid;

\copy (SELECT * FROM q) TO '/home/ubuntu/complex_event_video/src/datalog/data/postgres_output/q1_bdd.csv' (format csv);
-- select * from q;