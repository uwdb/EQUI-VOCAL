\timing

DROP VIEW IF EXISTS person, car, person_seq_view, person_seq_a, person_seq_filtered, car_seq_view, car_seq_a, car_seq_filtered;
DROP FUNCTION IF EXISTS get_iou;
DROP INDEX IF EXISTS idx_obj, idx_person, idx_person_spatial;

CREATE INDEX idx_obj
ON Obj (fid);

-- CREATE INDEX idx_obj_spatial
-- ON Obj USING btree_gist (x1, y1, x2, y2);

CREATE FUNCTION get_iou(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS double precision
    AS '/home/ubuntu/complex_event_video/src/datalog/postgres-recursion/functors', 'get_iou'
    LANGUAGE C STRICT;

CREATE TEMPORARY TABLE person AS
SELECT fid, x1, y1, x2, y2
FROM Obj
WHERE cid = 0 AND x1 < 540 AND x2 > 367 AND y1 < 418 AND y2 > 345;

CREATE TEMPORARY TABLE car AS
SELECT fid, x1, y1, x2, y2
FROM Obj
WHERE cid = 2 AND (
    (y1 > -0.191 * x1 + 480 AND y1 > 0.295 * x1 + 261) OR
    (y2 > -0.191 * x1 + 480 AND y2 > 0.295 * x1 + 261) OR
    (y1 > -0.191 * x2 + 480 AND y1 > 0.295 * x2 + 261) OR
    (y2 > -0.191 * x2 + 480 AND y2 > 0.295 * x2 + 261)
);

CREATE INDEX idx_person
ON person (fid);

-- EXPLAIN ANALYZE

-- BEGIN;
-- SET LOCAL enable_mergejoin TO false;
EXPLAIN ANALYZE CREATE TEMPORARY TABLE person_seq_view AS
WITH RECURSIVE person_seq (fid1, fid2, x1, y1, x2, y2) AS (
    -- base case
    SELECT fid, fid, x1, y1, x2, y2 FROM person
        UNION ALL
    -- step case
    SELECT s.fid1, p.fid, p.x1, p.y1, p.x2, p.y2
    FROM person_seq s, person p
    -- WHERE p.fid = s.fid2 + 1 AND get_iou(s.x1, s.y1, s.x2, s.y2, p.x1, p.y1, p.x2, p.y2) > 0.5
    WHERE p.fid = s.fid2 + 1
)
SELECT DISTINCT fid1, fid2 FROM person_seq WHERE fid2 - fid1 > 30;
-- COMMIT;

CREATE TEMPORARY TABLE person_seq_a AS
SELECT fid1, max(fid2) AS fid2
FROM person_seq_view
GROUP BY fid1;

CREATE TEMPORARY TABLE person_seq_filtered AS
SELECT min(fid1) AS fid1, fid2
FROM person_seq_a
GROUP BY fid2;

EXPLAIN ANALYZE CREATE TEMPORARY TABLE car_seq_view AS
WITH RECURSIVE car_seq (fid1, fid2, x1, y1, x2, y2) AS (
    -- base case
    SELECT fid, fid, x1, y1, x2, y2 FROM car
        UNION ALL
    -- step case
    SELECT s.fid1, p.fid, p.x1, p.y1, p.x2, p.y2
    FROM car_seq s, car p
    -- WHERE p.fid = s.fid2 + 1 AND get_iou(s.x1, s.y1, s.x2, s.y2, p.x1, p.y1, p.x2, p.y2) > 0.5
    WHERE p.fid = s.fid2 + 1
)
SELECT DISTINCT fid1, fid2 FROM car_seq WHERE fid2 - fid1 > 30;

CREATE TEMPORARY TABLE car_seq_a AS
SELECT fid1, max(fid2) AS fid2
FROM car_seq_view
GROUP BY fid1;

CREATE TEMPORARY TABLE car_seq_filtered AS
SELECT min(fid1) AS fid1, fid2
FROM car_seq_a
GROUP BY fid2;


CREATE TEMPORARY TABLE q AS
SELECT p.fid1 AS fid1, p.fid2 AS fid2, c.fid1 AS fid3, c.fid2 AS fid4
FROM person_seq_filtered p, car_seq_filtered c
WHERE p.fid1 < c.fid1 AND c.fid1 - p.fid1 < 100;

-- select * from q;