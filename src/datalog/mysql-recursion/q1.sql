CREATE VIEW person AS
SELECT oid, fid 
FROM Obj 
WHERE cid = 0 AND fid < 140 AND x1 < 540 AND x2 > 367 AND y1 < 418 AND y2 > 345;

CREATE VIEW car AS
SELECT oid, fid 
FROM Obj 
WHERE cid = 2 AND fid < 140 AND (
    (y1 > -0.191 * x1 + 480 AND y1 > 0.295 * x1 + 261) OR 
    (y2 > -0.191 * x1 + 480 AND y2 > 0.295 * x1 + 261) OR
    (y1 > -0.191 * x2 + 480 AND y1 > 0.295 * x2 + 261) OR
    (y2 > -0.191 * x2 + 480 AND y2 > 0.295 * x2 + 261)
);

CREATE VIEW person_seq_view AS
WITH RECURSIVE person_seq (fid1, fid2) AS (
    -- base case
    SELECT oid, oid FROM person
        UNION ALL
    -- step case
    SELECT s.fid1, p.fid
    FROM person p, person_seq s
    WHERE p.fid = s.fid2 + 1
)
SELECT DISTINCT * FROM person_seq;

CREATE VIEW car_seq_view AS
WITH RECURSIVE car_seq (fid1, fid2) AS (
    -- base case
    SELECT oid, oid FROM car
        UNION ALL
    -- step case
    SELECT s.fid1, c.fid
    FROM car c, car_seq s
    WHERE c.fid = s.fid2 + 1
)
SELECT DISTINCT * FROM car_seq;

CREATE VIEW q AS
SELECT p.fid1 AS fid1, p.fid2 AS fid2, c.fid1 AS fid3, c.fid2 AS fid4
FROM person_seq_view p, car_seq_view c
WHERE p.fid2 < c.fid1 AND p.fid2 - p.fid1 > 30 AND c.fid2 - c.fid1 > 30 AND c.fid2 - p.fid1 < 100;

CREATE VIEW q_neg AS 
SELECT q1.fid1, q1.fid2, q1.fid3, q1.fid4
FROM q q1, q q2 
WHERE q2.fid1 < q1.fid1 AND q1.fid2 < q2.fid2 AND q2.fid3 < q1.fid3 AND q1.fid4 < q2.fid4;

CREATE TEMPORARY TABLE q_filtered
SELECT * FROM q 
WHERE NOT EXISTS (
    SELECT 1 FROM q_neg n
    WHERE q.fid1 = n.fid1 AND q.fid2 = n.fid2
    AND q.fid3 = n.fid3 AND q.fid4 = n.fid4);

-- ERROR 1053 (08S01): Server shutdown in progress