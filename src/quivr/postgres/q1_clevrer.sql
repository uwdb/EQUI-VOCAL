/*
Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Near_1.05), True*), Conjunction(LeftOf, Behind)), True*), Duration(Conjunction(TopQuadrant, Far_0.9), 5)), True*)
*/
\timing

-- DROP FUNCTION IF EXISTS Near, Far, LeftOf, BackOf, RightQuadrant, TopQuadrant;
-- DROP INDEX IF EXISTS idx_obj_filtered, idx_g1, idx_g2, idx_g3, idx_g1_seq_view, idx_g2_seq_view, idx_g3_seq_view, idx_q1;

-- CREATE FUNCTION Near(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean
--     AS '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'near'
--     LANGUAGE C STRICT;

-- CREATE FUNCTION Far(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean
--     AS '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'far'
--     LANGUAGE C STRICT;

-- CREATE FUNCTION LeftOf(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean
--     AS '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'left_of'
--     LANGUAGE C STRICT;

-- CREATE FUNCTION BackOf(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean
--     AS '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'behind'
--     LANGUAGE C STRICT;

-- CREATE FUNCTION RightQuadrant(double precision, double precision, double precision, double precision) RETURNS boolean
--     AS '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'right_quadrant'
--     LANGUAGE C STRICT;

-- CREATE FUNCTION TopQuadrant(double precision, double precision, double precision, double precision) RETURNS boolean
--     AS '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'top_quadrant'
--     LANGUAGE C STRICT;

CREATE TEMPORARY TABLE Obj_filtered AS
SELECT * FROM Obj_trajectories;

CREATE INDEX idx_obj_filtered
ON Obj_filtered (vid, fid);

CREATE TEMPORARY TABLE g1 AS
SELECT a.oid as oid1, b.oid as oid2, a.vid, a.fid
FROM Obj_filtered a, Obj_filtered b
WHERE a.vid = b.vid and a.fid = b.fid and Near(1.05, a.x1, a.y1, a.x2, a.y2, b.x1, b.y1, b.x2, b.y2) = true and a.oid = 0 and b.oid = 1;

CREATE INDEX idx_g1
ON g1 (vid, oid1, oid2, fid);

CREATE TEMPORARY TABLE g1_seq_view AS
WITH RECURSIVE g1_seq (vid, fid1, fid2, oid1, oid2) AS (
    -- base case
    SELECT vid, fid, fid, oid1, oid2 FROM g1
        UNION ALL
    -- step case
    SELECT s.vid, s.fid1, g.fid, s.oid1, s.oid2
    FROM g1_seq s, g1 g
    WHERE s.vid = g.vid AND s.oid1 = g.oid1 AND s.oid2 = g.oid2 AND g.fid = s.fid2 + 1 AND g.fid - s.fid1 + 1 <= 1
)
SELECT DISTINCT * FROM g1_seq WHERE fid2 - fid1 + 1 = 1;

CREATE INDEX idx_g1_seq_view
ON g1_seq_view (vid, fid1, fid2, oid1, oid2);

CREATE TEMPORARY TABLE g2 AS
SELECT a.oid as oid1, b.oid as oid2, a.vid, a.fid
FROM Obj_filtered a, Obj_filtered b
WHERE a.vid = b.vid and a.fid = b.fid and LeftOf(a.x1, a.y1, a.x2, a.y2, b.x1, b.y1, b.x2, b.y2) = true and Behind(a.x1, a.y1, a.x2, a.y2, b.x1, b.y1, b.x2, b.y2) = true and a.oid = 0 and b.oid = 1;

CREATE INDEX idx_g2
ON g2 (vid, oid1, oid2, fid);

CREATE TEMPORARY TABLE g2_seq_view AS
WITH RECURSIVE g2_seq (vid, fid1, fid2, oid1, oid2) AS (
    -- base case
    SELECT vid, fid, fid, oid1, oid2 FROM g2
        UNION ALL
    -- step case
    SELECT s.vid, s.fid1, g.fid, s.oid1, s.oid2
    FROM g2_seq s, g2 g
    WHERE s.vid = g.vid AND s.oid1 = g.oid1 AND s.oid2 = g.oid2 AND g.fid = s.fid2 + 1 AND g.fid - s.fid1 + 1 <= 1
)
SELECT DISTINCT * FROM g2_seq WHERE fid2 - fid1 + 1 = 1;

CREATE INDEX idx_g2_seq_view
ON g2_seq_view (vid, fid1, fid2, oid1, oid2);

CREATE TEMPORARY TABLE g3 AS
SELECT a.oid as oid1, b.oid as oid2, a.vid, a.fid
FROM Obj_filtered a, Obj_filtered b
WHERE a.vid = b.vid and a.fid = b.fid and TopQuadrant(a.x1, a.y1, a.x2, a.y2) = true and Far(0.9, a.x1, a.y1, a.x2, a.y2, b.x1, b.y1, b.x2, b.y2) = true and a.oid = 0 and b.oid = 1;

CREATE INDEX idx_g3
ON g3 (vid, oid1, oid2, fid);

CREATE TEMPORARY TABLE g3_seq_view AS
WITH RECURSIVE g3_seq (vid, fid1, fid2, oid1, oid2) AS (
    -- base case
    SELECT vid, fid, fid, oid1, oid2 FROM g3
        UNION ALL
    -- step case
    SELECT s.vid, s.fid1, g.fid, s.oid1, s.oid2
    FROM g3_seq s, g3 g
    WHERE s.vid = g.vid AND s.oid1 = g.oid1 AND s.oid2 = g.oid2 AND g.fid = s.fid2 + 1 AND g.fid - s.fid1 + 1 <= 5
)
SELECT DISTINCT * FROM g3_seq WHERE fid2 - fid1 + 1 = 5;

CREATE INDEX idx_g3_seq_view
ON g3_seq_view (vid, fid1, fid2, oid1, oid2);

CREATE TEMPORARY TABLE q1 AS
SELECT g1.vid as vid, min(g2.fid2) as fid, g1.oid1 as oid1, g1.oid2 as oid2
FROM g1_seq_view g1, g2_seq_view g2
WHERE g1.vid = g2.vid AND g1.oid1 = g2.oid1 AND g1.oid2 = g2.oid2 AND g1.fid2 < g2.fid1
GROUP BY g1.vid, g1.oid1, g1.oid2;

CREATE INDEX idx_q1
ON q1 (vid, fid, oid1, oid2);

CREATE TEMPORARY TABLE q2 AS
SELECT g1.vid as vid, min(g2.fid2) as fid, g1.oid1 as oid1, g1.oid2 as oid2
FROM q1 g1, g3_seq_view g2
WHERE g1.vid = g2.vid AND g1.oid1 = g2.oid1 AND g1.oid2 = g2.oid2 AND g1.fid < g2.fid1
GROUP BY g1.vid, g1.oid1, g1.oid2;

SELECT count(distinct vid) from q2;
