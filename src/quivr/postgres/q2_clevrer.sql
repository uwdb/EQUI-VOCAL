/*
Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Near_1.05), True*), Conjunction(LeftOf, BackOf)), True*), Duration(Conjunction(TopQuadrant, Far_0.9), 5)), True*)
*/
\timing

-- DROP FUNCTION IF EXISTS Near, Far, LeftOf, BackOf, RightQuadrant, TopQuadrant;

-- CREATE FUNCTION Near(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean
--     AS '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'Near'
--     LANGUAGE C STRICT;

-- CREATE FUNCTION Far(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean
--     AS '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'Far'
--     LANGUAGE C STRICT;

-- CREATE FUNCTION LeftOf(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean
--     AS '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'LeftOf'
--     LANGUAGE C STRICT;

-- CREATE FUNCTION Behind(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean
--     AS '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'Behind'
--     LANGUAGE C STRICT;

-- CREATE FUNCTION RightQuadrant(double precision, double precision, double precision, double precision) RETURNS boolean
--     AS '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'RightQuadrant'
--     LANGUAGE C STRICT;

-- CREATE FUNCTION TopQuadrant(double precision, double precision, double precision, double precision) RETURNS boolean
--     AS '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'TopQuadrant'
--     LANGUAGE C STRICT;

CREATE TEMPORARY TABLE Obj_filtered AS
SELECT * FROM Obj_trajectories WHERE vid = 1;

CREATE INDEX idx_obj_filtered
ON Obj_filtered (vid, fid);

CREATE TEMPORARY TABLE g1 AS
SELECT a.oid as oid1, b.oid as oid2, a.vid, a.fid
FROM Obj_filtered a, Obj_filtered b
WHERE a.vid = b.vid and a.fid = b.fid and LeftOf(a.x1, a.y1, a.x2, a.y2, b.x1, b.y1, b.x2, b.y2) = true and a.oid = 0 and b.oid = 1;

CREATE TEMPORARY TABLE g2 AS
SELECT a.oid as oid1, b.oid as oid2, a.vid, a.fid
FROM Obj_filtered a, Obj_filtered b
WHERE a.vid = b.vid and a.fid = b.fid and Near(1.05, a.x1, a.y1, a.x2, a.y2, b.x1, b.y1, b.x2, b.y2) = true and a.oid = 0 and b.oid = 1;

CREATE INDEX idx_g1
ON g1 (vid, oid1, oid2, fid);

CREATE INDEX idx_g2
ON g2 (vid, oid1, oid2, fid);

-- EXPLAIN ANALYZE

-- BEGIN;
-- SET LOCAL enable_mergejoin TO false;
CREATE TEMPORARY TABLE g1_seq_view AS
WITH RECURSIVE g1_seq (vid, fid1, fid2, oid1, oid2) AS (
    -- base case
    SELECT vid, fid, fid, oid1, oid2 FROM g1
        UNION ALL
    -- step case
    SELECT s.vid, s.fid1, g.fid, s.oid1, s.oid2
    FROM g1_seq s, g1 g
    WHERE s.vid = g.vid AND s.oid1 = g.oid1 AND s.oid2 = g.oid2 AND g.fid = s.fid2 + 1
)
SELECT vid, max(fid1) AS fid1, fid2, oid1, oid2
FROM (
    SELECT vid, fid1, min(fid2) AS fid2, oid1, oid2 FROM g1_seq
    WHERE fid2 - fid1 + 1 >= duration_constraint
    GROUP BY vid, fid1, oid1, oid2
) AS t
GROUP BY vid, fid2, oid1, oid2;

SELECT COUNT(*) FROM g1_seq_view;

CREATE TEMPORARY TABLE g2_seq_view AS
WITH RECURSIVE g2_seq (vid, fid1, fid2, oid1, oid2) AS (
    -- base case
    SELECT vid, fid, fid, oid1, oid2 FROM g2
        UNION ALL
    -- step case
    SELECT s.vid, s.fid1, g.fid, s.oid1, s.oid2
    FROM g2_seq s, g2 g
    WHERE s.vid = g.vid AND s.oid1 = g.oid1 AND s.oid2 = g.oid2 AND g.fid = s.fid2 + 1
)
SELECT vid, max(fid1) AS fid1, fid2, oid1, oid2
FROM (
    SELECT vid, fid1, min(fid2) AS fid2, oid1, oid2 FROM g2_seq
    WHERE fid2 - fid1 + 1 >= 1
    GROUP BY vid, fid1, oid1, oid2
) AS t
GROUP BY vid, fid2, oid1, oid2;

SELECT COUNT(*) FROM g2_seq_view;

CREATE INDEX idx_g1_seq_view
ON g1_seq_view (vid, oid1, oid2, fid1, fid2);

CREATE INDEX idx_g2_seq_view
ON g2_seq_view (vid, oid1, oid2, fid1, fid2);

CREATE TEMPORARY TABLE q1 AS
SELECT DISTINCT g1.vid as vid, g1.fid1 as fid1, g2.fid2 as fid2, g1.oid1 as oid1, g1.oid2 as oid2
FROM g1_seq_view g1, g2_seq_view g2
WHERE g1.vid = g2.vid AND g1.oid1 = g2.oid1 AND g1.oid2 = g2.oid2 AND g1.fid2 < g2.fid1;

SELECT count(distinct vid) from q1;

-- \copy (SELECT * FROM q) TO '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/q1_clevrer.csv' (format csv);
-- select * from q;