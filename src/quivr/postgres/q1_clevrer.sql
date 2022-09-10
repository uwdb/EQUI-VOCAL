/*
Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(Sequencing(True*, Near_1.05), True*), Conjunction(LeftOf, BackOf)), True*), Duration(Conjunction(TopQuadrant, Far_0.9), 5)), True*)
*/
\timing

DROP FUNCTION IF EXISTS Near, Far, LeftOf, BackOf, RightQuadrant, TopQuadrant;
DROP INDEX IF EXISTS idx_obj_filtered, idx_g1, idx_g2, idx_g3, idx_g1_seq_view, idx_g2_seq_view, idx_g3_seq_view, idx_q1;

CREATE FUNCTION Near(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean
    AS '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'near'
    LANGUAGE C STRICT;

CREATE FUNCTION Far(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean
    AS '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'far'
    LANGUAGE C STRICT;

CREATE FUNCTION LeftOf(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean
    AS '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'left_of'
    LANGUAGE C STRICT;

CREATE FUNCTION BackOf(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) RETURNS boolean
    AS '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'behind'
    LANGUAGE C STRICT;

CREATE FUNCTION RightQuadrant(double precision, double precision, double precision, double precision) RETURNS boolean
    AS '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'right_quadrant'
    LANGUAGE C STRICT;

CREATE FUNCTION TopQuadrant(double precision, double precision, double precision, double precision) RETURNS boolean
    AS '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/functors', 'top_quadrant'
    LANGUAGE C STRICT;

CREATE TEMPORARY TABLE Obj_filtered AS
SELECT * FROM Obj_trajectories WHERE vid in generate_series(0, 299);

CREATE INDEX idx_obj_filtered
ON Obj_filtered (vid, fid);

CREATE TEMPORARY TABLE g1 AS
SELECT a.oid as oid1, b.oid as oid2, a.vid, a.fid
FROM Obj_filtered a, Obj_filtered b
WHERE a.vid = b.vid and a.fid = b.fid and near(a.x1, a.y1, a.x2, a.y2, b.x1, b.y1, b.x2, b.y2) = true and a.oid = 0 and b.oid = 1;

CREATE TEMPORARY TABLE g2 AS
SELECT a.oid as oid1, b.oid as oid2, a.vid, a.fid
FROM Obj_filtered a, Obj_filtered b
WHERE a.vid = b.vid and a.fid = b.fid and left_of(a.x1, a.y1, a.x2, a.y2, b.x1, b.y1, b.x2, b.y2) = true and behind(a.x1, a.y1, a.x2, a.y2, b.x1, b.y1, b.x2, b.y2) = true and a.oid = 0 and b.oid = 1;

CREATE TEMPORARY TABLE g3 AS
SELECT a.oid as oid1, b.oid as oid2, a.vid, a.fid
FROM Obj_filtered a, Obj_filtered b
WHERE a.vid = b.vid and a.fid = b.fid and top_quadrant(a.x1, a.y1, a.x2, a.y2) = true and far(a.x1, a.y1, a.x2, a.y2, b.x1, b.y1, b.x2, b.y2) = true and a.oid = 0 and b.oid = 1;

CREATE INDEX idx_g1
ON g1 (vid, oid1, oid2, fid);

CREATE INDEX idx_g2
ON g2 (vid, oid1, oid2, fid);

CREATE INDEX idx_g3
ON g3 (vid, oid1, oid2, fid);

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
SELECT DISTINCT vid, fid1, fid2, oid1, oid2 FROM g1_seq;

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
SELECT DISTINCT vid, fid1, fid2, oid1, oid2 FROM g2_seq;

CREATE TEMPORARY TABLE g3_seq_view AS
WITH RECURSIVE g3_seq (vid, fid1, fid2, oid1, oid2) AS (
    -- base case
    SELECT vid, fid, fid, oid1, oid2 FROM g3
        UNION ALL
    -- step case
    SELECT s.vid, s.fid1, g.fid, s.oid1, s.oid2
    FROM g3_seq s, g3 g
    WHERE s.vid = g.vid AND s.oid1 = g.oid1 AND s.oid2 = g.oid2 AND g.fid = s.fid2 + 1
)
SELECT DISTINCT vid, fid1, fid2, oid1, oid2 FROM g3_seq WHERE fid2 - fid1 + 1 >= 5;


CREATE INDEX idx_g1_seq_view
ON g1_seq_view (vid, fid1, fid2, oid1, oid2);

CREATE INDEX idx_g2_seq_view
ON g2_seq_view (vid, fid1, fid2, oid1, oid2);

CREATE INDEX idx_g3_seq_view
ON g3_seq_view (vid, fid1, fid2, oid1, oid2);

CREATE TEMPORARY TABLE q1 AS
SELECT DISTINCT g1.vid as vid, g1.fid1 as fid1, g2.fid2 as fid2, g1.oid1 as oid1, g1.oid2 as oid2
FROM g1_seq_view g1, g2_seq_view g2
WHERE g1.vid = g2.vid AND g1.oid1 = g2.oid1 AND g1.oid2 = g2.oid2 AND g1.fid2 < g2.fid1;

CREATE INDEX idx_q1
ON q1 (vid, fid1, fid2, oid1, oid2);

CREATE TEMPORARY TABLE q AS
SELECT DISTINCT g1.vid as vid, g1.fid1 as fid1, g2.fid2 as fid2, g1.oid1 as oid1, g1.oid2 as oid2
FROM q1 as g1, g3_seq_view g2
WHERE g1.vid = g2.vid AND g1.oid1 = g2.oid1 AND g1.oid2 = g2.oid2 AND g1.fid2 < g2.fid1;

-- SELECT distinct vid from q;

-- \copy (SELECT * FROM q) TO '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/q1_clevrer.csv' (format csv);
-- select * from q;