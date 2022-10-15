/*
Duration(LeftOf(o0, o1), 5); Conjunction(Conjunction(Conjunction(Conjunction(Behind(o0, o2), Cyan(o2)), FrontOf(o0, o1)), RightQuadrant(o2)), Sphere(o2)); Duration(RightQuadrant(o2), 3)
*/
\timing

EXPLAIN ANALYZE CREATE TEMPORARY TABLE Obj_filtered AS
SELECT * FROM Obj_clevrer WHERE vid < 300;

CREATE INDEX idx_obj_filtered ON Obj_filtered (vid, fid);

-- g0
EXPLAIN ANALYZE CREATE TEMPORARY TABLE g0 AS SELECT o0.vid as vid, o0.fid as fid, o0.oid as o0_oid, o1.oid as o1_oid FROM Obj_filtered as o0, Obj_filtered as o1 WHERE o0.vid = o1.vid and o0.fid = o1.fid and LeftOf(o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2, o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2) = true and o0.oid <> o1.oid;
-- CREATE INDEX IF NOT EXISTS idx_g0 ON g0 (vid, fid, o0_oid, o1_oid);

EXPLAIN ANALYZE CREATE TEMPORARY TABLE g0_rownum AS (
    SELECT vid, fid, o0_oid, o1_oid,
    lead(fid, 5 - 1, 0) OVER (PARTITION BY vid, o0_oid, o1_oid ORDER BY fid) as fid_offset
    FROM g0
);

-- CREATE INDEX IF NOT EXISTS idx_g0_rownum ON g0_rownum (vid, o0_oid, o1_oid);

EXPLAIN ANALYZE CREATE TEMPORARY TABLE g0_seq AS (
    SELECT t0.vid, t0.o0_oid, t0.o1_oid, min(t0.fid_offset) AS fid
    FROM g0_rownum t0
    WHERE t0.fid_offset = t0.fid + (5 - 1)
    GROUP BY t0.vid, t0.o0_oid, t0.o1_oid
);

-- g1
EXPLAIN ANALYZE CREATE TEMPORARY TABLE g1 AS SELECT o0.vid as vid, o0.fid as fid, o0.oid as o0_oid, o1.oid as o1_oid, o2.oid as o2_oid FROM Obj_filtered as o0, Obj_filtered as o2, Obj_filtered as o1 WHERE o0.vid = o1.vid and o0.fid = o1.fid and o1.vid = o2.vid and o1.fid = o2.fid and Behind(o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2, o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true and Cyan(o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true and FrontOf(o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2, o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2) = true and RightQuadrant(o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true and Sphere(o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true and o0.oid <> o1.oid and o0.oid <> o2.oid and o1.oid <> o2.oid;
-- CREATE INDEX IF NOT EXISTS idx_g1 ON g1 (vid, fid, o0_oid, o1_oid, o2_oid);


EXPLAIN ANALYZE CREATE TEMPORARY TABLE g1_filtered AS (
    SELECT g1.vid, g1.fid, g1.o0_oid, g1.o1_oid, g1.o2_oid
    FROM g0_seq, g1
    WHERE g0_seq.vid = g1.vid AND g0_seq.o0_oid = g1.o0_oid AND g0_seq.o1_oid = g1.o1_oid AND g1.fid > g0_seq.fid
);



EXPLAIN ANALYZE CREATE TEMPORARY TABLE g1_rownum AS (
    SELECT vid, fid, o0_oid, o1_oid, o2_oid,
    lead(fid, 1 - 1, 0) OVER (PARTITION BY vid, o0_oid, o1_oid, o2_oid ORDER BY fid) as fid_offset
    FROM g1_filtered
);

-- CREATE INDEX IF NOT EXISTS idx_g0_rownum ON g0_rownum (vid, o0_oid, o1_oid);

EXPLAIN ANALYZE CREATE TEMPORARY TABLE g1_seq AS (
    SELECT t0.vid, t0.o0_oid, t0.o1_oid, t0.o2_oid, min(t0.fid_offset) AS fid
    FROM g1_rownum t0
    WHERE t0.fid_offset = t0.fid + (1 - 1)
    GROUP BY t0.vid, t0.o0_oid, t0.o1_oid, t0.o2_oid
);

-- g2
EXPLAIN ANALYZE CREATE TEMPORARY TABLE g2 AS SELECT o2.vid as vid, o2.fid as fid, o2.oid as o2_oid FROM Obj_filtered as o2 WHERE RightQuadrant(o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true;
-- CREATE INDEX IF NOT EXISTS idx_g2 ON g2 (vid, fid, o2_oid);

EXPLAIN ANALYZE CREATE TEMPORARY TABLE g2_filtered AS (
    SELECT g2.vid, g2.fid, g1_seq.o0_oid, g1_seq.o1_oid, g2.o2_oid
    FROM g1_seq, g2
    WHERE g1_seq.vid = g2.vid AND g1_seq.o2_oid = g2.o2_oid AND g2.fid > g1_seq.fid
);

EXPLAIN ANALYZE CREATE TEMPORARY TABLE g2_rownum AS (
    SELECT vid, fid, o0_oid, o1_oid, o2_oid,
    lead(fid, 3 - 1, 0) OVER (PARTITION BY vid, o0_oid, o1_oid, o2_oid ORDER BY fid) as fid_offset
    FROM g2_filtered
);

-- CREATE INDEX IF NOT EXISTS idx_g0_rownum ON g0_rownum (vid, o0_oid, o1_oid);

EXPLAIN ANALYZE CREATE TEMPORARY TABLE g2_seq AS (
    SELECT t0.vid, t0.o0_oid, t0.o1_oid, t0.o2_oid, min(t0.fid_offset) AS fid
    FROM g2_rownum t0
    WHERE t0.fid_offset = t0.fid + (3 - 1)
    GROUP BY t0.vid, t0.o0_oid, t0.o1_oid, t0.o2_oid
);

SELECT distinct vid from g2_seq order by vid;