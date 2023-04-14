/*
Duration(Conjunction(Near_1.05(o0, o1), RightOf(o1, o2)), 3); Duration(Conjunction(Conjunction(Cyan(o1), LeftQuadrant(o0)), Metal(o0)), 2); Duration(Conjunction(LeftOf(o0, o2), TopQuadrant(o1)), 4)
*/
\timing

EXPLAIN ANALYZE CREATE TEMPORARY TABLE Obj_filtered AS
SELECT * FROM Obj_clevrer WHERE vid < 300;

CREATE INDEX idx_obj_filtered
ON Obj_filtered (vid, fid);

-- g0
CREATE TEMPORARY TABLE g0 AS SELECT o0.vid as vid, o0.fid as fid, o0.oid as o0_oid, o1.oid as o1_oid, o2.oid as o2_oid FROM Obj_filtered as o1, Obj_filtered as o0, Obj_filtered as o2 WHERE o0.vid = o1.vid and o0.fid = o1.fid and o1.vid = o2.vid and o1.fid = o2.fid and Near(1.05, o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2, o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2) = true and RightOf(o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2, o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true and o0.oid <> o1.oid and o0.oid <> o2.oid and o1.oid <> o2.oid;

-- CREATE INDEX IF NOT EXISTS idx_g0 ON g0 (vid);

CREATE TEMPORARY TABLE g0_windowed AS (
    SELECT vid, fid, o0_oid, o1_oid, o2_oid,
    lead(fid, 3 - 1, 0) OVER (PARTITION BY vid, o0_oid, o1_oid, o2_oid ORDER BY fid) as fid_offset
    FROM g0
);

CREATE TEMPORARY TABLE g0_contiguous AS (
    SELECT t0.vid, t0.o0_oid, t0.o1_oid, t0.o2_oid, min(t0.fid_offset) AS fid
    FROM g0_windowed t0
    WHERE t0.fid_offset = t0.fid + (3 - 1)
    GROUP BY t0.vid, t0.o0_oid, t0.o1_oid, t0.o2_oid
);

-- CREATE INDEX IF NOT EXISTS idx_g0_contiguous ON g0_contiguous (vid);

-- g1
CREATE TEMPORARY TABLE g1 AS SELECT o0.vid as vid, o0.fid as fid, o0.oid as o0_oid, o1.oid as o1_oid FROM Obj_filtered as o1, Obj_filtered as o0 WHERE o0.vid = o1.vid and o0.fid = o1.fid and Cyan(o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2) = true and LeftQuadrant(o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2) = true and Metal(o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2) = true and o0.oid <> o1.oid;

-- CREATE INDEX IF NOT EXISTS idx_g1 ON g1 (vid);

CREATE TEMPORARY TABLE g1_filtered AS (
    SELECT t0.vid, t1.fid, t0.o0_oid, t0.o1_oid, t0.o2_oid
    FROM g0_contiguous t0, g1 t1
    WHERE t0.vid = t1.vid AND t0.o0_oid = t1.o0_oid AND t0.o1_oid = t1.o1_oid AND t0.fid < t1.fid
);

CREATE TEMPORARY TABLE g1_windowed AS (
    SELECT vid, fid, o0_oid, o1_oid, o2_oid,
    lead(fid, 2 - 1, 0) OVER (PARTITION BY vid, o0_oid, o1_oid, o2_oid ORDER BY fid) as fid_offset
    FROM g1_filtered
);

CREATE TEMPORARY TABLE g1_contiguous AS (
    SELECT t0.vid, t0.o0_oid, t0.o1_oid, t0.o2_oid, min(t0.fid_offset) AS fid
    FROM g1_windowed t0
    WHERE t0.fid_offset = t0.fid + (2 - 1)
    GROUP BY t0.vid, t0.o0_oid, t0.o1_oid, t0.o2_oid
);

-- CREATE INDEX IF NOT EXISTS idx_g1_contiguous ON g1_contiguous (vid);

-- g2
CREATE TEMPORARY TABLE g2 AS SELECT o0.vid as vid, o0.fid as fid, o0.oid as o0_oid, o1.oid as o1_oid, o2.oid as o2_oid FROM Obj_filtered as o1, Obj_filtered as o0, Obj_filtered as o2 WHERE o0.vid = o1.vid and o0.fid = o1.fid and o1.vid = o2.vid and o1.fid = o2.fid and LeftOf(o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2, o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true and TopQuadrant(o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2) = true and o0.oid <> o1.oid and o0.oid <> o2.oid and o1.oid <> o2.oid;

-- CREATE INDEX IF NOT EXISTS idx_g2 ON g2 (vid);

CREATE TEMPORARY TABLE g2_filtered AS (
    SELECT t0.vid, t1.fid, t0.o0_oid, t0.o1_oid, t0.o2_oid
    FROM g1_contiguous t0, g2 t1
    WHERE t0.vid = t1.vid AND t0.o0_oid = t1.o0_oid AND t0.o1_oid = t1.o1_oid AND t0.o2_oid = t1.o2_oid AND t0.fid < t1.fid
);

CREATE TEMPORARY TABLE g2_windowed AS (
    SELECT vid, fid, o0_oid, o1_oid, o2_oid,
    lead(fid, 4 - 1, 0) OVER (PARTITION BY vid, o0_oid, o1_oid, o2_oid ORDER BY fid) as fid_offset
    FROM g2_filtered
);

CREATE TEMPORARY TABLE g2_contiguous AS (
    SELECT t0.vid, t0.o0_oid, t0.o1_oid, t0.o2_oid, min(t0.fid_offset) AS fid
    FROM g2_windowed t0
    WHERE t0.fid_offset = t0.fid + (4 - 1)
    GROUP BY t0.vid, t0.o0_oid, t0.o1_oid, t0.o2_oid
);

-- CREATE INDEX IF NOT EXISTS idx_g2_contiguous ON g2_contiguous (vid);

SELECT distinct vid from g2_contiguous;

-- \copy (SELECT * FROM q) TO '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/outputs/q1_clevrer.csv' (format csv);
-- select * from q;


-- Results
-- 13
-- [92, 280, 101, 296, 4, 46, 286, 224, 146, 289, 84, 27, 293]
-- time 9.154475688934326