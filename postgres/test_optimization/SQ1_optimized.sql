-- "Duration(Conjunction(Far_3(o0, o1), RightQuadrant(o1)), 15); Conjunction(FrontOf(o0, o1), Near_1(o0, o1)); Duration(Conjunction(Far_3(o0, o1), RightQuadrant(o1)), 5)"

\timing on

CREATE TEMPORARY TABLE Obj_filtered AS
SELECT * FROM Obj_clevrer;

CREATE INDEX idx_obj_filtered
ON Obj_filtered (vid, fid);

CREATE TEMPORARY TABLE g0 AS SELECT o0.vid as vid, o0.fid as fid, o0.oid as o0_oid, o1.oid as o1_oid FROM Obj_filtered as o0, Obj_filtered as o1 WHERE o0.vid = o1.vid and o0.fid = o1.fid and Far(3.0, o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2, o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2) = true and RightQuadrant(o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2) = true and o0.oid <> o1.oid;

CREATE TEMPORARY TABLE g0_windowed AS (
    SELECT vid, fid, o0_oid, o1_oid,
    lead(fid, 15 - 1, 0) OVER (PARTITION BY vid, o0_oid, o1_oid ORDER BY fid) as fid_offset
    FROM g0
);


CREATE TEMPORARY TABLE g0_contiguous AS (
    SELECT vid, o0_oid, o1_oid, min(fid_offset) AS fid
    FROM g0_windowed
    WHERE fid_offset = fid + (15 - 1)
    GROUP BY vid, o0_oid, o1_oid
);

CREATE TEMPORARY TABLE g1 AS SELECT o0.vid as vid, o0.fid as fid, o0.oid as o0_oid, o1.oid as o1_oid FROM Obj_filtered as o0, Obj_filtered as o1 WHERE o0.vid = o1.vid and o0.fid = o1.fid and FrontOf(o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2, o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2) = true and Near(1.0, o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2, o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2) = true and o0.oid <> o1.oid;

CREATE TEMPORARY TABLE g1_filtered AS (
    SELECT t0.vid, t1.fid, t0.o0_oid, t0.o1_oid
    FROM g0_contiguous t0, g1 t1
    WHERE t0.vid = t1.vid AND t0.o0_oid = t1.o0_oid and t0.o1_oid = t1.o1_oid AND t0.fid < t1.fid
);


CREATE TEMPORARY TABLE g1_windowed AS (
    SELECT vid, fid, o0_oid, o1_oid,
    lead(fid, 1 - 1, 0) OVER (PARTITION BY vid, o0_oid, o1_oid ORDER BY fid) as fid_offset
    FROM g1_filtered
);


CREATE TEMPORARY TABLE g1_contiguous AS (
    SELECT vid, o0_oid, o1_oid, min(fid_offset) AS fid
    FROM g1_windowed
    WHERE fid_offset = fid + (1 - 1)
    GROUP BY vid, o0_oid, o1_oid
);

CREATE TEMPORARY TABLE g2 AS SELECT o0.vid as vid, o0.fid as fid, o0.oid as o0_oid, o1.oid as o1_oid FROM Obj_filtered as o0, Obj_filtered as o1 WHERE o0.vid = o1.vid and o0.fid = o1.fid and Far(3.0, o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2, o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2) = true and RightQuadrant(o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2) = true and o0.oid <> o1.oid;

CREATE TEMPORARY TABLE g2_filtered AS (
    SELECT t0.vid, t1.fid, t0.o0_oid, t0.o1_oid
    FROM g1_contiguous t0, g2 t1
    WHERE t0.vid = t1.vid AND t0.o0_oid = t1.o0_oid and t0.o1_oid = t1.o1_oid AND t0.fid < t1.fid
);


CREATE TEMPORARY TABLE g2_windowed AS (
    SELECT vid, fid, o0_oid, o1_oid,
    lead(fid, 5 - 1, 0) OVER (PARTITION BY vid, o0_oid, o1_oid ORDER BY fid) as fid_offset
    FROM g2_filtered
);


CREATE TEMPORARY TABLE g2_contiguous AS (
    SELECT vid, o0_oid, o1_oid, min(fid_offset) AS fid
    FROM g2_windowed
    WHERE fid_offset = fid + (5 - 1)
    GROUP BY vid, o0_oid, o1_oid
);

SELECT distinct vid from g2_contiguous;