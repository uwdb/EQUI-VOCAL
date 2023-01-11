-- "Duration(Conjunction(LeftOf(o0, o1), Shape_sphere(o1)), 15); Conjunction(FrontOf(o0, o2), Near_1(o0, o2)); Duration(Conjunction(Behind(o0, o2), Far_3(o0, o2)), 5)"

\timing on

CREATE TEMPORARY TABLE Obj_filtered AS
SELECT * FROM Obj_clevrer;

CREATE INDEX idx_obj_filtered
ON Obj_filtered (vid, fid);

CREATE TEMPORARY TABLE g0 AS SELECT o0.vid as vid, o0.fid as fid, o0.oid as o0_oid, o1.oid as o1_oid FROM Obj_filtered as o0, Obj_filtered as o1 WHERE o0.vid = o1.vid and o0.fid = o1.fid and LeftOf(o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2, o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2) = true and Shape('sphere', o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2) = true and o0.oid <> o1.oid;

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

CREATE TEMPORARY TABLE g1 AS SELECT o0.vid as vid, o0.fid as fid, o0.oid as o0_oid, o2.oid as o2_oid FROM Obj_filtered as o0, Obj_filtered as o2 WHERE o0.vid = o2.vid and o0.fid = o2.fid and FrontOf(o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2, o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true and Near(1.0, o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2, o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true and o0.oid <> o2.oid;

CREATE TEMPORARY TABLE g1_filtered AS (
    SELECT t0.vid, t1.fid, t0.o0_oid, t0.o1_oid, t1.o2_oid
    FROM g0_contiguous t0, g1 t1
    WHERE t0.vid = t1.vid AND t0.o0_oid = t1.o0_oid and t0.o0_oid <> t1.o2_oid and t0.o1_oid <> t1.o2_oid AND t0.fid < t1.fid
);


CREATE TEMPORARY TABLE g1_windowed AS (
    SELECT vid, fid, o0_oid, o1_oid, o2_oid,
    lead(fid, 1 - 1, 0) OVER (PARTITION BY vid, o0_oid, o1_oid, o2_oid ORDER BY fid) as fid_offset
    FROM g1_filtered
);


CREATE TEMPORARY TABLE g1_contiguous AS (
    SELECT vid, o0_oid, o1_oid, o2_oid, min(fid_offset) AS fid
    FROM g1_windowed
    WHERE fid_offset = fid + (1 - 1)
    GROUP BY vid, o0_oid, o1_oid, o2_oid
);

CREATE TEMPORARY TABLE g2 AS SELECT o0.vid as vid, o0.fid as fid, o0.oid as o0_oid, o2.oid as o2_oid FROM Obj_filtered as o0, Obj_filtered as o2 WHERE o0.vid = o2.vid and o0.fid = o2.fid and Behind(o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2, o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true and Far(3.0, o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2, o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true and o0.oid <> o2.oid;

CREATE TEMPORARY TABLE g2_filtered AS (
    SELECT t0.vid, t1.fid, t0.o0_oid, t0.o1_oid, t0.o2_oid
    FROM g1_contiguous t0, g2 t1
    WHERE t0.vid = t1.vid AND t0.o0_oid = t1.o0_oid and t0.o2_oid = t1.o2_oid AND t0.fid < t1.fid
);


CREATE TEMPORARY TABLE g2_windowed AS (
    SELECT vid, fid, o0_oid, o1_oid, o2_oid,
    lead(fid, 5 - 1, 0) OVER (PARTITION BY vid, o0_oid, o1_oid, o2_oid ORDER BY fid) as fid_offset
    FROM g2_filtered
);


CREATE TEMPORARY TABLE g2_contiguous AS (
    SELECT vid, o0_oid, o1_oid, o2_oid, min(fid_offset) AS fid
    FROM g2_windowed
    WHERE fid_offset = fid + (5 - 1)
    GROUP BY vid, o0_oid, o1_oid, o2_oid
);

SELECT distinct vid from g2_contiguous;