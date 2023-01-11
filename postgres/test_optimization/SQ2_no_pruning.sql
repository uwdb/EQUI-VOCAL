-- "Conjunction(Conjunction(Color_cyan(o0), Far_3(o1, o2)), Shape_sphere(o1)); Conjunction(FrontOf(o1, o2), Near_1(o1, o2)); Conjunction(Far_3(o1, o2), TopQuadrant(o2))"

\timing on

CREATE TEMPORARY TABLE Obj_filtered AS
SELECT * FROM Obj_clevrer;

CREATE INDEX idx_obj_filtered
ON Obj_filtered (vid, fid);

CREATE TEMPORARY TABLE g0 AS SELECT o0.vid as vid, o0.fid as fid, o0.oid as o0_oid, o1.oid as o1_oid, o2.oid as o2_oid FROM Obj_filtered as o0, Obj_filtered as o1, Obj_filtered as o2 WHERE o0.vid = o1.vid and o0.fid = o1.fid and o1.vid = o2.vid and o1.fid = o2.fid and Color('cyan', o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2) = true and Far(3.0, o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2, o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true and Shape('sphere', o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2) = true and o0.oid <> o1.oid and o0.oid <> o2.oid and o1.oid <> o2.oid;

CREATE TEMPORARY TABLE g0_windowed AS (
    SELECT vid, fid, o0_oid, o1_oid, o2_oid,
    lead(fid, 1 - 1, 0) OVER (PARTITION BY vid, o0_oid, o1_oid, o2_oid ORDER BY fid) as fid_offset
    FROM g0
);

CREATE TEMPORARY TABLE g0_contiguous AS (
    SELECT vid, o0_oid, o1_oid, o2_oid, fid AS fid1, fid_offset AS fid2
    FROM g0_windowed
    WHERE fid_offset = fid + (1 - 1)
);

CREATE TEMPORARY TABLE g1 AS SELECT o1.vid as vid, o1.fid as fid, o1.oid as o1_oid, o2.oid as o2_oid FROM Obj_filtered as o1, Obj_filtered as o2 WHERE o1.vid = o2.vid and o1.fid = o2.fid and FrontOf(o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2, o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true and Near(1.0, o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2, o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true and o1.oid <> o2.oid;

CREATE TEMPORARY TABLE g1_windowed AS (
    SELECT vid, fid, o1_oid, o2_oid,
    lead(fid, 1 - 1, 0) OVER (PARTITION BY vid, o1_oid, o2_oid ORDER BY fid) as fid_offset
    FROM g1
);


CREATE TEMPORARY TABLE g1_contiguous AS (
    SELECT vid, o1_oid, o2_oid, fid AS fid1, fid_offset AS fid2
    FROM g1_windowed
    WHERE fid_offset = fid + (1 - 1)
);

CREATE TEMPORARY TABLE g2 AS SELECT o1.vid as vid, o1.fid as fid, o1.oid as o1_oid, o2.oid as o2_oid FROM Obj_filtered as o1, Obj_filtered as o2 WHERE o1.vid = o2.vid and o1.fid = o2.fid and Far(3.0, o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2, o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true and TopQuadrant(o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true and o1.oid <> o2.oid;

CREATE TEMPORARY TABLE g2_windowed AS (
    SELECT vid, fid, o1_oid, o2_oid,
    lead(fid, 1 - 1, 0) OVER (PARTITION BY vid, o1_oid, o2_oid ORDER BY fid) as fid_offset
    FROM g2
);

CREATE TEMPORARY TABLE g2_contiguous AS (
    SELECT vid, o1_oid, o2_oid, fid AS fid1, fid_offset AS fid2
    FROM g2_windowed
    WHERE fid_offset = fid + (1 - 1)
);

-- q0 = g0 with min
CREATE TEMPORARY TABLE q0 AS
SELECT g0.vid as vid, g0.fid2 as fid, g0.o0_oid as o0_oid, g0.o1_oid as o1_oid, g0.o2_oid as o2_oid
FROM g0_contiguous g0;

-- q1 = q0; g1
CREATE TEMPORARY TABLE q1 AS
SELECT DISTINCT t0.vid as vid, t1.fid2 as fid, t0.o0_oid as o0_oid, t0.o1_oid as o1_oid, t0.o2_oid as o2_oid
FROM q0 as t0, g1_contiguous as t1
WHERE t0.vid = t1.vid and t0.fid < t1.fid1 and t0.o1_oid = t1.o1_oid and t0.o2_oid = t1.o2_oid;

-- q2 = q1; g2
CREATE TEMPORARY TABLE q2 AS
SELECT DISTINCT t0.vid as vid, t1.fid2 as fid, t0.o0_oid as o0_oid, t0.o1_oid as o1_oid, t0.o2_oid as o2_oid
FROM q1 as t0, g2_contiguous as t1
WHERE t0.vid = t1.vid and t0.fid < t1.fid1 and t0.o1_oid = t1.o1_oid and t0.o2_oid = t1.o2_oid;

SELECT distinct vid from q2;