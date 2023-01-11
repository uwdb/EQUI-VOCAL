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
    SELECT vid, o0_oid, o1_oid, fid AS fid1, fid_offset AS fid2
    FROM g0_windowed
    WHERE fid_offset = fid + (15 - 1)
);

CREATE TEMPORARY TABLE g1 AS SELECT o0.vid as vid, o0.fid as fid, o0.oid as o0_oid, o1.oid as o1_oid FROM Obj_filtered as o0, Obj_filtered as o1 WHERE o0.vid = o1.vid and o0.fid = o1.fid and FrontOf(o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2, o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2) = true and Near(1.0, o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2, o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2) = true and o0.oid <> o1.oid;

CREATE TEMPORARY TABLE g1_windowed AS (
    SELECT vid, fid, o0_oid, o1_oid,
    lead(fid, 1 - 1, 0) OVER (PARTITION BY vid, o0_oid, o1_oid ORDER BY fid) as fid_offset
    FROM g1
);


CREATE TEMPORARY TABLE g1_contiguous AS (
    SELECT vid, o0_oid, o1_oid, fid AS fid1, fid_offset AS fid2
    FROM g1_windowed
    WHERE fid_offset = fid + (1 - 1)
);

CREATE TEMPORARY TABLE g2 AS SELECT o0.vid as vid, o0.fid as fid, o0.oid as o0_oid, o1.oid as o1_oid FROM Obj_filtered as o0, Obj_filtered as o1 WHERE o0.vid = o1.vid and o0.fid = o1.fid and Far(3.0, o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2, o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2) = true and RightQuadrant(o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2) = true and o0.oid <> o1.oid;

CREATE TEMPORARY TABLE g2_windowed AS (
    SELECT vid, fid, o0_oid, o1_oid,
    lead(fid, 5 - 1, 0) OVER (PARTITION BY vid, o0_oid, o1_oid ORDER BY fid) as fid_offset
    FROM g2
);


CREATE TEMPORARY TABLE g2_contiguous AS (
    SELECT vid, o0_oid, o1_oid, fid AS fid1, fid_offset AS fid2
    FROM g2_windowed
    WHERE fid_offset = fid + (5 - 1)
);

-- q0 = g0 with min
CREATE TEMPORARY TABLE q0 AS
SELECT g0.vid as vid, g0.fid2 as fid, g0.o0_oid as o0_oid, g0.o1_oid as o1_oid
FROM g0_contiguous g0;

-- q1 = q0; g1
CREATE TEMPORARY TABLE q1 AS
SELECT DISTINCT t0.vid as vid, t1.fid2 as fid, t0.o1_oid as o1_oid, t0.o0_oid as o0_oid
FROM q0 as t0, g1_contiguous as t1
WHERE t0.vid = t1.vid and t0.fid < t1.fid1 and t0.o1_oid = t1.o1_oid and t0.o0_oid = t1.o0_oid;

-- q2 = q1; g2
CREATE TEMPORARY TABLE q2 AS
SELECT DISTINCT t0.vid as vid, t1.fid2 as fid, t0.o1_oid as o1_oid, t0.o0_oid as o0_oid
FROM q1 as t0, g2_contiguous as t1
WHERE t0.vid = t1.vid and t0.fid < t1.fid1 and t0.o1_oid = t1.o1_oid and t0.o0_oid = t1.o0_oid;

SELECT distinct vid from q2;