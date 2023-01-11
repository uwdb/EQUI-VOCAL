-- "Conjunction(Conjunction(Color_cyan(o0), Far_3(o1, o2)), Shape_sphere(o1)); Conjunction(FrontOf(o1, o2), Near_1(o1, o2)); Conjunction(Far_3(o1, o2), TopQuadrant(o2))"

\timing on

CREATE TEMPORARY TABLE Obj_filtered AS
SELECT * FROM Obj_clevrer;

CREATE INDEX idx_obj_filtered
ON Obj_filtered (vid, fid);

CREATE TEMPORARY TABLE g0 AS SELECT o0.vid as vid, o0.fid as fid, o0.oid as o0_oid, o1.oid as o1_oid, o2.oid as o2_oid FROM Obj_filtered as o0, Obj_filtered as o1, Obj_filtered as o2 WHERE o0.vid = o1.vid and o0.fid = o1.fid and o1.vid = o2.vid and o1.fid = o2.fid and Color('cyan', o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2) = true and Far(3.0, o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2, o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true and Shape('sphere', o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2) = true and o0.oid <> o1.oid and o0.oid <> o2.oid and o1.oid <> o2.oid;

CREATE TEMPORARY TABLE g0_contiguous AS
WITH RECURSIVE g0_seq (vid, fid1, fid2, o0_oid, o1_oid, o2_oid) AS (
    SELECT vid, fid, fid, o0_oid, o1_oid, o2_oid FROM g0
        UNION
    SELECT s.vid, s.fid1, g.fid, s.o0_oid, s.o1_oid, s.o2_oid
    FROM g0_seq s, g0 g
    WHERE s.vid = g.vid and g.fid = s.fid2 + 1 and s.o0_oid = g.o0_oid and s.o1_oid = g.o1_oid and s.o2_oid = g.o2_oid and g.fid - s.fid1 + 1 <= 1
)
SELECT DISTINCT vid, min(fid2) as fid, o0_oid, o1_oid, o2_oid FROM g0_seq WHERE fid2 - fid1 + 1 = 1 GROUP BY vid, o0_oid, o1_oid, o2_oid;

CREATE TEMPORARY TABLE g1 AS SELECT o1.vid as vid, o1.fid as fid, o1.oid as o1_oid, o2.oid as o2_oid FROM Obj_filtered as o1, Obj_filtered as o2 WHERE o1.vid = o2.vid and o1.fid = o2.fid and FrontOf(o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2, o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true and Near(1.0, o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2, o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true and o1.oid <> o2.oid;

CREATE TEMPORARY TABLE g1_filtered AS (
    SELECT t0.vid, t1.fid, t0.o0_oid, t0.o1_oid, t0.o2_oid
    FROM g0_contiguous t0, g1 t1
    WHERE t0.vid = t1.vid AND t0.o1_oid = t1.o1_oid and t0.o2_oid = t1.o2_oid AND t0.fid < t1.fid
);

CREATE TEMPORARY TABLE g1_contiguous AS
WITH RECURSIVE g1_seq (vid, fid1, fid2, o0_oid, o1_oid, o2_oid) AS (
    SELECT vid, fid, fid, o0_oid, o1_oid, o2_oid FROM g1_filtered
        UNION
    SELECT s.vid, s.fid1, g.fid, s.o0_oid, s.o1_oid, s.o2_oid
    FROM g1_seq s, g1_filtered g
    WHERE s.vid = g.vid and g.fid = s.fid2 + 1 and s.o0_oid = g.o0_oid and s.o1_oid = g.o1_oid and s.o2_oid = g.o2_oid and g.fid - s.fid1 + 1 <= 1
)
SELECT DISTINCT vid, min(fid2) as fid, o0_oid, o1_oid, o2_oid FROM g1_seq WHERE fid2 - fid1 + 1 = 1 GROUP BY vid, o0_oid, o1_oid, o2_oid;

CREATE TEMPORARY TABLE g2 AS SELECT o1.vid as vid, o1.fid as fid, o1.oid as o1_oid, o2.oid as o2_oid FROM Obj_filtered as o1, Obj_filtered as o2 WHERE o1.vid = o2.vid and o1.fid = o2.fid and Far(3.0, o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2, o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true and TopQuadrant(o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true and o1.oid <> o2.oid;

CREATE TEMPORARY TABLE g2_filtered AS (
    SELECT t0.vid, t1.fid, t0.o0_oid, t0.o1_oid, t0.o2_oid
    FROM g1_contiguous t0, g2 t1
    WHERE t0.vid = t1.vid AND t0.o1_oid = t1.o1_oid and t0.o2_oid = t1.o2_oid AND t0.fid < t1.fid
);

CREATE TEMPORARY TABLE g2_contiguous AS
WITH RECURSIVE g2_seq (vid, fid1, fid2, o0_oid, o1_oid, o2_oid) AS (
    SELECT vid, fid, fid, o0_oid, o1_oid, o2_oid FROM g2_filtered
        UNION
    SELECT s.vid, s.fid1, g.fid, s.o0_oid, s.o1_oid, s.o2_oid
    FROM g2_seq s, g2_filtered g
    WHERE s.vid = g.vid and g.fid = s.fid2 + 1 and s.o0_oid = g.o0_oid and s.o1_oid = g.o1_oid and s.o2_oid = g.o2_oid and g.fid - s.fid1 + 1 <= 1
)
SELECT DISTINCT vid, min(fid2) as fid, o0_oid, o1_oid, o2_oid FROM g2_seq WHERE fid2 - fid1 + 1 = 1 GROUP BY vid, o0_oid, o1_oid, o2_oid;

SELECT distinct vid from g2_contiguous;