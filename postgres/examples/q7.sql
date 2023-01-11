/*
Conjunction(Conjunction(Conjunction(Brown(o0), Far_0.9(o0, o1)), LeftQuadrant(o2)), Metal(o0)); Conjunction(RightOf(o1, o2), RightQuadrant(o0)); LeftOf(o0, o2)
*/

EXPLAIN ANALYZE CREATE TEMPORARY TABLE Obj_filtered AS
SELECT * FROM Obj_clevrer WHERE vid < 300;

CREATE INDEX idx_obj_filtered
ON Obj_filtered (vid, fid);

EXPLAIN ANALYZE CREATE TEMPORARY TABLE g0 AS SELECT o0.vid as vid, o0.fid as fid, o0.oid as o0_oid, o1.oid as o1_oid, o2.oid as o2_oid FROM Obj_filtered as o0, Obj_filtered as o1, Obj_filtered as o2 WHERE o0.vid = o1.vid and o0.fid = o1.fid and o1.vid = o2.vid and o1.fid = o2.fid and Brown(o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2) = true and Far(0.9, o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2, o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2) = true and LeftQuadrant(o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true and Metal(o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2) = true and o0.oid <> o1.oid and o0.oid <> o2.oid and o1.oid <> o2.oid;
CREATE INDEX IF NOT EXISTS idx_g0 ON g0 (vid, fid, o0_oid, o1_oid, o2_oid);

                    EXPLAIN ANALYZE CREATE TEMPORARY TABLE g0_seq_view AS
                    WITH RECURSIVE g0_seq (vid, fid1, fid2, o0_oid, o1_oid, o2_oid) AS (
                        SELECT vid, fid, fid, o0_oid, o1_oid, o2_oid FROM g0
                            UNION
                        SELECT s.vid, s.fid1, g.fid, s.o0_oid, s.o1_oid, s.o2_oid
                        FROM g0_seq s, g0 g
                        WHERE s.vid = g.vid and g.fid = s.fid2 + 1 and s.o0_oid = g.o0_oid and s.o1_oid = g.o1_oid and s.o2_oid = g.o2_oid and g.fid - s.fid1 + 1 <= 1
                    )
                    SELECT DISTINCT vid, fid1, fid2, o0_oid, o1_oid, o2_oid FROM g0_seq WHERE fid2 - fid1 + 1 = 1;

CREATE INDEX IF NOT EXISTS idx_g0_seq_view ON g0_seq_view (vid, fid1, fid2, o0_oid, o1_oid, o2_oid);
EXPLAIN ANALYZE CREATE TEMPORARY TABLE g1 AS SELECT o0.vid as vid, o0.fid as fid, o0.oid as o0_oid, o1.oid as o1_oid, o2.oid as o2_oid FROM Obj_filtered as o0, Obj_filtered as o1, Obj_filtered as o2 WHERE o0.vid = o1.vid and o0.fid = o1.fid and o1.vid = o2.vid and o1.fid = o2.fid and RightOf(o1.shape, o1.color, o1.material, o1.x1, o1.y1, o1.x2, o1.y2, o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true and RightQuadrant(o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2) = true and o0.oid <> o1.oid and o0.oid <> o2.oid and o1.oid <> o2.oid;
CREATE INDEX IF NOT EXISTS idx_g1 ON g1 (vid, fid, o0_oid, o1_oid, o2_oid);

                    EXPLAIN ANALYZE CREATE TEMPORARY TABLE g1_seq_view AS
                    WITH RECURSIVE g1_seq (vid, fid1, fid2, o0_oid, o1_oid, o2_oid) AS (
                        SELECT vid, fid, fid, o0_oid, o1_oid, o2_oid FROM g1
                            UNION
                        SELECT s.vid, s.fid1, g.fid, s.o0_oid, s.o1_oid, s.o2_oid
                        FROM g1_seq s, g1 g
                        WHERE s.vid = g.vid and g.fid = s.fid2 + 1 and s.o0_oid = g.o0_oid and s.o1_oid = g.o1_oid and s.o2_oid = g.o2_oid and g.fid - s.fid1 + 1 <= 1
                    )
                    SELECT DISTINCT vid, fid1, fid2, o0_oid, o1_oid, o2_oid FROM g1_seq WHERE fid2 - fid1 + 1 = 1;

CREATE INDEX IF NOT EXISTS idx_g1_seq_view ON g1_seq_view (vid, fid1, fid2, o0_oid, o1_oid, o2_oid);
EXPLAIN ANALYZE CREATE TEMPORARY TABLE g2 AS SELECT o0.vid as vid, o0.fid as fid, o0.oid as o0_oid, o2.oid as o2_oid FROM Obj_filtered as o0, Obj_filtered as o2 WHERE o0.vid = o2.vid and o0.fid = o2.fid and LeftOf(o0.shape, o0.color, o0.material, o0.x1, o0.y1, o0.x2, o0.y2, o2.shape, o2.color, o2.material, o2.x1, o2.y1, o2.x2, o2.y2) = true and o0.oid <> o2.oid;
CREATE INDEX IF NOT EXISTS idx_g2 ON g2 (vid, fid, o0_oid, o2_oid);

                    EXPLAIN ANALYZE CREATE TEMPORARY TABLE g2_seq_view AS
                    WITH RECURSIVE g2_seq (vid, fid1, fid2, o0_oid, o2_oid) AS (
                        SELECT vid, fid, fid, o0_oid, o2_oid FROM g2
                            UNION
                        SELECT s.vid, s.fid1, g.fid, s.o0_oid, s.o2_oid
                        FROM g2_seq s, g2 g
                        WHERE s.vid = g.vid and g.fid = s.fid2 + 1 and s.o0_oid = g.o0_oid and s.o2_oid = g.o2_oid and g.fid - s.fid1 + 1 <= 1
                    )
                    SELECT DISTINCT vid, fid1, fid2, o0_oid, o2_oid FROM g2_seq WHERE fid2 - fid1 + 1 = 1;

CREATE INDEX IF NOT EXISTS idx_g2_seq_view ON g2_seq_view (vid, fid1, fid2, o0_oid, o2_oid);

                EXPLAIN ANALYZE CREATE TEMPORARY TABLE q0 AS
                SELECT DISTINCT t0.vid as vid, min(t1.fid2) as fid, t0.o0_oid as o0_oid, t0.o1_oid as o1_oid, t0.o2_oid as o2_oid
                FROM g0_seq_view as t0, g1_seq_view as t1
                WHERE t0.vid = t1.vid and t0.fid2 < t1.fid1 and t0.o0_oid = t1.o0_oid and t0.o1_oid = t1.o1_oid and t0.o2_oid = t1.o2_oid
                GROUP BY t0.vid, t0.o0_oid, t0.o1_oid, t0.o2_oid;

CREATE INDEX IF NOT EXISTS idx_q0 ON q0 (vid, fid, o0_oid, o1_oid, o2_oid);

                EXPLAIN ANALYZE CREATE TEMPORARY TABLE q1 AS
                SELECT DISTINCT t0.vid as vid, min(t1.fid2) as fid, t0.o0_oid as o0_oid, t0.o1_oid as o1_oid, t0.o2_oid as o2_oid
                FROM q0 as t0, g2_seq_view as t1
                WHERE t0.vid = t1.vid and t0.fid < t1.fid1 and t0.o0_oid = t1.o0_oid and t0.o2_oid = t1.o2_oid
                GROUP BY t0.vid, t0.o0_oid, t0.o1_oid, t0.o2_oid;

CREATE INDEX IF NOT EXISTS idx_q1 ON q1 (vid, fid, o0_oid, o1_oid, o2_oid);


SELECT count(distinct vid) from q1;


-- Results
-- 12
-- [88, 46, 79, 163, 287, 212, 59, 251, 215, 207, 27, 261]
-- time 5.543581962585449










