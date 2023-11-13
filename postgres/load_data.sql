\COPY Obj_clevrer FROM '/home/enhao/EQUI-VOCAL/postgres/obj_clevrer.csv' DELIMITER ',' CSV;
CREATE INDEX IF NOT EXISTS idx_obj_clevrer ON Obj_clevrer (vid);
\COPY Obj_trajectories FROM '/home/enhao/EQUI-VOCAL/postgres/obj_trajectories.csv' DELIMITER ',' CSV;
CREATE INDEX IF NOT EXISTS idx_obj_trajectories ON Obj_trajectories (vid);
-- \COPY Obj_shibuya FROM '/home/enhao/EQUI-VOCAL/postgres/shibuya.csv' DELIMITER ',' CSV;
-- CREATE INDEX IF NOT EXISTS idx_obj_shibuya ON Obj_shibuya (vid);
-- \COPY Obj_warsaw FROM '/home/enhao/EQUI-VOCAL/postgres/warsaw_trajectories.csv' DELIMITER ',' CSV;
-- CREATE INDEX IF NOT EXISTS idx_obj_warsaw ON Obj_warsaw (vid);