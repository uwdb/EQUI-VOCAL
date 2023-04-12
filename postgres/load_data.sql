\COPY Obj_clevrer FROM '/gscratch/balazinska/enhaoz/complex_event_video/postgres/obj_clevrer.csv' DELIMITER ',' CSV;
CREATE INDEX IF NOT EXISTS idx_obj_clevrer ON Obj_clevrer (vid);
\COPY Obj_trajectories FROM '/gscratch/balazinska/enhaoz/complex_event_video/postgres/obj_trajectories.csv' DELIMITER ',' CSV;
CREATE INDEX IF NOT EXISTS idx_obj_trajectories ON Obj_trajectories (vid);
-- \COPY Obj_shibuya FROM '/gscratch/balazinska/enhaoz/complex_event_video/postgres/shibuya.csv' DELIMITER ',' CSV;
-- CREATE INDEX IF NOT EXISTS idx_obj_shibuya ON Obj_shibuya (vid);
\COPY Obj_warsaw FROM '/gscratch/balazinska/enhaoz/complex_event_video/postgres/warsaw_trajectories.csv' DELIMITER ',' CSV;
CREATE INDEX IF NOT EXISTS idx_obj_warsaw ON Obj_warsaw (vid);