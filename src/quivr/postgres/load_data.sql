COPY Obj_clevrer FROM '/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/obj_clevrer.csv' DELIMITER ',' CSV;
CREATE INDEX IF NOT EXISTS idx_obj_clevrer ON Obj_clevrer (vid);
COPY Obj_trajectories FROM '/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/obj_trajectories.csv' DELIMITER ',' CSV;
CREATE INDEX IF NOT EXISTS idx_obj_trajectories ON Obj_trajectories (vid);
COPY Obj_collision FROM '/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/obj_collision.csv' DELIMITER ',' CSV;
CREATE INDEX IF NOT EXISTS idx_obj_collision ON Obj_collision (vid);