COPY Obj_clevrer FROM '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/obj_clevrer.csv' DELIMITER ',' CSV;
CREATE INDEX IF NOT EXISTS idx_obj_clevrer ON Obj_clevrer (vid);
-- COPY Obj_trajectories FROM '/mmfs1/gscratch/balazinska/enhaoz/complex_event_video/src/quivr/postgres/test_trajectories.csv' DELIMITER ',' CSV;