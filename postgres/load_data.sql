\COPY Obj_clevrer FROM 'postgres/obj_clevrer.csv' DELIMITER ',' CSV;
CREATE INDEX IF NOT EXISTS idx_obj_clevrer ON Obj_clevrer (vid);
\COPY Obj_trajectories FROM 'postgres/obj_trajectories.csv' DELIMITER ',' CSV;
CREATE INDEX IF NOT EXISTS idx_obj_trajectories ON Obj_trajectories (vid);