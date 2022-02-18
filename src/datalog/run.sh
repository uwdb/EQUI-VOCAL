#!/bin/bash
time /gscratch/balazinska/enhaoz/complex_event_video/src/datalog/souffle/install/bin/souffle queries/q1.2_filter-seq.dl -p data/profile_log_files/q1.2_filter-seq.log -j 1

# time /home/ubuntu/complex_event_video/src/datalog/souffle/install/bin/souffle queries/q2.2.dl -p data/profile_log_files/q2.2.log -j 1

# time /home/ubuntu/complex_event_video/src/datalog/souffle/install/bin/souffle queries/turning_car_and_pedestrain_at_intersection.dl -p data/profile_log_files/turning_car_and_pedestrain_at_intersection.log -j 8

# time /gscratch/balazinska/enhaoz/complex_event_video/src/datalog/souffle/install/bin/souffle queries/q1_bdd.dl -p data/profile_log_files/q1_bdd.log -j 4
