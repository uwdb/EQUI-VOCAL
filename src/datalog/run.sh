#!/bin/bash
time /home/ubuntu/complex_event_video/src/datalog/souffle/install/bin/souffle queries/q1.2_filter-seq.dl -p data/profile_log_files/q1.2_filter-seq.log -j 1 

# time /home/ubuntu/complex_event_video/src/datalog/souffle/install/bin/souffle queries/q2.2.dl -p data/profile_log_files/q2.2.log -j 1 