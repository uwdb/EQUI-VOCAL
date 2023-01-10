# EQUI-VOCAL

A prototype implementation of EQUI-VOCAL, which is a system to automatically synthesize compositional queries over videos from limited user interactions. See the [technical report](https://arxiv.org/abs/2301.00929) for more details.

# Example Usage

## Start your PostgreSQL server

```sh
pg_ctl -D /gscratch/balazinska/enhaoz/mylocal_db start
```

## Run query synthesis
To reproduce experiment, run this command:
```sh
# Trajectories dataset
./run_vocal_trajectory.sh
# Scene graphs dataset
./run_vocal_scene_graph.sh
```

## Evaluate query performance
To evaluate the performance of synthesized queries, run this command:
```sh
# Trajectories dataset
./eval_vocal_trajectory.sh
# Scene graphs dataset
./eval_vocal_scene_graph.sh
```