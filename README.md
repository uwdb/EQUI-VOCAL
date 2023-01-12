# EQUI-VOCAL

A prototype implementation of EQUI-VOCAL, which is a system to automatically synthesize compositional queries over videos from limited user interactions. See the [technical report](https://arxiv.org/abs/2301.00929) for more details.

## Cloning

Install Git Large File Storage before cloning the repository, then,

```sh
git clone https://github.com/uwdb/EQUI-VOCAL.git
```

Pulling large files using the following commands:
```sh
cd EQUI-VOCAL
git lfs install
git lfs pull
```

## Example Usage

### Set up your PostgreSQL server

1. Run the following commands to create a PostgreSQL server instance and then load data into the database.

```sh
# Create a PostgreSQL server instance
initdb -D mylocal_db --no-locale --encoding=UTF8
# Start the server
pg_ctl -D mylocal_db start
# Create a database
createdb --owner=enhaoz myinner_db
# Configure
psql -f postgres/alter_config-cpu_1-mem_100.sql  myinner_db
# Restart the server
pg_ctl -D mylocal_db restart
# Create relations
psql -f postgres/create_table.sql myinner_db
# Load data
psql -f postgres/load_data.sql myinner_db
# Load user-defined functions
psql -f postgres/create_udf.sql myinner_db
```

### Run query synthesis
To reproduce experiment, run this command:
```sh
# Trajectories dataset
./run_vocal_trajectory.sh
# Scene graphs dataset
./run_vocal_scene_graph.sh
```

### Evaluate query performance
To evaluate the performance of synthesized queries, run this command:
```sh
./eval_vocal.sh
```