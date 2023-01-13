# EQUI-VOCAL

A prototype implementation of EQUI-VOCAL, which is a system to automatically synthesize compositional queries over videos from limited user interactions. See the [technical report](https://arxiv.org/abs/2301.00929) for more details.

## Setup Instructions

The project uses `conda` to manage dependencies. To install conda, follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

```sh
# Clone the repository
git clone https://github.com/uwdb/EQUI-VOCAL.git
cd EQUI-VOCAL

# Create a conda environment (called equi-vocal) and install dependencies
conda env create -f environment.yml
conda activate equi-vocal
python -m pip install -e .
```

The project uses Git Large File Storage to track large files.

```sh
# Pull large files
git lfs install
git lfs pull
```

## Example Usage

### Set up your PostgreSQL server
Run the following commands to create a PostgreSQL server instance and then load data into the database.

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
To reproduce experiment, run the following commands:
```sh
cd scripts
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