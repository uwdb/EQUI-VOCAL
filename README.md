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
Run the following commands to create a PostgreSQL server instance and then load data into the database. This will create a databse cluster `mylocal_db` and the database data will be stored in `<project_root_dir>/mylocal_db`.

```sh
cd <project_root_dir>
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
To synthesis query, run this command under the `<project_root_dir>/src` directory:

```buildoutcfg
python synthesize.py [-h] [--method METHOD] [--n_init_pos N_INIT_POS] [--n_init_neg N_INIT_NEG]
                     [--dataset_name DATASET_NAME] [--npred NPRED] [--n_nontrivial N_NONTRIVIAL]
                     [--n_trivial N_TRIVIAL] [--depth DEPTH] [--max_duration MAX_DURATION] [--beam_width BEAM_WIDTH]
                     [--pool_size POOL_SIZE] [--k K] [--budget BUDGET] [--multithread MULTITHREAD]
                     [--strategy STRATEGY] [--max_vars MAX_VARS] [--query_str QUERY_STR] [--run_id RUN_ID]
                     [--output_to_file] [--port PORT] [--lru_capacity LRU_CAPACITY] [--reg_lambda REG_LAMBDA]
                     [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR]

options:
  -h, --help            show this help message and exit
  --method METHOD       Query synthesis method.
  --n_init_pos N_INIT_POS
                        Number of initial positive examples provided by the user.
  --n_init_neg N_INIT_NEG
                        Number of initial negative examples provided by the user.
  --dataset_name DATASET_NAME
                        Name of the dataset.
  --npred NPRED         Maximum number of predicates that the synthesized queries can have.
  --n_nontrivial N_NONTRIVIAL
  --n_trivial N_TRIVIAL
  --depth DEPTH         For EQUI-VOCAL: Maximum number of region graphs that the synthesized queries can have. For
                        Quivr: Maximum depth of the nested constructs that the synthesized queries can have.
  --max_duration MAX_DURATION
                        Maximum number of the duration constraint.
  --beam_width BEAM_WIDTH
                        Beam width.
  --pool_size POOL_SIZE
                        Number of queries sampled during example selection.
  --k K                 Number of queries in the final answer.
  --budget BUDGET       Labeling budget.
  --multithread MULTITHREAD
                        Number of CPUs to use.
  --strategy STRATEGY   Strategy for query sampling.
  --max_vars MAX_VARS   Maximum number of variables that the synthesized queries can have.
  --query_str QUERY_STR
                        Target query in compact notation.
  --run_id RUN_ID       Run ID.
  --output_to_file      Whether write the output to file or not.
  --port PORT           Port number of the database.
  --lru_capacity LRU_CAPACITY
                        LRU cache capacity. Only used for Quivr due to its large memory footprint.
  --reg_lambda REG_LAMBDA
                        Regularization parameter.
  --input_dir INPUT_DIR
                        Input directory.
  --output_dir OUTPUT_DIR
                        Output directory.
```

The following scripts provide example configurations for the trajectories dataset and the scene graphs dataset used in the paper:

```sh
cd scripts
# Trajectories dataset
./run_vocal_trajectory.sh
# Scene graphs dataset
./run_vocal_scene_graph.sh
```

### Evaluate query performance
To evaluate the performance of synthesized queries, run this command under the `<project_root_dir>/experiments/analysis` directory:
```buildoutcfg
python evaluate_vocal.py [-h]
                         [--dataset_name {synthetic_scene_graph_easy,synthetic_scene_graph_medium,synthetic_scene_graph_hard,without_duration-sampling_rate_4,trajectories_duration,trajectories_handwritten}]
                         [--query_str QUERY_STR]
                         [--method {vocal_postgres_no_active_learning-topk,vocal_postgres-topk}] [--port PORT]
                         [--multithread MULTITHREAD] [--budget BUDGET] [--task_name TASK_NAME] [--value VALUE]
                         [--run_id RUN_ID] [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR]

options:
  -h, --help            show this help message and exit
  --dataset_name {synthetic_scene_graph_easy,synthetic_scene_graph_medium,synthetic_scene_graph_hard,without_duration-sampling_rate_4,trajectories_duration,trajectories_handwritten}
                        Dataset to evaluate.
  --query_str QUERY_STR
                        Target query to evalaute, in compact notation.
  --method {vocal_postgres_no_active_learning-topk,vocal_postgres-topk}
                        Query synthesis method.
  --port PORT           Port number of the database.
  --multithread MULTITHREAD
                        Number of CPUs to use.
  --budget BUDGET       Labeling budget.
  --task_name TASK_NAME
                        Task name.
  --value VALUE         Value of the tested hyperparameter. If specified, evaluate on the single value; otherwise,
                        evaluate on all values.
  --run_id RUN_ID       Run ID.
  --input_dir INPUT_DIR
                        Input directory.
  --output_dir OUTPUT_DIR
                        Output directory.
```
The following script provides an example configuration used in the paper:
```sh
cd <project_root_dir>/scripts
./eval_vocal.sh
```