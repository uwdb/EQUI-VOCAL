#!/bin/bash
#
# Stop on errors
set -Eeuo pipefail
set -x

# python3 codes/remove_duplicate_descr.py

# python3 codes/generate_bin_range.py

# python3 codes/split_category.py

for i in {0..3562}
do
    python3 codes/train_lstm.py 0 $i

    python3 codes/train_lstm.py 1 $i

    python3 codes/join.py $i

    python3 codes/comp_rsquare.py $i
done
