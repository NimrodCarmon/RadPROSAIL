#!/bin/bash
#SBATCH --job-name=single-node-job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=08:00:00
#SBATCH --mem=180GB

# Function to stop Ray and kill Python processes run by 'carmon' when the script exits
cleanup() {
  ray stop
  pkill -f python -u carmon
}
trap cleanup EXIT

# Activate the Dreams2 Conda environment
source /beegfs/store/carmon/anaconda3/etc/profile.d/conda.sh
conda activate Dreams2

# Run Python script
python run_map_traits.py /scratch/carmon/venus/venus_run_zoomin.json