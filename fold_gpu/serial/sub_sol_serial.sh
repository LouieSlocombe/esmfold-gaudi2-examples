#!/bin/bash

#SBATCH --job-name=fold
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -p htc
#SBATCH -q public
#SBATCH --time=0-01:00:00
#SBATCH -G a100:1
#SBATCH --mem=32G
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="%u@asu.edu"
#SBATCH --export=NONE   # Purge the job-submitting shell environment"

module load cuda-13.0.1-gcc-12.1.0

cd $SLURM_SUBMIT_DIR

module load mamba/latest
source activate monsterproteinstability

start_time=$(date +%s.%N)
echo "Start time: $start_time"

# Run the Python script
srun python3 run.py

end_time=$(date +%s.%N)
echo "End time:   $end_time"

elapsed=$(awk "BEGIN {print $end_time - $start_time}")
elapsed_hours=$(awk "BEGIN {printf \"%.6f\", $elapsed/3600}")
printf "Elapsed time: %.6f hours\n" "$elapsed_hours"