#!/usr/bin/env python
import os

import monsterproteinstability as mps

if __name__ == "__main__":
    files = mps.list_files_with_extension("data/", ".faa")

    for j, file in enumerate(files):
        print(f'j = {j}, submitting jobs for file = {file}', flush=True)
        # Load the fasta entries
        entries, descriptions = mps.load_fasta_entries(file)
        n = len(entries)
        print(f'Number of entries in file: {n}', flush=True)
        srun_line = "${SLURM_ARRAY_TASK_ID}"
        sub_file_data = f"""#!/bin/bash
#SBATCH --job-name=fold_{j}
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -p htc
#SBATCH -q public
#SBATCH --time=0-01:00:00
#SBATCH -G a100:1
#SBATCH --array=0-{n - 1}
#SBATCH --mem=32G
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.out
#SBATCH --export=NONE

cd $SLURM_SUBMIT_DIR

module load cuda-13.0.1-gcc-12.1.0
ENV_NAME="monsterproteinstability"

module load mamba/latest
source activate $ENV_NAME

$HOME/.conda/envs/$ENV_NAME/bin/python3 run_fold_parallel.py {srun_line} {j}
    """

        folder = f"F{j:03d}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            print(f"Folder {folder} already exists, skipping creation.", flush=True)

        os.system(f"cp run_fold_parallel.py {folder}/run_fold_parallel.py")
        with open(os.path.join(folder, "sub_fold.sh"), "w") as f:
            f.write(sub_file_data)

        os.system(f"cd {folder} && sbatch sub_fold.sh && cd ..")
        print(f"Submitted jobs for file in folder {folder}", flush=True)
        print(flush=True)
