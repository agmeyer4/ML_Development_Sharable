#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=civil-np
#SBATCH --partition=civil-np
#SBATCH -o ./Slurm_Reports/gridsearch-%A.out # STDOUTS

source activate CO2_Eddy

mprof run --include-children --interval=1 python gridsearch_loop.py
mprof plot --output memory-profile.png
