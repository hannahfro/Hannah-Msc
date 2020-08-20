#!/bin/bash
#SBATCH --account=def-acliu
#SBATCH --time=0-1:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=<hannah.fronenberg@mail.mcgill.ca>
#SBATCH --mail-type=ALL
#SBATCH --mem=1.5G


python CC_trial.py 