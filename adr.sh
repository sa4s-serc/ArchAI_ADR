#!/bin/bash
#SBATCH -A irel
#SBATCH -w gnode081
#SBATCH -c 38
#SBATCH --gres gpu:4
#SBATCH --mem-per-cpu 2G
#SBATCH --time 1-00:00:00
#SBATCH --output job-logs/SE/adr_t5.log
#SBATCH --mail-user rudra.dhar@research.iiit.ac.in
#SBATCH --mail-type ALL
#SBATCH --job-name adr_t5

python3 /home2/rudra.dhar/codes/SE/adr/adr.py