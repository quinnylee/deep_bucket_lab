#!/bin/bash
#SBATCH --job-name=lstm_exp       # Job name
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu001
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --time=12:00:00                 # Walltimes
#SBATCH -o outputs/output.%j.out # STDOUT

# Run Python script
python lstm_conv_experiment.py