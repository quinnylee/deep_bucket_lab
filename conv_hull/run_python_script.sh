#!/bin/bash
#SBATCH --job-name=quinn_test       # Job name
#SBATCH --partition=normal
#SBATCH --nodelist=compute003
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --time=48:00:00                 # Walltime

# Run Python script
python conv_hull_experiment.py
