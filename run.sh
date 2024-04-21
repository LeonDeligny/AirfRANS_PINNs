#!/bin/bash
#SBATCH --job-name=out
#SBATCH --output=out_%j.out
#SBATCH --error=out_%j.err
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=1          # Number of tasks per node
#SBATCH --cpus-per-task=1            # Number of CPUs per task
#SBATCH --mem=100G                   # Memory per node
#SBATCH --time=00:05:00              # Time (hh:mm:ss)
#SBATCH --gres=gpu:1 
#SBATCH --partition=gpu 
#SBATCH --qos=gpu

# Execute the Python script
python __main__.py
