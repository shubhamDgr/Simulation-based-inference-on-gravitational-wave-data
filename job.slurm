#!/bin/bash
#BATCH --job-name=sbi-gw
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=sbi_%j.out
#SBATCH --error=sbi_%j.err

# Load system environment
module load 2025
module load Python/3.13.5-GCCcore-14.3.0

# Activate your virtual environment
source ~/venvs/sbi_2025/bin/activate

# Go to your project directory (adjust if needed)
cd $HOME/ML-SBI-project-group-6

# Run your program
python trainer_gpu.py
