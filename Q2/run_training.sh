#!/bin/bash

# --- SLURM Directives ---
#SBATCH -J Colorization_CPU            # Job name
#SBATCH -A research                    # Account to use (change if necessary)
#SBATCH -p general                     # Partition for CPU jobs (guaranteed compatibility)
#SBATCH --cpus-per-task=8              # Request 8 CPU cores for multiprocessing
#SBATCH --mem=32G                      # Request 32 GB of RAM (sufficient for CIFAR-10)
#SBATCH --time=0-08:00:00              # Max time (8 hours)
#SBATCH --output=slurm_logs/colorization_%j.out # Output log file
#SBATCH --error=slurm_logs/colorization_%j.err  # Error log file

#SBATCH --mail-user=bhogadi.likhit@research.iiit.ac.in # <--- REPLACE WITH YOUR ACTUAL IIIT EMAIL
#SBATCH --mail-type=END,FAIL           # Email on job completion or failure

# --- Setup ---
# Create log directory if it doesn't exist
mkdir -p slurm_logs

# Activate your Conda Environment
# Replace 'smaia4env' if your environment path is different
source ~/.bashrc
conda activate smaia4env

# Navigate to the directory containing train.py
# The current working directory is usually the directory where you run sbatch
cd ~/smai_a4/Q2

# --- Execution ---
echo "Starting training script on CPU..."
echo "Current Device setting in train.py must be set to 'cpu'"

# Set the device environment variable explicitly to ensure no accidental CUDA attempts
export CUDA_VISIBLE_DEVICES="" 

# Run the Python training script
# Note: You MUST manually change 'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")'
# in your train.py file to 'device = torch.device("cpu")' for this to work reliably.
python3 train.py

echo "Job finished."