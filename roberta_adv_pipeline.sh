#!/bin/bash
#SBATCH --job-name=roberta_adv
#SBATCH --output=/scratch/gilbreth/talusb01/EMNLP/logs/roberta_adv_%j.out
#SBATCH --error=/scratch/gilbreth/talusb01/EMNLP/logs/roberta_adv_%j.err
#SBATCH -A pfw-cs                  # Required: Your PFW CS allocation account
#SBATCH -p a100-40gb               # Required: Target GPU cluster partition
#SBATCH -q standby                 # Required: The standby QoS queue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1          # Correct Gilbreth GPU syntax
#SBATCH --time=02:59:00            # Safe limit (Must be under the 3-hour standby ceiling)
#SBATCH --mem=64G

# 1. Clean environment tracks and load the container
module purge
module load rcac
module load ngc/default
module load pytorch/25.01-py3

# 2. Match your workspace and container paths
export PYTHONNOUSERSITE=0          # Crucial: Must be 0 so Python can see your local installed packages!
export PYTHONPATH=""
export HF_HOME="/scratch/gilbreth/talusb01/EMNLP/.hf_cache"
export HF_HUB_OFFLINE=1            # Forces transformers to run completely offline
export PYTHONUNBUFFERED=1          # Keeps stdout unbuffered for instant logging

# 3. Setup workspace logs directory and execute
mkdir -p /scratch/gilbreth/talusb01/EMNLP/logs
cd /scratch/gilbreth/talusb01/EMNLP

python -u train_roberta_adv.py
