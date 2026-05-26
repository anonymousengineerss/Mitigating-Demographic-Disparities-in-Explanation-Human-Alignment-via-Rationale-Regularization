#!/bin/bash
#SBATCH --job-name=roberta_hatexplain
#SBATCH --output=/scratch/gilbreth/talusb01/EMNLP/logs/roberta_%j.out
#SBATCH --error=/scratch/gilbreth/talusb01/EMNLP/logs/roberta_%j.err
#SBATCH -A pfw-cs                  # Required: Your allocation account
#SBATCH -p a100-40gb               # Required: Target GPU nodes
#SBATCH -q standby                 # Required: The standby queue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1          # Correct Gilbreth GPU request syntax
#SBATCH --time=02:59:00            # Safe limit (Must be under the 3-hour standby ceiling)
#SBATCH --mem=64G

# 1. Clean environment and load the correct container
module purge
module load rcac
module load ngc/default
module load pytorch/25.01-py3

# 2. Match your working environment paths
export PYTHONNOUSERSITE=0          # Crucial: Allows container to read 'transformers' and 'seaborn' from your user profile
export PYTHONPATH=""
export HF_HOME="/scratch/gilbreth/talusb01/EMNLP/.hf_cache"
export HF_HUB_OFFLINE=1            # Enforces local weights loading safely

# 3. Directories setup and execution
mkdir -p /scratch/gilbreth/talusb01/EMNLP/logs
cd /scratch/gilbreth/talusb01/EMNLP

python train_roberta.py
