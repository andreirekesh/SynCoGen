#!/bin/bash
#SBATCH --job-name=syncogen_rewrite
#SBATCH --nodes=1                  
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G               
#SBATCH --gres=gpu:2
#SBATCH --time 96:00:00
#SBATCH --reservation=mkoziarski_gpu
#SBATCH --output=logs/runs/%x-%j.out

#########################
####### Configs #########
#########################
CONDA_ENV_NAME=syncogen
CONDA_HOME="/hpf/tools/alma8/miniconda/24.5"

# Default run name (can be overridden via command line)
RUN_NAME="${1:-default_run}"

#########################
####### Env loader ######
#########################
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_NAME}
module load cuda/12.4.1

# Create logs directory if it doesn't exist
mkdir -p logs/runs

#########################
####### Training ########
#########################
srun python -u train.py \
    --config configs/experiments/default.gin \
    --vocab_dir vocabulary/original \
    --gin "WandbLogger.name='${RUN_NAME}'" \
    --gin "Trainer.precision='32'" 
