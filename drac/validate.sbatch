#!/bin/bash
#SBATCH --job-name=gg_validate_v3
#SBATCH --account=def-bengioy
#SBATCH --time=0:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=${SYNCOGEN_WORK:-/scratch/$USER/vocab_rebuild}/jobs/validate_v3_%j.out
module --force purge
module load StdEnv/2023 python/3.11.5 rdkit/2024.03.5
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
source "${SYNCOGEN_VENV:-$HOME/venvs/syncogen}/bin/activate"
export PYTHONPATH="${SYNCOGEN_REPO:-$HOME/syncogen}:$PYTHONPATH"
cd ${SYNCOGEN_REPO:-$HOME/syncogen}
python -W ignore ${SYNCOGEN_WORK:-/scratch/$USER/vocab_rebuild}/jobs/validate_rules.py \
   ${SYNCOGEN_WORK:-/scratch/$USER/vocab_rebuild}/vocab_v4 14
echo VALIDATE_DONE
