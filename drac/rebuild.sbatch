#!/bin/bash
#SBATCH --job-name=vocab_rebuild2
#SBATCH --account=def-bengioy
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=${SYNCOGEN_WORK:-/scratch/$USER/vocab_rebuild}/jobs/rebuild2_%j.out
set -euo pipefail
module --force purge
module load StdEnv/2023 python/3.11.5 rdkit/2024.03.5
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
source "${SYNCOGEN_VENV:-$HOME/venvs/syncogen}/bin/activate"
cd ${SYNCOGEN_REPO:-$HOME/syncogen}
R=${SYNCOGEN_WORK:-/scratch/$USER/vocab_rebuild}
echo "node=$(hostname) start=$(date -Is)"

python -W ignore syncogen/vocabulary/preprocess.py \
  --bb_input  ${SYNCOGEN_DATA:?set SYNCOGEN_DATA}/aablock/filtered_by_reaction.csv \
  --bb_smiles_column SMILES \
  --rxn_input "$R/reactions_v4.txt" \
  --output_dir "$R/vocab_v4" \
  --max_atoms_per_bb_filter 20 \
  --allowed_atom_types "C,N,O,B,F,Cl,Br,S,I" \
  --max_centers_per_bb 5 \
  --no_substructure_filtering

echo "done=$(date -Is)"
echo "=== meta.json ==="; cat "$R/vocab_v4/meta.json"
echo "=== artifacts ==="; ls -la "$R/vocab_v4"
