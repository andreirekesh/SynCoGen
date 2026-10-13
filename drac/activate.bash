#!/bin/bash
# Activate the SynCoGen environment on this cluster:  source env.sh
#
# The venv lives in $HOME, not /project: the rrg-bengioy-ad project filesystem is
# at its file-count quota (~5M files), and venvs create tens of thousands of files.
#
# rdkit is provided by a module rather than by pip, and the module must be loaded
# BEFORE the venv is activated, so keep this order.

module --force purge
module load StdEnv/2023 python/3.11.5 rdkit/2024.03.5 cuda/12.6 xtb/6.7.1

export PYTHONNOUSERSITE=1   # keep the ~/.local python3.9 site-packages out of the way
source "$HOME/venvs/syncogen/bin/activate"
