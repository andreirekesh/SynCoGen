#!/bin/bash
# Generate 1M molecules at one fragment count. Submit as:
#   sbatch --array=2-5 generate.sh
# Array index IS the fragment count.
#SBATCH --job-name=gen4M
#SBATCH --account=def-bengioy
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --output=${SYNCOGEN_WORK:-/scratch/$USER/vocab_rebuild}/jobs/logs/gen_%A_%a.out
set -uo pipefail
module --force purge
module load StdEnv/2023 python/3.11.5 rdkit/2024.03.5
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
source "${SYNCOGEN_VENV:-$HOME/venvs/syncogen}/bin/activate"
cd ${SYNCOGEN_REPO:-$HOME/syncogen}
R=${SYNCOGEN_WORK:-/scratch/$USER/vocab_rebuild}
L=${SLURM_ARRAY_TASK_ID}
N=${N_MOL:-1000000}
export SYNCOGEN_MAX_MW=${MAXMW:-550}
export SYNCOGEN_MAX_CANDIDATES=${MAXCAND:-32}
TEMP_ARGS=""
[ -n "${TEMPERATURE:-}" ] && TEMP_ARGS="--sample_by_inverse_molwt --molwt_temperature ${TEMPERATURE}"

echo "node=$(hostname) length=$L n=$N cores=$SLURM_CPUS_PER_TASK"
echo "SYNCOGEN_MAX_MW=$SYNCOGEN_MAX_MW SYNCOGEN_MAX_CANDIDATES=$SYNCOGEN_MAX_CANDIDATES TEMP='${TEMPERATURE:-none}'"
echo "start=$(date -Is)"

# RDKit logging is deliberately NOT suppressed: its errors must land in this log.
/usr/bin/time -v python -W ignore syncogen/vocabulary/generate_graph_data.py \
  --vocab_dir "$R/vocab_v4" \
  --num_samples "$N" --length "$L" \
  --seed $((1000 + L)) --num_workers "$SLURM_CPUS_PER_TASK" --timeout 20 \
  --save_graphs --output_path "$R/graphs/len${L}_1M.pt" \
  $TEMP_ARGS 2> >(tee "$R/jobs/logs/gen_err_L${L}.txt" >&2)

echo "end=$(date -Is)"
LOG="$R/jobs/logs/gen_${SLURM_ARRAY_JOB_ID}_${L}.out"
ERR="$R/jobs/logs/gen_err_L${L}.txt"
echo "================ ERROR / REJECTION REPORT (length $L) ================"
echo "-- RDKit stderr messages by type --"
grep -oE "(ERROR|WARNING): [^$]*" "$ERR" 2>/dev/null | sed 's/[0-9]\+/N/g' | sort | uniq -c | sort -rn | head -20
echo "   rdkit ERROR lines: $(grep -c 'ERROR' "$ERR" 2>/dev/null || echo 0)"
echo "   rdkit WARNING lines: $(grep -c 'WARNING' "$ERR" 2>/dev/null || echo 0)"
echo "-- python-level sanitisation warnings --"
echo "   'could not be sanitized': $(grep -c 'could not be sanitized' "$LOG" 2>/dev/null || echo 0)"
echo "-- aggregated worker counters --"
python3 - "$LOG" <<'PYEOF'
import sys, json, re
from collections import Counter
tot=Counter(); n=0
for line in open(sys.argv[1], errors="ignore"):
    if line.startswith("WORKER_STATS "):
        tot.update(json.loads(line[len("WORKER_STATS "):])); n+=1
print(f"   workers reporting: {n}")
a=tot.get("attempts",0)
for k in ("attempts","accepted","rejected_unsanitizable","rejected_disconnected","dfs_failed_or_timeout"):
    v=tot.get(k,0)
    print(f"   {k:26s} {v:>10,}" + (f"  ({100*v/a:.3f}% of attempts)" if a and k!="attempts" else ""))
PYEOF
echo "GEN_DONE length $L"
