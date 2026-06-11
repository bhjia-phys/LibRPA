#!/bin/bash
#SBATCH -J qsgw-pair-diff
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 00:05:00
#SBATCH -o qsgw_pair_diff.%j.out
#SBATCH -e qsgw_pair_diff.%j.err

set -euo pipefail

SCRIPT_DIR="${COMPARE_SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ ! -f "${SCRIPT_DIR}/compare_qsgw_pair_outputs.py" ] && [ -f "${SCRIPT_DIR}/tools/compare_qsgw_pair_outputs.py" ]; then
  SCRIPT_DIR="${SCRIPT_DIR}/tools"
fi

: "${RUN_ROOT:?set RUN_ROOT to the QSGW dump root}"
: "${LEFT:?set LEFT case name}"
: "${RIGHT:?set RIGHT case name}"
: "${OUT:?set OUT output TSV path}"

ITERATIONS="${ITERATIONS:-1}"

python3 "${SCRIPT_DIR}/compare_qsgw_pair_outputs.py" \
  --run-root "${RUN_ROOT}" \
  --left "${LEFT}" \
  --right "${RIGHT}" \
  --iterations "${ITERATIONS}" \
  --out "${OUT}"
