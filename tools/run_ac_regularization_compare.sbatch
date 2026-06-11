#!/bin/bash
#SBATCH -J ac-ridge-compare
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -o ac_regularization_compare.%j.out
#SBATCH -e ac_regularization_compare.%j.err

set -euo pipefail

SCRIPT_DIR="${COMPARE_SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ ! -f "${SCRIPT_DIR}/compare_ac_regularization.py" ] && [ -f "${SCRIPT_DIR}/tools/compare_ac_regularization.py" ]; then
  SCRIPT_DIR="${SCRIPT_DIR}/tools"
fi

: "${RUN_ROOT:?set RUN_ROOT to the QSGW AC dump root}"
: "${PAIRS:?set PAIRS, e.g. t32_a:t32_b,t1_a:t32_a}"
: "${OUT_DIR:?set OUT_DIR for comparison output}"

ITERATIONS="${ITERATIONS:-1}"
METHODS="${METHODS:-thiele:32,ridge:14:1e-8:10,ridge:16:1e-8:10,ridge:16:1e-6:10}"

python3 "${SCRIPT_DIR}/compare_ac_regularization.py" \
  --run-root "${RUN_ROOT}" \
  --pairs "${PAIRS}" \
  --iterations "${ITERATIONS}" \
  --methods "${METHODS}" \
  --out-dir "${OUT_DIR}"
