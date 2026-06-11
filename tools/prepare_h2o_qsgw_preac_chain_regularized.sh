#!/bin/bash
set -euo pipefail

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LABEL="${LABEL:-regularized}"
NITER="${NITER:-1}"
NFREQ="${NFREQ:-32}"
N_PARAMS_ANACON="${N_PARAMS_ANACON:-16}"
ANACON_METHOD="${ANACON_METHOD:-ridge}"
PADE_RIDGE_LAMBDA="${PADE_RIDGE_LAMBDA:-1e-6}"
PADE_RIDGE_DEN_WEIGHT="${PADE_RIDGE_DEN_WEIGHT:-10}"
PADE_DENOMINATOR_FLOOR="${PADE_DENOMINATOR_FLOOR:-1e-12}"
PADE_THIELE_DEN_CUT="${PADE_THIELE_DEN_CUT:-1e-3}"
USE_SCALAPACK_GW_WC="${USE_SCALAPACK_GW_WC:-t}"
USE_ABACUS_GW_SYMMETRY="${USE_ABACUS_GW_SYMMETRY:-t}"
DUMP_ITER="${DUMP_ITER:-1}"
STAGE_MODE="${STAGE_MODE:-copy}"
ARRAY_LIMIT="${ARRAY_LIMIT:-1}"
THREADS="${THREADS:-32}"
PARTITION="${PARTITION:-p1}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"

SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-build}"
EXE="${EXE:-${SRC}/${BUILD_DIR}/chi0_main.exe}"
LD_PREFIX="${LD_PREFIX:-${SRC}/${BUILD_DIR}/src:${SRC}/${BUILD_DIR}/qsgw}"
BASE="${BASE:-/data/home/df_iopcas_bhj/ai-runs/qsgw-restart-h2o-20260326-1/really_tight}"
ROOT_BASE="${ROOT_BASE:-/data/home/df_iopcas_bhj/ai-runs/qsgw-ac-pade-regularization-20260605}"
ROOT="${ROOT:-${ROOT_BASE}/h2o_qsgw_preac_chain_${LABEL}_${STAMP}}"

if [ ! -x "$EXE" ]; then
  echo "ERROR: executable not found or not executable: $EXE" >&2
  exit 2
fi

mkdir -p "$ROOT"

setup_case() {
  local name="$1"
  local dir="$ROOT/$name"
  mkdir -p "$dir"
  cd "$dir"
  for pat in \
    aims.out band_out basis_out bz_sampling_out geometry.in control.in stru_out dielecfunc_out \
    vxc_out vexx_out xc_matr_spin_*.csc KS_eigenvector_*.txt \
    coulomb_cut_*.txt coulomb_mat_*.txt Cs_data_*.txt
  do
    for f in "$BASE"/$pat; do
      [ -e "$f" ] || continue
      if [ "$STAGE_MODE" = "link" ]; then
        ln -sfn "$f" "$(basename "$f")"
      else
        cp -a "$f" "$(basename "$f")"
      fi
    done
  done
  cat > librpa.in <<EOF_LRPA
task = qsgw
nfreq = ${NFREQ}
n_params_anacon = ${N_PARAMS_ANACON}
anacon_method = ${ANACON_METHOD}
pade_ridge_lambda = ${PADE_RIDGE_LAMBDA}
pade_ridge_den_weight = ${PADE_RIDGE_DEN_WEIGHT}
pade_denominator_floor = ${PADE_DENOMINATOR_FLOOR}
pade_thiele_den_cut = ${PADE_THIELE_DEN_CUT}
option_dielect_func = 2
replace_w_head = t
use_scalapack_gw_wc = ${USE_SCALAPACK_GW_WC}
use_abacus_gw_symmetry = ${USE_ABACUS_GW_SYMMETRY}
parallel_routing = libri
vq_threshold = 0
sqrt_coulomb_threshold = 0
qsgw_checkpoint_every = 1
EOF_LRPA
  printf "%s\t%s\t%s\n" "$name" "$THREADS" "$dir" >> "$ROOT/cases.tsv"
}

: > "$ROOT/cases.tsv"
setup_case t32_a
setup_case t32_b

cat > "$ROOT/run_preac_chain_pair.sbatch" <<EOF_SBATCH
#!/bin/bash
#SBATCH -J h2o-qsgw-${LABEL}
#SBATCH -p ${PARTITION}
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${THREADS}
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=${TIME_LIMIT}
#SBATCH --array=0-1%${ARRAY_LIMIT}
#SBATCH -o ${ROOT}/slurm-preac-chain-%A_%a.out
#SBATCH -e ${ROOT}/slurm-preac-chain-%A_%a.err

set -eo pipefail

ROOT="${ROOT}"
EXE="${EXE}"
LD_PREFIX="${LD_PREFIX}"

case "\${SLURM_ARRAY_TASK_ID}" in
  0) CASE_NAME=t32_a ;;
  1) CASE_NAME=t32_b ;;
  *) echo "bad array index \${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

RUN_DIR="\${ROOT}/\${CASE_NAME}"
cd "\$RUN_DIR"

source /data/app/intel/oneapi-2024.2/setvars.sh --force >/dev/null 2>&1
export LD_LIBRARY_PATH="\${LD_PREFIX}:\${LD_LIBRARY_PATH:-}"
export QSGW_AC_DUMP_ALL_ITER=1
export QSGW_PREAC_CHAIN_DUMP=1
if [ -n "${DUMP_ITER}" ]; then
  export QSGW_PREAC_CHAIN_DUMP_ITER="${DUMP_ITER}"
fi
export QSGW_MAX_ITER="${NITER}"
export QSGW_FORCE_MAX_ITER=1
export OMP_NUM_THREADS=${THREADS}
export MKL_NUM_THREADS=${THREADS}
export OPENBLAS_NUM_THREADS=${THREADS}
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_STACKSIZE=512M
export KMP_STACKSIZE=512m
export I_MPI_PIN_DOMAIN=omp
ulimit -s unlimited || true

{
  echo "CASE_NAME=\$CASE_NAME"
  echo "THREADS=${THREADS}"
  echo "QSGW_MAX_ITER=\$QSGW_MAX_ITER"
  echo "QSGW_FORCE_MAX_ITER=\$QSGW_FORCE_MAX_ITER"
  echo "NFREQ=${NFREQ}"
  echo "N_PARAMS_ANACON=${N_PARAMS_ANACON}"
  echo "ANACON_METHOD=${ANACON_METHOD}"
  echo "PADE_RIDGE_LAMBDA=${PADE_RIDGE_LAMBDA}"
  echo "PADE_RIDGE_DEN_WEIGHT=${PADE_RIDGE_DEN_WEIGHT}"
  echo "PADE_DENOMINATOR_FLOOR=${PADE_DENOMINATOR_FLOOR}"
  echo "PADE_THIELE_DEN_CUT=${PADE_THIELE_DEN_CUT}"
  echo "USE_SCALAPACK_GW_WC=${USE_SCALAPACK_GW_WC}"
  echo "USE_ABACUS_GW_SYMMETRY=${USE_ABACUS_GW_SYMMETRY}"
  echo "QSGW_PREAC_CHAIN_DUMP=\$QSGW_PREAC_CHAIN_DUMP"
  echo "QSGW_PREAC_CHAIN_DUMP_ITER=\${QSGW_PREAC_CHAIN_DUMP_ITER:-}"
  echo "SLURM_JOB_ID=\${SLURM_JOB_ID:-}"
  echo "SLURM_ARRAY_TASK_ID=\${SLURM_ARRAY_TASK_ID:-}"
  echo "HOST=\$(hostname)"
  echo "PWD=\$PWD"
  echo "EXE=\$EXE"
  echo "COMMAND=mpirun -np 1 \$EXE"
  echo "START=\$(date -Is)"
} > run_provenance_preac_chain.txt

mpirun -np 1 "\$EXE" > qsgw_preac_chain.out 2>&1

{
  echo "END=\$(date -Is)"
  grep -E "AC diagnostic|Diagnostic max_iterations|Solving quasi|Iteration [0-9]+:|Final Quasi|libRPA finished|ERROR|Error|Warning|terminate|what\\(\\)|bad_alloc" qsgw_preac_chain.out | tail -220 || true
  ls -1 qsgw_preac_chain_*.dat sigc_ac_input_*.dat sigc_ac_output_*.dat vc_gw_*.dat h0_gw_*.dat eigenvalues_*.dat 2>/dev/null | sed 's/^/dump: /' || true
} >> run_provenance_preac_chain.txt
EOF_SBATCH

cat > "$ROOT/README.txt" <<EOF_README
H2O qsgw OMP=${THREADS} regularized analytic-continuation diagnostic run
root: $ROOT
base input: $BASE
source: $SRC
build: $BUILD_DIR
exe: $EXE
QSGW_MAX_ITER=$NITER
NFREQ=$NFREQ
N_PARAMS_ANACON=$N_PARAMS_ANACON
ANACON_METHOD=$ANACON_METHOD
PADE_RIDGE_LAMBDA=$PADE_RIDGE_LAMBDA
PADE_RIDGE_DEN_WEIGHT=$PADE_RIDGE_DEN_WEIGHT
PADE_DENOMINATOR_FLOOR=$PADE_DENOMINATOR_FLOOR
PADE_THIELE_DEN_CUT=$PADE_THIELE_DEN_CUT
USE_SCALAPACK_GW_WC=$USE_SCALAPACK_GW_WC
USE_ABACUS_GW_SYMMETRY=$USE_ABACUS_GW_SYMMETRY
QSGW_PREAC_CHAIN_DUMP=1
QSGW_PREAC_CHAIN_DUMP_ITER=$DUMP_ITER
STAGE_MODE=$STAGE_MODE
ARRAY_LIMIT=$ARRAY_LIMIT
cases: t32_a, t32_b
EOF_README

jobid="$(sbatch --parsable "$ROOT/run_preac_chain_pair.sbatch")"
echo "$jobid" | tee "$ROOT/slurm_job_id.txt"
echo "ROOT=$ROOT"
echo "JOBID=$jobid"
