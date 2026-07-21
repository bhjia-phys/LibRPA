#!/bin/bash
#SBATCH -J si-qsgw-adv-k888
#SBATCH -p 48cp3
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -o qsgw-advance-%j.out
#SBATCH -e qsgw-advance-%j.err

set -eo pipefail
RUN_DIR="/data/home/df_iopcas_bhj/ai-runs/si-qsgw-band0-k888-sym-shrink-headwing-iter1-20260514-173351"
LIBRPA="/data/home/df_iopcas_bhj/software-stack/src/LibRPA_codex/build/chi0_main.exe"
PYTHON="/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python"
export EXPECT_NK=512
export EXPECT_NBANDS=44

source /data/app/intel/oneapi-2024.2/setvars.sh --force >/dev/null 2>&1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export I_MPI_PIN_DOMAIN=omp
export I_MPI_DEBUG=4
export LD_LIBRARY_PATH="/data/home/df_iopcas_bhj/software-stack/src/LibRPA_codex/build/src:/data/home/df_iopcas_bhj/software-stack/src/LibRPA_codex/build/qsgw:${LD_LIBRARY_PATH:-}"
NP="${SLURM_NTASKS:-4}"

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
need_file() { test -s "$1" || { echo "Missing required file: $1" >&2; exit 20; }; }

cd "$RUN_DIR"

log "Runtime provenance"
echo "RUN_DIR=$RUN_DIR"
echo "LIBRPA=$LIBRPA"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual} NP=$NP OMP_NUM_THREADS=$OMP_NUM_THREADS"
mpirun -np "$NP" hostname | sort | uniq -c

need_file librpa.d/qsgw_checkpoints/latest_iteration.txt
LATEST="$(tr -d '[:space:]' < librpa.d/qsgw_checkpoints/latest_iteration.txt)"
case "$LATEST" in
  ''|*[!0-9]*) echo "Invalid latest checkpoint iteration: $LATEST" >&2; exit 21 ;;
esac
NEXT=$((LATEST + 1))
LATEST4="$(printf '%04d' "$LATEST")"
NEXT4="$(printf '%04d' "$NEXT")"
LATEST5="$(printf '%05d' "$LATEST")"

log "Advance QSGW from checkpoint iteration $LATEST to $NEXT"
need_file "librpa.d/qsgw_checkpoints/iter_${LATEST5}/checkpoint.meta"
need_file "hrs1_nao_qsgw_iter_${LATEST4}.csr"
need_file OUT.ABACUS_scf/running_scf.log
need_file OUT.ABACUS_scf/srs1_nao.csr
need_file OUT.ABACUS_scf/rr.csr
need_file band_out
ls KS_eigenvector_*.dat >/dev/null
for f in irreducible_sector.txt symrot_R.txt symrot_k.txt symrot_abf_k.txt Cs_shrinked_data_0.txt shrink_sinvS_0.txt; do need_file "$f"; done
ls vxcs1k*_nao.txt >/dev/null
ls sks1k*_nao.txt >/dev/null
ls band_vxc*_nao.txt band_vxck*_nao.txt band_vxcs*_nao.txt >/dev/null 2>&1 || true

log "Refresh pyatb full-grid velocity from exported QSGW H(R)"
REFRESH_DIR="pyatb_refresh_iter_${LATEST4}"
rm -rf "$REFRESH_DIR"
mkdir -p "$REFRESH_DIR/OUT.ABACUS"
cp -L STRU KPT_scf get_diel.py output_librpa.py fix_pyatb.py "$REFRESH_DIR/"
cp -L OUT.ABACUS_scf/running_scf.log "$REFRESH_DIR/OUT.ABACUS/"
cp -L OUT.ABACUS_scf/srs1_nao.csr "$REFRESH_DIR/OUT.ABACUS/"
cp -L OUT.ABACUS_scf/rr.csr "$REFRESH_DIR/OUT.ABACUS/"
cp -L "hrs1_nao_qsgw_iter_${LATEST4}.csr" "$REFRESH_DIR/OUT.ABACUS/hrs1_nao.csr"
(
  cd "$REFRESH_DIR"
  mpirun -np 1 "$PYTHON" get_diel.py
  "$PYTHON" fix_pyatb.py
  "$PYTHON" - <<'PY'
import os
from pathlib import Path
expect_nk = int(os.environ['EXPECT_NK'])
expect_nb = int(os.environ['EXPECT_NBANDS'])
vel = Path('pyatb_librpa_df/velocity_matrix')
band = Path('pyatb_librpa_df/band_out')
kinfo = Path('pyatb_librpa_df/k_path_info')
for p in (vel, band, kinfo):
    if not p.is_file() or p.stat().st_size == 0:
        raise SystemExit(f'missing pyatb file: {p}')
with vel.open() as f:
    vel_header = [int(f.readline().strip()) for _ in range(4)]
if vel_header != [expect_nk, 1, expect_nb, expect_nb]:
    raise SystemExit(f'bad velocity header: {vel_header}')
with band.open() as f:
    band_header = [int(f.readline().strip()) for _ in range(4)]
if band_header != [expect_nk, 1, expect_nb, expect_nb]:
    raise SystemExit(f'bad pyatb band_out header: {band_header}')
print('pyatb refresh header check PASS', vel_header)
PY
)
rm -rf pyatb_librpa_df
cp -a "$REFRESH_DIR/pyatb_librpa_df" .
head -4 pyatb_librpa_df/velocity_matrix > "pyatb_velocity_header_iter_${LATEST4}.txt"

log "Run LibRPA restart for QSGW iteration $NEXT"
cat > librpa.in <<EOF
task = qsgw_band0
nfreq = 16
n_params_anacon = 16
option_dielect_func = 3
replace_w_head = t
use_scalapack_gw_wc = t
use_scalapack_ecrpa = t
parallel_routing = libri
vq_threshold = 0
sqrt_coulomb_threshold = 0
use_shrink_abfs = t
use_abacus_exx_symmetry = t
use_abacus_gw_symmetry = t
use_fullcoul_exx = t
use_pyatb = t
output_energy_qp = t
output_gw_sigc_mat_rf = f
libri_chi0_threshold_C = 1e-4
libri_chi0_threshold_G = 1e-5
libri_exx_threshold_V = 1e-1
libri_exx_threshold_C = 1e-4
libri_exx_threshold_D = 1e-4
libri_g0w0_threshold_C = 1e-5
libri_g0w0_threshold_G = 1e-5
libri_g0w0_threshold_Wc = 1e-6
max_iter = ${NEXT}
qsgw_restart = t
qsgw_restart_dir = librpa.d/qsgw_checkpoints/
qsgw_restart_iteration = ${LATEST}
qsgw_checkpoint_every = 1
qsgw_export_hamiltonian_for_pyatb = t
output_dir = librpa.d/
EOF
mpirun -np "$NP" "$LIBRPA" 16 1e-12
need_file "QSGW_band_spin_1_${NEXT}.dat"
need_file "hrs1_nao_qsgw_iter_${NEXT4}.csr"
need_file "librpa.d/qsgw_checkpoints/iter_$(printf '%05d' "$NEXT")/checkpoint.meta"
log "QSGW iteration $NEXT complete"
