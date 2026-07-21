#!/bin/bash
#SBATCH -J si-qsgw-b0-i1-k888
#SBATCH -p p1
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -eo pipefail
RUN_DIR="/data/home/df_iopcas_bhj/ai-runs/si-qsgw-band0-k888-sym-shrink-headwing-iter1-20260514-173351"
ABACUS="/data/home/df_iopcas_bhj/software-stack/src/abacus-old-libri/build/abacus_3p"
LIBRPA="/data/home/df_iopcas_bhj/software-stack/src/LibRPA_codex/build/chi0_main.exe"
PYTHON="/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python"
export EXPECT_NK=512
export EXPECT_NBANDS=44
cd "$RUN_DIR"

source /data/app/intel/oneapi-2024.2/setvars.sh --force >/dev/null 2>&1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-40}"
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
copy_out_glob() {
  for pat in "$@"; do
    for f in OUT.ABACUS/$pat; do
      [ -e "$f" ] && cp -L "$f" . || true
    done
  done
}

log "Runtime provenance"
echo "RUN_DIR=$RUN_DIR"
echo "ABACUS=$ABACUS"
echo "LIBRPA=$LIBRPA"
echo "LIBRPA_SRC=/data/home/df_iopcas_bhj/software-stack/src/LibRPA_codex"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual} NP=$NP OMP_NUM_THREADS=$OMP_NUM_THREADS"
mpirun -np "$NP" hostname | sort | uniq -c

log "Clean generated outputs in fresh directory"
rm -rf OUT.ABACUS OUT.ABACUS_scf pyatb_librpa_df librpa.d scf_librpa_root __pycache__
rm -f INPUT KPT band_KS_* band_vxc* band_kpath_info GW_band_spin_*.dat KS_band_spin_*.dat
rm -f QSGW_band_spin_*.dat EXX_band_spin_*.dat homo_lumo_vs_iterations.dat
rm -f self_energy* sigc* sigma* dielectric_function_*.dat dielecfunc_out LibRPA*.out librpa_para_*_myid_*.out
rm -f s1k*_nao.txt sks1k*_nao.txt vxcs1k*_nao.txt band_sk*_nao.txt band_vxck*_nao.txt

log "Stage 1: ABACUS SCF symmetry=1, shrink outputs, k888"
cp INPUT_scf INPUT
cp KPT_scf KPT
mpirun -np "$NP" "$ABACUS"
need_file OUT.ABACUS/running_scf.log
need_file OUT.ABACUS/ABACUS-CHARGE-DENSITY.restart
copy_out_glob 'hrs*_nao.csr' 'srs*_nao.csr' 'rr.csr' 'vxc_out.dat' 'irreducible_sector.txt' 'symrot_R.txt' 'symrot_k.txt' 'symrot_abf_k.txt' 'Cs_data_*.txt' 'Cs_shrinked_data_*.txt' 'coulomb_cut_*.txt' 'coulomb_mat_*.txt' 'shrink_sinvS_*.txt'
ln -sf vxc_out.dat vxc_out
mkdir -p scf_librpa_root
for f in band_out k_path_info KS_eigenvector_*.dat; do [ -e "$f" ] && cp -L "$f" scf_librpa_root/ || true; done
for f in irreducible_sector.txt symrot_R.txt symrot_k.txt symrot_abf_k.txt vxc_out.dat vxc_out; do [ -e "$f" ] && cp -L "$f" scf_librpa_root/ || true; done
need_file scf_librpa_root/band_out
ls scf_librpa_root/KS_eigenvector_*.dat >/dev/null
for f in irreducible_sector.txt symrot_R.txt symrot_k.txt symrot_abf_k.txt Cs_shrinked_data_0.txt shrink_sinvS_0.txt; do need_file "$f"; done

log "Stage 1b: ABACUS SCF matrix adapter for QSGW-band0"
"$PYTHON" abacus_qsgw_adapter.py --scf-dir OUT.ABACUS --outdir . --write-legacy-sks-alias --max-rms-error-ev 1e-2 > qsgw_adapter.out
need_file qsgw_adapter_validation.json
ls vxcs1k*_nao.txt >/dev/null
ls sks1k*_nao.txt >/dev/null

log "Stage 2: pyatb full-BZ velocity matrix for head-wing"
rm -rf pyatb_librpa_df
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
    header = [int(f.readline().strip()) for _ in range(4)]
if header != [expect_nk, 1, expect_nb, expect_nb]:
    raise SystemExit(f'bad velocity header: {header}, expected [{expect_nk}, 1, {expect_nb}, {expect_nb}]')
with band.open() as f:
    header = [int(f.readline().strip()) for _ in range(4)]
if header != [expect_nk, 1, expect_nb, expect_nb]:
    raise SystemExit(f'bad pyatb band_out header: {header}')
print('pyatb header check PASS', header)
PY
head -4 pyatb_librpa_df/velocity_matrix > pyatb_velocity_header.txt
mv OUT.ABACUS OUT.ABACUS_scf

log "Stage 3: ABACUS NSCF symmetry=-1 band path"
cp INPUT_nscf INPUT
cp KPT_nscf KPT
mpirun -np "$NP" "$ABACUS"
need_file OUT.ABACUS/running_nscf.log

log "Stage 4: preprocess ABACUS band path for LibRPA"
"$PYTHON" preprocess_abacus_for_librpa_band.py
bash rename_copy.sh
need_file band_kpath_info
ls band_KS_eigenvalue_k_*.txt >/dev/null
ls band_KS_eigenvector_k_*.txt >/dev/null
ls band_vxc_k_*.txt >/dev/null
ls band_vxck*_nao.txt >/dev/null
ls band_sk*_nao.txt >/dev/null
cp -L scf_librpa_root/band_out .
cp -L scf_librpa_root/KS_eigenvector_*.dat .
[ -e scf_librpa_root/k_path_info ] && cp -L scf_librpa_root/k_path_info . || true
cp -L scf_librpa_root/vxc_out.dat .
ln -sf vxc_out.dat vxc_out
need_file band_out
ls KS_eigenvector_*.dat >/dev/null

log "Stage 5: LibRPA qsgw_band0 first iteration, shrink+symmetry+head-wing"
cat > librpa.in <<'EOF'
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
max_iter = 1
qsgw_export_hamiltonian_for_pyatb = t
output_dir = librpa.d/
EOF
mpirun -np "$NP" "$LIBRPA" 16 1e-12
need_file QSGW_band_spin_1_1.dat
log "ALL STAGES COMPLETE"
