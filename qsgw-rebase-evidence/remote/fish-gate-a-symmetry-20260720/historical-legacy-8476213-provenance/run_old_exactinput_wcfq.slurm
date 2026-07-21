#!/bin/bash
#SBATCH -J si-old-wcfq
#SBATCH -p 48cp3
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 24:00:00
#SBATCH -o /ssd/work/df_iopcas_bhj/qsgw-newarch-runs/si-k444-qsgw-old-exactinput-wcfq-20260704-014854/librpa/logs/old-exactinput-wcfq-%j.out
#SBATCH -e /ssd/work/df_iopcas_bhj/qsgw-newarch-runs/si-k444-qsgw-old-exactinput-wcfq-20260704-014854/librpa/logs/old-exactinput-wcfq-%j.err
source /data/app/intel/oneapi-2024.2/setvars.sh --force >/dev/null 2>&1
set -eo pipefail
cd /ssd/work/df_iopcas_bhj/qsgw-newarch-runs/si-k444-qsgw-old-exactinput-wcfq-20260704-014854/librpa
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export I_MPI_PIN_DOMAIN=omp
export LIBRI_DETERMINISTIC_REDUCTION=1
export LD_LIBRARY_PATH="/data/home/df_iopcas_bhj/software-stack/src/LibRPA_codex/build/src:/data/home/df_iopcas_bhj/software-stack/src/LibRPA_codex/build/qsgw:${LD_LIBRARY_PATH:-}"
export LIBRPA_WCFQ_DUMP=1
echo RUN_DIR=$PWD
echo EXE=/data/home/df_iopcas_bhj/software-stack/src/LibRPA_codex/build/chi0_main.exe
grep -E '^[[:space:]]*(task|output_gw_sigc_mat_rf|debug|use_abacus_gw_symmetry|use_symmetry_gw)' librpa.in || true
mpirun -np 1 /data/home/df_iopcas_bhj/software-stack/src/LibRPA_codex/build/chi0_main.exe 16 1e-12
test -s QSGW_band_spin_1_1.dat
test -s hrs1_nao_qsgw_iter_0001.csr
test -s librpa.d/Wcfq_ifreq_0_iq_0_I_0_J_0_id_0.mtx
