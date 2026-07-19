#!/usr/bin/env bash
# Generate a small full-BZ ABACUS Si dataset (2x2x2 MP grid) for the
# no-symmetry QSGW regression case, following the si-k444-qsgw-final
# generation chain (SCF symmetry=1 -> NSCF symmetry=-1 with RPA outputs).
#SBATCH --job-name=abacus-si-k222-gen
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/abacus-si-k222-gen-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/abacus-si-k222-gen-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
source_dataset=/data/home/df_iopcas_bhj/qsgw-runs/si-k444-qsgw-final-c70ad2b-v9-20260713/dataset
abacus=/data/home/df_iopcas_bhj/software-stack/src/abacus-old-libri/build/abacus_3p
work=$root/runs/abacus-si-k222-gen-$SLURM_JOB_ID

record_failure() {
  local rc=$?
  if [[ -d ${work:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$work/FAILED"
  fi
  exit "$rc"
}
trap record_failure ERR

test ! -e "$work"
test -x "$abacus"
mkdir -p "$work/scf" "$work/nscf"
cp "$0" "$work/abacus-si-k222-gen.slurm"

for f in STRU Si_ONCV_PBE-1.0.upf Si_gga_8au_100Ry_3s3p2d.orb \
  Si_3s3p2d1f1g_pca1e-6.abfs; do
  cp "$source_dataset/$f" "$work/scf/"
  cp "$source_dataset/$f" "$work/nscf/"
done

cat >"$work/scf/KPT" <<'EOF'
K_POINTS
0
Gamma
2 2 2 0 0 0
EOF
cp "$work/scf/KPT" "$work/nscf/KPT"

cat >"$work/scf/INPUT" <<'EOF'
INPUT_PARAMETERS
pseudo_dir                      ./
orbital_dir                     ./
calculation                     scf
ntype                           1
nbands                          44
basis_type                      lcao
ks_solver                       genelpa
symmetry                        1
ecutwfc                         130
scf_thr                         1.0e-7
scf_nmax                        100
smearing_method                 gaussian
smearing_sigma                  1e-4
out_chg                         1
out_mat_hs2                     1
out_mat_r                       1
out_mat_xc                      1
rpa                             1
exx_separate_loop               1
exx_pca_threshold               10
exx_singularity_correction      massidda
exx_ccp_rmesh_times             3
rpa_ccp_rmesh_times             6
exx_real_number                 1
exx_cs_inv_thr                  1e-5
shrink_abfs_pca_thr             1e-4
shrink_lu_inv_thr               1e-3
latname                         user_defined_lattice
EOF

cat >"$work/nscf/INPUT" <<'EOF'
INPUT_PARAMETERS
pseudo_dir                      ./
orbital_dir                     ./
calculation                     nscf
ntype                           1
nbands                          44
basis_type                      lcao
ks_solver                       genelpa
symmetry                        -1
init_chg                        file
ecutwfc                         130
smearing_method                 gaussian
smearing_sigma                  1e-4
out_mat_xc                      1
out_eband_terms                 1
out_wfc_lcao                    1
exx_separate_loop               1
exx_pca_threshold             10
exx_singularity_correction      massidda
exx_ccp_rmesh_times             2
rpa_ccp_rmesh_times             5
latname                         user_defined_lattice
out_mat_hs                      1
out_mat_hs2                     1
EOF

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

(
  cd "$work/scf"
  "$abacus" >abacus.stdout 2>abacus.stderr
)
test -s "$work/scf/OUT.ABACUS/running_scf.log" || \
  test -s "$work/scf/OUT.ABACUS/INPUT"
cp -a "$work/scf/OUT.ABACUS" "$work/nscf/OUT.ABACUS"
(
  cd "$work/nscf"
  "$abacus" >abacus.stdout 2>abacus.stderr
)

find "$work" -maxdepth 3 -type f | sort >"$work/files.txt"
wc -l "$work/files.txt"
echo ABACUS_SI_K222_GEN_DONE
