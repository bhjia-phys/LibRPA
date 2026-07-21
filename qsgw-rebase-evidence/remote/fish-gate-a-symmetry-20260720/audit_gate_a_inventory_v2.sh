#!/usr/bin/env bash
set -euo pipefail

sym_root=/tmp/si444_sym_test
dataset=$sym_root/dataset
bare=/tmp/librpa-qsgw-cb294020-84e6b0a/repo.git
legacy=e08f4a130df7661e9ac355b9be45fb2bf9c3ed01
candidate=b7273e13c77d5ea781f192cea3c4201710b6f9fa
reader_build=/home/bhj/ai-runs/librpa-qsgw-reader-binding-20260720-v2

echo "host=$(hostname)"
echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo '--- symmetry run top-level'
test -d "$sym_root"
find "$sym_root" -maxdepth 1 -type f -printf '%f|%s\n' | sort
sha256sum "$sym_root"/librpa.in "$sym_root"/qsgw_iterations.dat \
  "$sym_root"/qsgw_eigenvalues.dat "$sym_root"/qsgw_matrices.dat

echo '--- symmetry run librpa.in'
cat "$sym_root/librpa.in"

echo '--- symmetry trace metadata and rows'
sed -n '1,40p' "$sym_root/qsgw_iterations.dat"
grep -E 'symmetry|QSGW iteration|HOMO|LUMO|gap|Input contract' \
  "$sym_root/librpa.out" | tail -n 80 || true

echo '--- dataset pattern counts'
for pattern in \
  'KS_eigenvector_*.dat' 'band_KS_eigenvalue_k_*.txt' \
  'band_KS_eigenvector_k_*.txt' 'Cs_data_*.txt' \
  'Cs_shrinked_data_*.txt' 'coulomb_mat_*.txt' \
  'coulomb_cut_*.txt' 'vxcs1k*_nao.txt' 'sks1k*_nao.txt'; do
  printf '%s|' "$pattern"
  find "$dataset" -maxdepth 2 -type f -name "$pattern" | wc -l
done
printf 'symlinks|'
find "$dataset" -type l | wc -l

echo '--- dataset key files'
find "$dataset" -maxdepth 1 -type f \
  \( -name 'band_out' -o -name 'stru_out' -o -name 'bz_sampling_out' \
     -o -name 'basis_*_out' -o -name 'vxc_out.dat' \
     -o -name 'qsgw_input*.contract' \) \
  -printf '%f|%s\n' | sort
for file in band_out stru_out bz_sampling_out basis_wfc_out basis_aux_out \
  vxc_out.dat qsgw_input.contract; do
  test -f "$dataset/$file" && sha256sum "$dataset/$file"
done

echo '--- bz_sampling_out head'
sed -n '1,80p' "$dataset/bz_sampling_out"

echo '--- legacy QSGW tree paths'
git --git-dir="$bare" ls-tree -r --name-only "$legacy" | \
  grep -E 'qsgw|inputfile|symmetr|task_gw|task_g0w0' || true

echo '--- legacy symmetry symbols'
git --git-dir="$bare" grep -n -I -E \
  'use_abacus_(gw|exx)_symmetry|use_symmetry_(gw|exx|rpa)|symmetry' \
  "$legacy" -- src driver 2>/dev/null | head -n 500 || true

echo '--- candidate symmetry symbols'
git --git-dir="$bare" grep -n -I -E \
  'use_abacus_(gw|exx)_symmetry|use_symmetry_(gw|exx|rpa)|symmetry' \
  "$candidate" -- src/qsgw driver/tasks/qsgw.cpp driver/inputfile.cpp \
  2>/dev/null | head -n 500 || true

echo '--- reader-binding build'
if test -d "$reader_build"; then
  find "$reader_build" -maxdepth 3 -type f \
    \( -name chi0_main.exe -o -name COMPLETE -o -name '*PROVENANCE*' \
       -o -name '*SHA256SUMS*' -o -name CMakeCache.txt \) \
    -printf '%p|%s\n' | sort
  find "$reader_build" -maxdepth 4 -type f -name chi0_main.exe \
    -exec sha256sum {} +
else
  echo "MISSING:$reader_build"
fi

echo '--- branch tips'
git --git-dir="$bare" show-ref \
  refs/heads/codex/scheme-a-legacy-oracle-exact-20260712 \
  refs/heads/codex/qsgw-symmetry-no-headwing-42d-20260720

echo INVENTORY_COMPLETE
