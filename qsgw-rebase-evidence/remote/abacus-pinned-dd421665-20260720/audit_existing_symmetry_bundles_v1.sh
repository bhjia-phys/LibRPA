#!/usr/bin/env bash
set -euo pipefail

roots=(
  /ssd/work/df_iopcas_bhj/si-k444-qsgw-sym-newarch-exactpyatb-20260630-032256/dataset
  /ssd/work/df_iopcas_bhj/qsgw-newarch-runs/si-k444-qsgw-newarch-previnput-wcfq-current-20260704-013610/librpa
  /data/home/df_iopcas_bhj/ai-runs/si-qsgw-k444-headwing-kconv-20260517-203305/base_iter1
)

echo "host=$(hostname)"
echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo '--- matching directories'
find /ssd/work/df_iopcas_bhj -maxdepth 4 -type d \
  \( -iname '*si*k444*sym*' -o -iname '*si*k444*qsgw*' \) \
  2>/dev/null | sort | head -n 300
find /data/home/df_iopcas_bhj/ai-runs -maxdepth 3 -type d \
  \( -iname '*si*k444*sym*' -o -iname '*si*k444*qsgw*' \) \
  2>/dev/null | sort | head -n 300

for root in "${roots[@]}"; do
  echo "===== root=$root ====="
  if ! test -d "$root"; then
    echo MISSING
    continue
  fi

  printf 'regular_files|'
  find "$root" -maxdepth 1 -type f | wc -l
  printf 'symlinks|'
  find "$root" -maxdepth 1 -type l | wc -l
  printf 'broken_symlinks|'
  find "$root" -maxdepth 1 -xtype l | wc -l

  echo '-- pattern counts'
  for pattern in \
    'KS_eigenvector_*.dat' 'band_KS_eigenvalue_k_*.txt' \
    'band_KS_eigenvector_k_*.txt' 'vxcs1k*_nao.txt' \
    'Cs_data_*' 'Cs_shrinked_data_*' 'coulomb_mat_*' 'coulomb_cut_*' \
    'irreducible_sector.txt' 'symrot_k.txt' 'symrot_R.txt' \
    'basis_index.txt' 'bz_sampling_out'; do
    printf '%s|' "$pattern"
    find "$root" -maxdepth 2 \( -type f -o -type l \) -name "$pattern" | wc -l
  done

  echo '-- key files'
  for name in \
    librpa.in band_out stru_out bz_sampling_out vxc_out.dat \
    basis_wfc_out basis_aux_out basis_index.txt \
    irreducible_sector.txt symrot_k.txt symrot_R.txt \
    qsgw_input.contract INPUT INPUT_scf KPT KPT_scf STRU; do
    path=$root/$name
    if [[ -e "$path" || -L "$path" ]]; then
      target=$(readlink -f "$path" || true)
      if test -n "$target" && test -f "$target"; then
        size=$(stat -Lc '%s' "$path")
        sha=$(sha256sum "$target" | awk '{print $1}')
        printf '%s|%s|%s|%s\n' "$name" "$target" "$size" "$sha"
      else
        printf '%s|BROKEN|%s\n' "$name" "$target"
      fi
    else
      printf '%s|MISSING\n' "$name"
    fi
  done

  echo '-- symlink target directories'
  find "$root" -maxdepth 1 -type l -print0 | \
    while IFS= read -r -d '' path; do
      dirname "$(readlink -f "$path")"
    done | sort | uniq -c | sort -nr | head -n 30 || true

  echo '-- selected file heads'
  for name in librpa.in bz_sampling_out irreducible_sector.txt \
    symrot_k.txt symrot_R.txt; do
    path=$root/$name
    if [[ -f "$path" || -L "$path" ]]; then
      echo "--- $name"
      sed -n '1,100p' "$path"
    fi
  done

  echo '-- nearby traces and manifests'
  find "$root" "$(dirname "$root")" -maxdepth 2 \
    \( -type f -o -type l \) \
    \( -name 'qsgw_iterations.dat' -o -name 'qsgw_eigenvalues.dat' \
       -o -name 'homo_lumo_vs_iterations.dat' -o -name 'progress.txt' \
       -o -name '*PROVENANCE*' -o -name '*SHA256SUMS*' \
       -o -name 'librpa.in' \) -print 2>/dev/null | sort | head -n 300
done

echo AUDIT_COMPLETE
exit 0
