#!/usr/bin/env bash
set -euo pipefail

exact=/ssd/work/df_iopcas_bhj/qsgw-newarch-runs/si-k444-qsgw-sym-newarch-exactpyatb-20260630-032256
dataset=$exact/dataset
stitched=/ssd/work/df_iopcas_bhj/qsgw-newarch-runs/si-k444-qsgw-newarch-previnput-wcfq-current-20260704-013610/librpa
producer=/data/home/df_iopcas_bhj/ai-runs/si-qsgw-k444-headwing-kconv-20260517-203305/base_iter1

echo "host=$(hostname)"
echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

for root in "$exact" "$dataset"; do
  echo "===== inventory=$root ====="
  test -d "$root"
  find "$root" -maxdepth 1 \( -type f -o -type l \) \
    -printf '%y|%f|%s|%l\n' | sort
done

echo '--- exact dataset hashes'
find "$dataset" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum

echo '--- exact parent small text'
for name in librpa.in run.sh submit.sh '*.slurm' '*.py' '*.txt' '*.out' '*.err'; do
  find "$exact" -maxdepth 1 -type f -name "$name" -size -2M -print0 | \
    sort -z | while IFS= read -r -d '' path; do
      echo "===== file=$path ====="
      sha256sum "$path"
      sed -n '1,260p' "$path"
    done
done

echo '--- four exact-derived links in stitched dataset'
find "$stitched" -maxdepth 1 -type l -print0 | \
  while IFS= read -r -d '' path; do
    target=$(readlink -f "$path")
    case "$target" in
      "$dataset"/*)
        printf '%s|%s|' "$(basename "$path")" "$target"
        sha256sum "$target" | awk '{print $1}'
        ;;
    esac
  done | sort

echo '--- producer versus stitched top-level counts'
for root in "$producer" "$stitched"; do
  echo "root=$root"
  for pattern in 'KS_eigenvector_*.dat' 'vxcs1k*_nao.txt' \
    'Cs_data_*' 'Cs_shrinked_data_*' 'coulomb_mat_*' 'coulomb_cut_*'; do
    printf '%s|' "$pattern"
    find "$root" -maxdepth 1 \( -type f -o -type l \) -name "$pattern" | wc -l
  done
done

echo '--- producer/exact mapping summary'
sed -n '1,24p' "$dataset/bz_sampling_out"
sed -n '1,80p' "$producer/irreducible_sector.txt"
for name in irreducible_sector.txt symrot_k.txt symrot_R.txt; do
  sha256sum "$producer/$name"
done

echo '--- historical symmetry run candidates'
for run in \
  /ssd/work/df_iopcas_bhj/qsgw-newarch-runs/si-k444-qsgw-sym-newarch-restart9-nomix-dump-20260630-024300 \
  /ssd/work/df_iopcas_bhj/qsgw-newarch-runs/si-k444-qsgw-sym-newarch-restart9-fixedbasis-20260630-030234 \
  /ssd/work/df_iopcas_bhj/qsgw-newarch-runs/si-k444-qsgw-sym-newarch-exactpyatb-20260630-032256 \
  /ssd/work/df_iopcas_bhj/qsgw-newarch-runs/si-k444-qsgw-old-exactinput-wcfq-20260704-014854; do
  echo "===== run=$run ====="
  test -d "$run" || { echo MISSING; continue; }
  find "$run" -maxdepth 2 -type f \
    \( -name 'librpa.in' -o -name 'homo_lumo_vs_iterations.dat' \
       -o -name 'qsgw_iterations.dat' -o -name 'qsgw_eigenvalues.dat' \
       -o -name 'progress.txt' -o -name '*.sh' -o -name '*.slurm' \
       -o -name '*PROVENANCE*' -o -name '*SHA256SUMS*' \) \
    -printf '%p|%s\n' | sort
  for file in "$run/librpa/librpa.in" "$run/librpa.in" \
    "$run/librpa/homo_lumo_vs_iterations.dat" \
    "$run/homo_lumo_vs_iterations.dat"; do
    if test -f "$file"; then
      echo "--- $file"
      sha256sum "$file"
      sed -n '1,180p' "$file"
    fi
  done
done

echo AUDIT_COMPLETE
exit 0
