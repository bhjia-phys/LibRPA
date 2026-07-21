#!/usr/bin/env bash
set -euo pipefail

source_root=/ssd/work/df_iopcas_bhj/qsgw-newarch-runs/si-k444-qsgw-newarch-previnput-wcfq-current-20260704-013610/librpa

echo "host=$(hostname)"
echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "source_root=$source_root"
test -d "$source_root"

echo '--- required scalar files'
for name in \
  stru_out band_out bz_sampling_out basis_wfc_out basis_aux_out \
  basis_aux_shrink_out irreducible_sector.txt symrot_k.txt symrot_R.txt; do
  path=$source_root/$name
  test -e "$path"
  printf '%s|type=%s|size=%s|target=%s|sha256=' \
    "$name" \
    "$(stat -c '%F' "$path")" \
    "$(stat -Lc '%s' "$path")" \
    "$(readlink -f "$path")"
  sha256sum "$path" | awk '{print $1}'
done

echo '--- candidate reader prefix sets'
for prefix in \
  KS_eigenvector vxcs1k Cs_data Cs_shrinked_data shrink_sinvS_ \
  coulomb_mat coulomb_cut; do
  echo "prefix=$prefix"
  find "$source_root" -maxdepth 1 \( -type f -o -type l \) \
    -name "${prefix}*" -printf '%f\0' | sort -z | \
    while IFS= read -r -d '' name; do
      path=$source_root/$name
      printf '%s|type=%s|size=%s|target=%s|sha256=' \
        "$name" \
        "$(stat -c '%F' "$path")" \
        "$(stat -Lc '%s' "$path")" \
        "$(readlink -f "$path")"
      sha256sum "$path" | awk '{print $1}'
    done
done

echo '--- mapping header'
sed -n '1,24p' "$source_root/bz_sampling_out"

echo '--- irreducible sectors'
sed -n '1,120p' "$source_root/irreducible_sector.txt"

echo '--- source link accounting'
printf 'regular_files='
find "$source_root" -maxdepth 1 -type f | wc -l
printf 'symlinks='
find "$source_root" -maxdepth 1 -type l | wc -l
printf 'broken_symlinks='
find "$source_root" -maxdepth 1 -xtype l | wc -l

echo INSPECT_COMPLETE
exit 0
