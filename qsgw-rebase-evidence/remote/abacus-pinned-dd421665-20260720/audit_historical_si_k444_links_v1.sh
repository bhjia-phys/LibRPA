#!/usr/bin/env bash
set -euo pipefail

dataset=${1:-/ssd/work/df_iopcas_bhj/qsgw-newarch-runs/si-k444-qsgw-newarch-previnput-wcfq-current-20260704-013610/librpa}

printf 'dataset=%s\n' "$dataset"
test -d "$dataset"

printf '%s\n' '--- representative paths ---'
for name in \
    librpa.in band_out stru_out bz_sampling_out vxc_out.dat basis_index.txt \
    coulomb_mat_0_0_0_0 Cs_data_0_0_0_0; do
    path="$dataset/$name"
    if [[ -e "$path" || -L "$path" ]]; then
        target=$(readlink -f "$path")
        size=$(stat -Lc '%s' "$path")
        sha=$(sha256sum "$target" | awk '{print $1}')
        printf '%s|%s|%s|%s\n' "$name" "$target" "$size" "$sha"
    else
        printf '%s|MISSING\n' "$name"
    fi
done

for path in $(find "$dataset" -maxdepth 1 -name 'KS_eigenvector_*.dat' -print | sort | head -2); do
    name=$(basename "$path")
    target=$(readlink -f "$path")
    size=$(stat -Lc '%s' "$path")
    sha=$(sha256sum "$target" | awk '{print $1}')
    printf '%s|%s|%s|%s\n' "$name" "$target" "$size" "$sha"
done

printf '%s\n' '--- symlink target directory counts ---'
find "$dataset" -maxdepth 1 -type l -print0 \
    | while IFS= read -r -d '' path; do dirname "$(readlink -f "$path")"; done \
    | sort | uniq -c | sort -nr | head -20

printf '%s\n' '--- local regular files ---'
find "$dataset" -maxdepth 1 -type f -printf '%f|%s\n' | sort

band_target=$(readlink -f "$dataset/band_out")
producer_dir=$(dirname "$band_target")
printf 'producer_dir=%s\n' "$producer_dir"
printf '%s\n' '--- nearby producer inputs and scripts ---'
find "$producer_dir" "$(dirname "$producer_dir")" -maxdepth 2 \
    \( -type f -o -type l \) \
    \( -name 'INPUT*' -o -name 'KPT*' -o -name 'STRU*' -o \
       -name '*.slurm' -o -name '*.sh' -o -name 'provenance*' -o \
       -name '*manifest*' \) -print 2>/dev/null | sort -u | head -240
