#!/usr/bin/env bash
set -euo pipefail

root=${1:-/data/home/df_iopcas_bhj/ai-runs/si-qsgw-k444-headwing-kconv-20260517-203305/base_iter1}

printf 'root=%s\n' "$root"
test -d "$root"

for name in \
    case_manifest.txt INPUT_scf KPT_scf INPUT_nscf KPT_nscf STRU \
    run_base_iter1.sh rename_copy.sh OUT.ABACUS_scf/INPUT.info \
    OUT.ABACUS/INPUT.info OUT.ABACUS/KPT.info; do
    path="$root/$name"
    printf '\n===== %s =====\n' "$name"
    if [[ -f "$path" ]]; then
        sha256sum "$path"
        sed -n '1,320p' "$path"
    else
        printf 'MISSING\n'
    fi
done

printf '\n%s\n' '===== producer artifact inventory ====='
for pattern in \
    'band_out' 'stru_out' 'bz_sampling_out' 'vxc_out.dat' \
    'KS_eigenvector_*.dat' 'Cs_data_*' 'Cs_shrinked_data_*' \
    'coulomb_cut_*' 'coulomb_mat_*' 'vxcs*k*_nao.txt' \
    'Vxc_R_spin*.csr' 'data-*'; do
    count=$(find "$root" -maxdepth 1 -type f -name "$pattern" | wc -l)
    bytes=$(find "$root" -maxdepth 1 -type f -name "$pattern" -printf '%s\n' \
        | awk '{sum += $1} END {print sum + 0}')
    printf '%s|count=%s|bytes=%s\n' "$pattern" "$count" "$bytes"
done

printf '\n%s\n' '===== representative artifact hashes ====='
for pattern in \
    'band_out' 'stru_out' 'bz_sampling_out' 'vxc_out.dat' \
    'KS_eigenvector_*.dat' 'Cs_data_*' 'coulomb_cut_*' 'coulomb_mat_*' \
    'vxcs*k*_nao.txt' 'Vxc_R_spin*.csr'; do
    find "$root" -maxdepth 1 -type f -name "$pattern" -print0 \
        | sort -z | head -z -2 | xargs -0 -r sha256sum
done
