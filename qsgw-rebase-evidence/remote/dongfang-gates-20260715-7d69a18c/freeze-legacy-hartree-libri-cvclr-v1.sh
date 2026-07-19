#!/bin/bash

set -euo pipefail

git_bin=/usr/bin/git
source_root=/data/home/df_iopcas_bhj/software-stack/src/LibRPA-kouxiang/thirdparty/LibRI_cvclr
snapshot_root=/data/home/df_iopcas_bhj/tmp/librpa_kouxiang_hr_selfcheck_src_20260623/thirdparty/LibRI_cvclr
evidence=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/provenance/legacy-hartree-libri-cvclr-f164e202-v1
expected_base=f164e202334fff2703998fc10c401d185bca268c
expected_hartree_header_sha=9e5cb86ea3bf8cfc50f419cff32119c9fef9f56d8626bfd22dcbd4076aa44c86
expected_hartree_impl_sha=d734909b7a60d1d77235cb909c903584a2226e8b523cb5715a86f627cf7c3c93
expected_lri_header_sha=60ae23f68dd6f9fbcb769d6e4cc5f7587a776c0e0ca05e2e36cc8176df26a281

test -x "$git_bin"
test -d "$source_root/.git"
test -d "$snapshot_root/include"
test "$($git_bin -C "$source_root" rev-parse HEAD)" = "$expected_base"
test "$(sha256sum "$source_root/include/RI/physics/Hartree.h" | awk '{print $1}')" = \
  "$expected_hartree_header_sha"
test "$(sha256sum "$source_root/include/RI/ri/LRI-cal_hartree.hpp" | awk '{print $1}')" = \
  "$expected_hartree_impl_sha"
test "$(sha256sum "$source_root/include/RI/ri/LRI.h" | awk '{print $1}')" = \
  "$expected_lri_header_sha"
grep -Fq 'cal_cvcd_k_hartree' \
  "$source_root/include/RI/ri/LRI-cal_hartree.hpp"
test ! -e "$evidence"

mkdir -p "$evidence"

manifest_tree() {
  local tree=$1
  local output=$2
  (
    cd "$tree"
    find . -path './.git' -prune -o -type f -print0 \
      | LC_ALL=C sort -z \
      | xargs -0 sha256sum
  ) >"$output"
}

manifest_tree "$source_root" "$evidence/source-SHA256SUMS.txt"
manifest_tree "$snapshot_root" "$evidence/snapshot-SHA256SUMS.txt"
cmp "$evidence/source-SHA256SUMS.txt" \
  "$evidence/snapshot-SHA256SUMS.txt"

$git_bin -C "$source_root" status --short >"$evidence/git-status.txt"
$git_bin -C "$source_root" diff --binary >"$evidence/libri-dirty.patch"
$git_bin -C "$source_root" diff --quiet --cached
test -s "$evidence/git-status.txt"
test -s "$evidence/libri-dirty.patch"
printf '%s\n' "$expected_base" >"$evidence/base-commit.txt"

(
  cd "$source_root"
  tar --exclude='./.git' -czf "$evidence/libri-cvclr-source.tar.gz" .
)

{
  printf 'source_root=%s\n' "$source_root"
  printf 'snapshot_root=%s\n' "$snapshot_root"
  printf 'base_commit=%s\n' "$expected_base"
  printf 'hartree_header_sha256=%s\n' "$expected_hartree_header_sha"
  printf 'hartree_impl_sha256=%s\n' "$expected_hartree_impl_sha"
  printf 'lri_header_sha256=%s\n' "$expected_lri_header_sha"
  printf 'frozen_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$evidence/PROVENANCE.txt"

sha256sum \
  "$evidence/base-commit.txt" \
  "$evidence/git-status.txt" \
  "$evidence/libri-dirty.patch" \
  "$evidence/source-SHA256SUMS.txt" \
  "$evidence/snapshot-SHA256SUMS.txt" \
  "$evidence/libri-cvclr-source.tar.gz" \
  "$evidence/PROVENANCE.txt" \
  >"$evidence/OUTPUT_SHA256SUMS.txt"

touch "$evidence/COMPLETE"
printf 'LEGACY_HARTREE_LIBRI_FREEZE=PASS evidence=%s\n' "$evidence"
