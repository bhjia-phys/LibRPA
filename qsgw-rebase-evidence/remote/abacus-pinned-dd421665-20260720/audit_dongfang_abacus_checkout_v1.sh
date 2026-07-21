#!/usr/bin/env bash

set -euo pipefail

source_repo=/data/home/df_iopcas_bhj/software-stack/src/abacus-old-libri
binary=$source_repo/build/abacus_3p
cmake_cache=$source_repo/build/CMakeCache.txt
run_root=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-audit-20260720-v1
pinned_commit=dd4216653386d32f79e3219f3ea5dd2d229c1c5a
pinned_repo=https://github.com/AroundPeking/abacus-develop.git
pinned_branch=master_ghj

test ! -e "$run_root"
test -d "$source_repo/.git"
test -x "$binary"
test -f "$cmake_cache"

mkdir -p "$run_root"
cp "$0" "$run_root/audit_dongfang_abacus_checkout_v1.sh"

git -C "$source_repo" rev-parse HEAD >"$run_root/current-head.txt"
git -C "$source_repo" remote -v >"$run_root/git-remotes.txt"
git -C "$source_repo" status --short --branch --untracked-files=all \
  >"$run_root/git-status.txt"
git -C "$source_repo" diff --binary >"$run_root/current-dirty.patch"
git -C "$source_repo" diff --cached --binary \
  >"$run_root/current-staged.patch"

pinned_object_available=false
if git -C "$source_repo" cat-file -e "$pinned_commit^{commit}" 2>/dev/null
then
  pinned_object_available=true
  git -C "$source_repo" show -s --format=fuller "$pinned_commit" \
    >"$run_root/pinned-commit.txt"
  git -C "$source_repo" branch -a --contains "$pinned_commit" \
    >"$run_root/pinned-containing-branches.txt"
  git -C "$source_repo" rev-parse "$pinned_commit^{tree}" \
    >"$run_root/pinned-tree.txt"
  git -C "$source_repo" diff --stat "$pinned_commit" \
    >"$run_root/current-vs-pinned-stat.txt"
  git -C "$source_repo" diff --binary "$pinned_commit" \
    >"$run_root/current-vs-pinned.patch"
else
  printf 'PINNED_OBJECT_NOT_AVAILABLE\n' >"$run_root/pinned-commit.txt"
fi

sha256sum "$binary" >"$run_root/abacus-executable.sha256"
sha256sum "$cmake_cache" >"$run_root/CMakeCache.sha256"
cp "$cmake_cache" "$run_root/CMakeCache.txt"
ldd "$binary" >"$run_root/abacus-executable.ldd.txt" 2>&1
grep -E \
  '^(CMAKE_(BUILD_TYPE|CXX_COMPILER|C_COMPILER|Fortran_COMPILER|GENERATOR|CXX_FLAGS|C_FLAGS)|ENABLE_|USE_|BUILD_|MPI_|MKL_|OpenMP_)' \
  "$cmake_cache" >"$run_root/CMakeCache.selected.txt" || true

current_head=$(cat "$run_root/current-head.txt")
binary_sha=$(awk '{print $1}' "$run_root/abacus-executable.sha256")
cache_sha=$(awk '{print $1}' "$run_root/CMakeCache.sha256")
tracked_dirty_count=$(git -C "$source_repo" status --porcelain \
  --untracked-files=no | wc -l)
untracked_count=$(git -C "$source_repo" status --porcelain \
  --untracked-files=all | awk '$1 == "??" {count++} END {print count+0}')

printf '%s\n' \
  'analysis=abacus_pinned_baseline_checkout_audit' \
  'acceptance_gate=false' \
  'host=dongfang-login-readonly-audit' \
  "source_repo=$source_repo" \
  "requested_repo=$pinned_repo" \
  "requested_branch=$pinned_branch" \
  "requested_commit=$pinned_commit" \
  "current_head=$current_head" \
  "pinned_object_available=$pinned_object_available" \
  "tracked_dirty_count=$tracked_dirty_count" \
  "untracked_count=$untracked_count" \
  "binary=$binary" \
  "binary_sha256=$binary_sha" \
  "cmake_cache_sha256=$cache_sha" \
  "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >"$run_root/PROVENANCE.txt"

(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name AUDIT_COMPLETE -print0 |
    sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/AUDIT_COMPLETE"
cat "$run_root/PROVENANCE.txt"
