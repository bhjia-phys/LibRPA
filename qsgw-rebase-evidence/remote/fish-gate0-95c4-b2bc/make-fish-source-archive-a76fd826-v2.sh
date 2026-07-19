#!/usr/bin/env bash
set -euo pipefail

run_root=/home/bhj/ai-runs/librpa-qsgw-source-archive-a76fd826-v2-20260716T0852
repo=/tmp/librpa-qsgw-cb294020-84e6b0a/repo.git
archive=$run_root/librpa-a76fd826-linux-git-archive.tar.gz
expected_commit=a76fd826eed3929185ea47cfcb992075a72f2b77
candidate_ref=refs/heads/codex/qsgw-independent-upstream-95c4-20260716

test ! -e "$run_root"
test -d "$repo"
test "$(git --git-dir="$repo" rev-parse "$candidate_ref")" = \
  "$expected_commit"

mkdir -p "$run_root"
git --git-dir="$repo" archive \
  --format=tar.gz \
  --prefix=source/ \
  --output="$archive" \
  "$expected_commit" -- \
  CMakeLists.txt cmake include driver src thirdparty

archive_sha=$(sha256sum "$archive" | awk '{print $1}')
set +o pipefail
archive_commit=$(gzip -dc "$archive" | git get-tar-commit-id)
set -o pipefail
archive_entries=$(tar -tzf "$archive" | wc -l)
test "$archive_commit" = "$expected_commit"
test "$archive_entries" -eq 1838
test "$(tar -tzf "$archive" | awk '/(^|\/)\.git(\/|$)/ {count++} END {print count+0}')" -eq 0

for path in \
  src/qsgw/fixed_basis.cpp \
  src/qsgw/fixed_basis.h \
  src/test/test_qsgw_fixed_basis.cpp \
  src/test/test_qsgw_fixed_basis_mpi.cpp \
  driver/tasks/qsgw.cpp \
  src/qsgw/occupation.cpp \
  src/core/gw.cpp \
  src/core/gw.h \
  src/core/exx.cpp \
  src/core/exx.h \
  src/core/dielecmodel.h; do
  digest=$(tar -xOzf "$archive" "source/$path" | sha256sum | awk '{print $1}')
  printf '%s  %s\n' "$digest" "$path"
done >"$run_root/critical-blobs.sha256"

grep -Fqx \
  '1517f60c6d1229442b653ac48523ad9d259cf87d964965db8de6f0a68ef20412  src/qsgw/fixed_basis.cpp' \
  "$run_root/critical-blobs.sha256"
grep -Fqx \
  '3e3637e4e774b6c43c249de4fa7b6b0e2123331a9ff9dc17f8af542502578ff8  src/qsgw/fixed_basis.h' \
  "$run_root/critical-blobs.sha256"
grep -Fqx \
  '1f1e5837743ea641175e6bc32a49308a30c1757af90cfe71196695a1a8876be6  src/test/test_qsgw_fixed_basis.cpp' \
  "$run_root/critical-blobs.sha256"
grep -Fqx \
  'fc5a2680406313ac152c32b5fee569df2aefc72ad016e13f203d4cc8b1da5060  src/test/test_qsgw_fixed_basis_mpi.cpp' \
  "$run_root/critical-blobs.sha256"
grep -Fqx \
  'bc4ae1af3883f8f976cd0d1526d703895b5581de6ab2e05b98f2af9efb9a9326  src/core/gw.cpp' \
  "$run_root/critical-blobs.sha256"
grep -Fqx \
  '536a251dfb60ebe5c20f964b2c646514d12c82907c998f9ad4e8a4d212c4483b  src/core/exx.cpp' \
  "$run_root/critical-blobs.sha256"

cp "$0" "$run_root/make-fish-source-archive-a76fd826-v2.sh"
{
  printf 'artifact=linux-git-source-archive\n'
  printf 'source_commit=%s\n' "$expected_commit"
  printf 'source_ref=%s\n' "$candidate_ref"
  printf 'archive=%s\n' "$archive"
  printf 'archive_sha256=%s\n' "$archive_sha"
  printf 'archive_embedded_commit=%s\n' "$archive_commit"
  printf 'archive_entry_count=%s\n' "$archive_entries"
  printf 'archive_selection=%s\n' \
    'CMakeLists.txt cmake include driver src thirdparty'
  printf 'producer_host=%s\n' "$(hostname)"
  printf 'producer_git=%s\n' "$(git --version)"
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$run_root/PROVENANCE.txt"

(
  cd "$run_root"
  sha256sum \
    PROVENANCE.txt \
    critical-blobs.sha256 \
    make-fish-source-archive-a76fd826-v2.sh \
    librpa-a76fd826-linux-git-archive.tar.gz \
    >SHA256SUMS.txt
)
touch "$run_root/COMPLETE"
printf 'FISH_SOURCE_ARCHIVE_A76FD826=PASS\n'
