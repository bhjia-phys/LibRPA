#!/usr/bin/env bash
set -euo pipefail

source_run=/home/bhj/ai-runs/librpa-qsgw-source-archive-a76fd826-v2-20260716T0852
source_archive=$source_run/librpa-a76fd826-linux-git-archive.tar.gz
run_root=/home/bhj/ai-runs/librpa-qsgw-source-archive-validation-a76fd826-v3-20260716T0856
archive=$run_root/librpa-a76fd826-linux-git-archive.tar.gz
expected_commit=a76fd826eed3929185ea47cfcb992075a72f2b77
expected_archive_sha=521936e27b18c325a79acedcecd7f0541d54347f3a0f831b89e4517d4104f641

test ! -e "$run_root"
test -f "$source_archive"
test "$(sha256sum "$source_archive" | awk '{print $1}')" = \
  "$expected_archive_sha"

mkdir -p "$run_root"
cp "$source_archive" "$archive"
test "$(sha256sum "$archive" | awk '{print $1}')" = \
  "$expected_archive_sha"

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

cat >"$run_root/expected-critical-blobs.sha256" <<'EOF'
1517f60c6d1229442b653ac48523ad9d259cf87d964965db8de6f0a68ef20412  src/qsgw/fixed_basis.cpp
3e3637e4e774b6c43c249de4fa7b6b0e2123331a9ff9dc17f8af542502578ff8  src/qsgw/fixed_basis.h
1f1e5837743ea641175e6bc32a49308a30c1757af90cfe71196695a1a8876be6  src/test/test_qsgw_fixed_basis.cpp
fc5a2680406313ac152c32b5fee569df2aefc72ad016e13f203d4cc8b1da5060  src/test/test_qsgw_fixed_basis_mpi.cpp
2daf2c43001ef062cba9e29511530815346d0ba0c171ac472187aafc9c971f15  driver/tasks/qsgw.cpp
4dd9cac87562da097425343076e21c00847ee2de18afbf013fa02145ebf5d7dc  src/qsgw/occupation.cpp
304cd95fe903b51dea8c407b9f7ff6b25d6b24f49faeb6060b2cedf9db185bd5  src/core/gw.cpp
b14bb433aace2e869e8267147c051dfc4be46674c333ad0e58df1fd0022ad333  src/core/gw.h
989810fd126c85cc24bd1fa8df8e3707f06cd4f456a731edfd8a79a668d9540f  src/core/exx.cpp
07617d506f3421334f2a2175e08315212b744f49325cbae8e0ffa112b67a9846  src/core/exx.h
f32c50778105e046d2ea635cd9d2969b458eff4120ab8660a808150dd94ea899  src/core/dielecmodel.h
EOF
cmp "$run_root/expected-critical-blobs.sha256" \
  "$run_root/critical-blobs.sha256"

cp "$0" "$run_root/validate-fish-source-archive-a76fd826-v3.sh"
{
  printf 'artifact=validated-linux-git-source-archive\n'
  printf 'source_commit=%s\n' "$expected_commit"
  printf 'source_failed_generator_run=%s\n' "$source_run"
  printf 'source_failure_scope=validator_expectations_only_archive_content_revalidated\n'
  printf 'archive=%s\n' "$archive"
  printf 'archive_sha256=%s\n' "$expected_archive_sha"
  printf 'archive_embedded_commit=%s\n' "$archive_commit"
  printf 'archive_entry_count=%s\n' "$archive_entries"
  printf 'archive_line_endings=git_blob_lf_bytes\n'
  printf 'producer_host=%s\n' "$(hostname)"
  printf 'producer_git=%s\n' "$(git --version)"
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$run_root/PROVENANCE.txt"

(
  cd "$run_root"
  sha256sum \
    PROVENANCE.txt \
    critical-blobs.sha256 \
    expected-critical-blobs.sha256 \
    validate-fish-source-archive-a76fd826-v3.sh \
    librpa-a76fd826-linux-git-archive.tar.gz \
    >SHA256SUMS.txt
  sha256sum --check --quiet SHA256SUMS.txt
)
touch "$run_root/COMPLETE"
printf 'FISH_SOURCE_ARCHIVE_A76FD826_V3=PASS\n'
