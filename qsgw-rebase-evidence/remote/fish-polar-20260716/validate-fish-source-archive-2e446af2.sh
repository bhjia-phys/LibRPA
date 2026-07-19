#!/usr/bin/env bash

set -euo pipefail

staging=/home/bhj/ai-runs/librpa-qsgw-source-archive-upload-2e446af2
source_archive=$staging/librpa-2e446af2-linux-git-archive.tar.gz
run_root=/home/bhj/ai-runs/librpa-qsgw-source-archive-validation-2e446af2-20260716
archive=$run_root/librpa-2e446af2-linux-git-archive.tar.gz
red_evidence=/home/bhj/ai-runs/librpa-qsgw-polar-tdd-red-20260716-785ac390/evidence-oneapi-v2
green_evidence=/home/bhj/ai-runs/librpa-qsgw-polar-tdd-green-20260716-262a6424/evidence-oneapi-v3
expected_commit=2e446af200a0e5a5fd1ee05c8db99ebb7dde9418
expected_parent=a76fd826eed3929185ea47cfcb992075a72f2b77
expected_archive_sha=433d971ab428d221ce122d295b784f5bd7530c4dbd9643ad417a36948beb82ce
expected_archive_entries=1838
expected_red_manifest_sha=92dae83f0bda1ea4cfec215a919fde544b02b7f6a095044404fbc131d1dc8130
expected_red_provenance_sha=58f0b77a3b9ebdeab0c5e595e75b13a3aeccb1cfad8740a03db702e85c018be1
expected_green_manifest_sha=5d857626f24ae69da5ba836fa4b6fc03c257b2b181b39979bb12653ff5a1b651
expected_green_provenance_sha=2598fe61c3ceaf927e69f4290e3f6221b63bc078fa1834eb0b8b65470c80d707

test ! -e "$run_root"
test -f "$source_archive"
test "$(sha256sum "$source_archive" | awk '{print $1}')" = \
  "$expected_archive_sha"
test -e "$red_evidence/TDD_RED_CONFIRMED"
test -e "$green_evidence/COMPLETE"
test "$(sha256sum "$red_evidence/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_red_manifest_sha"
test "$(sha256sum "$red_evidence/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_red_provenance_sha"
test "$(sha256sum "$green_evidence/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_green_manifest_sha"
test "$(sha256sum "$green_evidence/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_green_provenance_sha"
(
  cd "$red_evidence"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
(
  cd "$green_evidence"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

mkdir -p "$run_root"
cp "$source_archive" "$archive"
cp "$0" "$run_root/validate-fish-source-archive-2e446af2.sh"
test "$(sha256sum "$archive" | awk '{print $1}')" = \
  "$expected_archive_sha"

set +o pipefail
archive_commit=$(gzip -dc "$archive" | git get-tar-commit-id)
set -o pipefail
archive_entries=$(tar -tzf "$archive" | wc -l)
test "$archive_commit" = "$expected_commit"
test "$archive_entries" -eq "$expected_archive_entries"
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
0fcfa4901490a952a2af40f541229e912b5852a6eb7260c320b181de33d1ab7e  src/qsgw/fixed_basis.cpp
1ddf9250ae2b0fec21b9bb08bfd6d41c421822aa7a7e396ffc2e8a0d2819755f  src/qsgw/fixed_basis.h
65157f83fcc809d27bc58ff566dab4573432df4887b5885bfe7fee37880ba348  src/test/test_qsgw_fixed_basis.cpp
105392016d0fd5dc1998a06217d558a55ffeb5ab81d54cf3cf51a22caa84ffeb  src/test/test_qsgw_fixed_basis_mpi.cpp
016a889a46aeeb4046b09d20a67416cee4231be99e8e7ef6cc69e6fe4d342dc0  driver/tasks/qsgw.cpp
dfda28723b8a8790530a0a2a1bafc43c62de01fc2c327be8399d50c8739b37d4  src/qsgw/occupation.cpp
bc4ae1af3883f8f976cd0d1526d703895b5581de6ab2e05b98f2af9efb9a9326  src/core/gw.cpp
beec940d42b851f4b0221124b6d8f0ce923cd1ec2a047cf3c1bb98239751c56b  src/core/gw.h
536a251dfb60ebe5c20f964b2c646514d12c82907c998f9ad4e8a4d212c4483b  src/core/exx.cpp
d02693ab6126613e48537467951c2f4cc3bef9b62c5eea1002044e402c4e4ee8  src/core/exx.h
ab864c7c3da809a7e466ad874943698e91626cc10b8c8e630528207130def768  src/core/dielecmodel.h
EOF
cmp "$run_root/expected-critical-blobs.sha256" \
  "$run_root/critical-blobs.sha256"

cat >"$run_root/PROVENANCE.txt" <<EOF
artifact=validated-linux-git-source-archive
source_commit=$expected_commit
source_parent_commit=$expected_parent
archive=$archive
archive_sha256=$expected_archive_sha
archive_embedded_commit=$archive_commit
archive_entry_count=$archive_entries
archive_selection=CMakeLists.txt,cmake,include,driver,src,thirdparty
archive_line_endings=git_blob_lf_bytes
archive_producer=local_windows_git_archive
archive_transport=local_to_fish_scp
fish_red_evidence=$red_evidence
fish_red_manifest_sha256=$expected_red_manifest_sha
fish_red_result=expected_failure_confirmed_oneapi
fish_green_evidence=$green_evidence
fish_green_manifest_sha256=$expected_green_manifest_sha
fish_green_result=serial_mpi4_and_59_of_59_pass_oneapi
validator_host=$(hostname)
validator_git=$(git --version)
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$run_root"
  sha256sum \
    PROVENANCE.txt \
    critical-blobs.sha256 \
    expected-critical-blobs.sha256 \
    validate-fish-source-archive-2e446af2.sh \
    librpa-2e446af2-linux-git-archive.tar.gz \
    >SHA256SUMS.txt
  sha256sum --check --quiet SHA256SUMS.txt
)
touch "$run_root/COMPLETE"
printf 'FISH_SOURCE_ARCHIVE_2E446AF2=PASS\n'
