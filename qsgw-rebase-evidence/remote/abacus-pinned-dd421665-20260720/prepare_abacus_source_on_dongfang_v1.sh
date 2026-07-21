#!/usr/bin/env bash
set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-build-20260720-v1
artifacts="$root/source_artifacts"
source_dir="$root/source"
main_archive="$artifacts/abacus-master_ghj-dd421665-source.tar.gz"
libpaw_archive="$artifacts/libpaw_interface-c211c0ab-source.tar.gz"

expected_main=f39d426e17bb4bd1225630cb454e65886dac9ebf2660d20605f114f4a0d2121f
expected_libpaw=dec5fbe7349c4b843efeb8c16ac761549d641900aa6cc81833814f4c15ea01af

test ! -e "$root/SOURCE_PREPARED" || {
    echo "Refusing to mutate an already prepared source tree: $root" >&2
    exit 2
}

actual_main=$(sha256sum "$main_archive" | awk '{print $1}')
actual_libpaw=$(sha256sum "$libpaw_archive" | awk '{print $1}')
test "$actual_main" = "$expected_main"
test "$actual_libpaw" = "$expected_libpaw"

mkdir -p "$source_dir"
tar -xzf "$main_archive" -C "$source_dir" --strip-components=1
mkdir -p "$source_dir/deps/libpaw_interface"
tar -xzf "$libpaw_archive" -C "$source_dir/deps/libpaw_interface" --strip-components=1

(
    cd "$source_dir"
    sha256sum -c "$artifacts/TRACKED_SOURCE_SHA256SUMS.txt"
) > "$root/main-manifest-check.stdout" 2> "$root/main-manifest-check.stderr"

(
    cd "$source_dir/deps/libpaw_interface"
    sha256sum -c "$artifacts/LIBPAW_TRACKED_SOURCE_SHA256SUMS.txt"
) > "$root/libpaw-manifest-check.stdout" 2> "$root/libpaw-manifest-check.stderr"

cat > "$root/SOURCE_PROVENANCE.txt" <<EOF
analysis=abacus_pinned_source_prepare
host=$(hostname -f 2>/dev/null || hostname)
requested_repo_url=https://github.com/AroundPeking/abacus-develop.git
source_branch=master_ghj
source_commit=dd4216653386d32f79e3219f3ea5dd2d229c1c5a
source_tree=7284d42382b3555081badfdb4a15691d0bd1ec3d
submodule_path=deps/libpaw_interface
submodule_commit=c211c0ab330adf3cc374f50ab3edee46b174e64c
source_archive_sha256=$actual_main
submodule_archive_sha256=$actual_libpaw
tracked_manifest_sha256=$(sha256sum "$artifacts/TRACKED_SOURCE_SHA256SUMS.txt" | awk '{print $1}')
submodule_manifest_sha256=$(sha256sum "$artifacts/LIBPAW_TRACKED_SOURCE_SHA256SUMS.txt" | awk '{print $1}')
prepared_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

find "$source_dir" -type f -printf '%P\t%s\n' | LC_ALL=C sort > "$root/PREPARED_FILE_INVENTORY.txt"
sha256sum "$root/PREPARED_FILE_INVENTORY.txt" > "$root/PREPARED_FILE_INVENTORY.sha256"
touch "$root/SOURCE_PREPARED"

sha256sum \
    "$root/SOURCE_PROVENANCE.txt" \
    "$root/PREPARED_FILE_INVENTORY.txt" \
    "$root/main-manifest-check.stdout" \
    "$root/main-manifest-check.stderr" \
    "$root/libpaw-manifest-check.stdout" \
    "$root/libpaw-manifest-check.stderr" \
    > "$root/PREPARE_OUTPUT_SHA256SUMS.txt"

cat "$root/SOURCE_PROVENANCE.txt"
wc -l "$root/main-manifest-check.stdout" "$root/libpaw-manifest-check.stdout"
cat "$root/PREPARED_FILE_INVENTORY.sha256"
