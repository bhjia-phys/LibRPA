#!/usr/bin/env bash
set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-build-20260720-v2
deps="$root/dependencies"
source_libri=/data/home/df_iopcas_bhj/software-stack/src/LibRPA/thirdparty/LibRI
source_libcomm=/data/home/df_iopcas_bhj/software-stack/src/LibRPA/thirdparty/LibComm
source_cereal=/data/home/df_iopcas_bhj/software-stack/src/LibRPA/thirdparty/cereal-1.3.0
source_elpa=/data/home/df_iopcas_bhj/software-stack/install

test -e "$root/SOURCE_PREPARED"
test ! -e "$root/DEPENDENCIES_FROZEN" || {
    echo "Refusing to mutate already frozen dependencies: $root" >&2
    exit 2
}

mkdir -p "$deps"
cp -a "$source_libri" "$deps/LibRI"
cp -a "$source_libcomm" "$deps/LibComm"
cp -a "$source_cereal" "$deps/cereal-1.3.0"
mkdir -p "$deps/elpa/include" "$deps/elpa/lib/pkgconfig"
cp -a "$source_elpa/include/elpa_openmp-2025.01.001" "$deps/elpa/include/"
cp -a "$source_elpa/lib/libelpa_openmp.so.19.4.0" "$deps/elpa/lib/"
ln -s libelpa_openmp.so.19.4.0 "$deps/elpa/lib/libelpa_openmp.so"
cp -a "$source_elpa/lib/pkgconfig/elpa_openmp.pc" "$deps/elpa/lib/pkgconfig/"

write_file_manifest() {
    local tree=$1
    local output=$2
    (
        cd "$tree"
        find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
    ) > "$output"
}

write_symlink_manifest() {
    local tree=$1
    local output=$2
    (
        cd "$tree"
        find . -type l -print0 | LC_ALL=C sort -z |
            while IFS= read -r -d '' path; do
                target=$(readlink "$path")
                digest=$(printf '%s' "$target" | sha256sum | awk '{print $1}')
                printf '%s  %s -> %s\n' "$digest" "$path" "$target"
            done
    ) > "$output"
}

for name in LibRI LibComm cereal-1.3.0; do
    if test "$name" = cereal-1.3.0; then
        source_tree=$source_cereal
    elif test "$name" = LibRI; then
        source_tree=$source_libri
    else
        source_tree=$source_libcomm
    fi
    write_file_manifest "$source_tree" "$root/source-${name}-files.sha256"
    write_file_manifest "$deps/$name" "$root/frozen-${name}-files.sha256"
    write_symlink_manifest "$source_tree" "$root/source-${name}-symlinks.sha256"
    write_symlink_manifest "$deps/$name" "$root/frozen-${name}-symlinks.sha256"
    cmp "$root/source-${name}-files.sha256" "$root/frozen-${name}-files.sha256"
    cmp "$root/source-${name}-symlinks.sha256" "$root/frozen-${name}-symlinks.sha256"
done

write_file_manifest "$deps/elpa" "$root/frozen-elpa-files.sha256"
write_symlink_manifest "$deps/elpa" "$root/frozen-elpa-symlinks.sha256"

grep -E '^#define __LIBRI_VERSION_(MAJOR|MINOR|PATCH)' \
    "$deps/LibRI/include/RI/version.h" > "$root/libri-version.txt"
grep -E '^#define CEREAL_VERSION_(MAJOR|MINOR|PATCH)' \
    "$deps/cereal-1.3.0/include/cereal/version.hpp" > "$root/cereal-version.txt"
grep -E '^(Name|Version|Libs|Cflags):' \
    "$deps/elpa/lib/pkgconfig/elpa_openmp.pc" > "$root/elpa-version.txt"

tar -czf "$root/abacus-build-dependencies.tar.gz" -C "$root" dependencies
archive_sha=$(sha256sum "$root/abacus-build-dependencies.tar.gz" | awk '{print $1}')

cat > "$root/DEPENDENCY_PROVENANCE.txt" <<EOF
analysis=abacus_pinned_build_dependency_freeze
host=$(hostname -f 2>/dev/null || hostname)
libri_source=$source_libri
libri_version=2.1.1
libcomm_source=$source_libcomm
cereal_source=$source_cereal
cereal_version=1.3.0
elpa_source=$source_elpa
elpa_version=2025.01.001
elpa_library_sha256=$(sha256sum "$deps/elpa/lib/libelpa_openmp.so.19.4.0" | awk '{print $1}')
dependency_archive_sha256=$archive_sha
frozen_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

touch "$root/DEPENDENCIES_FROZEN"
find "$root" -maxdepth 1 -type f \
    ! -name DEPENDENCY_OUTPUT_SHA256SUMS.txt \
    ! -name DEPENDENCIES_FROZEN \
    -newer "$root/SOURCE_PREPARED" -print0 |
    LC_ALL=C sort -z | xargs -0 sha256sum > "$root/DEPENDENCY_OUTPUT_SHA256SUMS.txt"

cat "$root/DEPENDENCY_PROVENANCE.txt"
wc -l "$root"/frozen-*-files.sha256 "$root"/frozen-*-symlinks.sha256
