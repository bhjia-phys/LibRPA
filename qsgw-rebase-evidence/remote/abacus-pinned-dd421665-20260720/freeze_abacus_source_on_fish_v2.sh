#!/usr/bin/env bash

set -euo pipefail

requested_repo_url=https://github.com/AroundPeking/abacus-develop.git
accepted_transport_url=git@github.com:AroundPeking/abacus-develop.git
source_branch=master_ghj
pinned_commit=dd4216653386d32f79e3219f3ea5dd2d229c1c5a
source_root=/home/bhj/ai-runs/abacus-pinned-dd421665-source-20260720-v1/source
failed_v1=/home/bhj/ai-runs/abacus-pinned-dd421665-source-20260720-v1
run_root=/home/bhj/ai-runs/abacus-pinned-dd421665-source-20260720-v2

test ! -e "$run_root"
test -d "$failed_v1"
test ! -e "$failed_v1/SOURCE_FROZEN"
test -d "$source_root/.git"
mkdir -p "$run_root"
cp "$0" "$run_root/freeze_abacus_source_on_fish_v2.sh"

git ls-remote "$requested_repo_url" "refs/heads/$source_branch" \
  >"$run_root/ls-remote.txt"
test "$(awk '{print $1}' "$run_root/ls-remote.txt")" = "$pinned_commit"
test "$(git -C "$source_root" rev-parse HEAD)" = "$pinned_commit"
test -z "$(git -C "$source_root" status --porcelain --untracked-files=all)"

actual_origin=$(git -C "$source_root" remote get-url origin)
if test "$actual_origin" != "$requested_repo_url" && \
   test "$actual_origin" != "$accepted_transport_url"
then
  printf 'unexpected origin URL: %s\n' "$actual_origin" >&2
  exit 1
fi

git -C "$source_root" log -1 --format=fuller \
  >"$run_root/source-commit.txt"
git -C "$source_root" rev-parse "HEAD^{tree}" \
  >"$run_root/source-tree.txt"
git -C "$source_root" remote -v >"$run_root/git-remotes.txt"
git -C "$source_root" status --short --branch --untracked-files=all \
  >"$run_root/git-status.txt"
git -C "$source_root" submodule status --recursive \
  >"$run_root/submodule-status.txt" 2>"$run_root/submodule-status.stderr" || true
git -C "$source_root" fsck --full \
  >"$run_root/git-fsck.stdout" 2>"$run_root/git-fsck.stderr"

(
  cd "$source_root"
  git ls-files -z | sort -z | xargs -0 sha256sum \
    >"$run_root/TRACKED_SOURCE_SHA256SUMS.txt"
)

git -C "$source_root" bundle create \
  "$run_root/abacus-master_ghj-dd421665.bundle" \
  "refs/heads/$source_branch"
git -C "$source_root" bundle verify \
  "$run_root/abacus-master_ghj-dd421665.bundle" \
  >"$run_root/bundle-verify.stdout" \
  2>"$run_root/bundle-verify.stderr"
git -C "$source_root" archive --format=tar.gz \
  --prefix=abacus-master_ghj-dd421665/ \
  --output="$run_root/abacus-master_ghj-dd421665-source.tar.gz" \
  "$pinned_commit"

du -sh "$source_root" >"$run_root/source-size.txt"
du -h "$run_root/abacus-master_ghj-dd421665.bundle" \
  "$run_root/abacus-master_ghj-dd421665-source.tar.gz" \
  >"$run_root/artifact-sizes.txt"
sha256sum \
  "$run_root/abacus-master_ghj-dd421665.bundle" \
  "$run_root/abacus-master_ghj-dd421665-source.tar.gz" \
  "$run_root/TRACKED_SOURCE_SHA256SUMS.txt" \
  >"$run_root/FROZEN_SOURCE_SHA256SUMS.txt"

source_tree=$(cat "$run_root/source-tree.txt")
bundle_sha=$(awk 'NR == 1 {print $1}' "$run_root/FROZEN_SOURCE_SHA256SUMS.txt")
archive_sha=$(awk 'NR == 2 {print $1}' "$run_root/FROZEN_SOURCE_SHA256SUMS.txt")
manifest_sha=$(awk 'NR == 3 {print $1}' "$run_root/FROZEN_SOURCE_SHA256SUMS.txt")
printf '%s\n' \
  'analysis=abacus_pinned_source_freeze' \
  'acceptance_gate=true' \
  'host=Fisherd-Server100.96.1.64' \
  "requested_repo_url=$requested_repo_url" \
  "actual_origin_url=$actual_origin" \
  'transport_rewrite=https_to_ssh' \
  "source_branch=$source_branch" \
  "source_commit=$pinned_commit" \
  "source_tree=$source_tree" \
  'source_dirty_count=0' \
  "git_bundle_sha256=$bundle_sha" \
  "source_archive_sha256=$archive_sha" \
  "tracked_manifest_sha256=$manifest_sha" \
  "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >"$run_root/PROVENANCE.txt"

(
  cd "$run_root"
  find . -maxdepth 1 -type f \
    ! -name OUTPUT_SHA256SUMS.txt ! -name SOURCE_FROZEN -print0 |
    sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/SOURCE_FROZEN"
cat "$run_root/PROVENANCE.txt"
