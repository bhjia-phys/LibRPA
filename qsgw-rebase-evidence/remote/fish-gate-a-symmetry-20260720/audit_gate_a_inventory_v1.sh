#!/usr/bin/env bash
set -euo pipefail

echo "host=$(hostname)"
echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo '--- /tmp/si444_sym_test files'
if test -d /tmp/si444_sym_test; then
  find /tmp/si444_sym_test -maxdepth 4 \
    \( -type f -o -type l \) -printf '%y|%p|%s|%l\n' | sort
else
  echo 'MISSING:/tmp/si444_sym_test'
fi

echo '--- known candidate executable'
candidate=/home/bhj/LibRPA-qsgw/build/chi0_main.exe
if test -x "$candidate"; then
  sha256sum "$candidate"
  ldd "$candidate" | sort
else
  echo "MISSING:$candidate"
fi

echo '--- LibRPA worktrees'
for repo in /home/bhj/LibRPA-qsgw /home/bhj/LibRPA*; do
  test -d "$repo/.git" || continue
  printf 'repo=%s\n' "$repo"
  git -C "$repo" rev-parse HEAD
  git -C "$repo" status --short --branch --ignore-submodules=all
done

echo '--- executable inventory'
find /home/bhj -maxdepth 6 -type f -name chi0_main.exe -perm -u+x \
  -printf '%p\n' 2>/dev/null | sort | while IFS= read -r executable; do
    sha256sum "$executable"
  done

echo '--- fish bare branch tips'
bare=/tmp/librpa-qsgw-cb294020-84e6b0a/repo.git
if test -d "$bare"; then
  git --git-dir="$bare" for-each-ref \
    --format='%(objectname)|%(refname)' refs/heads | sort
else
  echo "MISSING:$bare"
fi
