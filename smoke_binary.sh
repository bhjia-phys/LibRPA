#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ref_dir="$script_dir/../references/regression-tests/bn-headwing"

usage() {
  cat <<'EOF'
Usage:
  smoke_binary.sh --binary <path> [--reference-dir <path>]

Behavior:
  Runs LibRPA on pre-packaged BN headwing regression test input,
  compares QP output against reference, and reports pass/fail.
  No ABACUS runtime needed — uses bundled pre-computed data.

Exit: 0 if all checks pass, 1 otherwise.
EOF
}

pass_count=0
fail_count=0

note_pass() { echo "PASS: $*"; pass_count=$((pass_count + 1)); }
note_fail() { echo "FAIL: $*" >&2; fail_count=$((fail_count + 1)); }

binary=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary) binary="$2"; shift 2 ;;
    --reference-dir) ref_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    *) shift ;;
  esac
done

[[ -n "$binary" ]] || { echo "ERROR: --binary is required" >&2; usage >&2; exit 2; }
[[ -x "$binary" ]] || { note_fail "Binary not executable: $binary"; }

# Resolve to absolute paths (tolerate non-existent path: note_fail already recorded)
binary="$(cd "$(dirname "$binary")" 2>/dev/null && pwd)/$(basename "$binary")" || true

# ---- Provenance ----
# Git commit from the binary's source tree (build/.. → source root)
src_dir="$(cd "$(dirname "$binary")/.." 2>/dev/null && pwd)" || true
if git -C "$src_dir" rev-parse --short HEAD >/dev/null 2>&1; then
  commit="$(git -C "$src_dir" rev-parse --short HEAD)"
  note_pass "git commit: $commit"
else
  note_fail "Could not determine git commit from $src_dir"
  commit="unknown"
fi

# Compiler flags embedded in binary
flags="$(strings "$binary" 2>/dev/null | grep -oE '\-O[0-9]|\-qopenmp|\-fopenmp|\-std=c\+\+[0-9]+' | sort -u | tr '\n' ' ')" || true
if [[ -n "$flags" ]]; then
  note_pass "compiler flags: $flags"
else
  note_fail "No compiler flags detected in binary"
fi

# ---- Run regression test ----
[[ -f "$ref_dir/input_librpa.tar.gz" ]] || { note_fail "Missing reference input: $ref_dir/input_librpa.tar.gz"; }
[[ -f "$ref_dir/librpa.in" ]] || { note_fail "Missing reference librpa.in: $ref_dir/librpa.in"; }

tmp="$(mktemp -d)"
cleanup() { rm -rf "$tmp"; }
trap cleanup EXIT

tar -xzf "$ref_dir/input_librpa.tar.gz" -C "$tmp" || { note_fail "Failed to unpack reference input"; }
cp "$ref_dir/librpa.in" "$tmp/"

cd "$tmp"

export LD_LIBRARY_PATH="$src_dir/build/src${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if OMP_NUM_THREADS=1 "$binary" 6 1e-12 > librpa.out 2>&1; then
  note_pass "LibRPA exited cleanly"
else
  note_fail "LibRPA exited with non-zero status"
fi

if grep -q 'libRPA finished successfully' librpa.out; then
  note_pass "Completion marker found"
else
  note_fail "Missing completion marker in output"
fi

# QP energy table comparison (matching the regression test's validation regex)
qp_pattern='^[[:space:]]*1[[:space:]]+2\.00000'
if grep -qE "$qp_pattern" librpa.out; then
  note_pass "QP energy table produced"
else
  note_fail "No QP energy table found"
fi

# Compare against reference if available
if [[ -f "$ref_dir/GW_band_spin_1.dat" ]]; then
  if diff -q <(grep -E "$qp_pattern" librpa.out) \
              <(grep -E "$qp_pattern" "$ref_dir/GW_band_spin_1.dat") > /dev/null 2>&1; then
    note_pass "QP energies match reference exactly"
  else
    note_fail "QP energies deviate from reference"
  fi
fi

echo ""
echo "SUMMARY: pass=$pass_count fail=$fail_count"
[[ "$fail_count" -eq 0 ]] && echo "DONE: binary validation passed" || echo "DONE: binary validation failed"
exit $fail_count
