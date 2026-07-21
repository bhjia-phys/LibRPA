#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
    printf 'usage: %s PRODUCER_ROOT POSTCHECK_ROOT OBSERVER_SHA256\n' "$0" >&2
    exit 2
fi

producer=$1
root=$2
expected_observer_sha=$3
validator="$root/validate_abacus_si_k444_symmetry_output_v1.py"
report="$root/OUTPUT_VALIDATION.json"

on_exit()
{
    code=$?
    if [[ $code -ne 0 && -d $root && ! -e $root/FAILED ]]; then
        {
            printf 'exit_code=%s\n' "$code"
            printf 'failed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        } > "$root/FAILED"
    fi
}
trap on_exit EXIT

test -d "$root"
test -e "$producer/FAILED"
test ! -e "$producer/COMPLETE"
test ! -e "$root/FAILED"
test ! -e "$root/COMPLETE"
test -f "$validator"

actual_observer_sha=$(sha256sum "$validator" | awk '{print $1}')
test "$actual_observer_sha" = "$expected_observer_sha"
sha256sum "$validator" > "$root/OBSERVER_SHA256SUMS.txt"
sha256sum "$producer/FAILED" > "$root/SOURCE_FAILED_SHA256SUMS.txt"

cat > "$root/POSTCHECK_PROVENANCE.txt" <<EOF
gate=pinned_abacus_si_k444_symmetry_output_postcheck_v1
source_producer=$producer
source_job_id=2381071
source_numeric_stage=completed
source_terminal_marker=FAILED_observer_only
observer_sha256=$actual_observer_sha
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

python3 -B "$validator" "$producer/run" --mpi-ranks 4 --report "$report" \
    > "$root/output-validation.stdout" \
    2> "$root/output-validation.stderr"

python3 -B - "$report" > "$root/POSTCHECK_REPORT_ASSERTIONS.txt" <<'PY'
import json
import math
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)

assert report["status"] == "PASS"
assert report["reader_version"] == 0
assert report["bz_sampling"]["grid"] == [4, 4, 4]
assert report["bz_sampling"]["n_scf"] == 8
assert report["bz_sampling"]["n_ibz"] == 8
assert report["structure"]["n_atoms"] == 2
assert report["structure"]["n_symops"] == 48
assert report["structure"]["identity_present"] is True
assert report["band"]["n_kpoints"] == 8
assert report["band"]["n_spins"] == 1
assert report["band"]["n_bands"] == 44
assert report["band"]["n_basis"] == 44
assert math.isclose(report["band"]["gap_ev"], 0.6523128697371297, rel_tol=0.0, abs_tol=1.0e-12)
assert report["file_counts"] == {
    "Cs_data": 4,
    "KS_eigenvector": 8,
    "coulomb_cut": 4,
    "coulomb_mat": 4,
    "vxc_ks_matrix": 8,
}
assert report["vxc_matrix_schema"] == {
    "columns": 44,
    "format": "abacus_modern_comment_row_upper_triangle",
    "rows": 44,
    "upper_triangle_entries_per_matrix": 990,
    "validated_matrix_count": 8,
}
print("PASS: postcheck report assertions")
PY

sha256sum \
    "$report" \
    "$root/output-validation.stdout" \
    "$root/output-validation.stderr" \
    "$root/POSTCHECK_REPORT_ASSERTIONS.txt" \
    > "$root/POSTCHECK_ARTIFACT_SHA256SUMS.txt"

printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$root/COMPLETE"
