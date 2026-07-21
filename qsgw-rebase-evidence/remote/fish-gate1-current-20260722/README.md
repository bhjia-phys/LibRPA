# Fish Gate 1 current G0W0 A/B

This gate compares frozen upstream LibRPA `42d3863c` with the clean QSGW
candidate build from accepted fish Gate 0. Both executables receive the same
Si k444 symmetry-reduced ABACUS payload and byte-identical `librpa.in` files.

The numerical oracle is the complete set of 48 correlation self-energy
matrices: one spin, eight IBZ k-points, and six imaginary frequencies. The
gate also requires byte-identical `energy_qp` output. Acceptance thresholds
for the SigmaC comparison are `1e-12 Ha` maximum absolute difference and
`1e-12` relative Frobenius difference.

The frozen QSGW bundle stores full AO-basis Vxc matrices, while upstream
`task = g0w0` requires the traditional diagonal KS-basis `vxc_out`. The
versioned `vxc_out` in this directory is the matching file from the same
historical ABACUS producer root. Its source path and SHA256 are recorded in
`VXC_SOURCE_PROVENANCE.txt`. The runner builds a run-local overlay consisting
of 55 symlinks to the frozen bundle plus physical `vxc_out` and `stru_out`
files; it does not modify or copy the 1.7 GiB frozen bundle.

Upstream `42d3863c` reconstructs symmetry from the symmetry-operation tail in
`stru_out` and ignores the historical `symrot_R.txt`, `symrot_k.txt`, and
`irreducible_sector.txt` sidecars. The historical frozen `stru_out` predates
that tail. `build_stru_symmetry_overlay_v1.py` appends the 48 row-convention
fractional operations emitted by pinned ABACUS commit `dd421665` while keeping
the historical lattice, atoms, and legacy k-point payload byte-for-byte. The
builder rejects an existing symmetry block and validates the legacy 4x4x4
payload, operation determinants, identity count, source lattice metric, and
two-atom mapping before writing the run-local file.

The complete pinned-producer `stru_out` is deliberately not used. Its direct
lattice omits `LATTICE_CONSTANT = 10.2` while its Cartesian atom coordinates
include that scale, so it violates LibRPA's documented Bohr-unit structure
contract. Only its dimensionless fractional rotations/translations are reused,
and those are independently validated against the historical Si structure.
The exact source, hashes, rejected fields, and reuse scope are recorded in
`STRU_SYMMETRY_SOURCE_PROVENANCE.txt`.

The runner fails closed on Gate 0 provenance, executable hashes, frozen
bundle manifests, runner/tool/input hashes, exact input identity, expected
matrix count and dimensions, and comparison thresholds. A successful run has
`GREEN_CONFIRMED`; any failed run has `FAILED`.

Rejected preflight attempts from runner commit `14059056` used run tags
`20260722-14059056-v1` and `20260722-14059056-debug1`. Both stopped before
creating a run directory or launching LibRPA because the comparator hashes had
been measured from a CRLF Windows working tree instead of the canonical LF Git
blobs. No numerical output from those attempts is accepted. The corrected
runner pins the hashes measured from the clean fish checkout.

Run `20260722-83b10134-v1` passed every provenance and input preflight but is
also rejected. Upstream stopped before producing SigmaC with
`Generated symmetry k-star count does not match Coulomb k-points`; candidate
execution never started. This identified the missing new-format `stru_out`
symmetry tail rather than a numerical difference. That runner also failed to
write `FAILED` because its `ERR` trap was not inherited through the run
function/subshell. The current runner uses `set -E` plus an `EXIT` trap that
writes `FAILED` after run-root creation unless the green path is completed.

Required variables:

```bash
RUNNER_COMMIT=<commit-containing-this-runner> \
RUNNER_SOURCE=<clean-checkout-at-runner-commit> \
RUNNER_SHA256=<sha256-of-runner> \
RUN_TAG=<unique-tag> \
bash run_fish_gate1_current_v1.sh
```
