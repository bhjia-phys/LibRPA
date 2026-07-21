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
of symlinks to every frozen bundle file plus this physical `vxc_out`; it does
not modify or copy the 1.7 GiB frozen bundle.

The runner fails closed on Gate 0 provenance, executable hashes, frozen
bundle manifests, runner/tool/input hashes, exact input identity, expected
matrix count and dimensions, and comparison thresholds. A successful run has
`GREEN_CONFIRMED`; any failed run has `FAILED`.

Required variables:

```bash
RUNNER_COMMIT=<commit-containing-this-runner> \
RUNNER_SOURCE=<clean-checkout-at-runner-commit> \
RUNNER_SHA256=<sha256-of-runner> \
RUN_TAG=<unique-tag> \
bash run_fish_gate1_current_v1.sh
```
