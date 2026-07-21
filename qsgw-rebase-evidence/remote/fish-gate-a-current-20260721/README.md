# Fish Gate A current runner

This directory prepares, but does not claim execution of, the clean-candidate
Si k444 symmetry gate.

`run_fish_gate_a_current_v2.sh` uses the same accepted Gate2 input overlay for a
historical compatibility harness and the current candidate. It runs:

- no mixing through iteration 2, mapping legacy direct beta 1 to current
  `qsgw_mixer=none` with configured-but-unused beta 0.2;
- linear Hamiltonian mixing with beta 0.2 through iteration 5;
- per-iteration component, eigenvalue, gap, closure, Hermiticity, fixed-basis,
  initial-state, and electron-count observers.

The legacy executable is explicitly labeled a compatibility harness, not raw
historical source. The v4-to-v6 adapter validates the real contracts first and
then normalizes contract headers only to reuse the frozen numerical row
aligner. Numeric rows are never rewritten.

The v2 runner binds the accepted Gate0 candidate executable, accepted Gate2
postcheck, full Gate2 source-run manifest, frozen ABACUS bundle, composite
`stru_out`, derived input contract, physical `vxc_out`, legacy harness, and all
observers by SHA256. Both executables read the exact same overlay directory.

`run_fish_gate_a_current_v1.sh` is retained as a pre-execution draft. It must
not be used because it points at the pre-overlay contract and an external
`/tmp` observer bundle.

Required runtime variables bind the v2 runner to a clean checkout:

```bash
RUNNER_COMMIT=<clean-runner-commit> \
RUNNER_SOURCE=<clean-runner-checkout> \
RUNNER_SHA256=<sha256-of-committed-v2-runner> \
RUN_TAG=<unique-tag> \
bash run_fish_gate_a_current_v2.sh
```

A successful aggregate run has `GREEN_CONFIRMED`. Any failure records
`FAILED` and leaves the immutable partial evidence in place.
