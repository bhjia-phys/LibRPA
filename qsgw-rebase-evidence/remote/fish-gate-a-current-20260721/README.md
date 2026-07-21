# Fish Gate A current runner

This directory prepares, but does not claim execution of, the clean-candidate
Si k444 symmetry gate.

`run_fish_gate_a_current_v1.sh` uses the same frozen ABACUS bundle for a
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

Required runtime variables bind the run to an accepted fish Gate 0:

```bash
CANDIDATE_COMMIT=<clean-commit> \
CANDIDATE_GATE0_ROOT=<immutable-gate0-run> \
CANDIDATE_GATE0_PROVENANCE_SHA256=<sha256> \
CANDIDATE_EXE_SHA256=<sha256> \
RUNNER_SHA256=<sha256-of-runner-from-candidate-commit> \
RUN_TAG=<unique-tag> \
bash run_fish_gate_a_current_v1.sh
```

A successful aggregate run has `RUN_COMPLETE`. Any failure records `FAILED`
and leaves the immutable partial evidence in place.
