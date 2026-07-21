# Fish Gate A current runner

This directory prepares, but does not claim execution of, the clean-candidate
Si k444 symmetry gate.

`run_fish_gate_a_current_v3.sh` uses two reader-compatibility views over the
same accepted Gate2 physical input overlay for a historical compatibility
harness and the current candidate. It runs:

- no mixing through iteration 2, mapping legacy direct beta 1 to current
  `qsgw_mixer=none` with configured-but-unused beta 0.2;
- linear Hamiltonian mixing with beta 0.2 through iteration 5;
- per-iteration component, eigenvalue, gap, closure, Hermiticity, fixed-basis,
  initial-state, and electron-count observers.

The legacy executable is explicitly labeled a compatibility harness, not raw
historical source. The v4-to-v6 adapter validates the real contracts first and
then normalizes contract headers only to reuse the frozen numerical row
aligner. Numeric rows are never rewritten.

The v3 runner binds the accepted Gate0 candidate executable, accepted Gate2
postcheck, full Gate2 source-run manifest, frozen ABACUS bundle, composite
`stru_out`, derived input contract, physical `vxc_out`, legacy harness, and all
observers by SHA256. The views contain identical file sets and share every
physical input byte except two reader metadata files: the legacy view uses the
numeric-only source `stru_out` and source contract, while the candidate view
uses the append-only symmetry `stru_out` and its derived contract. The input
view validator requires those to be the only SHA256 differences and requires
the contracts to differ only in the bound `stru_out` hash.

Both views also expose the same eight `s1k*_nao.txt` symlink aliases to the
frozen `sks1k*_nao.txt` overlap matrices. This is required because the legacy
reader constructs the `sks` filename but only opens its older `s1k` fallback.
The runner rejects any legacy overlap-file warning instead of accepting the
reader's identity-matrix fallback.

`run_fish_gate_a_current_v2.sh` is retained as failed evidence. It must not be
used because the legacy reader applies `stoi` to every trailing `stru_out`
token and therefore cannot parse the candidate-only symmetry metadata tail.

`run_fish_gate_a_current_v1.sh` is retained as a pre-execution draft. It must
not be used because it points at the pre-overlay contract and an external
`/tmp` observer bundle.

Required runtime variables bind the v2 runner to a clean checkout:

```bash
RUNNER_COMMIT=<clean-runner-commit> \
RUNNER_SOURCE=<clean-runner-checkout> \
RUNNER_SHA256=<sha256-of-committed-v3-runner> \
RUN_TAG=<unique-tag> \
bash run_fish_gate_a_current_v3.sh
```

A successful aggregate run has `GREEN_CONFIRMED`. Any failure records
`FAILED` and leaves the immutable partial evidence in place.
