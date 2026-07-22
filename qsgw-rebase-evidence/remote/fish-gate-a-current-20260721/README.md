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

The v3 overlap aliases are rejected evidence, but they are not the cause of
the spurious approximately `-2.9e4 eV` iteration-1 frontier. The legacy reader
never opens `sks1k*` directly: it reads `S_spin_*.csc`, then `s1k*`, and
otherwise keeps its initialized identity matrix. A no-alias rerun reproduced
the rejected prefix. Direct comparison of the alias and no-alias prefixes
finds maximum differences of about `1.2e-9 Ha`, while both differ from the
current candidate by `1e2-1e5 Ha` in EXX/Sigma/Vc. The overlap choice therefore
does not explain the failure.

`run_fish_gate_a_failed_prefix_postcheck_v1.sh` is a diagnostic-only recovery
for that rejected run. It compares the common `iter0:1` prefix with the
accepted current Gate2 trace, treats legacy `n_params_anacon=6` as equivalent
to all six frequency points, and records per-component magnitudes. It never
promotes the rejected run to an oracle.

`run_fish_gate_a_legacy_no_overlap_iter1_v1.sh` reruns only the historical
compatibility harness through iteration 1 with no `s1k` aliases. It requires
the eight expected identity-fallback warnings and compares every common trace
component against the SHA-bound accepted current Gate2 trajectory. The run
finished LibRPA successfully but failed the numerical comparator, so it is
diagnostic-only rejected evidence.

`run_fish_gate_a_no_overlap_failed_postcheck_v1.sh` SHA-binds both rejected
prefixes, validates their numerical equivalence with
`validate_gate_a_overlap_diagnostic_v1.py`, and archives compact diagnostics
without copying the 27 MB traces. The next controlled factor is
`use_shrink_abfs=true` versus `false`: the accepted full-BZ legacy/current
miniter10 gate used `false`, while both rejected symmetry prefixes used
`true`. No aggregate miniter2/miniter5 run is allowed until that factor is
isolated.

`run_fish_gate_a_shrink_off_iter1_v1.sh` performs that isolation. It reruns
both the legacy compatibility harness and the Gate0 candidate through
iteration 1 with symmetry enabled and `use_shrink_abfs=false` on both sides.
It retains the historical identity-overlap fallback, requires exactly eight S
and eight HF fallback warnings from the legacy reader, and compares EXX,
SigmaC, Vc, raw/mixed QSGW matrices, eigenvalues, frontier levels, closure,
Hermiticity, fixed-basis invariants, and electron count. This focused gate must
pass before the aggregate miniter2/miniter5 runner is reconsidered.

`run_fish_gate_a_current_v2.sh` is retained as failed evidence. It must not be
used because the legacy reader applies `stoi` to every trailing `stru_out`
token and therefore cannot parse the candidate-only symmetry metadata tail.

`run_fish_gate_a_current_v1.sh` is retained as a pre-execution draft. It must
not be used because it points at the pre-overlay contract and an external
`/tmp` observer bundle.

Required runtime variables bind the v3 runner to a clean checkout:

```bash
RUNNER_COMMIT=<clean-runner-commit> \
RUNNER_SOURCE=<clean-runner-checkout> \
RUNNER_SHA256=<sha256-of-committed-v3-runner> \
RUN_TAG=<unique-tag> \
bash run_fish_gate_a_current_v3.sh
```

A successful aggregate run has `GREEN_CONFIRMED`. Any failure records
`FAILED` and leaves the immutable partial evidence in place.

The focused shrink-off runner uses the same binding variables and a unique
run tag:

```bash
RUNNER_COMMIT=<clean-runner-commit> \
RUNNER_SOURCE=<clean-runner-checkout> \
RUNNER_SHA256=<sha256-of-committed-shrink-off-runner> \
RUN_TAG=<unique-tag> \
bash run_fish_gate_a_shrink_off_iter1_v1.sh
```
