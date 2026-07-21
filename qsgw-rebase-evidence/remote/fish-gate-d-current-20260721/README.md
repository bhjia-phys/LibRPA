# Fish Gate D current runner

`run_fish_gate_d_current_v1.sh` validates the mixer-disabled current-contract
QSGW band path on the immutable pinned-ABACUS Si k444 symmetry band bundle.
`run_fish_gate_d_current_linear_v2.sh` repeats the controlled cut-mode trio with
linear mixing at beta 0.2. Both run two QSGW updates for cut modes 0, 1, and 2
with every other input fixed.

The observer closes:

- raw and mixed effective Hamiltonians on the regular and band channels;
- fixed-basis diagonalization and live-wavefunction rotation from immutable
  `mf0`;
- Fourier diagnostics and H(R) translational Hermiticity;
- ABACUS CSR to AO H(k) to fixed-basis H(k) round trips;
- KS, EXX, and QSGW band tables;
- cut-mode invariants and the mode-1/mode-2 active subspace.

The linear runner additionally requires every iteration to close as
`cut(previous) + beta * (raw - cut(previous))`, followed by reapplication of
the configured cut. Its local observer has a negative fixture for the legacy
uncut-initial/no-recut ordering.

This is a structural and metamorphic gate for the current implementation. It is
not a substitute for a legacy-versus-current run on the same 143-point band
dataset. The output records this limitation as
`legacy_same_dataset_acceptance=false`.

The runner must be invoked from an exact clean candidate commit already accepted
by Gate 0. Supply the exact SHA-256 values from the Gate 0 and band-bundle
artifacts:

```bash
CANDIDATE_COMMIT=... \
CANDIDATE_GATE0_ROOT=... \
CANDIDATE_GATE0_PROVENANCE_SHA256=... \
CANDIDATE_EXE_SHA256=... \
BAND_BUNDLE_ROOT=... \
BAND_BUNDLE_PROVENANCE_SHA256=... \
BAND_BUNDLE_OUTPUT_SHA256=... \
BAND_DATASET_MANIFEST_SHA256=... \
BAND_CONTRACT_SHA256=... \
BAND_VXC_MANIFEST_SHA256=... \
RUNNER_SHA256=... \
RUN_TAG=... \
bash run_fish_gate_d_current_linear_v2.sh
```

`RUNNER_SHA256` is the hash of the exact LF-normalized runner present in the
candidate checkout. Select the v1 runner for the mixer-disabled lane and the v2
runner for the linear lane. No remote execution has been claimed for either
local runner draft.
