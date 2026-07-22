# exact847 full-ABF chi0 failure (not a parity oracle)

## Classification

This run is preserved as a failed input-contract diagnostic. It must not be
used as the legacy side of the QSGW parity gate.

- Remote run:
  `/home/bhj/ai-runs/librpa-qsgw-gate-a1-exact847-fullcoul-nohead-miniter2-20260722-6a85c7fc-v1`
- Legacy source: exact847 plus the Scheme-A occupation-only compatibility
  patch.
- Symmetry: ABACUS IBZ 8 k-points restored to the 64-point full BZ.
- Hartree: off.
- Iterative head/wing: off.
- Full-Coulomb EXX: on.
- `use_shrink_abfs`: true.
- `use_shrink_chi`: not specified, so exact847 used its historical default
  `true`.

The target candidate contract explicitly uses `use_shrink_chi=false`. The
failed run therefore exercised full-ABF chi0 followed by the shrink transform,
not the direct shrinked-Cs chi0 route required by the comparison.

## Observed failure

Iteration 0:

```text
HOMO = 6.28506 eV, LUMO = 6.97617 eV, Efermi = 6.48709 eV
```

Iteration 1:

```text
HOMO = 4.81178 eV, LUMO = 4.81216 eV, Efermi = 4.81197 eV
```

The iteration-1 gap was about `0.00038 eV`. Iteration 2 then aborted in the
ABACUS symmetry chi0 reduction path:

```text
terminate called after throwing an instance of 'std::out_of_range'
  what():  map::at
```

The run ended with exit code 255 and did not produce an iteration-2
checkpoint.

## Bound artifacts

```text
FAILED                                      9f1826056923ac1a29c3b6d5df0904c7e2f200c1f88c5051c2308a227a9eaf33
PROVENANCE.txt                              7fe87a9d85eeeb4badd1bbbd3f419054a714d7aff61a7b02a849ad6d31ac9602
legacy/librpa.in                            4fc3699bb2b49d9e59ff2cf1ed438bfd558cd35a7b7b639fabcb4cf39056c796
legacy/librpa.stdout                        5f9540cdd0903b2a5d8ed0a45f7f47ada777160350ff73808c8f79f38aeb4751
legacy/librpa.stderr                        4208e3a96cd66ef8121f1b33363a02734a75fcbcd55a7e3ed5eb1567f0cda147
legacy/homo_lumo_vs_iterations.dat          8a2e96435bc6c7604d69e500f87563c199df09d8546c7230ad14dd2083fd8d89
checkpoint/latest_iteration.txt             4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865
checkpoint/iter_00001/checkpoint.meta        ff3126cc8b6ca8a7a6ced9ec8189aa6ded94cb3b4cdd26bdd3aa7720f1773dd6
```

The eight iteration-1 Hamiltonian checkpoint matrices are individually bound
in `run_fish_gate_a_exact847_restart_mapat_gdb_v1.sh`.

## Diagnostic follow-up

The checkpoint-restart GDB run was started at:

`/home/bhj/ai-runs/librpa-qsgw-gate-a1-exact847-restart-mapat-gdb-20260722-5b7d669d-v1`

It completed the first chi0 time point under GDB instead of reproducing the
immediate abort. The run was intentionally terminated once the input-contract
mismatch was established; it is not an accepted diagnostic or numerical gate.

## Corrected gate

The corrected exact847 runner explicitly sets:

```text
use_shrink_abfs = t
use_shrink_chi = f
```

Its new immutable run ID is:

`librpa-qsgw-gate-a1-exact847-fullcoul-nohead-shrinkchi-off-miniter2-20260722-v3`

Only a completed, manifest-verified v3 run may unlock the candidate parity
runner.
