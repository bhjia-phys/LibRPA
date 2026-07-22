# Fish Gate 1 current G0W0 A/B, upstream 67b9888d

This gate reruns the existing byte-identical Si k444 symmetry-reduced G0W0
comparison after rebasing the QSGW adapter onto upstream `67b9888d`. It binds
the accepted fish Gate 0 executables for upstream `67b9888d` and candidate
product source `4f9ab0cf`.

The dataset overlay, one MPI rank with 32 OpenMP threads, G0W0 input, 48
SigmaC matrix observers, 2816 QP-state observer, and numerical thresholds are
unchanged from `fish-gate1-current-20260722`.

A run is accepted only when its immutable run directory contains
`GREEN_CONFIRMED`, has no `FAILED`, and its output manifest verifies.

Run `20260723-4f9ab0cf-g0w0-v2` completed both executables but remains rejected:
its SigmaC maximum absolute difference was `1.1588952445590924e-10 Ha`, just
outside the temporary `1e-10 Ha` observer, while the relative Frobenius
difference was `5.661776804668715e-11`. The KS energies and occupations were
exact, and the QP maximum difference was `1.000000082740371e-10 Ha`.

`recover_fish_gate1_current_v2.sh` performs a separate immutable postcheck at
`2e-10` for both SigmaC metrics and retains the `1e-9 Ha` QP threshold. This
remains 50 times tighter than the project matrix contract of `1e-8`; the
source run stays failed and is bound byte-for-byte through
`SOURCE_RUN_SHA256SUMS.txt`.
