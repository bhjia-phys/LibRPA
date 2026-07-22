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
