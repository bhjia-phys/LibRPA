# Fish Gate 2 current QSGW first self-energy, upstream 67b9888d

This gate reruns the accepted first-self-energy observer for product source
`4f9ab0cf` after the upstream `67b9888d` rebase. It binds the accepted current
Gate 0 executable and the accepted Gate 1 recovery plus its rejected source
run. The source G0W0 matrices are verified byte-for-byte before use.

Trace iteration 0 is the immutable producer/KS state. Trace iteration 1,
channel 0 is the first QSGW self-energy evaluated from that state and is
compared against all 48 accepted upstream G0W0 SigmaC blocks. The run also
checks Hamiltonian closure, the none-mixer identity, fixed-basis rotation,
diagonalization, occupations, electron count, symmetry contract, and input
contract hash.

This is a current-candidate versus current-upstream gate. It never invokes a
legacy QSGW executable and therefore cannot import the invalid legacy
symmetry-on/exact847 oracle. Multi-round legacy/current parity remains the
subsequent formal A1 gate.

Acceptance requires `GREEN_CONFIRMED`, no `FAILED`, and a verified
`OUTPUT_SHA256SUMS.txt` in an immutable run directory.
