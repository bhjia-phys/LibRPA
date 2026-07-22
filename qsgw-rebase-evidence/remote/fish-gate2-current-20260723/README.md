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

Run `20260723-dd7a75f2-v1` is accepted. All 48 SigmaC blocks pass with a
maximum absolute difference of `2.7994974373643978e-11 Ha` and relative
Frobenius difference of `2.1289455429165035e-11`. Hamiltonian closure is
`2.2737367544323206e-13 Ha`; the none-mixer and fixed-basis rotation residuals
are exactly zero.

The archive `dd7a75f2-v1` keeps every regular run file in the checksum-pinned
tarball. All files except the 22 MiB matrix trace are also extracted for
review; the matrix trace remains in the tarball and its SHA-256
`64cf0e2712697f7cabc4350aa9b28ac6cda9d1a51a5a20189dc3db2ed708b382`
matches the embedded output manifest.
