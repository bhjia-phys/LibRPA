# exact847 Sigma restart/projection probe

## Result

The accepted current executable read the 16 real-space `SigcRF` files from the
usable exact847 shrink-chi-off iteration-1 prefix and projected them through
the current k-local BLACS fixed-basis path.

Immutable artifact:

`/home/bhj/ai-runs/librpa-qsgw-gate-a1-exact847-legacy-sigcrf-projection-probe-20260722-2962fcd1-v1`

The candidate and exact847 projected `SigmaC(iw)` agree to:

- maximum absolute difference: `2.72568972891501e-10 Ha`;
- relative Frobenius difference: `1.3369659378509992e-10`;
- frequency-grid maximum difference: `0 Ha`.

The probe passed its `1e-6 Ha` absolute and `1e-8` relative diagnostic
thresholds. The current SigcRF reader, immutable `mf0` exposure, and k-local
BLACS projection therefore reproduce the exact847 fixed-basis Sigma when
given the same real-space operator.

## Boundary

This is diagnostic evidence, not Gate A1 acceptance. The ordinary current
one-update run still differs strongly from exact847 because its newly built
real-space Sigma differs before projection. No shared GW, EXX, or G0W0 source
was changed by this probe.

The next controlled comparison is the exact847 versus current full-compute
`SigcRF` block set under the same k444 symmetry-on, shrink-chi-off, no-head,
no-Hartree contract.
