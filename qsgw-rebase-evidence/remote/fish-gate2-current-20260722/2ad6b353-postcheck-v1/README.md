# Accepted fish Gate 2 postcheck

This compact archive records the accepted postcheck for the completed current
QSGW first-self-energy run. The full 22.8 MB matrix trace and accepted upstream
G0W0 SigmaC blocks remain on fish and are bound by
`SOURCE_RUN_SHA256SUMS.txt`; they are not duplicated here.

- Accepted remote archive:
  `/home/bhj/ai-runs/librpa-qsgw-gate2-current-postcheck-20260722-2ad6b353-v1`
- Completed numerical source run:
  `/home/bhj/ai-runs/librpa-qsgw-gate2-current-20260722-aca53374-v1`
- Postcheck commit: `2ad6b353295d8c9ae04aa0503667ac403147985b`
- Postcheck runner SHA256:
  `4e2ccd5b71587f93b08cc7e79be684ded2b8377f6688036f9eadea66031684a1`
- Remote `GREEN_CONFIRMED`: present
- Remote `FAILED`: absent
- Remote archive and source-run integrity checks: pass
- Focused observer tests: 10/10 pass

The source executable finished successfully. Its original runner retained a
`FAILED` marker only because the first observer required the raw EXX lower
triangle to be Hermitian and compared it directly in Hamiltonian closure.
Legacy Scheme A diagonalizes with `eigsh(UPLO='U')`; the QSGW assembly keeps
the upper triangle authoritative and materializes the lower triangle from it.
The corrected observer reproduces that exact rule and reports raw component
anti-Hermiticity separately. No numerical source code or tolerance changed.

Accepted results:

- QSGW trace iteration 1/channel 0 versus accepted upstream G0W0: 48 blocks,
  maximum absolute difference `2.0267304319323924e-11 Ha`, relative Frobenius
  difference `2.125913085616405e-11`; both are below `1e-10`.
- Upper-authoritative raw Hamiltonian closure:
  `2.2737367544323206e-13 Ha`; assembled Hamiltonian Hermiticity error: zero.
- Diagonalization eigenvalue error: `1.5916157281026244e-12 Ha`; off-diagonal
  error: `7.389644451905042e-13 Ha`; fixed-basis wavefunction rotation error:
  zero on fish.
- Initial eigenvalues, occupations, Fermi level, charge, and no-mixer identity
  all pass their `1e-10` gates.

This gate proves the current symmetry-on, head-wing-off, Hartree-off,
band-off, no-mixing first update. It does not replace the pending multi-round
legacy/current, Hartree, or `qsgw_band` gates.
