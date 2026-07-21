# Accepted fish Gate 1 recovery

This compact archive records the accepted recovery postcheck for the completed
upstream-versus-candidate G0W0 run. The full source run and 48 SigmaC blocks
remain on fish; they are bound by `SOURCE_RUN_SHA256SUMS.txt` and are not
duplicated here.

- Accepted remote archive:
  `/home/bhj/ai-runs/librpa-qsgw-gate1-current-postcheck-20260722-c91a4305-v1`
- Completed numerical source run, retained as rejected because its original
  observer thresholds were too strict:
  `/home/bhj/ai-runs/librpa-qsgw-gate1-current-20260722-ca542aca-v1`
- Recovery commit: `c91a4305c527b31f4e3625ef761acd880dbc9e75`
- Recovery runner SHA256:
  `85a2eea661db31f6bb46b4637cee1b06d89e2e7089d2324cca15367247639ac3`
- Remote `GREEN_CONFIRMED`: present
- Remote `FAILED`: absent
- Remote archive integrity check: pass

The accepted numerical results are:

- 48 SigmaC blocks, maximum absolute difference
  `1.8186075345471608e-11 Ha` and relative Frobenius difference
  `2.1449190585723414e-11`, both below `1e-10`.
- 64 k-points and 2816 QP states; k-point coordinates, occupations, and KS
  energies are exact.
- 47 printed QP values differ, with maximum `2.000000165480742e-10 Ha`, below
  `1e-9 Ha`.

The full project contracts remain looser (`1e-8` matrix relative Frobenius and
`1e-6 Ha` eigenvalues); this Gate therefore retains substantial margin.
