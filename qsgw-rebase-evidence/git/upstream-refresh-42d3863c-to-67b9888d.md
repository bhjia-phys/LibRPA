# Upstream refresh: 42d3863c to 67b9888d

Recorded: 2026-07-23

## Provenance

- Old upstream: `42d3863c1d865194d382a085851d1e2e8a39764f`
- New upstream: `67b9888dac0d09870361398165d0b3c1acc931ff`
- Pre-refresh candidate: `314b5c19005ff11ff53a0341c9a0b94640147471`
- Preserved branch: `codex/qsgw-symmetry-no-headwing-42d-20260720`
- Rebased product source: `4f9ab0cfc90f54910158ab01a877581b080f136e`
- Rebased branch: `codex/qsgw-symmetry-no-headwing-67b-20260723`
- Rebase result: 106 linear commits replayed without a text conflict.

The absence of a text conflict is not an acceptance argument. Classification
below comes from the commit diffs, changed APIs, call sites, and the QSGW task
boundary. The candidate diff against the new upstream is empty in `src/core`,
`src/api`, `driver/tasks/g0w0.cpp`, and `driver/tasks/g0w0_band.cpp`.

## Semantic classification

| Change ID | Commit | Class | Shared change and disposition |
|---|---|---|---|
| `UP-ELPA-DEVICE-ALLOC-67B-001` | `5e390487` | U1 | ELPA device allocation/free calls adopt the current DDLA API. Accept unchanged; build and inherited diagonalization observers apply. |
| `UP-DDLA-BUNDLE-67B-001` | `e99cdf01` | U1 | Bundled LibDDLA APIs, solvers, transport, device memory, and CMake integration change together. Accept unchanged; QSGW reaches these only through upstream shared solvers. |
| `UP-HEADWING-BODY-SOLVE-67B-001` | `85968a20` | U1 | The head/wing body inverse is formed by solving `B X = I`, with LU/Cholesky and CPU/device routes. Accept unchanged. |
| `UP-HEAD-RANK1-67B-001` | `054df5b7` | U1 | Gamma head correction becomes a rank-one update in the original Coulomb representation rather than a full basis rotation. Accept unchanged. |
| `UP-DDLA-REVISION-67B-001` | `0b9bdedb` | U0 | Top-level bundled LibDDLA revision metadata only. Accept unchanged. |
| `UP-HEADWING-BODY-CLEANUP-67B-001` | `bcf3e573` | U1 | Removes redundant shared body-inverse allocation/setup. Accept unchanged. |
| `UP-DIELECTRIC-SOLVE-ERROR-67B-001` | `67b9888d` | U1 | Reports a nonzero dielectric solve `info` as a runtime failure. Accept unchanged. |

There are no U2 QSGW adapter migrations in this range: none of the changed
internal head/wing or DDLA APIs is called from `src/qsgw` or
`driver/tasks/qsgw.cpp`. There are no U3 shared-core conflicts and no U4 QSGW
formula changes.

## Formula-to-code impact

### F-DDLA-DEVICE-CONTRACT-67B

- Invariant: a device allocation/free and distributed solve uses the stream
  and ownership encoded by its DDLA handle without changing matrix values.
- Upstream symbols: ELPA connector device allocation/free calls and bundled
  LibDDLA allocation, transport, factorization, and solve APIs.
- QSGW consumer: indirect, through upstream GW/distributed solver calls.
- Changes: `UP-ELPA-DEVICE-ALLOC-67B-001`, `UP-DDLA-BUNDLE-67B-001`.
- Before/after contract: DDLA API and implementation revision changes; matrix
  basis, units, and QSGW state contract do not.
- Required observers: fish configure/build/CTest and the same-input upstream
  versus candidate G0W0 gate.

### F-HEADWING-BODY-INVERSE-67B

- Formula: find `X` from `B X = I`; `X` is the body inverse used by the Schur
  head/wing correction.
- Upstream symbols: `invert_headwing_body_with_identity_solve`,
  `diele_func::get_body_inv`, `diele_func::rewrite_eps`.
- QSGW consumer: no active iterative consumer because QSGW head/wing requests
  fail during input validation; upstream G0W0 remains the owner.
- Changes: `UP-HEADWING-BODY-SOLVE-67B-001`,
  `UP-HEADWING-BODY-CLEANUP-67B-001`,
  `UP-DIELECTRIC-SOLVE-ERROR-67B-001`.
- Before/after contract: explicit matrix inversion is replaced by an identity
  solve; matrix basis, dimensions, and units are unchanged, while failure
  reporting becomes explicit.
- Required observers: upstream `test_rpa_headwing`, unchanged G0W0 head/wing
  regression, and the QSGW head/wing fail-fast tests.

### F-GAMMA-HEAD-RANK1-67B

- Formula: `epsilon <- epsilon + (H - x1^H epsilon x1) x1 x1^H`.
- Upstream symbols: the Gamma head correction in `src/core/epsilon.cpp`.
- QSGW consumer: indirect only through upstream G0W0; active QSGW head/wing is
  rejected.
- Change: `UP-HEAD-RANK1-67B-001`.
- Before/after contract: the corrected head is applied without rotating the
  full dielectric matrix to and from the Coulomb eigenbasis; the physical
  corrected head and matrix units are unchanged.
- Required observers: upstream head/wing unit tests and G0W0 numerical
  regression. Historical pre-67b head/wing numbers are not an oracle for this
  intentional upstream change.

## Gate consequence

All previously accepted Gate 0 executables were built on `42d3863c` and are
historical after this refresh. The required order is:

1. fish Gate 0: upstream `67b9888d` versus candidate `4f9ab0cf`;
2. same-input upstream/candidate G0W0 regression;
3. candidate iteration zero versus new-upstream G0W0;
4. formal A1-A3 using legacy symmetry-off/full-BZ as the only old-QSGW oracle;
5. post-refresh head/wing structural and G0W0 checks.
