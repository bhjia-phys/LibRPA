# Revised QSGW Goal Audit - 2026-07-20

## Freeze

- Branch: `codex/qsgw-symmetry-no-headwing-42d-20260720`
- Frozen parent: `7e11dd65050666a04361f2d2bd09c7b3aca81c9c`
- Preserved prior branch: `codex/qsgw-symmetry-validation-k888-20260720`
- Upstream base: `42d3863c1d865194d382a085851d1e2e8a39764f`
- No-sym candidate: `c27482016f70ece5a0e5ccad7199d93ac3f6ebf5`
- Protected shared source diff: `src/core/dielecmodel.h` only; the other `src/` differences outside QSGW are CMake/test registration.

## Confirmed Implementation State

| Requirement | Current evidence | Audit result |
|---|---|---|
| QSGW parameters in `librpa.in` | `driver/driver.*`, `driver/inputfile.cpp`, `driver/test/test_qsgw_inputfile.cpp` | Linear mixing, beta, iteration bounds, matrix traces, Hartree switches, and contract path are parsed. Default mixing is linear with beta 0.2. |
| Immutable fixed basis plus live mean field | `src/qsgw/fixed_basis.*`, `src/qsgw/occupation.*`, `driver/tasks/qsgw.cpp` | Implemented in code and covered by unit tests; end-to-end symmetry and Hartree acceptance remains open. |
| QSGW head/wing unsupported | `driver/inputfile.cpp:119-129`, `driver/tasks/qsgw.cpp` | **Fail.** The parser currently accepts analytic QSGW head/wing and the runtime executes both same-grid and independent-full-grid paths. |
| Existing G0W0 head/wing unchanged | protected diff versus upstream | Shared G0W0/GW/EXX/epsilon/chi0 implementations are unchanged. This must remain true while adding QSGW-only fail-fast behavior. |
| Approved head getter | `src/core/dielecmodel.h::get_head_matrices`, consumed only by `driver/tasks/qsgw.cpp` | Obsolete under the revised scope. Remove the getter with the QSGW head/wing path so the final shared numerical diff is zero. |
| ABACUS symmetry preflight | `driver/tasks/qsgw.cpp:412-429` | Partial. It requires all EXX/GW/RPA symmetry switches, an available context, complete k-star count, and full member count. It does not by itself prove rotations, phases, canonical mapping, or numerical equivalence. |
| Symmetry numerical acceptance | k444 supporting evidence and k888 inventory | **Open.** No merge-before symmetry-on two-round oracle and no three-way per-component comparison are frozen. |
| Hartree live density | `driver/tasks/qsgw.cpp` calls `build_hartree_delta_periodic_operator(*hartree_static, dataset->mf, reference, ...)` after each live update | The call uses live `dataset->mf` against immutable `reference`. Unit coverage exists, but no accepted Hartree-on two-round no-sym/symmetry gate exists. |
| `qsgw_band` AO/BvK/Fourier route | `src/qsgw/operator_fourier.*`, `driver/tasks/qsgw.cpp` | **Fail.** `operator_fourier` is only called through `src/qsgw/headwing_update.cpp`; the current band loop separately builds band EXX/Sigma and never invokes the grid-operator Fourier path. |
| Formal regression cases | `regression_tests/testsuite.xml` | **Fail.** Two QSGW entries exist, but both referenced testcase directories are absent. No committed numerical QSGW regression currently runs. |
| Full CTest | `qsgw-rebase-evidence/remote/dongfang-symmetry-k888-20260720/fish-full-ctest-v2` | Supporting evidence: 61/61 passed on fish at commit `76f5a144`; rerun is required after revised-scope code changes. |
| Si k888 input provenance | `dongfang-inventory-v2-failed` | Useful inventory despite failed strict assertions: 29 SCF IBZ, 512 full-grid/PyATB, 143 band points, 44 bands/AOs, complete ABACUS `KPT.info` mapping. Iterative head-wing assets are not to be used by revised-goal QSGW runs. |

## Stale Planning Records

`QSGW_REBASE_PLAN.md` and `qsgw-rebase-manifest.json` were created for an older no-sym-only goal. Before this audit they still recorded the old worktree, branch, upstream base, candidate HEAD, head-wing gates, and a `crystal_symmetry_qsgw=out_of_scope_not_tested` state. These fields cannot support a revised-scope acceptance claim.

The refreshed schema-1.0 manifest is intentionally invalid at this audit point. Its validator reports 61 unresolved items: 19 missing hashes, 37 missing required provenance strings, four missing references, and one stale upstream-inventory range. It reports no schema, gate-sequence, observer-schema, comparative commit-role, approval, or self-consistency errors. Raw output is in `qsgw-rebase-evidence/validation/manifest-validator-revised-goal.txt`.

## Required Corrections

1. Reject `replace_w_head=true` and `use_pyatb=true` for `task=qsgw` and `task=qsgw_band` during input validation; retain a runtime guard as defense in depth.
2. Remove QSGW head-wing execution reachability and related claims while leaving upstream G0W0 behavior byte-identical.
   Remove the now-unused approved five-line head getter as part of this scope correction.
3. Replace the current `qsgw_band` update with grid effective-operator AO restoration followed by BvK/Fourier projection into immutable `mf0_band`.
4. Build a head-wing-disabled ABACUS IBZ contract that binds the actual 29-point SCF inputs and the complete 29-to-512 mapping provenance.
5. Generate a merge-before symmetry-on two-round oracle, then compare new symmetry-on, new full-BZ/no-symmetry, and legacy symmetry-on per iteration and per component.
6. Add no-symmetry and symmetry-on Hartree gates proving live-density use, charge conservation, units, q=0 convention, and Hermiticity.
7. Replace dangling `testsuite.xml` placeholders with real committed two-round cases and references.

## Claim Boundary

No ABACUS crystal-symmetry QSGW, Hartree-on QSGW, or formal QSGW regression is accepted by this audit. The frozen 61/61 CTest and earlier numerical packets are supporting evidence only. No QSGW iterative head-wing calculation submitted after the revised goal may be used as acceptance evidence.
