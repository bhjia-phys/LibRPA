# QSGW Upstream Rebase Plan

Status: `in_progress`

Manifest: `qsgw-rebase-manifest.json`

Current gate: `revised-goal-audit` (entered on 2026-07-20; implementation gates are not yet accepted)

Scope: full-BZ/no-crystal-symmetry QSGW plus numerically accepted ABACUS IBZ/crystal-symmetry QSGW, linear fixed-basis Hamiltonian mixing, Hartree updates, and head-wing-off `qsgw_band`. QSGW iterative head/wing is explicitly unsupported and must fail fast. Existing G0W0 head/wing behavior remains upstream-owned and unchanged.

## Revised-goal freeze (2026-07-20)

| Field | Recorded value |
|---|---|
| Worktree | `F:/AI_Workspace/Theoretical-Physics/.sisyphus/drafts/_scratch/LibRPA-qsgw-independent-upstream-95c4-20260716` |
| Revised branch | `codex/qsgw-symmetry-no-headwing-42d-20260720` |
| Frozen parent | `7e11dd65050666a04361f2d2bd09c7b3aca81c9c` |
| Preserved prior branch | `codex/qsgw-symmetry-validation-k888-20260720` at the same frozen parent |
| Upstream base | `42d3863c1d865194d382a085851d1e2e8a39764f` (`upstream-ssh/master`) |
| No-sym candidate | `c27482016f70ece5a0e5ccad7199d93ac3f6ebf5` |
| Audit | `qsgw-rebase-evidence/impact/revised-goal-audit-20260720.md` |

The older live-state and gate tables below are retained as historical provenance. Where they conflict with this revised-goal freeze, this section and the revised audit are authoritative.

## Ownership boundary

- Upstream owns shared G0W0, GW, EXX, chi0, epsilon, LibRI, symmetry, distributed-matrix and MPI numerical behavior.
- QSGW is an independent adapter that repeatedly calls upstream GW/EXX machinery and updates QSGW-only state.
- No historical shared numerical routine may be restored to reproduce an old QSGW number.
- API/data-layout migration belongs in QSGW-only code whenever possible.
- A shared interface hunk is U3 and requires an evidence-linked exact approval before it can enter the clean candidate.

## Live Git and worktree state

| Field | Recorded value | Evidence |
|---|---|---|
| Host/shell | `JIABOHAN`, Windows, Windows PowerShell 5.1, not WSL | `qsgw-rebase-evidence/environment/local-freeze.json` |
| Repository/worktree | `F:/AI_Workspace/Theoretical-Physics/.sisyphus/drafts/_scratch/LibRPA-qsgw-independent-upstream-1376-20260714` | `git rev-parse --show-toplevel`; `qsgw-rebase-evidence/git/worktrees.txt` |
| Branch | `codex/qsgw-independent-upstream-1376-20260714` | `qsgw-rebase-evidence/git/status.txt` |
| Frozen HEAD | `1376ee4f45a7611a55c5b92c4ba41409d515bcea` | `qsgw-rebase-evidence/git/upstream-new.txt` |
| Tracking ref | local `upstream-ssh/master` at `1376ee4f`; fresh network fetch not yet authorized in the current permission state | `qsgw-rebase-evidence/git/remotes.txt`; `git for-each-ref` capture in session |
| Dirty state | QSGW implementation/tests/registration plus one protected getter; plan/manifest/evidence now also untracked | `qsgw-rebase-evidence/git/status.txt`; `current-tracked.patch`; `current-untracked-source-sha256.txt` |
| Existing changes | all frozen changes are from the ongoing QSGW port; inaccessible `tmp8nrx_fx6` and `tmphsxvjj9c` are excluded and not modified | status warnings and frozen inventory |
| Isolation | dedicated worktree retained; no reset, stash, clean or destructive operation performed | `qsgw-rebase-evidence/git/worktrees.txt` |

## Upstream range

| Field | Full commit | Reference |
|---|---|---|
| Upstream old base | `b484f2a9a252c8a7169c67e9781da6f9c07c310a` | `qsgw-rebase-evidence/git/upstream-old.txt` |
| Upstream new local tracking tip | `1376ee4f45a7611a55c5b92c4ba41409d515bcea` | `qsgw-rebase-evidence/git/upstream-new.txt` |
| Old QSGW source commit | `cb2940201b1f44b1c38b90169048b7d060e44732` | `qsgw-rebase-evidence/git/qsgw-old.txt` |
| Clean candidate commit | pending | U3 is approved; create only after source-lane audit |

The old/new upstream range contains 14 commits and 16 semantic change records, including the U3 interface conflict linked to the upstream head/wing contract. Commit and change-ID coverage are exact and machine-checked.

## Entire `src/` protected inventory

All of `src/` is protected. The frozen inventory contains 251 files:

| Ownership | Count | Rule |
|---|---:|---|
| Shared core | 185 | every non-QSGW `src/` implementation file |
| Shared build interface | 2 | `src/CMakeLists.txt`, `src/test/CMakeLists.txt` |
| QSGW-only source | 44 | explicit `src/qsgw/` root, including four pre-existing old draft files |
| QSGW tests | 20 | `src/test/test_qsgw*` |

Evidence: `qsgw-rebase-evidence/git/protected-src-inventory.json`.

Changed shared/build-interface paths relative to `1376ee4f`:

1. `src/CMakeLists.txt`: adds the QSGW subdirectory; QSGW integration/build hunk, not a numerical change.
2. `src/test/CMakeLists.txt`: registers QSGW tests; test integration hunk, not a numerical change.
3. `src/core/dielecmodel.h`: adds the approved five-line read-only head-tensor getter; this is the sole protected U3 hunk.

Every other shared `src/` file is byte-identical to the upstream base in the frozen worktree.

## Upstream U0-U2 inventory

Full structured inventory: `qsgw-rebase-evidence/git/upstream-hunk-inventory.json`.

| Change ID | Commit | Class | Reachable effect and disposition |
|---|---|---|---|
| `UP-BVK-LOG-001` | `bb0e2762` | U1 | qsgw-band BvK remap; logging only, inherit unchanged |
| `UP-HEAD-GAMMA-VOLUME-001` | `72559e92` | U1 | head normalization uses complete BvK cell count; inherit upstream numerical behavior |
| `UP-SYMMETRY-TEXT-TOL-001` | `a209b9e5` | U1 | protected symmetry parser; no-sym path bypasses it |
| `UP-HEADWING-SYMMETRY-ROUTE-001` | `4c302ffa` | U1 | shared head/wing setup; changed symmetry branch remains disabled in this scope |
| `UP-REGRESSION-DEBUG-001` | `0aaf08f7` | U0 | upstream regression input only |
| `UP-REGRESSION-CASE-001` | `cfc6b188` | U0 | upstream regression case only |
| `UP-REGRESSION-STABILITY-001` | `84a62bfc` | U0 | upstream regression validation only |
| `UP-REGRESSION-PHASE-001` | `35043ae5` | U0 | upstream phase-invariant comparison only |
| `UP-HEADWING-DIAGNOSTICS-001` | `133a6061` | U1 | reachable diagnostics/collectives; inherit and test MPI |
| `UP-REGRESSION-DIAGNOSTICS-001` | `133a6061` | U0 | regression metadata/output only |
| `UP-KBLACS-MF-OWNERSHIP-001` | `318e3e42` | U2 | distributed MF/GF/DM ownership changed; QSGW adapts only collection/projection boundaries |
| `UP-KPARA-HEADWING-CONTRACT-001` | `a033ec4c` | U2 | full-BZ head/wing active-k and velocity ownership changed; bind live MF/velocity in adapter |
| `U3-HEAD-MATRIX-GETTER-001` | `a033ec4c` | U3 | exact five-line read-only interface approved by `APPROVAL-U3-HEAD-MATRIX-GETTER-001` |
| `UP-ATOM-BASIS-MAP-001` | `e238a761` | U1 | GW/EXX/chi0 now use authoritative atom-basis map; inherit unchanged |
| `UP-COMMUNICATOR-OWNERSHIP-001` | `14704c11` | U2 | shared collectives use explicit Dataset communicator; adapter follows `comm_h` |
| `UP-REGRESSION-KPARA-001` | `1376ee4f` | U0 | upstream regression parameter only |

## Formula-to-code map

The complete table is in `qsgw-rebase-evidence/impact/formula-to-code.md`. It covers:

- BvK remap invariance;
- head gamma-cell normalization;
- enforced no-symmetry bypass;
- head/wing diagnostic collectives;
- kBLACS density/Green-function ownership;
- live full-BZ head/wing update;
- atom-basis map authority;
- Dataset communicator ownership;
- the proposed read-only head tensor interface.

## Frozen candidate source lanes

The current dirty source is frozen and must be separated before clean-candidate acceptance:

| Lane | Current paths | Preliminary class | Required decision |
|---|---|---|---|
| Upstream API adapter and task registration | `driver/*`, `driver/tasks/qsgw.cpp`, fixed-basis/distributed/input/trace modules | U2 | prove old formula is preserved and commit as the rebase candidate |
| Linear Hamiltonian mixing | `src/qsgw/mixing*`, `hamiltonian_mixing*` and driver wiring | U4 unless legacy equivalence is proved | separate immutable feature commit and two-sided benchmark |
| QSGW head/wing | current driver and `src/qsgw/headwing_update*` still implement it | excluded by revised goal | remove runtime reachability; any QSGW request must fail during input validation without changing G0W0 |
| Hartree update | `src/qsgw/hartree_*` and driver wiring | U2 only if it reproduces the old equation; otherwise U4 | real legacy Hartree oracle and provenance required |
| qsgw-band operator path | projection/operator Fourier and band wiring | U2 only if it reproduces the old equation; otherwise U4 | prohibit state-basis k-to-k Fourier; run dedicated band gate |
| Regression comparator | `regression_tests/backend/comparisons/cmp_qsgw.py` | test infrastructure | keep separate from numerical implementation commit |
| Shared head tensor getter | `src/core/dielecmodel.h` | U3 | retain only the exact approved five-line hunk and run all linked observers |

No lane is accepted merely because earlier dirty-source runs were numerically close.

The completed formula-level candidate classification is recorded in `qsgw-rebase-evidence/impact/qsgw-source-lane-audit.md`. It separates legacy-equivalent U2 adapter behavior from U4 occupation, mixing, independent-grid head-wing, Hartree and combined band behavior; each U4 lane remains numerically unaccepted until its ordered gate passes.

## Numerical attribution boundary

| Layer | Current evidence status | Conclusion boundary |
|---|---|---|
| Upstream G0W0 behavior | historical G0W0 regressions and no-head iter0 evidence exist, but clean upstream/candidate A/B has not been rerun | supporting evidence only |
| QSGW adapter | historical miniter10 no-head old/new comparison exists | supporting evidence only until reproduced from immutable candidate and legacy source |
| Independent formula features | head/live mixing/Hartree/band code exists | not accepted; each suspected U4 lane requires an immutable feature commit and dedicated observer |

## Protected shared-hunk audit

| Hunk ID | Path/symbol | Class | Formula row | Candidate diff | Approval | Observer | Status |
|---|---|---|---|---|---|---|---|
| `SHARED-HEAD-GETTER-001` | `src/core/dielecmodel.h::diele_func::get_head_matrices` | U3 | `F-HEAD-TENSOR-READ` | `qsgw-rebase-evidence/impact/U3-head-matrix-readonly-getter.patch` | `APPROVAL-U3-HEAD-MATRIX-GETTER-001` | build, G0W0 A/B, head-wing regression, QSGW head trace | approved; observers pending execution |

There are no changes to `driver/tasks/g0w0.cpp`, `driver/tasks/g0w0_band.cpp`, `src/core/gw.cpp/.h`, `exx.cpp/.h`, `chi0.cpp/.h`, `epsilon.cpp/.h`, or `dielecmodel.cpp` relative to upstream `1376ee4f`.

## U3 impact packet

| U3 ID | Formula | Call chain and exact diff | Affected scope | Required observers | Decision |
|---|---|---|---|---|---|
| `U3-HEAD-MATRIX-GETTER-001` | `F-HEAD-TENSOR-READ` | `qsgw-rebase-evidence/impact/U3-head-matrix-readonly-getter.md` | one inline const getter in `src/core/dielecmodel.h`; no shared numerical implementation | build/CTest, upstream-candidate G0W0 A/B, upstream head-wing regression, QSGW tensor trace | approved by `APPROVAL-U3-HEAD-MATRIX-GETTER-001` |

## Baseline provenance

| Item | State | Required evidence |
|---|---|---|
| Clean source commit | pending | candidate commit after source-lane audit; U3 decision is complete |
| Candidate executable | pending | remote build path, SHA256 and source commit |
| Pure upstream executable | pending | separate build from `1376ee4f` or fresher fetched tip, path and SHA256 |
| Old QSGW executable | pending clean rebuild | `cb294020` source snapshot, dirty diff if any, compiler/dependencies and executable SHA256 |
| Si k444 dataset and all inputs | prior artifacts exist but not yet accepted | complete input manifest and SHA256 for mf0/Vxc/eigenpairs/WFC/velocity/Coulomb/Cs/Hartree/k-map/units/basis/gauge |
| Environment | local freeze recorded; remote pending | compiler, libraries, MPI/OMP, host and deterministic-reduction environment |
| Comparator | local source/tests exist | immutable comparator commit/SHA and declared tolerances |

## Fixed gate sequence

| Order | Gate | Status | Exit evidence |
|---:|---|---|---|
| 0 | configure/build, complete CTest, unchanged upstream regressions | not entered | clean upstream and candidate builds, logs and results |
| 1 | upstream G0W0 vs candidate G0W0 | planned | byte-identical inputs and direct tensor/result A/B |
| 2 | QSGW iteration 0/1 vs upstream G0W0 | planned | Sigma/EXX/Vc/H/U/eigenvalue/WFC/invariant comparison |
| 3 | solid no-mixing old/new | planned | immutable legacy miniter5/miniter10 per-iteration replay |
| 4 | QSGW head/wing fail-fast | not entered | parser/runtime rejection plus unchanged upstream G0W0 head-wing regression |
| 5 | linear mixing beta=0.2 | planned | disabled-vs-linear controlled pair and old/new trajectory |
| 6 | Hartree no-symmetry | planned | delta-VH iter0, live-density charge, units, Hermiticity and old/new evidence |
| 7 | ABACUS symmetry-on two-round | not entered | 29-to-512 mapping, old/new/full-BZ component comparison, no head-wing |
| 8 | qsgw-band no-symmetry and symmetry | not entered | grid AO/real-space operator followed by BvK/Fourier projection |
| 9 | FHI-aims no-symmetry | planned | formal two-round regression and complete input provenance |
| 10 | ABACUS no-symmetry | planned | formal two-round regression and complete input provenance |
| 11 | ABACUS symmetry-on regression | planned | committed small two-round Hartree-on case with mapping provenance |
| 12 | MPI/OMP/determinism | planned | MPI 1/4, OMP 1/32 and deterministic comparisons |

Heavy builds and runs must use SSH on fish/dongfang; dongfang jobs must use `sbatch`. Earlier dirty-source results remain supporting evidence only.

## Verified evidence

| Evidence ID | Bounded claim | Artifact | Verification |
|---|---|---|---|
| `E-FREEZE-GIT-001` | live Windows worktree/branch/remotes/status were captured without destructive operations | `qsgw-rebase-evidence/git/*` | direct Git commands on 2026-07-15 |
| `E-UPSTREAM-COVERAGE-001` | 14 range commits and 16 semantic IDs are exactly covered | `upstream-hunk-inventory.json` | PowerShell JSON/rev-list comparison: both exact `True` |
| `E-PROTECTED-SRC-001` | all 251 current `src/` files are inventoried; only one shared numerical header differs | `protected-src-inventory.json` | SHA256/status generation and ownership query |
| `E-COMPARATOR-UNIT-001` | frozen Python QSGW comparator tests pass locally | source state before freeze | 18/18 focused and 41/41 comparison backend tests |
| `E-PROTECTED-IMPL-001` | listed GW/G0W0/EXX/chi0/epsilon implementations equal upstream `1376ee4f` | live Git diff | `PROTECTED_IMPLEMENTATIONS_IDENTICAL` |

## Open issues

| Issue ID | Class/gate | Exact issue | Required evidence or decision |
|---|---|---|---|
| `ISSUE-LATEST-UPSTREAM` | environment | local tracking ref is `1376ee4f`, but a fresh upstream fetch has not succeeded under current authorization | explicit network authorization, fetch, then update range and rerun audit |
| `ISSUE-CANDIDATE-SPLIT` | U2/U4 | frozen dirty source combines adapter and potentially independent formula features | formula-level split and clean immutable commits |
| `ISSUE-REMOTE-GATE0` | Gate 0 | no clean candidate executable or complete CTest from the frozen source | after candidate commit, SSH build on fish and sbatch numerical work on dongfang |
| `ISSUE-LEGACY-HARTREE` | Hartree | legacy pinned LibRI lacks required Hartree header; historical build used a dirty LibRI tree | rebuild from archived exact dirty LibRI with complete provenance |
| `ISSUE-QSGW-HEADWING-REACHABLE` | revised scope | QSGW parser and runtime still execute same-grid and independent-grid iterative head/wing | add parser and runtime fail-fast tests; remove dead QSGW-only implementation reachability |
| `ISSUE-OBSOLETE-U3-GETTER` | protected diff | the approved five-line `get_head_matrices` interface is now used only by excluded QSGW head/wing code | remove the consumer and getter; final protected shared numerical diff should be zero |
| `ISSUE-BAND-NOT-FOURIER-WIRED` | qsgw-band | `operator_fourier` is called only by the QSGW head-wing adapter; the current band loop separately evaluates band EXX/Sigma | wire the converged grid AO/real-space effective operator to the fixed band reference and validate full-BZ plus symmetry restoration |
| `ISSUE-SYMMETRY-ORACLE` | symmetry | only k-star count preflight and one-iteration supporting evidence exist; no legacy two-round oracle or full-BZ per-component comparison exists | regenerate a merge-before symmetry oracle and run the three-way two-round comparison |
| `ISSUE-REGRESSION-DANGLING` | CI | both QSGW `testsuite.xml` entries point to absent testcase directories | replace placeholders with committed two-round ABACUS no-sym, ABACUS symmetry, and FHI-aims cases |
| `ISSUE-STALE-MANIFEST` | provenance | the manifest records an older worktree, base, scope, HEAD, and gate | refresh with the revised freeze before accepting any new numerical gate |

## Exactly one next action

- [ ] Commit the revised audit and make all QSGW head/wing requests fail fast, with parser/runtime unit tests proving that G0W0 input behavior is untouched.
