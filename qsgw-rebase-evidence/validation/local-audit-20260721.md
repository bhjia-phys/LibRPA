# Local QSGW Audit, 2026-07-21

## Source State

- Worktree: `F:/AI_Workspace/Theoretical-Physics/.sisyphus/drafts/_scratch/LibRPA-qsgw-independent-upstream-95c4-20260716`
- Branch: `codex/qsgw-symmetry-no-headwing-42d-20260720`
- Committed HEAD: `b7273e13c77d5ea781f192cea3c4201710b6f9fa`
- Frozen upstream: `42d3863c1d865194d382a085851d1e2e8a39764f`
- State under test: uncommitted audited working tree; not yet an immutable
  candidate and not eligible for numerical acceptance.

## Current-Source Native Tests

All listed executables were rebuilt from the current source with MSVC 2022 and
run successfully on Windows. Scratch MPI/BLAS/MeanField compatibility code was
used only to link the isolated tests and is outside the repository worktree.

| Test | SHA256 | Result |
|---|---|---|
| `test_qsgw_abacus_csr_current.exe` | `f9033e797606e102111b2c3d5c950aaae327202c53e4cce9f1cdee3f432e6afa` | PASS |
| `test_qsgw_band_output_current.exe` | `fdb7b0b7fcbe80fc80038308bee684b32e6f76f81e5d56cf9f4f09a73e7ada42` | PASS |
| `test_qsgw_hamiltonian_cut.exe` | `901a5d160b31070c8f0cca539b6b39535499a1a2c7e4a9b5375b294cddc3104e` | PASS |
| `test_qsgw_hamiltonian_mixing.exe` | `b2b864ba1c5b391e80cd1ac163d945a5ac1ea772d29fd15d376fe85ccd3e85b6` | PASS |
| `test_qsgw_hartree_density_current.exe` | `e52b252c517929dd883070146d7f58beb8e8035b2a6d10cb9437d25041946bd6` | PASS |
| `test_qsgw_hartree_kernel_current.exe` | `1b62b13210a35c17a957d16e30ccfb9a34317207e3f420b729989273f5a5bb26` | PASS |
| `test_qsgw_hartree_dump_local.exe` | `b6548043e52c7c6b10dd15865bb9bdf265fd88a5769e6bbc98c986c7bfd868fc` | PASS |
| `test_qsgw_hartree_workflow_current.exe` | `a89c865895729cc31cf207aa96a0eaaf069ca9f1591ebfd57854cfdd324429d0` | PASS |
| `test_qsgw_inputfile_current.exe` | `6b7af50c148de435ed06737bde8e314c437d3ec7864f1e951669be83ad907cfa` | PASS |
| `test_qsgw_operator_fourier_current.exe` | `c9bd48cf994185405146c9d3d312b71c8e97cbdeffe46b3658e3aac62e6670e0` | PASS |
| `test_qsgw_sha256_current.exe` | `9291bee16a1a38a66cce35c6b038260f0a24f2b1e8a79bfa30b5e1e8e9f0ab89` | PASS |
| `test_qsgw_vxc_io_current.exe` | `0c1e1e7076b5419fe6c7bfe66d5d4da002b689f44191aa66ac3976d5845090a6` | PASS |

The operator Fourier executable links the real symmetry, PBC, atomic-basis,
Wigner-rotation, and Fourier sources. Its scratch stubs provide only external
math symbols and minimal MeanField construction. It covers complete-grid
roundtrip, nontrivial gauge, time reversal, IBZ/full-BZ parity, and real-space
AO output.

## Parser, Python, and Documentation

- `test_qsgw_inputfile_current.exe`: PASS, including strict QSGW values,
  Fortran `D` exponents, duplicate-last-value handling, symmetry switches,
  head/wing fail-fast, replicated-WFC requirement, band cut controls, and
  unchanged G0W0 behavior.
- Focused comparator and driver-wiring pytest: `29 passed`.
- Full regression backend excluding `test_driver_workspace.py`: `54 passed`.
- `generate_runtime_parameters.py --check --check-defaults`: PASS.
- Pytest emitted only the known Windows ACL warning for the repository-local
  cache; isolated test temporary directories and test results were unaffected.

## Gate C Hartree Observer Preparation

- The env-gated C++ Hartree dump now records the exact full-k order, BvK
  translation order, and atom-pair BvK remap. Its native test was syntax
  checked, linked, and executed successfully with a nontrivial
  `R=1 -> R=-2` remap.
- `validate_qsgw_hartree_dump_v2.py` no longer assumes that the trace k-point
  count equals the reconstructed full-BZ count. It validates exact-order
  Fourier reconstruction, remap application, density charge/Hermiticity, and
  fixed-basis projection onto every active trace channel.
- `validate_qsgw_hartree_trace_v6.py` validates grid-only charge conservation,
  iteration-1 zero response, iteration-2 nonzero response, Hermiticity, and
  `raw_h = h0 - Vxc + EXX + Vc + delta_vh`.
- `compare_qsgw_grid_channels_v1.py` requires the SCF-grid trajectories from
  otherwise controlled `qsgw` and `qsgw_band` runs to agree while ignoring the
  band-only channel.
- `validate_qsgw_hartree_contraction_v1.py` independently rebuilds the
  Gamma-point Hartree contraction from frozen `Cs_data`, Coulomb, density, and
  full-k order. It does not consume a C++ Hartree intermediate and checks both
  explicit normalization conventions.
- The current-source observers plus the versioned fish Gate C runner pass
  28/28 local synthetic/static tests.
- The historical v40 Hartree-on run is rejected as an oracle: its Hartree
  delta was exactly zero but its full Coulomb read replaced the normal
  distributed EXX/GW `Vq_cut`, changing non-Hartree channels. The corrected
  legacy patch keeps a dedicated full-Coulomb Hartree map and restores all
  reader state. Its patch/build/parity observers pass 24/24 local tests.
- These tests are structural evidence only; no corrected legacy build, pinned
  full-BZ bundle, fish numerical run, or dongfang numerical result is claimed.
- The current Gate A comparator passes 6/6 local tests. Gate D band/cut
  observers pass 8/8 tests plus 2 subtests. Pinned ABACUS producer, contract,
  validator, and bundle tools pass 78/78 tests.
- Gate D observer coverage is now 10/10 tests plus 2 comparator subtests. The
  current-only `run_fish_gate_d_current_linear_v2.sh` lane fixes linear beta at
  0.2 while varying only cut mode 0/1/2. Its synthetic observer proves the
  current mix-then-reapply-cut closure and rejects the legacy uncut-initial/
  no-recut ordering. The runner has not been executed remotely and cannot
  replace the required same-dataset legacy/current numerical comparison.
- The added B2 path starts from the pinned `INPUT_scf_fullbz` rather than a
  symmetry-derived dataset. Its observer requires an exact k444 64/64 grid,
  uniform `1/64` weights, periodic k-point uniqueness, 64 wavefunctions, and
  64 native state-basis Vxc matrices. The Slurm producer and immutable
  full/truncated-Hartree bundle assembler add 19/19 passing tests, including
  exact hashing of the Slurm script text executed by each job. The observer
  also follows the pinned writer contract that `symmetry=-1` emits no
  symmetry-operation tail in `stru_out`. No B2 job has been submitted yet.
- One initial Gate C and producer integration invocation failed only while
  creating Windows `TemporaryDirectory` children under a restricted ACL.
  Re-running through each test's fixed-root override passed; no numerical or
  observer assertion failed in that environmental invocation.

## Compile and Build Wiring

- MSVC `/Zs` syntax checks passed for every modified/new C++ implementation and
  test translation unit. The latest `driver/tasks/qsgw.cpp` Sigma-retention
  change also passes with `LIBRPA_USE_MPI` and with `LIBRPA_USE_LIBRI` both
  enabled and disabled; only the pre-existing code-page warning in
  `base_blacs.h` is emitted.
- This is a syntax-only Windows check. Linux compilation, linking, and the
  complete 63-test inventory remain required parts of the next fish Gate 0.
- `hartree_workflow.cpp` and `test_qsgw_hartree_workflow.cpp` passed syntax
  checks with and without `LIBRPA_USE_LIBRI`.
- All 22 `src/qsgw/*.cpp` files are present in `src/qsgw/CMakeLists.txt`.
- All 23 `src/test/test_qsgw_*.cpp` files are registered in
  `src/test/CMakeLists.txt`.
- The previous immutable fish build had 60 tests; the three newly registered
  tests make the next expected CTest inventory 63.
- The Gate 0 focused QSGW regex now includes `test_qsgw_hartree_dump`, making
  the next focused inventory 10 tests rather than 9.

## Diff and Ownership Checks

- `git diff --check`: PASS.
- Protected diff against `42d3863c` is empty for shared G0W0/GW/EXX and API
  paths.
- `src/core/dielecmodel.h` is upstream-identical; the obsolete approved getter
  is not present in the current diff.
- No new remote build or numerical result is claimed by this local audit.

## Defects Found During Audit

1. `sha256_file` placed a 1 MiB buffer on the stack and crashed on the default
   Windows stack. It now uses a heap-backed vector; hash semantics are unchanged.
2. The no-LibRI Hartree branch attempted to inspect a private compatibility
   tensor shape. It now uses the public shape accessor; the LibRI branch remains
   unchanged.
3. The shared input parser silently treated QSGW `nan` and other malformed
   assigned values as absent. QSGW-only strict parsing now rejects malformed
   strings, numbers, integers, and booleans while leaving G0W0 parsing unchanged.
4. QSGW permanently enabled the G0W0 full-Sigma output/retention flag solely to
   obtain in-memory fixed-basis matrices. A scoped guard now restores the
   original flag after projection, including exceptional exits. QSGW does not
   invoke the G0W0 binary matrix writer, and protected G0W0/GW/EXX sources are
   unchanged.

## Upstream Inventory and Manifest

- The frozen `b484f2a9..42d3863c` range contains 30 commits, and the manifest
  now contains exactly 30 commit-map rows ending at `42d3863c`.
- The semantic inventory contains 31 records: 9 U0, 11 U1, 11 U2, and no
  current U3/U4 record. Sixteen records cover the later
  `1376ee4f..42d3863c` refresh.
- `42d3863c` is U1, not U0: it changes only tests, but those tests live under
  the strictly protected `src/` tree.
- A live GitHub compare at `2026-07-21T18:37:20Z` returned `status=identical`,
  `ahead_by=0`, `behind_by=0`, and `total_commits=0` for
  `42d3863c...minyez/LibRPA:master`. No newer upstream commit currently requires
  classification or rebase. Evidence is frozen in
  `qsgw-rebase-evidence/git/upstream-master-compare-20260722.json`; repeat the
  comparison immediately before opening the PR.
- The obsolete approved head getter, its U3 formula row, shared hunk, and live
  approval link were retired from the current manifest because the candidate
  no longer contains that hunk. The approval document remains historical
  evidence.
- Versioned commit-list, name-status, and semantic-hunk artifacts are hashed
  in the manifest. Re-running the refresh script with candidate source
  `5cf996de098c349c741b0b6c74dd0951d1ea3171` is idempotent; manifest SHA-256 remains
  `56d7330aa172e2b7385da702f8262ae3b1a60dfbbb6493d4d64702dedf2fea4d`.
- Freeze-parent and candidate-source provenance are now separate. The generator
  accepts an explicit candidate source, verifies that it exists and descends
  from the freeze parent, and only then may close the clean-candidate issue.
- The positive current-HEAD path is byte-idempotent. A nonexistent SHA and the
  existing but pre-freeze upstream commit both fail before manifest writing.
- The current manifest validator result is intentionally invalid with 59
  unresolved provenance fields: 18 missing SHA-256 values, 37 missing required
  strings, and 4 missing references. It reports no upstream inventory,
  classification, formula-map, protected-hunk, or schema consistency error.
  Raw output is in `manifest-validator-20260721.txt`; its exit code is recorded
  separately as `1`.

## Acceptance Boundary

These checks establish local source consistency and unit/formula behavior only.
They do not replace fish Gate 0, legacy/candidate iterative comparisons, Hartree
end-to-end validation, or qsgw_band/cut numerical validation.

In particular, current physical-default Hartree validation
(`weighted_occupations`, full Coulomb, symmetry on) and corrected legacy parity
(`legacy_extra_inverse_nk`, truncated Coulomb, full BZ) are separate lanes.
Neither lane may be inferred from the other.

## Layered Commit Status

- The approved five-layer boundary and its explicit exclusions are frozen in
  `layered-commit-plan-20260721.md`.
- `generate-layered-staging-plan.ps1 -Check` passes against
  `layered-staging-plan-20260722.json`. The complete Git-visible workset has
  525 paths: 388 candidates split 29/4/145/194/16 across Layers 1-5, 137
  explicit exclusions, and zero unclassified paths. Candidate paths are
  unique and disjoint from exclusions; the pathset SHA-256 is
  `e49805a1471da89893d555b0d88fd5a9361b83fdefbd2d8ea07c8d146f0f757d`.
  The generated JSON SHA-256 is
  `69fd8ad65dd841d909bd1563add80fe86dcbd6c79a336b58b595aaefbca19820`.
- `verify-layered-staging-plan.ps1` passed all five alternate-index dry-runs:
  29/4/145/194/16 exact paths. Strict whitespace coverage was
  29/4/125/165/16; the remaining 20 Layer-3 and 29 Layer-4 files are the 49
  JSON-listed byte-preserved remote patches, source snapshots, and tool
  stdout/stderr/cache evidence. The real Git index remained empty.
- An injected `unexpected/staging-probe.txt` is rejected as unclassified and
  leaves the generated JSON byte-identical. Directory-level staging is
  forbidden because ACL-denied scratch directories are outside the visible
  Git workset.
- The first `git add` attempt was rejected by the platform approval usage quota;
  its reported retry time is 2026-07-25 15:02.
- The Git index remained empty. No local commit, push, SSH command, or remote
  numerical run was performed after that rejection.
