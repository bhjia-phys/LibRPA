# QSGW Layered Commit Plan

This file freezes the five approved local commit boundaries for the audited
working tree based on freeze parent
`b7273e13c77d5ea781f192cea3c4201710b6f9fa` and frozen upstream
`42d3863c1d865194d382a085851d1e2e8a39764f`.

## Layer 1: Independent QSGW Product Code

Planned subject: `qsgw: complete independent Hartree and band workflows`

- `docs/user_guide/runtime_parameters.yml`
- `driver/driver.cpp`, `driver/driver.h`, `driver/inputfile.cpp`
- `driver/tasks/qsgw.cpp`, `driver/test/test_qsgw_inputfile.cpp`
- the modified/new files under `src/qsgw/` shown by the audited status
- the modified/new `src/test/test_qsgw_*.cpp` files and
  `src/test/CMakeLists.txt`

This layer must have an empty protected diff for the shared G0W0/GW/EXX and
API paths listed in `local-audit-20260721.md`.

## Layer 2: Formal Regression Wiring

Planned subject: `test(qsgw): wire formal regression observers`

- `regression_tests/backend/comparisons/cmp_qsgw.py`
- `regression_tests/backend/comparisons/test_cmp_qsgw.py`
- `regression_tests/backend/test_qsgw_driver_wiring.py`
- `regression_tests/testsuite.xml`

The XML entries remain disabled until a small immutable two-round numerical
oracle is committed.

## Layer 3: Pinned Producer and Reader Contracts

Planned subject: `test(qsgw): freeze pinned ABACUS producer contracts`

- versioned scripts, tests, input decks, compact provenance, and SHA records in
  `qsgw-rebase-evidence/remote/abacus-pinned-dd421665-20260720/`
- the clean `dongfang-may-si-k888-reference`, `dongfang-si-k888-job2375481`,
  `fish-source-freeze-v3`, `producer-inputs-v1`, and `producer-inputs-v2`
  subdirectories
- versioned scripts and compact Gate evidence in
  `qsgw-rebase-evidence/remote/fish-reader-binding-20260720/`

The failed job 2375481 is failure evidence only and must never be named as an
ABACUS or QSGW numerical oracle.

## Layer 4: Versioned Gate Observers and Runners

Planned subject: `test(qsgw): add versioned A C and D gate runners`

- `fish-gate0-current-20260721/`
- `fish-gate-a-current-20260721/`
- `fish-gate-c-current-20260721/`
- `fish-gate-c-legacy-corrected-20260721/`
- `fish-gate-d-current-20260721/`
- curated source, runner, observer, patch, provenance, and comparison records
  from `fish-gate-a-symmetry-20260720/`
- compact historical-oracle provenance in
  `dongfang-historical-toolchain-baseline-2380797/`

Historical attempts are evidence, not acceptance. The current A/C/D runners
remain locally tested but remotely unexecuted for the clean candidate.

## Layer 5: Upstream and Audit Metadata

Planned subject: `docs(qsgw): bind rebase audit to candidate source`

- `QSGW_REBASE_PLAN.md`
- `qsgw-rebase-manifest.json`
- `qsgw-rebase-evidence/validation/refresh-revised-goal-manifest.ps1`
- `qsgw-rebase-evidence/validation/generate-layered-staging-plan.ps1`
- `qsgw-rebase-evidence/validation/verify-layered-staging-plan.ps1`
- `qsgw-rebase-evidence/validation/layered-staging-plan-20260722.json`
- `qsgw-rebase-evidence/validation/local-audit-20260721.md`
- manifest validator output and exit code
- the three `upstream-*-to-42d3863c` inventory files and refresh report
- the live `upstream-master-compare-20260722.json` evidence
- this commit plan

After Layers 1-4 exist, regenerate Layer 5 so candidate provenance points to
the Layer 4 commit. The metadata commit itself must not be its own source hash.

## Never Stage

- any `__pycache__`, `*.pyc`, object, executable, CMake build, or pytest cache
- any `tmp*`, `*-test-scratch`, `band-pipeline-*-p7q2d8gv`, or ACL-denied test
  workspace
- `patch-work-*` and `patch-apply-test-*` checkout copies
- `legacy-h0-comparator-tmp-*`
- large derived `legacy-runtime.strings`; retain its hash/provenance instead
- raw ignored logs unless a compact failure summary has not otherwise been
  preserved

The generated staging plan is the only approved path source. It currently
classifies the complete Git-visible workset as 388 candidate files in five
disjoint layers and 137 explicit exclusions, with zero unclassified paths.
Never replace its exact path lists with directory-level `git add` commands.

Run the staging-plan generator in `-Check` mode once immediately before Layer 1,
while HEAD and the empty index still match the frozen source state. The plan is
deliberately bound to that state and is expected to become stale after Layer 1
is committed. For each layer, stage only its JSON-listed paths, run
`verify-layered-staging-plan.ps1 -Layer layerN`, and inspect the cached diff.
The verifier enforces exact staged paths and strict whitespace on current code,
scripts, and documents. Its JSON-listed 49 byte-preserved remote evidence files
are exempt from rewriting. All five layers passed isolated-index dry-runs and
left the real index empty. After Layer 5, rerun the local test matrix, manifest
idempotence check, and protected-diff check.

## Current Blocker

On 2026-07-21 the approved Layer 1 `git add` was rejected by the platform's
approval usage quota. The reported retry time is 2026-07-25 15:02. No index
entry or commit was created, and no alternative path was used.
