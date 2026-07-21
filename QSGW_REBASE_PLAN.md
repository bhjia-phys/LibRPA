# QSGW Upstream Rebase Plan

Status: `in_progress`

Manifest: `qsgw-rebase-manifest.json`

Current gate: `local-audit-and-versioned-runner-preparation-before-layered-commit`

## Scope

Port QSGW as an independent adapter on top of upstream LibRPA. The accepted
scope is:

- ABACUS full-BZ/no-crystal-symmetry QSGW;
- ABACUS IBZ/crystal-symmetry QSGW;
- fixed-basis Hamiltonian iteration with no mixing or linear mixing;
- optional live-density Hartree updates;
- head-wing-off `qsgw_band` using a full-grid AO/real-space operator;
- `H_QSGW` cut modes 0, 1, and 2 in the QSGW band layer.

QSGW iterative head/wing is outside this scope and must fail during input
validation. G0W0 head/wing remains upstream-owned and unchanged.

## Current Freeze

| Field | Value |
|---|---|
| Worktree | `F:/AI_Workspace/Theoretical-Physics/.sisyphus/drafts/_scratch/LibRPA-qsgw-independent-upstream-95c4-20260716` |
| Branch | `codex/qsgw-symmetry-no-headwing-42d-20260720` |
| Committed HEAD | `b7273e13c77d5ea781f192cea3c4201710b6f9fa` |
| Frozen upstream | `42d3863c1d865194d382a085851d1e2e8a39764f` |
| Legacy source | raw `8476213` archive plus the separately recorded compatibility harness |
| ABACUS producer | `dd4216653386d32f79e3219f3ea5dd2d229c1c5a` |
| Dirty state | audited QSGW-only implementation/tests/docs plus driver QSGW wiring; no clean candidate commit yet |
| Live upstream comparison | GitHub compare at `2026-07-21T18:37:20Z`: `master` identical to `42d3863c`, 0 ahead/behind |

`42d3863c` remains the frozen base and was also the live master tip at the
recorded comparison time. Repeat the comparison immediately before opening the
PR.

## Ownership Boundary

- Upstream owns shared G0W0, GW, EXX, chi0, epsilon, symmetry, LibRI, MPI, and
  distributed-matrix numerical behavior.
- QSGW repeatedly calls upstream GW/EXX APIs and stores all iterative state in
  `driver/tasks/qsgw.cpp` and `src/qsgw/`.
- Shared numerical implementations must not be changed to reproduce an old
  QSGW result.
- The current diff against `42d3863c` is empty for:
  `src/core/dielecmodel.*`, `src/core/gw.*`, `src/core/exx.*`,
  `driver/tasks/g0w0*.cpp`, `src/api/compute_g0w0.cpp`, and
  `src/api/compute_exx.cpp`.
- The formerly approved head-matrix getter is now upstream-identical and no
  longer needed because QSGW head/wing is fail-fast.

## Implemented QSGW Contracts

| Area | Current implementation |
|---|---|
| Fixed basis | immutable reference `mf0/wfc0`; live eigenvalues and eigenvectors update after diagonalization; GW/EXX projections temporarily bind the immutable reference basis |
| Input | explicit QSGW contract file; strict QSGW-only parsing rejects malformed numbers, booleans, duplicate trailing invalid values, head/wing, and distributed SCF wavefunctions |
| Symmetry | inherited upstream EXX/GW/RPA switches; IBZ operators are restored to full BZ in AO gauge before Fourier interpolation |
| Mixing | `none` or linear Hamiltonian mixing; default beta `0.2`; the cut is reapplied after mixing for `qsgw_band` |
| Hartree | optional live-density delta-Hartree update; full/truncated Coulomb choice and occupation normalization are explicit and independent of `H_QSGW` cut |
| Band | converged full-grid fixed-basis operator is transformed to AO/real space and projected onto immutable band-path states |
| H cut | mode 0 uncut; mode 1 restores reference KS diagonal beyond the active subspace; mode 2 additionally applies the configured shift |
| Head/wing | every QSGW/QSGW-band `replace_w_head` or `use_pyatb` request is rejected; upstream G0W0 parsing is unchanged |

For the Si semiconductor benchmark, QSGW band occupations are display labels
derived from the grid chemical potential and remain 2/0. Finite-temperature or
metallic band-path occupation equivalence is not part of the current claim.

## Local Verification, 2026-07-21

Full details and executable hashes are in
`qsgw-rebase-evidence/validation/local-audit-20260721.md`.

| Check | Result |
|---|---|
| Current-source native executable tests | 12/12 PASS |
| Focused Python comparator/wiring tests | 29/29 PASS |
| Gate C current Hartree observer/comparator/runner tests | 28/28 PASS |
| Gate C corrected-legacy patch/build/parity tests | 24/24 PASS |
| Regression backend excluding known Windows ACL workspace test | 54/54 PASS |
| Runtime-parameter generation/default check | PASS |
| MSVC syntax checks including latest driver retention fix | PASS with and without `LIBRPA_USE_LIBRI`; Linux link/build pending fish Gate 0 |
| Hartree syntax checks with and without `LIBRPA_USE_LIBRI` | PASS |
| QSGW implementation files in CMake | 22/22 |
| QSGW C++ tests registered in CMake | 23/23 |
| `git diff --check` | PASS |
| Protected G0W0/GW/EXX/API diff | empty |
| Frozen upstream inventory | 30/30 commits and 30/30 commit-map rows |
| Current semantic records | 31 total: 9 U0, 11 U1, 11 U2, 0 U3/U4 |
| Manifest validation | intentionally INVALID only for 59 unfilled remote provenance fields |
| Gate A current comparator tests | 6/6 PASS |
| Gate D band/cut observer tests | 10/10 PASS plus 2 subtests; linear post-mix cut negative fixture included |
| Pinned ABACUS producer/contract tests | 78/78 PASS |
| Exact layered staging plan | 388 candidates, 137 exclusions, 0 unclassified; adversarial probe and all five isolated-index dry-runs passed |

The last immutable fish build at `b7273e13` passed 60/60 CTests. The current
source adds three registered C++ tests, so the next clean fish Gate 0 must list
and pass exactly 63 tests with zero failed and zero Not Run.

Two Windows-only portability defects were found and fixed during this audit:

1. `sha256_file` used a 1 MiB stack buffer and overflowed the default Windows
   stack; the buffer now uses heap storage without changing the hash contract.
2. The no-LibRI Hartree compatibility path accessed a private tensor shape;
   it now uses the public shape accessor while retaining the LibRI branch.

The audit also closed a QSGW input bug: malformed values such as
`qsgw_band0_cut_shift_ha = nan` were silently treated as absent by the shared
parser. QSGW now parses its own assigned values strictly, while G0W0 retains
the upstream parser behavior.

The QSGW driver also used the upstream G0W0 output flag as a permanent
full-Sigma retention switch. Retention is now scoped to the fixed-basis
projection and restores the original user setting on every exit path. QSGW
still consumes the matrices in memory and never calls the G0W0 binary matrix
writer; protected G0W0/GW/EXX sources remain unchanged. The new source-level
driver regression and MSVC syntax checks pass, while full Linux compilation,
linking, and CTest remain part of fish Gate 0.

## Evidence Boundary

The following are supporting evidence, not acceptance for the uncommitted
current source:

- fish 60/60 CTest at `b7273e13`;
- candidate Si k444 symmetry-on versus candidate no-symmetry eigenvalue parity
  within `6e-12 Ha`;
- historical no-head miniter5/miniter10 traces;
- the historical Gate A legacy compatibility harness.

The raw legacy `8476213` source and the compatibility harness are distinct.
The harness changes reader/occupation compatibility and cannot be described as
an unmodified historical executable. New acceptance runs must record that
boundary explicitly and feed the same frozen ABACUS bundle to both sides.

The historical v40 Hartree-on run is also rejected as an oracle. Although its
iteration-1 Hartree delta was exactly zero, its full Coulomb reader replaced
the distributed `Vq_cut` used by EXX/GW and changed non-Hartree channels. A
source-pinned correction now isolates the full Coulomb map, restores reader
state, and has 24/24 local static/synthetic tests; fish build and numerical
parity remain pending.

## Gate Sequence

| Gate | Requirement | Current status |
|---|---|---|
| Gate 0 | clean candidate configure/build; exactly 63/63 CTests; protected diff empty | versioned runner complete locally; remote execution pending |
| A0 | freeze legacy/candidate commits, executables, compiler, MPI/OMP, dependencies, and bundle hashes | historical records exist; refresh for clean candidate pending |
| A1 | same Si k444 symmetry bundle; no-mix miniter2 and linear beta=0.2 miniter5; per-iteration matrix/eigen/gap comparison | historical capability evidence only; clean-candidate rerun pending |
| A2 | candidate symmetry-on versus full-BZ comparison including weights, rotations, phases, time reversal, and Hermiticity | eigenvalue parity passed as supporting evidence; component gate pending |
| B0-B2 | clean pinned ABACUS build and independently frozen no-sym/sym bundles | symmetry producer provenance partly frozen; pinned no-sym/full-BZ producer, observer, contract, and immutable Hartree-bundle runners pass 19 local tests; their dongfang jobs and resulting bundle are pending |
| C0 | k888 29-to-512 symmetry mapping | pending; must not block k444 core conclusion |
| C1 | Hartree-on at least two rounds with charge, units, G=0, Hermiticity, and legacy comparison | current observer/runner pass 28 local tests; corrected legacy patch/build/parity lane passes 24 local tests; pinned full-BZ bundle, fish builds, current physical-default run, and corrected legacy numerical parity are pending |
| D0 | uncut qsgw_band H(k), H(R)/CSR, eigenvalues, gap, and PyATB export | versioned current structural/metamorphic runner complete locally; numerical run and same-dataset legacy acceptance pending |
| D1 | cut modes 0/1/2 and shifts versus legacy band artifacts | versioned current cut-mode runner passes structural checks with mixing disabled; numerical run and same-dataset legacy acceptance pending, including cut-plus-linear-mixing ordering |
| PR | formal committed regressions, latest-upstream rebase, G0W0/RPA/EXX regressions, deterministic MPI/OMP checks | pending |

Unified numerical thresholds:

- eigenvalue max absolute difference: `1e-6 Ha`;
- gap difference: `1e-5 eV`;
- matrix relative Frobenius difference: `1e-8`;
- Hermiticity and orthogonality residuals: `1e-10`.

## Formal Regression Status

`regression_tests/testsuite.xml` currently contains two disabled historical
placeholders. They are intentionally not accepted references because no small
oracle-backed QSGW dataset is committed. Enabling them before importing a
curated two-round bundle would create a false regression claim.

## Open Issues

1. Produce clean layered commits from the audited dirty source. The approved
   Git index write is currently blocked by the platform approval quota until
   2026-07-25 15:02 local time; no workaround is permitted.
2. Run fish Gate 0 for the clean candidate and archive compiler/build/CTest
   provenance.
3. Rerun A1/A2 from iteration 0 on the same Si k444 bundle for legacy and the
   clean candidate.
4. Produce the pinned no-sym/full-BZ ABACUS bundle. Run current physical-default
   Hartree C1 and band/cut D0-D1 separately from the corrected legacy
   truncated/legacy-normalization parity gate.
5. Resolve the mode-1/mode-2 linear-mixing compatibility question with a
   controlled legacy/current run. Legacy initializes the mixer from uncut
   `H_KS0` and does not reapply the cut after mixing; current `qsgw_band`
   initializes from the cut Hamiltonian and treats the cut as an exact
   post-mixing constraint. The current-only linear Gate D runner verifies the
   latter closure and rejects the legacy ordering synthetically, but it is not
   a substitute for the required same-dataset legacy/current numerical run.
6. Import a genuinely small, immutable, two-round ABACUS regression dataset.

## Next Action

The manifest audit and exact five-layer staging plan are complete. When Git
index approval returns, rerun the plan in `-Check` mode, stage only each
layer's listed paths, create the five commits, and regenerate the manifest
against the Layer 4 source commit. When network access returns, push those
exact commits, run fish Gate 0, assemble the Hartree bundle, and then start
the clean-candidate A/C/D numerical gates.
