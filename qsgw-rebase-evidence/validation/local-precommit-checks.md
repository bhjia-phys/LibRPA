# Local Pre-commit Checks

Date: `2026-07-15`

Source state: clean immutable candidate source `7d69a18cff419be3139d61b8405a8dbca6b53fd8` on `codex/qsgw-independent-upstream-1376-20260714`, based on `1376ee4f45a7611a55c5b92c4ba41409d515bcea`. Planning/evidence files are committed separately and are not part of the executable source identity.

These checks are supporting evidence only. This Windows host has CMake but no C/C++ compiler, MPI launcher or Ninja, so they do not replace remote configure/build/CTest or numerical gates.

## Results

| Check | Result |
|---|---|
| `python -B -m unittest test_cmp_qsgw -v` from `regression_tests/backend/comparisons` | PASS, 18/18 |
| `python -B -m unittest discover -s regression_tests/backend/comparisons -p "test_*.py"` | PASS, 41/41 |
| Python JSON parse of `qsgw-rebase-manifest.json` and import of `cmp_qsgw` | PASS |
| Source candidate `git diff --check` | PASS |
| Protected numerical paths against `HEAD` | PASS, zero modified files among G0W0/GW/EXX/chi0/epsilon/dielecmodel.cpp |
| `get_head_matrices` source references | PASS, only `src/core/dielecmodel.h` and QSGW-only `driver/tasks/qsgw.cpp` |
| Rebase manifest validator | EXPECTED INVALID, 62 remote build/dataset/input/environment provenance errors; zero commit-role, comparative-provenance, U3 or approval errors |

The first attempted focused invocation from the repository root used `python -m unittest regression_tests.backend.comparisons.test_cmp_qsgw`. The test module directly imports `cmp_qsgw`, so that invocation failed module discovery. Running from the comparison directory, which matches the repository's discovery convention, passed all 18 tests.

The evidence commit intentionally preserves raw Git patch and status snapshots byte-for-byte. A full staged `git diff --check` therefore reports pre-existing trailing whitespace embedded inside those snapshots; the authored Markdown, JSON and PowerShell evidence files pass the scoped whitespace check.

## Artifact hashes

| Artifact | SHA256 |
|---|---|
| `regression_tests/backend/comparisons/cmp_qsgw.py` | `7cb36796e97e88c3a3f4b02a1d9f2dcca4b50ccf6f99689eaeb997352178be9e` |
| `regression_tests/backend/comparisons/test_cmp_qsgw.py` | `a52a79fc803c48af413053b440bb172d10695686bba102f58ce6399695fe0ab1` |
| `qsgw-rebase-manifest.json` at check time | `67f438452cd2a6e22903de16fe252ca649aefd496c5547f9c6205d6a7c4d23ea` |
| `qsgw-rebase-evidence/validation/manifest-validator.txt` | `853cd8f8a082e82764c9e4137a0c57284cff5df3a16283ef9936aaa94417e329` |

## Remaining required evidence

- Linux configure/build and complete CTest for pure upstream and candidate commits.
- Byte-identical-input upstream/candidate G0W0 direct A/B.
- Old/new QSGW numerical gates in the required order.
- Executable, compiler/library, input and environment provenance needed to make the manifest valid.
