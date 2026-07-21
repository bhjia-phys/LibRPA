# Fish Gate 0 current runner

This runner was used for the accepted clean candidate Gate 0 at
`66bfe1cfd35c983222935d250039a8fe5c4b7af1`. The immutable output and complete
checksum manifest are stored in `66bfe1cf-v1/`.

The runner performs these checks on fish:

- detached, clean checkouts of frozen upstream `42d3863c` and the exact
  `CANDIDATE_COMMIT`;
- identical oneAPI/CMake configuration for both sides;
- exactly 39 upstream and 63 candidate CTests, all passing;
- ten focused QSGW parser/Hartree/Fourier/band/cut tests;
- focused Python comparator/driver-wiring tests and runtime-parameter docs;
- an explicit `DOCS_PYTHON` with PyYAML, generating runtime-parameter docs in
  the immutable run root before checking defaults and byte-idempotence;
- zero diff in the explicitly protected G0W0/GW/EXX files;
- immutable compiler, MPI, CMake, executable, test, and checksum evidence.

After the runner itself belongs to the clean candidate commit, materialize it
from the fish bare repository and hash those exact bytes before execution:

```bash
candidate=<40-character-commit>
script=qsgw-rebase-evidence/remote/fish-gate0-current-20260721/run_fish_gate0_current_v1.sh
runner=/tmp/run_fish_gate0_current_v1.sh
git --git-dir=/tmp/librpa-qsgw-cb294020-84e6b0a/repo.git show "$candidate:$script" >"$runner"
runner_sha=$(sha256sum "$runner" | awk '{print $1}')
CANDIDATE_COMMIT="$candidate" RUNNER_SHA256="$runner_sha" \
  RUN_TAG="20260721-${candidate:0:8}-v1" bash "$runner"
```

Do not reuse a `RUN_TAG`. A successful run ends with `GREEN_CONFIRMED`; a
failed run records `FAILED` and is never overwritten.
