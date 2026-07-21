# Upstream refresh audit: 95c4c080 to 42d3863c

Status: semantic classification complete; clean-candidate runtime revalidation pending.

## Frozen endpoints

- Previous audited upstream tip: `95c4c08009aa6752a6d386289abe1fb4358489ca`
- Frozen upstream base: `42d3863c1d865194d382a085851d1e2e8a39764f`
- Current dirty-tree parent: `b7273e13c77d5ea781f192cea3c4201710b6f9fa`

The public upstream repository was reachable on 2026-07-21, but shell network
access was unavailable. Therefore `42d3863c` is the frozen, locally verified
base for this audit and is not claimed to be the current remote `master` tip.

## Commit classification

| Commit | Subject | Class | Reason and QSGW action |
| --- | --- | --- | --- |
| `eefa0682` | perf(gw): real-valued output in dense Wc CT/FT | U2 | Adds an optional shared dense-Wc representation and CT/FT API that projects scalar complex storage to real storage after a residual check. The default remains off. QSGW must inherit this shared GW path unchanged and its iteration-zero observer must cover the resulting storage/API contract. |
| `deadd008` | test: enable real Wc path in LiH case | U0 | Changes only an upstream regression input outside `src/`. Accept unchanged and retain the upstream regression observer. |
| `9bb5fa4b` | fix(gw): release dense Wc memory per time point | U1 | Changes shared GW intermediate lifetime by clearing each time-point map after conversion. Matrix values, units, ownership, and normalization are intended to remain unchanged. Accept upstream and observe G0W0 plus QSGW iteration zero. |
| `51b9e08e` | tweak: warn large imaginary part instead of abort | U1 | Changes the diagnostic/error policy of the optional real-Wc projection while leaving the projected numerical formula unchanged. QSGW inherits this shared policy; non-finite values still fail. |
| `42d3863c` | refactor(test): split epsilon tests from headwing | U1 | Runtime implementation is unchanged, but the commit edits `src/test`, which is inside the strict protected `src/` inventory. It is therefore conservatively U1 rather than U0 and keeps the shared G0W0/QSGW observers. |

No U3 conflict is introduced by this range. None of these commits requires a
QSGW equation change or restoration of an older shared GW implementation.

## Shared-contract implications

### Real dense Wc path

For scalar/non-spinor calculations with `use_real_dense_gw_wc = true` and no
Wc file output, upstream may store `W_c(R,i\tau)` as real matrices after the
collective residual check

`max |Im W_c| <= 1e-12 + 1e-10 max |Re W_c|`.

Commit `51b9e08e` turns a finite residual violation into a warning before the
real projection; a non-finite value remains fatal. The normal complex path and
the default setting remain unchanged. This is shared G0W0 behavior and is not
reimplemented in `src/qsgw/`.

### Dense Wc lifetime

Commit `9bb5fa4b` releases each `R -> W_c(R,i\tau)` map after its LibRI tensors
are prepared. The required invariant is that the LibRI tensors already own or
otherwise preserve their data for the subsequent self-energy build. Gate 0
and the upstream LiH real-Wc regression observe this lifetime boundary; the
candidate G0W0 and QSGW iteration-zero comparisons observe numerical parity.

## Current protected boundary

Against `42d3863c`, the current working tree has zero diff in the explicitly
protected shared numerical files:

- `src/core/dielecmodel.cpp` and `src/core/dielecmodel.h`
- `src/core/gw.cpp` and `src/core/gw.h`
- `src/core/exx.cpp` and `src/core/exx.h`
- `src/api/compute_g0w0.cpp`
- `src/api/compute_exx.cpp`
- `driver/tasks/g0w0.cpp` and `driver/tasks/g0w0_band.cpp`

The formerly approved read-only head-matrix getter is not present in the
current candidate. QSGW iterative head/wing is instead rejected during input
validation and guarded again at runtime. Consequently the current candidate
does not modify shared G0W0/GW/EXX numerical logic.

## Required observers before acceptance

- clean fish build and exactly 63 CTests;
- unchanged upstream regressions, including LiH with the real dense-Wc option;
- frozen-upstream versus candidate G0W0 on identical inputs;
- candidate QSGW iteration zero versus frozen-upstream G0W0;
- later iterative, Hartree, symmetry, and `qsgw_band` gates only after those
  earlier gates pass.
