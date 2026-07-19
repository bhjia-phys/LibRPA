# Upstream refresh audit: 1376ee4f to 95c4c080

Status: runtime revalidation in progress.

## Frozen endpoints

- Previous upstream base: `1376ee4f45a7611a55c5b92c4ba41409d515bcea`
- Refreshed upstream tip: `95c4c08009aa6752a6d386289abe1fb4358489ca`
- Pre-refresh QSGW candidate: `ce1e1859316e1b659df0178b553bbbfba2ca1173`
- Rebasing result: `b2bc09d00ab49a2ff39c46c72d83ab89bc5ddac6`
- Rebase result: all six QSGW commits replayed without a text conflict.

The absence of text conflicts is not classification evidence. The runtime and
layout effects below were inspected separately.

## Commit classification

| Commit | Subject | Class | Reason and QSGW action |
| --- | --- | --- | --- |
| `293dacc5` | Add BN symmetry kpara GW headwing regression | U0 | Regression input, references, and testsuite registration only. Accept unchanged and run the new case on the candidate. |
| `b66d5283` | test: cover empty-rank symmetry W(q) restore | U1 | Shared head/wing observer under `src/test`; no runtime implementation hunk. Accept unchanged and retain the test in Gate 0. |
| `b5127386` | test: run BN symmetry RPA with k parallelism | U0 | Regression input/reference metadata only. Accept unchanged. |
| `2b34a92f` | redistribute head/wing Cs data by k point | U2 | Changes Cs Fourier-transform communicator and k-owner redistribution in shared Dataset/dielectric code. QSGW must consume the upstream Dataset contract without restoring the old path. |
| `c3563afd` | update headwing test file | U1 | Shared test coverage and regression input changes only. Accept unchanged and retain the observers. |
| `be966e25` | use 128 rectangular block size for Cs rotation | U2 | Changes the distributed matrix layout used by head/wing Cs rotation. QSGW inherits the upstream layout through Dataset and `diele_func`. |
| `3ffe9ab8` | read eigenvectors as 2D k-communicator blocks | U2 | Changes WFC API/layout/ownership across reader, Dataset, EXX, G0W0, dielectric, BLACS, and mean-field MPI code. QSGW must use the resulting Dataset state and may not patch shared GW/EXX numerics. |
| `2df7b08b` | fix local WFC rotation for symmetric head/wing | U2 | Corrects which local WFC block is rotated for symmetry. QSGW head/wing must inherit this ownership rule and is checked by the upstream BN regression plus QSGW head gates. |
| `12f43c38` | redistribute eigenvectors to capped rectangular blocks | U2 | Changes permanent SCF and band WFC descriptors and redistribution boundaries. QSGW must preserve the descriptors maintained by Dataset. |
| `14261e28` | pair full-BZ WFCs and velocities for symmetry | U2 | Changes the symmetry pairing contract for WFC and velocity data. QSGW same-grid head/wing uses the upstream pairing and its own immutable/live velocity state only after input alignment. |
| `95c4c080` | accept valid WFC block-cyclic layouts | U2 | Changes validation of legal WFC block-cyclic layouts. QSGW must not impose the superseded dense-root shape assumption. |

No new U3 conflict was found in this refresh. The previously approved U3
read-only head matrix getter remains the only shared-core candidate hunk.

## Shared-path boundary after rebase

The rebased candidate has zero diff from `95c4c080` in:

- `src/core/gw.cpp`
- `src/core/gw.h`
- `src/core/exx.cpp`
- `src/core/exx.h`
- `src/api/compute_g0w0.cpp`
- `src/api/compute_exx.cpp`

The only shared numerical-path diff is the approved five-line const getter in
`src/core/dielecmodel.h`:

```cpp
const std::vector<matrix_m<std::complex<double>>>&
get_head_matrices() const noexcept
{
    return head;
}
```

It exposes existing state read-only. It does not change construction,
calculation, MPI ownership, normalization, symmetry, or G0W0 control flow.

## Contract impact on QSGW

| Invariant | Upstream symbols | QSGW consumer | Required observer |
| --- | --- | --- | --- |
| Shared G0W0 and EXX numerics remain upstream-owned | `librpa_build_exx`, `librpa_build_g0w0_sigma`, `Exx`, `G0W0` | `driver/tasks/qsgw.cpp` calls Dataset API and projects in a scoped fixed basis | Latest upstream/candidate G0W0 A/B; Gate 2 iteration-one Sigma comparison |
| Fixed reference basis is immutable while live eigenpairs evolve | `Dataset::mf`, WFC descriptors | QSGW `reference`, live `dataset->mf`, fixed-basis projection scope | Gate 2 fixed-basis observer; Gate 3 miniter5/miniter10 trajectory replay |
| Head/wing WFC and velocity refer to the same physical full-BZ member | symmetry WFC/velocity pairing and `diele_func` | QSGW same-grid and independent head/wing state refresh | Upstream BN symmetry+kpara regression; QSGW head-only and wing controlled pairs |
| K-parallel WFC storage follows Dataset descriptors | `desc_wfc_kb`, `desc_band_wfc_kb`, redistribution methods | QSGW grid and future `qsgw_band` paths | Upstream tests, band gate, symmetry gate, MPI/OMP gate |

## Runtime evidence available

Fish Gate 0 for `95c4c080` versus `b2bc09d0`:

- Upstream CTest: 38/38 passed.
- Candidate CTest: 59/59 passed.
- Official `g0w0_aims_Si_libri`: candidate versus upstream, candidate versus
  official reference, and upstream versus official reference all have zero
  maximum difference over 486 rows.
- Candidate executable SHA-256:
  `955586f0b653e5ff53c1553e08b1d24db6954b86cebec87f4350cd6c9dddc0d6`.
- Evidence manifest SHA-256:
  `b94e92356bbfb6d7e4ed544de2d2d4b4b509bcc6fbfdd02f714011de9fb1d17d`.

New upstream BN symmetry+kpara+head/wing regression on the candidate:

- 1/1 case passed.
- Wing-mu real/absolute diagnostic difference: `4.623e-20`.
- Wing-lambda real/absolute diagnostic difference: `3.805e-18`.
- QP grid (208 rows), EXX band, and GW band comparisons have zero maximum
  difference from their upstream references.
- Evidence manifest SHA-256:
  `b547ef991e5c2aa924e770b31bda1e08b61ae857973b8de38a5f43fdf0a705a9`.

Dongfang clean builds:

- Upstream executable SHA-256:
  `af93da0fccd7c9346734234f0c2ef890f3a156938fd3048294de93173434e268`.
- Candidate executable SHA-256:
  `a898ef66390d0207158c0f24d1b311fabde1bb11f4a1cd89b6082aee75e54998`.

The refreshed Dongfang Gate 1 and later QSGW gates are not accepted until
their own `COMPLETE`, Slurm terminal state, manifests, and numerical observers
pass.
