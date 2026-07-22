# exact847 versus current QSGW call-chain audit

## Scope

This audit covers the regular-grid QSGW update used by the Si k444 symmetry
Gate A1. Hartree, iterative head/wing, band-path interpolation, and H_QSGW cut
are disabled. Numerical parity remains pending until the miniter2 comparator
passes.

## Matched numerical contract

| Stage | exact847 `task_qsgw_band0` | Current `task=qsgw` |
|---|---|---|
| chi0 coefficients | `Cs_shrinked_data` when `use_shrink_abfs=true` and `use_shrink_chi=false` | `cs_data_shrink` and `basis_aux_shrink` under the same flags |
| chi0 shrink transform | disabled; the direct shrinked-Cs route does not consume `sinvS` | disabled; an empty `sinvS` map is passed to the shrink-basis chi0 build |
| EXX coefficients | shrinked Cs | shrinked Cs |
| EXX Coulomb | full `Vq` because `use_fullcoul_exx=true` | full `vq` because `use_fullcoul_exx=true` |
| epsilon Coulomb | full `Vq` in the exact847 QSGW screened-interaction call | full `vq` because `use_fullcoul_eps=true` |
| Wc output Coulomb | truncated `Vq_cut` because `use_fullcoul_wc=false` | truncated `vq_cut` because `use_fullcoul_wc=false` |
| Sigma real-space coefficients | full `Cs_data` | full `cs_data` |
| Sigma shrink reconstruction | loaded `shrink_sinvS_*` plus shrink/full auxiliary bases | `sinvS`, shrink/full auxiliary bases, and the BLACS descriptors |
| projection basis | immutable initial eigenvectors (`eigenvectors0`) | scoped immutable `mf0` reference eigenvectors |
| live state between iterations | eigenvalues and zero-temperature occupations updated | eigenvalues and zero-temperature occupations updated; fixed reference basis retained |

## Source anchors

Legacy source is frozen by the Gate A0 executable and source hashes. The
relevant exact847 locations are:

- `driver/task_qsgw_band_0.cpp:1537`: direct shrinked-Cs chi0 selection.
- `driver/task_qsgw_band_0.cpp:1721`: EXX construction.
- `driver/task_qsgw_band_0.cpp:1738`: shrinked-Cs EXX input.
- `driver/task_qsgw_band_0.cpp:1758`: ScaLAPACK Wc route.
- `driver/task_qsgw_band_0.cpp:1807`: shrink inverse loading.
- `driver/task_qsgw_band_0.cpp:1814`: full-Cs Sigma real-space construction.
- `driver/task_qsgw_band_0.cpp:1824`: fixed-basis Sigma projection.

Current source anchors are:

- `src/api/compute_g0w0.cpp:718`: shrink-basis EXX selection.
- `src/api/compute_g0w0.cpp:721`: full versus truncated EXX Coulomb selection.
- `src/api/compute_g0w0.cpp:759`: direct shrinked-Cs chi0 selection.
- `src/api/compute_g0w0.cpp:830`: independent epsilon and Wc Coulomb selection.
- `src/api/compute_g0w0.cpp:880`: full-Cs Sigma build with shrink reconstruction inputs.
- `driver/tasks/qsgw.cpp:1103`: exact upstream G0W0 builder invocation.
- `driver/tasks/qsgw.cpp:1114`: scoped immutable reference basis.
- `driver/tasks/qsgw.cpp:1116`: EXX and Sigma fixed-basis projection calls.

## Protected-source boundary

Relative to frozen upstream commit
`42d3863c1d865194d382a085851d1e2e8a39764f`, the accepted candidate does not
modify `src/core/gw.cpp`, `src/core/gw.h`, `src/core/exx.cpp`,
`src/core/exx.h`, or the upstream G0W0 task implementations. QSGW invokes the
upstream builders and performs its fixed-basis collection and iterative update
in the independent QSGW adapter.

## Important exclusion

The failed run documented in `EXACT847_FULL_ABF_FAILURE_20260722.md` used the
exact847 default `use_shrink_chi=true`. That route is not equivalent to the
current Gate A1 contract and is excluded from the parity comparison.

## Claim boundary

The source call chains are aligned for the stated flags. This is a structural
result only. It becomes a numerical merge result only after the exact847 v3
legacy run, current contract-v6 run, and strict checkpoint comparator all pass.
