# QSGW Hartree source audit v1

Status: code-path equivalence is supported by source inspection; numerical
equivalence remains pending Gate 6A/6B.

## Frozen legacy sources

- `legacy_hartree.cpp`: `aebea67a960753aed35f492b784e978797dd820871acfd8d1643491b82169215`
- `legacy_task_qsgw.cpp`: `6c79c6731d3420b3c9f70c97dea8c63236bca2952216bb67d9e6b9839552c9d1`
- `legacy_LRI-cal_hartree.hpp`: `d734909b7a60d1d77235cb909c903584a2226e8b523cb5715a86f627cf7c3c93`

These files were copied from the immutable v36 legacy Hartree build.

## Iteration semantics

The legacy driver builds the total Hartree operator from the live mean field
on every iteration. On iteration 1 it stores `Hartree_0`; on later iterations
it adds `Hartree_i - Hartree_0` to the pure GW correlation potential. The
current adapter builds the density difference `rho(live) - rho(mf0)` and
contracts it directly. The Hartree contraction and all Fourier/projection
steps are linear, so these two constructions are algebraically equivalent.

Both paths produce a zero Hartree correction on iteration 1. Gate 6B must
verify this from both traces and compare the nonzero iteration-2 correction.

## Fixed basis

The legacy path calls `Hartree.build_KS_kgrid0()`, which projects with
`meanfield.get_eigenvectors0()`. The current path calls
`project_periodic_operator_to_fixed_basis(..., reference, ...)`, where
`reference` is the immutable `mf0`. Both therefore project the live-density
Hartree correction into the initial fixed state basis.

## Normalization

The legacy LibRI contraction divides the auxiliary Hartree potential by
`Nk`. The legacy k-to-R transform divides by `Nk` again.

The current full-grid density path first forms an unnormalized R-space
Fourier sum and then divides by `Nk` in the inverse R-to-k reconstruction;
the two transforms recover the unweighted density matrix `D(k)`. The
`legacy_extra_inverse_nk` kernel option then supplies the legacy LibRI
auxiliary-potential factor, and the current k-to-R transform supplies the
second legacy factor. Therefore the benchmark must set:

`qsgw_hartree_normalization = legacy_extra_inverse_nk`

The default `weighted_occupations` mode is not the legacy-oracle contract.

## Shrink reader binding

Both the legacy driver and current `select_hartree_reader_route()` choose
`Cs_shrinked_data_*` when `use_shrink_abfs=true`. The current reader also
uses the shrink RI files themselves as the auxiliary-basis provenance when
no explicit `basis_aux_shrink_out`, `basis_out_shrink`, or
`basis_out.shrink_backup` exists.

Historical contract generator v31 incorrectly declared `Cs_data_*` and
`basis_aux_out` for Hartree even in shrink mode. It cannot be used for Gate 6.
Generator v32 now mirrors the actual reader route and is covered by four
full/shrink role-selection tests on fish.

- v32 generator SHA256: `6beb94b0dc8c314cc253f996b7cc31f4ce1efd3737ece30da4100636dc443989`
- v32 test SHA256: `ba02e4b58acdab8a5ed8e8b6c047649cfcea8d84716583e437627a7b31efc40f`
- fish result: `4/4` passed

## Numerical gates

1. Gate 6A: same v36 legacy executable and current candidate, shrink on,
   Hartree off, linear beta 0.2, miniter2.
2. Gate 6B: identical inputs, binaries, mixing and resources, Hartree on,
   truncated Coulomb, legacy normalization, miniter2.
3. Gate 6B observers: component comparison including `delta_vh`, first
   correction zero, occupation-weighted charge conservation, Hermiticity,
   closure, eigenvalues, wavefunctions and gap.

No shared G0W0/GW/EXX/epsilon/chi0 numerical source was changed by this audit.
