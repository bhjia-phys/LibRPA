# Pinned ABACUS Si producer inputs v1

## Frozen source

- Repository: `https://github.com/AroundPeking/abacus-develop.git`
- Branch provenance: `master_ghj`
- Commit: `dd4216653386d32f79e3219f3ea5dd2d229c1c5a`
- Tree: `7284d42382b3555081badfdb4a15691d0bd1ec3d`
- LibPAW submodule: `c211c0ab330adf3cc374f50ab3edee46b174e64c`

The accepted source freeze is under `fish-source-freeze-v3/`. The executable
hash is not frozen until Slurm job 2377816 completes successfully.

## Source-proved producer contract

- `rpa = 1` calls `RPA_LRI::postSCF` and emits `band_out`,
  `KS_eigenvector_*.dat`, `stru_out`, `bz_sampling_out`, RI coefficients,
  cut Coulomb, and full Ewald Coulomb.
- `out_librpa_reader_version = 0` is the pinned legacy text format. Version 1
  is a distinct packed/binary format and is not used for the shared
  legacy/candidate oracle until legacy-reader compatibility is separately
  proved.
- `out_mat_xc = 1` emits upper-triangular `C^dagger Vxc C` text matrices in
  Ry plus `vxc_out.dat`; LibRPA converts the matrix values to Ha.
- `out_mat_xc2 = 1` emits `Vxc_R_spin0.csr` in the AO real-space basis for
  later `qsgw_band` validation.
- `out_wfc_lcao = 1` is used only for the NSCF band path. The SCF RPA producer
  already emits the `KS_eigenvector_*.dat` files consumed by LibRPA.
- `out_ri_cv = 0` avoids diagnostic real-space C/V dumps not consumed by the
  normal LibRPA reader.
- `fold_C` and `n_params_anacon` are not ABACUS producer inputs.

The exact source excerpts are archived in
`pinned_abacus_output_source_audit_v1.out` and
`pinned_abacus_output_filenames_v1.out`.

## Controlled cases

- `INPUT_scf_fullbz` plus `KPT_k444` or `KPT_k888`: `symmetry = -1`, full BZ.
- `INPUT_scf_symmetry` plus `KPT_k444` or `KPT_k888`: `symmetry = 1`, crystal
  symmetry and real-space EXX symmetry enabled.
- `INPUT_nscf_band` plus `KPT_band`: full band path from a frozen SCF charge.

The first oracle deliberately sets `shrink_abfs_pca_thr = -1`. This keeps
`Cs_data`, `coulomb_mat`, and `coulomb_cut` in one unshrunk auxiliary basis and
matches LibRPA `use_shrink_abfs = false`. The historical approximately 0.97 eV
QSGW gap used shrink and remains an orientation value, not the exact oracle.

Every run must copy, not relink, the three assets named in `ASSET_SHA256SUMS`.
The complete generated output bundle is immutable and must be consumed by both
legacy QSGW and the candidate. A run is not accepted until `INPUT.info`, k-grid
cardinality/mapping, every required output, and a recursive SHA256 manifest are
recorded.

## Required SCF outputs

- `band_out`, `KS_eigenvector_*.dat`, `stru_out`, `bz_sampling_out`
- `vxc_out.dat` and every `vxcs*k*_nao.txt` matrix
- `Cs_data_*`, `coulomb_mat_*`, `coulomb_cut_*`
- `Vxc_R_spin0.csr`
- `data-HR-sparse_SPIN0.csr`, `data-SR-sparse_SPIN0.csr`, and position matrix
  outputs selected by `out_mat_r`
- symmetry metadata encoded by `stru_out` and `bz_sampling_out`; legacy
  sidecars are inventoried when present but are not substituted from another
  producer run

Expected full-grid cardinalities are 64 for k444 and 512 for k888. The
symmetry-reduced cardinalities, including the expected k888 29 -> 512 map, are
runtime assertions to verify from the pinned output rather than assumptions to
write into the oracle.
