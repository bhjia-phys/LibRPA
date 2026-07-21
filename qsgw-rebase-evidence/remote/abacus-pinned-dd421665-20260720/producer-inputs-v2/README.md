# Pinned ABACUS Si producer inputs v2

## Frozen source

- Repository: `https://github.com/AroundPeking/abacus-develop.git`
- Branch provenance: `master_ghj`
- Commit: `dd4216653386d32f79e3219f3ea5dd2d229c1c5a`
- Tree: `7284d42382b3555081badfdb4a15691d0bd1ec3d`
- LibPAW submodule: `c211c0ab330adf3cc374f50ab3edee46b174e64c`

The accepted source freeze is under `fish-source-freeze-v3/`. The clean pinned
build completed as Slurm job 2378198 under
`abacus-pinned-dd421665-build-20260720-v2/build-attempt-v3/`:

- executable: `build/abacus_3p`
- executable SHA256:
  `a2676c36e318da831339cfb147c6027f3b7aed0b40925fa4a6086a947a403b17`

The earlier held job 2377816 is not part of this freeze and must not be used.

The first pinned Si k444 symmetry producer run is Slurm job 2381071 under
`abacus-pinned-dd421665-si-k444-symmetry-20260720-v1/`. Its ABACUS numerical
stage completed, but the immutable run root remains marked `FAILED` because
the original observer assumed an incorrect Ha/eV constant and legacy Vxc
headers. Recovery is only through a separate successful postcheck directory;
bundle assembly requires postcheck-v4 `COMPLETE` and refuses to rewrite the
original producer markers.

## Source-proved producer contract

- `rpa = 1` calls `RPA_LRI::postSCF` and emits `band_out`,
  `KS_eigenvector_*.dat`, `stru_out`, `bz_sampling_out`, RI coefficients,
  cut Coulomb, and full Ewald Coulomb.
- `out_librpa_reader_version = 0` is the pinned legacy text format. Version 1
  is a distinct packed/binary format and is not used for the shared
  legacy/candidate oracle until legacy-reader compatibility is separately
  proved.
- The pinned ABACUS source writes `basis_wfc_out`, `basis_aux_out`, and the
  combined `basis_out` only on its reader-v1 branch. A clean reader-v0 producer
  run must therefore contain no such files. Dataset assembly generates the
  split basis metadata once from the frozen ORB/ABFS shell layout and freezes
  that same assembled bundle for both legacy and candidate consumers.
- `out_mat_xc = 1` emits upper-triangular `C^dagger Vxc C` text matrices in
  Ry as `vxck*_nao.txt`, plus `vxc_out.dat`; LibRPA converts the matrix values
  to Ha. Dataset assembly v2 retains only those producer-native names. The raw
  exact-847 reader and its Scheme-A harness accept the modern comment/`Row`
  schema directly, as does `src/qsgw/vxc_io.cpp`; both apply the same 0.5
  Ry-to-Ha factor. Their frozen source contract is recorded by
  `native_vxck_reader_compatibility_v1.json`. Numerical legacy/candidate
  parity remains a later same-bundle runtime gate, so no `vxcs1k*` aliases or
  format conversion are introduced here.
- `out_mat_xc2 = 1` emits `Vxc_R_spin0.csr` in the AO real-space basis for
  later `qsgw_band` validation.
- `out_wfc_lcao = 1` is used only for the NSCF band path. The SCF RPA producer
  already emits the `KS_eigenvector_*.dat` files consumed by LibRPA.
- `out_app_flag = 0` is explicit in every controlled INPUT so a fresh run
  truncates output instead of appending to any pre-existing artifact.
- `out_ri_cv = 0` avoids diagnostic real-space C/V dumps not consumed by the
  normal LibRPA reader.
- `fold_C` and `n_params_anacon` are not ABACUS producer inputs.

The exact source excerpts are archived in
`pinned_abacus_output_source_audit_v1.out` and
`pinned_abacus_output_filenames_v1.out`.

### Vxc basis contract correction

The frozen producer INPUT assets remain version 2, but
`prepare_abacus_qsgw_ibz_contract_v2.py` and
`assemble_pinned_abacus_si_k444_symmetry_bundle_v2.slurm` are rejected
pre-runtime evidence. They mislabeled the producer-native `out_mat_xc`
matrices as `nao/ao_bloch`. Because ABACUS has already evaluated
`C^dagger Vxc_AO C` before writing an `nbands`-square matrix, that manifest
would make the candidate apply a second `C^dagger V C` projection whenever
`n_bands == n_aos` hid the dimension mismatch.

The source-backed correction is frozen in
`native_vxck_state_basis_v1.json`. Runtime bundle assembly starts at version
3 and binds native `vxck*_nao.txt` as `Ry/state/mf0_state`, with dimensions
checked against `n_bands` and no basis transform. The true
`Ry/nao/ao_bloch` candidate path remains available only for an actual AO-basis
matrix producer.

This v2 contract supersedes `producer-inputs-v1`, which is retained as evidence
but omitted the explicit `out_app_flag = 0` setting. Run
`python3 validate_producer_inputs_v2.py` before staging, and pass
`--assets-dir DIR` after copying the frozen UPF, ORB, and ABFS files.

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
- `vxc_out.dat` and every producer-native `vxck*_nao.txt` matrix
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

The k444 full-BZ B2 lane is versioned by
`run_pinned_abacus_si_k444_fullbz_v1.slurm`,
`validate_abacus_si_k444_fullbz_output_v1.py`,
`prepare_abacus_qsgw_fullbz_contract_v1.py`, and
`assemble_pinned_abacus_si_k444_fullbz_hartree_bundle_v1.slurm`. Their local
tests require 64/64 sampling, uniform `1/64` weights, 64 native state-basis
Vxc matrices, and separately hashed full/truncated Hartree contracts. These
checks do not claim that the dongfang producer or bundle job has run.
