# Corrected legacy Hartree oracle

This directory contains the source-only correction required before the old
QSGW Hartree implementation can be used as a parity oracle.

The patch applies to the exact v36 `driver/task_qsgw.cpp` archived at:

`../dongfang-gates-20260715-7d69a18c/legacy-hartree-v36-source-audit/legacy_task_qsgw.cpp`

Its SHA256 is
`6c79c6731d3420b3c9f70c97dea8c63236bca2952216bb67d9e6b9839552c9d1`.
After `dos2unix`, the source SHA256 is
`50215d59480b46bea7e3edb34ac5e453165afcf0a66bc6a48053b17b491b303f`.
The patch SHA256 is
`127b8c3927de328f095ce1be287fb492b32c7e6e3bd09fc23526252da56dd234`;
it applies cleanly to the normalized file under `git apply --check`, and the
patched LF source SHA256 is
`54a5eae4f96b7a39e1400881ce96824f40b8e4ac026dc819b26e519db3e02ef9`.

The normal distributed `Vq_cut` remains the EXX/GW input. A temporary full
Coulomb read is moved into `hartree_full_vq_cut`; all global reader state is
restored on both success and exception paths. The Hartree-only Fourier
transform consumes the dedicated map.

The versioned tools in this directory are:

- `build_fish_legacy_hartree_corrected_v1.sh`: builds the exact normalized
  legacy source with the isolated-reader patch and binds the resulting binary
  to immutable provenance;
- `compare_legacy_hartree_null_delta_v1.py`: rejects any non-Hartree trajectory
  change when the first Hartree response is mathematically zero;
- `compare_qsgw_legacy_hartree_v4_current_v6.py`: adapts the legacy v4 and
  current v6 trace contracts without changing numerical rows;
- `run_fish_gate_c_legacy_corrected_v1.sh`: runs legacy/current controls and
  two-round same-dataset parity with truncated Coulomb and
  `legacy_extra_inverse_nk` normalization.

Their local static/synthetic suite passes 24/24 tests. This is not numerical
oracle acceptance. The pinned ABACUS full-BZ/no-symmetry bundle is still
missing, and the patch must still be built on fish and pass the same-dataset
two-update gate described in
`../../validation/hartree-legacy-oracle-audit-20260721.md`.
