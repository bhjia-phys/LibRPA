# Experimental fixed-basis QSGW

Use `task = qsgw` for both molecular and periodic calculations. The driver
constructs self-energies in the immutable initial KS basis and updates the
eigenvalues and wavefunctions each iteration.
The effective Hamiltonian replaces the initial DFT exchange-correlation
potential with exchange and a Hermitian correlation potential. Hartree updates
are not implemented. Same-grid analytic head and wing updates are supported
with an explicit velocity convention; independent-grid updates are rejected.

QSGW reuses the G0W0 input readers and EXX/Sigma kernels. Task-specific input
and output live in `driver/qsgw`; numerical operations in `src/core/qsgw` accept
matrices and mean-field objects.

## Optional band-path channel

The driver selects the band channel from the input files. If
`fn_band_kpath_info` (default `band_kpath_info`) exists under `input_dir`, QSGW
runs the full existing band-path channel alongside the SCF grid. Supply the
matching band energies, wavefunctions, and **full** band Vxc matrices; the
metadata file does not substitute for these inputs. Missing or inconsistent
band data are errors. If the metadata file is absent, QSGW runs only the SCF
grid. A periodic calculation without a band path is valid, and the driver
does not infer a path from molecular or periodic geometry.

`qsgw` is the only QSGW task name. Older inputs containing
`task = qsgw_band` must change that value to `task = qsgw`; the former task has
been removed. Band-path selection still follows the input metadata. Option 5
uses the unchanged current-HR prescription described below.

The existing `qsgw_band0_unoccupied_keep`, `qsgw_band0_cut_mode`, and
`qsgw_band0_cut_shift_ha` keys retain their cut behavior and apply only when
band-path inputs are present.

## Vxc input

QSGW needs the full DFT Vxc matrix, including off-diagonal elements; the diagonal
Vxc tables used by G0W0 are insufficient. The driver reads the full matrices
directly from `input_dir`, using spin and k-point filenames with these defaults:

| Producer | SCF example (spin 1, k point 1) | Band example (spin 1, k point 1) |
| --- | --- | --- |
| FHI-aims | `xc_matr_spin_1_kpt_000001.csc` | `band_vxc_mat_spin_1_k_00001.csc` |
| ABACUS, one spin channel | `vxck1_nao.txt` | `band_vxck1_nao.txt` |
| ABACUS, two spin channels | `vxck1s1_nao.txt` | `band_vxck1s1_nao.txt` |

No additional file list or manifest is required. Optional `prefix_vxc_scf` and
`prefix_vxc_band` override the prefixes shown above (`xc_matr`/`band_vxc_mat`
for FHI-aims and `vxc`/`band_vxc` for ABACUS). Prefixes are relative to
`input_dir` unless absolute. For a separate ABACUS band output directory,
`prefix_vxc_band = OUT.band/vxc` reads its native `vxck1_nao.txt`, etc., without
renaming; the default `band_vxc` prefix distinguishes files staged alongside
SCF input. Spin and k indices are one-based and must follow
the same ordering as the corresponding SCF or band reference. Vxc files do not
provide an independent k-coordinate check.

Use `constants_choice = aims` for FHI-aims ELSI matrices in Hartree, and
`constants_choice = internal` for ABACUS complex text matrices in Ry, which are
converted to Hartree. The default `qsgw_vxc_basis = state` reads matrices in the
initial KS-state basis. This includes standard ABACUS `out_mat_xc` output,
despite its `_nao` filename suffix. Set `qsgw_vxc_basis = nao` only for genuine
ABACUS AO matrices; the driver then projects them with the reference
wavefunctions. This setting applies to both SCF and band matrices. Matrix
dimensions, finiteness and Hermiticity are checked on input.

## Analytic head and wing feedback

The standalone `qsgw` driver accepts `option_dielect_func = 5`
to recompute velocity internally from the complete current Hamiltonian and
then apply the existing analytic head/wing prescription at every iteration.
This preset selects `qsgw_headwing_velocity = current_hr` and resolves to
numerical option 3. An omitted selector or explicit `current_hr` is accepted;
explicit `reference` and `rotate_reference` are rejected for option 5. It
requires `replace_w_head = true`, `use_scalapack_gw_wc = true`, and
`use_pyatb = false`. The log records requested option 5 and effective option 3.
The native mode remains experimental.

Existing option 3 provides both analytic head and wing, and option 4 provides
head only, with `replace_w_head = true`. These options retain their explicit
`qsgw_headwing_velocity` selection:

| Value | Velocity used by each entering update |
| --- | --- |
| `reference` (default) | Legacy fixed initial matrix entries, option 4 only |
| `rotate_reference` | Fixed initial velocity operator rotated into the current eigenstates |
| `current_hr` | Velocity recomputed from the complete current AO Hamiltonian and current eigenpairs |

The last two choices clear the previous head/wing data and rebuild them from
the actual live energies, wavefunctions, occupations and frequency nodes before
each body/self-energy calculation. The analytic and body nodes are checked individually, including
when automatic minimax endpoints change. `rotate_reference` needs the original
same-grid velocity input. It does not differentiate the updated Hamiltonian.

For option 5, supply dimensionless native ABACUS `S(R)` and Bohr `r(R)` CSR
files, with their complete independent real-space supports, in addition to the
ordinary QSGW inputs:

```ini
task = qsgw
input_preset = abacus
replace_w_head = true
option_dielect_func = 5
use_scalapack_gw_wc = true
use_pyatb = false
qsgw_fn_sr = srs1_nao.csr
qsgw_fn_rr = rr.csr
use_kpara_scf_eigvec = false
use_symmetry_rpa = false
use_symmetry_gw = false
use_symmetry_exx = false
```

The previous equivalent selection remains supported with the same inputs:

```ini
option_dielect_func = 3
qsgw_headwing_velocity = current_hr
```

Paths are relative to `input_dir` unless absolute. This mode requires a full,
unshifted uniform mesh, replicated wavefunctions and a complete square AO/state
basis, with SOC and symmetry reduction disabled. S/r remain fixed; a separate
producer H(R) input is not needed. At each entering iteration, the driver
embeds the reference-basis Hamiltonian as `H_AO = S U0 h U0† S`, then Fourier
transforms the *whole* effective Hamiltonian and evaluates velocity in the
already selected current E/C frame. Even-grid Nyquist images have equal weights.
The differentiated operator is this finite-BvK model;
its initial velocity can differ from differentiation of the producer's longer
real-space PBE Hamiltonian. No independent PBE velocity file is used.

The implementation follows the executed PyATB H/S/r velocity convention, with
energies in Ha and Cartesian derivatives/position in Bohr. It checks the
reference and live S metrics, current eigenpair frame and Fourier round trip.
The AO basis and S/r operators remain fixed throughout QSGW.

The reusable internal C++ interface in `src/core/qsgw/headwing_velocity.h`
accepts localized matrices rather than filenames.
`solve_localized_velocity(H, operators, k_fractional, lattice_bohr)` performs
one generalized eigensolve and returns all `eigenvalues_ha`, AO-row/band-column
`ket_coefficients`, and three `velocity_ha_bohr` matrices in that same frame.
H contains full complex entries in Ha; S is dimensionless and r is in Bohr.
The lattice contains row vectors in Bohr, and fractional k uses
`exp(+i 2*pi*k.R)` while the derivative is Cartesian, in inverse Bohr.
S/dS and r use the documented upper-triangle completion, followed by
Hermitian upper-triangle completion of the band velocity. Native S/dS and
velocity diagonals are explicitly real, as in the existing live path;
PyATB retains complex diagonal roundoff. The consistency claim concerns
physically Hermitian models, with the residuals reported, and does not
extend to arbitrary invalid complex diagonal inputs.

For the iterative consumer, `compute_localized_velocity_in_frame` accepts
the already selected complete entering E/C and uses the same numerical core.
`compute_live_headwing_velocity` calls this form after building the current
finite-BvK H(R), preserving the live orbital frame used by the RI response.
Consequently each head/wing refresh needs no external PyATB process and no
second eigensolve. These are internal numerical interfaces; no public C or
Fortran binding is added. Scalar full-matrix controls cover the model solve
and the retained three entering Si frames; SOC, symmetry reduction and
fractional-occupation response remain outside this interface benchmark.

Fourier round-trip and eigenpair-frame checks establish agreement only on the
input mesh. Before using `current_hr` for physical band, velocity or topology
predictions, compare the initial interpolant with direct producer bands away
from that mesh and check mesh convergence. The 26-state Si 4x4x4 control passes
native-mesh closure but produces spurious off-mesh low-energy states: its
five-eV near-gap window differs from direct PBE bands by up to 116.74 eV.
That coarse interpolant is not an accepted physical model. Option 5 uses the
same finite-BvK interpolation and does not change this acceptance boundary or
automatically replace whole-H interpolation with an alternative construction.

With `qsgw_write_iteration_matrices = true`, traces include all three velocity
components, the head tensors and auxiliary-space wing arrays. Current-HR mode
also writes `qsgw_head_HR_update<N>_spin<S>.eV.csr` with full complex entries
and no sparsity threshold. This is the entering Hamiltonian differentiated for
update N (after N-1 completed updates), in **eV**; use the matching raw S/r.
On a fresh band/cut calculation, update 1 still screens the original PBE state;
the cut reference used for residuals does not replace those entering eigenpairs.

## Iteration and output

After reading and validating the SCF data, the driver selects the mixing route
from the global loaded SCF k-point count and full BvK mesh. The single-k route
requires both one SCF k point and a full mesh product of 1; it permits the
requested `qsgw_mixer = none` or `linear`. All other inputs use the periodic
route and force effective `none` with beta 1, even if `linear` was requested.
An IBZ containing one point from a larger full mesh therefore uses the
periodic route. The log records requested/effective mixer, SCF count and mesh.

The default mixer remains `none`; selecting a single-k input does not enable
mixing automatically. A single-k supercell mixing example is:

```ini
qsgw_mixer = linear
qsgw_mixing_beta = 0.2
```

For single-k `linear`, `H_next = H_in + beta * (H_out - H_in)` with
`beta = qsgw_mixing_beta` (default 0.2); beta 1 gives the direct-update result.
`none` accepts the new Hamiltonian directly. Grid and band path use the same
effective fraction; band-path presence and its point count do not select the
mixing route. This is an input convention for single-k supercells, not a
physical isolation test: a Gamma-only solid also enters the single-k route
and can select `none`. It does not change geometry, Coulomb, or head/wing.
The obsolete `qsgw_system_type` key must be removed from inputs; specifying it
is an input error. The band Hamiltonian cut is applied before and after
mixing, so excluded states obey the cut exactly.

`qsgw_iterations.dat` records the iteration number, largest eigenvalue change
(eV), Hamiltonian residual L2 and maximum norms (Ha), Fermi energy and gap (eV),
electron count, mixing fraction and convergence flag. Residuals are measured
before mixing on the SCF grid; the maximum is the largest complex-entry
magnitude. Stopping uses the largest eigenvalue change after `qsgw_min_iter`.
`qsgw_eigenvalues.dat` contains every grid eigenvalue and, when enabled, every
path eigenvalue at initialization and after each update. With band inputs,
each completed iteration also writes `KS_band_spin_S_N.dat`,
`EXX_band_spin_S_N.dat`, and `QSGW_band_spin_S_N.dat` under `output_dir`, where
S is the one-based spin index and N is the iteration number. These tables
provide the data for band plotting; the driver does not generate PNG figures.
Optional matrix diagnostics are enabled by
`qsgw_write_iteration_matrices`.

## Checkpoint continuation

QSGW saves the last completed update by default in
`output_dir/qsgw_checkpoint.bin` (normally `librpa.d/qsgw_checkpoint.bin`).
The optional `qsgw_checkpoint_file` selects another filename relative to
`output_dir`, or an absolute path; `none` disables saving. The driver writes
a temporary file and renames it only after the complete state has been written.
Interruption during a later update therefore leaves the previous completed
checkpoint available. Work inside an unfinished self-energy calculation is
not restored.

To continue, retain the original producer input and numerical settings, use
a fresh working/output directory, and set:

```ini
qsgw_restart_file = /path/to/initial/librpa.d/qsgw_checkpoint.bin
qsgw_max_iter = 3
```

The restart path is relative to the working directory unless absolute.
Restoration is explicit: a checkpoint already present in `output_dir` does
not start a restart unless selected with `qsgw_restart_file`. The default
`none` starts from the original input.

Iteration limits are global. Restoring completed iteration 1 with a maximum
of 3 computes updates 2 and 3. The maximum must exceed the saved iteration;
the minimum and stopping tolerance may also change. New traces begin at the
restored boundary, so a separate output directory preserves earlier traces.

The checkpoint retains the grid and optional band-path Hamiltonians in the
immutable initial KS basis, live eigenvalues, wavefunctions, occupation weights,
Fermi energy and the last iteration summary. Directly restoring wavefunctions
preserves their gauge at the boundary. Screening and exchange are rebuilt
for each new update; the original reference remains the basis for Vxc and
self-energy projection. Native RI and Coulomb files remain required, as do
the velocity input or S/r files selected by the velocity convention. Preserve
the band channel across a restart: a checkpoint created with band inputs
requires the same path metadata, band E/C, and full band Vxc inputs; a
grid-only checkpoint cannot acquire a band channel by adding those files.

The reader checks dimensions, reference eigenpairs, Vxc, k coordinates/weights,
cut policy, effective fixed mixing fraction, frequency count and head mode.
The automatically selected route uses the existing effective-beta checkpoint
check without changing the checkpoint format. Older mixed single-k molecular
calculations can continue with the same mixer and beta, without a system-type
setting. Periodic-route checkpoints with effective beta below 1 cannot
continue under the new policy because the existing beta check rejects the
change; effective-beta-1 checkpoints remain compatible.
Other producer files and numerical controls must also remain unchanged; the
checkpoint does not validate every input. MPI size can change because these
tasks require replicated wavefunctions. This experimental binary format
rejects incompatible byte order and accepts legacy fixed-mode versions 1--3.
Live velocity maps write version 5, recording their convention and actual
immutable velocity or S/r operators (plus current-HR geometry). A restart
cannot switch maps or replace those numerical operators. Version 4 belongs to
a separate equation/mixer branch and is not interpreted here.
This implementation supports `qsgw_mixer = none` and `linear`; adaptive
automatic-mixing history requires the separate mixer-aware extension.

## Numerical checks

The small regression suite covers two updates of molecular H2O from FHI-aims
with linear mixing, Si k333 with analytic head correction, and
FHI-aims H2O through `qsgw` with a band path, mixing and Hamiltonian truncation.
An ABACUS H2O case checks one full update through the native complex Vxc input
and Ry conversion against results from before the input simplification.
The band case uses the default 16 minimax
frequencies and the same Gamma reference on grid and path. Tests compare actual
iteration energies, Hamiltonian residuals, electron count and eigenvalue
trajectories with the existing numeric table comparator used by G0W0.

`regression_tests/verify_qsgw_restart.py` compares uninterrupted trajectories
with independently started continuation processes for Si without mixing and
molecular H2O with a band path and Hamiltonian cut. Earlier controls also
covered Si with linear mixing; those results remain historical, and that case
is excluded from the active verifier under the single-k mixing policy.
The verifier checks all eigenvalues,
summary fields, reference-basis operators and exact restored WFC/occupations;
later eigenvector phases and degenerate rotations are excluded. Use a fresh
`--work-dir`, the built `--binary`, and explicit `--ranks` in the matching MPI
runtime. These controls test numerical continuation, not material convergence.

A manual Si k666 band case is under
`regression_tests/manual/Si_k666_qsgw_band`. These cases check numerical workflows;
they do not establish convergence of material predictions.

`regression_tests/verify_qsgw_headwing.py` compares three uninterrupted updates
with a one-plus-two restart, including saved velocities, head tensors, wings,
self-energies, effective Hamiltonians and restored boundary states. It also
checks the actual analytic/body frequency nodes. Select `--case aims-rotate`
for the bundled initial-operator control or `--case native-current --dataset`
for a complete full-mesh native ABACUS dataset; the manual Si k444 input gives
the latter's operator and velocity convention. These are feedback and restart
controls, separate from independent velocity and off-mesh band acceptance.
