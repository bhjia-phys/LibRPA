# QSGW Candidate Source-Lane Audit

Scope: frozen dirty candidate based on upstream `1376ee4f45a7611a55c5b92c4ba41409d515bcea`.

Status: classification complete; numerical acceptance remains pending the ordered gates in `QSGW_REBASE_PLAN.md`.

## Protected shared-code boundary

- `driver/tasks/g0w0.cpp` and `driver/tasks/g0w0_band.cpp` are byte-identical to upstream.
- `src/core/gw.cpp/.h`, `exx.cpp/.h`, `chi0.cpp/.h`, `epsilon.cpp/.h`, and `dielecmodel.cpp` are byte-identical to upstream.
- The only shared numerical-source hunk is the approved five-line read-only getter in `src/core/dielecmodel.h`.
- `get_head_matrices()` has exactly two source references: its declaration and the QSGW-only caller in `driver/tasks/qsgw.cpp`.
- The live getter patch and the approved packet patch have the same SHA256: `07d3d7a3b73c8860184d32ebb3895d2b3ae56fa03d6f6f19155547c00f84c296`.

The driver, input parser and CMake changes add QSGW state, validation, task dispatch and tests. QSGW-only keywords are parsed and validated only when the normalized task is `qsgw` or `qsgw_band`. Existing G0W0 numerical options and call sites are unchanged; this still requires the Gate-0 parser/CTest and direct G0W0 A/B observers.

## Formula and feature lanes

| Lane ID | Formula or behavior | Candidate symbols | Class | Reason | Required acceptance |
|---|---|---|---|---|---|
| `Q2-FIXED-BASIS` | Keep `mf0/C0/v0` immutable; build real-space objects from live `mf`; project EXX/Sigma with `C0`; diagonalize in the fixed basis; update live `C_n=U_n^T C0` in row-stored MeanField convention and `v_n=U_n^dagger v0 U_n`. | `ScopedReferenceEigenvectors`, `diagonalize_in_reference_basis` | U2 | Reproduces the old `get_eigenvectors0()` projection and `diagonalize_and_store` behavior without changing upstream GW/EXX. | fixed-basis unit tests, iter0/G0W0, old/new no-mixing matrices and WFC, orthogonality |
| `Q2-CORRELATION-MODE-B` | Diagonal elements use Hermitian `Sigma_c(epsilon_n-E_F)`; off-diagonal elements use Hermitian `Sigma_c(0)`. | `build_qsgw_correlation_potential` | U2 | Direct transcription of legacy Mode B using upstream `AnalyContPade`; default `n_params_anacon=-1` uses all points and requires no QSGW-only input. | unit oracle plus old/new `Sigma(iw)` and `Vc` per iteration |
| `Q2-EFFECTIVE-H` | `H_QSGW = H_KS(mf0) - Vxc_DFT(mf0) + Sigma_x(mf0) + Vc(mf0) + delta_VH(mf0)`. | `build_reference_hamiltonian`, `assemble_effective_hamiltonian` | U2 without `delta_VH`; U4 with `delta_VH` | The first four terms reproduce legacy `construct_H0_GW`; Hartree is a separately gated extension. The legacy upper triangle remains authoritative before Hermitian diagonalization. | unit formula oracle, iter0/G0W0, old/new raw Hamiltonian |
| `Q2-DISTRIBUTED-ADAPTER` | Collect the upstream distributed fixed-basis Sigma exactly once and broadcast only the mixed QSGW Hamiltonian maps. | `collect_blacs_matrix_root`, `collect_sigma_root`, `broadcast_spin_k_matrix_map` | U2 | Adapts to upstream kBLACS ownership without changing `G0W0` storage or reductions. | CTest at MPI 4 plus MPI 1/2/4 numerical identity |
| `Q2-HEADWING-SAME-GRID` | Rebind upstream dielectric head/wing to live `mf` and rotate velocity from immutable `v0`. | `refresh_headwing`, `diagonalize_in_reference_basis` | U2 | Required adapter for the upstream live MeanField/velocity contract. | head-only and head-wing old/new traces, MPI/OMP |
| `Q4-HEADWING-INDEPENDENT-GRID` | Fourier/interpolate a fixed-basis operator from the SCF grid to an independent full grid, then update that grid's live WFC and velocity. | `interpolate_fixed_basis_operator`, `update_independent_headwing_state` | U4 | This projection path is not a literal legacy call and needs independent round-trip and Hermiticity evidence. | projection unit tests, full-grid head-wing per-iteration matrices and residual bounds |
| `Q4-OCCUPATIONS` | Global zero-temperature charge-conserving fill with explicit k weights and degenerate-group handling. | `physical_electron_count`, `update_qsgw_occupations` | U4, conditionally equivalent for the Si full-BZ insulating oracle | It is more general and better specified than the legacy local filling routine, so equality must not be assumed outside the accepted Si scope. | charge conservation, iter0 equality, old/new Si per-iteration occupations, metallic behavior out of current claim |
| `Q4-LINEAR-MIXING` | `H_in^(n+1)=(1-beta)H_in^n+beta H_out^n`, default `beta=0.2`; use the same step and grid-derived decision for grid and band channels. | `HamiltonianMixer`, `SpinKHamiltonianMixer` | U4 | Explicitly requested convergence control; mixing-off must pass first. | no-mixing oracle, linear-beta-0.2 old/new trace, convergence curve, synchronized qsgw_band test |
| `Q4-HARTREE` | Build `delta rho=rho[live]-rho[mf0]`, contract RI Coulomb once, inverse Fourier to an AO periodic operator, and project to fixed grid/band bases. | `hartree_density.*`, `hartree_kernel.*`, `hartree_workflow.*` | U4 | No usable clean Hartree oracle exists yet; normalization and archived dirty LibRI provenance are unresolved numerical risks. | iteration-0 exact zero, charge conservation, direct contraction oracle, old/new Hartree off/on, grid/band projection |
| `Q2/Q4-BAND` | Reuse upstream real-space EXX/Sigma and `build_*_KS_band_blacs` BvK projection into immutable `mf0_band`; apply synchronized linear mixing when enabled. | `run_qsgw_stage_one(true)` | U2 for upstream real-space projection; U4 for synchronized mixing and Hartree | It does not Fourier-transform state-basis matrices between unrelated k points. The combined iterative behavior still requires a dedicated gate. | qsgw_band no-mixing then beta-0.2, raw/mixed operator and band eigenvalues |
| `Q-INFRA-CONTRACT` | Bind every required input file, producer, basis/gauge, unit and SHA256 before iteration. | `input_contract.*`, `vxc_io.*`, `projection_target.*`, `sha256.*`, `iteration_trace.*` | test/provenance infrastructure | No numerical formula is changed; failures are fail-closed. | FHI-aims and ABACUS input-contract regressions |

## Dormant or excluded code

- `src/qsgw/Hamiltonian.cpp/.h` and `fermi_energy_occupation.cpp/.h` are old upstream drafts and are not listed in `src/qsgw/CMakeLists.txt`. They remain untouched until all no-symmetry gates pass, as required by the cleanup policy.
- `MixingMode::Pulay` has implementation and unit coverage but is not accepted by the `librpa.in` parser and is not a required merge gate. No Pulay numerical claim is made in this candidate.
- Crystal-symmetry QSGW is rejected by input validation and is outside this goal.
- Optional two-stage analytic-continuation resampling exists in the QSGW-only helper but is not populated by the current driver; the accepted legacy lane is direct upstream Pade with normal LibRPA options.

## Static conclusions

1. QSGW is implemented as a new driver task and QSGW-only modules that call unchanged upstream GW/EXX/head-wing machinery.
2. No static evidence shows a change to G0W0 numerical logic. Runtime non-regression remains unproved until Gate 0.
3. U2 and U4 results must be attributed separately. In particular, a passing mixed/Hartree/band gap cannot substitute for the no-mixing fixed-basis matrix gates.
4. The clean candidate may retain the approved U3 getter, but no other shared numerical-source hunk is allowed.
