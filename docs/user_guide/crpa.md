# Local interactions from constrained RPA

`task = crpa` computes bare, partially screened and fully screened onsite
interaction tensors on the native positive imaginary-frequency grid. The two
Green-function branches use the same selected original KS states to form
`Pd`; the screened interaction uses `Pr = P0 - Pd`. KS energies, occupations
and the frequency grid are unchanged.

## Orbital analysis before screening

By default, `crpa_output_analysis = true` evaluates Mulliken and symmetric
Lowdin AO weights from the loaded scalar or two-component spinor KS
coefficients and supplied S(k).
Here \(C\) contains the already exported KS eigenvectors in the atomic-orbital
basis, with \(C_{\mu n}\) the coefficient of AO \(\mu\) in band \(n\).
For scalar states each column \(c_n\) obeys \(Hc_n=\varepsilon_n Sc_n\).
For spinor states the overlap metric is \(I_{\rm spinor}\otimes S\), and
\(\sum_s c_{ns}^\dagger Sc_{ns}=1\); k and stored-channel indices are
suppressed here.
No Hamiltonian is read and no physical KS eigenproblem is solved. S alone is
diagonalized to evaluate its positive square root. The weight definitions are

$$
w^M_\mu=\sum_s\operatorname{Re}[c_{\mu s}^*(Sc_s)_\mu],\qquad
w^L_\mu=\frac{\sum_s |(S^{1/2}c_s)_\mu|^2}
{\sum_s c_s^\dagger Sc_s}.
$$

The existing `basis_wfc_out` shell metadata and explicit basis convention
identify atom/l groups. All radial functions are summed, with d split into
global-axis eg and t2g channels. No `pbands.xml`, `Orbital` or `PDOS.dat`
reference file is required. This AO character diagnostic is independent of
the first-radial projected local frame used for the interaction tensors.

The driver writes the following before screening:

- `crpa_orbital_groups.dat`: zero-based group/atom indices, species, l and channel.
- `crpa_orbital_weights.dat`: each spin/k/band/group weight in both conventions,
  absolute and relative energy, occupation, k weight and original S norm.
- `crpa_pdos.dat`: both Gaussian PDOS conventions on one energy grid, with
  separate stored spin channels, in states/(eV cell stored-spin-channel).
- `crpa_orbital_analysis.txt`: definitions, energy reference, Gaussian sigma,
  normalization and overlap-square-root diagnostics.

For a two-component input, weights sum both components of the same KS state
and are written in one stored spin channel (`spin = 0`). This index does not
identify a conserved spin projection. Scalar collinear inputs retain their
separate spin-up and spin-down channels. No spin-projected vector or spin
texture is computed by this orbital-character analysis.

PDOS uses the loaded normalized k weights, all loaded bands and no occupation
or additional spin-degeneracy factor. Mulliken weights can be negative; they
are not clipped. The default grid is -10 to 14 eV in steps of 0.02 eV, relative
to the highest occupied KS energy (`crpa_pdos_reference = vbm`), with Gaussian
standard deviation `crpa_pdos_sigma_ev = 0.15`. Set the reference to `fermi`
for a Fermi-level zero, and control the range with `crpa_pdos_min_ev`,
`crpa_pdos_max_ev` and `crpa_pdos_step_ev`. A finite grid covers only that
energy range; it is not a complete-spectrum normalization integral.
Energy conversion uses the configured LibRPA constants. A producer's printed
eV column can use a slightly different Hartree/eV constant; comparisons must
match this convention as well as poles, k weights and Gaussian sigma.

For a first pass that helps select A/B windows, use:

```text
task = crpa
input_preset = abacus
input_dir = ./input/
fn_stru = stru_out
fn_eigocc_scf = band_out
crpa_species_labels = Ni O
crpa_overlap_file = srs1_nao.csr
crpa_residual_tol = 1e-8
crpa_analysis_only = true
use_spinor_wfc = false
use_kpara_scf_eigvec = false
use_symmetry_rpa = false
```

This mode reads structure/k metadata, the split wave-function basis, energies,
coefficients and S; it skips auxiliary-basis, RI and Coulomb inputs entirely.
For SOC or a noncollinear spinor input, change `use_spinor_wfc` to `true`.
Either a spatial S(R) export or a validated interleaved SOC export can be
used. Optical velocity inputs are also skipped in analysis-only mode.
No correlated species or A/B selections are required. After inspecting the
analysis, supply the normal cRPA selections and set `crpa_analysis_only = false`
to run screening. Orbital character never selects windows automatically or
enters Pd as fractional weights. `crpa_output_analysis = false` skips the
additional analysis during an interaction run.

## Response selection and local frames

The response subspace **A** and output-orbital subspace **B** are independent.
Each can be selected by inclusive absolute-Hartree intervals (lower/upper pairs)
or by an explicit zero-based KS band list shared by all spin/k points. PDOS and
KS-band inspection determine the material-specific selection.

For each spin, atomic trials are projected into B and orthonormalized using
the AO overlap. `crpa_parent_orbitals` selects `s`, `p`, `sp`, `d`, `dp`, `eg` or `t2g` trials;
`crpa_output_orbitals` selects a subset of the resulting common frame. Thus
`d|dp` and `eg|dp` retain columns of the jointly normalized d+p frame. The
first radial shell is used; `eg` and `t2g` refer to the global Cartesian axes.
`sp` combines `s, pz, px, py` on each correlated atom in one output site, retaining
the cross-shell direct and exchange interactions in its four-index tensor. In
contrast, `dp` combines correlated-atom d orbitals with distinct ligand-atom p
orbitals. Selecting `s` or `p` from an `sp` parent retains columns of that same
jointly normalized frame, rather than constructing separate local frames.
An explicit ligand species prevents spectator atoms such as Li or Sr from
being interpreted as oxygen.

An example using the existing AFM NiO d|dp windows is:

```text
task = crpa
input_preset = abacus
parallel_routing = libri
use_symmetry_rpa = false
use_kpara_scf_eigvec = false
use_fullcoul_eps = true
use_fullcoul_wc = true
replace_w_head = false
option_dielect_func = 0
n_bands_chi0 = -1
tfgrids_type = minimax
nfreq = 24
crpa_overlap_file = srs1_nao.csr
crpa_species_labels = Ni O
crpa_correlated_species = Ni
crpa_ligand_species = O
crpa_parent_orbitals = dp
crpa_output_orbitals = d
crpa_response_windows_ha = 0.422538107206142 0.6066522113061734
crpa_orbital_windows_ha = 0.231441631892736 0.6066522113061734
crpa_residual_tol = 1e-8
```

If a group is isolated at each k but overlaps other bands in its global energy
range, specify `crpa_response_bands = 20 21 22` and/or
`crpa_orbital_bands = 20 21 22`. Omit the respective `_windows_ha` input when
using a band list. This selects the original KS indices directly; no orbital
character weighting or modified eigenvalues enter the response.

Species labels follow the numeric species order in the structure input.
`crpa_overlap_file` is a native S(R) CSR file, relative to `input_dir`.
Spatial exports and interleaved SOC exports are accepted; SOC exports must
have identical diagonal spin blocks and zero off-diagonal blocks.
Scalar or two-component spinor wavefunctions, the full k/q mesh and LIBRI routing are required. A
compressed auxiliary basis additionally requires `use_shrink_abfs = true`,
`use_shrink_chi = true` and the corresponding native transformation inputs.

Outputs under `output_dir` are:

- `crpa_tensors.dat`: complete ordered onsite v/U/W tensors for every output
  spin pair and positive imaginary frequency, in eV.
- `crpa_summary.dat`: intraorbital, distinct-orbital direct and exchange
  averages, with explicit atom/spin indices. Exchange averages are not Slater J.
- `crpa_windows.dat`: the actual selected original KS indices for A and B.
- `crpa_orbitals.dat`: atom, local column and atomic-trial label.
- `crpa_metadata.txt`: orbital conventions, input selections and units.

No direct zero-frequency value or analytic continuation is added by this task.
The first minimax node must not be identified with U(0).

For SOC, set `use_spinor_wfc = true`. Each spatial trial is lifted to both
spin components, projected into the original full-spinor B states, and jointly
normalized in the metric `I_spinor tensor S`. A p site therefore has six columns,
a d site ten, and an sp site eight. Within each site, up trials precede down
trials; these labels identify the trial orientation and are not conserved spin
quantum numbers. The response has one channel of occupied spinor states with
occupation at most one, without an additional factor of two. It retains all
four Green-function spin-block products.

SOC emits one complete spinor four-index tensor per site/frequency. Its
`spin_left = spin_right = 0` denotes that one channel; spinor information is
carried by `a,b,c,d`. The density vertex sums same-component products before
screening. The complete-spinor summary exchange average includes opposite
trial-spin columns and is not a scalar-shell Hund parameter. Inspect resolved
entries using the emitted orbital labels. Keep entire degenerate groups when
choosing A or B.

For a gapped three-dimensional input, `replace_w_head = true` with
`option_dielect_func = 3` uses analytic head and wings from the current
one-particle eigenpairs and their velocity matrices. The driver preset
`option_dielect_func = 5` selects the same full head/wing calculation and
computes velocity from native H(R), S(R) and r(R), using the existing
`g0w0_fn_hr`, `g0w0_fn_sr`, `g0w0_fn_rr` and unit parameters. With option 3,
select either that native source or the existing `velocity_matrix` file. Full
screening uses every physical transition; constrained screening omits a pair
only when both original endpoints belong to A. The RI optical charge vertex
sums equal spinor components. Only the angular Gamma-cell weights are
normalized; the regular dielectric body is unchanged. Optical sums retain all
original full-BZ KS states and velocities even when q-star screening is enabled.
Distributed KS inputs and the two-dimensional optical route remain excluded.

With `replace_w_head = false` and `option_dielect_func = 0`, an uncorrected
periodic result remains a finite-mesh diagnostic until the Gamma contribution
is bounded. Using a full-response optical head unchanged for nonzero Pd would
screen with the wrong response. Small-case optical tests do not establish
material mesh or basis convergence.

Library callers use `librpa_crpa.h`: initialize `LibrpaCrpaInput`, supply S(k)
and the window/species data, call `librpa_compute_crpa_window` collectively,
then inspect the owned result and release it with `librpa_delete_crpa_result`.
The API accepts no filenames and writes no files.

The separate `librpa_orbital.h` C interface computes an owned orbital-analysis
result through `librpa_analyze_orbitals`, exposes spatial AO/group weights and
state metadata including the number of spinor components, and evaluates
Gaussian PDOS in Hartree units. C++ callers can use `Handler::analyze_orbitals`,
which returns an owning result. Neither path modifies the handler, KS states,
response masks or local output frames. Orbital analysis uses the original full
k-grid states even when cRPA screening applies q-star symmetry reduction.

Every participating rank must supply identical numerical options, species,
selections and S(k) arrays. Input buffers remain owned by the caller and must
stay valid until the collective call returns. Each rank receives its own result
and releases it locally. The driver supplies this replicated input by reading
the same native source on every rank. KS orthogonality and frame checks validate
the supplied data; they do not reorthogonalize or repair the KS eigenvectors.

The scalar nspin=1 response retains the native spin-degeneracy convention.
For nspin=2, screening includes both physical spin channels, while output
vertices use their respective spin-resolved local frames. The reported left and
right spin indices identify those frames, not two independently screened U's.
Finite-temperature or metallic static-limit accuracy requires separate response
and convergence controls; selecting positive minimax nodes alone does not establish it.

With `use_symmetry_rpa = true`, the first symmetry route retains full original
k-grid Green functions and reduces only response q sampling and dense screening.
Every screened kernel is restored through the supplied unitary/antiunitary
operation table before contraction with the full-grid local spinor vertices.
The bare term also uses all original q points. Multiplying one representative
orbital tensor by star multiplicity would not preserve its individual components.
Selectors must keep complete degenerate groups and agree within each magnetic
k star. A supplied group must be a symmetry of the reference and constrained
model; grid reduction alone does not establish that property. The driver accepts
an explicit `spin_symmetry` structure tail, and library callers use
`librpa_set_symmetry_spin_operations`. SOC symmetry calculations reject a legacy
spatial-only declaration. A grey-group declaration appends time
reversal; it is appropriate for the nonmagnetic GaAs SOC control, and must not
be applied automatically to a magnetic state. This initial route does not
extend the native even-frequency response to general magnetic odd-frequency
components or validate k-star wavefunction reconstruction for cRPA.

## Compose or reuse numerical stages

The internal C++ stage declarations are in `src/core/crpa_workflow.h`, with
their implementations together in `src/core/crpa_workflow.cpp`. The public
standalone API prepares the model through these stages, and `compute_crpa_onsite`
is the composition that retains the existing complete v/U/W calculation.
The driver continues to own native-file reading and output writing.

| Stage | Result |
| --- | --- |
| `prepare_crpa_model` | Owning A/B selections and scalar/spinor local frames |
| `prepare_crpa_local_projection` | RI density vertices and retained bare interaction |
| `build_crpa_total_response` | Owning P0 on the current state and actual grid |
| `build_crpa_excluded_response` | Owning Pd with both endpoints selected by A |
| `form_crpa_constrained_response` | Owning Pr = P0 - Pd |
| `screen_crpa_response` | Native correlation-only Wc from an owning response |
| `contract_crpa_interaction` | Local bare + Wc tensors, without changing Wc |

For a caller that already holds P0, the combination
`evaluate_crpa_onsite_from_response(context, selection, frames, p0)` computes
only bare and U. It builds Pd and screens Pr; it does not build P0 or screen
the full response. Its returned `CrpaResult` has empty W vectors. The caller
can use the projection/contraction stages separately for an already computed
full Wc when a local fully screened comparison is wanted.

The borrowed P0, resident Chi0 response map and incoming band selector are
preserved, including when P0 aliases `context.chi0.get_chi0_q()`. Response
stages return owning maps; screening consumes its map by value/move. Temporary
Green-function workspaces may rebuild. The numerical stages do not update
the mean field or reinitialize the Dataset, frequency grid or descriptors.

Reuse requires the same current E/C/f, native frequency nodes, response q view,
active auxiliary space, MPI atom-pair layout and Coulomb/optical conventions.
Call before the GW backend releases its P0. The public collective
`librpa_build_g0w0_sigma_crpa` combines the ordinary GW build with U-only
constrained screening immediately after its P0 is available. The corresponding
C++ method is `Handler::build_g0w0_sigma_crpa`. It returns an owned cRPA result
while retaining the usual GW EXX and self-energy objects. The public
`librpa_compute_crpa_u_window` / `Handler::compute_crpa_u_window` builds a
current-state P0 and U without an additional full-W or self-energy calculation.
The standalone `librpa_compute_crpa_window` still returns complete v/U/W.

## Compute U during G0W0 or QSGW

For a single G0W0 calculation, enable the shared cRPA workflow with:

```text
task = g0w0
option_dielect_func = 5
crpa_enabled = true
crpa_model_file = crpa-model.in
```

The calculation reuses G0W0's P0 and the original PBE eigenpairs for U, while
retaining the usual GW screened interaction and quasiparticle calculation.
It writes the cRPA tensors, orbital weights and PDOS to
`output_dir/crpa/state_0000`. The metadata records the unchanged reference
state (`state_index = 0`, `gw_update = 0`) as final. The interval option
applies to QSGW iteration; G0W0 evaluates U once.

Add the following to a working QSGW input using the native full head/wing preset:

```text
task = qsgw
option_dielect_func = 5
crpa_enabled = true
crpa_model_file = crpa-model.in
```

The model file contains only the cRPA choices already used by a standalone
calculation: species labels, correlated/ligand species, overlap, parent/output
orbitals and the A/B selectors. An existing standalone cRPA input can itself
be the model file. Its shared GW settings are not imported. The model-file
path is relative to the main input file; native data paths are relative to the
main `input_dir`. Main-input cRPA assignments override the imported model;
selecting bands replaces an inherited energy-window selection and conversely.
Do not explicitly specify both selector forms for the same A or B.

`crpa_interval = 1` evaluates every entering electronic state. To sample every
second state, set `crpa_interval = 2`. The final state is always evaluated,
including when it does not fall on the interval. Ordinary QSGW supplies its
current E, C and occupations; each selected calculation rebuilds the model
frame and Pd from that state. The overlap, atomic trial definition and input
selection rule remain fixed. An energy-window rule may select different bands
as energies move; explicit band indices retain the stated indices.

Outputs are written below `output_dir/crpa/state_0000`, `state_0001`, etc.
State 0 is the initial reference; update 1 produces state 1. Metadata records
the electronic state, GW update and whether this is the final state. A selected
entering state borrows the GW P0 before full screening. After the last QSGW
update, a response-only call evaluates the final U with the final eigenpairs
and current-HR velocity. That call does not perform another self-energy update.
A checkpoint restart keeps the saved state index and the same interval rule.

These outputs contain bare and partially screened tensors and their direct/
exchange summaries. The U-only route omits W columns. Orbital weights and PDOS
are enabled by the existing `crpa_output_analysis` default and use the same
current eigenvectors; no extra physical KS solve is required.

The coupled path currently requires replicated full-BZ states, native minimax
frequencies and the LibRI response path. It rejects response symmetry reduction
and `read_sigc_mat_rf`, which bypasses P0. Use full head/wing option 3 (or driver
preset 5) for a corrected three-dimensional calculation. Head-only option 4
does not provide the constrained optical wings required by this implementation.
Driver preset 5 is normalized to option 3 before the library call.

The cRPA kernel uses the dataset's full Coulomb interaction for its own bare
term and constrained screening. GW retains its separately selected full/cut
Coulomb treatment. P0 reuse requires the same response, not an identical final
screened interaction. cRPA normalizes the angular Gamma-cell weights in its
onsite kernel; the GW optical convention remains the one selected by the
ordinary GW calculation. Consequently a cRPA fully screened local comparison
and GW W should be compared only with explicitly matched optical conventions.
