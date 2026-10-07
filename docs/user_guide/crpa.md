# Local interactions from constrained RPA

`task = crpa_u` computes bare, partially screened and fully screened onsite
interaction tensors on the native positive imaginary-frequency grid. The two
Green-function branches use the same selected original KS states to form
`Pd`; the screened interaction uses `Pr = P0 - Pd`. KS energies, occupations
and the frequency grid are unchanged.

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
task = crpa_u
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
`option_dielect_func = 3` uses analytic head and wings from the original KS
velocity matrices. Supply the existing native `velocity_matrix` input. Full
screening uses every physical transition; constrained screening omits a pair
only when both original endpoints belong to A. The RI optical charge vertex
sums equal spinor components. Only the angular Gamma-cell weights are
normalized; the regular dielectric body is unchanged. Symmetry, distributed
KS inputs and the two-dimensional optical route are currently excluded from
this cRPA task.

With `replace_w_head = false` and `option_dielect_func = 0`, an uncorrected
periodic result remains a finite-mesh diagnostic until the Gamma contribution
is bounded. Using a full-response optical head unchanged for nonzero Pd would
screen with the wrong response. Small-case optical tests do not establish
material mesh or basis convergence.

Library callers use `librpa_crpa.h`: initialize `LibrpaCrpaInput`, supply S(k)
and the window/species data, call `librpa_compute_crpa_window` collectively,
then inspect the owned result and release it with `librpa_delete_crpa_result`.
The API accepts no filenames and writes no files.

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
