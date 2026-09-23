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
the AO overlap. `crpa_parent_orbitals` selects `d`, `dp`, `eg` or `t2g` trials;
`crpa_output_orbitals` selects a subset of the resulting common frame. Thus
`d|dp` and `eg|dp` retain columns of the jointly normalized d+p frame. The
first radial shell is used; `eg` and `t2g` refer to the global Cartesian axes.
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
`crpa_overlap_file` is a native scalar S(R) CSR file, relative to `input_dir`.
Scalar wavefunctions, the full k/q mesh and LIBRI routing are required. A
compressed auxiliary basis additionally requires `use_shrink_abfs = true`,
`use_shrink_chi = true` and the corresponding native transformation inputs.

Outputs under `output_dir` are:

- `crpa_tensors.dat`: complete ordered onsite v/U/W tensors for every output
  spin pair and positive imaginary frequency, in eV.
- `crpa_summary.dat`: intraorbital, distinct-orbital direct and exchange
  averages, with explicit atom/spin indices. Exchange averages are not Slater J.
- `crpa_windows.dat`: the actual selected original KS indices for A and B.
- `crpa_metadata.txt`: orbital conventions, input selections and units.

No direct zero-frequency value or analytic continuation is added by this task.
The first minimax node must not be identified with U(0).

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
