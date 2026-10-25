# ABACUS input contracts for scalar, collinear and spinor GW

This note derives the contracts at the ABACUS-to-LibRPA boundary. It applies
to the source families distinguished in the
[producer audit](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/producer-audit.md),
not to an unspecified ABACUS version. Source agreement, a finite numerical
oracle and material convergence are separate claims. The first pass below
is a writer–reader source audit; numerical witnesses are identified in the
report as they are completed.

## 1. A nonorthogonal basis does not insert S into every contraction

Let spatial atom-centred functions have Bloch sums `phi_a(k)` and overlap

\[
 S_{ab}(k)=\langle\phi_a(k)|\phi_b(k)\rangle,\qquad
 |\psi_{nk}\rangle=\sum_{a,s}|\phi_a(k),s\rangle C_{as,n}(k).
\]

For SOC, S is tensored with the two-dimensional spin identity. The
coefficients satisfy

\[
 H^{AO} C=S C\epsilon,\qquad C^\dagger S C=I.
\]

An AO operator matrix `O_ab=<phi_a|O|phi_b>` is projected as

\[
 O^{KS}=C^\dagger O^{AO}C.
\]

There is no additional S in this equation. By contrast the density kernel
is a coefficient outer product,

\[
 P^{ss'}_{ab}(k)=\sum_n w_k f_{nk}
 C_{as,n}(k)C^*_{bs',n}(k),\qquad
 N_e=\sum_k\operatorname{Tr}[S(k)P(k)].
\]

This distinction is crucial for EXX and reconstruction: P is not an AO
operator matrix and should not be transported with the same factors as O.
For a complete S-normalized frame, `O_AO=S C O_KS C^dagger S`. For a
truncated set of states this only specifies the operator in its represented
subspace. A dual-frame or residual completion must state the complementary
operator explicitly; replacing it by an identity is not justified.

The independent metric-aware density/EXX control is retained in
[the signed-time/self-energy note](soc-self-energy-signed-time.md). It tests
this distinction with a nonidentity S, rather than relying on an
orthogonal two-level special case.

### Physical length and reciprocal units are a separate invariant

ABACUS stores a dimensionless lattice L, dual G_int and Cartesian tau in
units of its actual lattice constant a0 (Bohr). The LibRPA wire contract is

\[
 A=a_0 L,\qquad G_{wire}=\frac{2\pi}{a_0}G_{int},\qquad
 r=a_0\tau,\qquad k_{wire}=\frac{2\pi}{a_0}k_{int}.
\]

Native structure parsing has already made tau Cartesian for both Direct
and Cartesian input. LibRPA divides external G and k by2pi internally,
so its map keys are Cartesian cycles/Bohr, while Bloch phases use
`2pi k_fractional dot R`. These are distinct types.

The active f20 writer instead hard-coded the scale corresponding to one
Angstrom and omitted a0 for Cartesian positions. An actual8-Bohr cell was
exported as15.117815 Bohr. A*G_wire^T=2pi I still passed, so reciprocal
self-consistency and a Gamma EXX match could not certify physical geometry.
The [native equivalent-input repair](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/producer-controls/producer-geometry-units/README.md)
uses a0 and2pi/a0 in both structure and BZ writers, with unconditional
Cartesian tau*a0. Four before groups fail absolute geometry; four corrected
Cartesian/Direct, a0=1/2 groups pass, with native H/S/density/spectra unchanged
in their common fixed-charge NSCF state. Other coordinate modes have source
convention support, not new runtime coverage. The older candidate already
had the structure helper but retained the separate BZ scale defect.

This is relevant to physical volume, Gamma integration and atom geometry;
integer lattice phases can cancel a shared scale error without fixing those
observables. No historical Bi2Se3 numerical effect is assigned without its
exact producer/input provenance. Supplementary raw auxiliary-overlap q
metadata is outside these two repaired writer contracts.

## 2. Spin degeneracy, k weights and the stored occupation

Write `w_k` for geometric BZ weights with sum one, and f for a single
spin-resolved level occupation. ABACUS `wg` already contains its k weight
and implicit degeneracy. With stored spatial k count N_k, the exporter
writes

\[
 b_{nk}=N_k\,\mathrm{wg}_{nk},\qquad
 \epsilon^{file}_{nk}=\epsilon^{ABACUS}_{nk}/2,
 \qquad E_F^{file}=E_F^{ABACUS}/2.
\]

The energies are converted from Ry to Ha. LibRPA reads `b/N_k`, recovering
wg; it does not multiply by another geometric k weight at this stage.

| ABACUS nspin | LibRPA `(n_spins,n_spinor)` | ABACUS weight in each stored channel | Single-spin density factor |
| --- | --- | --- | --- |
| 1 | (1,1) | `2*w_k*f_nk` | 1/2 |
| 2 | (2,1) | `w_k*f_nks` | 1 |
| 4 | (1,2) | `w_k*f_nk` for a spinor state | 1 |

The factor in the final column is `.5*n_spins*n_spinor`. SOC states are
already distinct spinor eigenstates, so their occupations must not be
halved again. The scalar density used for total charge needs its implicit
factor two restored; exchange uses one spin channel. The definitions, not
only the numerical value at full occupation, matter on an irreducible mesh.

`bz_sampling_out` independently normalizes its geometric weights by their
sum. Some Coulomb/overlap records still use the historical expression
`wk*nspin/2`, which is wrong as geometric metadata for nspin=4. Current
Coulomb readers discard that record field and use the BZ data. This makes
the metadata mismatch real without establishing an error in the present
numerical contraction. Any correction must identify all consumers first.

The first header/occupation precision is version dependent. The clean
candidate producer sets its stream precision before writing them; the
archived f20+patch producer sets it only at the first eigenvalue. The active
f20 writer had the same defect and now sets its existing fixed fifteen-
decimal precision immediately after opening the stream. For an encoded
first weight b0 with error delta_b0, the reader's weight error is
delta_b0/Nk and its spin-block density error is
`gamma*(delta_b0/Nk)*c0*c0†`, with gamma=1/2 for scalar implicit spin and
gamma=1 for explicit collinear/SOC input. The
[actual writer/reader and nonorthogonal density control](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/producer-controls/occupation-precision/README.md)
improves from0/12 to12/12 at unchanged2e-14 gates. Energy, spin and weight
conventions are unchanged. A reader cannot infer the original unrounded
value from an old file; the finite ensemble control accepts no metallic GW.

### The reference density must survive a restart representation

Even exact energy/occupation packing does not prove that a subsequent NSCF
uses the original SCF operator. Native LCAO charge output has a full real
grid rho_j, while the binary restart retains only a finite PW set Lambda:

\[
 \rho_G=N^{-1}\sum_j e^{-iG\cdot r_j}\rho_j,\qquad
 (P_\Lambda\rho)_j=\sum_{G\in\Lambda}e^{iG\cdot r_j}\rho_G.
\]

`ESolver_FP::iter_finish` applies `real2recip` before binary output;
`Charge::init_rho` prefers that binary to cubes and applies `recip2real`.
The native cube exporter instead writes every value of the same rho_save.
Retaining G=0 preserves total electron number but does not make P_Lambda
the identity on all real-grid LCAO densities. In particular, nonlinear
Vxc evaluated on the reloaded density need not equal its original value.
The [native charge-representation control](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/producer-controls/charge-restart-fidelity/README.md)
confirms the actual binary coefficients independently and recovers the
original H to roundoff only on the high-precision cube route. It isolates
the earlier coarse SCF/NSCF discrepancy without choosing a new production
format, changing read priority or assigning a Bi2Se3 error. Full density
fidelity and changed-support memory safety are different questions.

The [native changed-support control](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/producer-controls/charge-restart-fidelity/pw-projection-support/README.md)
then establishes three actual reader defects. An FFT-box/stick index can
be owned locally while lying outside the target spherical PW set; its
initialized local index is -1 and must not be written. Excess source spin
channels must reject collectively before accessing undeclared destination
pointers. Finally, the native two-channel writer serializes separate up/down
densities, while four-channel input stores charge and spin magnetization:

\[
 (\rho_0,\rho_1,\rho_2,\rho_3)
 =(\rho_\uparrow+\rho_\downarrow,0,0,
   \rho_\uparrow-\rho_\downarrow).
\]

The old binary two-to-four branch instead retained rho_up as total charge
and copied rho_down to mz, unlike the correct native cube conversion.
The fixes retain the intended intersection/zero-fill of PW coefficients,
reject unsupported excess channels, and apply the sum/difference after
projection. Native coefficient, field-sample, charge/moment and independent
Hartree checks pass28/28 at the unchanged2e-13 gate; the safety-only
intermediate passes26/28 and exposes the independent physical conversion
error. No general spin-conversion policy or arbitrary-cell acceptance follows.

The precision defect also matters beyond fractional occupations. In the
actual fixed-occupation periodic H2O input, the rounded mu lies1.15234e-7 Ha
below its highest occupied state. Existing Green-function builders cap a
positive exponential argument at zero. On an occupied negative-time branch
this uses min(epsilon_i,mu), rather than epsilon_i, when mu falls below that
state; the empty positive-time branch analogously uses max(epsilon_a,mu).
Consequently an exact Lindhard reference based on the original energies
need not equal that clamped finite equation. A
[fresh ordinary native export](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/producer-controls/charge-restart-fidelity/mu-corrected-rpa/verification.json)
with the coherent precision/support-repaired producer changes only mu,
now equal to the occupied VBM: all H/S/E/f/C/Cs/V values are unchanged.
It supplies an unedited corrected input to the downstream numerical
control, without changing the Green-function formula or explaining away
the separate transform-conditioning/absolute-accuracy failures.

## 3. Coefficient packing and actual state basis

The internal ABACUS SOC row is `2*a+s`. The RPA v1 exporter separates it
into the following contiguous payload:

\[
 \mathrm{offset}(\sigma,s,n,a)=
 ((\sigma N_{spinor}+s)N_{band}+n)N_{AO}+a.
\]

Here sigma labels collinear channels; SOC uses one sigma and two s. The
v1 eigenvector reader uses exactly this offset. For a scalar collinear
calculation the extra spinor extent is one. A complex coefficient is
copied, not conjugated. Reversing a Bloch convention is a separate operation
and cannot be hidden in a storage conversion.

The legacy writer loops spatial AO, spinor, band and channel; the matching
reader option is `BasisSpinorBandSpin`. The native `write_wfc_nao` text
export and the v1 binary RPA export have different precision and structure.
The band-path converter also has its own `(band,AO,spinor)` payload. File
extension and similar dimensions do not make these formats interchangeable.

The public library setters accept a complete matrix on each owning k rank,
not a local BLACS tile. They have no AO/band offsets to describe a tile.
That contract and its numerical packing/guard-page witnesses are recorded
in [the self-energy note](soc-self-energy-signed-time.md); native
redistribution creates BLACS tiles after the complete per-k input is set.

### Ordinary density storage is a transpose, not the physical AO density

The ordinary native `cal_dm_psi` forms its first GEMM factor from conjugated
weighted coefficients. Its raw column-major storage is therefore

\[
 Q_{ab}=\sum_n f_n C_{an}^*C_{bn}=D_{ba},\qquad
 D=CfC^\dagger.
\]

The public DMK accessor intentionally transposes this raw view. The core
cal_DMR path instead repacks raw Q without exchanging its mathematical
indices. Its Pauli conversion must consequently use

\[
 (n,m_x,m_y,m_z)=
 (Q_{uu}+Q_{dd},Q_{ud}+Q_{du},-i(Q_{ud}-Q_{du}),Q_{uu}-Q_{dd}).
\]

For psi=(1,i)/sqrt2, raw Q_ud=+i/2 while physical D_ud=-i/2, and m_y=+1.
The previous converter applied the physical-D formula to Q and returned-1.
The [actual native caller discriminator](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/producer-controls/active-hybrid-caller/magnetic-soc-multik/density-storage-repair/caller-contract.md)
supports the two-template sign repair with six C-to-DMK-to-DMR cases.
Raw/global storage and the separate full-spinor RI/EXX density remain
unchanged. A hand-constructed physical-D helper test does not exercise
this caller contract. Actual cube/continuum spin signs now agree, with a
separate approximately1e-5 finite-grid integration difference; that is not
the exact caller's1e-12 gate or accepted magnetic SCF convergence.

## 4. What ABACUS Vxc actually contains

`source/source_io/module_hs/write_vxc.hpp::cVc` first forms V*C and then
C^dagger*(V*C). The matrix passed to `save_mat` is nbands-by-nbands in the
current **DFT state basis**, despite the `_nao.txt` suffix. Matrix entries
remain in Ry. The LibRPA reader divides by two and completes the Hermitian
triangle, yielding

\[
 (V_0)_{mn}=\langle\psi^0_m|V^{DFT}_{xc}|\psi^0_n\rangle
 \quad\text{in Ha}.
\]

For PBE this is the semilocal XC potential. The same producer routine can
include a hybrid EXX term and DFT+U when those are present in the reference
Hamiltonian. Therefore a universal label “PBE Vxc” would be wrong for all
inputs. The intended GW subtraction must remove precisely the static
terms already present in its chosen reference; a PBE proof does not accept
every hybrid/U starting point.

With reference-state representation held fixed,

\[
 H^{QS}=H_0-V_0+\Sigma_x+V_c^{stat}.
\]

For current eigenvectors U_t in that frame, each operator must be projected
with the same U_t when solving the current diagonal QP equation. The
static term already in H_t is `H_t-H_0+V_0`. This does not change the input
meaning of V_0; it changes its representation when the current states
rotate.

`vxc_out.dat` independently writes the diagonal in Ha and eV. Comparing
that first column with the imported matrix diagonal is a direct unit and
index check before any GW calculation. Off-diagonal/gauge correctness
still requires a matrix comparison, not only matching those diagonals.

The audit found a separate input limitation: the Gamma-only real solver
writes plain real values through `save_mat<double>`, whereas the previous
QSGW reader only recognized complex `(real,imag)` tokens. Its Gamma filename
fallback did not cure the payload mismatch. The staged repair accepts
plain real values only when the native `# gamma only 1` header declares
that format; other layouts retain their complex parsing and Ry-to-Ha
conversion. General-k non-SOC export is complex and was not covered by
this defect. The linked report retains the native-writer negative control
and finite Hamiltonian assembly comparison.

### Full local Pauli Hermiticity couples opposite spin channels

For real NAOs and real local Pauli fields, each spatial integral obeys
V_mu(ba,-R)=V_mu(ab,R)^T. Hence H_ud=Bx-iBy spatially transposes in a fixed
spin channel, while full Hermiticity is

\[
 H_{ss'}^{ba}(-R)=\bigl[H_{s's}^{ab}(R)\bigr]^\dagger.
\]

Conjugating the same ud/du channel reverses By only in the lower atom
block. The [native full-operator witness and repair](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/producer-controls/active-hybrid-caller/magnetic-soc-multik/README.md#1-fixed-spin-channel-requires-a-spatial-transpose)
fix that local Gint assembly. Raw H/Vxc non-Hermiticity0.01853/0.07558Ry
falls to roundoff, and the two-atom SU2/full-complex regressions pass after
failing on the preserved old native object. Reading only an upper triangle
and completing it can conceal the omitted-block defect; neither a smooth
spectrum nor that completion proves the original operator was Hermitian.
These are CPU magnetic operator controls, not converged magnetic hybrid
references or attribution of the nonmagnetic Bi2Se3 discrepancy.

A complete hybrid potential additionally needs its actual reference K,
not an unavailable map interpreted as zero in V_local+alpha*K. The
[minimized native HSE failure](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/producer-controls/active-hybrid-caller/magnetic-soc-multik/density-storage-repair/hse-postprocessing-crash-minimized/README.md)
reaches output after initial SCF failure with zero reference spin channels;
the R writer previously indexed channel0 and the earlier KS route could
leave an incomplete local-only file. A common collective guard now checks
the selected reference before both potential writers. Legitimate empty
local maps remain allowed; a missing channel vector is not a computed
zero exchange operator. Native missing-state probes reject cleanly and
normal converged HSE full-R/KS operators remain unchanged. This does not
establish magnetic SCF convergence or general sparse-support allocation.

The reference's spatial support must independently include every spin
channel, not just channel0:

\[
 \mathcal S_{\rm required}=\mathcal S_{\rm local}\cup
 \bigcup_c\operatorname{supp}K_c,
\]

with the same rank ownership and optional nearest-cell mapping used by
the actual addition. A computed-zero up channel may coexist with a
nonlocal down channel; a Hermitian SOC \(K(k)=-x(k)x(k)^\dagger\) may have
off-diagonal R=+/-1 while both diagonal channels only need R=0. The
[native support-union repair](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/producer-controls/active-hybrid-caller/magnetic-soc-multik/density-storage-repair/hybrid-vxc-support-union/README.md)
reproduces the two old shape-valid crashes and restores complete R/k
operators at np1/np4. Allocation precedes local Veff and exchange
contributions; it is not an operation that promises to preserve an already
populated numerical container. Valid converged HSE caller operators
remain unchanged, while general nearest-cell/NSCF/backend and magnetic
convergence acceptance remain outside these finite controls.

## 5. Charge RI, Coulomb and active auxiliary projection

For a local product expansion, schematically,

\[
 \phi_{Ii}(r)\phi_{Jj}(r-R)
 \simeq\sum_\mu C^{I\mu}_{Ii,Jj}(R)P_{I\mu}(r)
       +\sum_\nu C^{J\nu}_{Jj,Ii}(-R)P_{J\nu}(r-R).
\]

The stored tensor uses internal order `(mu,i,j)`. The legacy/v1 exporter
writes `(i,j,mu)`; the reader must perform that permutation. The charge
auxiliary functions and spatial coefficients are shared by spinor
components. SOC spin contractions belong in P, G and the bubble vertices.

The real-space Coulomb transform used by the inspected exporter is

\[
 v_{I\mu,J\nu}(q)=\sum_R e^{+2\pi i q\cdot R}
 v_{I\mu,J\nu}(R).
\]

For this static Hermitian operator, upper atom-pair storage is sufficient
with the correct conjugate-transposed completion. This does not permit
the same completion for a general magnetic positive-iw response; see
[general screening](nonreciprocal-screening.md). The charge and optical
head/wing inputs must use the same Bloch/auxiliary convention.

Let parent functions be P and active functions p. Least-squares projection
in their overlap metric gives

\[
 S_{ss} U=S_{sP},\qquad U=S_{ss}^{-1}S_{sP},
 \qquad U\in\mathbb C^{N_s\times N_P}.
\]

The producer computes this rectangular q-dependent map. Its response and
screened-interaction use is

\[
 \chi_s(q,z)=U(q)\chi_P(q,z)U(q)^\dagger,\qquad
 W_P(q,z)=U(q)^\dagger W_s(q,z)U(q).
\]

These equations preserve a general non-Hermitian frequency response as
well as Hermitian static matrices. Replacing daggers by transposes or
projecting only one side is wrong once U is complex. Symmetry restoration
of U must use its different left/right representations. Spinor symmetry
does not make U itself a spinor matrix.

Neither raw `out_coulomb_k` nor the v1 writer applies an explicit Ry-to-Ha
factor, so the writer alone cannot resolve the unit established earlier
in the RI integral generator. The subsequent [native unequal-atom interchange](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/producer-controls/unequal-atom-interchange/README.md)
supplies the missing direct generator/writer/reader/EXX comparison for an
unshrunk Gamma H2O+SOC input. Its AO partition is (4,1,1), auxiliary
partition (10,1,1), and all nine Cs and six upper Coulomb blocks are checked.
The independently formed charge vertex is B_ij=C_Iij+C_Jji, including
both same-center terms. Then

\[
 (ik|lj)=B_{ik}vB_{lj},\qquad
 (H_x)_{ij}^{st}=-\sum_{kl}D_{kl}^{st}(ik|lj),\qquad
 E_x=\tfrac12\operatorname{Tr}(D H_x),\quad D=CfC^\dagger.
\]

The actual exported Coulomb normalization gives this result in Ha. Native
ABACUS Hexx is divided by two from Ry; complete-matrix error is at most
1.516e-14 Ha and printed-energy error 4.441e-16 Ha. Actual LibRPA readers
and API EXX agree with the independent matrix within 4.441e-16 Ha at one
and four ranks. No additional S belongs in this exchange-energy trace;
Tr DS and C†SC independently check the nonorthogonal density convention.
This settles that finite parent-basis interchange, not nontrivial-q phases,
all producer versions, compressed U or material convergence.

The two measured MPI boundaries differ. ABACUS diagnostic files contain
complete atom blocks selected by its AO judge; unequal atoms need explicit
judge masks, not an assumption that each rank contains the full matrix or
that a numerical zero denotes absent ownership. LibRPA's pre-projection
`exx_IJR_cplx` instead contains additive positive-contraction fragments;
the physical operator is their negative sum, as in the native KS projector.
Cs input uses the driver's disjoint LibRI distribution. The failed
replicated-loading/sign and complete-replica fixtures are retained in the
same report; they are not evidence of production EXX defects.

### Periodic phases and actual rectangular U generation

The [periodic native control](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/producer-controls/periodic-shrink-interchange/README.md)
uses the corrected physical metadata, a3x1x1 full mesh and matched fixed-
charge SOC states with a nonidentity AO metric. Nonzero-cell Cs and EXX,
and complex Coulomb/U entries, are measured explicitly. In the native Bloch
convention the product of a bra at k with a ket at k' has coefficient

\[
 B_{ac}^{\mu}(k,k')=
 \delta_{\mu\in a}\sum_R e^{2\pi i k'R}C_{ac}^{a\mu}(R)
 +\delta_{\mu\in c}\sum_R e^{-2\pi i kR}C_{ca}^{c\mu}(R).
\]

The finite Fock equation is

\[
 (H_x)_{ab}^{st}(k)=-\sum_{k'}w_{k'}\sum_{cd\mu\nu}
 D_{cd}^{st}(k') B_{ac}^{\mu}(k,k')v_{\mu\nu}(k-k')
 [B_{bd}^{\nu}(k,k')]^*.
\]

All k/spin entries agree with actual reader/API EXX within9e-16 Ha in
both parent12 and active11 auxiliary dimensions, at one/four ranks.
The native producer diagnostic agrees within1.27e-12 Ha. Those bounds
compare the same finite RI approximation, not exact radial-fit accuracy.

To check U itself, an equal-threshold native arm exports the full parent
overlap S_P. Source PCA ordering identifies E as the retained basis selector,
and independent overlap and Coulomb submatrix checks establish that identity.
Then

\[
 S_s=E^\dagger S_P E,\qquad
 U_S=(E^\dagger S_P E)^{-1}E^\dagger S_P,\qquad U_S E=I.
\]

The actual complex11x12 export agrees with this independent solve within
1.111e-15; it is not merely compared with its loader. The product vertex
carries q_product=k'−k, so its projected version is U(k'−k)B_P. Since the
real auxiliary basis obeys U(−q)=U(q)*, the equivalent bilinear interaction
at q=k−k' is U(q)† v_s(q) U(q). Wrong transpose/conjugation or wrong q
is detectable in this control.

Native direct Cs fitting solves a local pair Coulomb-metric system,
schematically c_pair=V_pair^-1 a_pair. That is distinct from the global
L2 projection U_S. Even for a global fit in the same parent span, the
Coulomb projection would instead be

\[
 U_V=(E^\dagger V_P E)^{-1}E^\dagger V_P.
\]

The measured U_V and U_S differ by0.2076. Direct active fitting and projected
parent vertices differ by2.344e-3, and their Fock operators by7.113e-3 Ha
at deliberately coarse shrink threshold0.05. Native EXX uses directly fitted
active Cs. The response has two actual API routes: `use_shrink_chi=ON`
builds parent response and projects with U; when it is OFF, response uses
directly fitted active Cs. In both cases the correlation self-energy
contracts parent Cs with U-unfolded W. These are different finite
approximations whose convergence must be assessed; passing
reader, projection and covariance checks cannot make them equal. No fitting
metric, projection, threshold or default is silently replaced here, and this
finite example gives no Bi2Se3 or topological error estimate.

To state the distinction at the operator level, let a_P be an orbital
transition vertex and a_s^dir its independently refitted counterpart.
Exact representation equivalence requires a_s=U a_P, because

\[
 \chi_s=\sum_\ell F_\ell a_{s\ell}a_{s\ell}^\dagger
        =U\chi_P U^\dagger,
 \qquad
 a_P^\dagger U^\dagger W_s U a_P=a_s^\dagger W_s a_s.
\]

This identity includes the appropriate product q in a, as fixed above.
It does not hold merely because U is read correctly if a_s^dir differs
from U a_P. A response built from a_s^dir and a self-energy contracted
with U a_P is the actual mixed finite approximation, not an exact change
of representation of one retained vertex. It is not replaced by another
scheme in this audit. Lossy fitting convergence and the relation between
body and optical-wing vertices require separate controls.

The [actual-input body-GW conjunction](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/producer-controls/periodic-shrink-interchange/gw-conjunction/README.md)
traces the above choices through actual readers and the complete native
API. At24nodes with explicit1e-12 regularization, source-exact full-matrix
contractions and one/four-rank comparisons pass; at coarse0.05 the direct/
projected KS Sigma(iw) matrices still differ by5.307e-4 Ha. That is an
imaginary-frequency self-energy difference, not a quasiparticle gap error.
The unchanged16-node/unregularized routes fail accuracy. At the original
bare-transition fit interval, the separate absolute finite-pole Sigma gate
also fails after explicit stabilization. The
[signed-kernel decomposition](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/producer-controls/periodic-shrink-interchange/gw-conjunction/forward-kernel-accuracy/README.md)
isolates the missing electron-plus-boson decay coverage, and a subsequent
[complete explicit-window native pair](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/producer-controls/periodic-shrink-interchange/gw-conjunction/forward-kernel-accuracy/complete-native-expanded/README.md)
passes the unchanged absolute gate at one/four ranks, maximum1.499e-9 Ha.
That upper bound uses the finite pole oracle, not an adopted general
automatic prescription. These results keep numerical failures distinct
from projection and contraction correctness, and accept no head-wing
correction, continuation, magnetic material or physical topology.

The later [signed input-Gram bound](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/producer-controls/periodic-shrink-interchange/gw-conjunction/forward-kernel-accuracy/conservative-pole-bound/README.md)
removes the need for known screened poles when selecting this finite body
window: the PSD signed-RPA Schur bound gives2.376509Ha directly from the
actual vertices, weights, energies and dielectric v. The full native body
pair then has KS Sigma error at most9.708e-10Ha. The subsequent
[native distributed CPU prepass](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/producer-controls/periodic-shrink-interchange/gw-conjunction/forward-kernel-accuracy/conservative-pole-bound/native-gram-prepass/README.md)
now assembles both S_+/-=-chi0(q,0+/-) through the actual GF/LibRI paths and
screening Cs/U/v branch before quadrature. Full-array parent/direct/projected,
loaded-space, magnetic IBZ and k-BLACS controls pass, with explicit lifecycle
and collective recovery checks. Automatic grid selection, large-system
costs and optical/indefinite/metallic extensions remain unaccepted; no
default is changed.

### The optical vertices must follow the selected response representation

For a projected body, the augmented response obeys

\[
 \Pi_P=\begin{pmatrix}h&l_P\\r_P&B_P\end{pmatrix},\qquad
 \Pi_s=\operatorname{diag}(I_3,U)\Pi_P
       \operatorname{diag}(I_3,U)^\dagger.
\]

Thus h is unchanged, r_s=U r_P and l_s=l_P U-dagger. Since native left
storage is l-dagger, both stored auxiliary-by-three wings use the same
left multiplication by U(Gamma), with no extra conjugation or AO overlap.
This precedes the active-basis Coulomb eigenvector/Schur operation.

The [native optical-vertex discriminator](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/lossy-optical-vertex-consistency/README.md)
found that the prior API always selected Cs_shrink for wings, even when
`use_shrink_chi=ON` projected the parent body. Changing only the directly
refitted vertices left that body unchanged but changed corrected W. The
repair threads the explicit parent auxiliary basis through the existing
full/IBZ contractions, reduces parent-sized wing buffers and projects
both independent sides into active storage. It copies no MF/WFC state.
Direct-response mode retains its direct vertices, and a source-mode cache
switch now triggers recomputation. Same-mode in-place U/Cs/frequency
updates remain a separate invalidation boundary.

The tests include independent Lehmann sums, nonorthogonal SOC,3D/strict2D
Gamma corrections, complex neutral auxiliary gauges and16 periodic
full/direct/projected/delayed, symmetryOFF/magnetic-IBZ, np1/np4 routes.
The original refit-induced projected-W change becomes zero; periodic
matched KS Sigma error is at most1.442e-13 Ha. This does not establish
lossy convergence, arbitrary complex charge-head gauges, absolute optical
Sigma accuracy or a material topology.

### Periodic position matrices require affine completion

An absolute position matrix is not a translation-invariant localized
operator. With r_ab(R)=<a0|r|bR> in Bohr and real-space overlap S_ab(R),
translation of the integration coordinate gives

\[
 r_{ab}^\alpha(R)=r_{ba}^\alpha(-R)^*
                 +R_\alpha S_{ab}(R).
\]

For the native positive Fourier phase, differentiation with respect to
physical Cartesian k consequently gives

\[
 r^\alpha(k)^\dagger=r^\alpha(k)+i\partial_\alpha S(k),\qquad
 \bar r^\alpha=r^\alpha+{i\over2}\partial_\alpha S.
\]

The centered matrix bar-r is Hermitian; in real space it subtracts
R_alpha*S(R)/2. The [native velocity implementation](../../../src/core/headwing_velocity.cpp)
therefore reconstructs a discarded lower entry as the conjugate of
upper-r plus i*dS, rather than plain Hermitian completion. Its band formula
is

\[
 v^\alpha_{mn}=(C^\dagger\partial_\alpha HC)_{mn}
 -\epsilon_m(C^\dagger\partial_\alpha SC)_{mn}
 +i(\epsilon_m-\epsilon_n)(C^\dagger r^\alpha C)_{mn}.
\]

Substituting centered bar-r gives the equivalent symmetric-energy
coefficient -(epsilon_m+epsilon_n)*dS/2. Thus upper storage alone is not
evidence of a velocity error. A matched producer check must first retain
the full raw r/S blocks, test the affine relation with nonzero R*S and
non-Gamma dS, and compare full ordered velocity with the actual reader's
upper-plus-affine path. The final velocity's enforced Hermiticity cannot
substitute for that input/operator check. Matched actual-input acceptance
belongs to the producer output control, not to this source derivation.

Equal orbital labels do not guarantee equal finite radial representations
for r and S. A consistent numerical construction may use the same active
overlap integrator for coordinate-multiplied bras. Write the local vector
as \(\boldsymbol{\xi}=\mathbf r-\boldsymbol{\tau}_a\) and its length as
\(\xi=|\boldsymbol{\xi}|\). In a real NAO basis,

\[
 \xi_\alpha R_l(\xi)Z_{lm}(\hat\xi)
 =\xi R_l(\xi)\sum_{L=l\pm1,M}
 \eta_\alpha\sqrt{4\pi/3}\,
 G(l,1,L;m,m_\alpha,M)Z_{LM}(\hat\xi),
\]

with native Cartesian conventions (m_x,m_y,m_z)=(1,-1,0) and
(eta_x,eta_y,eta_z)=(-1,-1,1). The moment radial r*R_l is unnormalized;
normalizing it would change the operator. Its mixed overlap and tau_a*S
term must use the active radial transformer, grids and origin convention.
Stored implicit radial powers must be decoded explicitly before forming r*R.
Retain geometric pairs even if their S entry vanishes: an onsite s-p dipole
can remain nonzero. Finite interpolation/quadrature errors still require the
full raw affine and ordered-velocity checks above; a common engine alone
does not prove numerical acceptance.

## 6. Symmetry and ownership are part of the input contract

A unitary spatial operation transports a state coefficient with an AO
representation and, for SOC, its spin rotation. Antiunitary operations
also conjugate the source coefficient and reverse the momentum as defined
by the producer operation. Bloch sewing phases depend on the same lattice,
atom mapping and AO ordering used for C, S, RI and H. Matching eigenvalues
alone cannot test that consistency.

The discriminating test uses one converged producer reference and compares
full versus reduced exports at the matrix level. Independently converged
SCFs confound symmetry transport with changed density/magnetic state. A
consumer full/reduced pair on the same noncovariant input can instead
demonstrate where covariance is violated; it does not automatically name
the producer as its cause.

Likewise, an MPI sum is correct only for uniquely owned entries. The
LCAO input code resets ordinary kpar to one but retains `sys.kpar_lcao`.
The inspected CPU `parakSolve` redistributes each pool's solved
wavefunctions to the global BLACS descriptor before returning. The
subsequent exporter sums those uniquely distributed global tiles; the
presence of a world reduction therefore does not establish double-counted
pool replicas on this path. This source trace is not a new MPI numerical
acceptance of every solver/export combination.

Acceptance therefore proceeds by independent numerical identities: metric
normalization and charge; Vxc and energy units; native EXX through exported
Cs/v; complex rectangular projection; and full/reduced matrix covariance.
Each check needs a negative control or a dataset that distinguishes the
relevant ordering, phase, unit or spin-factor error. None substitutes for
band/auxiliary/mesh convergence or establishes a material topology.

## 7. A spinor density need not be real, even with time reversal

The RPA export wrapper in the inspected producer families instantiates a
real post-SCF density/EXX route. This is distinct from writing real spatial
RI coefficients: the latter can be correct while the former loses SOC
information. In the candidate, `ctrl_scf_lcao.cpp:726` selects
`RPA_LRI<TK,double>` and `RI_2D_Comm.hpp:56–84` converts each density
contribution to real. Spatial restoration is restricted to nspin<4.

An exact finite counterexample clarifies the lost information. In two
spatial orbitals and two spin components, let

\[
 P={1\over2}(I_4+\tau_y\otimes\sigma_z),\qquad
 \Theta=I_2\otimes(i\sigma_y)K.
\]

Then `P=P^dagger=P^2`, Tr(P)=2, and
`Theta P Theta^-1=P`: complex conjugation reverses tau_y and the spin
rotation reverses sigma_z. P is a valid occupied Kramers-pair projector.
Taking its entrywise real part gives I4/2, which is no longer a projector:

\[
 (\operatorname{Re}P)^2-\operatorname{Re}P=-I_4/4.
\]

The discarded matrix entries have magnitude 1/2. In an independently
chosen interaction channel with exchange proportional to P, those entries
also disappear from exchange. This first argument is analytic; the native finite discriminator below
now executes the density transform and a controlled EXX contraction. It rules out the claim
that a nonmagnetic SOC real-space density can always be converted to real.

Likewise, at a time-reversal-related pair the correct coefficient-kernel
sewing is

\[
 P(-k)=J P(k)^* J^\dagger,\qquad J=I_{AO}\otimes i\sigma_y,
\]

including spatial sewing when required by the Bloch convention. The
candidate helper's bare conjugate average inferred from a weight of two
does not implement this relation. For a larger star, the full spatial and
spin transformations are also required. A weight alone does not identify
which operation relates the states.

This producer restriction directly affects its diagnostic post-SCF EXX.
It does not by itself alter the original complex C exported from psi,
E/occupations exported from the converged state, or the density-independent
spatial RI integrals. Those input paths and the post-SCF EXX must be
validated separately. The already repaired LibRPA four-block EXX route
must not use this real producer diagnostic as its universal SOC oracle.

### 7.1 Native finite discriminator and exact exchange normalization

The [retained controls](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/producer-controls/native-soc-density-README.md)
now compile the unchanged density/Fourier templates from both producer
families and the actual LibRI exchange contraction. They do not execute an
ABACUS SCF or regenerate material integrals. All six controls in each family
use two spatial orbitals, interleaved spins and two occupied states. For
orthonormal spatial orbitals the explicit projector already supplies a
valid coefficient kernel; no overlap ambiguity enters this witness.

For a local positive charge channel choose a spatial vertex Q=I2 and
Coulomb scalar g=0.8 Ha. Its independent exchange equation is

\[
 (\Sigma_x)_{as,bt}=-g\sum_{cd}Q_{ac}D_{cs,dt}(0)Q_{db}
                   =-gD_{as,bt}(0),\qquad
 E_x=-{g\over2}\operatorname{Tr}D(0)^2.
\]

LibRI's local two-end RI representation has C=Q/2 at each endpoint, so its
four products reconstruct Q on both charge vertices. The native contraction
returns the positive gD; adding the physical exchange sign gives the
expression above. The ABACUS wrapper convention is consistent:
`post_process_Hexx` multiplies by -2 to obtain Ry, while its SOC energy
postprocessing multiplies the raw contraction by -1 and its printed Ha
energy divides by 2. The fixture compares operators in Ha and evaluates
the independent energy formula; it does not claim to have executed that
complete wrapper.

For the complex Gamma Kramers projector, D=P and Re(D)=I4/2. Hence

\[
 \|\Delta D\|_{\max}=\tfrac12,\qquad
 \|\Delta\Sigma_x\|_{\max}=g/2=0.4\ {\rm Ha},
 \qquad E_x[\operatorname{Re}P]-E_x[P]=g/2=0.4\ {\rm Ha}.
\]

The sigma_z and sigma_x versions distinguish imaginary spin-diagonal and
spin-off-diagonal coherence. Both are lost by the real template; the actual
complex template preserves them exactly. The real Gamma projector is a
positive control with zero error on both paths.

A second independent control isolates the spin sewing error. On three
uniform points k=0,+1/3,-1/3 choose

\[
 P_0=\operatorname{diag}(1,1,0,0),\quad
 P_+=\operatorname{diag}(1,0,1,0),\quad
 P_-=J P_+^*J^\dagger=\operatorname{diag}(0,1,0,1).
\]

Each is a rank-two projector and the full set obeys time reversal. The
correct onsite density and its incorrectly conjugated reduced counterpart
are

\[
 D_0={P_0+P_++P_-\over3}
     =\operatorname{diag}(2/3,2/3,1/3,1/3),
\]
\[
 \widetilde D_0={P_0+2P_+\over3}
     =\operatorname{diag}(1,1/3,2/3,0).
\]

Thus even complex output leaves
`max|Delta D|=1/3`, `max|Delta Sigma_x|=g/3`, and the energy difference
`E_x[tilde D]-E_x[D]=-2g/9`. The observed absolute errors are
0.33333333333333337, 0.26666666666666672 Ha and
0.17777777777777776 Ha. Native LibRI contraction errors relative to the
independent -gD relation are zero for all controls. With the full three
points, the complex transform is exact; the real transform loses an
imaginary nonzero-R density of magnitude sqrt(3)/6. The local interaction
then has no onsite error, correctly distinguishing density failure from
this specific interaction's sensitivity.

The Gamma real projection can still obey time reversal: J is real, so
entrywise real projection commutes with this TR constraint. Its failure
is loss of required spinor coherence and exchange, not by itself proof
of Kramers splitting. By contrast the reduced-pair tilde D0 is not
TR invariant.

These numbers prove two separate defects in this native diagnostic route;
they are not estimates of a material's energy/topological error. The
metadata-backed existing spinor restoration is a reuse candidate, but
`kstars` are only populated for symmetry=1 in both inspected families.
TR-only symmetry=0 therefore needs explicit partner metadata before that
restoration can be reused. Parser and runtime source tracing shows explicit
magnetic SOC symmetry=0 can reach an MP reducer that assumes pure time
reversal, whereas the default SOC setting is -1. Magnetic analysis is
itself skipped under flag0, so a flag set only by that analysis cannot be
its sole safety criterion. Explicit flag0 is documented as a pure-TR
assumption, so a magnetic input violating it is not by itself a flag0
implementation defect. Retained native callback/reducer executions now
confirm this bounded path, without executing a complete parser/SCF. The
separate automatic-fallback discriminator and isolated correction are given
below; their injected failure precondition must not be confused with an
executed lattice-analysis failure.

### 7.2 Automatic fallback cannot infer time reversal from failed spatial analysis

Explicit `symmetry=0` is documented to assume time reversal. That assumption
is physical input, not a theorem inferred by the reducer. In contrast an
automatic change from symmetry=1 to 0, after a reciprocal/real-lattice
compatibility check fails, adds a symmetry premise that the failed check
does not justify. This matters for magnetic spinors even when some combined
antiunitary operation Theta*g belongs to their magnetic group: pure Theta
need not belong to it.

A simple gapped periodic spinor Hamiltonian makes the integration condition
explicit. Choose, in arbitrary model energy units,

\[
 H(k)=[B+\lambda\sin(2\pi k)]\sigma_z,
 \qquad B=1,\quad\lambda=0.3,\quad\Theta=i\sigma_yK.
\]

Its gap is at least `2(B-|lambda|)=1.4`, but

\[
 H(-k)-\Theta H(k)\Theta^{-1}=2B\sigma_z\ne0.
\]

The occupied energy is `e_-(k)=-B-lambda sin(2 pi k)`. On k=0,+1/3,-1/3,
with weights 1/3, the exact mean is -B. The native TR-only fold retains
Gamma with weight 1/3 and +1/3 with weight 2/3, so

\[
 \overline e_{\rm folded}=-B-{\lambda\sqrt3\over3},
 \qquad
 |\Delta\overline e|=0.1732050807568877\ldots.
\]

Native coordinate wrapping changes the final printed value to
0.1732050807568899. The independent even observable cos(2 pi k) remains
correct within 1.272e-14, providing a positive control for the TR-compatible
quadrature. The magnetic-model observable is evaluated diagnostically on
every mesh; it is not an energy assigned to the fixture's nonmagnetic or
scalar labels, or to a material calculation.

The [native control](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/producer-controls/k-reducer-README.md)
executes unchanged selected INPUT callbacks, the actual k-reducer translation
unit, and the byte-identical extracted automatic-fallback block. It supplies
`match=false` as a controlled precondition, not as a result of running lattice
analysis. Both old source families switch flag1 to 0 even when the supplied
magnetic-spinor flag is true. Disabling autoclose follows the native error
branch. Thus the automatic operation's physical defect is established under
this precondition; occurrence in any retained material is not established.

The isolated producer repair now retains the complete k mesh with runtime
symmetry=-1 for original nspin_in=4, skips the second reducer call, and clears
stale stars. It does not infer pure TR from the absence of a valid spatial
reduction. Original nspin_in is necessary because K_Vectors internally converts
spinor nspin=4 to one k-point spin channel. Passing integer -1 to the reducer's
bool use_symm argument would evaluate to true, so the call itself must be
skipped. Scalar/collinear fallback and explicit symmetry=0 retain their
previous meanings. The default autoclose value is unchanged.

Sixteen-case native before/after controls change exactly the two intended
SOC automatic-fallback cases; the other fourteen outcomes are identical.
The full-grid model error after repair is zero. This verifies the bounded
policy correction, not full MPI distribution, an ABACUS SCF, the separate
spinor EXX restoration, or material topology. The
[producer report](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/producer-audit.md#native-automatic-spinor-fallback-and-isolated-repair)
records exact sources, artifacts, checks and remaining boundaries.

The subsequent [ordinary-executable wrapper](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/producer-controls/autoclose-integration/README.md)
closes the actual-failed-lattice/MPI gap for one native spinor path. A weakly
strained primitive cell triggers real/reciprocal Bravais incompatibility;
four ranks retain all twelve k points and match explicit symmetry=-1 in
H/S, density, metric projectors and spectra at printed precision. Twelve
fixed gates pass, with independent generalized-eigensolve density error
1.278e-14 and eigenvalue error 7.461e-14 eV. Both arms reload the same frozen
charge into that cell with identical missing-G/renormalization warnings;
equal H/S checks the resulting common operator. Its negative indirect gap
excludes a gapped material/GW inference. It does not validate magnetic SCF,
general stars, repeated geometry changes or scalar/collinear MPI fallback.

## 8. A hybrid reference requires the original exchange fraction in V0

For a hybrid DFT starting point write its static contribution as

\[
 V_0=C_0^\dagger[V_{loc}(\alpha_0)+\alpha_0 K_0+V_{U,0}]C_0,\qquad
 H^{GW}=H_0-V_0+\Sigma_x+V_c.
\]

The exported V0 must be the state-basis operator actually present in H0.
Here K0 is the reference AO exchange operator with its outer fraction
factored out. If an unrelated post-SCF RPA export sets a shared alpha to
one and the later Vxc writer reads it, the general error is

\[
 \Delta V_0=C_0^\dagger[\Delta V_{loc}+(1-\alpha_0)K_0]C_0,\qquad
 H^{GW}_{wrong}-H^{GW}=-\Delta V_0.
\]

The candidate-r2 call order has precisely this source-level hazard:
`RPA_LRI.hpp:578` writes the global alpha during `after_scf`; Vxc is
written during `after_all_runners`; its EXX operator reads that same
global alpha. No alpha reset is present between these calls. Named PBE0
uses a separately cached local coefficient, giving Delta Vloc=0. Default
named HSE instead rebuilds LibXC with the mutated global alpha, so
Delta Vloc=-(1-alpha0) vx_PBE,SR and the total error contains
`K_SR-vx_PBE,SR`. External XC parameters can override the local dependence.
The [native caller/component account](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/producer-controls/hybrid-caller-audit.md)
derives these cases and checks unchanged native LibXC initialization and
evaluation. The subsequent [actual historical caller witness](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/producer-controls/hybrid-caller-integration/README.md)
now executes that original unmodified path through successful SCF and RPA
postprocessing. Matched PBE states give no Vxc change. Matched PBE0 states
give a full-matrix shift0.4907855262 Ha and agree with the predicted error
matrix to2.221e-16 Ha. A native HSE H2O pair converges at the same strict
threshold and reproduces both the fourfold nonlocal contribution and local
change, with total matrix shift0.3061114556 Ha. Earlier nonconverged Si
attempts remain failed; they are not relabeled as accepted material results.
The historical default selector coupling is handled by explicitly matching
reference prescriptions in each accepted pair. These finite operator
shifts are not general material gap corrections.

The archived f20+patch NiO source uses a local Exx_Info_RI copy and has no such
global-alpha assignment. The retained PBE Bi2Se3 input has no hybrid EXX
term in its Vxc export. Thus the candidate defect is a version- and
starting-point-specific finding; neither the modern producer nor the PBE
material should be assigned its hybrid error by association.

The [current f20 ordinary-caller positive control](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/producer-controls/active-hybrid-caller/README.md)
now verifies the repaired local-owned design through24 converged scalar
H2O calculations: PBE/PBE0/HSE, explicit limits/omitted selectors, np1/np4,
with only RPA off/on changed within each pair. All12 pairs preserve the
complete later Vxc within1.388e-15 Ha and the reference states to roundoff.
An independent four-index RI Fock energy agrees within1.333e-15 Ha.
At a fixed selector, all three functionals export identical Cs and cut/full
Coulomb payloads. The omitted private-RPA selector resolves to Massidda
without mutating the reference hybrid kernel; its full V differs from
explicit limits by0.0488941 Ha in this finite fixture. No SOC/magnetic
hybrid SCF, multi-k material or complete hybrid GW acceptance follows.

The separate [native nonmagnetic SOC three-k control](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/producer-controls/active-hybrid-caller/soc-multik/README.md)
now extends the actual caller to twelve converged SCFs/six matched pairs
with explicit limits and all four complex reference-exchange spin blocks.
Complete AO Vxc agrees within3.112e-15 Ha and the independent bare-RPA
exchange energy within1.777e-15 Ha. Its raw KS-matrix mismatch is retained:
the comparison must use Q=C_off-dagger S C_on and transport the complete
operator, while checking energy/occupation intertwining, rather than
comparing arbitrary spinor frames entrywise. Nonzero SOC and actual TR
sewing are tested. Magnetic or reduced-grid hybrid export, SOC defaults
and complete hybrid GW remain unaccepted.

## 9. Reference exchange and the physical GW Coulomb kernel are distinct

For an HSE reference, the static nonlocal exchange uses

\[
 v_{\rm SR}(r)=\frac{\operatorname{erfc}(\omega r)}{r},\qquad
 v_{\rm SR}(q)=\frac{4\pi}{q^2}
       \left[1-\exp\left(-\frac{q^2}{4\omega^2}\right)\right].
\]

The physical GW/RPA interaction must instead be the full bare Coulomb
kernel, or its explicitly declared geometric truncation. The reference's
alpha and short-range kernel belong in H0 and V0, not in a silently
substituted physical GW interaction. At q=0.2 bohr^-1 and omega=0.11 bohr^-1,
the short-range/bare multiplier is 0.56239836. Changing a singularity
prescription to massidda does not remove this inherited Erfc factor.
A screened RI fitting metric by itself is not wrong; the error here is
the actual exported physical V inheriting that reference kernel.

The [native component witness](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/producer-controls/hybrid-kernel-audit.md)
traces actual ReadInput/Convert and radial dispatch in both source
families. The [owned-settings repair](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/producer-controls/hybrid-kernel-repair-design.md)
copies the RPA settings by value, replaces its interaction map with unit
Fock and resolves the original singularity request through the existing
PBE/RPA default. Explicit geometric prescriptions and 2D settings survive;
reference alpha/maps remain unchanged. The original request is retained
before HSE/PBE0 reference default rewriting, broadcast and emitted in a
replayable RPA INPUT. A same-Parameter repeated-read defect was first
reproduced and then corrected by capturing only explicit read values.
This preserves existing omitted-field retention; it does not promise a
general reset on arbitrary functional changes in a reused object.

Twenty native one/four-rank cases pass, including private/reference default
separation, every explicit HSE prescription, 2D setting preservation,
parameter reuse and written-INPUT replay. Maximum radial multiplier error
is 1.388e-17 against the fixed 1e-13 gate. The independently found
[multi-component conversion defect](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/producer-controls/multikernel-audit.md)
is separate: whole-vector assignment inside a component loop discarded
earlier terms. Five indexed assignments preserve the requested sum, with
18 native LCAO/PW/LCAO-in-PW cases passing after a 0/3 baseline. Neither
set of component controls accepts an HSE material's complete SCF/RPA/Vxc
export or repairs the older candidate's reference-alpha mutation by itself.
