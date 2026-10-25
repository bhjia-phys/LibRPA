# General magnetic screening in the existing space-time GW route

This implementation restores the signed-time charge-response and GW equations in the
existing RI, minimax and dielectric-solve route. It does not change the selected
fixed-initial-basis QSGW staticization, Padé continuation, Coulomb truncation,
auxiliary projection or material input. The current coplanar magnetic controls
have a reciprocal charge response; adding `Bz=0.21 Ha` to the same gapped
three-site SOC model supplies a nonreciprocal counterexample. Symmetry reduction
being off does not remove this distinction.

The equations below are derived for the zero-temperature, gapped sector already
used by the space-time consumer. Metallic intraband terms and finite-temperature
periodic imaginary time are separate extensions. The numerical examples include
isolated `R=0` models and a periodic nine-point optical/magnetic control. They
establish finite implementation results, not material error estimates or
production acceptance. Sections 10–13 distinguish the native controls from
inverse-transform conditioning, the residual angular error and unsupported
consumers. Source references identify the audit branch based on `52b43609`.

## 1. Signed time, charge vertices and the missing term

Use Hermitian localized charge probes `rho_mu(R)` in a real AO/auxiliary gauge,
and the connected time-ordered convention

\[
 \chi_{\mu\nu}(R,\tau)=
 -\langle T_\tau\delta\rho_\mu(R,\tau)\delta\rho_\nu(0,0)\rangle.
\]

At q=0, an occupied m and empty n give

\[
 d_{\mu p}=\langle m|\rho_\mu|n\rangle,\quad
 A_p=d_pd_p^\dagger,\quad \Delta_p=\epsilon_n-\epsilon_m>0,
\]
\[
 \chi(+t)=-\sum_p A_p e^{-\Delta_p t},\qquad
 \chi(-t)=-\sum_p A_p^* e^{-\Delta_p t},\qquad t>0.
\]

The positive-time bubble is Hermitian as a matrix at fixed q, but this does not
make its **frequency transform** Hermitian at positive imaginary frequency.
With `F(iw)=integral d tau exp(i w tau) F(tau)`, the exact result is

\[
 \chi(i\omega)=-\sum_p\left[
 {A_p\over\Delta_p-i\omega}+{A_p^*\over\Delta_p+i\omega}\right]
 =\sum_p{-2\Delta_p\operatorname{Re}A_p
                 +2\omega\operatorname{Im}A_p\over\Delta_p^2+\omega^2}.
\]

The imaginary part in the last numerator denotes entrywise Im. Since A is
Hermitian, it is a real antisymmetric matrix. Thus the odd-frequency term is
real and antisymmetric at q=0. Forcing chi(iw) to be Hermitian removes it.
The native positive-time cosine expression instead keeps
`-2 Delta A/(Delta^2+w^2)`, both missing the required real antisymmetric term
and retaining an inappropriate imaginary antisymmetric term.

In the present real-space convention, time ordering and spectral Hermiticity
give, including transpose of the orbital dimensions in the exchanged block,

\[
 \chi_{\mu\nu}(R,-t)=\chi_{\nu\mu}(-R,+t)
                    =\chi_{\mu\nu}(R,+t)^*.
\]

Therefore a second expensive LibRI bubble contraction is unnecessary. The
corrected existing complex positive-time contraction supplies both signs.
The spin sum and the conjugation on the reversed **fermion** leg already
implemented in `chi0.cpp` remain unchanged. This bosonic signed-time identity
is a different operation; it must not undo that earlier SOC correction.

At fixed q the useful identities are

\[
 \chi(q,+t)^\dagger=\chi(q,+t),\quad
 \chi(q,-t)=\chi(-q,+t)^T,\quad
 \chi(q,i\omega)^\dagger=\chi(q,-i\omega).
\]

In a complex auxiliary gauge, the first and last statements retain their
matrix meaning, but bare conjugation between q and -q must include that
gauge's reality sewing. A q-dependent rectangular compression must transform
both signed-time or full-frequency operators as `U(q) chi U(q)^dagger`.

## 2. Exact minimax normalization and the available inverse sine transform

The current GreenX weights use the following test functions; there is no
additional factor of two or pi to insert into the matrix contractions:

\[
 C e^{-x t}\simeq {2x\over x^2+\omega^2},\qquad
 S e^{-x t}\simeq {2\omega\over x^2+\omega^2},
\]
\[
 \bar C{2x\over x^2+\omega^2}\simeq e^{-x t},\qquad
 \bar S{2\omega\over x^2+\omega^2}\simeq e^{-x t}.
\]

Here C/S are `get_costrans_t2f()` / `get_sintrans_t2f()`, with frequency rows,
and the barred matrices have time rows. For general matrix samples A=F(+t),
B=F(-t),

\[
 F_+=C{A+B\over2}+iS{A-B\over2},\qquad
 F_-=C{A+B\over2}-iS{A-B\over2}.
\]
\[
 F(\pm t)=\bar C{F_++F_-\over2}
            \mp i\bar S{F_+-F_-\over2}.
\]

For screening, use `F_-=F_+^dagger` **at fixed q**. This is Hermitian
analyticity, not an assumption of time reversal. Do not replace F_- by F_+.
For the real-space positive-time charge bubble the forward formula simplifies
to `C Re(A)-S Im(A)`, which is real entrywise. This yields another exact
reality relation in the original real auxiliary gauge:

\[
 \chi(-q,+i\omega)=\chi(q,+i\omega)^*.
\]

It does not require the physical Hamiltonian to be time-reversal invariant.
The same reality relation holds for W with consistently transformed Coulomb
matrices. Consequently an existing q/-q conjugation reconstruction may remain
valid as a **charge-reality** operation, even though an antiunitary crystal
operation at positive frequency has the different rule below.

The optional inverse sine weights already pass through
`src/core/minimax.h`, `minimax_gx.cpp`, and the Fortran wrapper. The trailing
arguments are `double *sinft_tw`, `double *sinft_duality_error`, and
`double *max_error_sin_wt`. Before this extension, `TFGrids::generate_minimax`
did not request them; allocation/getter existence did not mean valid weights.
The root-owned capability change requests them explicitly, checks finiteness
and tracks availability, leaving the legacy generation default unchanged.

The vendored GreenX diagnostic at
`thirdparty/greenX/GX-TimeFrequency/src/minimax_grids.F90` near 219 computes
`sinft_wt * cosft_tw` in its non-bare sine-duality branch. That reported scalar
is not `S*Sbar-I`. Recompute the latter independently if wanted. More
importantly, neither a finite coefficient matrix nor a matrix-duality statistic
establishes accurate inversion of the physical kernel. These transforms are
fitted to a restricted exponential family and can be ill-conditioned on
arbitrary sample vectors.

The read-only [`inverse-sine control`](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/self-energy-controls/inverse-sine-controls.cpp)
checks 301 logarithmically spaced x in each existing test interval. Requesting
the optional weights leaves the existing nodes, weights, C, Cbar and S
**bitwise unchanged** in all eight cases. Every optional matrix is finite.
The maximum absolute error in `Sbar[2w/(x^2+w^2)]-exp(-xt)` and maximum
absolute row sum of Sbar are:

| Grid and interval (Ha) | Existing regularization | Inverse sine kernel error | Maximum row sum |
| --- | ---: | ---: | ---: |
| 6, [0.657768,30.1366] | 0 | 5.6931e-2 | 2.84e1 |
| 16, [0.173388,23.7738] | 0 | 8.7631e-5 | 6.07e1 |
| 24, [0.3619945912,4.874507377] | 0 | 2.0513e-5 | 1.13e10 |
| 24, [0.4286910871,3.021313236] | 1e-12 | 4.7359e-7 | 2.03e4 |
| 32 H2O, [0.2634202450,22.01822598] | 0 | 9.0160e-7 | 5.19e7 |
| Same 32 H2O | 1e-12 | 9.2632e-7 | 1.17e4 |
| 16 magnetic, [0.4,4] | 0 | 7.1678e-8 | 5.14e7 |
| 24 magnetic, [0.4,4] | 0 | 4.8055e-5 | 1.31e11 |

The [full output](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/self-energy-controls/inverse-sine-controls.log)
also retains forward errors and both duality statistics. Exponential accuracy
and downstream Sigma error, not universal identity-matrix accuracy, are the
acceptance observables. Explicit use of the existing regularization in a
fixture is recorded; changing the production regularization default is a
separate numerical-method choice.

### Native transform capability implemented and checked

The audit branch now exposes optional inverse-sine generation through overloads
of `TFGrids::generate` and `generate_minimax`. The legacy signatures and class
member layout are retained; ordinary calls still leave inverse sine unavailable.
`has_inverse_sine_transform()` reports actual stored availability, and the
inverse-sine getter rejects use before generation. Generated weights are checked
for finiteness before installation. All frequency-only grid generators clear
the old time nodes, weights and transforms, preventing a stale inverse from
surviving a change of grid.

The native `test_timefreq` control uses 24 nodes and explicitly supplies the
existing regularization `1e-12`. For independent complex signed spectral kernels,
the maximum forward error is **1.08056e-8**, and the maximum inverse error is
**5.00698e-8**, below its declared `1e-6` gate. It also verifies unchanged legacy
nodes and forward/cosine-inverse weights, regenerates signed minimax after a
frequency-only Gauss-Legendre grid, and consumes the serialized inverse matrix
on the analytic odd kernel. The matrix writer now serializes vector elements
through `.data()`, rather than the vector object header, and records actual
inverse-sine availability.

The compiled time-frequency test passed in both LibRI and non-LibRI builds;
the non-LibRI targeted suite passed 4/4. The corresponding
[non-LibRI log](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/ctest-density-signed-input-nolib.log)
and [LibRI log](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/ctest-density-signed-input-libri.log)
preserve the executions. This is acceptance of the optional transform capability
and its lifecycle/export contract. Production chi, dielectric screening and
signed-time W consumption have not yet been changed by this stage; the interval
and conditioning limitations in the preceding table still apply.

## 3. Complete charge storage and dielectric algebra

The current active transforms `chi_libri_ct_accumulate_R` and
`chi_libri_ft_tw` in `chi0.cpp` use only C. Both must consume the even and odd
signed-time parts. The delayed real-space path and immediate compressed-q
path must represent the same complete operator.

At **positive time**, missing opposite atom blocks may still be reconstructed
from `chi_JI(R,+t)=chi_IJ(-R,+t)^dagger`. At positive **frequency**, that
same-frequency Hermitian completion is invalid. An implementation can avoid
doubling LibRI work by completing the time-domain pair once, or by accumulating
the even and odd Hermitian q-space components separately and combining them
into the full frequency matrix. It cannot preserve only one Hermitian triangle
of the combined chi(iw). An all-ordered-pair frequency map is the simpler
consumer contract; two explicitly distinguished Hermitian components are an
equivalent storage optimization, not a permission to discard the odd component.

The smallest next native contract can retain the existing `Chi0QMap` container
and expand each rank's owned upper atom pairs to include their reversed pairs.
Every ordered frequency block then has one owner; there is no new distributed
matrix representation. The existing LibRI contraction already produces full
`rpa.chi0s` before the requested-pair collection, except for its explicit
irreducible-sector output filter. Requesting the opposite blocks therefore
does not require another bubble contraction. The full frequency output requires
roughly twice the off-diagonal atom-block storage of one Hermitian triangle;
that storage holds independent information. The q/upper-pair work split and its
final redistribution must expand the same pair ownership consistently, including
ranks owning no physical blocks.

Restore the complex positive-time R-space response before applying C/S. In
particular, move the current no-shrink symmetry restoration ahead of the time
transform; its present placement after the cosine accumulation is not a general
odd-frequency restoration law. In the immediate-shrink path, constructing both
signed-time maps from P(R) and P(R)* allows the existing Hermitian signed-time
collection to be retained separately. Fourier-transform and compress both,
then combine them with the signed formula. Conjugating only the already
compressed P(q) at the same q does not construct its negative-time partner.
The delayed-shrink path instead needs general collection of the already
combined frequency matrix.

The native comparison must collect the full output without Hermitian
completion and assert one owner for each ordered element. The previous
`test_magnetic_gw_model::capture_response_operators` asserted `i<=j` and
conjugate-completed the result; the updated collector consumes every explicit
entry for the full-frequency contract while retaining its scalar behavior.
Its [serial/four-rank replay](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/ctest-magnetic-ordered-collector.log)
passes both tests with the original grid, regularization and numerical gates.
Compare all entries of
the Bz=0.21 response with the independent Lindhard sum, including its nonzero
odd part, alongside Bz=0, complete spin rotations, both shrink orderings and
one/four-rank cases. Standalone native chi can be tested before W is connected,
but a full-frequency map must never silently enter an old Hermitian screening
consumer. This is an intermediate integration boundary, not a proposal to
leave magnetic GW permanently disabled.

The native implementation slice now exposes
`Chi0::full_frequency_matrix()`, derived from `mf.get_n_spinor()>1`, without
adding class members. Its full-map contract applies to the complex spinor LibRI
space-time path. Other spinor routes are rejected instead of using their
real-only Green-function fallback. The scalar route is unchanged: supporting
orbital-current, nonreciprocal scalar Hamiltonians would require selecting a
complex Green-function contraction for `n_spinor==1`, which is a distinct
extension. This accessor does not infer whether the input is physically
time-reversal symmetric.

Implementation preparation includes both signed immediate-shrink maps, general
delayed-frequency shrink, pre-transform complex R-space restoration, reverse
pair ownership and memory accounting, and lower-block acceptance in cRPA
response subtraction. The immediate path uses the general ordered collector
for both signed maps as well, preserving explicit data and making its ownership
identical to the frequency path. Its additional time-map storage is released
per tau. The existing `Chi0::unfold_abfs_Wc_q` is intentionally separate: its
current GW caller supplies W(q,tau), where Hermitian collection is valid, so the
frequency accessor must not be applied indiscriminately there.

The new `test_chi0_signed_frequency` uses direct full-map ownership counts and
an independent occupied-empty Lindhard sum. It covers three one-orbital atoms
and one three-orbital atom, the latter testing the internal non-Hermitian part
of an atom-diagonal block and empty MPI owners. Controls include Bz=0/0.21,
complex common SU(2) rotations, a dispersive three-k model, q/atom-pair splitting,
both compression orderings, square and rectangular complex auxiliary maps,
and response subtraction. The first compiled serial and four-rank executions
passed all 15 cases each; the maximum Lindhard error was **1.2385e-11**. The
Bz=0.21 Gamma response error was **8.4204e-12**, compared with the preserved
pre-repair cosine error **5.63367e-5**. Its nonzero anti-Hermitian part,
7.38573e-5 in the uncompressed case, survived storage and collection. The
complex projected cases retained anti-Hermitian parts up to 1.28658e-4. The
[CTest summary](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/ctest-signed-chi0.log)
and [first raw execution log](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/self-energy-controls/native-chi0-signed-first-LastTest.log)
preserve these checks. They establish the native full-grid response calculation,
not the dielectric solve or the complete GW result.

After that execution, the output-zero allocation was moved to one frequency
at a time on the delayed path, avoiding simultaneous residency of its entire
full q-frequency output and R-frequency cache. The focused test has also been
extended with a genuine 3x3 hexagonal crystal: C3 gives five input stars, and
C3 plus the valid magnetic antiunitary Theta M_y gives four. For the radial
120-degree exchange texture with Bz=0.21, Theta M_y acts as site reflection
times bare complex conjugation on spinors and preserves the Hamiltonian.
An independent nine-k matrix check gave a maximum covariance residual
8.74e-16. The extended native cases subsequently passed in both serial and
four-rank builds, including the allocation adjustment. C3 full/IBZ inputs
used five response representatives, while C3 plus Theta M_y used four; the
real-space maps covered all 81 blocks, including **36 antiunitary sectors**.
The Hamiltonian covariance residual was 8.88178e-16, the full-versus-reduced
native chi difference was at most **9.888e-17**, and the crystal Lindhard error
was **1.05739e-11**. All first 15 numerical cases retained their previous
maximum error of 1.2385e-11. The
[extended native execution log](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/ctest-general-optical-libri-detail.log)
records the 20 cases per process configuration. This accepts the tested
unitary/antiunitary crystal response restoration, including genuinely reduced
input, in addition to the earlier full-grid signed-response controls.

Required collection changes include:

- `epsilon.cpp::build_dense_symmetry_hermitian_matrix_from_local_blocks` and
  ordinary `collect_block_from_ALL_IJ_Tensor*` currently conjugate-complete
  opposite blocks and sometimes Hermitize atom-diagonal blocks. Use full
  assembly for the general response, with ownership for every ordered block.
- Replace upper-pair request construction for this response with the existing
  full-pair descriptor helper. Do not change Hermitian Coulomb collection.
- `shrink_abfs_chi0` currently collects with the Hermitian flag. Immediate
  shrink at signed time may retain that property separately at each sign;
  delayed shrink of the full frequency response needs full collection.
  The existing GEMMs `U chi U^dagger` themselves are correct.
- The public/internal compressed W transfer to `G0W0::build_spacetime` must
  preserve both triangles at frequency. `complete_hermitian_Wc_q_blocks` is
  valid after a correctly signed time transform, not before it at positive iw.

The screening equations remain

\[
 E(q,z)=I-v_\epsilon^{1/2}\chi(q,z)v_\epsilon^{1/2},\qquad
 W_c(q,z)=v_W^{1/2}[E(q,z)^{-1}-I]v_W^{1/2}.
\]

The Coulomb square roots remain Hermitian and keep the existing filtered
subspaces. E and its inverse are general matrices. The ordinary distributed
body path in `epsilon.cpp` already calls `LaConnector::pgesv` when
`use_cholesky_gw_wc=false`; it solves `E X=sqrt(v_W)` and forms
`sqrt(v_W)(X-sqrt(v_W))`. No new inverse algorithm is needed there.
Cholesky is valid only for a verified Hermitian positive-definite E, and the
serial `power_hemat_onsite(E,-1)` must be replaced by a general solve on this
route. Selecting an existing general solve restores the stated linear system;
silently symmetrizing E would instead change the screening equation.

## 4. Unitary and antiunitary reconstruction: keep the frequency domain explicit

Let D be the actual auxiliary representation, including spatial, atomic and
Bloch-gauge factors. For unitary g, `q'=R q`,

\[
 F(q',z)=D F(q,z)D^\dagger.
\]

For antiunitary g, `q'=-R q`,

\[
 F(q',z)=D F(q,z^*)^*D^\dagger
        =D F(q,z)^T D^\dagger.
\]

The last equality uses Hermitian analyticity. At real signed imaginary time
the corresponding operation is `D F(q,tau)* D^dagger`; the i in the Fourier
sine factor must not be treated as a static matrix coefficient under this
antiunitary operation.

For the dense W helper, its antiunitary effective D is
`conj(transform_dense)`: the baseline `rotate_symmetry_blacs_wq` evaluated
`T* W* T^T`. At positive frequency the middle operation must be transpose,
giving `T* W^T T^T`. Its inverse-to-star-source counterpart similarly uses
`T^dagger W_source^T T`. Block helpers must transpose **and exchange the
source atom indices**; local transpose of a non-square I,J block alone is
not the same operation. Little-group projection of chi at positive frequency
needs the same dynamic rule.

The baseline delayed-CT chi path restored real-space symmetry after its cosine
transform. For a first correct general implementation, restore the complex
positive-time bubble before forming C/S parts at each tau. Then the existing
time-domain transformation law applies without a new parity protocol. A later
optimization may restore even/odd frequency parts separately, with opposite
antiunitary signs; feeding their sum to the old static conjugation helper is
incorrect.

Likewise, one may first invert W to signed time at representative q points,
where each W(q,+/-t) is Hermitian, and only then restore the q stars with the
time-domain helper before the q-to-R transform. This aligns with the present
compressed atom-pair route. If the dense path retains frequency-to-R ordering,
its q-star helper must instead implement the positive-frequency transpose law.
Frequency-output restoration must always identify which law it uses. The
charge-reality q/-q relation from section 2 is distinct and must not be globally
replaced by a time-reversal law.

## 5. W(+tau), W(-tau) and the Sigma consumer

At fixed q construct both W time samples from the Cbar/Sbar formula using
`W(q,-iw)=W(q,+iw)^dagger`. In the original full real auxiliary basis, after
q-to-R transformation,

\[
 W_c(R,-t)=W_c(R,+t)^*.
\]

The negative sign can therefore be represented by a conjugated view/copy of
the positive-time full-basis tensor. With a complex compressed gauge, unfold
with its correct q-dependent U before using a bare real-space conjugation.
No assumption that W(R,t) is real is allowed on the general magnetic route.

The active public [Sigma caller](../../../src/api/compute_g0w0.cpp) supplies
the parent auxiliary basis and real parent Cs to build_spacetime, even when
the response uses active Cs. The compressed route in
[gw.cpp](../../../src/core/gw.cpp) first inverse-transforms W in its active
coordinates, then calls unfold_abfs_Wc_q before q-to-R transformation and
LibRI cache installation. Its two PGEMMs in
[chi0.cpp](../../../src/core/chi0.cpp) implement U-dagger W_active U, and the
consumer's complex Cs tensors are conversions of the real parent tensors.
Thus an arbitrary complex active U does not itself put active-coordinate W
into this bare-conjugation cache. Exact-subspace controls test this contract;
they do not establish harmless lossy projection or compatibility of arbitrary
q-dependent input maps.

The subsequent absolute optical controls also exercise this actual
atom-pair compressed route: its active3 frequency data unfold into parent6
time data before FT_Wc_q2R and LibRI. Explicit U-dagger W_active U and
independent time unfolding agree within5.17e-16 and6.59e-15, respectively.
The constant commuting Householder map, however, makes a deliberately wrong
active-conjugation witness vacuous. That failed nonvacuity test is retained;
it cannot establish general complex-active cache safety. The
[interface and coverage report](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/optical-generalization/absolute-sigma/README.md#exact-compressed-periodic-public-sigma-route)
keeps this distinction from the successful parent-operator comparisons.

The baseline `gw.cpp` set LibRI W once outside the loop over `taus={t,-t}`,
then changed G inside that loop. It must install the W with
the **same signed time** before `cal_Sigmas`:

\[
 \Sigma_{st}(R,+t)=G^{\rm lib}_{st}(R,+t)W_c(R,+t),\qquad
 \Sigma_{st}(R,-t)=G^{\rm lib}_{st}(R,-t)W_c(R,-t).
\]

The existing `G_lib=-G_Matsubara` convention supplies the usual `-GW` sign.
The external spin indices and both signed-time G definitions remain unchanged.
Restoration of all four spin blocks is done for each real signed time, then
the repaired union-of-support C/S transform produces Sigma(iw).

A minimal correct implementation can install matching W inside the signed-time
loop. A more economical loop order is: positive t, all four spin channels and
their joint restore; negative t, all four channels and their joint restore;
then frequency accumulation. That sets each W only once per sign and avoids
repeating W communication for each spin channel. Frequency accumulation can
also be linear per sign, with `(C+iS)/2` and `(C-iS)/2`, preserving absent-key
zero semantics. These are equivalent contraction orders, not changed physics.

### Native signed inverse and the real-space adjoint

The dense native implementation keeps the existing q-to-R ordering. Hermitian
analyticity gives the exact Fourier identity

\[
 F(R,-i\omega)=F(-R,+i\omega)^\dagger,
\]

and therefore

\[
 F(R,+t)=\frac12\sum_j\left[
 (\bar C_{tj}-i\bar S_{tj})F(R,+i\omega_j)
 +(\bar C_{tj}+i\bar S_{tj})F(-R,+i\omega_j)^\dagger\right].
\]

Replacing this adjoint by a transpose additionally assumes that F(R,+iw) is
entrywise real. The implementation retains the adjoint: no reality projection
is performed and a complex auxiliary-gauge test distinguishes the two rules.
A distributed conjugate transpose communicates the opposite R block, including
ranks with empty tiles. R/-R pairs are released together after both inverses
have been computed. Only one transposed frequency tile is temporarily stored;
this initial implementation has not been performance-tuned.

The compressed path instead co-owns the two independent ordered frequency
blocks (I,J) and (J,I), forms F(q,-iw) locally by their adjoint, and produces
Hermitian positive-time matrices. Only at that time boundary may upper-pair
storage, existing auxiliary unfolding and static q-star restoration be reused.
The complex-gauge relation is handled before unfolding; bare conjugation in R
is used only after returning to the full real auxiliary basis. Compressed
frequency export separately retains both ordered blocks, unfolds them without
Hermitian completion and applies dynamic antiunitary transport. Section 12
derives and checks this path; section 13 validates the actual serialized caller
export against an independent Fourier sum. Its former blanket guard is removed.

For Sigma the implementation prepares two named LibRI W caches once per time
node, selects the cache with the same sign as G, and frees both after all spin
channels. This avoids recommunicating W for each spinor channel but retains an
additional signed W cache; no memory or speed improvement is claimed here.
Dataset initialization requests inverse sine only for the spinor minimax route,
without changing the existing regularization. A manually supplied time grid
must expose that capability explicitly rather than silently treating its
missing sine inverse as zero.

These changes are part of the coordinated native implementation under test.
The optical, RPA/cRPA and compressed-frequency-output consumers retain explicit
staging guards until their complete equations and controls are connected.

## 6. Head and wing require two independent wings

Fixing the body alone does not implement general magnetic option-6 screening.
`dielecmodel.cpp::accumulate_wing_mu_for_pair` and `compute_wing` use reciprocal
denominators; `construct_L`, `replace_rpa_response_headwing`, and
`rewrite_eps_abf_space` infer one wing from the conjugate transpose of the
other. The latter also uses Hermiticity when projecting the regular body.

An independent optical oracle should start from the **same full Lehmann
response** for an enlarged set of charge/dipole probes. For off-diagonal
eigenstates, `v=i[H,r]` gives

\[
 r_{mn}={v_{mn}\over i(\epsilon_m-\epsilon_n)}.
\]

This supplies the interband position vertex from the existing correctly
computed velocity. The external finite-cluster oracle below chooses the head
coordinate `+i q_alpha r_alpha`. The native transition convention instead uses
`<u_mk|u_n,k+q> = q_alpha v_mn,alpha/Delta = -i q_alpha r_mn,alpha`, for occupied
m and empty n. These head coordinates differ by one overall sign: both wings
must change sign together, leaving the Schur body correction invariant. Their
factor i and q direction must be retained consistently on the two external
vertices. No intraband term is being added. Compute each enlarged matrix entry
with both `1/(Delta-iw)` and `1/(Delta+iw)` denominators before taking the
scaled q limit. Applying the q-limit factors to that complete response avoids
guessing the missing sine term from a single existing wing formula. The
existing spin, occupation and quadrature normalization must be retained.

In the native convention let `p_mu,alpha=c_mn,mu v_nm,alpha` and let kappa
denote the existing square-root Coulomb/volume normalization. The lower and
upper dielectric wings for one occupied-empty pair are respectively

\[
 Z_{R,\mu\alpha}=\kappa\left[
 {p_{\mu\alpha}\over\Delta(\Delta-i\omega)}-
 {p_{\mu\alpha}^*\over\Delta(\Delta+i\omega)}\right],\qquad
 Z_{L,\alpha\mu}=\kappa\left[
 {p_{\mu\alpha}^*\over\Delta(\Delta-i\omega)}-
 {p_{\mu\alpha}\over\Delta(\Delta+i\omega)}\right].
\]

The relative minus sign follows from reversing q on the head vertex in the
backward transition. If the reciprocal full-BZ sum of p is purely imaginary,
the lower expression reduces to `-2 kappa p*/(Delta^2+omega^2)`, agreeing with
the current native `cal_wing` dielectric convention. A changed global head
gauge cannot repair missing odd-frequency terms or justify inferring one wing
from the other's adjoint at the same positive frequency.

For each direction n, write the dielectric matrix as

\[
 E(n)=\begin{pmatrix} n^T H n&n^T Z_L\\Z_R n&B\end{pmatrix},\quad
 D=B^{-1},\quad L=H-Z_L D Z_R.
\]

Z_L and Z_R are independent at the same positive frequency. The full inverse
is the ordinary non-Hermitian Schur complement with
`s=n^T L n`, `A_R=D Z_R`, `A_L=Z_L D`:

\[
 E(n)^{-1}=\begin{pmatrix}
 s^{-1}&-s^{-1}n^T A_L\\
 -A_R n s^{-1}&D+A_R n s^{-1}n^T A_L
 \end{pmatrix}.
\]

For the present inversion-symmetric angular quadrature, the two off-diagonal
blocks average to zero because s(-n)=s(n). With
`a0=sum_n weight/s` and `K=sum_n weight n n^T/s`, the embedded average is

\[
 \langle E^{-1}\rangle=a_0 P+D+A_R K A_L.
\]

It reduces to the existing `a0 P+D+A K A^dagger` only when the required
Hermitian relations hold. This preserves the angular approximation and
filtered Coulomb space; it restores the matrix identity for general inputs.

The thin projection optimization also has a direct generalization. For
`T=I-YY^dagger`, with Y spanning the head and filtered directions, define
`G_R=E Y` and `G_L=Y^dagger E` independently. Then

\[
 M=TET+YY^\dagger
 =E-G_RY^\dagger-YG_L+
       Y(I+Y^\dagger E Y)Y^\dagger.
\]

The baseline used `G_R^dagger` for G_L, which is wrong when E is not Hermitian.
After a general identity solve, `D=M^{-1}-YY^dagger` is the embedded regular
body inverse. The LU capability already exists in
`invert_headwing_body_with_identity_solve`; this audit implements the full
input algebra and independent wings. The three-dimensional controls in
section 13 and the [strict-2D complete-Wc controls](nonreciprocal-strict2d-screening.md)
check their respective optical consumers. Strict-2D signed W-to-Sigma,
nonorthogonal/shrink/IBZ conjunctions and continuation remain separate.
The [body RPA trace-log consumer](nonreciprocal-rpa-energy.md) is also checked;
optical RPA remains guarded. None inherits Hermiticity solely from an
imaginary-frequency label.

The finite-cluster oracle uses actual diagonal Hermitian position probes on
the same three sites, inserts the density-head factor i explicitly, and
compares direct 4x4 inversions over five antipodal direction pairs with the
formula above. At Bz=0.21 and the 24-node frequency of largest antisymmetry:

| Check | Maximum absolute residual |
| --- | ---: |
| Independent left wing minus adjoint right wing | 1.02340e-4 |
| General thin projector versus explicit TET+YY^dagger | 2.22e-16 |
| Existing Hermitian-only thin expression versus explicit projector | 1.02340e-4 |
| General angular Schur formula versus direct inversions | 2.22e-16 |
| Conjugate-wing assembly versus direct inversions | 3.10754e-7 |

These are dimensionless model dielectric/inverse entries. The Bz=0 reciprocal
control gives all disagreements below 3.34e-16. This checks the necessary
block algebra, not the native periodic optical implementation.

## 7. Independent complete R=0 oracle and observed errors

The [native model/grid exporter](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/self-energy-controls/nonreciprocal-grid-model.cpp)
only obtains the existing model eigenpairs and calls the existing optional
GreenX generator. The [independent Python route](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/self-energy-controls/nonreciprocal-route.py)
implements spectral sums and general linear algebra; it does **not** call a
purportedly repaired native Chi0, W or Sigma implementation.

Take the model's local charge vertices and `v_aa=1`, `v_ab=0.15` for a!=b.
Its exact reference chi follows the Lehmann sum in section 1. A 3x3 general
Dyson solve gives W. Independently, introduce 9 occupied/empty transitions,
`Bv=[d,d*]`, `S=diag(I,-I)` and `D0=diag(Delta,-Delta)`. The Woodbury identity
gives

\[
 W_c(z)=v B_v[zI-D_0-SB_v^\dagger vB_v]^{-1}S B_v^\dagger v.
\]

The resulting 18x18 transition matrix has real positive/negative pole pairs.
Its residues reproduce the direct 3x3 Dyson W to 3.8e-16. Repeated poles are
treated as complete eigenspaces: the sum of their residues is invariant and
Hermitian; individual residues from arbitrary nonnormal eigenvectors need
not be. The initial ungrouped-residue assertion failure is preserved in
`nonreciprocal-route-ungrouped-poles-failed.log`; grouping the repeated pole
corrects the oracle, without Hermitizing any data or changing the target.

Writing the positive-pole residues R_p and frequencies Omega_p,

\[
 W_c(+t)=-\sum_p R_p e^{-\Omega_p t},\quad
 W_c(-t)=-\sum_p R_p^* e^{-\Omega_p t}.
\]

For local charge vertices the spatial part of the Sigma contraction is an
entrywise product; all four spinor blocks remain. With
`P_n=c_n c_n^dagger`, each R is lifted over the spin blocks, and the exact
frequency oracle is

\[
 \Sigma(i\omega)=
 \sum_{n\in\mathrm{occ},p}{P_n\odot R_p^*\over
             i\omega+\mu-\epsilon_n+\Omega_p}
 -\sum_{n\in\mathrm{empty},p}{P_n\odot R_p\over
             \epsilon_n-\mu+\Omega_p-i\omega}.
\]

At Bz=0.21 the electronic gap is 1.6930819335 Ha, the particle-hole
transitions lie in [1.69308,2.42790] Ha, and the RPA poles in
[1.69549,2.43009] Ha. Positive-pole residues are Hermitian to 1.5e-14 and
positive semidefinite to 4.1e-19. These independent conditions check the
causal sign and the time-domain oracle.

All following cases use minimax interval [0.4,4] Ha. Errors are absolute
maximum matrix elements; Sigma errors are Ha:

| Bz=0.21 comparison | 16, reg=0 | 24, reg=0 | 16, reg=1e-12 | 24, reg=1e-12 |
| --- | ---: | ---: | ---: | ---: |
| Signed-time chi versus exact Lindhard | 9.18e-11 | 8.42e-12 | 1.28e-10 | 2.88e-11 |
| Numerically reconstructed W(+/-t) versus exact poles | 8.44e-8 | 3.36e-7 | 6.17e-12 | 3.40e-11 |
| Sigma transform using exact signed-time W | 1.97e-9 | 2.07e-11 | 5.03e-10 | 3.89e-11 |
| Complete signed chi -> W -> Sigma route | 2.66e-6 | 5.27e-9 | 5.02e-10 | 3.89e-11 |
| Incorrect reuse of exact W(+t) at -t | 1.14e-6 | 1.14e-6 | 1.14e-6 | 1.14e-6 |

The old cosine response error at 24 nodes is 5.63367e-5, reproducing the
existing native counterexample. The full W norm is 0.00749507 and Sigma norm
0.00242344 Ha in that model. The reciprocal Bz=0 full-route Sigma errors are
1.97e-6, 5.47e-9, 3.65e-10 and 4.31e-11 Ha in the same column order. Thus
inverse-transform conditioning is separate from nonreciprocity; adding the
correct equation alone is not a grid-convergence certificate.

An independently time-reversed partner and a dense complex auxiliary basis
rotation check the dynamic antiunitary law. At 24 nodes and Bz=0.21, the
positive-frequency transpose law agrees to 3.48e-18; substituting entrywise
conjugation disagrees by 1.06303e-4. This is a symmetry-related pair of models,
not a claim that reversing the magnetic field is a symmetry of one model.

The [JSON results](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/self-energy-controls/nonreciprocal-route-results.json)
retain every case and intermediate residual. Raw matrices are retained in
`nonreciprocal-route-n*-bz*.npz`. These independent oracles preceded the native
implementation. Sections 9–13 record its subsequent comparisons, including the
unchanged-default conditioning failures and the explicit-regularization
optical pass; they do not establish universal production acceptance.

## 8. Coordinated implementation and acceptance order

1. Expose optional inverse sine generation with explicit availability, unchanged
   default forward weights, finite checks and analytic even/odd transform
   controls. Keep the measured conditioning limits visible. This is the
   root-owned first implementation stage.
2. Make the complex charge path form both signed-time parts and preserve the
   full frequency response. Update immediate/delayed shrink and collectors
   together. Compare all entries of the Bz witness directly with the independent
   Lindhard sum before any screening. Keep Bz=0 and arbitrary complete spin
   rotations as controls.
3. Feed the full response into the existing general dielectric solve. Preserve
   full W at positive frequency. Compare with direct Dyson, then invert with
   both Cbar/Sbar and compare W(+/-t) with exact RPA poles. Keep coarse-grid
   approximation error separate from assembly error.
4. Install matching signed-time W for each Sigma contraction and preserve the
   four-channel restoration/union semantics. Compare every full complex Sigma
   matrix with the spectral oracle, with separate exact-W and numerical-W
   controls. Exercise one and multiple MPI ranks, including empty local tiles.
5. Implement/verify the dynamic q-star law or perform reconstruction strictly
   in signed time. Require genuine reduced stars and an independent full-BZ
   band sum; a two-model antiunitary algebra check alone does not accept the
   periodic reconstruction. Include complex compatible shrink maps and both
   Fourier/compression orderings.
6. Extend the optical vertex pair, thin body projection and independent wings
   before enabling general magnetic head/wing. Verify the Schur inverse against
   a directly assembled enlarged dielectric matrix and directional small-q
   Lindhard limits. Include the user's ordinary option-6 route; a body-only
   run is a development control, not completion of magnetic option-6 GW.

These stages restore the unrestricted gapped GW response equations without
replacing the explicitly selected QSGW staticization. Changing quadrature
regularization defaults, adopting a different continuation, changing the QSGW
map, or accepting lossy compression is a separate choice. The initial
derivation did not itself change production screening/Sigma or submit a
material job. The staged implementation controls below record later source
changes and coordinated local builds; they do not accept a material run.

## 9. Independent native dielectric-body control

The [direct Lindhard-to-Dyson oracle](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/self-energy-controls/general-screening-oracle.cpp)
bypasses the time transform to isolate the frequency consumers. It uses
exact charge vertices from a gapped three-site SOC Hamiltonian with Bz=0
and 0.21 Ha, eight positive frequencies, and the full spectral expression
in section 1. It separately tests a real auxiliary frame and a complex
DFT rotation of that frame. Known positive Hermitian square roots define
two different Coulomb matrices, one in epsilon and one in the W sandwich.

For each frequency the independent reference is

\[
 E=I-S_\epsilon\chi S_\epsilon,\qquad
 W_c=S_W(E^{-1}-I)S_W.
\]

The reference uses an independently written 3-by-3 partial-pivot
Gauss–Jordan inverse. It checks both W and the Dyson residual against the
actual serial and BLACS consumers. Atom groupings `(3)`, `(1,2)` and
`(1,1,1)` distinguish full atom-diagonal blocks, unequal atom dimensions
and ordered off-diagonal blocks. BLACS block sizes 1 and 3 on four ranks
include empty local tiles. The full-frequency collector must copy exact
ordered entries; disabling conjugation in the old Hermitian collector
would still infer the lower triangle by transposition and fails this
contract.

The parent audit's final one-/four-rank runs pass all 48 cases (24 each):
maximum native W error `4.193694003173931e-16`, serial W error
`9.887964771484038e-17`, and Dyson residual `4.440892098500626e-16`.
[Case records](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/self-energy-controls/general-screening-verification.json)
retain the numerical controls and source/binary hashes. A test-fixture
descriptor initially lacked its BLACS grid and raised SIGFPE; that failure
and its backtrace are retained separately and are not a production defect.

Two real consumer defects were exposed after correcting that fixture.
The serial W sandwich mixed row/column-major matrices. The BLACS Gamma
shortcut also converted a Hermitian complex Coulomb matrix to real merely
because q=0; a complex auxiliary gauge makes that assumption false. The
fixed route uses a consistent matrix major and applies the real shortcut
only when every Coulomb entry is actually real. The former complex-gauge
control had W error about `4.08e-3` and Dyson residual `6.16e-3`; the
failed logs remain preserved.

This establishes the finite general-frequency **body** solve, including
ordered collection and MPI distribution, not optical wings, rectangular
compression, complete Sigma, minimax accuracy or a material spectrum.
The one-rank serial general path is explicitly bounded; multi-rank
general screening uses the BLACS route. General Cholesky is rejected.
Section 11 records the subsequent cRPA consumer validation; the separate
[RPA energy note](nonreciprocal-rpa-energy.md) owns its complex BLACS2D body
controls. Those acceptances do not come from the W body test itself.

## 10. Complete native dense signed-time GW control

The [native exporter](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/self-energy-controls/native-signed-gw.cpp)
now runs the actual complex LibRI chi build, full BLACS dielectric solve,
signed frequency-to-time transform and G0W0 contraction on the same gapped
R=0 model. It enables no optical correction, auxiliary compression, Padé
continuation or QSGW staticization. The Coulomb matrix is
`v=0.85 I+0.15 ones`, with exact local density RI vertices. Both Bz=0 and
Bz=0.21 are included. These are finite operator controls, not Bi2Se3 results.

Each execution exports native chi(iw), W(iw), W(+tau), and an independently
transformed W(-tau), obtained by exchanging the positive/negative frequency
halves. It also writes **all four native Sigma(R,iw) spin-block checkpoint
shards** and the complete 6-by-6 KS matrix. The
[independent checker](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/self-energy-controls/native-signed-gw-check.py)
reads the actual raw AO shards, verifies disjoint ownership and compares every
element against the Lindhard, direct Dyson and 18-dimensional RPA-pole oracles
of section 7. It separately evaluates the explicit G(tau)W(tau) quadrature
using the exported native W samples, isolating contraction error from inverse
transform error. KS projection is checked against `C^dagger Sigma_AO C`.

The executed batch comprises 16/24 nodes, regularization 0/1e-12, Bz=0/0.21
and one/four MPI ranks, plus a four-rank descriptor with empty local matrix
tiles: **17/17 cases pass their declared implementation checks**. All 12,384
AO self-energy entries are present exactly once across the raw shards. The
native W-versus-Dyson(native chi) error is at most **4.20671e-16**, and the KS
projection error is at most **2.16853e-18**. At 24 nodes with explicit
regularization 1e-12, the native Sigma contraction differs from the independent
quadrature by at most **1.85269e-18 Ha**. Reusing W(+tau) in the negative-time
contraction would instead differ by **1.13979e-6 Ha** for Bz=0.21.

Quadrature accuracy is a separate observable. The following errors are maxima
over the one/four-rank executions; Sigma errors are in Hartree:

| Nodes | Explicit regularization | Sigma vs exact poles, Bz=0 | Sigma vs exact poles, Bz=0.21 | Largest one/four-rank Sigma difference | Maximum row sum of inverse sine |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0 | 1.50480e-6 | 4.92136e-7 | 3.20984e-7 | 5.14133e7 |
| 16 | 1e-12 | 3.64965e-10 | 5.00038e-10 | 8.93056e-12 | 9.44134e4 |
| 24 | 0 | 2.64694e-9 | 4.12328e-9 | 3.57226e-9 | 1.30706e11 |
| 24 | 1e-12 | 4.30878e-11 | 3.89180e-11 | 2.04488e-14 | 3.28015e4 |

The unregularized inverse amplifies frequency-domain roundoff: native chi is
identical between one and four ranks in these cases, while W(iw) differs by
only 2.2e-16--3.4e-16. Nevertheless, the 16-node unregularized Sigma differs
by up to 3.21e-7 Ha between those layouts. This is evidence for transform
conditioning, not a missing spin block or a change of GW equations. At
24 nodes with explicit regularization, W(+/-tau) agrees with exact poles to
3.39715e-11 and the full Sigma errors are approximately the exact-W time
quadrature error. The empty-tile control changes Sigma by at most 1.93e-14 Ha
relative to the ordinary four-rank layout.

Across all unregularized cases, independent summation of the same inverse
weights differs from the native inverse by up to 1.44e-8, within a recorded
coefficient-norm roundoff bound; the worst native Sigma-versus-quadrature
residual is 1.21e-10 Ha. Those conditioning-dependent implementation gates do
not certify arbitrary-grid physical accuracy. No production regularization
default was changed. The tested explicit 1e-12 is a documented control of the
existing numerical option, and its success does not establish one universally
optimal regularization.

The [full case/rank comparison JSON](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/self-energy-controls/native-signed-gw-results.json),
[one-rank batch log](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/self-energy-controls/native-signed-gw-batch-np1.log)
and [four-rank batch log](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/self-energy-controls/native-signed-gw-batch-np4.log)
link the exported arrays, checkpoint shards and per-case checks. Exact replay
commands are in the [build script](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/self-energy-controls/build-native-signed-gw.sh)
and [batch runner](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/self-energy-controls/run-native-signed-gw-batch.sh).
Two initial fixture failures are retained: a cyclic descriptor violated the
existing consecutive-tile GW transfer contract, and the exporter initially
queried full KS matrices without enabling their retention flag. Neither was
treated as a new production defect.

This accepts the complete tested **dense, uncompressed, body-only** native
signed-time GW route and its actual negative-time W selection. Optical
conjunctions, compressed GW and material-scale numerical acceptance remain
separate work. Subsequent frequency-only cRPA and RPA body controls are
documented separately and do not require this inverse time transform.


## 11. General-frequency cRPA consumer and independent control

The cRPA response selection changes the transition weights, not the frequency
symmetry. For a real, nonnegative band weight $w_{nk}$ on the original KS poles,

$$
P^d_{\mu\nu}(q,z)=\frac{1}{N_k}\sum_{knm}
 w_{nk}w_{m,k+q}\frac{f_{nk}-f_{m,k+q}}{z+\epsilon_{nk}-\epsilon_{m,k+q}}
 d^\mu_{nm}(k,q)d^{\nu *}_{nm}(k,q),\qquad P^r=P^0-P^d.
$$

The selected Green functions weight both occupied and empty branches. Hence
both terms of the signed-time response survive, with the same endpoint
weight. A covariant selector must transform as an operator under the actual
magnetic group; choosing equal same-index arrays is insufficient in a changed
KS gauge. Existing selection validation remains in force. The independent
control below uses nondegenerate Bz-split levels and validates the magnetic
Hamiltonian before using its group.

At every positive imaginary node the two screened interactions are

$$
U(q,z)=[I-v(q)P^r(q,z)]^{-1}v(q),\qquad
W(q,z)=[I-v(q)P^0(q,z)]^{-1}v(q).
$$

Neither may generally be replaced by its Hermitian part at the same positive
frequency. The cRPA output is an ordered density tensor. With the existing
vertex convention $D^\mu_{ab}(q)$, the implementation contracts

$$
U_{ab,cd}(z)=\frac1{N_q}\sum_{q\mu\nu}
 [D^\mu_{ba}(q)]^*U_{\mu\nu}(q,z)D^\nu_{cd}(q).
$$

`contract_crpa_blacs` and `contract_rectangular_ordered_pair_vertex` already
retain every kernel entry and implement the displayed left conjugation,
orbital-index reversal and ordinary transpose on the right. There is no
additional tensor Hermitization to remove. Nor does this frequency-only
consumer need an inverse sine transform. The necessary prerequisites were
the full ordered $P^0-P^d$ subtraction, general dielectric inverse and dynamic
antiunitary q restoration implemented in earlier sections. The bare Coulomb
input still has a Hermitian upper-block contract, which is legitimate for
$v$ and does not imply a triangular dynamic response.

The staged SOC guard in `screen_crpa_response` is therefore removed in this
source slice. Core optical guards and the scalar response contract are
unchanged. The coordinated one-/four-rank validation described below now
passes; this removes the temporary guard without changing the selected
cRPA transition model or its ordered tensor convention.

The extended `test_crpa_soc` retains its original Bz=0 model and adds Bz=0.21
with the valid magnetic group $C_3+\Theta M_y C_3$. The former
$\Theta M_z$ group is not reused, because it would reverse Bz. A six-orbital
cluster frame mixes sites and spin components by a known complex unitary,
so comparing every four-index tensor entry probes the full response rather
than only same-site diagonal screening. The control uses fractional endpoint
weights, exact finite Lindhard sums for all ordered $P^0,P^d,P^r$ entries,
and an independently pivoted three-dimensional solve of $(I-vP)W=v$.
All native response entries must have exactly one MPI owner. Full-q and
actual antiunitary q-star routes, shared-P0 alias preservation, the untouched
incoming selector and unchanged mean-field inputs are checked separately.
The existing one-/four-rank test registration is reused; no new parser or
error-message tests are introduced.


The rebuilt serial and four-rank `test_crpa_soc` both pass (37.47 s and
118.04 s). Their [preserved detailed log](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/ctest-general-consumers-libri-detail.log)
contains the following maxima over both layouts and their full/reduced paths:

| Control | Bz=0 | Bz=0.21 |
| --- | ---: | ---: |
| Actual full to representative q grid | 9 to 3 | 9 to 4 |
| Full/reduced ordered U/W difference | 1.77636e-15 | 1.44329e-15 |
| P0/Pd/Pr versus finite Lindhard | 1.39990e-11 | 8.42040e-12 |
| Complete ordered v/U/W versus independent solve and vertex contraction | 1.71299e-11 | 8.38954e-12 |
| Maximum anti-Hermitian part of exact P0(+iw) | 1.69198e-17 | 7.38573e-5 |

The nonzero last entry is the discriminating witness: replacing the dynamic
response by a same-frequency Hermitian matrix would erase a resolved part
of the magnetic response. Shared-P0 U-only evaluations agree exactly with
their complete calculations, while the resident response, incoming selector
and all mean-field E/C/f inputs remain unchanged. These tests use forward
cosine/sine transforms only, so the inverse-sine conditioning issue of
section 10 is not part of this cRPA result. The
[coordinated suite](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/ctest-general-consumers-libri.log)
passes 7/7 LibRI checks; the separate no-LibRI consumer suite passes 3/3.

This accepts the tested dense, uncompressed, imaginary-frequency cRPA
body route, including true magnetic antiunitary q-star restoration and
ordered orbital contraction. It is not a fresh validation of compressed
cRPA, nonreciprocal optical cRPA, elementwise continuation covariance or a
material interaction tensor. Existing optical fixtures still run in the
regression, but the new magnetic conjunction is body-only.

## 12. Compressed positive-frequency W export

The dynamical kernel and its positive-time value have different storage and
antiunitary contracts. The ordinary GW contraction unfolds W(q,+tau), for
which a Hermitian pair representation is valid. Optional W(q,+iw) export
must instead preserve every ordered block. For a complex rectangular
auxiliary map U(q), the common algebra is

\[
 W_{\rm parent}(q,z)=U(q)^\dagger W_{\rm active}(q,z)U(q).
\]

This formula does not make a non-Hermitian W(q,+iw) Hermitian. The new
`Chi0::unfold_abfs_Wc_q` overload selects full ordered collection and requests
all required output pair directions for frequency export. The existing
signature still selects the time/Hermitian behavior. The same two GEMMs
implement the projection; no overlap or spin factor is inserted. The path
streams one frequency at a time and does not retain every unfolded frequency
matrix at once. It requires the existing LibRI unfolding implementation.

After unfolding, the original real auxiliary basis supplies the crystal
operation matrices. The full-frequency Fourier helper first checks one
owner per ordered atom pair, gathers each explicit entry without conjugate
completion, then applies

\[
 W(gq,+i\omega)=D_g W(q,+i\omega)D_g^\dagger
 \quad\hbox{(unitary)},
\]
\[
 W(aq,+i\omega)=D_a W(q,+i\omega)^T D_a^\dagger
 \quad\hbox{(antiunitary)}.
\]

The existing block rotation routine implements the static antiunitary
conjugation. Supplying the complete adjoint block map to that operation
therefore gives the required transpose, since `(W^dagger)*=W^T`. Orbital
permutations, lattice translations and representative phase factors continue
to come from the existing operation plan. The general-frequency path does
not apply the old Hermitian little-group completion or average. It then
uses the ordinary discrete transform

\[
 W_{IJ}(R,+i\omega)=\frac1{N_q}\sum_q
   e^{-2\pi i q\cdot R}W_{IJ}(q,+i\omega).
\]

The separate q/-q **charge-reality** fold still uses complex conjugation in
the real auxiliary gauge. It is not replaced by the physical antiunitary
law. Existing helper signatures and time-domain behavior are retained.

Focused controls in `test_epsilon` compare the rectangular complex
projection against an explicit four-index contraction, then compare all
27 real-space matrix entries with a direct discrete sum. One- and two-atom
partitions test both the internal diagonal-atom triangle and independent
opposite atom blocks; empty MPI owners participate. A separate algebraic
antiunitary test compares the frequency transpose against a direct Fourier
sum and does not claim that a pure-Theta material can have a nonreciprocal
charge kernel. The focused test now passes in the
[LibRI batch](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/ctest-general-consumers-libri.log),
with all asserted entrywise projection/Fourier errors below 1e-12. The
antiunitary helper also passes in the
[non-LibRI batch](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/ctest-general-consumers-nolib.log).
The initial non-LibRI build exposed a LibRI-only template instantiation;
an explicit compile-time capability branch fixes that build without changing
the LibRI calculation. The actual compressed GW export now also passes the physical optical caller
replay in section 13, including a non-Hermitian magnetic response with
genuine antiunitary IBZ reconstruction and serialized ordered matrices.


## 13. Complete nonorthogonal magnetic optical conjunction

The scalar-charge test basis is nonorthogonal, with
S=(1-s)I+sJ in each spin component and s=0.12. Set A=S^(1/2),
H_AO=A H_orth A and C_AO=A^(-1) U. Its three site-charge matrices are
Gamma_a=(P_a S+S P_a)/2, so sum_a Gamma_a=S. Position matrices use the
same vertices, r_AO,alpha=sum_a x_a,alpha Gamma_a. The independent
orthogonal velocity is the analytic hopping derivative plus
`i[H_orth,r_orth]`, where r_orth=A^(-1) r_AO A^(-1). This constructs
matching H/S/r/RI data instead of independently assigning incompatible
optical and charge gauges.

For Bz=0.21 Ha the valid magnetic group includes C3 and Theta My. In the
real fractional row-lattice convention, My is the matrix
`[[1,0,0],[1,-1,0],[0,0,1]]`. Its antiunitary action sends
`k -> (-kx,ky-kx,-kz)`, followed by the C3 rotation. The spinor action before
K sends site a to -a+r modulo three and multiplies spin up/down by
`exp(∓i pi r/3)`. This preserves Bz; the earlier Theta Mz operation does not.
For O=R_z(2 pi r/3) diag(1,-1,1), the preflight independently checks

\[
H_{\rm AO}(k')=D H_{\rm AO}(k)^*D^\dagger,
\qquad S=D S^*D^\dagger,
\]
\[
r_\alpha=\sum_\beta O_{\alpha\beta}D r_\beta^*D^\dagger,
\qquad
v_\alpha(k')=-\sum_\beta O_{\alpha\beta}D v_\beta(k)^*D^\dagger.
\]

All five H/S/r/charge/velocity checks are below 8.89e-16. Independent star
counting gives 9 k points reduced to four stars with sizes 1,2,3,3. The
native IBZ inputs actually contain four points; they do not retain a full
input and merely turn on a flag. Exact auxiliary compression retains the
active vector (0.8,0.6) on each atom and removes the charge-free orthogonal
channel. The eight full/folded, full/IBZ, one/four-rank routes use 24 nodes
in [0.4,4] Ha and explicitly set the existing regulation option to 1e-12.
All pass the unchanged 1e-10 Ha complete-Sigma comparison gate:
paired Sigma <=2.585e-11 Ha and one/four-rank Sigma <=4.351e-14 Ha.
Paired corrected W(i omega) is within 6.23e-16 and captured W(tau) within
5.03e-13. The actual public API exports the first two positive frequencies,
all nine R cells and all ordered 6x6 auxiliary entries; those files agree
with an independent discrete Fourier transform within 4.26e-17, including
the compressed consumer. The diagnostic W capture is a separate
identical-input invocation, not a hook into the exact object consumed by
API Sigma.

The physical dynamic-antiunitary witness is resolved: W-W^dagger reaches
6.803e-5. The correct Theta My transpose law has residual below 4.69e-16;
substituting same-frequency conjugation instead gives 6.803e-5. Away from
Gamma all rotated antiunitary operations agree within 4.69e-16. Gamma
retains a finite angular C3 residual of 6.58e-10 in this Bz fixture, also
seen in the rotated antiunitary checks. An independent full directional
inverse reproduces this quadrature residual in the companion controls;
no angular projection or averaging is applied to the production result.

The broader pair of Bz=0/0.21 optical controls completes sixteen routes
at each explicit regulation setting. Regulation 0 fails: paired Sigma
6.593e-9 Ha and one/four-rank Sigma 5.428e-9 Ha. The inverse-sine row sum
is 1.30705e11, which amplifies machine-scale frequency-domain differences.
At regulation 1e-12 the corresponding row sum is 32801.48; paired Sigma
is 3.473e-11 Ha and one/four-rank Sigma 1.600e-13 Ha. This demonstrates
conditioning sensitivity and route agreement, not absolute optical
self-energy accuracy. Regulation also changes forward quadrature fits;
cross-setting Sigma differs by as much as 4.663e-8 Ha, and no independent
exact-pole reference for the complete periodic optical Sigma is provided.
No default, Padé rule, or acceptance gate is changed by these controls.

The [response audit](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/response-audit.md#general-magnetic-optical-response-derivation-and-implementation-checkpoint)
links the original failures, corrected captures, independent directional
inverse and frozen source/library provenance. The completed physical
[Theta My comparison](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/optical-generalization/theta-my-reg1e-12-comparison.json)
and [dynamic covariance record](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/optical-generalization/theta-my-reg1e-12-dynamic-covariance.json)
are the bounded acceptance evidence. These results do not certify a
material input, lossy auxiliary truncation, arbitrary complex Gamma gauges,
metallic intraband response, or every optical consumer.

## 14. Absolute optical Sigma, bare subtraction and an independent spectral reference

The [absolute-error control](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/optical-generalization/absolute-sigma/README.md)
starts from the same finite response, angular directions and integration
weights as the native operator. It distinguishes an error in that operator
from inverse-transform conditioning and finite forward-quadrature error.
Neither a full/IBZ agreement nor a same-input dielectric inverse proves this
absolute self-energy test.

### 14.1 A nonunit weight sum leaves an instantaneous term

Use separate Coulomb matrices throughout: v_eps generates the dielectric
and its retained head/body projectors, whereas C_w=sqrt(v_wc) supplies the
two external screened-interaction legs. They need not be equal or commute.
Let P_h=u*u-dagger, T=R*R-dagger and P_r=P_h+T. For actual antipodally paired
angular directions and weights rho_a, set s_rho=sum_a rho_a. The free native
rewrite in section6 returns the embedded inverse

\[
 \mathcal I(z)=\sum_a\rho_a Q E_a(z)^{-1}Q^\dagger
 +(1-s_\rho)R B(z)^{-1}R^\dagger,\qquad Q=[u,R].
 \tag{14.1}
\]

The unweighted regular-body baseline is part of this existing quadrature
formula. At zero polarizability E_a=I and B=I, so

\[
 \mathcal I_0=s_\rho P_h+T,
 \qquad
 W_{c,\mathrm{old}}(z)=C_w[\mathcal I(z)-P_r]C_w,
\]
\[
 K_\infty=\lim_{|z|\to\infty}W_{c,\mathrm{old}}(z)
 =(s_\rho-1)C_wP_hC_w.
 \tag{14.2}
\]

Thus subtracting the complete P_r is inconsistent with the numerical bare
head moment whenever s_rho differs from1. This is already a zero-response
failure, independent of the number of time/frequency nodes. A frequency-
independent K_infinity is a contact distribution in imaginary time, not a
screened finite-frequency decay pole. It cannot be assigned an arbitrary
large pole to make a pure-decay oracle fit. Its contribution to a continuum
self-energy also requires an equal-time prescription; this audit does not
silently assign an additional Fock term.

The minimal matched correlation subtraction is

\[
 W_{c,\mathrm{corr}}(z)
 =C_w[\mathcal I(z)-s_\rho P_h-T]C_w
 =W_{c,\mathrm{old}}(z)+(1-s_\rho)C_wP_hC_w.
 \tag{14.3}
\]

Implementation adds (1-s_rho)P_h at the correlation-subtraction boundary,
before the external C_w multiplications. It leaves the averaged inverse
(14.1), every angular weight and the existing optional normalization setting
unchanged. The head is still the one selected from v_eps, including the
distinct option53 s-head. This is different from replacing rho by rho/s_rho,
which changes the finite dynamical residues and is a numerical-policy choice.
Head-only4/5 use a dielectric head overwrite from Tr(head)/3, with no rho sum;
their source zero-response argument is separate. Strict2D already returns a
complete Wc and is outside (14.3).

Define the directional and regular-body dynamical interactions by subtracting
their own bare matrices:

\[
 W_{c,a}=C_wQ(E_a^{-1}-I)Q^\dagger C_w,\qquad
 W_{c,B}=C_wR(B^{-1}-I)R^\dagger C_w.
\]

Then the exact fixed-rule decompositions are

\[
 W_{c,\mathrm{corr}}=\sum_a\rho_a W_{c,a}+(1-s_\rho)W_{c,B},
 \qquad W_{c,\mathrm{old}}=W_{c,\mathrm{corr}}+K_\infty.
 \tag{14.4}
\]

Real pole support bounds survive either decomposition. If s_rho>1, however,
the body coefficient is negative; a pole-support theorem alone does not
establish positivity/passivity of the averaged spectral residues. A finite
signed combination of these stable real-pole resolvents still remains
analytic in the upper half-plane: a spectral-sign failure is not by itself
an upper-half-plane instability. That quadrature-convergence question remains separate from removing the
zero-response constant. The older saved rule with sum1.0000252926 and the
newer rule with sum0.9998441222 must not be conflated.

### 14.2 Hermitian congruence supplies poles and residues without a fitted W

For either a direction or the body, collect the actual weighted ordered
transition columns into X. In the directional case each column is

\[
 x_l(n)=\sqrt{w_l|\Delta f_l|}
 \begin{pmatrix}
 \sqrt{4\pi/\Omega}\,n^T d_l\\R^\dagger\sqrt{v_\epsilon}\,\rho_l
 \end{pmatrix},\qquad
 d_{mn}={v_{mn}\over\epsilon_n-\epsilon_m}.
 \tag{14.5}
\]

The signed denominator is retained for reversed transitions: in a real
parent auxiliary basis rho_nm=rho_mn* but d_nm=-d_mn*. Both compressed
columns transform with the same U, not a conjugate U inferred on one side.
Let Delta be the positive diagonal matrix of absolute transition gaps and
J the diagonal transition sign. The exact signed response and inverse obey

\[
 \chi_X(z)=X(zI-J\Delta)^{-1}JX^\dagger,\qquad
 E(z)^{-1}-I=X[zI-JH]^{-1}JX^\dagger,
 \quad H=\Delta+X^\dagger X>0.
 \tag{14.6}
\]

The generally nonnormal JH is similar to a Hermitian matrix:

\[
 A=H^{1/2}JH^{1/2}=Y\Lambda Y^\dagger,\qquad
 (zI-JH)^{-1}J=H^{-1/2}A(zI-A)^{-1}H^{-1/2}.
 \tag{14.7}
\]

For external leg F=C_wQ (or C_wR for the body), set
a_j=F X H^(-1/2)Y_j. The complete dynamical spectral reference is

\[
 W_c(z)=\sum_j{\mathcal R_j\over z-\lambda_j},\qquad
 \mathcal R_j=\lambda_j a_j a_j^\dagger.
 \tag{14.8}
\]

This avoids left/right eigenvector conditioning of a nonnormal transition
matrix. Degenerate eigenvalues are compared through their complete spectral
projectors, not arbitrarily selected individual columns. Each directional
positive-pole residue is PSD and each negative-pole residue NSD. These
properties do not automatically survive a negative body coefficient in
(14.4). Filtered, singular dielectric Coulomb channels require no inverse
Coulomb here. External legs change residues, not pole positions.

### 14.3 Signed time and the complete self-energy

For t>0 the frequency convention of section1 gives

\[
 W_c(+t)=-\sum_{\lambda_j>0}\mathcal R_j e^{-\lambda_jt},\qquad
 W_c(-t)=+\sum_{\lambda_j<0}\mathcal R_j e^{\lambda_jt}.
 \tag{14.9}
\]

Native fermionic G has the opposite sign to the conventional Green function:
it is positive on empty states at +t and negative on occupied states at -t.
The native LibRI Sigma contraction has no additional minus. With xi_n=eps_n-mu,
let C_nq[R] denote the independently evaluated full RI contraction of residue
R with the internal-state projector, including every spinor block. In KS
notation its entries are sum_PQ Gamma^P_mn R_PQ Gamma^{Q*}_m'n, with the
same momentum/Fourier convention as the input vertices. Then

\[
 \Sigma_c(k,z)=\sum_q w_q\left[
 \sum_{n\in\mathrm{empty},\lambda_j>0}
 {\mathcal C_{nkq}[\mathcal R_j]\over z-\xi_{n,k-q}-\lambda_j}
 -\sum_{n\in\mathrm{occ},\lambda_j<0}
 {\mathcal C_{nkq}[\mathcal R_j]\over z-\xi_{n,k-q}-\lambda_j}
 \right].
 \tag{14.10}
\]

No same-q positive/negative pairing, static antiunitary conjugation or
entrywise Padé fit is used. Apply (14.10) separately to each directional
and body spectral term, with the actual signed coefficients in (14.4).
The full nonorthogonal AO/KS comparison uses C-dagger S C=I and complete
metric-unitary frame transport; dropping off-diagonal spin blocks is not
an allowed simplification.

### 14.4 Separate operator, inverse, contraction and forward errors

For fixed G, vertices and numerical transform matrices, the finite native
map from sampled W(iw) to Sigma(iw) is linear. Therefore the old constant
can be diagnosed without assigning a continuum contact prescription:

\[
 \mathcal L_{\rm grid}[W_{c,\mathrm{old}}]
 =\mathcal L_{\rm grid}[W_{c,\mathrm{corr}}]
  +\mathcal L_{\rm grid}[K_\infty].
 \tag{14.11}
\]

The last term is propagated through the unchanged inverse C/S weights,
native-sign G, RI contraction and forward C/S weights. It is explicitly a
discrete artifact, not an independently justified physical exchange term.
The staged controls retain native sampled W, the positive-time maps actually
returned by CT_FT_Wc_freq_q, all four raw Sigma spin blocks and the final
complete KS matrices. These real-parent controls do not capture an independent
negative-time map: the actual LibRI consumer constructs its negative cache
by entrywise conjugation at the same R. The independent signed spectral
reference verifies W(-t,R)=W(+t,R)* before this cache law is used or an exact
time payload is injected. Both signs still enter the complete Sigma reference;
the magnetic option53 case has unequal positive/negative time matrices and
tests that distinction. Bare conjugation in an arbitrary complex active
auxiliary gauge is not established by this real-parent cache contract.

The raw real-space/time Sigma files contain additive rank-local LibRI
contributions before collection. They must be summed, not deduplicated as
replicas; the collected real-space/frequency files have unique output owners.
Compare same-grid contraction separately from exact-time injection and the
full native path. Matching a contraction cannot establish transform accuracy;
matching a fitted W at its sampled nodes cannot establish its time values.

Frozen gates are2e-11 for complete W,2e-12Ha for the new same-grid contraction
controls and2e-8Ha for absolute Sigma on actual input. Earlier stricter finite
fixtures retain their own gates. Existing16/default-zero and24/explicit1e-12
settings are retained as controls, without choosing a new numerical default.
Report the same-input operator inverse and the exact-Lehmann/native-frequency
comparison separately at their retained W gates: finite response-transform
error can fail the latter even when the former agrees to roundoff. A passed
Sigma gate must not retroactively mark that frequency comparison passed.
Full Gamma, corrected actual SOC3k and genuine magnetic full/IBZ periodic
coverage remain distinct requirements; a passing small Gamma oracle is not
the completion of this stage or a material topology result.

For a fixed input, the
[telescoping diagnostic](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/optical-generalization/absolute-sigma/error-decomposition.py)
defines F_G as the forward transform after contracting the supplied signed
time interaction with the same native-sign G. Let I_grid be the independent
inverse transform, W_N the captured native sampled frequency interaction,
T_N the consumed native time interaction, and W_E,T_E the exact spectral
frequency/time samples. All quantities below are complete matrices in the
same KS frame. The error identity is

\[
\begin{aligned}
 \Sigma_N-\Sigma_E={}&[\Sigma_N-F_G(T_N)]\\
 &+[F_G(T_N)-F_G(I_{\rm grid}W_N)]\\
 &+[F_G(I_{\rm grid}W_N)-F_G(I_{\rm grid}W_E)]\\
 &+[F_G(I_{\rm grid}W_E)-F_G(T_E)]\\
 &+[F_G(T_E)-\Sigma_E].
\end{aligned}
\tag{14.11a}
\]

These five differences distinguish contraction/forward implementation,
native versus independent inverse evaluation on the same samples,
propagated response-fit error, inverse-transform approximation on exact
frequency samples, and forward-transform approximation on exact time
samples. The second term includes numerical summation/evaluation order;
it is not a new physical contribution. Maxima of different terms need not
occur at the same matrix entry. Their sum is exact before taking norms;
adding their norms supplies only an upper bound, not percentages of a
physical uncertainty. On the corrected actual SOC3k input, their maxima
over one/four ranks are respectively5.57e-17,4.50e-13,2.59e-11,3.21e-11
and7.72e-10Ha. The forward approximation is the largest term in this fixed
comparison. This observation selects no universal grid or regularizer.

### 14.5 A normalization comparison is a numerical-policy decision

After matched bare subtraction, let W_U denote the existing unnormalized
Gamma dynamical interaction and W_B its regular-body term. With s=sumrho>0,
the existing normalized rule gives the exact fixed-operator relation

\[
 W_N=W_B+{W_U-W_B\over s}.
 \tag{14.12}
\]

All other q points remain unchanged. For fixed G and vertices the same
linearity applies to the Gamma self-energy contributions:

\[
 \Sigma_N-\Sigma_U=
 (s^{-1}-1)(\Sigma_{\Gamma,U}-\Sigma_{\Gamma,B}).
 \tag{14.13}
\]

This comparison needs no new angular eigensolve. It does change the
dynamical residues and is separate from the zero-response repair. Positive
normalized weights form a positive sum of the exact directional spectral
measures, eliminating the signed body remainder. This guarantees the intended
finite Lehmann model's spectral sign, not the accuracy of its subsequent
numerical transforms or convergence to a continuum material calculation.

For the matched actual SOC3k input, s=1.0009410009946196. The preserved
fixed-rule spectrum contains25 positive-body pole groups with negative
total residue at2e-12Ha coincidence resolution; the worst minimum eigenvalue
is-3.9030e-4 in native residue units. A direct body-resolvent contour and the
cached directional spectra independently verify it. This is failed spectral
positivity/passivity, with no upper-half-plane pole implied.

The [cached normalization comparison](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/optical-generalization/absolute-sigma/reference/actual-streamed/normalized-counterfactual.json)
changes a complete Matsubara self-energy entry by at most8.8573335e-6Ha,
or0.2410203meV, at the first sampled frequency and Gamma band8 diagonal.
It is a fixed-input matrix difference, not a quasiparticle level, band-gap
or material error. Normalizing the existing weights is therefore a concrete
numerical-policy alternative requiring researcher selection before any
default change. The source default remains unchanged during these controls.

An [external native flag comparison](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/optical-generalization/absolute-sigma/normalization-native-check.json)
now realizes that same alternative without changing the production default.
The complete normalized pole-Sigma error is7.6997e-10Ha, while all-spin
same-grid contraction agrees within5.57e-17Ha. Native DeltaSigma differs
from the continuum-pole DeltaSigma by2.01857e-12Ha, retaining a failed strict
2e-12Ha target. Independent fixed-grid propagation of the same sampled
DeltaW instead agrees within2.93e-14Ha at2e-12Ha. Thus the method-comparison
formula and its finite-transform implementation are tested separately from
continuum-difference accuracy; the latter is not silently made a contraction
pass. The original W-response frequency failure also remains failed.

### 14.6 Finite quadrature covariance and representative-sector restoration

For the complete signed-time AO contraction write X=(X_plus,X_minus),
where each component contains all spin blocks and spatial sectors. Let
Pi_ir select the actual native irreducible spatial sectors and R_G restore
them using their recorded orbital, spin and antiunitary operations. This
is a representative-selection and restoration map, not a group average.
In particular, its behavior on noncovariant data must not be inferred from
an orthogonal symmetry projection. Let F include the forward C/S transform,
real-space Fourier sum and metric-correct KS transformation. The two finite
targets are F[X] and F[R_G Pi_ir X], with the exact difference

\[
 \Delta_{\rm restore}
 =\mathcal F[(\mathcal R_G\Pi_{\rm ir}-I)X].
 \tag{14.14}
\]

Antiunitary restoration is real-linear and acts on both signed-time spinor
matrices before F. Replacing it with entrywise conjugation of a single
positive-frequency Sigma would test a different equation. The native
GW wrapper and restore_symmetry_spinor_rspace_blocks supply the actual
sector/member map needed for an independent replay.

With symmetry-compatible G, vertices and W, covariance gives
R_G Pi_ir X=X. A finite angular quadrature need not have that covariance,
even when every directional dielectric inverse is exact and the physical
input has the symmetry. Thus a complete finite full-grid contraction and
its representative-restored counterpart can differ without a faulty
matrix contraction. Checking (14.14) separates that discrepancy from an
incorrect restore kernel. Agreement with the restored target does not
retroactively pass the original direct full-sum gate; numerical covariance
and the implementation of restoration remain different requirements.

The [periodic discriminator](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/optical-generalization/absolute-sigma/reference/periodic-angular-covariance.json)
has pointwise C3 inverse error7.29e-17 and rotated-rule identity error
1.74e-17, while the original finite5810 rule has C3 frequency defect
6.5767e-10 and positive-time defect3.4386e-10. Exact-time injection retains
the IBZ direct full-sum Sigma discrepancy. Independently selecting21 sectors
and restoring81 members reproduces native observed/injected IBZ outputs
within1.40e-17Ha at one/four ranks. This tests (14.14) without replacing
the failed full-sum target: the finite angular rule supplies noncovariant
input to a covariance-assuming sector reduction. It is not evidence of a
new spinor contraction or restoration-kernel defect.

Normalization and angular covariance are also separate. For a symmetry
transport T_g define D_g(W)=T_g(W)-W, using its correct signed-time or
dynamic-frequency law. If the regular body is invariant, equation(14.12)
implies

\[
 D_g(W_N)=s^{-1}D_g(W_U).
 \tag{14.15}
\]

Hence normalizing a noncovariant angular rule only rescales its defect;
it does not remove it. Positive spectral weights and covariance under the
material's point group are different properties of a numerical quadrature.

## 15. A nonvacuous complex active-space reality test

The [nontrivial-map control](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/optical-generalization/absolute-sigma/nontrivial-active-map/README.md)
retains the physical input of the preceding periodic control. Its purpose
is to test the order of active-space unfolding and real-space conjugation,
which the old commuting Householder map could not discriminate.

Let U(q)U(q)-dagger=I and U(q)-dagger U(q)=P, with the same real retained
projector P at q and -q. The parent interaction has support in P and obeys
Wp_minus(q)=Wp_plus(-q)*. For Wa=U Wp U-dagger define

\[
 J(q)=U(q)U(-q)^T.
 \tag{15.1}
\]

Insertion of the two retained projectors gives

\[
 W_a(q,-t)=J(q)W_a(-q,+t)^*J(q)^\dagger,
 \quad J(q)J(q)^\dagger=I,
 \quad J(-q)J(q)^*=I.
 \tag{15.2}
\]

Bare conjugation in active real-space coordinates omits this sewing. It
can accidentally be correct for a particular W even when J is nontrivial;
the witness must compare the complete actual matrices. The production
parent-space cache instead unfolds each signed operator first, so its
real-basis conjugation law is unchanged. Equations(15.1–15.2) assume this
exact isometric/common-real-subspace control, not arbitrary lossy projection.

An explicit compatible map uses U(q)=G(q)U0, where U0 is the original real
onsite(.8,.6) compression. At q0=(0,1/3,0), Pm K fixes q0 and Pm swaps
sites1/2. Let h=(1,1,1)/sqrt3, e=(2,-1,-1)/sqrt6 and o=(0,1,-1)/sqrt2.
Then

\[
 M=eo^T+oe^T,\quad G_0=e^{i\theta M},\quad
 P_mMP_m=-M,\quad G_0h=h,\quad P_mG_0^*P_m=G_0.
 \tag{15.3}
\]

Propagate G0 by the actual C3 permutation on that three-point star and
use I on the opposite star, Gamma and the K pair. Distinct opposite stars
allow G(-q) different from G(q)* without breaking the magnetic group.
The active Coulomb .85 I+.15 ones*ones-transpose commutes with this neutral
rotation. The projected-chi route leaves parent Cs and the physical
Hamiltonian unchanged. Bloch r_atom=0 makes reciprocal-shift sewing I in
this fixture; actual registry keys remain required. Complete-operation,
signed-operator and native controls are necessary in addition to the
generator algebra, with the old failed nonvacuity result preserved.

The complete independent prototype now checks all actual operations and
stabilizers, reciprocal sewing and both signed chi/W branches. Correct
parent unfolding and fixed-grid Sigma invariance agree within6.68e-17
and2.61e-18Ha. The omitted-J active real-space conjugation differs by
1.27841e-5, above the unchanged1e-6 discriminator; its wrong parent Sigma
changes by1.14289e-6Ha. These are explicit wrong-route counterfactuals.
The actual source already unfolds before conjugating. The prototype closes
the old witness's algebraic vacuity. The subsequent
[native full/IBZ control](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/optical-generalization/absolute-sigma/nontrivial-active-map/native-controls/verification.json)
now separately checks actual U payloads and all six operation laws at
one/four ranks. Captured-chi inversion agrees within8.201e-15 and the
active-to-parent time interface within6.073e-15. The actual consumed-W
wrong-active-cache witness is1.278404e-5. Complete absolute Sigma is within
1.3262e-10Ha, and the separate representative-restored target within
1.391e-17Ha. The direct full-sum IBZ discrepancy3.1331e-11Ha and strict
exact-response W failure4.320e-11 remain failed at2e-12Ha and2e-11,
respectively. Positive parent time is captured; the effective negative
cache is source-derived and independently checked, not a second observed
negative-frequency solve. No source/default change or generic lossy-map
acceptance follows from this exact retained-subspace control.
