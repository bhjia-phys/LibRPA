# Signed-time support and spinor self-energy covariance

The SOC self-energy consumer must preserve the complex operator, its signed-time
Fourier transform and its transformation law. This note fixes the support error
in the space-time transform without changing the screened interaction, quadrature,
continuation or fixed-reference QSGW prescription. The accompanying control is
`test_magnetic_gw_model --sparse-time`.

## Spin indices and the operator being contracted

For spin-independent Coulomb vertices, exchange and correlation retain all four
spinor blocks. In continuous notation their structure is

\[
 \Sigma^x_{st}(r,r')=-v(r,r')\rho_{st}(r,r'),\qquad
 \Sigma^c_{st}(r,r';\tau)=-G_{st}(r,r';\tau)W_c(r,r';\tau).
\]

There is no spin trace on the external self-energy indices. Charge screening,
by contrast, traces the two propagators at the charge vertices. The real AO
product approximation replaces the products by RI vertices; it does not remove
complex spin off-diagonals. The current `MeanField` signed-time convention uses
`G_lib(+tau)=+C(1-f)exp[-(epsilon-mu)tau]C^dagger` and
`G_lib(-tau)=-Cf exp[(epsilon-mu)tau]C^dagger`, opposite to the usual
Matsubara Green function. Consequently the LibRI product of `G_lib` and `Wc`
contains the self-energy sign already. An extra global minus sign here would
change the implemented equation.

The Bloch-basis projection is

\[
 \Sigma^{\rm KS}_{mn}(k,z)=
 \sum_{abst} C^*_{as,m}(k)\Sigma^{\rm AO}_{as,bt}(k,z)C_{bt,n}(k).
\]

The AO coefficients already carry the covariant/contravariant pairing appropriate
to the supplied basis. Inserting a second overlap matrix into this projection
would be a separate convention error.

## The two signed-time supports must be united

Write the actual LibRI outputs at a positive time node as `A_t=Sigma(+tau_t)`
and `B_t=Sigma(-tau_t)`. With the code's cosine and sine transform weights,

\[
 \Sigma(i\omega_j)=\sum_t\left[
 {A_t+B_t\over2}c_{jt}
 +i{A_t-B_t\over2}s_{jt}\right].
\]

This is linear in each signed-time operand. In particular, `B_t=0` gives
`A_t(c_jt+i s_jt)/2`, not zero. The same holds with only `B_t` present.
Sparse storage is only a representation: the domain of this operation is the
**union** of the stored atom-pair/cell keys. An absent key denotes a zero tensor
of the same dimensions as its present partner.

Before the repair, `G0W0::build_spacetime` visited only positive-time keys and
continued past a key whenever its negative-time partner was absent. Independently
filtered occupied and empty propagators can have different support, even in exact
arithmetic. This discarded genuine self-energy contributions; lowering a threshold
does not repair an exactly zero complementary propagator.

The repaired loop constructs the union of keys, uses the original tensors when
both exist and allocates a zero tensor only for a missing partner. It neither
changes a present block nor fills a missing atom/cell by a symmetry-related value.
The common loop serves real and complex tensors, and both unreduced output and
jointly restored four-channel SOC output.

## Independent finite consumer control

Take one spatial orbital with two spin states, three k points, energies
`(-1,+1) Ha`, chemical potential zero and exactly one occupied state. Then

\[
 G_{\rm lib}(+\tau)=e^{-\tau}|\downarrow\rangle\langle\downarrow|,
 \quad
 G_{\rm lib}(-\tau)=-e^{-\tau}|\uparrow\rangle\langle\uparrow|.
\]

Supply a scalar causal screened interaction
`Wc(i nu)=-4/(nu^2+4)` identical at all q, and a local RI coefficient `1/2`
(the two equivalent centers produce a unit charge vertex). This tests the Sigma
consumer with external screening; this W is not asserted to be the RPA response
of the one-orbital reference. Its finite-quadrature oracle is evaluated directly
from the scalar W samples and the analytic occupied/empty exponentials. The two
spin diagonal self-energies are nonzero, while their complementary signed-time
supports are absent.

The magnetic operations `E`, `C2z`, `Theta Mx`, `Theta My` preserve this polarized
reference. The symmetry-enabled case asserts that its real-space sector is
actually reduced. The test runs both reduced and unreduced consumers and compares
the complete complex projected matrices at every original frequency and k point.
It is exercised by the existing serial and four-rank magnetic model targets.
This controls the support/transform defect and MPI accumulation, not the accuracy
of minimax quadrature or a material quasiparticle energy.

## Static and dynamic covariance are different statements

Let `A=UK` be an antiunitary symmetry from k to k'. In an orthonormal frame,

\[
 \Sigma(k',z)=U\Sigma(k,z^*)^*U^\dagger.
\]

Using Hermitian analyticity, `Sigma(k,z*)=Sigma(k,z)^dagger`, this becomes

\[
 \Sigma(k',z)=U\Sigma(k,z)^T U^\dagger.
\]

Thus a raw positive-imaginary-frequency self-energy must not be tested using the
static-Hamiltonian expression `U Sigma(k,+i omega)^* U^dagger`. That expression
also changes the frequency. In signed imaginary time, the spectral weights are
real and the antiunitary operation does act by complex conjugation. This is why
the implementation restores all four spin blocks jointly **before** multiplying
the sine part by `i`. Applying plain complex conjugation to the already combined
positive-frequency matrix would conjugate this `i` incorrectly. AO metrics and
KS sewing matrices must be included when changing the representation of U.

## Boundaries not repaired by the support change

The current fixed-reference QSGW task projects onto the immutable initial KS
frame. Its retained Mode-B construction samples the diagonal at live sorted
energies while the off-diagonal Hermitian part is sampled at the Fermi level.
That declared prescription is not replaced here by eigenframe staticization.
Independent scalar rational fits also need not commute with a dense change of
basis. Finally, upper-triangle Hermitian completion can convert an existing
anti-Hermitian input error into a symmetry defect; it is harmless only when
input Hermiticity already holds to the required accuracy. These separate issues
require their own operator-level controls and do not follow from the sparse-time
repair or from agreement of reduced and unreduced contractions.

## Empty distributed screening tiles

A process grid can give a rank zero local rows or columns of W even though the
global auxiliary basis is nonempty. The complex W frequency-to-time batching
must not divide its workspace limit by that local zero size. It now retains
shaped empty tau/R blocks and joins the common final barrier, as the real-storage
path already did. The one-orbital four-rank consumer control tests this case
without weakening the complete-matrix oracle.

On the fresh repair build the signed-time control's maximum error is
5.59432e-17 Ha for both symmetry settings and both one/four-rank executions;
the oracle norm is 0.332272 Ha. The old unreduced consumer loses this whole
signal. The old reduced spinor restore happens to materialize the missing
zero channels and therefore masks the defect in this particular model.

## What sixteen-node continuation does and does not establish

A bounded audit of the actual scalar Padé and complete static-potential builder
uses 16 positive imaginary nodes, 16 parameters and no resampling. The GreenX
nodes come from transition range 0.03–30 Ha; these are model frequencies, not
recovered material samples. The exact reference is a causal matrix function

\[
 \Sigma(z)=\sum_p{A_p\over z-\xi_p},\qquad
 A_p=A_p^\dagger>0,\quad A_p=J A_p^T J^\dagger,
 \quad J=I_2\otimes i\sigma_y.
\]

For two isolated Kramers pairs, take energies `(-0.2,-0.2,+0.2,+0.2) Ha`
and `U=U1 direct-sum U2` with independent U(2) pair rotations. Since U commutes
with these energies, this is a valid KS gauge comparison. Using different
frames at +k and -k produces the sewing matrix `B=U^dagger J`. Exact input obeys
`Sigma_-(z)=B Sigma_+(z)^T B^dagger`.

One- and two-pole positive controls preserve this relation to about 1e-16 Ha.
A twenty-pole input with its nearest poles at ±0.28 Ha gives a continued-matrix
sewing residual of 2.03545e-4 Ha at targets 0 and ±0.2 Ha. Thus finite scalar
continuation can remain noncovariant with sixteen samples. However, the complete
**Mode-B static sewing residual is only 9.34929e-10 Ha in this model**, while its
absolute error against the exact Mode-B function is 6.20674e-4 Ha. The much
better off-diagonal evaluation at zero suppresses the symmetry error without
fixing the on-shell diagonal error. Neither number estimates the material error.

Each frame's local Kramers relation remains accurate to at worst 5.01e-13 Ha
for the continued function. At an isolated TRIM doublet the sewing matrix is
a phase times `i sigma_y`; entrywise fitting can preserve this sparse relation
while violating a dense relation between different k-point frames. The two
checks must not be conflated.

### Diagonal Mode B versus degenerate-subspace covariance

For a fourfold degenerate energy e, Mode B has the exact-function form

\[
 V_B=\Sigma_H(0)+D[\Sigma_H(e)-\Sigma_H(0)],
\]

where D selects the coordinate diagonal. D does not commute with an arbitrary
rotation within that four-dimensional degenerate subspace. An exact one-pole
control gives a covariance difference of 2.40426e-3 Ha, despite continuation
errors below 9e-17 Ha; equal-energy Mode A is covariant. This degeneracy is a
separate case from two isolated Kramers pairs. In a Kramers pair the exactly
self-dual Hermitian self-energy block is scalar, so exact Mode B commutes with
pairwise frame rotations.

A possible block prescription is

\[
 V_B^{\rm block}=\Sigma_H(0)+
 \sum_g P_g[\Sigma_H(e_g)-\Sigma_H(0)]P_g,
\]

using projectors onto degenerate energy subspaces. It has the desired internal
frame covariance, but adopting it requires a degeneracy convention and changes
the coordinate-diagonal prescription. It is not implemented by this repair.

### Live eigenframes versus fixed-reference energy indices

With live eigenvectors `C_live=C0 U`, current-eigenframe staticization would
construct `U F_E[U^dagger Sigma0 U] U^dagger`. The retained fixed-reference task
constructs `F_E[Sigma0]`. A one-pole, exactly time-reversal-preserving model with
a 0.37-radian mixing of initial nondegenerate pairs and live energies
`(-0.18,-0.18,+0.24,+0.24) Ha` gives a Mode-B difference of 5.74750e-3 Ha; the
continuation errors are below 7.64e-17 Ha. This isolates a method difference
rather than numerical error. Update 1 has the identical initial/live frame.
Changing this prescription requires a separate method decision.

Covariance can alternatively be enforced between k points by fitting a verified
symmetry representative and transporting its matrix, or intrinsically by a
matrix continuation with shared scalar basis functions and covariant coefficient
extraction. The former does not fix absolute fit error or the chosen frame at
the representative. The latter needs new conditioning, pole-selection and
causality controls. Neither is silently substituted for the current method here.

## Truncated Green functions require an ordering-independent subspace check

In an orthonormal AO frame the selected projector and Green function are

\[
 P_M=C D_M C^\dagger,\qquad
 G_M(\tau)=C D_M g_E(\tau) C^\dagger,
 \qquad D_M=\operatorname{diag}(m_n).
\]

Under a joint permutation `C'=C Q`, `E'=Q^dagger E Q`,
`f'=Q^dagger f Q`, and `D_M'=Q^dagger D_M Q`, the physical operators are
unchanged. A multiplet-closure decision must therefore be invariant under that
same permutation. At Gamma the antiunitary condition is
`P_M=J P_M* J^dagger`. More generally it relates the selected projectors at
different k points through their supplied symmetry actions; energy degeneracy
alone is not a complete test of that cross-k condition.

Before this repair, `validate_kstar_band_cutoff_closure` compared neighboring input
indices without imposing an increasing-energy input contract. For two Kramers
pairs with energies (-1,-1,+1,+1), selecting states (0,2) splits both pairs.
After reordering to (0,2,1,3), the same selection has mask (1,1,0,0) and the
only adjacent-index mask boundary has a nonzero energy gap. The old guard accepted
it, despite rejecting the identical projector before permutation. An actual
public-input/pre-repair-library control retains H, density and signed-time G to
5.56e-17 while the split projector has a TR residual of 0.813878 in its complex
spinor frame. A complete Kramers pair passes and has zero projector residual.

The implemented correction compares adjacent **energy-ordered
indices**, using the existing O(N) path when the energies are sorted and a
temporary sorted index vector otherwise. It does not reorder E/C/f or the mask.
The independent post-repair witness rejects the split selection in both orders,
accepts the complete pair in both orders, and preserves all of the physical
matrix residuals above. The existing mean-field numerical test also compares
the four complex G blocks at both tau signs under the same joint permutation.
This correction restores the existing conservative rule against splitting any
degenerate cluster; it does not redefine truncation, certify cross-k sewing,
or establish unsorted-input support in older energy-bound helpers. The original
pre-repair external control and output are preserved separately from the
post-repair witness.

## Extension to a general magnetic screened interaction

The [nonreciprocal-screening derivation](nonreciprocal-screening.md) gives the
coordinated chi/W/Sigma extension, including the source paths, independent
spectral oracles and optical left/right-wing algebra. The essential additional
consumer condition is `Sigma(+/-tau)=G_lib(+/-tau) W(+/-tau)`: the existing
signed-time Sigma transform is correct only when its W input has the matching
sign. Deriving W from a cosine-only response and using it for both signs does
not supply unrestricted magnetic GW. This extension restores the full screening
equations and leaves the selected QSGW staticization as a separate contract.


## Density, nonorthogonal metrics and an independent exchange oracle

Let the AO coefficient matrix have basis functions as rows and states as
columns. For a nonorthogonal AO metric S, normalized states obey

\[
 C^\dagger S C=I,\qquad P=C f C^\dagger,\qquad
 N=\operatorname{Tr}(S P).
\]

Here f includes the k-point weights and the explicit-spin convention; P is the
contravariant density kernel, not a covariant AO operator. A physical AO
operator O projects as C^dagger O C, with **no additional S**. In contrast,
reconstructing its AO covariant matrix from a complete orthonormal state frame
uses S C O_state C^dagger S. Confusing these directions introduces or removes
metric factors. These identities apply componentwise to the full spinor matrix;
LibRPA stores separate band-by-AO arrays for up/down coefficients and assembles
AO operators in AO-major, spinor-minor order.

The active density convention multiplies stored occupations by
`n_spins*n_spinor/2`: unpolarized scalar occupations include both spins, whereas
explicit collinear channels and SOC spinors already use unit occupation per
state. This convention is unchanged by the repair. The existing threshold
`weight < 1e-4/Nk` is likewise unchanged. Its physical effect is to define a
thresholded f, not to define a prefix of the state list.

For a simultaneous state permutation Q, `C'=C Q` and `f'=Q^dagger f Q`,
so P is unchanged. The old density loops instead stopped at the first
below-threshold occupation. Occupations (1,0,1,0), obtained by permuting two
occupied and two empty states, incorrectly retained only one electron. The
repaired serial and three distributed density builders scan every state,
zero below-threshold columns, and retain a GEMM extent ending at the last
above-threshold column. Thus an internal zero occupation no longer removes
later occupied states; the trailing-zero optimization is retained.

A discriminating finite RI model checks more than Hermiticity. For real
symmetric auxiliary vertex matrices Gamma_mu and symmetric positive Coulomb
matrix V, the independent exact exchange is

\[
 K=-\sum_{\mu\nu}V_{\mu\nu}\Gamma_\mu P\Gamma_\nu,
 \qquad K_{\rm state}=C^\dagger K C.
\]

The fixture has two spatial orbitals, two spinor components and two occupied
Kramers states in a complex unitary frame. It uses

\[
 \Gamma_0=\begin{pmatrix}1&0.2\\0.2&0.4\end{pmatrix}\otimes I_2,
 \quad
 \Gamma_1=\begin{pmatrix}0.3&-0.1\\-0.1&0.8\end{pmatrix}\otimes I_2,
 \quad V=\begin{pmatrix}1.1&0.2\\0.2&0.7\end{pmatrix}.
\]

The nonorthogonal counterpart uses
`T=[[1.15,0.25],[0.12,0.9]] tensor I_2`, with

\[
 S'=T^\dagger T,\quad C'=T^{-1}C,\quad
 \Gamma'_\mu=T^\dagger\Gamma_\mu T,\quad
 P'=T^{-1}P T^{-\dagger},\quad K'=T^\dagger K T.
\]

Consequently C'^dagger K' C'=C^dagger K C. The independently constructed
vertex matrices are supplied through the public RI input: the two identical
local centers each carry Gamma/2, with the documented `(i,j,mu)` buffer order.
Native LibRI real-space exchange, its actual KS projection, AO Vxc projection
and `H0-Vxc+K` assembly can then be compared against explicit small matrix
products without treating one production route as the oracle for another.
The original negative control preserves both signed G and the older 2-Ha
transition diagnostics while producing an exchange error as large as
1.25029 Ha in the nonorthogonal AO frame. This isolates occupation selection
from energy-bound selection and from metric/projection conventions. The post-repair
32-case replay at one/four MPI ranks reduces the density error to 5.56e-17,
AO exchange to 4.45e-16 Ha and KS exchange to 3.34e-16 Ha. Electron count is
2 within 4.45e-16. Native Vxc projection and `H0-Vxc+K` assembly agree with the
independent products to 2.78e-17 and 4.45e-16 Ha, respectively.

The numerical fixture toggles the serial/distributed WFC route and symmetry
option, but its sole Gamma point does not test genuine IBZ reduction. Separate
full-zone/reduced-k-star repository regressions exercise the scalar and spinor
restoration paths under joint band permutations. Conversely, neither those
finite controls nor this density repair certify arbitrary unsorted input for
older gap/energy-range/index-prefix helpers. That broader input contract remains
unresolved. The normally sorted archived Bi2Se3 input is not shown to have
triggered this defect, and these model errors are not material error estimates.

## Public wave-function extents are complete matrices, not BLACS tiles

All eight SCF/band-path and scalar/spinor, split/packed wave-function setters
previously allocated `nstates_local*nbasis_local` entries but copied the global
`n_states*n_aos` entries. A smaller declared extent therefore caused memory
corruption before later native redistribution could diagnose its shape.
The setter arguments carry no band or AO offsets, and the native input
redistribution explicitly requires one complete matrix per spin/k owner.
Copying only the smaller buffer would not make a tile representable.

The repair requires positive complete dimensions equal to the initialized mean
field before creating/replacing any coefficient matrix. Unowned k points omit
the setter; native redistribution later creates legitimate empty local tiles.
Validation remains rank-local because the setters may be called only by their
owners. It follows the existing C/C++ runtime-error mechanism; introducing a new
collective into these calls would deadlock valid ownership patterns. This does
not add a new distributed-input ABI or validate arbitrary pointer extents.
