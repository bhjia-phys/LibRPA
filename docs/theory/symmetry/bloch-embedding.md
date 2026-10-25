# Atomic Bloch embedding through localized GW contractions

This note specifies a change of representation, not a new GW approximation.
It accompanies the SOC/magnetic audit. The accepted EXX controls and the
subsequent, initially unvalidated full-GW integration are distinguished in
the research evidence; this derivation alone is not numerical acceptance.
The tested native Bloch sign is s=-1. Changing the independent Fourier-sign
convention is not accomplished by changing the atomic embedding below.

## Raw reciprocal inputs and localized real-space vertices

Let tau_I be the fractional center of every orbital on atom I. For the
declared embedding eta, define the diagonal unitary matrices

\[
 B_w(k)_{a a}=\exp[-2\pi i s\eta k\cdot\tau_{I(a)}],\qquad
 B_p(q)_{\mu\mu}=\exp[-2\pi i s\eta q\cdot\tau_{I(\mu)}].
\]

Here w denotes electronic AOs, p the parent auxiliary basis, and a an active
auxiliary basis. Fractional k and q must be used in these exponentials.
The internal Cartesian reciprocal coordinate is converted with the lattice
before forming the dot product. Each actual target label, including its
integer reciprocal shift, defines its own B.

For a cell-frame input, the equivalent declared raw input is

\[
 C_\eta(k)=B_w(k)C_0(k),\quad
 V_\eta(q)=B_p(q)V_0(q)B_p(q)^\dagger,\quad
 U_\eta(q)=B_a(q)U_0(q)B_p(q)^\dagger.
\]

The localized RI coefficients and H(R), S(R), r(R) remain in their existing
cell convention. A nonorthogonal electronic metric obeys
S_eta=B_w S_0 B_w^dagger, so C_eta^dagger S_eta C_eta=I. Consequently the
state-basis matrix of a covariant AO operator matrix
O_ab=<phi_a|O|phi_b>, such as H, Sigma or an RI charge vertex, is
C_eta^dagger O_eta C_eta: an extra metric must not be inserted in this
projection. The spectral Green/density coefficient matrices below are a
different object; their covariant AO matrix contains a metric on each side.

## Electronic propagators and the response

At fixed k and either time sign, a spectral propagator has the form
G_eta=C_eta g(E,f,t) C_eta^dagger=B_w G_0 B_w^dagger. Density matrices obey
the same law. Remove B in the existing per-k work buffer before applying
the unchanged native cell Fourier transform and localized RI contractions:

\[
 G_0(k,t)=B_w(k)^\dagger G_\eta(k,t)B_w(k).
\]

In particular, for G=C g C^dagger the covariant operator matrix is S G S,
and C^dagger S G S C=g. This distinction changes the metric factors in a
state projection, but both kinds of matrix obey the same B congruence and
therefore the same phase conversion used here.

Raw coefficients remain unchanged. For restored members, rotate and sew the
raw coefficients/propagator first, then dephase at the actual member k.
Dephasing at the representative before using a raw symmetry representation
would change that representation without changing the restoration operator.

The localized contraction produces the cell parent response chi_0,p(q,iw).
Before pairing it with raw V_eta, embed it into the raw parent frame:

\[
 \chi_{\eta,p}=B_p\chi_{0,p}B_p^\dagger,\qquad
 \chi_{\eta,a}=U_\eta\chi_{\eta,p}U_\eta^\dagger
              =B_a(U_0\chi_{0,p}U_0^\dagger)B_a^\dagger.
\]

Thus parent embedding precedes compression. The active result must not be
embedded a second time. Independent signed endpoint/transition Gram matrices
obey the same bilinear law. This does not make a direct active-basis RI refit
equivalent to projecting the parent vertices with a lossy U.

## Screened interaction and the return to real space

Screening is covariant under this unitary representation change. In either
parent or active coordinates, W_eta=B W_0 B^dagger and the same holds for
Wc. Parent unfolding is

\[
 W_{\eta,p}^{c}=U_\eta^\dagger W_{\eta,a}^{c}U_\eta
              =B_p(U_0^\dagger W_{0,a}^{c}U_0)B_p^\dagger.
\]

Keep raw frames through screening, unfolding, raw crystal restoration and
reciprocal sewing. Only then form the physical cell Fourier sum, with the
existing native W convention:

\[
 W_{0}^{c}(R,i\omega)=\frac1{N_q}\sum_{q\in\mathrm{BZ}}
 e^{-2\pi i q_{\mathrm{frac}}\cdot R}
 B_p(q)^\dagger W_{\eta,p}^{c}(q,i\omega)B_p(q).
\]

A legacy inversion-only reconstruction must dephase the supplied source
before applying the real-parent conjugation law. A bare conjugation of raw
embedded matrices is not generally that law. Likewise, a real-parent
negative-time cache must be constructed after parent unfolding and cell
conversion; arbitrary complex active coordinates do not inherit elementwise
real-basis cache identities.

The dense BLACS restoration has a separate storage requirement. ScaLAPACK
reads column-major local tiles. Row-major representatives are copied and
converted before multiplication; restored members return to the supplied
layout before flat Fourier packing. This preserves the matrix elements and
raw inputs, including on ranks with zero local elements. It is not an
additional symmetry or Hermitian completion.

## Correlation self-energy and optical vertices

The localized G/W/Cs contraction constructs cell Sigma(R,t). After its
unchanged time/frequency and R/k transforms, embed the AO bilinear at the
actual requested mesh or band target:

\[
 \Sigma_\eta(k,z)=B_w(k)\Sigma_0(k,z)B_w(k)^\dagger,\qquad
 C_\eta^\dagger\Sigma_\eta C_\eta=C_0^\dagger\Sigma_0 C_0.
\]

AO export uses Sigma_eta; KS projection uses raw C_eta. Applying B only after
KS projection cannot repair a mismatched contraction. The two spinor legs
share the atomic electronic phase, without changing the established spin
ordering or occupation normalization.

For a Gamma optical RI vertex, B_p(0)=I but electronic k need not vanish.
The AO work matrix therefore requires

\[
 \Gamma_{\eta,\mu}(k)=B_w(k)\Gamma_{0,\mu}(k)B_w(k)^\dagger.
\]

Its raw-C projection is invariant. Prefourier/restored vertex routines must
receive the actual target k explicitly; a source coefficient index does not
identify that target's Bloch embedding. Both independent wings retain their
existing signed-frequency definition.

### Direct optical coefficients at a reciprocal alias

A direct full-BZ optical packet contains already supplied member states and
their band-basis velocity. It need not store them at the coordinate used by
the RI Fourier work. Let its actual source coordinate be kappa and the
actual Fourier target be k_t. For reciprocal-equivalent points, convert the
electronic AO coordinates by

\[
 T_w(k_t,\kappa)=B_w(k_t)B_w(\kappa)^\dagger,\qquad
 C_t=T_w C_{\rm source}.
\]

The native coefficient storage has band rows W=C^T, so this is a right
diagonal multiplication, with no conjugation of its diagonal element:

\[
 W_t(n,a)=W_{\rm source}(n,a)
 \exp[-2\pi i s\eta(k_t-\kappa)\cdot\tau_{I(a)}].
\]

The same factor multiplies both SOC components and every scalar spin
channel. If kappa=k_t+G it is exp(+2pi i s eta G.tau_I). For eta=0 or equal
coordinates it is identity. The common AO-coordinate conversion does not
change the supplied band-basis velocity; any independent band gauge remains
in both its coefficients and velocity. No additional spatial/spin rotation
or antiunitary conjugation is applied to an already supplied direct member.

This conversion is required even though the source matcher identifies k and
k+G as the same mesh point. Indeed, for an unchanged cell-frame state C0,

\[
 C_t^\dagger\Gamma_\eta(k_t)C_t
 =C_{\rm source}^\dagger B_w(\kappa)\Gamma_0(k_t)
   B_w(\kappa)^\dagger C_{\rm source}
 =C_0^\dagger\Gamma_0(k_t)C_0.
\]

Using raw C_source directly with Gamma_eta(k_t) leaves the relative atomic
phase inside the transition vertex. The head can remain unchanged because
it uses the supplied band velocity, while both independently ordered wings
change. Converting only to a stored member label is also insufficient when
the actual Fourier target is a different reciprocal alias.

The immutable input must therefore retain its actual source-coordinate
table and member-to-source indices. Cell-frame legacy packets need no AO
rephase; an active atomic embedding cannot infer a missing source table.
The existing coordinate matching contract is 1e-5 in both the current driver
and the core mapping helper. That admission criterion and the phase are
different operations: the phase uses the actual target-minus-source
difference, without rounding away an admitted residual. Coordinate matching
does not independently establish that states evaluated at physically
different k points are identical; producer-coordinate precision is a
separate input question.

Dense identity access can borrow the original const component; nonidentity
access needs only a current scratch component. With k-BLACS input, localize
once and phase only the owned AO columns, using the descriptor's global AO
indices. Neither path requires another full-BZ coefficient/velocity payload,
changes raw input, or supplies QSGW current-frame transport by itself. Native
alias controls, source ownership and current QSGW/checkpoint transport remain
separately scoped acceptance questions.

Native velocity instead consumes localized cell H/S/r and their existing
derivatives. Dephase the existing coefficient transfer/ket work buffer with
B_w^dagger before that construction, for both spinor components. Do not
transform the absolute position operator as if it were an ordinary bounded
bilinear; its affine real-space identity is a separate contract.

## Numerical acceptance and remaining boundaries

Equivalent eta=0,+1,-1 raw inputs must give the same physical cell G, chi,
W and Sigma and the same complete KS Sigma; declared AO outputs must satisfy
the corresponding B congruence. Tests require nontrivial atomic centers,
signed magnetic response, exact nontrivial q-dependent U, full and reduced
grids, reordered/shifted band targets, MPI ownership and input immutability.
Independent absolute spectral, same-consumed-operator and captured-response
inversion errors are distinct observables.

A finite angular rule can itself fail crystal covariance. Agreement with
representative-selected/restored data then does not establish equality to
the direct finite full-BZ sum. Keep both targets and their original gates.
Neither representation covariance nor a successful build establishes
quadrature, continuation, material convergence or a topological invariant.
Cache mutation, QSGW/checkpoint consumers, cRPA and the opposite Bloch sign
require their own coherent tests beyond the present fixed-input stage.

## Distributed optical reconstruction from IBZ representatives

The previous source explicitly rejected an IBZ-only SOC optical context
when its electronic coefficients were k-BLACS distributed. Precomputing
dense optical tensors before distributing the body coefficients does not
test this missing route. The new integration keeps the raw representative
coefficient tiles distributed and applies the same ket transformation as
the independently checked dense reconstruction. Numerical acceptance of
this integration is separate from the earlier dense optical controls.

In native storage C_sigma is band-by-AO and row-major. Its flat buffer is
also the AO-by-band, column-major matrix X_sigma=C_sigma^T described by the
wavefunction BLACS descriptor. Let O be the same orbital matrix used by
the dense spinor helper, excluding the final target reciprocal gauge. Then

\[
 C_\sigma^{\rm spatial}=C_\sigma O^T,
 \qquad X_\sigma^{\rm spatial}=O X_\sigma.
\]

The spatial multiplication uses distributed PGEMM(N,N). If U is the
resolved spin action of the member, form both local spin components before
overwriting either one:

\[
 Y_\uparrow=U_{00}X_\uparrow^{\rm spatial}
              +U_{01}X_\downarrow^{\rm spatial},\qquad
 Y_\downarrow=U_{10}X_\uparrow^{\rm spatial}
              +U_{11}X_\downarrow^{\rm spatial}.
\]

For a unitary member X'_sigma=D Y_sigma. For an antiunitary member the
same dense-helper convention is

\[
 X'_\uparrow=D\,\overline{Y_\downarrow},\qquad
 X'_\downarrow=-D\,\overline{Y_\uparrow}.
\]

D is evaluated at the actual target, including its reciprocal shift, and
multiplies each local AO row after conjugation. Moving D inside Theta would
conjugate the target sewing phase incorrectly. The restored raw tiles
then enter the unchanged raw optical vertex and KS projection.

A scalar component obeys the same ket law with no spin mixing:

\[
 X'=D OX\quad\hbox{(unitary)},\qquad
 X'=D\,\overline{OX}\quad\hbox{(antiunitary)}.
\]

For real diagonal occupations f, its coefficient projector therefore obeys

\[
 P'=D\overline O\,\overline P\,O^T D^\dagger,
 \qquad P=X f X^\dagger,
\]

exactly the antiunitary operator law. In band-by-AO storage this means
C'=C O^T D or C'=conj(C O^T)D. Both scalar and SOC spatial products use
PGEMM(N,N); the scalar output is conjugated locally after that product,
and the actual-target D is applied last.

The inherited dense scalar helper instead formed conj(C)R. Its R builder
contains the unconjugated orbital block and a conjugated target phase for
an antiunitary member. That builder is not a ready-to-apply scalar ket
matrix: the helper missed conjugation of the spatial block and placed the
target gauge inside the wrong conjugation. The first distributed candidate
copied that convention. A nonorthogonal two-AO physical TR model with a
folded +2/3 partner reproduces the same complete wrong head/wing arrays in
both dense and distributed routes. Head remains correct; eta=+1 right and
left-adjoint wing errors are0.05154850 and0.09965772 at the unchanged2e-12
gate. Full-grid and eta=0 controls pass. Its wrong-D-dagger independent
response agrees with the failed arrays within4.45e-16. The scalar function
is byte-identical in base52b43609; its first introduction is unassigned.
The scalar ket repair changes these two consumers, not the general rotation
builder or the numerical prescription. The old source, failed controls and
synthetic test's earlier incorrect complex-block expectation are preserved.

The existing AO rotation builder returns a row-major dense matrix. Pack it
element by element into a column-major root descriptor before distributing
it to the owning group's AO-by-AO matrix. The source WFC tiles already have
the required column-major interpretation and must not be transposed again.
Use a separate output tile for the spatial product, including valid dummy
storage on empty owners. No full representative or full-BZ WFC copy is
introduced. A temporary dense AO-squared rotation still exists on each
owning group root; neither a universal memory reduction nor a speedup is
established by this design.

Different k groups can own different numbers of representatives and star
members. Shape checks, AO distribution, spatial multiplication and paired
spin restoration therefore communicate within the owning k-BLACS group.
After all local groups finish, a common world failure handoff precedes the
existing global head, wing or endpoint reduction. A world collective inside
an owned-member loop can deadlock even for otherwise correct formulas.
Head accumulation likewise hands off a root-only reconstruction error before
the global reduction; any legacy owner lookups precede that independent
accumulation stage. Supplied direct full-BZ physical inputs retain their
separate selection and are not replaced by representative restoration.
