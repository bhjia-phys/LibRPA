# General-frequency strict-2D optical screening

This note tests the strict-2D complete-Wc consumer and its native signed-time
correlation self-energy conjunction in the SOC GW audit. The completed
controls include nonorthogonal AO congruence, exact rectangular auxiliary
compression, and genuine magnetic IBZ restoration in a periodic model.
The [general signed-frequency derivation](nonreciprocal-screening.md) fixes
its charge-response and independent-left/right-wing conventions. The 3D
optical controls do not establish the two-dimensional radial integral or
its Coulomb normalization. This is a finite operator control, not a material
convergence claim.

## The directional operator that must be inverted

Use physical in-plane momentum $q>0$ in inverse Bohr and a real in-plane
unit vector $n=(\cos\theta,\sin\theta,0)$. Let $B(i\omega)$ be the regular
Coulomb-basis dielectric body, $Z_R(i\omega)$ its body-to-head Cartesian
coefficient, and $Z_L(i\omega)$ its independent head-to-body coefficient.
The native optical tensor $H$ includes the identity. The strict-2D
long-wavelength dielectric operator is

$$
E(q,n;i\omega)=\begin{pmatrix}
1+q\,n^T(H-I)n & \sqrt q\,n^TZ_L\\
\sqrt q\,Z_Rn & B
\end{pmatrix}.
$$

A general magnetic response does not justify $B=B^\dagger$ or
$Z_L=Z_R^\dagger$ at a fixed positive imaginary frequency. With
$D=B^{-1}$, define

$$
a(n)=n^T[H-I-Z_LDZ_R]n,\quad
b_R(n)=DZ_Rn,\quad b_L(n)=n^TZ_LD.
$$

Then $\det E=\det B\,[1+qa(n)]$, and block inversion gives

$$
E^{-1}=\begin{pmatrix}
(1+qa)^{-1} & -\sqrt q\,b_L/(1+qa)\\
-\sqrt q\,b_R/(1+qa) & D+q\,b_Rb_L/(1+qa)
\end{pmatrix}.
$$

These formulas restore the existing strict-2D GW equation for general
complex matrices. They introduce neither a new screening approximation nor
a different continuation/staticization method.

## Physical Coulomb legs and producer normalization

Let $L_b$ be the regular Hermitian Coulomb square root in the fixed Coulomb
eigenbasis. In the plane-wave head normalization, the external Coulomb legs
are $L(q)=\operatorname{diag}(\sqrt{2\pi/q},L_b)$. The correlation interaction
is $W_c=L(E^{-1}-I)L$. Its blocks are

$$
(W_c)_{hh}=-\frac{2\pi a}{1+qa},\qquad
(W_c)_{bh}=-\frac{\sqrt{2\pi}L_bb_R}{1+qa},\qquad
(W_c)_{hb}=-\frac{\sqrt{2\pi}b_LL_b}{1+qa},
$$

$$
(W_c)_{bb}=L_b\left[D-I+\frac{q\,b_Rb_L}{1+qa}\right]L_b.
$$

The ABACUS producer defines the l=0 multipole as the radial integral
$m_\mu=\int r^2\psi_{\mu 0}(r)dr$. Since $Y_{00}=1/\sqrt{4\pi}$,
the integrated charge is $Q_\mu=\sqrt{4\pi}m_\mu$, and the metadata is
$M^2=\sum_\mu m_\mu^2$, not $\sum_\mu Q_\mu^2$.
For in-plane primitive area $A$, the native metadata contract therefore gives
the auxiliary head singularity $8\pi^2M^2/(Aq)$, equivalently
$2\pi\sum_\mu Q_\mu^2/(Aq)$. Thus transforming the plane-wave head to the normalized
auxiliary head multiplies each external head leg by
$s=\sqrt{4\pi M^2/A}$. The head block receives $s^2$ and the two wings
receive $s$ independently. The cell height does not enter this normalization.
A supplied height is still needed to define the three-dimensional input
lattice and its reciprocal bookkeeping, with the active integration plane
strictly in xy.

This normalization is bound to the actual producer, rather than fitted to
the screening result: ABACUS
[`get_multipole`](../../../../ABACUS-rpa-soc-fix-20260927/source/source_lcao/module_ri/exx_abfs_ctor_orbs.cpp)
integrates $r^{2+L}\psi_L/(2L+1)$ (line 497 onward), its
[real spherical harmonics](../../../../ABACUS-rpa-soc-fix-20260927/source/source_lcao/module_ri/direct_2d_coulomb.cpp)
use $\sqrt{(2l+1)/(4\pi)}$ (line 139), and the
[RPA exporter](../../../../ABACUS-rpa-soc-fix-20260927/source/source_lcao/module_rpa/rpa_lri.hpp)
uses the l=0 moments of the actual exported auxiliary basis (line 1422).
The [producer metadata formula](../../../../ABACUS-rpa-soc-fix-20260927/source/source_lcao/module_rpa/module_librpa_io/librpa_2d_coulomb_head.h)
and [LibRPA normalization](../../../src/core/dielecmodel.cpp) both give
$8\pi^2M^2/A$. The control's three unit-charge probes and the independent
auxiliary-rescaling test below constrain this convention independently.

## The radial and angular averages

The physical Gamma cell is the in-plane reciprocal Voronoi cell of the
actual BvK mesh. Its area is $A_\Gamma=(2\pi)^2/(A N_xN_y)$. If its radial
boundary in direction $n$ is $Q(n)$, the cell average is

$$
\overline W_c=\frac1{A_\Gamma}\int_0^{2\pi}d\theta
\int_0^{Q(n)}q\,dq\,W_c(q,n).
$$

The native analytic radial integration uses
$I_0(a,Q)=\int_0^Qq\,dq/(1+aq)$ and
$I_1(a,Q)=\int_0^Qq^2\,dq/(1+aq)$, with their finite small-a limits.
A direct numerical inverse of the displayed full matrix, followed by an
independent Gauss-Legendre radial quadrature, tests both integrals without
calling their native helpers. The angular rule remains the existing 5000
uniform directions. Comparing at the same angular nodes tests the
implementation. Rotating a cell against a finite angular grid is a separate
quadrature-convergence observable, not an exact algebraic covariance gate.

## Native control and acceptance boundaries

The separate audit control under
`workspace/symmetry/runs/20261023-soc-gw-symmetry-audit/strict2d-general/`
uses the gapped six-state, three-site SOC model with Bz=0/0.21, exact real
local density vertices, and a nonorthogonal in-plane primitive cell
$(7.1,0)$, $(1.3,5.9)$ Bohr. Each density probe has unit integral, so
$M^2=3/(4\pi)$. Native H/S/r velocity preparation receives that area and monopole
metadata through its ordinary public API. The frequency grid remains 24
nodes on [0.4,4] Ha with explicit existing regularization 1e-12; no production
default is changed.

The independent oracle reconstructs all head, right-wing, left-wing and
body matrices from the complete finite Lehmann sum. It independently builds
the reciprocal lattice and Gamma-cell radial boundary, and inverts the
full directional dielectric matrix at each radial quadrature point.
The native inner strict-2D rewrite is measured separately from the complete
`compute_Wc_freq_q_blacs` entry point. Native H/S/r preparation uses the
ordinary AO-major spinor packing and all six states, with a real diagonal
position operator; the independent spectral oracle evaluates its matrix
elements after diagonalizing the Hamiltonian. The control has identity AO
overlap; nonorthogonality here describes the oblique primitive lattice, not
a tested nontrivial AO metric. It tests one Gamma cell, with no k-star/IBZ
reduction.

The entry point was initially guarded for all SOC strict-2D cases. The
only additional production change in this slice is the
[epsilon admission guard](../../../src/core/epsilon.cpp), which now admits
general-frequency strict-2D screening only with analytic head/wing option 3,
active complete-Wc replacement, a nonempty optical activation vector, and
configured producer monopole normalization. The existing complex Schur and
radial formulas require no further change. The LU requirement is retained;
the general response cannot use a Hermitian Cholesky factorization.

Two API details matter for reaching this actual consumer: an empty
`epsmac_LF_imagfreq` vector selects the body-only path even with a head/wing
object, and `wing_mu_to_lambda` consumes the auxiliary-basis wing vectors.
The external control explicitly supplies the activation vector and uses a
separate copy for the inner helper; the outer call receives fresh original
head/wing inputs. Thus a successful body-only path or a previously transformed
wing object cannot masquerade as a strict-2D optical pass.

## Independent complete-consumer results

The [seven-case summary](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/strict2d-general/candidate-summary.json)
contains 24 positive frequencies per case. Every actual complete-Wc call
passed, for Bz=0 and 0.21 Ha, one/four MPI ranks, auxiliary scale 1/1.7,
height 18/31 Bohr, and simultaneous in-plane rotation of cell and position
vertices by 0.29 radians. The seven cases are bounded controls, not the full
Cartesian product of these parameters. Errors below are maximum absolute
matrix entries in the control's atomic-unit conventions.

| Check | Maximum error or signal |
| --- | ---: |
| Complete native Wc versus direct inverse, same native input | $6.676\times10^{-14}$ |
| Complete native Wc versus fully independent Lehmann input and inverse | $8.311\times10^{-11}$ |
| Complete caller versus native inner strict-2D rewrite | $2.396\times10^{-14}$ |
| Independent 32/64-point radial quadrature difference | $1.562\times10^{-15}$ |
| Reciprocal-cell radial boundary versus independent lattice construction | $2.221\times10^{-16}$ |
| Native positive-frequency response versus Lehmann response | $2.880\times10^{-11}$ |
| Native optical head versus Lehmann head | $1.111\times10^{-15}$ |
| Native independent wings versus Lehmann wings | $3.536\times10^{-18}$ |

The same-input inverse gate remains $2\times10^{-10}$ and radial-convergence
gate $2\times10^{-12}$. No gate was enlarged. The larger full-Lehmann error
is consistent with the measured finite minimax response error; it is not an
error in the radial inverse. Scaling the auxiliary basis changes absolute
matrix units by $1.7^2$, so its unnormalized full-Lehmann error is correspondingly
larger. The Bz=0.21 unit-basis case has a response anti-Hermitian signal
$7.38573\times10^{-5}$ and independent-wing difference
$3.96350\times10^{-5}$. Replacing the left wing with the right-wing adjoint
changes Wc by $2.27979\times10^{-8}$, well above the unchanged numerical gate.
This negative control distinguishes the repaired general route from an
accidentally reciprocal fixture. For Bz=0 those signals disappear to roundoff.

The complete-caller invariances are independently computed from its saved
matrices, not inferred from the inner helper:

| Transformation | Maximum Wc difference |
| --- | ---: |
| Auxiliary functions scaled by 1.7, with Wc divided by $1.7^2$ | $5.959\times10^{-16}$ |
| Vacuum height 18 to 31 Bohr at fixed in-plane physics | $6.246\times10^{-17}$ |
| One to four ranks, unit auxiliary scale | $3.730\times10^{-16}$ |
| One to four ranks, auxiliary scale 1.7 | $7.148\times10^{-16}$ |

The rescaling control changes the RI coefficients inversely, Coulomb matrices
quadratically, and producer radial-moment metadata quadratically. It thereby
checks the external head legs against the same auxiliary-basis convention,
rather than adjusting only an arbitrary scalar in the oracle.

### Finite angular quadrature remains a separate limitation

The native 5000-direction rule gives a $1.48890\times10^{-9}$ change under
the 0.29-radian rotation. This is larger than the same-input inverse residual
and is **not** an exact rotation-covariance pass. An
[independent angular convergence calculation](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/strict2d-general/angular-convergence.json),
using freshly generated directions and direct dielectric inversion at two
frequencies, gives rotation differences $1.48889\times10^{-9}$ at 5000,
$9.62242\times10^{-10}$ at 10000, and $3.90502\times10^{-11}$ at 20000
directions. The physical radial boundary is piecewise smooth at reciprocal
Voronoi corners. These data identify the finite angular rule as a numerical
limitation; they do not change the production quadrature or establish its
convergence for every anisotropic material. Conditioning near a vanishing
directional Schur complement is also outside this gapped control.

## Regression, provenance, and preserved failed controls

The new repository
[strict-2D signed-frequency regression](../../../src/test/test_strict2d_signed_frequency.cpp)
uses the same physical normalization but an independent direct full-matrix
inverse and 16/32-point radial quadrature. It checks two frequencies after
the actual caller constructs all 24. Both serial and four-rank runs give
Wc error $2.20906\times10^{-11}$, radial difference
$7.22775\times10^{-17}$, incorrect-left-wing signal
$2.27979\times10^{-8}$, and odd-response signal $7.38573\times10^{-5}$.
Its Wc gate remains $10^{-10}$ and radial gate $2\times10^{-12}$.
The focused [LibRI suite](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/strict2d-general/ctest-libri.log)
passed 5/5 in 12.46 seconds (new serial/MPI regression, existing head/wing
serial/MPI and epsilon tests). The
[non-LibRI suite](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/strict2d-general/ctest-nolib.log)
passed its three applicable existing tests in 6.68 seconds. The new SOC
control requires LibRI and is not claimed as a non-LibRI SOC-route test.

The external [native source](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/strict2d-general/native-strict2d.cpp),
[independent checker](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/strict2d-general/check.py),
[replay entry](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/strict2d-general/replay.sh),
[saved-result aggregation](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/strict2d-general/summarize.py),
build logs, per-case raw matrices, and
[binary hashes](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/strict2d-general/binary-provenance.txt)
are preserved together. The tested shared-library SHA256 is
`7e0ea39faa4dc918e3afeedac44a47327a9e05df4a33efb2f04f612fd2f7a6bf`.

Two initial failures are retained rather than rewritten. The first control
reused already consumed wings and failed before a valid outer optical call.
The second supplied synthetic $M^2=3$ for unit-integral probes and used an
incorrect $4\pi^2$ coefficient in its oracle, producing a head factor-of-two
disagreement. Its source, raw outputs, and diagnostic inverse are preserved
as `native-strict2d-before-metadata.cpp`, `baseline-bz021-np1-r1/`,
`check-before-inner-diagnostic.py`, and `diagnose-inner.log`. Checking the
actual producer's radial $Y_{00}$ convention corrected the fixture to
$M^2=3/(4\pi)$ and $8\pi^2M^2/A$ before acceptance. Neither preserved failure
is attributed to a production screening defect.

The complete-Wc cases above accept the bounded **native general-frequency strict-2D
complete-Wc consumer**, including independent wings, physical Gamma geometry,
producer normalization, and rank ownership. Those cases alone do not test
the signed-time self-energy; the following control supplies that additional
dense conjunction. Analytic continuation, QSGW iteration, nontrivial AO
metric, k-star reduction, and a material band/topology result remain outside
this strict-2D acceptance.

## Signed W and the native correlation self-energy

The resumed control measures every auxiliary entry of $W_c(\pm\tau)$ and
every AO spin block of $\Sigma_c(i\omega)$ after the actual complete-Wc
consumer. It distinguishes a signed inverse-transform or RI-contraction
defect from approximation/conditioning in the fixed minimax grid. The
independent reference is the complete ordered particle-hole spectral
resolvent at each physical radial/angular point, followed by the same Gamma
cell integral and an analytic pole contraction with the finite Green
function. Both positive and negative pole sets are retained independently;
pairing at a single direction is not assumed.

Before execution the gates remain those of the earlier dense native GW
control: native contraction against an independent contraction of its
sampled W has tolerance $2\times10^{-10}$ Ha, AO-to-KS projection
$2\times10^{-12}$ Ha, and the fully independent pole comparison with
explicit regularization $10^{-12}$ has tolerance $2\times10^{-8}$ Ha.
The strict-2D same-input W inverse and radial quadrature gates remain
$2\times10^{-10}$ and $2\times10^{-12}$. Unregularized output is retained
as a conditioning comparator, not silently accepted by relaxing a gate.
The initial conjunction uses the same Gamma-only, identity-AO-metric,
uncompressed model; it does not test continuation, QSGW iteration,
nonorthogonal/IBZ/shrink conjunctions or material topology. Results are
reported below only for the completed controls.

### An independent spectral oracle at each radius and direction

Let $m$ be occupied, $n$ empty, $\Delta_{mn}=\epsilon_n-\epsilon_m>0$,
and $d_{mn}$ the auxiliary density matrix element. In the native head
convention define $h_{mn}=-i\langle m|\mathbf r|n\rangle$. At every physical
radius $\rho$ and in-plane direction $\hat n$, form the two **ordered**
transition vertices

$$
b^+_{mn}(\rho,\hat n)=\begin{pmatrix}
\sqrt{2\pi\rho/A}\,\hat n^T h_{mn}\\
L_bU_b^\dagger d_{mn}
\end{pmatrix},\qquad
b^-_{mn}(\rho,\hat n)=\begin{pmatrix}
\sqrt{2\pi\rho/A}\,\hat n^T h_{nm}\\
L_bU_b^\dagger d_{nm}
\end{pmatrix}.
$$

The head vertex is imaginary even for real position operators, so blindly
conjugating the complete first column does not construct the reversed
transition: $h_{nm}=-h_{mn}^*$, whereas $d_{nm}=d_{mn}^*$ in this real
auxiliary gauge. Constructing both from their matrix elements retains the
actual sign. Collect all 18 ordered columns into
$B=(b^+,b^-)$, and let
$J=\operatorname{diag}(I,-I)$ and
$D_s=\operatorname{diag}(\Delta,-\Delta)$. The complete directional
dielectric matrix is

$$
E(z)=I-B(z-D_s)^{-1}JB^\dagger.
$$

This identity reproduces precisely the $\rho$-dependent head and
$\sqrt\rho$ wings displayed above. It does not replace the internal
strict-2D interaction with a unit head. Woodbury inversion instead gives

$$
E^{-1}(z)-I=B(z-K)^{-1}JB^\dagger,\qquad
K=D_s+JB^\dagger B.
$$

Let $K=R\operatorname{diag}(\Omega_p)R^{-1}$ and let $r_p$ and $\ell_p$
denote its right eigenvector and the corresponding row of $R^{-1}$.
With the physical external-leg matrix

$$
T=U\operatorname{diag}\left(
\sqrt{4\pi M^2/A}\sqrt{2\pi/\rho},L_b\right),
$$

the auxiliary-basis residues are
$A_p=TB r_p\ell_p JB^\dagger T^\dagger$. Consequently

$$
W_c(z;\rho,\hat n)=\sum_p\frac{A_p}{z-\Omega_p},\qquad
W_c(+\tau)=-\sum_{\Omega_p>0}A_pe^{-\Omega_p\tau},\qquad
W_c(-\tau)=+\sum_{\Omega_p<0}A_pe^{\Omega_p\tau}.
$$

The oracle integrates these expressions with
$\rho\,d\rho\,d\theta/A_\Gamma$. It retains both signed pole sets at every
direction; it does not assume that opposite poles pair at that direction.
Each local spectral sum is separately checked against a direct inverse of
the full $3\times3$ dielectric matrix before Gamma averaging. This avoids a
false pass from angular cancellation. In this three-site control there are
nine positive and nine negative poles at every quadrature point, with no
small collective pole.

### Exact self-energy and independent finite-grid contraction

Use the native space-time Green-function convention

$$
g^+_{ij}(\tau)=\sum_{n\in\mathrm{empty}}
P^n_{ij}e^{-(\epsilon_n-\mu)\tau},\qquad
g^-_{ij}(\tau)=-\sum_{n\in\mathrm{occ}}
P^n_{ij}e^{-(\mu-\epsilon_n)\tau},\qquad
P^n_{ij}=C_{in}C_{jn}^*.
$$

With exact local density RI vertices the contraction is
$\Sigma^\pm_{ij}(\tau)=g^\pm_{ij}(\tau)
W^\pm_{a(i)a(j)}(\tau)$, where $a(i)$ identifies the spatial site of the
spinor AO. If the auxiliary functions are scaled by $s$, both RI vertices
contribute the explicit factor $s^{-2}$. For every radial/angular point the
analytic Fourier integral is therefore

$$
\Sigma_{ij}(i\omega)=
-\sum_{n\in\mathrm{empty}}\sum_{\Omega_p>0}
\frac{P^n_{ij}(A_p)_{a(i)a(j)}}{
\epsilon_n-\mu+\Omega_p-i\omega}
-\sum_{n\in\mathrm{occ}}\sum_{\Omega_p<0}
\frac{P^n_{ij}(A_p)_{a(i)a(j)}}{
\mu-\epsilon_n-\Omega_p+i\omega},
$$

followed by the Gamma integral and, where applicable, $s^{-2}$. This
reference uses no native frequency-to-time weights. Separately, applying
the exported native cosine/sine weights to the exact signed time values
measures the forward finite-grid error. Contracting the actual native
sampled signed W measures the RI and inverse-transform implementation
error. Finally $C^\dagger\Sigma_\mathrm{AO}C$ checks the independently
exported KS matrices. Every frequency and all four AO spin blocks are
decoded from their actual checkpoint shards, with exactly one owner
required for each nonzero matrix entry.

### Native conjunction and fixed-grid results

The separate [native conjunction](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/strict2d-general/native-strict2d-sigma.cpp)
calls the same complete-Wc consumer, then
`CT_FT_Wc_freq_q` and `G0W0::build_spacetime`, and exports the actual
AO checkpoint shards and KS matrices. It independently calls the native
inverse on the exchanged positive/negative frequency halves to obtain a
second negative-time diagnostic. The production path instead uses
$W(R,-\tau)=W(R,+\tau)^*$ in the real localized auxiliary basis, through
the two named LibRI caches in [gw.cpp](../../../src/core/gw.cpp), lines
1801--1814. These are analytically equivalent only with the required
charge-reality identity; their floating-point difference is measured,
not silently identified with an RI-contraction defect.

The [independent oracle](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/strict2d-general/check-sigma.py)
uses 5000 angular nodes and 16 radial nodes, or 80000 independently
diagonalized transition-space operators. A separate 32-radial-node check
has 160000 operators. It reproduces the full directional Dyson inverse
with maximum error $7.806\times10^{-14}$ across these runs. Pole imaginary
parts are below $1.4\times10^{-15}$ Ha and minimum absolute pole exceeds
1.69 Ha. The 16/32 radial differences are $1.562\times10^{-17}$ for W,
$8.674\times10^{-18}$ for signed-time W, and
$1.087\times10^{-17}$ Ha for Sigma, far below the unchanged
$2\times10^{-12}$ radial gate.

The [saved seven-case summary](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/strict2d-general/sigma-summary.json)
contains five regularized passes and two retained unregularized failures
of the independent-negative-time gate. Each case covers all 864 complex
AO self-energy entries (24 frequencies, six by six spinor matrix), each
with exactly one native shard owner. Thus 6048 AO entries are checked
across the seven cases, with the 32-radial check reusing the first native
case rather than being counted as another native execution.

| Case | Explicit regularization | Sigma versus exact pole reference (Ha) | Native Sigma versus independently inverted signed-W contraction (Ha) |
| --- | ---: | ---: | ---: |
| Bz=0, one rank | $10^{-12}$ | $8.361\times10^{-11}$ | $4.718\times10^{-13}$ |
| Bz=0, four ranks | $10^{-12}$ | $8.361\times10^{-11}$ | $4.719\times10^{-13}$ |
| Bz=0.21, one rank | $10^{-12}$ | $6.872\times10^{-11}$ | $4.950\times10^{-13}$ |
| Bz=0.21, four ranks | $10^{-12}$ | $6.872\times10^{-11}$ | $4.994\times10^{-13}$ |
| Bz=0.21, auxiliary scale 1.7, one rank | $10^{-12}$ | $6.872\times10^{-11}$ | $5.065\times10^{-13}$ |
| Bz=0.21, one rank | 0 | $3.729\times10^{-9}$ | $3.333\times10^{-9}$: **fails** |
| Bz=0.21, four ranks | 0 | $3.902\times10^{-9}$ | $3.328\times10^{-9}$: **fails** |

For every case, independently contracting the **actual conjugated cache**
used by production agrees with native Sigma to at most
$1.840\times10^{-18}$ Ha. Native KS projection agrees with
$C^\dagger\Sigma_\mathrm{AO}C$ to below $1.8\times10^{-18}$ Ha. Thus the
two failed rows do not establish a new RI or spin-block implementation
defect. Their failure is specifically the finite-precision equivalence of
the two signed inverse constructions.

With explicit regularization $10^{-12}$, the signed-W pole error is at
most $3.821\times10^{-11}$ in the unit auxiliary basis, or
$1.106\times10^{-10}$ after scaling that basis by 1.7. The exact-W
forward time-quadrature errors in Sigma are
$8.365\times10^{-11}$ Ha for Bz=0 and $6.865\times10^{-11}$ Ha for
Bz=0.21, essentially the full-pipeline error. The rank differences are
$1.132\times10^{-14}$ and $4.131\times10^{-14}$ Ha respectively; changing
auxiliary scale leaves physical Sigma invariant to
$8.197\times10^{-14}$ Ha. Using the positive-time W cache for **both**
time signs instead gives a $1.213\times10^{-6}$ Ha error in the magnetic
case, so this model discriminates the signed-W repair.

### Why the unchanged unregularized inverse still fails

The inverse-sine maximum absolute row sum is
$1.30705\times10^{11}$ with zero regularization, versus
$3.28015\times10^4$ with explicit $10^{-12}$. At zero regularization the
native frequency W agrees with the exact spectral reference to
$8.23\times10^{-12}$, but the time-domain error grows to approximately
$6.49\times10^{-7}$. Even applying the unregularized inverse to the
independent exact-frequency samples leaves a
$5.669\times10^{-8}$ time error. This is evidence of the conditioning
limitation independently of native Chi0 and W formation.

In this real auxiliary, Gamma-averaged control, the exact frequency W is
real. Native construction leaves a charge-reality residual
$\|W-W^*\|_\max$ of only $8.67\times10^{-15}$, but the poorly conditioned
inverse amplifies it: independently inverted negative W and conjugated
positive W differ by approximately $6.01\times10^{-8}$. Contracting
those two versions accounts for the failed $3.33\times10^{-9}$ Ha
comparison. One/four-rank Sigma differs by
$3.836\times10^{-10}$ Ha at zero regularization. The code still contracts
the cache it is given accurately; the inverse does not preserve the desired
equivalent representations at the declared precision. No tolerance was
relaxed, no charge-reality projection was added, and the default
regularization remains unchanged.

### Receipt and remaining boundary

The external conjunction was compiled against the unchanged shared library
already identified above; [its separate binary receipt](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/strict2d-general/sigma-binary-provenance.txt)
records executable SHA256
`3715024747a04807abf719e05f576da32b5e3c46dafa2140dead0072ab8dbf5d`.
The [build script](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/strict2d-general/build-sigma.sh),
native logs, pole-oracle logs, 16/32 radial NPZ arrays and
[error disaggregation](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/strict2d-general/summarize-sigma.py)
are retained in the same evidence directory. The unregularized checks
return failure and their assertion logs remain in
`sigma-bz021-reg0-np1/pole16.log` and
`sigma-bz021-reg0-np4/pole16.log`; they are not relabeled as passes because
one looser aggregate error happens to be small. Earlier interrupted fixture
scripts are preserved separately from these numerical outcomes.

This conjunction needs no further production source change or shared-library
rebuild. The existing formula is correct for the tested operators, and five
configured regularized executions pass the retained gates. It establishes
the **dense, real-auxiliary, Gamma-only strict-2D complete-Wc to signed-time
to all-block Sigma route** in this gapped finite model. The finite
5000-angle limitation above remains. A nontrivial AO metric, auxiliary
compression, IBZ reduction, metallic response, continuation, QSGW iteration
and material topology require their own conjunctions. The two unregularized
failures remain an explicit numerical limitation of this otherwise
implemented route.

## Nonorthogonal AOs and an exact rectangular auxiliary representation

The next conjunction keeps the same physical Gamma model but changes both
representations. It tests all AO self-energy blocks under a nonorthogonal
AO congruence, and the actual compressed atom-pair/time consumer under a
rectangular auxiliary map. Removing a direction that carries physical
response would instead change the approximation; the following construction
deliberately has an exact inactive subspace so that invariance is required.
The established $10^{-10}$ Ha representation gate, $2\times10^{-8}$ Ha
pole gate and $2\times10^{-12}$ Ha KS-projection gate are unchanged.

### Preserve the operator, including its RI charge vertices

Let $T$ be the real, spin-identical AO transformation with spatial block

$$
T_0=\begin{pmatrix}1.10&0.12&-0.05\\0.04&0.91&0.09\\-0.07&0.08&1.05\end{pmatrix}.
$$

For the orthonormal reference operators and eigenvectors use

$$
S'=T^\dagger T,\quad H'=T^\dagger HT,\quad
r'_\alpha=T^\dagger r_\alpha T,\quad C'=T^{-1}C,\quad
\Gamma'_a=T^\dagger\Gamma_aT.
$$

Hence $C'^\dagger S'C'=I$, $H'C'=S'C'E$, and every transition vertex
$C'^\dagger\Gamma'_aC'=C^\dagger\Gamma_aC$ is invariant. The charge
identity is $\sum_a\Gamma'_a=S'$, not $I$. The native optical API receives
the transformed H/S/r operators and the native RI calculation receives
the transformed charge vertices. Transforming only the eigenvectors would
change the physical model.

The three spatial orbitals are grouped on one RI center so that all entries
of these dense vertices can be represented exactly by local Cs. There are
still six physical spinor states, the same H, r and three charge channels;
the regrouping introduces no extra band or response channel. At Gamma the
fixed position operator, rather than an inferred atom coordinate, supplies
the same dipole physics. The test verifies the complete transition vertices
and charge operator before comparing screening. This construction is not a
claim that the regrouped center has the original multi-center spatial
symmetry representation.

### Remove only an exactly dark auxiliary subspace

Embed the three physical charge channels in six parent auxiliary channels
with the real isometry $E\in\mathbb R^{6\times3}$,
$E_{2a,a}=0.8$, $E_{2a+1,a}=0.6$ and all other entries zero. The declared
active map is $U=E^\dagger$, so $UE=I_3$ but $EU\ne I_6$. Define

$$
\Gamma^\mathrm{parent}_\mu=\sum_aE_{\mu a}\Gamma'_a,\quad
V_\mathrm{parent}=EV_3E^\dagger+0.27(I-EE^\dagger),\quad
Q_\mathrm{parent}=E(1,1,1)^T.
$$

The positive inactive Coulomb block prevents this from being a trivial
zero-padded matrix. Nevertheless
$(I-EE^\dagger)\Gamma^\mathrm{parent}=0$ and
$(I-EE^\dagger)Q_\mathrm{parent}=0$, so the discarded channels have no
response or head charge. In particular

$$
\sum_\mu Q_\mu\Gamma^\mathrm{parent}_\mu=S',\qquad
M^2=\frac{Q_\mathrm{parent}^\dagger Q_\mathrm{parent}}{4\pi}
=\frac{3}{4\pi}.
$$

The native metadata is therefore fixed by the actual transformed charge,
not by the number of parent basis functions. No area, angular rule,
minimax interval, or regularization is changed. The required identities are

$$
\chi_3=U\chi_6U^\dagger,\qquad (W_c)_6=U^\dagger (W_c)_3U,
\qquad\Sigma'_\mathrm{AO}=T^\dagger\Sigma_\mathrm{AO}T,
\qquad C'^\dagger\Sigma'_\mathrm{AO}C'=C^\dagger\Sigma_\mathrm{AO}C.
$$

These statements test the implementation of a represented projection. They
do not establish small error for a PCA cutoff that removes a nonzero
physical charge vertex or mixes the retained head with discarded charge.
The W identity is for the correlation interaction: the total screened
interaction also retains the bare $0.27(I-EE^\dagger)$ dark block, which
does not couple to the RI charge vertices in this control.

### Native complete-consumer results

The [new native fixture](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/strict2d-general/native-strict2d-metric-shrink.cpp)
first captures Chi0, both optical wings and corrected frequency W, then
calls the ordinary `Handler::build_g0w0_sigma` API with fresh optical data.
Thus compressed W remains compressed through the actual API collector,
signed atom-pair inverse, U unfolding and parent-Cs self-energy contraction.
It does not substitute an already unfolded W into the dense Sigma route.
The full and compressed paths use the same physical input; the independent
reference is the previously computed exact-pole operator transformed by T.
The [checker](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/strict2d-general/check-metric-shrink.py)
first verifies that the complete model and frequency nodes match that
reference, rather than importing its result on the basis of a filename.

All **eight** cases pass: full six auxiliaries, three auxiliaries before
Chi0, six-to-three transformation after Chi0, and the delayed-FT variant of
the latter, each at one and four MPI ranks. All 6912 AO entries across
these executions have exactly one checkpoint-shard owner. The
[combined results](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/strict2d-general/metric-shrink-summary.json)
give the following maxima:

| Check | Maximum absolute difference |
| --- | ---: |
| $C'^\dagger S'C'-I$ | $2.554\times10^{-15}$ |
| Complete charge transition vertices before/after AO change | $3.886\times10^{-16}$ |
| Parent charge operator minus S' | $2.3\times10^{-16}$ |
| Discarded charge or vertex component | $5.552\times10^{-17}$ |
| Full/compressed or rank-reordered Chi0 | $3.470\times10^{-17}$ |
| Full/compressed or rank-reordered corrected W | $7.296\times10^{-16}$ |
| Matched full/compressed AO Sigma | $1.455\times10^{-13}$ Ha |
| Matched full/compressed KS Sigma | $9.617\times10^{-14}$ Ha |
| AO congruence versus original three-center calculation | $1.850\times10^{-13}$ Ha |
| AO Sigma versus independent transformed pole operator | $7.110\times10^{-11}$ Ha |
| KS Sigma versus independent pole operator | $1.510\times10^{-10}$ Ha |
| Native KS projection versus $C'^\dagger\Sigma'_\mathrm{AO}C'$ | $1.302\times10^{-18}$ Ha |

The overlap is nontrivial and well conditioned: its largest off-diagonal
element is 0.1628 and condition number is 1.87943. An intentionally incorrect
projection inserting extra S factors differs by
$1.28203\times10^{-3}$ Ha, so the metric test is discriminating. The KS
pole error is larger than the AO maximum-entry error because a basis
transformation need not preserve that norm; both satisfy the unchanged
predeclared $2\times10^{-8}$ Ha pole gate. This is not a relaxed
$10^{-10}$ Ha representation-equivalence test.

No production source, header, numerical default or shared library changed
in this stage. The [binary receipt](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/strict2d-general/metric-shrink-binary-provenance.txt)
and all original outputs are retained. An initial external setup omitted
its auxiliary BLACS descriptors and aborted before a valid W calculation;
the corrected fixture runs have separate directories. The earlier
unregularized failures remain unchanged. Genuine periodic k-star/IBZ
restoration needs a spatially represented multi-center model and is
addressed separately; it cannot be inferred from this regrouped Gamma
control.

## Periodic nonorthogonal/shrink/IBZ conjunction

The periodic control is a **distinct finite model**, not a claim that the
one-center Gamma regrouping carries the original spatial symmetry. It
extends the already independently checked C3 plus $\Theta M_y$
[periodic optical fixture](nonreciprocal-screening.md#13-complete-nonorthogonal-magnetic-optical-conjunction)
to strictly two-dimensional screening. Every comparison within this stage
uses the same Hamiltonian, charge vertices, optical operators and numerical
settings. It tests whether the complete native pipeline respects that fixed
operator when its mean-field input and auxiliary representation are reduced.
It is not a new exact-pole reference for the absolute periodic optical
self-energy.

### A two-dimensional Hamiltonian with a valid magnetic group

The three-center lattice has in-plane rows
$(6,0)$ and $(3,3\sqrt3)$ Bohr, height 8 Bohr and a $3\times3\times1$
mesh. Native H(R) has only $R_z=0$ blocks. The inherited model's
$\pm z$ hopping is folded exactly into its onsite operator, defining
$H_{2D}(k_x,k_y)=H_\mathrm{parent}(k_x,k_y,0)$; its constant off-diagonal
addition is $0.035\times0.8$ Ha. Thus the native input is explicitly
in-plane while preserving the previously checked sampled Hamiltonian and
velocities. This folding is part of this control's definition, not a
modification to a material input or the library.

The AO metric is $S=(1-s)I+sJ$ with $s=0.12$, repeated for both spin
components. The three charge vertices are
$\Gamma_a=(P_aS+SP_a)/2$, their sum is S, and
$r_\alpha=\sum_a x_{a\alpha}\Gamma_a$. The native H/S/r velocity is
checked against the analytic hopping derivative plus the commutator in
the orthonormal frame. Bz=0.21 Ha breaks ordinary time reversal but retains
C3 and $\Theta M_y$. The independently supplied operations satisfy the
Hamiltonian, metric, position, charge and velocity covariance relations;
all preflight residuals are below $8.89\times10^{-16}$.

The active auxiliary vector $(0.8,0.6)$ on each atom and its orthogonal
dark channel give the same exact six-to-three construction as above,
now respecting the actual atom permutations. Parent and active head
charges have squared norm 3, hence radial metadata $M^2=3/(4\pi)$.
The optical coefficients use $2\pi/A$ and $\sqrt{2\pi/A}$, with
$A=18\sqrt3$ Bohr squared; the former three-dimensional $4\pi/\Omega$
normalization is not retained. The Gamma integration cell area is
$(2\pi)^2/(9A)$. The existing 5000 directions, frequency interval
[0.4,4] Ha, 24 nodes and explicit regularization $10^{-12}$ are unchanged.

Independent star counting gives sizes 1,2,3,3. The reduced input actually
contains four k points rather than nine with a symmetry flag. The native
real-space restoration covers all 81 atom-pair/cell sectors, including
36 antiunitary sectors, from 21 irreducible sectors. Non-origin and
antiunitary density entries are respectively as large as 0.00659 and
0.0991, so the restoration is numerically active.

### Numerical transport and independent Gamma inverse

The final [native source](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/strict2d-general/native-strict2d-periodic-ibz.cpp)
executes four routes at one and four MPI ranks: full k/full auxiliaries,
full k/rectangular compression, actual IBZ/full auxiliaries, and actual
IBZ/rectangular compression. All eight native executions pass the unchanged
$10^{-10}$ Ha complete-Sigma representation gate. Their complete KS
matrices cover 62208 entries in total. Across all eight routes against the
one-rank unreduced reference, the
[final comparison](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/strict2d-general/periodic-ibz-inplane-summary.json)
gives Sigma differences at most $1.125\times10^{-13}$ Ha, Chi0 differences
$7.286\times10^{-17}$, and corrected W differences
$1.350\times10^{-15}$. Captured signed-time W differences within each
rank configuration are at most $5.247\times10^{-13}$.

The full-band, full-k Lehmann checks give native Chi0 error at most
$3.979\times10^{-11}$ and uncorrected body W error
$2.765\times10^{-11}$. Native optical head and both independent wings
agree with their ordered Lehmann construction within
$8.89\times10^{-15}$. A separate
[directional checker](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/strict2d-general/check-periodic-ibz-inplane.py)
reconstructs the reciprocal Voronoi boundary from the actual lattice and
inverts the full Gamma dielectric matrix at 32/64 radial quadrature nodes.
It reproduces the corrected W at all 24 frequencies to
$7.016\times10^{-15}$, with radial convergence
$6.783\times10^{-16}$ and geometry residual
$8.327\times10^{-17}$. Forcing the left wing to equal the right-wing
adjoint changes W by $3.233\times10^{-8}$, above the retained inverse
gate.

At fixed positive imaginary frequency, the magnetic auxiliary operator
obeys the dynamic antiunitary law

$$
W_c(g\mathbf q,i\omega)=D_g(\mathbf q)
W_c(\mathbf q,i\omega)^T D_g(\mathbf q)^\dagger.
$$

The independent auxiliary atom-permutation representation gives residuals
below $3.522\times10^{-16}$ for unitary and antiunitary operations away
from Gamma. Substituting same-frequency conjugation yields
$1.063\times10^{-4}$, comparable to the resolved anti-Hermitian signal
$\|W_c-W_c^\dagger\|_\max=1.063\times10^{-4}$. At Gamma the fixed angular
rule leaves a residual $1.48943\times10^{-12}$; the independent radial
inverse reproduces $1.48822\times10^{-12}$. This is retained as finite
angular error, not described as exact covariance or removed by an
additional projection.

### Preservation and acceptance limit

The final directories are `periodic-ibz-2d-np1/` and
`periodic-ibz-2d-np4/`, with the
[final binary receipt](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/strict2d-general/periodic-ibz-inplane-binary-provenance.txt).
The earlier sampled-kz=0 version and its source/binary remain preserved
as `native-strict2d-periodic-ibz-before-zfold*` and their original run
directories. The first copied fixture also reused a head/wing object after
the diagnostic strict-2D W capture consumed its raw wings; that failing
source/binary/output is preserved as `*before-optical-copy*` and
`periodic-ibz-np1/`. The corrected diagnostic uses its own copied optical
object, and the public Sigma API receives the original inputs. Neither
fixture correction is a production-code repair.

This stage validates a bounded **periodic, nonorthogonal, magnetic
strict-2D complete-Wc to signed-time to Sigma conjunction with genuine IBZ
and exact rectangular compression**, including nonzero antiunitary
restoration. It does not establish absolute periodic optical Sigma error
against a pole oracle, harmless lossy compression, arbitrary complex
auxiliary head gauges, continuation, metallic response, QSGW convergence or
material topology. The earlier zero-regularization failures remain. No
library source, algorithm, default or acceptance threshold changed in
these extensions.
