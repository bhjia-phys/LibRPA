# Spinor Green functions, charge response, and the frequency-symmetry boundary

This note describes the CPU space-time/LibRI path. It distinguishes a corrected
distributed band-selection omission from the existing restriction to an even
imaginary-frequency charge response. Correcting the omission does not implement
general nonreciprocal magnetic screening.

## Spectral convention and the retained subspace

For spatial AO index $i$, spinor index $s$, and generalized eigenvectors
$C^\dagger S C=I$, define the code's imaginary-time buffer by

$$
\mathcal G_{is,jt}(R,\tau)=\sum_{kn}e^{-2\pi i k\cdot R}
C_{is,nk}C_{jt,nk}^*m_n w_{nk}(\tau),
$$

where $m_n\in\{0,1\}$ is the requested retained-band mask. With physical
occupation $f_{nk}$ and full-grid quadrature weight $w_k$,

$$
w_{nk}(\tau)=
\begin{cases}
w_k(1-f_{nk})e^{-(\epsilon_{nk}-\mu)\tau},&\tau>0,\\
-w_k f_{nk}e^{-(\epsilon_{nk}-\mu)\tau},&\tau<0.
\end{cases}
$$

Thus $\mathcal G=-G$ relative to the conventional
$G=-\langle T_\tau\psi\psi^\dagger\rangle$; the two signs cancel in the
bubble. The implementation clips positive exponents and is intended here for
the gapped occupation convention. This is not a finite-temperature metallic
Green-function implementation.

An overlap matrix is not inserted between the two coefficient factors:
$C g C^\dagger$ is the contravariant AO propagator; the real-space vertices
supply its contractions. Adding $S$ to this buffer would change its meaning.

For a reduced input, stored occupations include the entire k-star weight.
Dividing each reconstructed member by the star size recovers the full-grid
weight. Spinor SOC has one explicit spin channel and two components; its
charge trace must not receive an additional factor of two. The scalar
spin-degenerate route instead supplies its existing $2/n_{\rm spins}$ factor.

The physically selected object is the retained projector, not a list of
numbers:

$$
Q(k)=C(k)mC(k)^\dagger S(k).
$$

It must transform covariantly under every operation used in reconstruction.
For exact covariant eigenpairs, retaining entire degenerate eigenspaces makes
this automatic. The current runtime helper checks that a mask does not split
adjacent energies after ordering a temporary index list when needed. This is
a conservative local guard, stronger than closure under some smaller symmetry
groups; it is not a complete covariance audit of arbitrary producer input.
The [joint-permutation control](soc-self-energy-signed-time.md#truncated-green-functions-require-an-ordering-independent-subspace-check)
explains why the decision must not depend on incidental band labels.

## Distributed restoration must apply the same mask

The replicated spinor builder already passed the mask through its four source
blocks. The distributed `Chi0` caller previously dropped it in both restored
branches: the scalar routine accepted a default empty mask, and the spinor
routine had no mask argument. Direct distributed evaluation did apply it.
Consequently `n_bands_chi0` and `n_bands_exclude_chi0` could change meaning when
k-star reconstruction became active. Disabling real-space response reduction
for an incomplete band space did not disable this earlier k-star operation.

The repair passes the mask from `Chi0`, applies it to each distributed spinor
band weight before matrix multiplication, and uses the same degenerate-space
guard as the replicated path. An empty mask retains the original arithmetic.

The regression in `test_meanfield_mpi.cpp` constructs complex orthonormal
spinor eigenvectors in two complete Kramers pairs, checks the actual selected
projector under $J K$, and compares both signs of $\tau$ against a direct
full-BZ spectral sum. Keeping only the occupied pair must make every
positive-time block zero. This is a numerical operator test, independent of
printed band labels or scalar quasiparticle energies. A separate end-to-end
control of `Chi0::build` uses a dispersive three-site magnetic SOC model,
nine full k points and three genuine IBZ representatives with star sizes
1, 2 and 6. Its retained-band response is compared directly with the original
full-BZ Lindhard sum, rather than with another Green-function contraction.
Thus the caller wiring is checked in addition to the low-level operation.

## The charge bubble and the reverse Green function

For the spin-independent density vertex,

$$
\chi^0(1,2;\tau)=\sum_{st}
\mathcal G_{st}(1,2;\tau)\mathcal G_{ts}(2,1;-\tau).
$$

Real spectral weights imply, without assuming time reversal,

$$
\mathcal G^{ji}_{ts}(-R,-\tau)
=\left[\mathcal G^{ij}_{st}(R,-\tau)\right]^*.
$$

LibRI contracts the two buffers in the same AO/translation ordering. The
negative-time buffer must therefore use the same $(s,t)$ indices followed by
elementwise conjugation. Swapping spin labels alone is insufficient. This
adjoint repair is already present in the branch's master baseline; it is
separate from the newly repaired distributed cutoff omission.

Unitary spin/orbital actions restore a bilinear as $D X D^\dagger$.
In the code's explicit convention an antiunitary operation first applies
that unitary action, then applies $J(\cdot)^*J^\dagger$, with
$J=I_{\rm AO}\otimes i\sigma_y$ after accounting for storage order. The
spin trace removes the spin rotation from the charge response. It does not
make a general magnetic response real or even in time.

## Why a cosine transform needs an additional physical condition

At $q=0$, take Hermitian charge vertices $V_\mu$, occupied state $m$, empty
state $n$, $\Delta=\epsilon_n-\epsilon_m>0$, and

$$
A_{\mu\nu}=\langle m|V_\mu|n\rangle
\langle n|V_\nu|m\rangle.
$$

The exact positive/negative-time contributions are
$-A_{\mu\nu}e^{-\Delta\tau}$ and
$-A_{\mu\nu}^*e^{-\Delta\tau}$, respectively. Therefore

$$
\begin{aligned}
\chi^0_{\mu\nu}(i\omega)
&=-\frac{A_{\mu\nu}}{\Delta-i\omega}
  -\frac{A_{\mu\nu}^*}{\Delta+i\omega}\\
&=-\frac{2\Delta\operatorname{Re}A_{\mu\nu}}{\Delta^2+\omega^2}
  +\frac{2\omega\operatorname{Im}A_{\mu\nu}}{\Delta^2+\omega^2},
\end{aligned}
$$

summed over transitions. In contrast, applying twice the cosine transform
to the positive-time buffer gives

$$
\chi^{\rm cos}_{\mu\nu}(i\omega)
=-\frac{2\Delta A_{\mu\nu}}{\Delta^2+\omega^2}.
$$

These coincide when the weighted imaginary transition amplitudes cancel,
as in a reciprocal charge response. Nonmagnetic time-reversal symmetry is
a sufficient condition for the real-space charge correlator used here.
Some magnetic groups also enforce that property. It is not implied by
Hermiticity of the Hamiltonian or by a gap.

For a general magnetic response the cosine-only result both misses the
frequency-odd contribution and retains an inappropriate imaginary part of
the positive-time amplitude. Calling this only a missing small correction
would understate the representation change required. This issue exists with
symmetry reduction switched off; equality of full and reduced computations
cannot settle it.

`chi_libri_ct_accumulate_R` and the alternative
`chi_libri_ft_tw` in `src/core/chi0.cpp` use cosine coefficients only.
Both immediate and delayed auxiliary shrink inherit this assumption.

## Screening and compressed auxiliaries

Within the implemented reciprocal/even-frequency scope,

$$
\widetilde\epsilon=I-v_\epsilon^{1/2}\chi^0v_\epsilon^{1/2},
\qquad
W_c=v_W^{1/2}(\widetilde\epsilon^{-1}-I)v_W^{1/2}.
$$

The possibly different Coulomb matrices denote the existing dielectric and
screening Coulomb choices; this notation does not change their truncation
prescription. The code gathers Hermitian auxiliary blocks, takes Hermitian
matrix powers/inverses, and transforms $W_c$ back to time with a cosine
transform. General nonreciprocal screening would require full positive and
negative time/frequency information through all of these stages, followed
by a compatible self-energy contraction. Adding a sine transform to one
isolated function is insufficient.

For rectangular auxiliary compression $U(q)$, the implemented contraction is

$$
\chi^0_s(q)=U(q)\chi^0_l(q)U(q)^\dagger,
\qquad W_{c,l}(q)=U(q)^\dagger W_{c,s}(q)U(q).
$$

The conjugate transpose is essential for complex Bloch phases. Symmetry
restoration of $U$ must transform both its left and right auxiliary spaces;
it is not an AO spinor rotation. The source restores this rectangular map
and checks supplied full-grid members against the restored values. This
algebra does not establish a basis-truncation error bound: comparison with
the uncompressed, matched operator is still required.

For an antiunitary action the current even-frequency q-star implementation
uses the complex-conjugated orbital transformation. Extending its contract
to a nonreciprocal $\chi(q,i\omega)$ requires keeping the frequency reversal
as well; the static-Hamiltonian conjugation identity cannot simply be
reused at a fixed complex frequency.


## Exact-subspace auxiliary compression as an independent control

To isolate implementation error from lossy basis truncation, introduce two
auxiliary s channels per site and retain the normalized direction
$a=(0.8,0.6)^T$. Its orthogonal complement is $b=(-0.6,0.8)^T$. Write
$A=I_{\rm sites}\otimes a$, $B=I_{\rm sites}\otimes b$ and $U=A^T$.
Construct the full RI vertices and Coulomb matrix as

$$
\Gamma_l=A\Gamma_s,\qquad
v_l=A v_s A^T+\lambda B B^T,\qquad \lambda>0.
$$

Every component of each active pair participates, while the discarded
channel has identically zero charge vertex. Then

$$
\chi_l=A\chi_s A^T,\qquad
W_{c,l}=A W_{c,s}A^T.
$$

The latter follows by splitting the dielectric problem into the orthogonal
$A$ and $B$ sectors: the $B$ sector has zero polarization, so its screened
interaction equals its bare interaction and its correlation part vanishes.
Consequently the following native paths should agree to rounding error:
full auxiliary contraction; direct contraction of compressed RI vertices;
full contraction followed by $U\chi_lU^T$, with either immediate or delayed
Fourier transformation; and unfolding of compressed W before q-star restore.

This construction commutes with the magnetic model's spatial operation:
the same two radial s channels are used on every equivalent atom, and every
site uses the same $a,b$. The spinor states and magnetic operations remain
nontrivial. It is an exact-subspace CPU control, not a bound for arbitrary PCA
thresholds, missing angular shells, or a material's optical head/wing correction.
The following extension retains this exactness through the self-energy consumer.


## Extension through the self-energy and a compatible complex U(q)

At fixed Green function, the time-domain correlation self-energy is linear
in the screened correlation interaction. Schematically, with both RI centers
and all spinor indices understood,

$$
\Sigma^c_{st}(\tau)=-\sum_{\mu\nu}C^\mu G_{st}(\tau)
                          C^{\nu\dagger}W^c_{\mu\nu}(\tau).
$$

Consequently exact equality of the full auxiliary $W^c$, with unchanged
full RI coefficients and Green function, implies equality of the full
self-energy, including off-diagonal spinor and band components. This is a
linearity statement; no quasiparticle equation or analytic continuation is
needed to test it. In the production compressed route, W is collected as
compressed atom-pair blocks, transformed from frequency to time there,
unfolded at each time node, and then transformed from q to R before the
ordinary full-RI contraction. Since U depends on q but not frequency,

$$
\mathcal C_{\omega\to\tau}[U(q)^\dagger W_s(q,i\omega)U(q)]
=U(q)^\dagger\mathcal C_{\omega\to\tau}[W_s(q,i\omega)]U(q).
$$

The same-subspace result therefore applies to this memory-saving order of
operations. A meaningful consumer check compares the full complex matrix
$C_{\rm target}^\dagger\Sigma^c(i\omega)C_{\rm target}$ in one fixed target
basis for every k and frequency, not only its diagonal or its fitted roots.

To test genuinely complex, q-dependent compression while retaining the
supplied orbital representation, set

$$
D_{aa}(q)=\exp[i\alpha\sin(2\pi q\cdot t_a)],\quad
\alpha=0.43,\quad
(t_0,t_1,t_2)=((1,0),(-1,1),(0,-1)),\qquad U(q)=D(q)A^T.
$$

For the model's $C_3$ action $Rq=(-q_y,q_x-q_y)$ and atom permutation
$a\mapsto a+1$, these phases obey

$$
D_{a+1,a+1}(Rq)=D_{aa}(q),\qquad D_{aa}(-q)=D_{aa}(q)^*.
$$

They are periodic under reciprocal lattice shifts. Thus both the unitary
and antiunitary operations are compatible with the already supplied
auxiliary s-shell representation; this is not an arbitrary rephasing with
an inconsistent symmetry table. The compressed inputs must change together:

$$
v'_s=Dv_sD^\dagger,\qquad
\chi'_s=D\chi_sD^\dagger,\qquad
W'_s=DW_sD^\dagger,
\qquad U^\dagger W'_sU=AW_sA^T.
$$

The native full-response/folded-shrink path directly realizes this test.
A direct compressed-RI calculation would additionally require the Fourier
consistent transformed RI coefficients. Reusing the original constant
compressed RI tensor in that route would not be a matching input; the
complex-U control therefore does not claim that direct-RI combination.

Finite-model results and retained native arrays are linked from the
[response audit](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/response-audit.md).
The matched full/native self-energy reference establishes equivalence of
these implementations at the selected minimax grid. It does not establish
convergence of the self-energy to an exact real-frequency or spectral
answer. In particular, the cosine transform for W retains the earlier
reciprocal-response boundary.


## Nonorthogonal optical control and the reciprocal-cell geometry defect

A consistent nonorthogonal variant of the magnetic model can retain the
same orbital count. Let the spatial overlap have $S_{aa}=1$ and
$S_{a\ne b}=0.12$, repeated in both spin components. It commutes with the
supplied magnetic operations. Write $B=S^{1/2}$, and transform the original
orthonormal Hamiltonian and eigenvectors as

$$
H_{AO}=B H_o B,\qquad C=B^{-1}U,\qquad C^\dagger S C=I.
$$

Local RI charge partitions can be chosen as
$\Gamma_a=\{P_a,S\}/2$. They have support only on pairs touching atom a,
so both RI centers represent them without an unsupported three-center
vertex. Their sum is S, which preserves the total-charge matrix element.
The position operator $r_{AO,\alpha}=\sum_a r_{a\alpha}\Gamma_a$ is
consistent with the first moment of those same charge partitions.
For constant S and $r_{o,\alpha}=B^{-1}r_{AO,\alpha}B^{-1}$, the independent
velocity in the orthonormal frame is

$$
v_{o,\alpha}(k)=\partial_{k_\alpha}H_o(k)
                  +i[H_o(k),r_{o,\alpha}],\qquad
v^{band}_\alpha=U^\dagger v_{o,\alpha}U.
$$

Its derivatives come directly from the manufactured real-space hopping
coefficients. The control verifies native H/S/r velocity against this
expression, its vector covariance under every magnetic operation, and
complete-band optical head/wing sums before comparing any GW result.
Thus the optical input is not justified solely by symmetry-on/off agreement.

### Reciprocal rows and all relevant Gamma-cell faces

LibRPA stores reciprocal lattice vectors as rows, $G=A^{-T}$, without an
explicit $2\pi$ in its internal k vectors. On the mesh with dimensions
$N_i$, define reciprocal mesh rows $B_{i\alpha}=G_{i\alpha}/N_i$.
For integer coefficients n the physical reciprocal displacement is

$$
g_\alpha=\sum_i n_iB_{i\alpha},\qquad g=nB.
$$

The former head/wing geometry used $G_{\alpha i}$ instead, combining columns.
For a nonorthogonal lattice that constructs a different reciprocal lattice,
with the same determinant but a different Wigner-Seitz shape. A correct
Gamma-cell volume therefore does not diagnose this error.

The radial boundary in direction $\hat q$ is

$$
q_{max}(\hat q)=\min_{g\cdot\hat q>0}\frac{|g|^2}{2g\cdot\hat q}.
$$

Restricting n to $\{-1,0,1\}^d$ is not sufficient on an unreduced basis:
$b_1=(1,0)$, $b_2=(3,0.2)$ generates the rectangular lattice whose shortest
y vector is $b_2-3b_1$. The correct positive-y boundary is 0.1, while the
fixed-neighbor box, even with correct row orientation, would give 10.1.

A finite sufficient enumeration follows without assuming a reduced basis.
Let

$$
R_{cover}=\max_{s_i=\pm1/2}\left|\sum_i s_i b_i\right|.
$$

The centered fundamental parallelepiped supplies a lattice representative
within $R_{cover}$ of every point. Thus the Voronoi cell lies in that ball.
Any vector defining one of its faces must have $|g|/2\le R_{cover}$.
Since $n=gB^{-1}$, every relevant vector obeys

$$
|g|\le2R_{cover},\qquad
|n_i|\le2R_{cover}\,\|\operatorname{column}_i(B^{-1})\|_2.
$$

The implementation pads these floating-point bounds outward, checks finite
values and integer/product overflow, enumerates the resulting coefficient
box, and retains vectors inside the radial bound. The two-dimensional
version applies the same proof in the xy plane. It rejects searches above
2,000,000 integer candidates before allocating or iterating; this is a
resource limit, not a mathematical truncation. At that limit the raw vector
payload alone could be of order 48 MB. Extremely ill-conditioned unreduced
cells remain unsupported until their lattice basis is reduced; they produce
an explicit error rather than an approximate Gamma geometry. No new lattice
reduction algorithm is claimed here.

### Geometry error versus finite angular integration

For the unchanged 3D head/wing rule, write $\rho_l=w_lq_{max,l}^3/(3V_\Gamma)$.
The Schur representation uses

$$
a_0=\sum_l\frac{\rho_l}{\hat q_l^T L\hat q_l},\qquad
K_{\alpha\beta}=\sum_l
\frac{\rho_l\hat q_{l\alpha}\hat q_{l\beta}}{\hat q_l^T L\hat q_l},\qquad
\epsilon^{-1}_{avg}=D+a_0uu^\dagger+\mathcal A K\mathcal A^\dagger.
$$

Here D is the embedded inverse regular body, u is the head direction,
$Z=\sqrt v\,w_\mu$, $\mathcal A=DZ$, and $L=H-Z^\dagger DZ$.
This notation follows the existing kernel's unweighted constant-body
convention. Its finite weight sum is recorded separately.

In the manufactured magnetic control L is uniaxial to roundoff, while the
column-built radial geometry makes K violate C3. An independent inverse of
each complete directional dielectric reproduces the native W defect.
Correct reciprocal rows remove that geometric violation, but the fixed
Lebedev rule itself is not exactly closed under arbitrary crystal rotations.
A smaller finite-integration covariance residual consequently remains.
Group-closing the directions removes that residual in a separate diagnostic;
it has **not** been applied to production W or adopted as the production
quadrature. Geometry correctness, numerical angular convergence and exact
symmetry of a finite quadrature remain distinct checks.

The modeled consequence of the old geometry is C3 violation. Its positive
and negative reciprocal vectors can remain paired, so this result does not
by itself explain a material's time-reversal or inversion violation, nor
its Kramers splitting. Those require their own operator checks.

The compiled production repair passes the unchanged full optical/Sigma
conjunction gate: the largest full complex Sigma difference across the
full/IBZ and uncompressed/folded routes is 3.24072e-11 Ha on four ranks,
compared with about 2.16e-8 Ha before the repair. The remaining native
full-auxiliary Gamma-W C3 residual is 7.89141e-10 with the unchanged finite
angular rule. The linked response audit retains all failed baseline runs,
complete arrays, and before/after binary hashes.

## Complete optical Lehmann pairs and band ordering

The geometry result above concerns the reciprocal response subspace. General
magnetic screening needs the complete frequency response and independent
optical wings. The full derivation and general Schur identities are in
[the nonreciprocal screening note](nonreciprocal-screening.md#6-head-and-wing-require-two-independent-wings).
The implementation uses the following native conventions. For two arbitrary
distinct band labels m,n, let

$$
\Delta=\epsilon_n-\epsilon_m,\quad d=f_m-f_n,\quad
p_{\mu\alpha}=C^\mu_{mn}v^\alpha_{nm},\quad
P_{\alpha\beta}=v^\alpha_{nm}v^\beta_{mn}.
$$

The sign of Delta and d is retained; m<n is only a way to count each pair
once. In a real auxiliary basis, the native small-q overlap uses the head
probe -i r. With g=sqrt(4 pi/Omega), one pair contributes

$$
\delta H_{\alpha\beta}=g^2\frac{d}{\Delta^2}
\left[\frac{P_{\alpha\beta}^*}{\Delta-i\omega}
      +\frac{P_{\alpha\beta}}{\Delta+i\omega}\right],
$$

$$
(Z_R)_{\mu\alpha}=g\frac d\Delta
\left[\frac{p_{\mu\alpha}}{\Delta-i\omega}
      -\frac{p_{\mu\alpha}^*}{\Delta+i\omega}\right],\qquad
(Z_L)_{\alpha\mu}=g\frac d\Delta
\left[\frac{p_{\mu\alpha}^*}{\Delta-i\omega}
      -\frac{p_{\mu\alpha}}{\Delta+i\omega}\right].
$$

The existing k-point weights and spin prefactor multiply these expressions.
They use the difference of the two occupation products, because
f_m(1-f_n)-f_n(1-f_m)=f_m-f_n. At nonzero frequency, Z_L need not equal
Z_R dagger. Changing m and n simultaneously conjugates p and P and changes
the signs of d and Delta, leaving every contribution unchanged. Dropping
negative d therefore violates invariance under a joint band permutation.
Equal occupations contribute zero before division by Delta; nonzero
occupation difference at zero gap is outside this interband formula.

Internally the left wing is stored as Z_L dagger, with the same auxiliary
row and Cartesian column layout as Z_R. Both stored arrays transform with
the same auxiliary-basis matrix. The general thin kernel independently forms
E Y and E dagger Y, and the final body correction is A_R K A_L, with
A_R=D Z_R and A_L=Z_L D. A complex basis change must transform both sides;
bare conjugation of a forward vertex is justified only in the real auxiliary
representation used by the RI band contraction, before that basis change.

The linked response audit records the independent ordered-transition and
directional-inverse controls, the preserved invalid degenerate-occupation
probe, and the compiled-validation status. These controls do not establish
metallic intraband response, convergence, or a material topology.


### Conditioning of the signed inverse transform

In the real auxiliary representation define W_R^+(omega)=W(R,+i omega).
Hermitian analyticity gives W_R^-(omega)=[W_{-R}^+(omega)]^dagger. The finite
inverse transform used by the complete response route is

$$
W(R,\tau_t)=\frac12\sum_j\left[
 C_{tj}(W_R^+(\omega_j)+W_R^-(\omega_j))
 -iS_{tj}(W_R^+(\omega_j)-W_R^-(\omega_j))\right].
$$

Consequently, machine-scale differences in W(i omega) cannot be judged
independently of the row sums of C and S. An entrywise bound for a common
frequency-space perturbation of magnitude eta is
`|delta W(tau_t)| <= eta sum_j (|C_tj|+|S_tj|)`, before accounting for the
roundoff in the transform itself. Algebraic cancellation of positive and
negative terms does not remove that conditioning issue.

The complete periodic optical control in the linked response audit keeps
24 nodes in [0.4,4] Ha, the physical inputs and the 1e-10 Ha full-Sigma gate
fixed. With existing regulation 0, `max row-sum |S|=1.30705e11`, paired
W(i omega) differences of 6.02e-16 become W(tau) differences of 1.005e-6
and Sigma differences of 6.593e-9 Ha. With the existing regulation option
explicitly set to 1e-12, the sine row sum is 32801.48 and the paired Sigma
difference is 3.473e-11 Ha. This is a conditioning discriminator, not a
new default or an absolute-accuracy proof: the regulation also changes the
forward fits, and cross-setting Sigma differences reach 4.663e-8 Ha. The
complete periodic optical control has no independent exact-pole Sigma
reference here. Its directional optical inversion and actual ordered
frequency-export comparisons are separately verified in the response audit.
