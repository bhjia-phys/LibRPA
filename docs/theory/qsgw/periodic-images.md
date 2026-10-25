# Symmetry-covariant images of a finite-mesh QSGW correction

QSGW supplies a complete Hamiltonian on a finite, unshifted reciprocal
mesh. Export and current-HR velocity use the same real-space reconstruction:
preserve the supplied native DFT H(R), and interpolate only the complete
current correction. A discrete Fourier transform fixes sums of equivalent
supercell images, not each image separately. The image assignment must
respect the physical centers of the two atomic orbitals in every matrix
element. Otherwise exact mesh reconstruction and Hermiticity can coexist
with broken spatial symmetry between the sampled points.

The implemented change chooses all shortest physical atom-pair images with
equal weights. It changes the off-mesh interpolant and its derivatives,
while preserving every native matrix. It is not a symmetry projection:
TR, inversion or magnetic-symmetry errors already present in the native
input remain. Better symmetry does not by itself prove better band
interpolation or a converged physical topological invariant.

## Fourier and basis conventions

Let A contain lattice vectors as rows, in Bohr. Fractional coordinates are
row vectors, and the Fourier convention is

$$
H(k)=\sum_R e^{2\pi i k\cdot R}H_R.
$$

Let D=diag(N1,N2,N3), so native points have components k_i=j_i/N_i.
Let tau_a be the fractional physical center of AO a in the same unit-cell
representative and Bloch gauge as H/S. The displacement from the bra AO
a in cell zero to ket AO b in cell R is

$$
d_{ab}(R)=(R+\tau_b-\tau_a)A.
$$

For SOC, both adjacent spin components of a spatial AO have the same center.
Every orbital on a given atom shares its atom's center. Collinear spin
channels use the same geometry independently. No real projection, spin
average or assumption of time-reversal symmetry is made.

The production task derives these centers from its existing atomic
coordinates and physical AO-to-atom map. The numerical reconstruction
receives an explicit center per AO row. Neither a new file convention nor
a new public C input field is introduced. Independently wrapping a center
into another unit cell without transforming the matrices would change the
Bloch gauge and is invalid.

## What the native mesh determines

For the target correction delta H(k_j), define the discrete coefficients

$$
C_{ab}(r)=\frac1{N_k}\sum_j e^{-2\pi i k_j\cdot r}
\delta H_{ab}(k_j),\qquad 0\le r_i<N_i.
$$

Every translation R=r+mD, with integer row vector m, has the same phase
on the native mesh. A lift can therefore assign weights w_ab(R|r) satisfying

$$
\sum_{R\equiv r\ ({\rm mod}\ D)}w_{ab}(R|r)=1,
\qquad (\delta H_R)_{ab}=w_{ab}(R|r)C_{ab}(r).
$$

Summing the Fourier series at a native point recovers the complete target
exactly, including any symmetry defect in that target. The finite-grid
identity does not determine which lift is physically appropriate off mesh.
The native DFT blocks are added unchanged after lifting the correction.

The implemented lift selects

$$
\mathcal M_{ab}(r)=\left\{r+mD:\ m\in
\operatorname*{arg\,min}_{n\in\mathbb Z^3}
\|(r+nD+\tau_b-\tau_a)A\|^2\right\}
$$

and gives each minimizer weight 1/|M_ab(r)|. Weights are shared by all
orbital and spin entries on the same ordered center pair. Exact equal
centers must be grouped using an exact ordering; a tolerance-based map
comparator is not a valid strict weak ordering and must not silently merge
distinct centers.

## Why the previous rectangular images could break inversion

For inversion about the chosen origin plus fractional translation v, write
v-tau_a=tau_p(a)+t_a, where t_a is an integer cell translation and p(a)
is the mapped atom. Including local orbital parity s_a=(-1)^l_a,

$$
P(k)_{p(a),a}=s_a e^{2\pi i k\cdot t_a},\qquad
H(k)=P(k)^\dagger H(-k)P(k).
$$

The corresponding real-space condition is

$$
(H_R)_{ab}=s_as_b(H_{t_b-t_a-R})_{p(a)p(b)}.
$$

The old lift used the same centered rectangular translations for all AO
entries, splitting an even-mesh Nyquist coefficient between +/-N_i/2.
It respects R -> -R but generally fails the additional atom-dependent
translation t_b-t_a. On the native mesh the extra images remain equivalent,
so its Fourier-roundtrip check cannot expose the failure.

An exact one-dimensional example makes the distinction explicit. Let
q=2 pi k, P(q)=diag(1,exp(iq)), and

$$
H(q)=\begin{pmatrix}0&f(q)\\f(q)^*&0\end{pmatrix},\qquad
f(q)=e^{3iq}+e^{-2iq}.
$$

It satisfies f(q)=exp(iq)f(-q), hence inversion, as well as Hermiticity
and spinless TR. On six samples the old lift replaces exp(3iq) with
cos(3q). Its off-mesh inversion defect is

$$
\widetilde f(q)-e^{iq}\widetilde f(-q)
=-i(1+e^{iq})\sin(3q).
$$

This vanishes at every sample and is nonzero between them. Thus an exact
mesh roundtrip does not imply a symmetry-covariant interpolant. The new
physical-pair distance chooses the compatible images; it need not recover
every original long-range hopping of an arbitrary undersampled model.

## Symmetry covariance of shortest images

For a spatial operation g, write
g tau_a=tau_p(a)+t_a. Let O denote its fractional rotation acting on row
coordinates on the right. The transformed hopping translation is

$$
R'=RO+t_b-t_a,
\qquad R'+\tau_{p(b)}-\tau_{p(a)}
=(R+\tau_b-\tau_a)O.
$$

The Cartesian transformation is orthogonal, so it preserves the physical
norm. If O preserves the sampling superlattice, meaning

$$
D O D^{-1}\in GL(3,\mathbb Z)
$$

in this row-vector convention, it maps the whole candidate set bijectively
onto the transformed candidate set. The nearest-image sets and their
multiplicities therefore correspond exactly. Equal weights are invariant
under that bijection. With column coordinates the same condition is written
D^-1 O D integer; the two notations must not be mixed.

Orbital and spin rotations mix entries within mapped atomic blocks, where
the image weight is the same real scalar. Weighting commutes with these
rotations and with complex conjugation. The lift therefore preserves
unitary symmetries and antiunitary magnetic symmetries already obeyed by
the sampled model, provided the basis supports the relevant rotations.
It does not impose a symmetry absent from the input. Fractional translations
in nonsymmorphic operations cancel in the displacement difference, while
their atom-dependent integer translations are retained.

Hermiticity follows separately: the candidate set for (b,a,-r) is the
negative of that for (a,b,r), with the same distances and tie count.
Consequently H_R=H_-R^dagger remains true. For real NAOs with spinors,
onsite TR has the coefficient condition H_R=J H_R^* J^dagger; real image
weights commute with it. An onsite Zeeman term remains TR breaking.

Inversion and TR are compatible with every diagonal unshifted mesh.
Other spatial operations may fail the superlattice condition, for example
a rotation exchanging axes with different mesh periods. This routine does
not manufacture the missing symmetry of an incompatible mesh.

## Complete search in a skew lattice and tie handling

A fixed search over the 27 neighboring images is not sufficient for a
general unreduced cell. Set B=D A and y=(r+tau_b-tau_a)D^-1. Choose any
integer m0 and let rho=||(y+m0)B||. Every minimizer m satisfies

$$
|y_i+m_i|\le\rho\,\|\operatorname{column}_i(B^{-1})\|.
$$

This follows by writing v=(y+m)B, using ||v||<=rho, and applying
Cauchy--Schwarz to (v B^-1)_i. Taking ceil/floor gives a finite complete
integer box. In the implementation, the equivalent bound uses the inverse
of A and divides column i's norm by N_i. A sufficiently enlarged radius
includes the geometric tie tolerance. Invalid nonfinite/singular geometry
and integer overflow are rejected rather than truncated.

The current squared-distance tie tolerance is
1e-10 times max(1,d_min^2), with distances in Bohr. The search first finds
the global minimum, then collects all candidates within that one tolerance;
incremental, order-dependent near-tie selection is avoided. This numerical
policy approximates exact equal-distance ties. Symmetry claims are subject
to geometry precision and that finite tolerance, not an energy threshold.
All qualifying ties receive equal weights, with no lexicographic tie break.

For example, the row lattice [[1,0,0],[10,0.1,0],[0,0,2]], unit periods,
and fractional displacement (0,0.49,0) have shortest shift (-5,0,0), with
squared distance 0.012401. A fixed neighboring-image search misses it.
Highly skew cells can make the rigorous box expensive; a reduced-basis
or sphere enumeration may optimize that search without changing its
completeness. The implementation currently performs the complete box scan.

## Shared velocity path, validation, and restart boundary

Both export and current-HR velocity call the same reconstructor with the
same centers. Finite-mesh equality of H does not imply equality of its
derivatives, so this change also changes the velocity interpolant and can
affect later head/wing feedback. Update 1 uses the entering DFT reference;
its final correction image choice cannot retroactively change the self-energy
already computed in that update.

The atom-pair displacement selects the images; it does not replace the
displacement in the Fourier derivative. With Cartesian k in inverse Bohr,

$$
\partial_{k_\alpha}H(k)=
i\sum_R (RA)_\alpha e^{i k\cdot(RA)}H_R.
$$

The factor is `(RA)_alpha`, not `((R+tau_b-tau_a)A)_alpha`, because the
operator still uses the cell-periodic Fourier phase indexed by R. The native
position matrix and overlap connection already account for basis geometry
in the localized velocity formula. Adding a separate tau contribution to
this derivative while retaining that formula would count the basis connection
twice.

Validation must retain native-mesh values, verify symmetry using operations
with nonzero cell translations and nontrivial orbital/spin structure,
cover skew cells and ties, and retain intentional magnetic symmetry
breaking. A smooth band curve or repaired symmetry is not an accuracy test:
the new interpolant must still be compared against a matching direct path.
Neither finite controls nor a single saved update establish a material
topological invariant.

The current-HR feedback checkpoint now writes version 12 (uncut) or 13
(shifted cut), with the HWV3 numerical-convention tag. It binds the shortest
atom-pair image rule and the complete fractional AO-center vector, in addition
to the existing lattice, periods and native operators. Restoring the state
requires matching centers before any live eigenpairs or Hamiltonian are
replaced. This is necessary because identical native matrices with different
image choices have different derivatives even at native k points.
Indeed, a difference satisfying
`sum_R exp(2 pi i k_j.R) delta H_R = 0` at every native point need not satisfy

$$
\partial_{k_\alpha}\delta H(k_j)
=2\pi i\sum_R R_\alpha e^{2\pi i k_j\cdot R}\delta H_R=0.
$$

Here the derivative is with respect to fractional k. Cartesian derivatives
use the corresponding lattice-vector component. The velocity formula also
uses the same unchanged native position/overlap operators, so equality of
the sampled Hamiltonian alone cannot certify an unchanged optical update.

Historical current-HR versions 10/11 use the previous image policy and omit
centers. They are rejected for continuation rather than silently changing
the feedback map. Original files remain readable by their original build
and usable for independent postprocessing; no automatic migration or fresh
GW result is implied. Reference/rotated-reference feedback keeps its existing
compatible formats. Output-only HR export, without current-HR head/wing
feedback, also keeps its existing checkpoint contract because its image choice
does not enter the update map. Matching physical H/S/r cell representatives
remain an input requirement; binding centers does not infer that match.

Implementation: `src/math/localized_operator.cpp::shortest_periodic_images`,
`src/core/qsgw/headwing_velocity.cpp::reconstruct_qsgw_hamiltonian`, and
the two callers in `src/core/qsgw/task.cpp`.
