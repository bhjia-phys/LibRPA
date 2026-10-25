# RPA correlation energy from a general magnetic charge response

This is the RPA-energy consumer of the signed charge response derived in
[the magnetic screening note](nonreciprocal-screening.md). It keeps the
existing Coulomb, frequency nodes and weights, and zero-temperature gapped
reference. It introduces no metallic intraband model and changes no minimax
regularization. The native control is
[`test_rpa_signed_frequency.cpp`](../../../src/test/test_rpa_signed_frequency.cpp).
Execution and numerical acceptance belong to the
[audit report](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/rpa-energy-audit.md).

## The half-line formula does not require Hermitian chi at positive frequency

Let the bare Coulomb matrix be Hermitian and let the time-ordered charge
response obey Hermitian analyticity,

\[
 v(q)^\dagger=v(q),\qquad
 \chi(q,-i\omega)=\chi(q,+i\omega)^\dagger.
\]

Define the scalar frequency integrand

\[
 g(q,z)=\operatorname{Tr}[v(q)\chi(q,z)]
          +\log\det[I-v(q)\chi(q,z)].
\]

The product `v chi` need not be Hermitian. Nevertheless cyclicity of the
trace and `det(I-AB)=det(I-BA)` imply

\[
 g(q,-i\omega)=g(q,+i\omega)^*.
\]

For the logarithm this relation uses the stable dielectric branch connected
continuously to the identity at infinite frequency. The real part is
`log(abs(det(E)))` and is unaffected by the integer logarithm branch. It
does not license continuing through a singular dielectric matrix; failed
LU factorization remains an error.

Pairing the two frequency halves gives the existing positive-frequency
normalization

\[
 E_c^{\rm RPA}
 =\frac1{4\pi}\sum_q w_q\int_{-\infty}^{\infty}d\omega\,g(q,i\omega)
 =\frac1{2\pi}\sum_q w_q\int_0^\infty d\omega\,
       \operatorname{Re}g(q,i\omega).
\]

This takes the real part of the **scalar energy after** evaluating the full
matrix determinant. It is not equivalent to replacing chi by
`(chi+chi^dagger)/2` before the determinant. The odd matrix part can affect
the real determinant even when its contribution to the linear trace is zero.
The native control reports that erroneous Hermitian-projection energy
separately and requires a nonzero difference in the magnetic Gamma model.

Likewise, q/-q folding is distinct from the frequency pairing. In a real
auxiliary gauge, charge reality gives
`chi(-q,+iw)=chi(q,+iw)*` and the corresponding relation for v. Thus the
two q points have equal **real** energy integrands, and their geometric
weights may be combined. Crystal unitary transport preserves trace and
determinant; positive-frequency antiunitary transport uses a transpose and
also preserves these scalars. No same-frequency Hermitian completion is
inferred from any of these statements.

## Native matrix and ownership contract

`compute_RPA_correlation_blacs_2d` keeps the Coulomb matrix in its existing
upper/Hermitian atom-pair representation. For
`Chi0::full_frequency_matrix()==true`, it instead requests all ordered chi
blocks and uses the strict general collector. Both off-diagonal atom
directions and both triangles inside an atom-diagonal block are independent
data at positive imaginary frequency.

Every rank participates in the block-cyclic redistribution, including a
rank with no requested atom pair or no local matrix elements. A rank-local
empty communication map is not a condition for skipping that collective.
The local buffer is zeroed before collection. The general chi collector
continues to reject missing required entries rather than silently filling
them by conjugation.

The body calculation forms `v chi`, extracts its trace, forms `I-v chi`, and
uses the existing general complex LU determinant. Its pivot array has the
ScaLAPACK required extent `m_local+mb`, including an empty row owner. The
LU status is checked before the scalar is consumed. This pivot-extent
contract also applies to `compute_rpa_response_trace_logdet_blacs_2d`, used
by the optical helper.

The public RPA API previously selected the real Gamma-only routine solely
from one k point and absent optical correction. The Gamma entry now delegates
a full-frequency spinor response to the complex BLACS2D body routine.
Consequently a complex auxiliary gauge at Gamma retains its imaginary
Coulomb entries. Scalar input retains the original real Gamma shortcut.
The complex body routine uses CPU BLACS, as it already did for general k;
this extension is not validation of a GPU general-response RPA implementation.

## Numerical question and retained boundaries

The control supplies exact finite-model Lindhard samples to the energy
consumer. A separate explicit 1x1/3x3 determinant supplies the expected
trace/log quadrature on the same 16-node Gauss-Legendre frequency grid.
It does not call the native response time transform, matrix collector or LU
for that expected energy. It therefore isolates the RPA consumer from the
previously measured inverse-sine conditioning problem, which RPA energy
does not need.

The cases cover a reciprocal Gamma model and a nonreciprocal magnetic
Gamma model, a dispersive three-q model, real and complex unitary auxiliary
gauges, one/two/three atom partitions, q/-q weights, a one-auxiliary-function
case with empty MPI tiles, and the retained scalar route. The sum of all
full-q direct-oracle contributions is compared even when the native route
uses the folded q representatives. Energy and each q contribution have
an absolute implementation gate of `3e-12 Ha`.

These are finite-grid operator checks, not convergence of a material RPA
energy or proof that 16 frequency nodes suffice universally. General SOC
optical energy remains explicitly guarded until its complete head/wing
energy consumer is independently checked. Legacy row and non-BLACS/Pi
consumers retain their temporary general-SOC guards; the active BLACS2D
body route is the scope of this repair.

The metadata-corrected native executions pass all 32 cases at one and four
ranks. Maximum energy/q-contribution error is `1.57991e-14 Ha`; the complex
Gamma dispatch error is at most `1.11760e-14 Ha`. The deliberately incorrect
Hermitian projection changes the expected energy by up to `8.96121e-10 Ha`.
The [raw execution](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/ctest-rpa-metadata-detail.log)
and [case-level record](../../../../../symmetry/runs/20261023-soc-gw-symmetry-audit/rpa-energy-verification.json)
support this bounded consumer acceptance. They use the same specified grid
for native and direct-oracle quadrature, so the measured difference is an
implementation error, not a continuum-frequency integration error.
