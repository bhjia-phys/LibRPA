# Physical input, representation and completed GW operators

This note describes the input-lifetime invariant implemented by the local cache
repair candidate. Source integration is not numerical acceptance. The independent
before/fresh controls and subsequent acceptance belong to the research workspace's
`cache-lifetimes` report. No angular rule, frequency transformation, continuation,
cutoff or physical approximation is selected by this repair.

## Which mathematical object may be reused?

Let the loaded input be

\[
 X=(E,f,C,\mu,\Gamma,V,U,v,\epsilon_{\rm ext},\mathcal G),
\]

where G denotes geometry, basis conventions and k-point/symmetry metadata.
The auxiliary index of Gamma is the selected RI source; U maps its parent
auxiliary basis into a compressed basis. A numerical recipe theta includes
resolved routing, truncation/pruning, the complete quadrature and transformation
recipe, and the actual distributed representation used by a consumer.
A completed cached result A is eligible only when

\[
 A=F_\theta[X|_{\operatorname{deps}(A)}].
\]

Equivalently, at the same final loaded input and recipe,

\[
 F_\theta^{\rm reused}[X_{\rm final}]
 =F_\theta^{\rm fresh}[X_{\rm final}].
\]

This is an operator equality. A pointer, allocated arrays, a smooth spectrum or
unchanged frequency nodes cannot establish it.

For nonorthogonal AOs, C^dagger S C=I, while the coefficient density and covariant
Fock matrix have the schematic contractions

\[
 D=C f C^\dagger,\qquad
 K=-\sum_{\mu\nu}\Gamma_\mu D\Gamma_\nu V_{\mu\nu}.
\]

There is no extra S between these vertices and D. At fixed C/f/Gamma/V, changing
E or the chemical potential does not change this real-space K. It does change
Green functions and optical denominators. For example the ordered response is

\[
 \Pi_{AB}(i\omega)=\sum_{k,mn}w_k
 \frac{f_{mk}-f_{nk}}{E_{mk}-E_{nk}+i\omega}
 A_{mn}(k)B_{mn}(k)^*,
 \qquad A^\alpha_{mn}=-v^\alpha_{mn}/(E_m-E_n),
\]

for the interband optical probe. Thus an energy-only update at fixed supplied v
changes both its denominator and the inferred optical probe. Rebuilding body chi
while reusing old optical head/wings is not a coherent response. A change of C
mixing occupied and empty states changes D and hence K; a harmless rotation
inside a uniformly occupied subspace need not do so.

For a band-only target, a real-space operator is unchanged. Its new projection

\[
 A_{\rm KS}(k_b)=C_b^\dagger A_{\rm AO}(k_b)C_b
\]

must nevertheless use the new C_b and requested Fourier image convention.
Resetting only this projection is distinct from recomputing real-space Sigma.
Output directories and continuation/root-solver settings do not change that
imaginary-frequency real-space operator. New output requests still have to be
honored from the current result rather than constructor-stamped old flags.

## Setters and the collective handoff

An owner-local setter may be called on only one MPI rank. It records a local
change mask before mutating input and does not add a world collective. An already
collective compute, prepare or computed getter forms

\[
 d_{\rm global}=\mathop{\mathrm{bitwise\ OR}}_r d_r
\]

before any pointer-dependent branch. All ranks invalidate the same dependencies,
apply pending representation updates in the same global key order, and clear
pending state only after a successful handoff. EXX/GW pointer presence and their
existing successful-completion flags are included in common readiness decisions.
A failed candidate is not a completed result on a later call.

The masks distinguish spectrum, occupations, SCF C, band targets, RI coefficients,
Coulomb matrices, U, velocity, external dielectric/direct optical input, and
structural metadata. Options are compared field by field. Padding and output
strings are not physical recipes; equal node arrays alone do not imply equal
transformation weights.

A completed time-frequency generation is stamped by every actual generation
argument: count/type, effective minimum and maximum transition energies,
intervals, time arguments, regularization and signed inverse-sine selection.
An unchanged public frequency-grid query reuses the completed grid and leaves
its resident solvers valid. A genuine recipe change invalidates borrowing
chi/Sigma objects before resetting the stable Dataset-owned grid, and commits
the generation stamp only after collective success. No solver borrows a local
temporary grid. Legacy output vectors may themselves be the caller's output
buffers during this handoff; invalidation must not clear or reallocate them.

## Updating a consumed representation

Public C input is a complete matrix for each supplied (spin,component,k), with
band rows and AO columns. Its row-major buffer represents the column-major
AO-by-band matrix X=C^T. A permanent kBLACS tile is not a complete dense API
matrix. After distribution, a replacement enters a pending map rather than
being written over one local tile. Only the supplied keys are transferred:

\[
 X'_{\rm local}(k)=\operatorname{restrict}_{\rm descriptor}X_{\rm supplied}(k).
\]

Unsupplied tiles remain unchanged. Identical replicated input can be normalized;
conflicting duplicates reject collectively. Transfer work is bounded to the
current k matrix and communication chunks, not a whole-MF clone.

Root-dense k-parallel storage and replicated dense storage are different facts.
OFF consumers require complete coefficients for every k on every rank. Returning
tiles only to each k group's root does not satisfy that contract. Explicit
normalization at actual coefficient consumers performs the requested transfer;
E/grid-only setup and full-grid head-only preparation do not require unused C.
Small per-key progress sets retain truthful ownership after a failed streaming
conversion, so a retry cannot label a mixed map as completed replicated or tiled
storage. Public ready flags are committed only after the conversion completes.

Consumed two-dimensional Coulomb input is updated by supplied rectangles:

\[
 V'_{ab}(q)=\begin{cases}
 V^{\rm supplied}_{ab}(q),&(a,b)\text{ lies in a supplied rectangle},\\
 V_{ab}(q),&\text{otherwise}.
 \end{cases}
\]

Exact zeros overwrite old entries; other q points and the other full/cut channel
survive. This is value replacement, not communicator reinitialization. A genuinely
incompatible basis/q-input layout instead requires an explicit input reset and
reload after destroying objects that borrow its contexts.

## Physical optical inputs outlive a calculated dielectric object

The direct full-BZ C/velocity/member-map payload is physical input. The driver's
selected PyATB active-grid E/C/f can also differ from the main mean field. Their
existing allocations move into shared immutable owners; destroying a computed
head/wing must not erase that selection or silently replace it with a restored
representative. Selected coefficients retain their actual descriptor and local
owned-k keys independently of the main body's later representation.

Physical velocity, external dielectric samples and strict-2D monopole
normalization also remain distinct from generated analytic results. Structural
changes mark a retained selected optical frame ineligible until an explicit
compatible replacement/reload; matching dimensions alone do not prove matching
basis/frame. Explicit native H/S/r preparation selects its newly matched live
inputs and preserves their normalization, including head preparation deferred
to a later call.

Native velocity already transfers complete dense coefficients one k/component
at a time and accepts that input without a k-group context. Only permanent
tiles require the preliminary collection to k roots. Requiring an otherwise
unused k-group layout would incorrectly reject the existing dense input route.

Head, both auxiliary wings and both regular Coulomb-frame wings have separate
successful-completion states. Allocation precedes calculation and is not a ready
predicate. Failed head/wing construction cannot publish a successful candidate;
a failed regular transform can preserve a completed auxiliary source for retry.
An ordinary head-only snapshot predating C redistribution must be reconstructed
before new wings use the prepared live coefficient layout.

The public optical snapshot is rank-local. It adds no MPI collective, rejects
locally pending optical changes, and reads matching committed dimensions and
velocity. It cannot discover a remote owner's uncommunicated setter. A common
prepare must establish the committed state before distributed consumers rely
on such snapshots.

## QSGW entering and exit states

A QSGW response at iteration t belongs to E_t/C_t/f_t. Diagonalizing the update
advances the live mean field to t+1; that does not turn Sigma_t into Sigma_(t+1).
The native task notifies this update without routing its native coefficients
through dense API setters. At ordinary exit it prepares the selected final
velocity and invalidates entering-state results without another chi/W/Sigma
calculation. Later public computation must build from the final state.

The current candidate explicitly guards retained-IBZ QSGW with selected direct
full-BZ optical data: current-frame transport of that supplied payload is not
yet implemented or accepted. The input is retained rather than silently replaced
by representative restoration. This guard is an unfinished boundary of the audit,
not a completed implementation of that route. Full-grid and other finite paths
need their own QSGW-exit versus fresh controls. None of these lifetime checks
establishes convergence, numerical-quadrature accuracy or material topology.

## External sampled functions retain their frequency labels

An external dielectric input is the set of pairs
`(omega_i, epsilon_i)`, with epsilon_i=epsilon(i omega_i), rather than an
unlabelled N-vector. Direct modes0/5 may copy epsilon_i to a target grid only
when all three lengths agree and the input/target labels match pointwise.
At fixed N, replacing omega_i by a different omega'_i does not preserve the
represented function: epsilon(i omega_i) generally differs from
epsilon(i omega'_i). Reusing the physical input after TF regeneration therefore
still requires an eligibility check; preserving its owner alone is insufficient.

The implementation reuses the existing analytic head/body compatibility rule,
in internal Hartree units:

\[
 |\omega_i-\omega'_i|\le 10^{-12}\max(1,|\omega'_i|).
\]

Both labels and samples must be finite. This numerical matching contract
allows serialization/roundoff within the existing tolerance; it proves neither
frequency convergence nor a bound on the derivative of epsilon. Accepted direct
samples remain byte-for-byte the same floating values. The ordinary GW caller
checks supplied data through a common collective error handoff before chi/EXX;
the core copy function also checks its own contract for other callers. Empty
external input retains its existing no-external-head behavior. A mismatch asks
for regenerated data or an explicitly selected interpolation/fitting mode,
without silently changing the prescribed method. General SOC optical0/5
screening remains separately guarded; fixing labels does not accept that route.

The C++ two-vector boundary must validate the pair before exposing raw buffers:

\[
 N_\omega=|\texttt{omegas}|=|\texttt{epsilon}|=N_\epsilon=N,
 \qquad 0\le N\le\texttt{INT\_MAX}.
\]

Passing only N_omega with epsilon.data() loses N_epsilon. If the latter is
shorter, the C setter can read outside that vector before a later Dataset-size
check; if it is longer, values are silently discarded. The C++ wrapper therefore
rejects unequal lengths and an unrepresentable C count before invoking the raw
setter. Rejection leaves the previous physical input and dirty state untouched.
Matching empty vectors retain the existing explicit-clear behavior. Raw-pointer
C callers remain responsible for providing N readable values in each buffer;
the vector guard is not a new guarantee about arbitrary raw-pointer allocations.
