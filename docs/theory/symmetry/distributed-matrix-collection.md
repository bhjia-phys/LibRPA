# Local matrix collection, zero ownership and scaling

A distributed descriptor assigns local row and column counts m_r and n_r.
Its physical entries are the Cartesian product of the two local index sets,
with global indices supplied by the descriptor's local-to-global maps:

\[
 M^{(r)}_{ij}=M_{g_r(i),g_c(j)},\qquad
 0\le i<m_r,\quad 0\le j<n_r.
\]

In particular, m_r>0 and n_r=0 is a legal ownership pattern. It has no
physical matrix entries even though a row loop has iterations. The temporary
row vector has length n_r and the intended copy has byte count

\[
 b_r=n_r\,\mathrm{sizeof}(T).
\]

When b_r=0, no row-vector element, destination pointer arithmetic or memcpy
argument needs to be evaluated. A zero byte count does not make element0 of
an empty vector valid. The seven tensor/matrix-transform collectors therefore
form pointers and copy only for nonzero b_r. Their existing temporary creation,
matrix-major conversion and destination replacement still run. No enclosing
collective operation or another rank's numerical work is skipped.

This distinction also preserves the existing empty-matrix representations.
Some callers use logical zero extents; others retain a padded one-by-one buffer
while their descriptor owns zero entries. Allocation size alone is not an
ownership test. Full numerical assembly must use the descriptor, and test
ROW/COL major plus the requested representation separately. The complete
ordered collector already rejects work on either zero local extent; its
existing guard is retained as a control.

The scaling parameters have a separate value contract. For the tensor and
complete symmetric tensor collectors, after the prescribed lower-block
completion the replacement value is

\[
 M_{ij}=\alpha\,A_{ij}.
\]

Conjugation of a stored opposite block acts on the source element before
multiplication by alpha, matching the corresponding pointer and sparse
collectors. A unit-alpha numerical baseline cannot establish this nonunit
contract. Scaling tests and zero-ownership tests must therefore be kept
distinct; neither is a performance benchmark or a physical GW validation.

The finite control route first checks ordinary nonempty layouts against
independent complete matrices. Legal empty layouts are executed only on the
guarded implementation; there is no old erroneous-access execution. Tests
retain exact global ownership, full entry coverage, major/representation and
collective completion. Source evidence of the old empty-vector expression is
separate from numerical after acceptance or an impact on historical materials.
