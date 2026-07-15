# User Approval: U3 Head Matrix Getter

Approval ID: `APPROVAL-U3-HEAD-MATRIX-GETTER-001`

Decision: **approved**

Approved at: `2026-07-15T16:40:50+08:00`

Approved by: workspace user in Codex task `019f0839-7e60-7770-bedc-59c75cde7fa5`

Exact user decision:

> 批准 U3-HEAD-MATRIX-GETTER-001，仅限 impact packet 中的五行只读 getter 及所列回归 observers。

## Evidence links

- Upstream change: `U3-HEAD-MATRIX-GETTER-001`
- Shared hunk: `SHARED-HEAD-GETTER-001`
- Formula row: `F-HEAD-TENSOR-READ`
- Candidate diff: `qsgw-rebase-evidence/impact/U3-head-matrix-readonly-getter.patch`
- Candidate diff SHA256: `07d3d7a3b73c8860184d32ebb3895d2b3ae56fa03d6f6f19155547c00f84c296`
- Regression observer: `g0w0-upstream-vs-rebased`
- Regression observer: `qsgw-iter0-vs-upstream`
- Full observer design: `qsgw-rebase-evidence/impact/U3-head-matrix-readonly-getter.md#required-observers`

## Scope boundary

This approval covers only the exact five added lines implementing the inline `const noexcept` read-only getter `diele_func::get_head_matrices()` and the observers listed in the impact packet. It does not approve any change to `dielecmodel.cpp`, G0W0, GW, EXX, chi0, epsilon, LibRI, symmetry, normalization, MPI ownership, class data layout, or other shared numerical logic.
