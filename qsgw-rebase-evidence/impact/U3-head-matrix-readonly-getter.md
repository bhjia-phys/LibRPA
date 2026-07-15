# U3 Impact Packet: Read-only Head Tensor Accessor

Status: **approved for the clean candidate, limited to the exact hunk and observers below**

U3 ID: `U3-HEAD-MATRIX-GETTER-001`

Formula row: `F-HEAD-TENSOR-READ`

Candidate hunk: `src/core/dielecmodel.h`, inline method `diele_func::get_head_matrices()`.

## Why the shared interface is requested

The upstream dielectric implementation computes and stores the full complex `3 x 3` head tensor for every imaginary-frequency point in the private member `diele_func::head`. Existing public `get_head_vec()` exposes only the derived scalar dielectric-head vector, which is insufficient for the required per-iteration tensor comparison and Hermiticity/structure evidence.

The QSGW adapter does not need a new numerical algorithm. It needs a read-only view of the tensor that upstream has already computed.

## Call chain

1. `driver::run_qsgw`
2. `refresh_headwing(Dataset&, Options&, const MeanField& live)`
3. `diele_func::get_meanfield_df() = live`
4. upstream `diele_func::init(...)`
5. upstream `diele_func::cal_head()`
6. QSGW-only `copy_head_tensor(const diele_func&, frequencies)`
7. proposed `diele_func::get_head_matrices() const noexcept`
8. QSGW trace output `qsgw_matrices.dat`

No G0W0, GW, EXX, chi0, epsilon or symmetry caller invokes the proposed method.

## Exact candidate diff

```diff
 public:
     double cal_factor(std::string name);
     void test_head();
     std::vector<double> get_head_vec();
+    const std::vector<matrix_m<std::complex<double>>>&
+    get_head_matrices() const noexcept
+    {
+        return head;
+    }
     bool has_wing() const { return !wing.empty() || !wing_mu.empty(); }
```

## Numerical and ABI impact

- Adds no data member and no virtual function.
- Changes neither class layout nor existing method signatures.
- Performs no allocation, copy, reduction, mutation or floating-point operation.
- Returns `const&`; callers cannot mutate through this interface.
- Is inline and `noexcept`.
- Existing G0W0/head-wing call ordering and output remain unchanged unless a new QSGW caller explicitly invokes the method.

## Required observers

1. Candidate configure/build and complete CTest.
2. Pure-upstream versus candidate G0W0 direct A/B on byte-identical no-symmetry input.
3. Upstream head-wing regressions unchanged.
4. QSGW head-only/head-wing trace contains the full tensor and satisfies the declared matrix comparator.
5. Protected-diff check confirms this is the only shared numerical-source hunk in the QSGW candidate.

## Approval boundary

Approval, if granted, applies only to the exact five added lines above, U3 ID `U3-HEAD-MATRIX-GETTER-001`, formula row `F-HEAD-TENSOR-READ`, and the listed observers. It does not approve changes to `dielecmodel.cpp`, G0W0, GW, EXX, chi0, epsilon, LibRI, symmetry, normalization, MPI ownership or any other shared numerical logic.

## Approval record

The workspace user granted exact approval on 2026-07-15. The evidence-linked record is `qsgw-rebase-evidence/impact/APPROVAL-U3-HEAD-MATRIX-GETTER-001.md`. The approved candidate patch has SHA256 `07d3d7a3b73c8860184d32ebb3895d2b3ae56fa03d6f6f19155547c00f84c296`.
