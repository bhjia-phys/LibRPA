# QSGW Hartree legacy-oracle audit

Status: local source-and-artifact audit complete; corrected remote parity run
not yet executed.

## Scope and evidence identity

This note audits the rejected Gate 6b/7 Hartree comparison and defines the
only acceptable replacement. It does not promote historical diagnostics to a
numerical benchmark.

- Attribution job: `2351376`
- Hartree-off side: old v36 job `2348804`
- Hartree-on side: oracle v40 job `2348987`
- Compared trajectory: iterations 0 and 1
- Attribution provenance explicitly records:
  `benchmark_acceptance=false` and `diagnostic_only=true`
- Comparison JSON SHA256:
  `6057e537b75a2d91a0a279ea2b750b5fda92940024a4a1947a132edc44b126f2`

Primary archived evidence:

- `remote/dongfang-gates-20260715-7d69a18c/gate6b7-attribution-v3-2351376/PROVENANCE.txt`
- `remote/dongfang-gates-20260715-7d69a18c/gate6b7-attribution-v3-2351376/old-off-vs-old-on-comparison.json`
- `remote/dongfang-gates-20260715-7d69a18c/gate6b7-attribution-v3-2351376/attribution-component-summary.txt`

## Proved failure signature

The old Hartree-on and Hartree-off trajectories have exactly equal zero
`delta_vh` over 64 blocks and 43,264 scalar entries, but other components
diverge before a nonzero Hartree response exists:

| Component | Maximum absolute difference (Ha) |
| --- | ---: |
| `delta_vh` | 0 |
| `exx` | 23.95446771085701 |
| `raw_h` | 23.894972653582414 |
| `mixed_h` | 4.778994530716478 |
| `sigma_c_iw` | 1.9978797879137928 |
| `vc` | 0.3957145291588309 |
| `gap` | 0.020185348744305165 (about 0.549 eV) |

This signature rules out a Hartree-potential numerical effect as the cause of
that comparison: the dumped Hartree increment is zero while EXX, GW, and the
Hamiltonian already differ.

## Proved reader contamination

The audited v36 source has SHA256
`6c79c6731d3420b3c9f70c97dea8c63236bca2952216bb67d9e6b9839552c9d1`.
The v40 full-reader source has SHA256
`ecb95fdf80caac32254a0fed7905ff29d6481fe9fe355cb3f809c6c0e0d6ed5e`.
Their only functional source change is:

```diff
 if (parallel_routing == R_TAU
-    || need_full_cut_coulomb_for_abacus_symmetry())
+    || need_full_cut_coulomb_for_abacus_symmetry()
+    || oracle_env_bool("QSGW_ORACLE_UPDATE_HARTREE", false))
```

In the ordinary v36 path, `read_Vq_row(...)` populates the distributed
`Vq_cut` used by EXX/GW. The v40 condition instead calls `read_Vq_full(...)`
into that same global `Vq_cut`, and the Hartree block later calls
`FT_Vq(Vq_cut, ...)`. Consequently, enabling the oracle changes the Coulomb
representation consumed by non-Hartree code. The archived EXX/GW divergence
with zero `delta_vh` is the observed consequence of this shared-state change.

The rejected v40 executable is therefore not a legacy Hartree oracle.

## Normalization audit

The frozen legacy LibRI source
`legacy_LRI-cal_hartree.keydiag-v2.hpp` (SHA256
`7c8a3ae24f0ab8852d916d8a3c1532a5e3aadb8e17b6c5c7ef1569f919420dc4`)
contains an explicit kernel factor:

```cpp
// N_mu = 1/Nk sum_nu Vq_mu_nu * M_nu
const Tdata fac = Tdata(1.0 / static_cast<double>(nk));
N_Mu = fac * N_Mu;
```

The frozen old LibRPA `src/hartree.cpp` (SHA256
`b7a7acf2b22a6eb729b86b52883b9ba49ac45faaa8c1c59603e3d5cf7fd7382c`)
then performs the k-to-R transform with another `1/nk` at line 308. Thus:

- `legacy_extra_inverse_nk` is the required current-code setting for exact
  legacy numerical parity.
- `weighted_occupations` removes the legacy kernel's additional `1/nk` and is
  the current physical-default hypothesis for already weighted densities.
- The second statement is not accepted by source inspection alone; it requires
  independent contraction and metamorphic numerical validation.

## Required replacement acceptance

### Corrected legacy parity lane

1. Start from the exact audited v36 source and executable provenance.
2. Preserve the normal distributed `Vq_cut` used by EXX/GW.
3. Stage `read_Vq_full(...)` into a dedicated Hartree-only map, then restore
   every global reader state item before entering the iteration.
4. Run legacy and candidate on the same frozen full-BZ bundle with
   `legacy_extra_inverse_nk`, linear beta 0.2, and two completed updates.
5. Require iteration 1 to equal Hartree-off because the density increment is
   zero; compare iteration 2 `delta_vh`, EXX, Sigma, raw/mixed Hamiltonians,
   eigenvalues, and summaries at the project thresholds.

### Current physical-default lane

1. Run symmetry-on `qsgw` and `qsgw_band` for two completed updates with
   `weighted_occupations`.
2. Independently recompute the `C(k)-V(q=0)-density` contraction from frozen
   producer files and the dumped density increment.
3. Require the iteration-1 Hartree-on trajectory to equal a Hartree-off control
   in every non-Hartree component and require `delta_vh` to be zero.
4. Validate density charge/Hermiticity, exact full-k/R order, BvK remap,
   inverse Fourier closure, fixed-basis projection, and grid/band equivalence.

The two lanes answer different questions and must not be substituted for each
other. No legacy Hartree numerical parity claim is valid until the corrected
legacy lane has run successfully on the frozen same-dataset bundle.
