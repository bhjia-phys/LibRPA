# Regularized Pade analytic continuation

## Current LibRPA path

`AnalyContPade` uses Thiele reciprocal differences to interpolate
`Sigma(i omega)` and evaluates the resulting continued fraction on the real
axis. `n_params_anacon` controls how many input points are selected. This branch
also fixes the `n_params_anacon < nfreq` point-selection bug by assigning the
last selected `par_x`.

The relevant QSGW/GW call sites construct one continuation object per
`Sigma_mn(i omega)` matrix element and evaluate at `epsilon_m - E_F` and `0`.

## Ridge rational model

Use a normalized rational form

```text
R(u) = P_m(u) / Q_n(u),
u = z / x_scale,
Q_n(u) = 1 + b_1 u + ... + b_n u^n.
```

For Matsubara samples `(u_i, y_i)`, with `y_i = Sigma(i omega_i) / y_scale`,
linearize by

```text
P_m(u_i) - y_i Q_n(u_i) ~= 0.
```

This gives a complex linear least-squares system

```text
A theta ~= y,
theta = (a_0 ... a_m, b_1 ... b_n).
```

The regularized objective is

```text
min_theta ||A theta - y||_2^2
        + lambda * sum_{j>=1} j^2 |a_j|^2
        + lambda * den_weight * sum_{k>=1} k^2 |b_k|^2.
```

The implementation solves the normal equations with a small fallback diagonal
shift if needed. `n_params_anacon` is the number of unknown coefficients; ridge
uses all Matsubara points instead of discarding points.

## Runtime knobs

```text
anacon_method = thiele       # default old behavior
anacon_method = ridge        # always use ridge rational fit
anacon_method = ridge_guard  # use Thiele unless the target denominator is small
n_params_anacon = 16
pade_ridge_lambda = 1e-6
pade_ridge_den_weight = 10
pade_denominator_floor = 1e-12
pade_thiele_den_cut = 1e-3
```

`pade_ridge_lambda = 0` does not recover the old Thiele path. It selects an
unregularized rational least-squares fit with the ridge numerator/denominator
model. Use `anacon_method = thiele` for legacy continuation behavior.

## Dongfang offline validation

The helper `tools/compare_ac_regularization.py` replays dumped QSGW AC inputs
and compares methods without rerunning the full QSGW driver. It was submitted
through `tools/run_ac_regularization_compare.sbatch`.

Key result on the sensitive H2O `t32_a_vs_t32_b`, one-iteration dump
`h2o_qsgw_preac_chain_t32_pair_20260530_preac_bin_filter_iter1`:

```text
method                 QP max          QP p99          amp          min |Q|
thiele32               3.4416e-02      3.8619e-04      2.268e6      6.60e-05
ridge16 lambda=1e-6    3.8244e-05      1.3901e-06      8.271e2      9.81e-03
ridge16 lambda=1e-8    1.4177e-04      3.1104e-06      2.302e3      2.39e-02
ridge32 lambda=1e-10   7.2187e-05      3.9465e-06      2.801e3      1.63e-02
```

On the 3-iteration composite dump, `ridge16 lambda=1e-6` consistently lowers
the `t32_a_vs_t32_b` QP max in iterations 1-3, but it is not uniformly best for
the more aggressive `t1_a_vs_t32_a` comparison. `ridge_guard` based only on the
Thiele denominator is conservative and does not catch every pairwise-amplifying
case.

## Averaged/filtered Pade check

The offline tool also supports a deterministic averaged Pade candidate:

```text
avg:n_min:n_max:den_cut:trim[:max_abs]
```

It builds a family of lower-order Thiele fits, rejects non-finite values and
small target denominators, and returns a trimmed component-wise complex median.
On the same H2O `t32_a_vs_t32_b` dump, `avg12_24_dc1e-4_tr0.2` lowers the QP
max to about `2.23e-03`. This is much better than `thiele32`, but still worse
than the best ridge candidate by about two orders of magnitude. It remains a
useful robust backup and diagnostic, not the current leading implementation.

## Dongfang end-to-end validation

The original GitHub-master checkout was not the right endpoint for QSGW because
it did not include the dirty but working `acdiag`/head-wing/QSGW changes used by
the H2O diagnostics. To avoid touching that source tree, an isolated copy was
created at:

```text
/ssd/work/df_iopcas_bhj/LibRPA-acdiag-pade-regularization
```

The AC changes were merged into that copy only. It configured and built with
Intel oneAPI, `USE_LIBRI=ON`, and `USE_GREENX_API=ON`; both `rpa_exe`
(`chi0_main.exe`) and `test_analycont` built successfully. The test was run
through Slurm job `1956555` and passed.

A one-iteration H2O QSGW pair was then submitted with:

```text
nfreq = 32
n_params_anacon = 16
anacon_method = ridge
pade_ridge_lambda = 1e-6
pade_ridge_den_weight = 10
pade_denominator_floor = 1e-12
```

Run root:

```text
/data/home/df_iopcas_bhj/ai-runs/qsgw-ac-pade-regularization-20260605/h2o_qsgw_preac_chain_ridge16_l1e-6_dw10_niter1_20260605_ridge16_1iter
```

Slurm array job `1956540` completed both `t32_a` and `t32_b`. Both runs finished
successfully with finite and identical one-iteration QP summaries:

```text
HOMO = -14.4654 eV
LUMO =   1.84166 eV
Efermi = -6.31186 eV
```

The actual dumped C++ output was compared against replayed methods in Slurm
job `1956557`. For the actual ridge run:

```text
method                 QP max          QP p99          max shift from dump
thiele32 replay        2.0545e-02      2.0810e-04      5.9209e-01
ridge16 lambda=1e-6    1.5819e-05      7.5225e-07      5.7905e-12
ridge16 lambda=1e-8    8.9461e-05      1.5004e-06      6.7275e-01
ridge32 lambda=1e-10   4.8274e-05      2.0736e-06      2.9290e+00
avg12_24 filtered      1.4897e-03      6.4509e-05      6.0604e-01
```

The `5.8e-12` maximum shift from dump for `ridge16 lambda=1e-6` verifies that
the C++ code path is using the intended ridge continuation. This one-iteration
run also provides the first physical sanity check: the QP energies remain
finite and repeatable across the pair while the AC pairwise amplification is
strongly suppressed.

Longer-run status: a `NITER=3` ridge16 pair was submitted as job `1956656`.
Both cases reached the first-iteration dump with the same one-iteration
HOMO/LUMO/Efermi shown above, then failed in the next Wc construction with
`std::bad_alloc`. A matching `thiele32 NITER=3` control job (`1956714`) was not
clean either: one array member completed 3 iterations, while the other failed
with the same `std::bad_alloc` pattern after the first iteration. The common
failure site is the ScaLAPACK Wc/sqrt-Coulomb path, so the next QSGW checks used
`use_scalapack_gw_wc = f` to isolate analytic continuation from that runtime
blocker.

## Three-iteration QSGW validation

With non-ScaLAPACK Wc, the H2O `t32_a`/`t32_b` pair completed for several ridge
settings. The table below summarizes the actual C++ method in each run. QP max
is the pairwise maximum difference of replayed AC outputs. Eigen max is the
pairwise maximum difference of dumped `eigenvalues_iter*` in Hartree.

```text
candidate                 AC QP max iter1/2/3       eigen max iter1/2/3
ridge16 lambda=1e-6       1.2e-6 / 5.4e-1 / 4.3e-1  1.0e-6 / 1.88e-1 / 2.26e-2
ridge12 lambda=1e-3       1.2e-8 / 1.0e-2 / 5.8e-2  ~0     / 8.54e-3 / 5.49e-2
ridge12 lambda=1e-2       3.3e-9 / 1.2e-2 / 4.3e-2  ~0     / 1.16e-2 / 3.86e-2
ridge8  lambda=1e-1       7.4e-9 / 2.9e-2 / 1.6e-2  ~0     / 2.73e-2 / 1.36e-2
```

The replay shift from the dumped C++ output is approximately zero for the
actual ridge method in each run, confirming that the runtime path is using the
intended parameters. Stronger regularization also raises the minimum target
denominator from problematic Thiele values near `1e-4` to about `0.87` for
`ridge12 lambda=1e-2` and `0.96` for `ridge8 lambda=1e-1`.

The tradeoff is bias and feedback behavior. `ridge16 lambda=1e-6` is excellent
for the first iteration but too weak once QSGW feedback has altered the input.
`ridge12 lambda=1e-3` and `ridge12 lambda=1e-2` are the best balanced options
tested so far: they suppress the AC tail by one to two orders of magnitude
relative to Thiele while keeping the regularization bias at the p99 level below
about `4e-3 Ha` for `lambda=1e-2`. The very strong `ridge8 lambda=1e-1` gives the
smallest final-iteration pair matrix difference, but its second-iteration QP
trajectory is visibly oscillatory and should be treated as a stress-test mode,
not a production recommendation.

The main remaining physics caveat is that none of these three-iteration H2O
pairs is fully self-consistent in the sense of staying pair-identical at every
iteration. For example, `ridge12 lambda=1e-2` finishes cleanly, but the third
iteration HOMO differs by about `1.05 eV` between the pair; `ridge8 lambda=1e-1`
reduces the third-iteration split but creates a larger second-iteration
excursion. This indicates that ridge regularization fixes the worst Pade pole
amplification, but does not by itself close every QSGW feedback sensitivity.

## Working recommendation

Keep `thiele` as the default for backward compatibility. Merge `ridge` as an
opt-in robust continuation method, because the one-iteration C++ validation and
the offline sweeps show a clear reduction of pathological Pade amplification.

Suggested modes:

```text
# low-bias one-shot / first-iteration diagnostic
n_params_anacon = 16
anacon_method = ridge
pade_ridge_lambda = 1e-6
pade_ridge_den_weight = 10

# stronger multi-iteration diagnostic
n_params_anacon = 12
anacon_method = ridge
pade_ridge_lambda = 1e-3  # or 1e-2 when denominator robustness matters more
pade_ridge_den_weight = 10
```

Do not promote `ridge_guard` yet: denominator-only guarding missed several
pairwise-amplifying cases. Do not promote `ridge8 lambda=1e-1` as a default:
it is useful to bound pole pathologies, but the current H2O trajectory shows
too much feedback bias. Before changing the production default, rerun a clean
multi-iteration validation on a known-stable QSGW runtime path and compare
HOMO/LUMO, H0, eigenvalues, and AC pairwise tails across more molecules and at
least two deterministic repeats.

## Literature anchors

- Vidberg and Serene, "Solving the Eliashberg equations by means of N-point
  Pade approximants", J. Low Temp. Phys. 29, 179-192 (1977),
  https://doi.org/10.1007/BF00655090. This is the classic Matsubara-frequency
  Pade AC reference and matches LibRPA's current Thiele continued-fraction
  lineage.
- Gonnet, Guettel, and Trefethen, "Robust Pade Approximation via SVD", SIAM
  Review 55, 101-117 (2013), https://doi.org/10.1137/110853236. The relevant
  lesson is to treat near-singular rational fitting and Froissart doublets as
  numerical linear-algebra pathologies, not merely physics noise.
- Schoett et al., "Analytic continuation by averaging Pade approximants",
  Phys. Rev. B 93, 075104 (2016), https://doi.org/10.1103/PhysRevB.93.075104
  and https://arxiv.org/abs/1511.03496. This motivates the next candidate:
  average or filter a family of Pade fits instead of trusting a single
  high-order interpolant.
- Leucke et al., "Analytic continuation component of the GreenX library:
  robust Pade approximants with symmetry constraints", JOSS 10, 7859 (2025),
  https://doi.org/10.21105/joss.07859. This is a recent GW-oriented reference
  for multiple precision, greedy stabilization, and optional symmetry
  constraints.
