# exact847 legacy W(R) diagnostic

## Result

The diagnostic-only build replaced the current upstream direct full-`W(R)`
accumulation with the pre-`318e3e42` generic irreducible-sector accumulation
and real-space symmetry restore. The accepted QSGW branch and executable were
not modified. The completed numerical prefix and its SHA-bound no-rerun
postcheck are:

- source run:
  `/home/bhj/ai-runs/librpa-qsgw-gate-a1-exact847-legacy-wr-diagnostic-20260722-e5407096-v3`;
- postcheck:
  `/home/bhj/ai-runs/librpa-qsgw-gate-a1-exact847-legacy-wr-postcheck-20260722-ca010863-v1`;
- classification SHA256:
  `f6ce042bfbcf0f0f6010333c4e93c5d0acf3a1f13ce5a866fd591cca17c0a140`.

The source runner stopped after a successful LibRPA update because one
postprocessing dependency was absent. The postcheck binds the failed marker,
stdout, matrix trace, `SigcRF` tree, exact847 component tree, and current
accepted trace before reading those outputs. It emits `POSTCHECK_COMPLETE`
only and does not promote the source run to an accepted benchmark.

## Numerical comparison

Restoring the old generic route did not recover exact847:

| Comparison | Maximum absolute difference | Relative Frobenius difference |
|---|---:|---:|
| exact847 vs current direct full-`W(R)` `SigcRF` | `5.0189e-4 Ha` | `5.1685e-3` |
| exact847 vs diagnostic old generic `SigcRF` | `1.0527e-3 Ha` | `1.0420e-2` |
| current vs diagnostic `SigcRF` | `5.5102e-4 Ha` | `5.3224e-3` |
| exact847 vs diagnostic fixed-basis `SigmaC(iw)` | `3.9529 Ha` | `1.2793` |

The frequency grids agree exactly. `H0` and `Vxc_DFT` also agree exactly;
EXX differs by `1.25897e-4 Ha` maximum. The large projected Sigma difference
is consistent with amplification of the real-space AO difference through the
ill-conditioned nonorthogonal AO-to-KS projection already isolated by the
restart probe. It is not evidence that the current projection implementation
is wrong: feeding exact847 `SigcRF` to the current reader/projector reproduces
the exact847 projected Sigma to `2.73e-10 Ha` maximum.

## Inference

The generic q-to-R change in upstream commit `318e3e42` is not sufficient to
explain the exact847 mismatch. Reverting only that route makes the real-space
Sigma disagreement larger. The next controlled boundary is therefore:

1. compare exact847's ABACUS-specific irreducible-sector accumulator with the
   pre-`318e3e42` generic implementation used by this diagnostic;
2. dump and compare `Wc(q)` before either q-to-R route if the implementations
   do not explain the difference.

## Claim boundary

This result rejects one causal hypothesis only. Gate A1 one-update parity,
miniter2/miniter5, Hartree, qsgw_band, H_QSGW cut, and PR readiness remain
unaccepted. It does not authorize a permanent edit to shared GW/EXX source.
