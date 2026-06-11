# Input Parameters

For Driver Usage, you can create a file named `librpa.in` in your working directory and input the parameter settings as follows:

```
task = rpa
nfreq = 16
gf_R_threshold = 1e-4
```
If `librpa.in` or the related keyword is not found, the default value will be used.

## Common Parameter Settings for LibRPA

| Parameter Name         | Description                                                           | Type   | Default Value (Options)      |
|------------------------|-----------------------------------------------------------------------|--------|------------------------------|
| `task`                 | Task type                                                             | string | rpa (rpa, g0w0, exx)         |
| `tfgrid_type`          | Type of time-frequency integration grid                               | string | minimax (minimax)            |
| `nfreq`                | Number of frequency integration grid points                           | int    | 6                            |
| `gf_R_threshold`       | Real-space Green's function screening threshold for response function | double | 1e-3                         |
| `cs_threshold`         | Auxiliary basis coefficient tensor screening threshold                | double | 1e-4                         |
| `parallel_routing`     | Parallel scheme of LibRPA                                             | string | auto (atompair, rtau, libri) |
| `use_scalapack_ecrpa`  | Flag to use ScaLapack for calculating $E_\text{c}^{\text{RPA}}$       | bool   | true                         |
| `debug`                | Flag to enable debug mode                                             | bool   | false                        |

## Analytic Continuation Parameters

These parameters affect Pade analytic continuation in GW and QSGW tasks.
The default keeps the original Thiele reciprocal-difference continuation.
`ridge` is an opt-in robust mode for noisy or pair-sensitive QSGW diagnostics.
Setting `pade_ridge_lambda = 0` does not restore the old Padé path; it means an
unregularized rational least-squares ridge model. Use `anacon_method = thiele`
for the legacy Thiele continuation.

| Parameter Name              | Description                                                                 | Type   | Default Value (Options)              |
|-----------------------------|-----------------------------------------------------------------------------|--------|--------------------------------------|
| `n_params_anacon`           | Number of analytic-continuation parameters; negative values use `nfreq`      | int    | -1                                   |
| `anacon_method`             | Analytic-continuation method                                                | string | thiele (thiele, pade, ridge, ridge_guard) |
| `pade_ridge_lambda`         | Ridge/L2 penalty strength for regularized rational Pade                     | double | 1e-10                                |
| `pade_ridge_den_weight`     | Multiplier on denominator-coefficient ridge penalty                         | double | 1                                    |
| `pade_denominator_floor`    | Minimum denominator magnitude used when evaluating ridge Pade               | double | 1e-12                                |
| `pade_thiele_den_cut`       | Thiele denominator cutoff used only by `ridge_guard`                        | double | 1e-3                                 |

Recommended diagnostic settings:

```text
# Low-bias one-shot or first-iteration diagnostic
n_params_anacon = 16
anacon_method = ridge
pade_ridge_lambda = 1e-6
pade_ridge_den_weight = 10

# Stronger multi-iteration diagnostic
n_params_anacon = 12
anacon_method = ridge
pade_ridge_lambda = 1e-3
pade_ridge_den_weight = 10
```

`ridge_guard` is kept for diagnostics but is not the recommended robust mode,
because guarding only on the Thiele denominator can miss pairwise amplification
cases. See {doc}`../regularized_pade_ac` for validation notes and limitations.

For details on all parameters, you can visit the API documentation of struct {librpa}`LibRPAParams`.
