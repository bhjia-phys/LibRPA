# QSGW head/wing live-state ownership audit

Candidate commit: `ce1e1859316e1b659df0178b553bbbfba2ca1173`

Scope: full-BZ, no-crystal-symmetry QSGW only. This is static ownership
evidence; the head-only and head-plus-wing numerical gates remain required.

## Same SCF grid

- `diele_func::velocity_` is declared as `const velocity_matrix_t&` in
  `src/core/dielecmodel.h` and initialized directly from the constructor
  argument. It is not a value copy.
- `read_headwing_input` constructs `Dataset::p_headwing` from
  `Dataset::velocity_matrix`. The QSGW fixed-basis update writes the live
  velocity back to that same dataset member as
  `v_n = U_n^dagger v_0 U_n`.
- `refresh_headwing` assigns the current live `MeanField` to
  `p_headwing->get_meanfield_df()` before `init`, `cal_head`, and optional
  `cal_wing`.
- The iteration loop calls `refresh_headwing` before each GW build and updates
  the live mean field and velocity at the end of the previous iteration.
  Therefore iteration `n` consumes the state produced by iteration `n-1`.

## Independent full grid

- `IndependentHeadwingState` owns immutable `reference` and
  `reference_velocity` members and distinct mutable `live` and
  `live_velocity` members.
- `Dataset::p_headwing` is constructed with references to
  `state->live` and `state->live_velocity` storage. The dielectric object
  keeps the velocity by reference and receives the live mean field on every
  `refresh_headwing` call.
- `update_independent_headwing_state` projects the SCF-grid fixed-basis
  Hamiltonian to the independent grid, diagonalizes against the immutable
  reference, and transactionally replaces both live mean field and live
  velocity.
- `IndependentHeadwingState::~IndependentHeadwingState` resets
  `owner->p_headwing` in the destructor body. C++ destroys data members only
  after the destructor body, so the dielectric object's velocity reference is
  released before `live_velocity` is destroyed.

## Direct full-BZ symmetry storage

The value-owned `direct_full_bz_velocity_` and `direct_full_bz_wfc_` members
belong to the crystal-symmetry restoration route. They are not selected by the
current full-BZ/no-symmetry same-grid or independent-grid QSGW path and cannot
be used as evidence for symmetry support.

## Required numerical observers

The static audit is accepted only together with observers that verify:

1. `C_n = U_n^T C_0` in row-stored `MeanField` convention.
2. `v_n = U_n^dagger v_0 U_n` for all Cartesian components.
3. Velocity Hermiticity and unitary residuals.
4. Exact iteration-zero/G0W0 head replay within tolerance.
5. A nonzero post-update live head response.
6. A controlled disabled versus head-only pair with all provenance fixed
   outside scalar `parameters.head_mode`.

