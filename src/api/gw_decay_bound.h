#pragma once

#include "../core/gw_decay_bound.h"
#include "dataset.h"
#include "librpa_options.h"

namespace librpa_int
{
// Internal diagnostic; preserves resident compute objects and never selects a grid.
GwBodyDecayBound compute_ds_gw_body_decay_bound(Dataset &ds, const LibrpaOptions &opts);
// Exact finite3D optical augmentation; before-grid and resident contexts alike.
// Analytic optics use all loaded bands, so a restricted response is ineligible.
GwOpticalDecayBound compute_ds_gw_optical_decay_bound(Dataset &ds, const LibrpaOptions &opts);
}  // namespace librpa_int
