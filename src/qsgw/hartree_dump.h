#pragma once

#include "hartree_density.h"
#include "hartree_kernel.h"
#include "hartree_workflow.h"

namespace librpa_int
{
namespace qsgw
{

// Env-gated diagnostic dump of the QSGW Hartree delta pipeline.
// When LIBRPA_QSGW_HARTREE_DUMP_DIR is set, each call appends a new
// call_NNN subdirectory holding the weighted density delta, the
// contracted Hartree operator in k space, the periodic operator in R
// space, and a manifest with the pipeline conventions. This observer
// hook changes no numerical behavior and is a no-op when the variable
// is unset or empty. Dumped data is intended for independent
// recomputation by external observers.
void maybe_dump_hartree_pipeline(
    const HartreeStaticData& static_data,
    const WeightedDensityKMap& density_delta_k,
    const HartreeDkMap& hartree_k,
    const PeriodicOperatorRMap& hartree_r);

} // namespace qsgw
} // namespace librpa_int
