#include "band_bvk_remap.h"

#include "../api/compute_helper.h"

namespace librpa_int
{
namespace qsgw
{

AtomPairBvKRemap<atom_t> build_legacy_band_bvk_remap(const Atoms& atoms,
                                                     const PeriodicBoundaryData& pbc,
                                                     const int remap_convention)
{
    // Keep the QSGW entry point, but do not undo the shared opposite-pair
    // consistency by deleting one displaced-atom corner mapping.
    return api::build_band_bvk_remap(atoms, pbc, remap_convention);
}

}  // namespace qsgw
}  // namespace librpa_int
