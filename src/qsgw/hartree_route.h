#pragma once

#include "../math/vector3_order.h"

#include <string>
#include <vector>

namespace librpa_int
{
namespace qsgw
{

struct HartreeReaderRoute
{
    std::string ri_prefix;
    bool use_shrink_basis = false;
};

HartreeReaderRoute select_hartree_reader_route(
    bool use_shrink_abfs,
    const std::string& full_ri_prefix,
    const std::string& shrink_ri_prefix);

bool hartree_density_requires_symmetry_restore(
    const std::vector<Vector3_Order<double>>& scf_kpoints,
    const std::vector<Vector3_Order<double>>& full_kpoints,
    bool symmetry_enabled);

} // namespace qsgw
} // namespace librpa_int
