#include "hartree_route.h"

#include <cmath>
#include <stdexcept>

namespace librpa_int
{
namespace qsgw
{
namespace
{

bool same_periodic_kpoint(const Vector3_Order<double>& lhs,
                          const Vector3_Order<double>& rhs,
                          const double tolerance)
{
    return std::abs(lhs.x - rhs.x - std::round(lhs.x - rhs.x)) <= tolerance &&
           std::abs(lhs.y - rhs.y - std::round(lhs.y - rhs.y)) <= tolerance &&
           std::abs(lhs.z - rhs.z - std::round(lhs.z - rhs.z)) <= tolerance;
}

} // namespace

HartreeReaderRoute select_hartree_reader_route(
    const bool use_shrink_abfs,
    const std::string& full_ri_prefix,
    const std::string& shrink_ri_prefix)
{
    if (full_ri_prefix.empty() || shrink_ri_prefix.empty() ||
        full_ri_prefix == shrink_ri_prefix)
    {
        throw std::invalid_argument(
            "QSGW Hartree requires distinct nonempty full and shrink RI prefixes");
    }
    return {use_shrink_abfs ? shrink_ri_prefix : full_ri_prefix,
            use_shrink_abfs};
}

bool hartree_density_requires_symmetry_restore(
    const std::vector<Vector3_Order<double>>& scf_kpoints,
    const std::vector<Vector3_Order<double>>& full_kpoints,
    const bool symmetry_enabled)
{
    constexpr double tolerance = 1.0e-10;
    if (scf_kpoints.empty() || full_kpoints.empty() ||
        scf_kpoints.size() > full_kpoints.size())
    {
        throw std::invalid_argument(
            "QSGW Hartree SCF/full k-grid sizes are inconsistent");
    }

    std::vector<bool> matched(full_kpoints.size(), false);
    for (const auto& scf_kpoint : scf_kpoints)
    {
        bool found = false;
        for (std::size_t index = 0; index < full_kpoints.size(); ++index)
        {
            if (!matched[index] &&
                same_periodic_kpoint(scf_kpoint, full_kpoints[index],
                                     tolerance))
            {
                matched[index] = true;
                found = true;
                break;
            }
        }
        if (!found)
        {
            throw std::invalid_argument(
                "QSGW Hartree SCF k grid is not a subset of the full BvK grid");
        }
    }

    if (scf_kpoints.size() == full_kpoints.size())
    {
        return false;
    }
    if (!symmetry_enabled)
    {
        throw std::invalid_argument(
            "QSGW Hartree received a reduced SCF k grid without use_symmetry_gw");
    }
    return true;
}

} // namespace qsgw
} // namespace librpa_int
