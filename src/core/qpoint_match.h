#pragma once

#include <algorithm>
#include <complex>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "pbc.h"
#include "symmetry_context.h"
#include "../mpi/base_mpi.h"

namespace librpa_int
{

// Internal q vectors are Cartesian, in 2pi/Bohr. Only their FRACTIONAL
// difference may be compared with reciprocal-lattice integers. This geometric
// predicate is intentionally separate from ordered-map key comparison.
inline bool match_internal_qpoints(const PeriodicBoundaryData& pbc,
                                  const Vector3_Order<double>& source,
                                  const Vector3_Order<double>& target,
                                  double tolerance, Vector3_Order<int>* target_minus_source = nullptr)
{
    const auto difference = pbc.latvec * (target - source);
    if (!nearly_integer_vector(difference, tolerance)) return false;
    if (target_minus_source)
        *target_minus_source = {static_cast<int>(std::lround(difference.x)),
                                static_cast<int>(std::lround(difference.y)),
                                static_cast<int>(std::lround(difference.z))};
    return true;
}

template <class QMap>
auto find_internal_qpoint(QMap& values, const PeriodicBoundaryData& pbc,
                         const Vector3_Order<double>& target, double tolerance = 1e-5)
    -> decltype(values.begin())
{
    const auto direct = values.find(target);
    if (direct != values.end() && match_internal_qpoints(pbc, direct->first, target, tolerance))
        return direct;
    return std::find_if(values.begin(), values.end(), [&](const auto& item) {
        return match_internal_qpoints(pbc, item.first, target, tolerance);
    });
}

// d_I maps a SOURCE operator frame to the equivalent TARGET frame. Reuse the
// established Bloch convention: exp[-2pi i * bloch_phase * bloch_ratom * G.tau_I].
// Exact/same-label rounded points require no metadata. Coordinates are needed
// only when the declared atom-position coefficient is nonzero.
inline std::vector<std::complex<double>> internal_q_gauge_phases(
    const SymmetryContext& context, const PeriodicBoundaryData& pbc,
    const Vector3_Order<double>& source, const Vector3_Order<double>& target,
    std::size_t atom_count, double tolerance = 1e-5)
{
    Vector3_Order<int> shift;
    if (!match_internal_qpoints(pbc, source, target, tolerance, &shift))
        throw std::runtime_error("q-frame transport requires reciprocal-equivalent points");
    if (shift == Vector3_Order<int>{0, 0, 0}) return std::vector<std::complex<double>>(atom_count, {1., 0.});
    if (!is_basis_bloch_convention_set(context.basis_convention))
        throw std::runtime_error("reciprocal q aliases require a declared basis Bloch convention");
    if (context.basis_convention.bloch_ratom == 0) return std::vector<std::complex<double>>(atom_count, {1., 0.});
    SymmetryKStarMember identity;
    identity.k_bz = {0., 0., 0.};
    const Vector3_Order<double> target_fractional{double(shift.x), double(shift.y), double(shift.z)};
    return build_symmetry_kstar_member_target_gauge_phases(
        context, identity, atom_count, &target_fractional);
}

inline void check_internal_q_error_collective(MPI_Comm comm, const std::string& error)
{
    int local = !error.empty(), any = 0;
    MPI_Allreduce(&local, &any, 1, MPI_INT, MPI_MAX, comm);
    if (any)
        throw std::runtime_error(error.empty() ? "q-frame validation failed on another rank" : error);
}

} // namespace librpa_int
