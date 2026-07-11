#include "occupation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace librpa_int
{
namespace qsgw
{

namespace
{

struct State
{
    double energy;
    double capacity;
    int spin;
    int kpoint;
    int band;
};

void validate_same_layout(const MeanField& live, const MeanField& reference)
{
    if (!live.initialized() || !reference.initialized())
    {
        throw std::invalid_argument("QSGW occupation mean fields must be initialized");
    }
    if (live.get_n_spins() != reference.get_n_spins() ||
        live.get_n_kpoints() != reference.get_n_kpoints() ||
        live.get_n_bands() != reference.get_n_bands() ||
        live.get_n_spinor() != reference.get_n_spinor())
    {
        throw std::invalid_argument("QSGW live and reference mean-field layouts differ");
    }
}

double state_capacity_factor(const MeanField& meanfield)
{
    return 2.0 /
           static_cast<double>(meanfield.get_n_spins() * meanfield.get_n_spinor());
}

} // namespace

OccupationResult update_qsgw_occupations(
    MeanField& live_meanfield,
    const MeanField& reference_meanfield,
    const std::vector<double>& kpoint_weights,
    const double total_electrons,
    const OccupationSettings& settings)
{
    validate_same_layout(live_meanfield, reference_meanfield);
    if (settings.temperature_kelvin != 0.0)
    {
        throw std::invalid_argument("Finite-temperature QSGW occupations are not implemented");
    }
    if (!(settings.degeneracy_tolerance_ha > 0.0) ||
        !(settings.electron_tolerance > 0.0))
    {
        throw std::invalid_argument("QSGW occupation tolerances must be positive");
    }
    if (kpoint_weights.size() !=
        static_cast<std::size_t>(live_meanfield.get_n_kpoints()))
    {
        throw std::invalid_argument("QSGW k-point weight count does not match mean field");
    }

    double kpoint_weight_sum = 0.0;
    for (const double weight: kpoint_weights)
    {
        if (!std::isfinite(weight) || weight < 0.0)
        {
            throw std::invalid_argument("QSGW k-point weights must be finite and nonnegative");
        }
        kpoint_weight_sum += weight;
    }
    if (std::abs(kpoint_weight_sum - 1.0) > settings.electron_tolerance)
    {
        throw std::invalid_argument("QSGW k-point weights must sum to one");
    }

    const double capacity_factor = state_capacity_factor(reference_meanfield);
    double reference_electrons = 0.0;
    double total_capacity = 0.0;
    std::vector<State> states;
    states.reserve(static_cast<std::size_t>(live_meanfield.get_n_spins()) *
                   live_meanfield.get_n_kpoints() * live_meanfield.get_n_bands());

    for (int spin = 0; spin < live_meanfield.get_n_spins(); ++spin)
    {
        live_meanfield.get_weight()[spin].zero_out();
        for (int kpoint = 0; kpoint < live_meanfield.get_n_kpoints(); ++kpoint)
        {
            const double capacity = capacity_factor * kpoint_weights[kpoint];
            for (int band = 0; band < live_meanfield.get_n_bands(); ++band)
            {
                const double reference_weight =
                    reference_meanfield.get_weight()[spin](kpoint, band);
                if (!std::isfinite(reference_weight) ||
                    reference_weight < -settings.electron_tolerance ||
                    reference_weight > capacity + settings.electron_tolerance)
                {
                    throw std::invalid_argument(
                        "QSGW reference occupation exceeds its symmetry-weighted capacity");
                }
                reference_electrons += reference_weight;
                total_capacity += capacity;
                states.push_back({live_meanfield.get_eigenvals()[spin](kpoint, band),
                                  capacity, spin, kpoint, band});
            }
        }
    }

    if (!std::isfinite(total_electrons) || total_electrons < 0.0 ||
        total_electrons > total_capacity + settings.electron_tolerance)
    {
        throw std::invalid_argument("QSGW target electron count is outside state capacity");
    }
    if (std::abs(reference_electrons - total_electrons) > settings.electron_tolerance)
    {
        throw std::invalid_argument(
            "QSGW target electron count differs from immutable reference occupations");
    }

    std::sort(states.begin(), states.end(), [](const State& lhs, const State& rhs) {
        if (lhs.energy != rhs.energy)
        {
            return lhs.energy < rhs.energy;
        }
        if (lhs.spin != rhs.spin)
        {
            return lhs.spin < rhs.spin;
        }
        if (lhs.kpoint != rhs.kpoint)
        {
            return lhs.kpoint < rhs.kpoint;
        }
        return lhs.band < rhs.band;
    });

    double remaining = total_electrons;
    for (std::size_t begin = 0; begin < states.size();)
    {
        std::size_t end = begin + 1;
        while (end < states.size() &&
               std::abs(states[end].energy - states[begin].energy) <=
                   settings.degeneracy_tolerance_ha)
        {
            ++end;
        }

        double group_capacity = 0.0;
        for (std::size_t index = begin; index < end; ++index)
        {
            group_capacity += states[index].capacity;
        }
        const double fraction = group_capacity > 0.0
                                    ? std::clamp(remaining / group_capacity, 0.0, 1.0)
                                    : 0.0;
        for (std::size_t index = begin; index < end; ++index)
        {
            const State& state = states[index];
            live_meanfield.get_weight()[state.spin](state.kpoint, state.band) =
                fraction * state.capacity;
        }
        remaining -= fraction * group_capacity;
        if (remaining < settings.electron_tolerance)
        {
            remaining = 0.0;
        }
        begin = end;
    }
    if (remaining > settings.electron_tolerance)
    {
        throw std::runtime_error("QSGW global occupation filling did not conserve charge");
    }

    OccupationResult result;
    result.vbm = -std::numeric_limits<double>::infinity();
    result.cbm = std::numeric_limits<double>::infinity();
    for (const State& state: states)
    {
        const double weight =
            live_meanfield.get_weight()[state.spin](state.kpoint, state.band);
        if (weight > settings.electron_tolerance)
        {
            result.vbm = std::max(result.vbm, state.energy);
        }
        if (weight < state.capacity - settings.electron_tolerance)
        {
            result.cbm = std::min(result.cbm, state.energy);
        }
        if (weight > settings.electron_tolerance &&
            weight < state.capacity - settings.electron_tolerance)
        {
            result.metallic = true;
        }
        result.electron_count += weight;
    }

    if (!std::isfinite(result.vbm))
    {
        result.vbm = states.front().energy;
    }
    if (!std::isfinite(result.cbm))
    {
        result.cbm = states.back().energy;
    }
    result.gap = result.metallic ? 0.0 : std::max(0.0, result.cbm - result.vbm);
    result.chemical_potential = result.metallic
                                    ? result.vbm
                                    : 0.5 * (result.vbm + result.cbm);

    if (std::abs(result.electron_count - total_electrons) >
        settings.electron_tolerance)
    {
        throw std::runtime_error("QSGW updated occupations do not conserve charge");
    }
    live_meanfield.get_efermi() = result.chemical_potential;
    return result;
}

} // namespace qsgw
} // namespace librpa_int
