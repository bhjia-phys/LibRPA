#include "mixing.h"

#include <stdexcept>

namespace librpa_int
{
namespace qsgw
{

namespace
{

void require_same_shape(const matrix& lhs, const matrix& rhs, const char* channel)
{
    if (lhs.nr != rhs.nr || lhs.nc != rhs.nc)
    {
        throw std::invalid_argument(std::string("QSGW ") + channel +
                                    " mixing matrix dimensions do not match");
    }
}

matrix linear_mix(const matrix& input, const matrix& output, const double beta)
{
    matrix residual = output - input;
    residual *= beta;
    return input + residual;
}

} // namespace

HamiltonianMixer::HamiltonianMixer(MixingOptions options)
    : options_(options)
{
    if (!(options_.beta > 0.0 && options_.beta <= 1.0))
    {
        throw std::invalid_argument("QSGW mixing beta must be in (0, 1]");
    }
}

void HamiltonianMixer::initialize(const matrix& grid_input)
{
    grid_input_ = grid_input;
    band_input_.reset();
    initialized_ = true;
}

void HamiltonianMixer::initialize(const matrix& grid_input, const matrix& band_input)
{
    grid_input_ = grid_input;
    band_input_ = band_input;
    initialized_ = true;
}

HamiltonianMixResult HamiltonianMixer::mix(const matrix& grid_output)
{
    return mix_impl(grid_output, std::nullopt);
}

HamiltonianMixResult HamiltonianMixer::mix(
    const matrix& grid_output,
    const matrix& band_output)
{
    return mix_impl(grid_output, band_output);
}

HamiltonianMixResult HamiltonianMixer::mix_impl(
    const matrix& grid_output,
    const std::optional<matrix>& band_output)
{
    if (!initialized_)
    {
        throw std::logic_error("QSGW Hamiltonian mixer must be initialized before use");
    }
    if (options_.mode != MixingMode::Linear)
    {
        throw std::logic_error("QSGW Pulay mixing is not implemented");
    }

    require_same_shape(grid_input_, grid_output, "grid");
    if (band_input_.has_value() != band_output.has_value())
    {
        throw std::invalid_argument("QSGW band mixing input/output presence does not match");
    }

    matrix mixed_grid = linear_mix(grid_input_, grid_output, options_.beta);
    std::optional<matrix> mixed_band;
    if (band_input_)
    {
        require_same_shape(*band_input_, *band_output, "band");
        mixed_band = linear_mix(*band_input_, *band_output, options_.beta);
    }

    MixingDecision decision;
    decision.requested_mode = options_.mode;
    decision.applied_mode = MixingMode::Linear;
    decision.beta = options_.beta;

    grid_input_ = mixed_grid;
    band_input_ = mixed_band;
    return {mixed_grid, mixed_band, decision};
}

} // namespace qsgw
} // namespace librpa_int
