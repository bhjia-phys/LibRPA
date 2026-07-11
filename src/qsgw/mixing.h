#pragma once

#include "../math/matrix.h"

#include <optional>
#include <string>

namespace librpa_int
{
namespace qsgw
{

enum class MixingMode
{
    Linear,
    Pulay,
};

struct MixingOptions
{
    MixingMode mode = MixingMode::Linear;
    double beta = 0.2;
};

struct MixingDecision
{
    MixingMode requested_mode = MixingMode::Linear;
    MixingMode applied_mode = MixingMode::Linear;
    double beta = 0.2;
    bool fell_back = false;
    std::string fallback_reason;
};

struct HamiltonianMixResult
{
    matrix grid;
    std::optional<matrix> band;
    MixingDecision decision;
};

class HamiltonianMixer
{
public:
    explicit HamiltonianMixer(MixingOptions options = {});

    void initialize(const matrix& grid_input);
    void initialize(const matrix& grid_input, const matrix& band_input);

    HamiltonianMixResult mix(const matrix& grid_output);
    HamiltonianMixResult mix(const matrix& grid_output, const matrix& band_output);

private:
    MixingOptions options_;
    bool initialized_ = false;
    matrix grid_input_;
    std::optional<matrix> band_input_;

    HamiltonianMixResult mix_impl(
        const matrix& grid_output,
        const std::optional<matrix>& band_output);
};

} // namespace qsgw
} // namespace librpa_int
