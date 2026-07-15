// These tests intentionally use assert; keep it active in Release test builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../qsgw/mixing.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using librpa_int::matrix;
using librpa_int::qsgw::HamiltonianMixer;
using librpa_int::qsgw::MixingMode;
using librpa_int::qsgw::MixingOptions;

namespace
{

void assert_close(const double actual, const double expected)
{
    assert(std::abs(actual - expected) < 1.0e-14);
}

template <typename Function>
void assert_throws(Function&& function)
{
    bool threw = false;
    try
    {
        function();
    }
    catch (const std::exception&)
    {
        threw = true;
    }
    assert(threw);
}

void test_default_linear_mixing_updates_grid_and_band_together()
{
    MixingOptions options;
    HamiltonianMixer mixer(options);

    matrix grid0(2, 1, true);
    grid0(0, 0) = 1.0;
    grid0(1, 0) = 2.0;

    matrix band0(1, 1, true);
    band0(0, 0) = 10.0;
    mixer.initialize(grid0, band0);

    matrix grid_output(2, 1, true);
    grid_output(0, 0) = 3.0;
    grid_output(1, 0) = 6.0;

    matrix band_output(1, 1, true);
    band_output(0, 0) = 20.0;

    const auto result = mixer.mix(grid_output, band_output);

    assert(result.decision.requested_mode == MixingMode::Linear);
    assert(result.decision.applied_mode == MixingMode::Linear);
    assert_close(result.decision.beta, 0.2);
    assert(!result.decision.fell_back);
    assert(result.band.has_value());

    assert_close(result.grid(0, 0), 1.4);
    assert_close(result.grid(1, 0), 2.8);
    assert_close(result.band->operator()(0, 0), 12.0);
}

void test_pulay_coefficients_are_solved_from_grid_and_reused_for_band()
{
    MixingOptions options;
    options.mode = MixingMode::Pulay;
    options.beta = 0.2;
    HamiltonianMixer mixer(options);

    matrix grid0(2, 1, true);
    matrix band0(2, 1, true);
    band0(0, 0) = 10.0;
    band0(1, 0) = 20.0;
    mixer.initialize(grid0, band0);

    matrix grid_output1(2, 1, true);
    grid_output1(0, 0) = 1.0;
    matrix band_output1 = band0;
    band_output1(0, 0) = 12.0;
    const auto first = mixer.mix(grid_output1, band_output1);
    assert(first.decision.applied_mode == MixingMode::Linear);

    matrix grid_output2 = first.grid;
    grid_output2(1, 0) = 1.0;
    matrix band_output2 = *first.band;
    band_output2(1, 0) = 24.0;
    const auto second = mixer.mix(grid_output2, band_output2);

    assert(second.decision.applied_mode == MixingMode::Pulay);
    assert(!second.decision.fell_back);
    assert(second.decision.coefficients.size() == 2);
    assert_close(second.decision.coefficients[0], 0.5);
    assert_close(second.decision.coefficients[1], 0.5);

    assert_close(second.grid(0, 0), 0.2);
    assert_close(second.grid(1, 0), 0.1);
    assert(second.band.has_value());
    assert_close(second.band->operator()(0, 0), 10.4);
    assert_close(second.band->operator()(1, 0), 20.4);
}

void test_pulay_conditioning_is_invariant_to_residual_scale()
{
    MixingOptions options;
    options.mode = MixingMode::Pulay;
    options.beta = 0.2;
    options.min_reciprocal_condition = 1.0e-12;
    HamiltonianMixer mixer(options);

    constexpr double scale = 1.0e-8;
    matrix grid0(2, 1, true);
    mixer.initialize(grid0);

    matrix output1(2, 1, true);
    output1(0, 0) = scale;
    const auto first = mixer.mix(output1);

    matrix output2 = first.grid;
    output2(1, 0) += scale;
    const auto second = mixer.mix(output2);

    assert(second.decision.applied_mode == MixingMode::Pulay);
    assert(!second.decision.fell_back);
    assert(second.decision.coefficients.size() == 2);
    assert_close(second.decision.coefficients[0], 0.5);
    assert_close(second.decision.coefficients[1], 0.5);
    assert_close(second.grid(0, 0), 0.2 * scale);
    assert_close(second.grid(1, 0), 0.1 * scale);
}

void test_singular_pulay_history_falls_back_to_finite_linear_step()
{
    MixingOptions options;
    options.mode = MixingMode::Pulay;
    options.beta = 0.2;
    options.min_reciprocal_condition = 1.0e-12;
    HamiltonianMixer mixer(options);

    matrix grid0(2, 1, true);
    mixer.initialize(grid0);

    matrix output1(2, 1, true);
    output1(0, 0) = 1.0;
    const auto first = mixer.mix(output1);

    matrix output2 = first.grid;
    output2(0, 0) += 1.0;
    const auto second = mixer.mix(output2);

    assert(second.decision.requested_mode == MixingMode::Pulay);
    assert(second.decision.applied_mode == MixingMode::Linear);
    assert(second.decision.fell_back);
    assert(!second.decision.fallback_reason.empty());
    assert(std::isfinite(second.decision.reciprocal_condition));
    assert(second.decision.reciprocal_condition < options.min_reciprocal_condition);
    assert_close(second.grid(0, 0), 0.4);
    assert_close(second.grid(1, 0), 0.0);
}

void test_large_pulay_coefficients_fall_back_to_linear_step()
{
    MixingOptions options;
    options.mode = MixingMode::Pulay;
    options.beta = 0.2;
    options.min_reciprocal_condition = 1.0e-20;
    options.max_coefficient_l1 = 10.0;
    HamiltonianMixer mixer(options);

    matrix grid0(2, 1, true);
    mixer.initialize(grid0);

    matrix output1(2, 1, true);
    output1(0, 0) = 1.0;
    const auto first = mixer.mix(output1);

    matrix output2 = first.grid;
    output2(0, 0) += 1.01;
    output2(1, 0) += 1.0e-4;
    const auto second = mixer.mix(output2);

    assert(second.decision.fell_back);
    assert(second.decision.applied_mode == MixingMode::Linear);
    assert(second.decision.reciprocal_condition >= options.min_reciprocal_condition);
    assert(second.decision.fallback_reason.find("coefficient") != std::string::npos);
    assert_close(second.grid(0, 0), 0.402);
    assert_close(second.grid(1, 0), 2.0e-5);
}

void test_residual_growth_falls_back_and_restarts_pulay_history()
{
    MixingOptions options;
    options.mode = MixingMode::Pulay;
    options.beta = 0.2;
    options.max_residual_growth = 2.0;
    HamiltonianMixer mixer(options);

    matrix grid0(2, 1, true);
    mixer.initialize(grid0);

    matrix output1(2, 1, true);
    output1(0, 0) = 1.0;
    const auto first = mixer.mix(output1);

    matrix output2 = first.grid;
    output2(0, 0) += 10.0;
    output2(1, 0) += 1.0;
    const auto second = mixer.mix(output2);
    assert(second.decision.fell_back);
    assert(second.decision.fallback_reason.find("growth") != std::string::npos);
    assert_close(second.grid(0, 0), 2.2);
    assert_close(second.grid(1, 0), 0.2);

    matrix output3 = second.grid;
    output3(1, 0) += 1.0;
    const auto third = mixer.mix(output3);
    assert(third.decision.applied_mode == MixingMode::Linear);
    assert(!third.decision.fell_back);
}

void test_reported_residual_is_unmixed_even_when_beta_is_one()
{
    MixingOptions options;
    options.beta = 1.0;
    HamiltonianMixer mixer(options);

    matrix grid0(2, 1, true);
    mixer.initialize(grid0);
    matrix output(2, 1, true);
    output(0, 0) = 3.0;
    output(1, 0) = 4.0;

    const auto result = mixer.mix(output);
    assert_close(result.residual_l2, 5.0);
    assert_close(result.residual_max, 4.0);
    assert_close(result.grid(0, 0), 3.0);
    assert_close(result.grid(1, 0), 4.0);
}

void test_rejected_band_shape_does_not_change_pulay_history()
{
    MixingOptions options;
    options.mode = MixingMode::Pulay;
    options.beta = 0.2;
    HamiltonianMixer tested(options);
    HamiltonianMixer fresh(options);

    matrix grid0(2, 1, true);
    matrix band0(1, 1, true);
    tested.initialize(grid0, band0);
    fresh.initialize(grid0, band0);

    matrix rejected_grid(2, 1, true);
    rejected_grid(0, 0) = 1.0;
    matrix rejected_band(2, 1, true);
    assert_throws([&] { tested.mix(rejected_grid, rejected_band); });

    matrix accepted_grid(2, 1, true);
    accepted_grid(1, 0) = 1.0;
    matrix accepted_band(1, 1, true);
    accepted_band(0, 0) = 2.0;
    const auto actual = tested.mix(accepted_grid, accepted_band);
    const auto expected = fresh.mix(accepted_grid, accepted_band);

    assert(actual.decision.applied_mode == expected.decision.applied_mode);
    assert(actual.decision.fell_back == expected.decision.fell_back);
    assert_close(actual.grid(0, 0), expected.grid(0, 0));
    assert_close(actual.grid(1, 0), expected.grid(1, 0));
    assert_close(actual.band->operator()(0, 0),
                 expected.band->operator()(0, 0));
}

void test_nonfinite_input_or_output_is_rejected_before_state_mutation()
{
    MixingOptions options;
    options.mode = MixingMode::Pulay;
    HamiltonianMixer tested(options);
    HamiltonianMixer fresh(options);

    matrix grid0(2, 1, true);
    matrix band0(1, 1, true);
    tested.initialize(grid0, band0);
    fresh.initialize(grid0, band0);

    matrix nonfinite_grid(2, 1, true);
    nonfinite_grid(0, 0) = std::numeric_limits<double>::quiet_NaN();
    assert_throws([&] { tested.mix(nonfinite_grid, band0); });

    matrix accepted_grid(2, 1, true);
    accepted_grid(0, 0) = 1.0;
    matrix accepted_band(1, 1, true);
    accepted_band(0, 0) = 2.0;
    const auto actual = tested.mix(accepted_grid, accepted_band);
    const auto expected = fresh.mix(accepted_grid, accepted_band);
    assert(actual.decision.applied_mode == expected.decision.applied_mode);
    assert_close(actual.grid(0, 0), expected.grid(0, 0));
    assert_close(actual.band->operator()(0, 0),
                 expected.band->operator()(0, 0));
}

void test_overflowing_residual_is_rejected_before_state_mutation()
{
    HamiltonianMixer tested;
    HamiltonianMixer fresh;

    matrix grid0(1, 1, true);
    grid0(0, 0) = std::numeric_limits<double>::max();
    tested.initialize(grid0);
    fresh.initialize(grid0);

    matrix overflowing_output(1, 1, true);
    overflowing_output(0, 0) = -std::numeric_limits<double>::max();
    assert_throws([&] { tested.mix(overflowing_output); });

    const auto actual = tested.mix(grid0);
    const auto expected = fresh.mix(grid0);
    assert_close(actual.grid(0, 0), expected.grid(0, 0));
    assert_close(actual.residual_l2, expected.residual_l2);
    assert_close(actual.residual_max, expected.residual_max);
}

} // namespace

int main()
{
    test_default_linear_mixing_updates_grid_and_band_together();
    test_pulay_coefficients_are_solved_from_grid_and_reused_for_band();
    test_pulay_conditioning_is_invariant_to_residual_scale();
    test_singular_pulay_history_falls_back_to_finite_linear_step();
    test_large_pulay_coefficients_fall_back_to_linear_step();
    test_residual_growth_falls_back_and_restarts_pulay_history();
    test_reported_residual_is_unmixed_even_when_beta_is_one();
    test_rejected_band_shape_does_not_change_pulay_history();
    test_nonfinite_input_or_output_is_rejected_before_state_mutation();
    test_overflowing_residual_is_rejected_before_state_mutation();
    std::cout << "test_qsgw_mixing: all tests passed\n";
    return 0;
}
