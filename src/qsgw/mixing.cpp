#include "mixing.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

extern "C"
{
void dgetrf_(const int* m, const int* n, double* a, const int* lda, int* ipiv, int* info);
void dgetrs_(const char* trans, const int* n, const int* nrhs, const double* a,
             const int* lda, const int* ipiv, double* b, const int* ldb, int* info);
void dgecon_(const char* norm, const int* n, const double* a, const int* lda,
             const double* anorm, double* rcond, double* work, int* iwork, int* info);
}

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

double inner_product(const matrix& lhs, const matrix& rhs)
{
    require_same_shape(lhs, rhs, "residual");
    double result = 0.0;
    for (int i = 0; i < lhs.size; ++i)
    {
        result += lhs.c[i] * rhs.c[i];
    }
    return result;
}

double l2_norm(const matrix& value)
{
    return std::sqrt(inner_product(value, value));
}

double max_abs(const matrix& value)
{
    double result = 0.0;
    for (int i = 0; i < value.size; ++i)
    {
        result = std::max(result, std::abs(value.c[i]));
    }
    return result;
}

double matrix_one_norm(const matrix& value)
{
    double result = 0.0;
    for (int column = 0; column < value.nc; ++column)
    {
        double column_sum = 0.0;
        for (int row = 0; row < value.nr; ++row)
        {
            column_sum += std::abs(value(row, column));
        }
        result = std::max(result, column_sum);
    }
    return result;
}

bool solve_pulay_coefficients(
    const std::vector<matrix>& residuals,
    const double min_reciprocal_condition,
    std::vector<double>& coefficients,
    double& reciprocal_condition,
    std::string& failure_reason)
{
    const int history_size = static_cast<int>(residuals.size());
    const int system_size = history_size + 1;
    matrix system(system_size, system_size, true);
    matrix rhs(system_size, 1, true);

    for (int i = 0; i < history_size; ++i)
    {
        for (int j = i; j < history_size; ++j)
        {
            const double value = inner_product(residuals[i], residuals[j]);
            system(i, j) = value;
            system(j, i) = value;
        }
        system(i, history_size) = -1.0;
        system(history_size, i) = -1.0;
    }
    rhs(history_size, 0) = -1.0;

    const double norm = matrix_one_norm(system);
    matrix lu = system;
    std::vector<int> pivots(system_size, 0);
    int info = 0;
    dgetrf_(&system_size, &system_size, lu.c, &system_size, pivots.data(), &info);
    if (info != 0)
    {
        reciprocal_condition = 0.0;
        failure_reason = "Pulay residual system is singular";
        return false;
    }

    const char norm_type = '1';
    std::vector<double> work(4 * system_size, 0.0);
    std::vector<int> iwork(system_size, 0);
    dgecon_(&norm_type, &system_size, lu.c, &system_size, &norm,
            &reciprocal_condition, work.data(), iwork.data(), &info);
    if (info != 0 || !std::isfinite(reciprocal_condition))
    {
        reciprocal_condition = 0.0;
        failure_reason = "Pulay reciprocal condition estimate failed";
        return false;
    }
    if (reciprocal_condition < min_reciprocal_condition)
    {
        failure_reason = "Pulay residual system is ill-conditioned";
        return false;
    }

    const char transpose = 'N';
    const int nrhs = 1;
    dgetrs_(&transpose, &system_size, &nrhs, lu.c, &system_size, pivots.data(),
            rhs.c, &system_size, &info);
    if (info != 0)
    {
        failure_reason = "Pulay residual system solve failed";
        return false;
    }

    coefficients.resize(history_size);
    for (int i = 0; i < history_size; ++i)
    {
        coefficients[i] = rhs(i, 0);
        if (!std::isfinite(coefficients[i]))
        {
            failure_reason = "Pulay coefficients are not finite";
            return false;
        }
    }
    return true;
}

matrix apply_pulay(
    const std::vector<matrix>& inputs,
    const std::vector<matrix>& residuals,
    const std::vector<double>& coefficients,
    const double beta)
{
    matrix mixed(inputs.front().nr, inputs.front().nc, true);
    for (std::size_t i = 0; i < coefficients.size(); ++i)
    {
        matrix term = inputs[i] + beta * residuals[i];
        term *= coefficients[i];
        mixed += term;
    }
    return mixed;
}

} // namespace

HamiltonianMixer::HamiltonianMixer(MixingOptions options)
    : options_(options)
{
    if (!(options_.beta > 0.0 && options_.beta <= 1.0))
    {
        throw std::invalid_argument("QSGW mixing beta must be in (0, 1]");
    }
    if (options_.max_history < 2)
    {
        throw std::invalid_argument("QSGW Pulay history must contain at least two samples");
    }
    if (!(options_.min_reciprocal_condition > 0.0 &&
          options_.min_reciprocal_condition < 1.0))
    {
        throw std::invalid_argument("QSGW Pulay reciprocal-condition threshold must be in (0, 1)");
    }
    if (!(options_.max_coefficient_l1 >= 1.0))
    {
        throw std::invalid_argument("QSGW Pulay coefficient L1 limit must be at least one");
    }
    if (!(options_.max_residual_growth > 1.0))
    {
        throw std::invalid_argument("QSGW Pulay residual-growth limit must exceed one");
    }
}

void HamiltonianMixer::initialize(const matrix& grid_input)
{
    grid_input_history_ = {grid_input};
    grid_residual_history_.clear();
    band_input_history_.clear();
    band_residual_history_.clear();
    initialized_ = true;
}

void HamiltonianMixer::initialize(const matrix& grid_input, const matrix& band_input)
{
    grid_input_history_ = {grid_input};
    grid_residual_history_.clear();
    band_input_history_ = {band_input};
    band_residual_history_.clear();
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
    const matrix current_grid_input = grid_input_history_.back();
    require_same_shape(current_grid_input, grid_output, "grid");
    const bool has_band = !band_input_history_.empty();
    if (has_band != band_output.has_value())
    {
        throw std::invalid_argument("QSGW band mixing input/output presence does not match");
    }

    matrix grid_residual = grid_output - current_grid_input;
    const double residual_l2 = l2_norm(grid_residual);
    const double residual_max = max_abs(grid_residual);
    const double previous_residual_l2 = grid_residual_history_.empty()
                                            ? residual_l2
                                            : l2_norm(grid_residual_history_.back());
    const bool residual_grew_too_fast =
        !grid_residual_history_.empty() &&
        residual_l2 > options_.max_residual_growth * previous_residual_l2;
    grid_residual_history_.push_back(grid_residual);

    if (has_band)
    {
        require_same_shape(band_input_history_.back(), *band_output, "band");
        band_residual_history_.push_back(*band_output - band_input_history_.back());
    }

    if (static_cast<int>(grid_residual_history_.size()) > options_.max_history)
    {
        grid_residual_history_.erase(grid_residual_history_.begin());
        grid_input_history_.erase(grid_input_history_.begin());
        if (has_band)
        {
            band_residual_history_.erase(band_residual_history_.begin());
            band_input_history_.erase(band_input_history_.begin());
        }
    }

    MixingDecision decision;
    decision.requested_mode = options_.mode;
    decision.beta = options_.beta;

    const bool pulay_ready = options_.mode == MixingMode::Pulay &&
                             grid_residual_history_.size() >= 2;
    if (pulay_ready)
    {
        std::string failure_reason;
        bool solved = false;
        if (residual_grew_too_fast)
        {
            failure_reason = "Pulay residual growth exceeds configured limit";
        }
        else
        {
            solved = solve_pulay_coefficients(
                grid_residual_history_, options_.min_reciprocal_condition,
                decision.coefficients, decision.reciprocal_condition, failure_reason);
        }
        if (solved)
        {
            double coefficient_l1 = 0.0;
            for (const double coefficient: decision.coefficients)
            {
                coefficient_l1 += std::abs(coefficient);
            }
            if (coefficient_l1 > options_.max_coefficient_l1)
            {
                solved = false;
                failure_reason = "Pulay coefficient L1 norm exceeds configured limit";
            }
        }
        if (solved)
        {
            decision.applied_mode = MixingMode::Pulay;
            matrix mixed_grid = apply_pulay(
                grid_input_history_, grid_residual_history_, decision.coefficients,
                options_.beta);
            std::optional<matrix> mixed_band;
            if (has_band)
            {
                mixed_band = apply_pulay(
                    band_input_history_, band_residual_history_, decision.coefficients,
                    options_.beta);
                band_input_history_.push_back(*mixed_band);
            }
            grid_input_history_.push_back(mixed_grid);
            return {mixed_grid, mixed_band, decision, residual_l2, residual_max};
        }
        decision.fell_back = true;
        decision.fallback_reason = failure_reason;
    }

    decision.applied_mode = MixingMode::Linear;
    decision.coefficients = {1.0};
    matrix mixed_grid = linear_mix(current_grid_input, grid_output, options_.beta);
    std::optional<matrix> mixed_band;
    if (has_band)
    {
        mixed_band = linear_mix(band_input_history_.back(), *band_output, options_.beta);
    }

    if (decision.fell_back)
    {
        grid_input_history_ = {mixed_grid};
        grid_residual_history_.clear();
        if (has_band)
        {
            band_input_history_ = {*mixed_band};
            band_residual_history_.clear();
        }
    }
    else
    {
        grid_input_history_.push_back(mixed_grid);
        if (has_band)
        {
            band_input_history_.push_back(*mixed_band);
        }
    }
    return {mixed_grid, mixed_band, decision, residual_l2, residual_max};
}

} // namespace qsgw
} // namespace librpa_int
