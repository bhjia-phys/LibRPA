/*!
 * @author    Min-Ye Zhang
 * @date      2024-04-25
 */
#include "analycont.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <limits>
// #include <iostream>

#include "complexmatrix.h"
#include "params.h"
// #include "stl_io_helper.h"

namespace LIBRPA
{

namespace
{

std::string lower_string(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
            [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return s;
}

bool isfinite(const cplxdb &z)
{
    return std::isfinite(z.real()) && std::isfinite(z.imag());
}

cplxdb eval_poly(const std::vector<cplxdb> &coeff, const cplxdb &x)
{
    cplxdb value = {0.0, 0.0};
    for (auto it = coeff.rbegin(); it != coeff.rend(); ++it)
    {
        value = value * x + *it;
    }
    return value;
}

bool solve_linear_system(std::vector<std::vector<cplxdb>> a,
                         std::vector<cplxdb> b,
                         std::vector<cplxdb> &x)
{
    const int n = static_cast<int>(b.size());
    x.assign(n, {0.0, 0.0});

    for (int k = 0; k < n; k++)
    {
        int pivot = k;
        double pivot_abs = std::abs(a[k][k]);
        for (int i = k + 1; i < n; i++)
        {
            const double cand = std::abs(a[i][k]);
            if (cand > pivot_abs)
            {
                pivot_abs = cand;
                pivot = i;
            }
        }

        if (pivot_abs < 1.0e-28)
        {
            return false;
        }

        if (pivot != k)
        {
            std::swap(a[pivot], a[k]);
            std::swap(b[pivot], b[k]);
        }

        for (int i = k + 1; i < n; i++)
        {
            const cplxdb factor = a[i][k] / a[k][k];
            a[i][k] = {0.0, 0.0};
            for (int j = k + 1; j < n; j++)
            {
                a[i][j] -= factor * a[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    for (int i = n - 1; i >= 0; i--)
    {
        cplxdb sum = b[i];
        for (int j = i + 1; j < n; j++)
        {
            sum -= a[i][j] * x[j];
        }
        if (std::abs(a[i][i]) < 1.0e-28)
        {
            return false;
        }
        x[i] = sum / a[i][i];
    }

    return true;
}

} /* end anonymous namespace */

AnalyContPade::AnalyContPade(int n_pars_in, const std::vector<cplxdb> &xs, const std::vector<cplxdb> &data)
    : method(Method::Thiele),
      n_pars(n_pars_in),
      ridge_valid(false),
      ridge_num_degree(0),
      ridge_den_degree(0),
      ridge_x_scale(1.0),
      ridge_y_scale(1.0)
{
    build_thiele(n_pars_in, xs, data);

    const auto method_name = lower_string(Params::anacon_method);
    if (method_name == "ridge" || method_name == "ridge_guard" || method_name == "ridge-guard")
    {
        build_ridge(n_pars_in, xs, data);
        if (ridge_valid)
        {
            method = (method_name == "ridge_guard" || method_name == "ridge-guard") ?
                Method::RidgeGuard: Method::Ridge;
        }
    }
}

void
AnalyContPade::build_thiele(int n_pars_in, const std::vector<cplxdb> &xs, const std::vector<cplxdb> &data)
{
    n_pars = n_pars_in;
    int n_data = data.size();
    std::vector<cplxdb> data_npar;

    assert (n_pars > 0);
    assert (n_data > 0);
    assert (xs.size() == data.size());

    if (n_data <= n_pars)
    {
        // Use all data points
        n_pars = n_data;
        par_x = xs;
        data_npar = data;
    }
    else
    {
        // Select the data points evenly, when number of parameters are fewer than data points
        par_x.resize(n_pars);
        data_npar.resize(n_pars);
        if (n_pars == 1)
        {
            par_x[0] = xs[0];
            data_npar[0] = data[0];
        }
        else
        {
            int step = n_data / (n_pars - 1);
            for (int ipar = 0; ipar < n_pars - 1; ipar++)
            {
                par_x[ipar] = xs[ipar * step];
                data_npar[ipar] = data[ipar * step];
            }
            par_x[n_pars-1] = xs[n_data-1];
            data_npar[n_pars-1] = data[n_data-1];
        }
    }

    // Calculate the continuation coefficients, using Thiel's reciprocal difference method
    ComplexMatrix g(n_pars, n_pars);
    for (int i_par = 0; i_par < n_pars; i_par++)
    {
        g(i_par, 0) = data_npar[i_par];
    }

    for (int i_par = 1; i_par < n_pars; i_par++)
    {
        for (int i = i_par; i < n_pars; i++)
        {
            g(i, i_par) = 
                (g(i_par-1, i_par-1) - g(i, i_par-1)) / ((par_x[i] - par_x[i_par-1]) * g(i, i_par-1));
        }
    }

    par_y.resize(n_pars);
    for (int i_par = 0; i_par < n_pars; i_par++)
    {
        par_y[i_par] = g(i_par, i_par);
    }
}

void
AnalyContPade::build_ridge(int n_pars_in, const std::vector<cplxdb> &xs, const std::vector<cplxdb> &data)
{
    const int n_data = static_cast<int>(data.size());
    assert (n_pars_in > 0);
    assert (n_data > 0);
    assert (xs.size() == data.size());

    const int n_unknowns = std::min(std::max(n_pars_in, 1), n_data);
    const int n_num = (n_unknowns + 1) / 2;
    const int n_den = n_unknowns - n_num;
    ridge_num_degree = n_num - 1;
    ridge_den_degree = n_den;

    ridge_x_scale = 0.0;
    ridge_y_scale = 0.0;
    for (int i = 0; i < n_data; i++)
    {
        ridge_x_scale = std::max(ridge_x_scale, std::abs(xs[i]));
        ridge_y_scale = std::max(ridge_y_scale, std::abs(data[i]));
    }
    if (ridge_x_scale < 1.0e-30)
    {
        ridge_x_scale = 1.0;
    }
    if (ridge_y_scale < 1.0e-30)
    {
        ridge_y_scale = 1.0;
    }

    std::vector<std::vector<cplxdb>> normal(
            n_unknowns, std::vector<cplxdb>(n_unknowns, {0.0, 0.0}));
    std::vector<cplxdb> rhs(n_unknowns, {0.0, 0.0});

    for (int i = 0; i < n_data; i++)
    {
        const cplxdb x_scaled = xs[i] / ridge_x_scale;
        const cplxdb y_scaled = data[i] / ridge_y_scale;
        std::vector<cplxdb> row(n_unknowns, {0.0, 0.0});

        cplxdb x_power = {1.0, 0.0};
        for (int j = 0; j < n_num; j++)
        {
            row[j] = x_power;
            x_power *= x_scaled;
        }

        x_power = x_scaled;
        for (int j = 0; j < n_den; j++)
        {
            row[n_num + j] = -y_scaled * x_power;
            x_power *= x_scaled;
        }

        for (int j = 0; j < n_unknowns; j++)
        {
            rhs[j] += std::conj(row[j]) * y_scaled;
            for (int k = 0; k < n_unknowns; k++)
            {
                normal[j][k] += std::conj(row[j]) * row[k];
            }
        }
    }

    const double lambda = std::max(0.0, Params::pade_ridge_lambda);
    const double den_weight = std::max(0.0, Params::pade_ridge_den_weight);
    for (int j = 1; j < n_num; j++)
    {
        normal[j][j] += lambda * static_cast<double>(j * j);
    }
    for (int j = 0; j < n_den; j++)
    {
        const int power = j + 1;
        normal[n_num + j][n_num + j] += lambda * den_weight * static_cast<double>(power * power);
    }

    std::vector<cplxdb> solution;
    bool solved = solve_linear_system(normal, rhs, solution);
    for (int attempt = 0; !solved && attempt < 5; attempt++)
    {
        auto shifted = normal;
        const double shift = std::pow(10.0, -14 + attempt);
        for (int j = 0; j < n_unknowns; j++)
        {
            shifted[j][j] += shift;
        }
        solved = solve_linear_system(shifted, rhs, solution);
    }

    if (!solved)
    {
        ridge_valid = false;
        return;
    }

    ridge_num.assign(solution.begin(), solution.begin() + n_num);
    ridge_den.assign(n_den + 1, {0.0, 0.0});
    ridge_den[0] = {1.0, 0.0};
    for (int j = 0; j < n_den; j++)
    {
        ridge_den[j + 1] = solution[n_num + j];
    }

    ridge_valid = true;
    for (const auto &coeff: ridge_num)
    {
        ridge_valid = ridge_valid && isfinite(coeff);
    }
    for (const auto &coeff: ridge_den)
    {
        ridge_valid = ridge_valid && isfinite(coeff);
    }
}

cplxdb
AnalyContPade::get_thiele(const cplxdb &x, double *denominator_abs) const
{
    cplxdb tmp = {1.0, 0.0};

    for (int i_par = n_pars - 1; i_par > 0; i_par--)
    {
        tmp = 1.0 + par_y[i_par] * (x - par_x[i_par-1]) / tmp;
    }
    if (denominator_abs != nullptr)
    {
        *denominator_abs = std::abs(tmp);
    }
    return par_y[0] / tmp;
}

cplxdb
AnalyContPade::get_ridge(const cplxdb &x) const
{
    const cplxdb x_scaled = x / ridge_x_scale;
    const cplxdb numerator = eval_poly(ridge_num, x_scaled);
    cplxdb denominator = eval_poly(ridge_den, x_scaled);

    const double denominator_floor = std::max(0.0, Params::pade_denominator_floor);
    if (denominator_floor > 0.0 && std::abs(denominator) < denominator_floor)
    {
        if (std::abs(denominator) < std::numeric_limits<double>::min())
        {
            denominator = {denominator_floor, 0.0};
        }
        else
        {
            denominator *= denominator_floor / std::abs(denominator);
        }
    }

    return ridge_y_scale * numerator / denominator;
}

cplxdb
AnalyContPade::get(const cplxdb &x) const
{
    if (method == Method::Ridge && ridge_valid)
    {
        const cplxdb y = get_ridge(x);
        if (isfinite(y))
        {
            return y;
        }
    }
    else if (method == Method::RidgeGuard && ridge_valid)
    {
        double thiele_den_abs = 0.0;
        const cplxdb thiele_y = get_thiele(x, &thiele_den_abs);
        const double den_cut = std::max(0.0, Params::pade_thiele_den_cut);
        if (isfinite(thiele_y) && (den_cut == 0.0 || thiele_den_abs >= den_cut))
        {
            return thiele_y;
        }
        const cplxdb ridge_y = get_ridge(x);
        if (isfinite(ridge_y))
        {
            return ridge_y;
        }
    }
    return get_thiele(x);
}

}
