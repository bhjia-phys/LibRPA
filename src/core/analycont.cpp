/*!
 * @author    Min-Ye Zhang
 * @date      2024-04-25
 */
#include "../utils/constants.h"
#include "analycont.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
// #include <iostream>

#include "../math/complexmatrix.h"
// #include "../io/stl_io_helper.h"

namespace librpa_int
{

namespace
{

bool finite_complex(const cplxdb &value)
{
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}

} // namespace

AnalyContPade::AnalyContPade(int n_pars_in, const std::vector<cplxdb> &xs, const std::vector<cplxdb> &data)
    : n_pars(n_pars_in)
{
    const int n_data = data.size();

    // Use all data if requested number of parameters is invalid
    if (n_pars < 0) n_pars = n_data;
    if (xs.size() != data.size() || n_data == 0 || n_pars == 0)
        throw std::invalid_argument(
            "Padé requires nonempty matching data and a nonzero parameter count");

    if (n_data <= n_pars)
    {
        // Use all data points
        n_pars = n_data;
        source_xs = xs;
        source_data = data;
    }
    else
    {
        // Select the data points evenly, when number of parameters are fewer than data points
        source_xs.resize(n_pars);
        source_data.resize(n_pars);
        int step = n_pars == 1 ? 0 : n_data / (n_pars - 1);
        for (int ipar = 0; ipar < n_pars - 1; ipar++)
        {
            source_xs[ipar] = xs[ipar * step];
            source_data[ipar] = data[ipar * step];
        }
        source_xs[n_pars-1] = xs[n_pars == 1 ? 0 : n_data-1];
        source_data[n_pars-1] = data[n_pars == 1 ? 0 : n_data-1];
    }

    // Band exclusion can make the correlation self-energy identically zero.
    if (!source_data.empty() && std::all_of(source_data.begin(), source_data.end(),
                                          [](const cplxdb &value) { return value == cplxdb(0.0, 0.0); }))
    {
        n_pars = 1;
        par_y = {cplxdb(0.0, 0.0)};
        return;
    }

    // Calculate the continuation coefficients, using Thiel's reciprocal difference method
    ComplexMatrix g(n_pars, n_pars);
    double data_scale = 0.0;
    for (int i_par = 0; i_par < n_pars; i_par++)
    {
        if (!finite_complex(source_xs[i_par]) || !finite_complex(source_data[i_par]))
            throw std::invalid_argument("Padé source data must be finite");
        g(i_par, 0) = source_data[i_par];
        data_scale = std::max(data_scale, std::abs(source_data[i_par]));
    }
    par_y.resize(n_pars);
    par_y[0] = g(0, 0);
    const double epsilon = std::numeric_limits<double>::epsilon();
    const auto lower_order_fits = [&](int n_terms)
    {
        for (int i = 0; i != static_cast<int>(source_xs.size()); ++i)
        {
            cplxdb tmp{1.0, 0.0};
            for (int p = n_terms - 1; p > 0; --p)
                tmp = 1.0 + par_y[p] * (source_xs[i] - source_xs[p - 1]) / tmp;
            const cplxdb value = par_y[0] / tmp;
            if (!finite_complex(value) ||
                std::abs(value - source_data[i]) > 64.0 * epsilon * data_scale)
                return false;
        }
        return true;
    };
    for (int i_par = 1; i_par < n_pars; i_par++)
    {
        const cplxdb pivot = g(i_par - 1, i_par - 1);
        double column_scale = std::abs(pivot);
        double deviation = 0.0;
        bool zero_pivot = pivot == cplxdb{0.0, 0.0};
        for (int i = i_par; i < n_pars; ++i)
        {
            const cplxdb previous = g(i, i_par - 1);
            column_scale = std::max(column_scale, std::abs(previous));
            deviation = std::max(deviation, std::abs(previous - pivot));
            zero_pivot = zero_pivot || previous == cplxdb{0.0, 0.0};
        }
        // Terminate a lower-order fraction only after checking all selected data.
        if ((zero_pivot || deviation <= 32.0 * epsilon * column_scale) && lower_order_fits(i_par))
        {
            n_pars = i_par;
            par_y.resize(n_pars);
            return;
        }
        if (zero_pivot)
            throw std::runtime_error(
                "Padé reciprocal differences have a zero pivot without a valid lower-order fit");
        for (int i = i_par; i < n_pars; i++)
        {
            const cplxdb denominator = (source_xs[i] - source_xs[i_par - 1]) * g(i, i_par - 1);
            if (denominator == cplxdb{0.0, 0.0} || !finite_complex(denominator))
                throw std::runtime_error("Padé reciprocal-difference denominator is singular");
            g(i, i_par) = (pivot - g(i, i_par - 1)) / denominator;
            if (!finite_complex(g(i, i_par)))
                throw std::runtime_error(
                    "Padé reciprocal differences produced non-finite coefficients");
        }
        par_y[i_par] = g(i_par, i_par);
    }
}

cplxdb
AnalyContPade::get(const cplxdb &x) const
{
    cplxdb tmp = {1.0, 0.0};

    for (int i_par = n_pars - 1; i_par > 0; i_par--)
    {
        tmp = 1.0 + par_y[i_par] * (x - source_xs[i_par-1]) / tmp;
    }
    return par_y[0] / tmp;
}

cplxdb
AnalyContPade::get_derivative(const cplxdb &x) const
{
    if (n_pars <= 1)
    {
        return {0.0, 0.0};
    }

    std::vector<cplxdb> g(n_pars);
    std::vector<cplxdb> dg(n_pars, {0.0, 0.0});

    g[n_pars-1] = par_y[n_pars-1];
    for (int i_par = n_pars - 2; i_par >= 0; i_par--)
    {
        const cplxdb denominator = 1.0 + (x - source_xs[i_par]) * g[i_par+1];
        g[i_par] = par_y[i_par] / denominator;
        dg[i_par] = -par_y[i_par]
                    * (g[i_par+1] + (x - source_xs[i_par]) * dg[i_par+1])
                    / (denominator * denominator);
    }
    return dg[0];
}

const std::vector<double> get_specfunc(const AnalyCont &ac, const std::vector<cplxdb> &omegas,
                                       const double &ref, const double &e_ks, const double &v_xc,
                                       const double &v_exx,
                                       const double &sigc_omega_imag_shift,
                                       const double &gf_omega_imag_shift)
{
    const int n_freq = omegas.size();
    std::vector<double> sf(n_freq, 0.0);
    for (int i = 0; i < n_freq; i++)
    {
        const auto omega_gf = omegas[i] + cplxdb(0.0, gf_omega_imag_shift);
        const auto omega_sigc = omegas[i] + cplxdb(0.0, sigc_omega_imag_shift);
        cplxdb sf_c = omega_gf - e_ks + v_xc - v_exx - ac.get(omega_sigc - ref);
        sf[i] = - ((1.0 / PI) / sf_c).imag();
    }
    return sf;
}

}
