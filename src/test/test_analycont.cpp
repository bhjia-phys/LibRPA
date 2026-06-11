#include "../analycont.h"
#include "../params.h"

#include <cassert>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <string>
#include "testutils.h"
#include <iostream>

namespace
{

cplxdb legacy_thiele_eval(
        int n_pars_in,
        const std::vector<cplxdb> &xs,
        const std::vector<cplxdb> &data,
        const cplxdb &x)
{
    int n_pars = n_pars_in;
    const int n_data = static_cast<int>(data.size());
    std::vector<cplxdb> par_x;
    std::vector<cplxdb> data_npar;
    assert(n_pars > 0);

    if (n_data <= n_pars)
    {
        n_pars = n_data;
        par_x = xs;
        data_npar = data;
    }
    else
    {
        par_x.resize(n_pars);
        data_npar.resize(n_pars);
        const int step = n_data / (n_pars - 1);
        for (int ipar = 0; ipar < n_pars - 1; ipar++)
        {
            par_x[ipar] = xs[ipar * step];
            data_npar[ipar] = data[ipar * step];
        }
        data_npar[n_pars - 1] = data[n_data - 1];
    }

    std::vector<std::vector<cplxdb>> g(
            n_pars, std::vector<cplxdb>(n_pars, {0.0, 0.0}));
    for (int i_par = 0; i_par < n_pars; i_par++)
    {
        g[i_par][0] = data_npar[i_par];
    }

    for (int i_par = 1; i_par < n_pars; i_par++)
    {
        for (int i = i_par; i < n_pars; i++)
        {
            g[i][i_par] =
                (g[i_par - 1][i_par - 1] - g[i][i_par - 1])
                / ((par_x[i] - par_x[i_par - 1]) * g[i][i_par - 1]);
        }
    }

    std::vector<cplxdb> par_y(n_pars);
    for (int i_par = 0; i_par < n_pars; i_par++)
    {
        par_y[i_par] = g[i_par][i_par];
    }

    cplxdb tmp = {1.0, 0.0};
    for (int i_par = n_pars - 1; i_par > 0; i_par--)
    {
        tmp = 1.0 + par_y[i_par] * (x - par_x[i_par - 1]) / tmp;
    }
    return par_y[0] / tmp;
}

} /* end anonymous namespace */


void test_analycont_pade()
{
    using std::vector;
    using std::cout;

    Params::anacon_method = "thiele";
    int n = 6;
    // original function: y = -1 / (x - x0), x0 = {1.0, 1.0}
    const cplxdb x0 = {2.0, 2.0};
    std::vector<cplxdb> xs(n);
    std::vector<cplxdb> data(n);
    for (int i = 0; i < n; i++)
    {
        xs[i] = {0.0, static_cast<double>(i+1)};
        data[i] = -1.0 / (xs[i] - x0);
    }
    cout << "Input x: " << xs << "\n";
    cout << "Input y: " << data << "\n";

    LIBRPA::AnalyContPade pade(n, xs, data);
    // std::cout << data << "\n";
    const cplxdb test_x = {1.0, 1.0};
    const cplxdb ref = -1.0 / (test_x - x0);
    cplxdb test_y = pade.get(test_x);
    cout << "Reference: "<< ref << "\n";
    cout << "     Test: "<< test_y << "\n";
    assert(fequal(ref, test_y));
}

void test_analycont_default_thiele_matches_legacy()
{
    using std::vector;

    const std::string old_method = Params::anacon_method;
    const double old_lambda = Params::pade_ridge_lambda;
    const double old_den_weight = Params::pade_ridge_den_weight;
    const double old_den_floor = Params::pade_denominator_floor;

    const int n = 12;
    const cplxdb pole1 = {1.4, -0.2};
    const cplxdb pole2 = {-2.0, 0.5};
    vector<cplxdb> xs(n);
    vector<cplxdb> data(n);
    for (int i = 0; i < n; i++)
    {
        xs[i] = {0.0, 0.11 * static_cast<double>(i + 1)};
        data[i] = 0.6 / (xs[i] - pole1)
            - 0.15 / (xs[i] - pole2)
            + cplxdb{0.02, -0.01};
    }

    Params::anacon_method = "thiele";
    Params::pade_ridge_lambda = 1.0e8;
    Params::pade_ridge_den_weight = 1.0e7;
    Params::pade_denominator_floor = 1.0;
    LIBRPA::AnalyContPade pade(n, xs, data);

    const vector<cplxdb> targets = {
        {0.0, 0.0},
        {0.3, 0.0},
        {-0.7, 0.25},
        {0.0, 0.9},
    };
    for (const auto &target: targets)
    {
        const cplxdb legacy = legacy_thiele_eval(n, xs, data, target);
        const cplxdb current = pade.get(target);
        assert(std::abs(legacy - current) < 1.0e-12);
    }

    Params::anacon_method = old_method;
    Params::pade_ridge_lambda = old_lambda;
    Params::pade_ridge_den_weight = old_den_weight;
    Params::pade_denominator_floor = old_den_floor;
}

void test_analycont_ridge_zero_lambda_is_not_thiele()
{
    using std::vector;

    const std::string old_method = Params::anacon_method;
    const double old_lambda = Params::pade_ridge_lambda;
    const double old_den_weight = Params::pade_ridge_den_weight;
    const double old_den_floor = Params::pade_denominator_floor;

    const int n = 30;
    vector<cplxdb> xs(n);
    vector<cplxdb> data(n);
    for (int i = 0; i < n; i++)
    {
        const double t = static_cast<double>(i + 1);
        xs[i] = {0.0, 0.09 * t + 0.002 * std::pow(t, 1.2)};
        data[i] = 0.8 / (xs[i] - cplxdb{-1.2, 0.4})
            - 0.35 / (xs[i] - cplxdb{-3.4, -0.6})
            + 0.18 / (xs[i] - cplxdb{1.1, 1.2})
            + cplxdb{1.0e-3 * std::sin(1.7 * t),
                      -7.0e-4 * std::cos(0.9 * t)};
    }

    const int n_params = 8;
    const cplxdb target = {0.1, 0.0};

    Params::anacon_method = "thiele";
    LIBRPA::AnalyContPade thiele(n_params, xs, data);
    const cplxdb y_thiele = thiele.get(target);

    Params::anacon_method = "ridge";
    Params::pade_ridge_lambda = 0.0;
    Params::pade_ridge_den_weight = 10.0;
    Params::pade_denominator_floor = 0.0;
    LIBRPA::AnalyContPade ridge_zero(n_params, xs, data);
    const cplxdb y_ridge_zero = ridge_zero.get(target);

    assert(std::isfinite(y_ridge_zero.real()));
    assert(std::isfinite(y_ridge_zero.imag()));
    assert(std::abs(y_ridge_zero - y_thiele) > 1.0e-3);

    Params::anacon_method = old_method;
    Params::pade_ridge_lambda = old_lambda;
    Params::pade_ridge_den_weight = old_den_weight;
    Params::pade_denominator_floor = old_den_floor;
}

void test_analycont_ridge_pade()
{
    using std::vector;
    using std::cout;

    const int n = 32;
    const cplxdb pole1 = {-1.3, 0.4};
    const cplxdb pole2 = {-4.0, -0.2};
    auto ref_func = [&](const cplxdb &x)
    {
        return 0.7 / (x - pole1) - 0.25 / (x - pole2) + cplxdb{0.05, 0.02};
    };

    vector<cplxdb> xs(n);
    vector<cplxdb> data(n);
    for (int i = 0; i < n; i++)
    {
        xs[i] = {0.0, 0.15 * static_cast<double>(i + 1)};
        const double noise_re = 1.0e-8 * std::sin(1.7 * static_cast<double>(i + 1));
        const double noise_im = 1.0e-8 * std::cos(2.3 * static_cast<double>(i + 1));
        data[i] = ref_func(xs[i]) + cplxdb{noise_re, noise_im};
    }

    Params::anacon_method = "ridge";
    Params::pade_ridge_lambda = 1.0e-8;
    Params::pade_ridge_den_weight = 10.0;
    Params::pade_denominator_floor = 1.0e-8;

    LIBRPA::AnalyContPade pade(14, xs, data);
    const cplxdb test_x = {0.45, 0.0};
    const cplxdb ref = ref_func(test_x);
    const cplxdb test_y = pade.get(test_x);
    cout << "Ridge reference: "<< ref << "\n";
    cout << "      Ridge test: "<< test_y << "\n";

    assert(std::isfinite(test_y.real()));
    assert(std::isfinite(test_y.imag()));
    assert(std::abs(test_y) < 10.0);
    assert(std::abs(test_y - ref) < 1.0e-3);

    Params::anacon_method = "thiele";
}

void test_analycont_method_params()
{
    const std::string old_method = Params::anacon_method;
    const int old_nfreq = Params::nfreq;
    const int old_n_params = Params::n_params_anacon;
    const double old_lambda = Params::pade_ridge_lambda;
    const double old_den_weight = Params::pade_ridge_den_weight;
    const double old_den_floor = Params::pade_denominator_floor;
    const double old_thiele_cut = Params::pade_thiele_den_cut;

    Params::nfreq = 8;
    Params::n_params_anacon = -1;
    Params::anacon_method = "PaDe";
    Params::pade_ridge_lambda = -1.0;
    Params::pade_ridge_den_weight = -2.0;
    Params::pade_denominator_floor = -3.0;
    Params::pade_thiele_den_cut = -4.0;
    Params::check_consistency();
    assert(Params::n_params_anacon == 8);
    assert(Params::anacon_method == "thiele");
    assert(Params::pade_ridge_lambda == 0.0);
    assert(Params::pade_ridge_den_weight == 0.0);
    assert(Params::pade_denominator_floor == 0.0);
    assert(Params::pade_thiele_den_cut == 0.0);

    Params::anacon_method = "ridge-guard";
    Params::check_consistency();
    assert(Params::anacon_method == "ridge_guard");

    bool rejected = false;
    Params::anacon_method = "ridge_den";
    try
    {
        Params::check_consistency();
    }
    catch (const std::logic_error &)
    {
        rejected = true;
    }
    assert(rejected);

    Params::anacon_method = old_method;
    Params::nfreq = old_nfreq;
    Params::n_params_anacon = old_n_params;
    Params::pade_ridge_lambda = old_lambda;
    Params::pade_ridge_den_weight = old_den_weight;
    Params::pade_denominator_floor = old_den_floor;
    Params::pade_thiele_den_cut = old_thiele_cut;
}

int main(int argc, char *argv[])
{
    test_analycont_pade();
    test_analycont_default_thiele_matches_legacy();
    test_analycont_ridge_zero_lambda_is_not_thiele();
    test_analycont_ridge_pade();
    test_analycont_method_params();
}
