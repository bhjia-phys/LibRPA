/*!
 * @file      analycont.cpp
 * @brief     Implementation of analytic continuation utilities
 * @author    Min-Ye Zhang
 * @date      2024-04-25
 */

#include "analycont.h"

#include <algorithm>
#include <cassert>
#include <complex>
#include <vector>

#include "complexmatrix.h"

namespace LIBRPA
{

AnalyContPade::AnalyContPade(int n_pars_in, const std::vector<cplxdb> &xs, const std::vector<cplxdb> &data)
    : n_pars(n_pars_in)
{
    const int n_data = static_cast<int>(data.size());
    std::vector<cplxdb> data_npar;

    assert(n_pars > 0);

    // Same point selection scheme as the historical driver: evenly sample n_pars points.
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
        par_x[n_pars - 1] = xs[n_data - 1];
        data_npar[n_pars - 1] = data[n_data - 1];
    }

    ComplexMatrix g(n_pars, n_pars);
    for (int i = 0; i < n_pars; i++)
    {
        g(i, 0) = data_npar[i];
    }

    for (int i_par = 1; i_par < n_pars; i_par++)
    {
        for (int i = i_par; i < n_pars; i++)
        {
            g(i, i_par) = (g(i_par - 1, i_par - 1) - g(i, i_par - 1)) /
                          ((par_x[i] - par_x[i_par - 1]) * g(i, i_par - 1));
        }
    }

    par_y.resize(n_pars);
    for (int i = 0; i < n_pars; i++)
    {
        par_y[i] = g(i, i);
    }
}

cplxdb AnalyContPade::get(const cplxdb &x) const
{
    cplxdb tmp = {1.0, 0.0};
    for (int i_par = n_pars - 1; i_par > 0; i_par--)
    {
        tmp = 1.0 + par_y[i_par] * (x - par_x[i_par - 1]) / tmp;
    }
    return par_y[0] / tmp;
}

} // namespace LIBRPA