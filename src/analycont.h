/*!
 * @file      analycont.h
 * @brief     Utilities for analytic continuation
 * @author    Min-Ye Zhang
 * @date      2024-04-23
 */
#pragma once
#include <vector>

#include "base_utility.h"

namespace LIBRPA
{

class AnalyContPade
{
private:
    enum class Method
    {
        Thiele,
        Ridge,
        RidgeGuard,
    };

    Method method;
    int n_pars;
    std::vector<cplxdb> par_x;
    std::vector<cplxdb> par_y;
    bool ridge_valid;
    int ridge_num_degree;
    int ridge_den_degree;
    double ridge_x_scale;
    double ridge_y_scale;
    std::vector<cplxdb> ridge_num;
    std::vector<cplxdb> ridge_den;

    void build_thiele(int n_pars_in,
                      const std::vector<cplxdb> &xs,
                      const std::vector<cplxdb> &data);
    void build_ridge(int n_pars_in,
                     const std::vector<cplxdb> &xs,
                     const std::vector<cplxdb> &data);
    cplxdb get_thiele(const cplxdb &x, double *denominator_abs = nullptr) const;
    cplxdb get_ridge(const cplxdb &x) const;

public:
    AnalyContPade(int n_pars_in,
                  const std::vector<cplxdb> &xs,
                  const std::vector<cplxdb> &data);

    /*!
     * @brief get the value of continued function at complex number
     *
     * @param [in]    x    complex argument of function
     *
     * @return    a complex double, the value of function at x
     */
    cplxdb get(const cplxdb &x) const;
};

}
