/*!
 * @file      example_greenx_ac.cpp
 * @brief     Example demonstrating the use of GreenX infinite precision Pade analytic continuation
 * @author    Generated for LibRPA
 * @date      2025-01-21
 */

#include "analycont.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <complex>
#include <iomanip>

using namespace LIBRPA;
using cplxdb = std::complex<double>;

int main()
{
    std::cout << "========================================\n";
    std::cout << "GreenX Infinite Precision Pade Example\n";
    std::cout << "========================================\n\n";

    // 示例：双极点模型函数
    // y = 0.9*(x - 0.25)^(-1) + (x - 0.75)^(-1) - 0.7*(x + 0.25)^(-1) - (x + 0.75)^(-1)
    auto reference_function = [](cplxdb x) -> cplxdb {
        return 0.9 / (x - 0.25) + 1.0 / (x - 0.75) 
               - 0.7 / (x + 0.25) - 1.0 / (x + 0.75);
    };

    // 参数设置
    const int n_pars = 16;           // Pade参数数量
    const int precision = 128;       // 精度（bit）
    const int use_greedy = 1;        // 使用贪心算法
    const int symmetry = 0;           // 无对称性

    // 沿虚轴生成参考点
    std::vector<cplxdb> x_ref(n_pars);
    std::vector<cplxdb> y_ref(n_pars);

    double x_start = -1.1;
    double x_end = 1.0;
    double step = (x_end - x_start) / (n_pars - 1);

    std::cout << "Reference points along imaginary axis:\n";
    std::cout << "  Creating " << n_pars << " points from i*" << x_start << " to i*" << x_end << "\n\n";

    for (int i = 0; i < n_pars; i++)
    {
        x_ref[i] = cplxdb(0.0, x_start + i * step);
        y_ref[i] = reference_function(x_ref[i]);
    }

    // 创建GreenX无限精度Pade模型
    std::cout << "Creating GreenX Pade model:\n";
    std::cout << "  Precision: " << precision << " bits\n";
    std::cout << "  Greedy algorithm: " << (use_greedy ? "ON" : "OFF") << "\n";
    std::cout << "  Symmetry: " << symmetry << " (none)\n\n";

    AnalyContPadeGreenX pade_greenx(n_pars, x_ref, y_ref, precision, use_greedy, symmetry);

    // 在实轴上评估（带小虚偏移）
    const int n_query = 20;
    const double imag_shift = 0.01;

    std::cout << "Evaluating along real axis with imaginary shift = " << imag_shift << ":\n";
    std::cout << "----------------------------------------------------------------\n";
    std::cout << "  Re(x)     |  Im(y_pade)   |  Im(y_exact)   |  |diff|/|y_exact|\n";
    std::cout << "------------+---------------+----------------+------------------\n";

    double max_rel_error = 0.0;

    for (int i = 0; i < n_query; i++)
    {
        double x_real = x_start + (x_end - x_start) * i / (n_query - 1);
        cplxdb x_query = cplxdb(x_real, imag_shift);

        cplxdb y_pade = pade_greenx.get(x_query);
        cplxdb y_exact = reference_function(x_query);

        double rel_error = std::abs(y_pade - y_exact) / std::abs(y_exact);
        max_rel_error = std::max(max_rel_error, rel_error);

        if (i % 2 == 0)  // 只打印每第二个点以节省空间
        {
            std::cout << std::fixed << std::setprecision(4);
            std::cout << " " << std::setw(10) << x_real << " | ";
            std::cout << std::setw(13) << y_pade.imag() << " | ";
            std::cout << std::setw(14) << y_exact.imag() << " | ";
            std::cout << std::scientific << std::setw(16) << rel_error << "\n";
        }
    }

    std::cout << "----------------------------------------------------------------\n";
    std::cout << "Maximum relative error: " << max_rel_error << "\n\n";

    // 比较不同精度
    std::cout << "Comparing different precisions:\n";
    std::cout << "----------------------------------------\n";

    int precisions[] = {64, 128, 256};
    for (int prec : precisions)
    {
        AnalyContPadeGreenX pade_test(n_pars, x_ref, y_ref, prec, use_greedy, symmetry);

        cplxdb x_test = cplxdb(0.5, imag_shift);
        cplxdb y_pade_test = pade_test.get(x_test);
        cplxdb y_exact_test = reference_function(x_test);
        double error_test = std::abs(y_pade_test - y_exact_test) / std::abs(y_exact_test);

        std::cout << "  Precision = " << std::setw(3) << prec << " bits: ";
        std::cout << "error = " << std::scientific << error_test << "\n";
    }

    std::cout << "\nExample completed successfully!\n";
    return 0;
}
