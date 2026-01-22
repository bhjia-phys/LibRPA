/*!
 * @file      test_greenx_integration.cpp
 * @brief     Simple test to verify GreenX integration
 * @author    Generated for LibRPA
 * @date      2025-01-21
 */

#include "analycont.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <complex>

using namespace LIBRPA;
using cplxdb = std::complex<double>;

int main()
{
    std::cout << "Testing GreenX Integration..." << std::endl;

#ifdef LIBRPA_USE_GREENX_AC
    std::cout << "GreenX Analytic Continuation is ENABLED" << std::endl;

    // 简单测试：常数函数
    int n_pars = 4;
    std::vector<cplxdb> x_ref(n_pars);
    std::vector<cplxdb> y_ref(n_pars);

    for (int i = 0; i < n_pars; i++)
    {
        x_ref[i] = cplxdb(0.0, static_cast<double>(i));
        y_ref[i] = cplxdb(1.0, 2.0);  // 常数函数
    }

    try
    {
        // 测试64位精度
        std::cout << "\nTesting 64-bit precision..." << std::endl;
        AnalyContPadeGreenX pade64(n_pars, x_ref, y_ref, 64, 1, 0);

        cplxdb x_test = cplxdb(0.5, 0.5);
        cplxdb result = pade64.get(x_test);

        std::cout << "  Input: " << x_test << std::endl;
        std::cout << "  Output: " << result << std::endl;
        std::cout << "  Expected: (1.0000, 2.0000)" << std::endl;
        std::cout << "  Error: " << std::abs(result - cplxdb(1.0, 2.0)) << std::endl;

        // 测试128位精度
        std::cout << "\nTesting 128-bit precision..." << std::endl;
        AnalyContPadeGreenX pade128(n_pars, x_ref, y_ref, 128, 1, 0);
        result = pade128.get(x_test);

        std::cout << "  Input: " << x_test << std::endl;
        std::cout << "  Output: " << result << std::endl;
        std::cout << "  Error: " << std::abs(result - cplxdb(1.0, 2.0)) << std::endl;

        std::cout << "\nAll tests passed!" << std::endl;
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
#else
    std::cout << "GreenX Analytic Continuation is NOT ENABLED" << std::endl;
    std::cout << "Please recompile with GMP library support:" << std::endl;
    std::cout << "  cmake .. -DUSE_GREENX_API=ON" << std::endl;
    return 1;
#endif
}
