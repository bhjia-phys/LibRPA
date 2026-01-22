/*!
 * @file      analycont.h
 * @brief     Utilities for analytic continuation
 * @author    Min-Ye Zhang
 * @date      2024-04-23
 */
#pragma once
#include <vector>

#include "base_utility.h"
#include "complexmatrix.h"

// 如果启用了GreenX解析延拓，包含相关头文件
#ifdef LIBRPA_USE_GREENX_AC
extern "C" {
    struct pade_model;  // 前向声明GreenX的pade_model结构体
    // 前向声明GreenX的C接口函数
    struct pade_model *thiele_pade_mp(int n_par, const std::complex<double> *x_ref,
                                       const std::complex<double> *y_ref, int do_greedy,
                                       int precision, int symmetry);
    std::complex<double> evaluate_thiele_pade_mp(const std::complex<double> x,
                                                 struct pade_model *params_ptr);
    void free_pade_model(struct pade_model *model);
}
#endif

namespace LIBRPA
{

class AnalyContPade
{
private:
    int n_pars;
    std::vector<cplxdb> par_x;
    std::vector<cplxdb> par_y;

public:
    AnalyContPade(int n_pars_in,
                  const std::vector<cplxdb> &xs,
                  const std::vector<cplxdb> &data);

    cplxdb get(const cplxdb &x) const;
};

// 新增的Nevanlinna类声明
class AnalyContNevanlinna
{
private:
    int n_pars;
    std::vector<cplxdb> par_x;
    std::vector<cplxdb> par_y;
    std::vector<cplxdb> phis_;
    std::vector<ComplexMatrix> abcds_;

public:
    AnalyContNevanlinna(int n_pars_in, const std::vector<cplxdb> &xs, const std::vector<cplxdb> &data);
    cplxdb get(const cplxdb &x) const;
};

// 新增的自能专用Nevanlinna类声明
class AnalyContNevanlinnaSelfEnergy
{
private:
    int n_pars;
    bool apply_physical_constraints_;
    std::vector<cplxdb> par_x;
    std::vector<cplxdb> par_y;
    std::vector<cplxdb> phis_;
    std::vector<ComplexMatrix> abcds_;

public:
    AnalyContNevanlinnaSelfEnergy(int n_pars_in, const std::vector<cplxdb> &xs,
                                 const std::vector<cplxdb> &data, bool apply_constraints = true);
    cplxdb get(const cplxdb &x) const;
};

// 新增：使用GreenX库的无限精度Pade近似类
// 注意：此类仅在LIBRPA_USE_GREENX_AC定义时可用
#ifdef LIBRPA_USE_GREENX_AC
class AnalyContPadeGreenX
{
private:
    int n_pars;
    int precision;
    int use_greedy;
    int symmetry;
    struct pade_model *model;  // 使用全局命名空间的pade_model
    std::vector<cplxdb> par_x;
    std::vector<cplxdb> par_y;

public:
    // 构造函数
    // n_pars_in: Pade参数数量
    // xs: 参考点数组
    // data: 参考函数值数组
    // precision_in: 精度(bit), 如64, 128, 256等，默认128
    // use_greedy_in: 是否使用贪心算法(推荐), 1=是, 0=否, 默认1
    // symmetry_in: 对称性, 0=无, 默认0
    AnalyContPadeGreenX(int n_pars_in, const std::vector<cplxdb> &xs,
                        const std::vector<cplxdb> &data,
                        int precision_in = 128, int use_greedy_in = 1, int symmetry_in = 0);

    // 析构函数：自动释放GreenX模型
    ~AnalyContPadeGreenX();

    // 在点x处评估Pade近似
    cplxdb get(const cplxdb &x) const;

    // 获取精度
    int get_precision() const { return precision; }

    // 获取参数数量
    int get_n_pars() const { return n_pars; }
};
#endif

} // namespace LIBRPA
