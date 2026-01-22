/*!
 * @author    Min-Ye Zhang
 * @date      2024-04-25
 */

#include "analycont.h"
#include <cassert>
#include <vector>
#include <complex>
#include <algorithm>
#include <iostream>

// 如果启用了GreenX解析延拓，包含相关头文件
#ifdef LIBRPA_USE_GREENX_AC
// 注意：不要直接包含pade_mp.h，因为它包含内联函数会导致多重定义
// 函数声明已在analycont.h中通过extern "C"提供
#endif

namespace LIBRPA
{

// AnalyContPade 实现
AnalyContPade::AnalyContPade(int n_pars_in, const std::vector<cplxdb> &xs, const std::vector<cplxdb> &data)
    : n_pars(n_pars_in)
{
    int n_data = data.size();
    std::vector<cplxdb> data_npar;

    assert (n_pars > 0);

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
        int step = n_data / (n_pars - 1);
        for (int ipar = 0; ipar < n_pars - 1; ipar++)
        {
            par_x[ipar] = xs[ipar * step];
            data_npar[ipar] = data[ipar * step];
        }
        par_x[n_pars-1] = xs[n_data-1];
        data_npar[n_pars-1] = data[n_data-1];
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

cplxdb AnalyContPade::get(const cplxdb &x) const
{
    cplxdb tmp = {1.0, 0.0};

    for (int i_par = n_pars - 1; i_par > 0; i_par--)
    {
        tmp = 1.0 + par_y[i_par] * (x - par_x[i_par-1]) / tmp;
    }
    return par_y[0] / tmp;
}

// AnalyContNevanlinna 实现
AnalyContNevanlinna::AnalyContNevanlinna(int n_pars_in, const std::vector<cplxdb> &xs, const std::vector<cplxdb> &data)
    : n_pars(n_pars_in)
{
    int n_data = data.size();
    std::vector<cplxdb> data_npar;

    assert(n_pars > 0);

    // 数据点选取策略与Pade相同
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
        int step = n_data / (n_pars - 1);
        for (int ipar = 0; ipar < n_pars - 1; ipar++)
        {
            par_x[ipar] = xs[ipar * step];
            data_npar[ipar] = data[ipar * step];
        }
        par_x[n_pars-1] = xs[n_data-1];
        data_npar[n_pars-1] = data[n_data-1];
    }

    // Nevanlinna核心算法 - 直接在构造函数中实现
    int M = par_x.size();
    phis_.resize(M);
    abcds_.resize(M);
    
    cplxdb I(0.0, 1.0);
    
    // 转换为Nevanlinna的θ值
    std::vector<cplxdb> theta_vals(M);
    for (int i = 0; i < M; i++) {
        cplxdb NG = -data_npar[i]; // 如果是格林函数：G -> NG
        theta_vals[i] = (NG - I) / (NG + I); // Mobius变换
    }
    
    // 反转顺序（按文档建议提高稳定性）
    std::reverse(theta_vals.begin(), theta_vals.end());
    std::vector<cplxdb> reversed_xs = par_x;
    std::reverse(reversed_xs.begin(), reversed_xs.end());
    
    // Schur算法核心
    phis_[0] = theta_vals[0];
    
    // 初始化2x2单位矩阵
    for (int k = 0; k < M; k++) {
        abcds_[k] = ComplexMatrix(2, 2, false);
        abcds_[k].set_as_identity_matrix();
    }
    
    for (int j = 0; j < M - 1; j++) {
        for (int k = j; k < M; k++) {
            ComplexMatrix prod(2, 2, false);
            
            cplxdb freq_diff = (reversed_xs[k] - reversed_xs[j]) / 
                              (reversed_xs[k] - std::conj(reversed_xs[j]));
            
            prod(0,0) = freq_diff;
            prod(0,1) = phis_[j];
            prod(1,0) = std::conj(phis_[j]) * freq_diff;
            prod(1,1) = cplxdb(1.0, 0.0);
            
            abcds_[k] = abcds_[k] * prod;
        }
        
        // 计算下一个φ
        cplxdb denom = abcds_[j+1](1,0) * theta_vals[j+1] - abcds_[j+1](0,0);
        if (std::abs(denom) < 1e-300) {
            phis_[j+1] = cplxdb(0.0, 0.0);
        } else {
            phis_[j+1] = (-abcds_[j+1](1,1) * theta_vals[j+1] + abcds_[j+1](0,1)) / denom;
        }
    }
    
    // 保存参数（与Pade接口兼容）
    par_y = phis_;
}

cplxdb AnalyContNevanlinna::get(const cplxdb &x) const
{
    // 直接在get函数中实现求值逻辑
    int M = phis_.size();
    cplxdb I(0.0, 1.0);
    cplxdb one(1.0, 0.0);
    
    ComplexMatrix result(2, 2, false);
    result.set_as_identity_matrix();
    
    std::vector<cplxdb> reversed_xs = par_x;
    std::reverse(reversed_xs.begin(), reversed_xs.end());
    
    for (int j = 0; j < M; j++) {
        ComplexMatrix prod(2, 2, false);
        
        cplxdb freq_diff = (x - reversed_xs[j]) / (x - std::conj(reversed_xs[j]));
        prod(0,0) = freq_diff;
        prod(0,1) = phis_[j];
        prod(1,0) = std::conj(phis_[j]) * freq_diff;
        prod(1,1) = one;
        
        result = result * prod;
    }
    
    // 参数函数（默认为0，可根据需要修改）
    cplxdb param_function(0.0, 0.0);
    
    // 计算θ(z)
    cplxdb theta_z = (result(0,0) * param_function + result(0,1)) /
                    (result(1,0) * param_function + result(1,1));
    
    // 逆变换
    cplxdb NG = I * (one + theta_z) / (one - theta_z);
    cplxdb result_val = -NG;  // NG -> G
    
    return result_val;
}

// AnalyContNevanlinnaSelfEnergy 实现
AnalyContNevanlinnaSelfEnergy::AnalyContNevanlinnaSelfEnergy(int n_pars_in, const std::vector<cplxdb> &xs, 
                             const std::vector<cplxdb> &data, bool apply_constraints)
    : n_pars(n_pars_in), apply_physical_constraints_(apply_constraints)
{
    int n_data = data.size();
    std::vector<cplxdb> data_npar;

    assert(n_pars > 0);

    // 相同的数据选取策略
    if (n_data <= n_pars) {
        n_pars = n_data;
        par_x = xs;
        data_npar = data;
    } else {
        par_x.resize(n_pars);
        data_npar.resize(n_pars);
        int step = n_data / (n_pars - 1);
        for (int ipar = 0; ipar < n_pars - 1; ipar++) {
            par_x[ipar] = xs[ipar * step];
            data_npar[ipar] = data[ipar * step];
        }
        par_x[n_pars-1] = xs[n_data-1];
        data_npar[n_pars-1] = data[n_data-1];
    }

    // 直接在构造函数中实现Nevanlinna参数计算
    int M = par_x.size();
    phis_.resize(M);
    abcds_.resize(M);
    
    cplxdb I(0.0, 1.0);
    
    // 自能数据的特殊处理：直接使用数据，不进行Mobius变换
    std::vector<cplxdb> theta_vals = data_npar;
    
    // 可选：对自能数据进行缩放以提高数值稳定性
    double max_abs = 0.0;
    for (const auto& val : data_npar) {
        max_abs = std::max(max_abs, std::abs(val));
    }
    if (max_abs > 1e-10) {
        double scale = 1.0 / max_abs;
        for (auto& val : theta_vals) {
            val *= scale;
        }
    }
    
    std::reverse(theta_vals.begin(), theta_vals.end());
    std::vector<cplxdb> reversed_xs = par_x;
    std::reverse(reversed_xs.begin(), reversed_xs.end());
    
    // Schur算法
    phis_[0] = theta_vals[0];
    
    for (int k = 0; k < M; k++) {
        abcds_[k] = ComplexMatrix(2, 2, false);
        abcds_[k].set_as_identity_matrix();
    }
    
    for (int j = 0; j < M - 1; j++) {
        for (int k = j; k < M; k++) {
            ComplexMatrix prod(2, 2, false);
            
            cplxdb freq_diff = (reversed_xs[k] - reversed_xs[j]) / 
                              (reversed_xs[k] - std::conj(reversed_xs[j]));
            
            prod(0,0) = freq_diff;
            prod(0,1) = phis_[j];
            prod(1,0) = std::conj(phis_[j]) * freq_diff;
            prod(1,1) = cplxdb(1.0, 0.0);
            
            abcds_[k] = abcds_[k] * prod;
        }
        
        cplxdb denom = abcds_[j+1](1,0) * theta_vals[j+1] - abcds_[j+1](0,0);
        if (std::abs(denom) < 1e-300) {
            phis_[j+1] = cplxdb(0.0, 0.0);
        } else {
            phis_[j+1] = (-abcds_[j+1](1,1) * theta_vals[j+1] + abcds_[j+1](0,1)) / denom;
        }
    }
    
    par_y = phis_;
}

cplxdb AnalyContNevanlinnaSelfEnergy::get(const cplxdb &x) const 
{
    // 直接在get函数中实现求值逻辑
    int M = phis_.size();
    cplxdb one(1.0, 0.0);
    
    ComplexMatrix result(2, 2, false);
    result.set_as_identity_matrix();
    
    std::vector<cplxdb> reversed_xs = par_x;
    std::reverse(reversed_xs.begin(), reversed_xs.end());
    
    for (int j = 0; j < M; j++) {
        ComplexMatrix prod(2, 2, false);
        
        cplxdb freq_diff = (x - reversed_xs[j]) / (x - std::conj(reversed_xs[j]));
        prod(0,0) = freq_diff;
        prod(0,1) = phis_[j];
        prod(1,0) = std::conj(phis_[j]) * freq_diff;
        prod(1,1) = one;
        
        result = result * prod;
    }
    
    // 参数函数
    cplxdb param_function(0.0, 0.0);
    
    cplxdb result_val = (result(0,0) * param_function + result(0,1)) /
                       (result(1,0) * param_function + result(1,1));
    
    // 应用物理约束（如果需要）
    if (apply_physical_constraints_) {
        // 基本的物理约束
        cplxdb constrained = result_val;
        
        // 约束1: 对于实数频率，ImΣ(ω)应该有正确的因果性符号
        if (std::abs(x.imag()) < 1e-12) {
            // 根据具体物理体系调整符号约束
            // 对于费米子体系，通常ImΣ(ω) ≤ 0
            if (constrained.imag() > 0) {
                constrained = cplxdb(constrained.real(), std::min(constrained.imag(), 0.0));
            }
        }
        
        return constrained;
    }
    
    return result_val;
}

// AnalyContPadeGreenX 实现
AnalyContPadeGreenX::AnalyContPadeGreenX(int n_pars_in, const std::vector<cplxdb> &xs, 
                                          const std::vector<cplxdb> &data,
                                          int precision_in, int use_greedy_in, int symmetry_in)
    : n_pars(n_pars_in), precision(precision_in), use_greedy(use_greedy_in), symmetry(symmetry_in), model(nullptr)
{
    int n_data = data.size();
    std::vector<cplxdb> data_npar;
    std::vector<cplxdb> xs_npar;

    assert(n_pars > 0);

    // 数据点选取策略与AnalyContPade相同
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
        int step = n_data / (n_pars - 1);
        for (int ipar = 0; ipar < n_pars - 1; ipar++)
        {
            par_x[ipar] = xs[ipar * step];
            data_npar[ipar] = data[ipar * step];
        }
        par_x[n_pars-1] = xs[n_data-1];
        data_npar[n_pars-1] = data[n_data-1];
    }

    // 保存数据用于后续使用
    par_y = data_npar;

    // 调用GreenX的thiele_pade_mp函数创建Pade模型
    // 注意：pade_mp.h中已经通过extern "C"声明了这些函数
    ::pade_model *model_ptr = thiele_pade_mp(n_pars, par_x.data(), data_npar.data(), use_greedy, precision, symmetry);
    model = model_ptr;

    if (model == nullptr)
    {
        std::cerr << "Error: Failed to create GreenX Pade model" << std::endl;
    }
}

AnalyContPadeGreenX::~AnalyContPadeGreenX()
{
    if (model != nullptr)
    {
        ::free_pade_model(model);
        model = nullptr;
    }
}

cplxdb AnalyContPadeGreenX::get(const cplxdb &x) const
{
    if (model == nullptr)
    {
        std::cerr << "Error: GreenX Pade model not initialized" << std::endl;
        return cplxdb(0.0, 0.0);
    }

    // 使用全局命名空间中的evaluate_thiele_pade_mp函数
    std::complex<double> result = ::evaluate_thiele_pade_mp(x, model);
    return result;
}

} // namespace LIBRPA