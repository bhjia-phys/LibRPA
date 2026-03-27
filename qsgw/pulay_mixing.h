#ifndef PULAY_MIXER_H
#define PULAY_MIXER_H

#include "matrix.h"
#include <vector>
#include <deque>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <iostream>

struct PulayMixerState {
    int max_history = 0;
    int current_step = 0;
    double mixing_beta = 0.0;
    bool initialized = false;
    int nrows = 0;
    int ncols = 0;
    std::vector<matrix> input_history;
    std::vector<matrix> residual_history;
};

class PulayMixer {
private:
    int max_history_;           // 最大历史记录数
    int current_step_;          // 当前步数
    double mixing_beta_;        // 混合参数
    bool initialized_;          // 是否已初始化

    // 历史数据存储
    std::vector<matrix> input_history_;   // 输入矩阵历史
    std::vector<matrix> residual_history_; // 残差矩阵历史

    // 矩阵尺寸
    int nrows_;
    int ncols_;

    // 自适应参数
    bool adaptive_enabled_;               // 是否启用自适应调整
    double beta_min_;                    // 最小 beta 值
    double beta_max_;                    // 最大 beta 值
    std::deque<double> residual_history_norms_;
    std::deque<double> eigenvalue_change_history_;  // 残差范数历史（用于自适应分析）
    int last_beta_adjustment_step_;      // 上次调整 beta 的步数

    // 私有成员函数声明
    double get_adaptive_beta();  // 自适应 beta 调整
    double matrix_inner_product(const matrix& A, const matrix& B);
    matrix solve_linear_system(const matrix& A, const matrix& b);
    
public:
    // 构造函数
    PulayMixer(int max_history = 5, double mixing_beta = 0.1);
    
    /**
     * @brief 初始化混合器，必须在第一次调用 mix 前调用
     * @param initial_guess 初始猜测矩阵
     */
    void initialize(const matrix& initial_guess);
    
    /**
     * @brief 执行 Pulay 混合
     * @param current_output 当前迭代的输出矩阵
     * @return 混合后的新输入矩阵
     */
    matrix mix(const matrix& current_output);
    matrix mix(const matrix& current_output, double eigenvalue_change_ev);
    
    /**
     * @brief 获取当前历史记录大小
     */
    int get_history_size() const;
    
    /**
     * @brief 获取当前迭代步数
     */
    int get_current_step() const;
    
    /**
     * @brief 重置混合器
     */
    void reset();
    
    /**
     * @brief 设置混合参数
     */
    void set_mixing_beta(double beta);
    
    /**
     * @brief 获取混合参数
     */
    double get_mixing_beta() const;

    /**
     * @brief 启用或禁用自适应参数调整
     */
    void set_adaptive_enabled(bool enabled) { adaptive_enabled_ = enabled; }

    /**
     * @brief 设置 beta 值的范围
     */
    void set_beta_bounds(double min_beta, double max_beta) {
        beta_min_ = min_beta;
        beta_max_ = max_beta;
    }

    /**
     * @brief 获取当前残差范数历史
     */
    const std::deque<double>& get_residual_history() const {
        return residual_history_norms_;
    }

    /**
     * @brief 导出/恢复 mixer 续算所需的最小状态
     */
    PulayMixerState snapshot() const;
    void restore(const PulayMixerState& state);
};

#endif // PULAY_MIXER_H
