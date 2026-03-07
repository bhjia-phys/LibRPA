#include "pulay_mixing.h"
#include "lapack_connector.h"
#include <deque>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>

// 注意：MKL 使用 LAPACKE 接口而不是 Fortran _ 后缀接口
// 我们将通过 C 接口调用 LAPACKE 函数
extern "C" {
    // LAPACKE C 接口 (Intel MKL)
    int LAPACKE_dgetrf(int matrix_layout, int m, int n, double* a, int lda, int* ipiv);
    int LAPACKE_dgetrs(int matrix_layout, char trans, int n, int nrhs,
                       const double* a, int lda, const int* ipiv, double* b, int ldb);
}
class ConvergenceMonitor {
private:
    std::deque<double> residual_history_;
    int window_size_;
    double convergence_threshold_;
    int stagnation_counter_;
    int oscillation_counter_;

public:
    ConvergenceMonitor(int window = 5, double threshold = 1e-6)
        : window_size_(window), convergence_threshold_(threshold),
          stagnation_counter_(0), oscillation_counter_(0) {}

    enum ConvergenceState {
        CONVERGING,      // 残差稳定下降
        OSCILLATING,     // 残差震荡
        STAGNATING,      // 收敛停滞
        DIVERGING,       // 发散
        STABLE           // 稳定
    };

    // 获取收敛状态描述
    std::string get_state_name(ConvergenceState state) const {
        switch(state) {
            case CONVERGING: return "CONVERGING";
            case OSCILLATING: return "OSCILLATING";
            case STAGNATING: return "STAGNATING";
            case DIVERGING: return "DIVERGING";
            case STABLE: return "STABLE";
            default: return "UNKNOWN";
        }
    }

    // 获取建议的 beta 调整因子
    double get_beta_factor(ConvergenceState state, double current_residual) const {
        switch(state) {
            case CONVERGING:
                // 收敛良好，可以适当增大 beta
                return std::min(1.2, 1.0 + current_residual * 0.1);
            case OSCILLATING:
                // 震荡，大幅减小 beta
                return 0.5;
            case STAGNATING:
                // 停滞，稍微减小 beta
                return 0.8;
            case DIVERGING:
                // 发散，大幅减小 beta
                return 0.3;
            case STABLE:
                // 稳定，保持或微调
                return 1.0;
            default:
                return 1.0;
        }
    }

    // 获取建议的历史记录大小
    int get_history_size(ConvergenceState state, int current_size, int max_size) const {
        switch(state) {
            case OSCILLATING:
                // 震荡时减少历史记录
                return std::max(2, current_size / 2);
            case DIVERGING:
                // 发散时大幅减少历史记录
                return std::max(2, 3);
            case STAGNATING:
                // 停滞时增加历史记录
                return std::min(max_size, current_size + 2);
            default:
                return current_size;
        }
    }

    // 分析残差趋势
    ConvergenceState analyze(const std::vector<double>& recent_residuals) {
        if (recent_residuals.size() < 3) return CONVERGING;

        double avg_decrease = 0.0;
        int oscillation_count = 0;
        double max_residual = recent_residuals[0];
        double min_residual = recent_residuals[0];

        for (size_t i = 1; i < recent_residuals.size(); i++) {
            double change = recent_residuals[i] - recent_residuals[i-1];
            avg_decrease += change;

            max_residual = std::max(max_residual, recent_residuals[i]);
            min_residual = std::min(min_residual, recent_residuals[i]);

            // 检测震荡：符号交替变化
            if (i >= 2 && (change * (recent_residuals[i-1] - recent_residuals[i-2])) < 0) {
                oscillation_count++;
            }
        }
        avg_decrease /= (recent_residuals.size() - 1);

        double oscillation_ratio = static_cast<double>(oscillation_count) / (recent_residuals.size() - 2);
        double relative_change = (max_residual - min_residual) / (min_residual + 1e-10);

        // 发散检测
        if (avg_decrease > 0 && avg_decrease > min_residual * 0.1) {
            return DIVERGING;
        }

        // 震荡检测
        if (oscillation_ratio > 0.6 || relative_change > 0.5) {
            oscillation_counter_++;
            if (oscillation_counter_ >= 2) {
                oscillation_counter_ = 0;
                return OSCILLATING;
            }
        } else {
            oscillation_counter_ = 0;
        }

        // 停滞检测
        if (std::abs(avg_decrease) < convergence_threshold_) {
            stagnation_counter_++;
            if (stagnation_counter_ >= 3) {
                stagnation_counter_ = 0;
                return STAGNATING;
            }
        } else {
            stagnation_counter_ = 0;
        }

        // 稳定检测
        if (relative_change < 0.05 && avg_decrease < 0) {
            return STABLE;
        }

        // 默认收敛中
        return CONVERGING;
    }

    // 重置计数器
    void reset() {
        stagnation_counter_ = 0;
        oscillation_counter_ = 0;
    }
};
// 构造函数
PulayMixer::PulayMixer(int max_history, double mixing_beta)
    : max_history_(max_history), current_step_(0), mixing_beta_(mixing_beta),
      initialized_(false), input_history_(), residual_history_(),
      nrows_(0), ncols_(0),
      adaptive_enabled_(true),
      beta_min_(0.02),
      beta_max_(0.5),
      residual_history_norms_(),
      last_beta_adjustment_step_(0) {}

// 初始化混合器
void PulayMixer::initialize(const matrix& initial_guess) {
    nrows_ = initial_guess.nr;
    ncols_ = initial_guess.nc;
    input_history_.clear();
    residual_history_.clear();
    
    // 存储初始猜测
    input_history_.push_back(initial_guess);
    initialized_ = true;
    current_step_ = 0;

    std::cout << "[PulayMixer] Initialized with matrix of size " 
              << nrows_ << "x" << ncols_ << std::endl;
}

// 执行 Pulay 混合
matrix PulayMixer::mix(const matrix& current_output) {
    return mix(current_output, -1.0);
}

// Overload with eigenvalue-change feedback (eV; pass <0 to disable)
matrix PulayMixer::mix(const matrix& current_output, double eigenvalue_change_ev) {
    if (!initialized_) {
        throw std::runtime_error("[PulayMixer] Not initialized. Call initialize() first.");
    }

    if (current_output.nr != nrows_ || current_output.nc != ncols_) {
        throw std::runtime_error("[PulayMixer] Matrix dimensions do not match initialization.");
    }

    current_step_++;


    // Optional: track eigenvalue-diff (in eV) to guide damping
    if (eigenvalue_change_ev >= 0.0) {
        eigenvalue_change_history_.push_back(eigenvalue_change_ev);
        if (eigenvalue_change_history_.size() > 20) {
            eigenvalue_change_history_.pop_front();
        }

        // If eigenvalues are still changing a lot, keep beta conservative.
        double beta_target = mixing_beta_;
        if (eigenvalue_change_ev > 0.5) beta_target = std::min(mixing_beta_, 0.06);
        else if (eigenvalue_change_ev > 0.2) beta_target = std::min(mixing_beta_, 0.08);
        else if (eigenvalue_change_ev > 0.05) beta_target = std::min(mixing_beta_, 0.12);
        else if (eigenvalue_change_ev < 0.01) beta_target = std::min(beta_max_, std::max(mixing_beta_, 0.12));

        if (beta_target != mixing_beta_) {
            set_mixing_beta(std::max(beta_min_, std::min(beta_max_, beta_target)));
        }
    }

    // 计算当前残差
    matrix current_residual = current_output - input_history_.back();
    double current_residual_norm = std::sqrt(matrix_inner_product(current_residual, current_residual));

    // 保存残差历史用于自适应调整
    residual_history_norms_.push_back(current_residual_norm);
    if (residual_history_norms_.size() > 20) {
        residual_history_norms_.erase(residual_history_norms_.begin());
    }

    bool residual_increased = false;
    double previous_residual_norm = 0.0;
    double residual_change_ratio = 0.0;

    // 检查残差是否增长
    if (current_step_ > 1) {
        previous_residual_norm = residual_history_norms_[residual_history_norms_.size() - 2];

        // 计算残差变化比例
        if (previous_residual_norm > 1e-12) {
            residual_change_ratio = (current_residual_norm - previous_residual_norm) / previous_residual_norm;
        }

        // 如果残差增长超过15%，标记为残差增长
        if (current_residual_norm > previous_residual_norm * 1.30) {
            residual_increased = true;
            std::cout << "[PulayMixer] Warning: Residual increased from " << previous_residual_norm
                      << " to " << current_residual_norm << " (" << (current_residual_norm/previous_residual_norm-1)*100
                      << "% increase)" << std::endl;
        }
    }

    // 将当前残差添加到历史记录中（无论是否增长）
    residual_history_.push_back(current_residual);

    // 历史记录管理
    if (residual_history_.size() > max_history_) {
        residual_history_.erase(residual_history_.begin());
        input_history_.erase(input_history_.begin());
    }

    int history_size = residual_history_.size();
    // Force linear mixing for early steps to avoid catastrophic divergence
    const int MIN_PULAY_STEP = 6;  // Linear mixing for first 5 steps
    bool use_linear_mixing = (history_size <= 1) || (current_step_ < MIN_PULAY_STEP);
    matrix alpha;

    // === 自适应参数调整 ===
    if (adaptive_enabled_ && residual_history_norms_.size() >= 5 &&
        (current_step_ - last_beta_adjustment_step_) >= 2) {

        ConvergenceMonitor monitor(5, 1e-6);

        std::vector<double> recent_residuals(
            residual_history_norms_.end() - std::min(5, (int)residual_history_norms_.size()),
            residual_history_norms_.end()
        );

        ConvergenceMonitor::ConvergenceState state = monitor.analyze(recent_residuals);

        std::cout << "[PulayMixer] Convergence state: " << monitor.get_state_name(state)
                  << " at step " << current_step_ << std::endl;

        // 根据收敛状态调整参数
        if (state != ConvergenceMonitor::CONVERGING) {
            double beta_factor = monitor.get_beta_factor(state, current_residual_norm);
            double new_beta = std::max(beta_min_, std::min(beta_max_, mixing_beta_ * beta_factor));

            if (new_beta != mixing_beta_ && current_step_ > 3) {
                std::cout << "[PulayMixer] Adaptive beta adjustment: " << mixing_beta_
                          << " -> " << new_beta << " (factor=" << beta_factor << ")" << std::endl;
                set_mixing_beta(new_beta);
                last_beta_adjustment_step_ = current_step_;
            }
        }
    }

    // 如果残差增长，强制使用线性混合并调整参数
    if (residual_increased) {
        use_linear_mixing = true;

        // 减小beta值
        double current_beta = get_mixing_beta();
        double new_beta = std::max(beta_min_, current_beta * 0.6);
        if (new_beta < current_beta) {
            set_mixing_beta(new_beta);
            std::cout << "[PulayMixer] Emergency beta reduction: " << current_beta
                      << " -> " << new_beta << " due to residual increase" << std::endl;
        }

        // 清除部分历史记录，保留最近2个
        while (residual_history_.size() > 2) {
            residual_history_.erase(residual_history_.begin());
            input_history_.erase(input_history_.begin());
        }
        history_size = residual_history_.size();
    }

    // 尝试Pulay混合
    if (!use_linear_mixing) {
        try {
            // Pulay混合：构建并求解线性方程组
            matrix B(history_size + 1, history_size + 1, true);  // 内积矩阵（带约束）
            matrix rhs(history_size + 1, 1, true);               // 右端项

            // 构建内积矩阵B
            double residual_norm = current_residual_norm;
            double regularization = 1.0e-5;

            // 根据残差大小和历史记录数量动态调整正则化参数
            if (residual_norm > 1.0) {
                regularization = 1.0e-4 * (1.0 + history_size * 0.1);
            } else if (residual_norm < 1.0e-3) {
                regularization = 1.0e-7;
            } else {
                regularization = 1.0e-6 * (1.0 + history_size * 0.05);
            }

            for (int i = 0; i < history_size; i++) {
                for (int j = i; j < history_size; j++) {
                    double inner_prod = matrix_inner_product(residual_history_[i], residual_history_[j]);

                    // 添加对角正则化
                    if (i == j) {
                        inner_prod += regularization;
                    }

                    B(i, j) = inner_prod;
                    B(j, i) = inner_prod;  // 对称矩阵
                }
                B(i, history_size) = -1.0;
                B(history_size, i) = -1.0;
            }
            B(history_size, history_size) = 0.0;  // 约束位置的0

            // 构建右端项 [0, 0, ..., 0, -1]^T
            rhs(history_size, 0) = -1.0;

            // 求解线性方程组 B * alpha = rhs
            alpha = solve_linear_system(B, rhs);

            // Sanity check DIIS coefficients to avoid catastrophic steps
            bool bad_alpha = false;
            for (int i = 0; i < history_size; i++) {
                double ci = alpha(i, 0);
                if (!std::isfinite(ci) || std::abs(ci) > 5.0) {
                    bad_alpha = true;
                    break;
                }
            }
            if (bad_alpha) {
                throw std::runtime_error("[PulayMixer] Unstable Pulay coefficients");
            }

        } catch (const std::exception& e) {
            std::cerr << "[PulayMixer] Warning: Pulay mixing failed (" << e.what()
                      << "). Falling back to linear mixing with smaller beta." << std::endl;

            // 混合失败时，减小beta值
            double current_beta = get_mixing_beta();
            set_mixing_beta(std::max(beta_min_, current_beta * 0.4));

            // 仅保留最近的历史记录
            while ((int)residual_history_.size() > 2) {
                residual_history_.erase(residual_history_.begin());
                input_history_.erase(input_history_.begin());
            }
            history_size = (int)residual_history_.size();
            use_linear_mixing = true;
        }
    }

    // 输出调试信息
    if (use_linear_mixing) {
        std::cout << "[PulayMixer] Linear mixing with beta=" << mixing_beta_
                  << ", residual norm=" << current_residual_norm
                  << ", change=" << (residual_change_ratio*100) << "%" << std::endl;
    } else {
        std::cout << "[PulayMixer] Pulay mixing at step " << current_step_
                  << ", history_size=" << history_size
                  << ", beta=" << mixing_beta_
                  << ", residual norm=" << current_residual_norm
                  << ", change=" << (residual_change_ratio*100) << "%" << std::endl;
    }

    matrix new_input;
    if (use_linear_mixing) {
        // 简单线性混合 (Linear Mixing)
        new_input = input_history_.back() + mixing_beta_ * current_residual;
    } else {
        // Pulay 混合：计算新的输入矩阵
        new_input = matrix(nrows_, ncols_, true);
        for (int i = 0; i < history_size; i++) {
            matrix term = input_history_[i] + mixing_beta_ * residual_history_[i];
            new_input += alpha(i, 0) * term;
        }
    }


    // Trust-region limiter for Pulay steps: prevent overly large extrapolation
    if (!use_linear_mixing) {
        matrix delta = new_input - input_history_.back();
        double delta_norm = std::sqrt(matrix_inner_product(delta, delta));
        // limit step size relative to current residual norm
        const double step_factor = 0.8;  // <=1 keeps updates conservative
        double max_delta = step_factor * current_residual_norm;
        if (delta_norm > max_delta && delta_norm > 1e-14) {
            double scale = max_delta / delta_norm;
            new_input = input_history_.back() + scale * delta;
            std::cout << "[PulayMixer] Step limited: scale=" << scale
                      << ", delta_norm=" << delta_norm
                      << ", max_delta=" << max_delta << std::endl;
        }
    }

    // Keep histories consistent. Residual r_n corresponds to input x_n (input_history_.back() before update).
    // Even if residual increased, we must still advance the input for the next iteration; otherwise
    // residual_history_ can grow ahead of input_history_ and cause out-of-bounds access.
    // Advance input history for next iteration.
    // Note: residual_history_ already includes current residual for this step; input_history_ is updated once here.
    input_history_.push_back(new_input);

    std::cout << "[PulayMixer] Performed " << (use_linear_mixing ? "linear" : "Pulay")
              << " mixing at step " << current_step_ << "." << std::endl;

    return new_input;
}

// 获取当前历史记录大小
int PulayMixer::get_history_size() const {
    return residual_history_.size();
}

// 获取当前迭代步数
int PulayMixer::get_current_step() const {
    return current_step_;
}

// 重置混合器
void PulayMixer::reset() {
    initialized_ = false;
    input_history_.clear();
    residual_history_.clear();
    nrows_ = 0;
    ncols_ = 0;
    current_step_ = 0;

    std::cout << "[PulayMixer] Reset mixer." << std::endl;
}

// 设置混合参数
void PulayMixer::set_mixing_beta(double beta) {
    mixing_beta_ = beta;
}

// 获取混合参数
double PulayMixer::get_mixing_beta() const {
    return mixing_beta_;
}

// 私有成员函数实现

// 计算两个矩阵的内积（视为向量的点积）
double PulayMixer::matrix_inner_product(const matrix& A, const matrix& B) {
    if (A.nr != B.nr || A.nc != B.nc) {
        throw std::runtime_error("[PulayMixer] Matrix dimensions must match for inner product.");
    }

    double result = 0.0;
    for (int i = 0; i < A.size; i++) {
        result += A.c[i] * B.c[i];
    }
    return result;
}

// 求解线性方程组 Ax = b
matrix PulayMixer::solve_linear_system(const matrix& A, const matrix& b) {
    if (A.nr != A.nc) {
        throw std::runtime_error("[PulayMixer] Matrix A must be square for linear system solving.");
    }
    if (b.nc != 1) {
        throw std::runtime_error("[PulayMixer] Right-hand side must be a column vector.");
    }
    if (A.nr != b.nr) {
        throw std::runtime_error("[PulayMixer] Matrix A and vector b must have compatible dimensions.");
    }

    int n = A.nr;
    matrix A_copy = A;  // 工作副本
    matrix x = b;       // 解向量

    // 使用LU分解求解 (LAPACKE 接口)
    int* ipiv = new int[n];
    int info;

    // LAPACKE_dgetrf 参数: matrix_layout (101=row-major), m, n, a, lda, ipiv
    info = LAPACKE_dgetrf(101, n, n, A_copy.c, n, ipiv);
    if (info != 0) {
        delete[] ipiv;
        throw std::runtime_error("[PulayMixer] LU factorization failed.");
    }

    // LAPACKE_dgetrs 参数: matrix_layout, trans, n, nrhs, a, lda, ipiv, b, ldb (row-major: ldb=nrhs)
    char trans = 'N';
    int nrhs = 1;
    info = LAPACKE_dgetrs(101, trans, n, nrhs, A_copy.c, n, ipiv, x.c, nrhs);

    delete[] ipiv;

    if (info != 0) {
        throw std::runtime_error("[PulayMixer] Linear system solving failed.");
    }

    return x;
}