# GreenX 无限精度解析延拓使用指南

## 概述

LibRPA 现在集成了 GreenX 库的无限精度解析延拓功能，提供比传统双精度 Pade 近似更高的数值稳定性和精度。

## GreenX 解析延拓介绍

GreenX 库提供了基于 GMP（GNU Multiple Precision Arithmetic Library）的任意精度 Pade 近似算法。

### 主要特性

1. **无限精度支持**：基于 GMP 库，支持 64位、128位、256位或任意更高精度
2. **贪心算法**：提高数值稳定性，自动选择最优的参考点顺序
3. **对称性约束**：支持 8 种不同的对称性约束
4. **Wallis 方法**：避免计算过程中的数值溢出

### API 接口

#### 创建 Pade 模型

```cpp
AnalyContPadeGreenX::AnalyContPadeGreenX(
    int n_pars,                           // Pade 参数数量
    const std::vector<cplxdb> &xs,       // 参考点数组
    const std::vector<cplxdb> &data,      // 参考函数值数组
    int precision_in = 128,               // 精度（bit），默认 128
    int use_greedy_in = 1,                // 是否使用贪心算法，默认 1（推荐）
    int symmetry_in = 0                    // 对称性，默认 0（无对称性）
);
```

#### 评估 Pade 模型

```cpp
cplxdb AnalyContPadeGreenX::get(const cplxdb &x) const;
```

#### 获取信息

```cpp
int get_precision() const;  // 获取精度
int get_n_pars() const;     // 获取参数数量
```

### 参数说明

#### 精度（precision）

- `64`：双精度（约 15-16 位有效数字）
- `128`：四倍精度（约 33-36 位有效数字，**推荐默认值**）
- `256`：八倍精度（约 66-71 位有效数字）
- 更高值：可根据需要设置，但会增加计算时间和内存使用

#### 贪心算法（use_greedy）

- `1`：启用贪心算法（**推荐**），自动选择最优的参考点顺序以提高数值稳定性
- `0`：不使用贪心算法，按原始顺序使用参考点

#### 对称性（symmetry）

- `0`：无对称性（默认）
- `1`：沿实轴镜像
- `2`：沿虚轴镜像
- `3`：沿实轴和虚轴镜像
- `4`：偶函数
- `5`：奇函数
- `6`：共轭对称 f(z) = conj(f(-z))
- `7`：反共轭对称 f(z) = -conj(f(-z))

## 编译配置

### 1. 安装 GMP 库

在使用无限精度功能之前，需要先安装 GMP 库：

#### Ubuntu/Debian:
```bash
sudo apt-get install libgmp-dev libgmpxx-dev
```

#### CentOS/RHEL:
```bash
sudo yum install gmp-devel
```

#### macOS:
```bash
brew install gmp
```

#### 从源码编译:
```bash
wget https://gmplib.org/download/gmp/gmp-6.3.0.tar.xz
tar xvf gmp-6.3.0.tar.xz
cd gmp-6.3.0
./configure --prefix=/usr/local
make
sudo make install
```

### 2. 配置和编译 LibRPA

```bash
cd /path/to/LibRPA-develop
mkdir build && cd build
cmake .. -DUSE_GREENX_API=ON
make -j
```

如果 GMP 库未找到，CMake 会显示警告，但仍会编译 LibRPA，只是无限精度功能将不可用。

### 3. 验证编译

检查编译日志中是否包含：
```
-- GreenX Analytic Continuation enabled with GMP support
```

如果没有看到此消息，说明 GMP 库未找到。

## 使用示例

### 基本用法

```cpp
#include "analycont.h"

using namespace LIBRPA;
using cplxdb = std::complex<double>;

// 准备参考数据
int n_pars = 16;
std::vector<cplxdb> x_ref(n_pars);
std::vector<cplxdb> y_ref(n_pars);

// 沿虚轴生成参考点（推荐用于格林函数解析延拓）
for (int i = 0; i < n_pars; i++)
{
    x_ref[i] = cplxdb(0.0, -1.1 + i * 0.1);  // 从 -1.1i 到 0.5i
    y_ref[i] = calculate_self_energy(x_ref[i]);  // 你的函数
}

// 创建 GreenX Pade 模型（128位精度，贪心算法）
AnalyContPadeGreenX pade(n_pars, x_ref, y_ref, 128, 1, 0);

// 在实轴上评估（带小虚偏移）
cplxdb x_query = cplxdb(0.5, 0.01);  // ω = 0.5 + i*0.01
cplxdb result = pade.get(x_query);

// 使用结果
double real_part = result.real();
double imag_part = result.imag();
```

### 用于 GW 准粒子能量计算

```cpp
#include "qpe_solver.h"
#include "analycont.h"

// 假设你有虚频自能数据
std::vector<cplxdb> omega_imag(n_freq);
std::vector<cplxdb> sigma_c_imag(n_freq);

// 使用 GreenX 创建高精度 Pade 模型
AnalyContPadeGreenX pade_sigma(n_freq, omega_imag, sigma_c_imag, 128, 1, 0);

// 求解准粒子能量
double e_mf = 1.0;      // 平均场能量
double e_fermi = 0.0;   // 费米能级
double vxc = 0.5;       // 交换相关势
double sigma_x = 0.1;   // Hartree-Fock 交换项
double e_qp;            // 准粒子能量（输出）
cplxdb sigc;            // 自能（输出）
double thres = 1e-8;    // 收敛阈值

int info = qpe_solver_pade_self_consistent(pade_sigma, e_mf, e_fermi, vxc, sigma_x, e_qp, sigc, thres);

if (info == 0)
{
    std::cout << "Quasiparticle energy: " << e_qp << std::endl;
    std::cout << "Self-energy: " << sigc << std::endl;
}
```

### 比较不同精度

```cpp
int precisions[] = {64, 128, 256};

for (int prec : precisions)
{
    AnalyContPadeGreenX pade(n_pars, x_ref, y_ref, prec, 1, 0);
    cplxdb result = pade.get(x_query);
    std::cout << "Precision " << prec << ": " << result << std::endl;
}
```

## 性能考虑

### 计算时间

- **双精度 (64 bits)**：与传统 Pade 相当
- **四倍精度 (128 bits)**：约 2-3 倍于双精度
- **八倍精度 (256 bits)**：约 5-8 倍于双精度

### 内存使用

- 内存使用与精度成正比
- 对于大多数应用，128 位精度是性能和精度的最佳平衡点

### 推荐配置

1. **初始测试**：使用 64 位精度快速验证结果
2. **生产运行**：使用 128 位精度（推荐）
3. **高精度需求**：使用 256 位或更高精度（仅在必要时）

## 与现有 `AnalyContPade` 的对比

| 特性 | `AnalyContPade` | `AnalyContPadeGreenX` |
|------|----------------|----------------------|
| 精度 | 双精度 | 可配置（64-无限） |
| 算法 | Thiele 倒数差分 | Thiele 倒数差分 + GMP |
| 贪心算法 | 不支持 | 支持（推荐） |
| 对称性约束 | 不支持 | 支持（8种） |
| 数值稳定性 | 中等 | 高 |
| 计算速度 | 快 | 中等（取决于精度） |
| 依赖 | 无 | GMP 库 |
| 适用场景 | 一般应用 | 高精度需求、困难情况 |

## 故障排除

### 编译错误：找不到 pade_mp.h

**原因**：未正确链接 GreenX 解析延拓库

**解决方案**：
1. 确保 GMP 库已安装
2. 检查 CMake 配置输出，确认找到 GMP 库
3. 清理构建目录并重新编译：
   ```bash
   rm -rf build
   mkdir build && cd build
   cmake .. -DUSE_GREENX_API=ON
   make -j
   ```

### 运行时错误：Pade model not initialized

**原因**：模型创建失败

**解决方案**：
1. 检查参考数据是否有效（非零，有限）
2. 尝试减少参数数量
3. 禁用贪心算法（设置 `use_greedy_in = 0`）

### 数值不稳定

**原因**：参考点选择不当或精度不够

**解决方案**：
1. 启用贪心算法（推荐）
2. 增加精度到 128 或 256 位
3. 检查参考点的分布（建议沿虚轴均匀分布）
4. 考虑使用对称性约束

## 示例代码

完整示例请参考：
- `examples/example_greenx_ac.cpp`：基本使用示例

## 参考资料

- GreenX 库文档：`/thirdparty/greenX-6ff8a00/GX-AnalyticContinuation/README.md`
- GMP 库：https://gmplib.org/
- Pade 近似和 Thiele 方法相关文献

## 技术支持

如有问题，请检查：
1. GMP 库是否正确安装
2. CMake 配置是否正确
3. 编译日志中的警告信息
4. 示例代码是否能正确编译和运行
