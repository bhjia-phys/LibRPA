# GreenX 无限精度解析延拓 - 快速参考

## 快速开始

### 1. 安装 GMP 库

```bash
# Ubuntu/Debian
sudo apt-get install libgmp-dev libgmpxx-dev

# CentOS/RHEL
sudo yum install gmp-devel

# macOS
brew install gmp
```

### 2. 编译

```bash
mkdir build && cd build
cmake .. -DUSE_GREENX_API=ON
make -j
```

### 3. 使用示例

```cpp
#include "analycont.h"

using namespace LIBRPA;
using cplxdb = std::complex<double>;

// 准备数据
int n_pars = 16;
std::vector<cplxdb> x_ref(n_pars);
std::vector<cplxdb> y_ref(n_pars);

// 沿虚轴生成参考点（格林函数推荐）
for (int i = 0; i < n_pars; i++) {
    x_ref[i] = cplxdb(0.0, -1.1 + i * 0.15);
    y_ref[i] = calculate_sigma_c(x_ref[i]);
}

// 创建 GreenX Pade 模型（128位精度，贪心算法）
AnalyContPadeGreenX pade(n_pars, x_ref, y_ref, 128, 1, 0);

// 评估
cplxdb omega = cplxdb(0.5, 0.01);
cplxdb sigma_c = pade.get(omega);
```

## API 快速参考

### 构造函数

```cpp
AnalyContPadeGreenX(
    int n_par,                          // 参数数量
    const std::vector<cplxdb> &xs,      // 参考点
    const std::vector<cplxdb> &data,    // 参考值
    int precision = 128,                 // 精度：64/128/256/...
    int use_greedy = 1,                 // 1=贪心算法(推荐), 0=否
    int symmetry = 0                    // 对称性：0=无
);
```

### 评估函数

```cpp
cplxdb get(const cplxdb &x) const;
```

### 信息查询

```cpp
int get_precision() const;  // 返回精度
int get_n_pars() const;     // 返回参数数量
```

## 推荐参数

### 精度选择

| 精度 | 有效数字 | 计算时间 | 适用场景 |
|------|---------|---------|---------|
| 64 | ~15-16 | 1x | 快速测试 |
| 128 | ~33-36 | 2-3x | **推荐默认** |
| 256 | ~66-71 | 5-8x | 高精度需求 |

### 贪心算法

- **推荐使用**（`use_greedy = 1`）：提高数值稳定性，自动选择最优参考点顺序
- **特殊情况禁用**（`use_greedy = 0`）：参考点已经是最优顺序时

### 对称性

- `0`：无对称性（默认，适用于大多数情况）
- `4`：偶函数（如果函数满足 f(z) = f(-z)）
- `5`：奇函数（如果函数满足 f(z) = -f(-z)）
- 其他对称性请参考完整文档

## 常见问题

### Q: 如何选择参数数量？

A: 通常 8-32 个参数足够。从 16 个开始，根据精度需求调整。

### Q: 128位精度足够吗？

A: 对于大多数 GW 计算和应用，128位精度已经足够。只有在数值不稳定时才考虑更高精度。

### Q: 如何检查 GMP 是否正确安装？

A: 编译时查看输出，应该看到：
```
-- GreenX Analytic Continuation enabled with GMP support
```

### Q: 计算太慢怎么办？

A:
1. 降低精度到 64 位
2. 减少参数数量
3. 禁用贪心算法（可能影响精度）

## 完整文档

详见：`docs/GREENX_ANALYTIC_CONTINUATION.md`

## 示例程序

- `examples/example_greenx_ac.cpp`：完整示例
- `examples/test_greenx_integration.cpp`：集成测试
