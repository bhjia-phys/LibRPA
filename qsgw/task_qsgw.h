#ifndef TASK_QSGW_H
#define TASK_QSGW_H

#include "meanfield.h"           // 包含 MeanField 类
#include "chi0.h"                // 包含 Chi0 类
#include "exx.h"                 // 包含 Exx 类
#include "gw.h"                  // 包含 GW 类
#include "ri.h"                  // 包含 RI 类
#include "analycont.h"           // 包含 AnalyCont 类
#include "coulmat.h"             // 包含 Coulmat 类
#include "qpe_solver.h"          // 包含 QPE Solver 类
#include "params.h"              // 包含参数设置
#include "profiler.h"            // 包含 Profiler
#include "dielecmodel.h"         // 包含 Dielectric Model
#include "epsilon.h"             // 包含 Epsilon 计算
#include "convert_csc.h"         // 包含 convert_csc 函数的定义
#include "Hamiltonian.h"         // 包含 Hamiltonian 函数的定义
#include "fermi_energy_occupation.h"  // 包含 Fermi 能量和占据数计算

// 声明 QSGW 计算任务的函数
void task_qsgw();

#endif // TASK_QSGW_H
