#!/usr/bin/env python3
"""Add GreenX AC comparison code to task_qsgw.cpp"""
import re

# Read the file
with open('/home/elhacedor/LibRPA-develop/driver/task_qsgw.cpp', 'r') as f:
    content = f.read()

# Find the exact location and replace
# We need to find the "data足够大且非全零，使用 Padé" line and insert GreenX code after it
pattern_to_find = r'(                                        // 数据足够大且非全零，使用 Padé\).*?\s*                                    // 直接取第一个点（通常对应最低虚频或零频）或者取平均'
pattern_replacement = r'''\s*                                    // 数据足够大且非全零，使用 Padé\).*?\s*                                    // 直接取第一个点（通常对应最低虚频或零频）或者取平均''
# Find the location in content
lines = content.split('\n')
insert_line = -1
for i, line in enumerate(lines):
    if '数据足够大且非全零，使用 Padé' in line:
        insert_line = i + 1  # Next line to insert after

# Check if we found the insertion point
if insert_line == -1:
    print(f"ERROR: Could not find the 'data足够大且非全零，使用 Padé' line to insert!")
    exit(1)

# Generate the new code block
new_code = r'''                                    // GreenX AC 比较：计算 Pade、GreenX 64-bit 和 GreenX 128-bit
                                    cplxdb result_pade = {0.0, 0.0};
                                    cplxdb result_pade0 = {0.0, 0.0};
                                    cplxdb result_greenx64 = {0.0, 0.0};
                                    cplxdb result_greenx64_0 = {0.0, 0.0};
                                    cplxdb result_greenx128 = {0.0, 0.0};

                                    // 计算标准 Pade (64-bit)
                                    try {
                                        LIBRPA::AnalyContPade pade(Params::n_params_anacon, imagfreqs,
                                                                    sigc_mn);
                                        result_pade = pade.get(energy0 - efermi);
                                        result_pade0 = pade.get(0.0);
                                    } catch (...) {
                                        result_pade = sigc_mn[0];
                                        result_pade0 = sigc_mn[0];
                                    }

                                    // 计算 GreenX 64-bit Pade
                                    try {
                                        LIBRPA::AnalyContPadeGreenX pade_gx64(Params::n_params_anacon, imagfreqs, sigc_mn,
                                                                               0,  // 64-bit precision
                                                                               0,  // use_greedy = 0
                                                                           0); // symmetry = 0
                                        result_greenx64 = pade_gx64.get(energy0 - efermi);
                                        result_greenx64_0 = pade_gx64.get(0.0);
                                    } catch (...) {
                                        result_greenx64 = sigc_mn[0];
                                        result_greenx64_0 = sigc_mn[0];
                                    }

                                    // 计算 GreenX 128-bit Pade
                                    try {
                                        LIBRPA::AnalyContPadeGreenX pade_gx128(Params::n_params_anacon, imagfreqs, sigc_mn,
                                                                               1,  // 128-bit precision
                                                                               0,  // use_greedy = 0
                                                                           0); // symmetry = 0
                                        result_greenx128 = pade_gx128.get(energy0 - efermi);
                                        result_greenx128_0 = pade_gx128.get(0.0);
                                    } catch (...) {
                                        result_greenx128 = sigc_mn[0];
                                        result_greenx128_0 = sigc_mn[0];
                                    }

                                    // 根据 analycont_method 参数选择使用哪个结果
                                    std::string method_used = "";
                                    if (Params::analycont_method == "pade") {
                                        result = result_pade;
                                        result1 = result_pade0;
                                        method_used = "pade (64-bit)";
                                    } else if (Params::analycont_method == "greenx64") {
                                        result = result_greenx64;
                                        result1 = result_greenx64_0;
                                        method_used = "greenx (64-bit)";
                                    } else if (Params::analycont_method == "greenx128") {
                                        result = result_greenx128;
                                        result1 = result_greenx128_0;
                                        method_used = "greenx (128-bit)";
                                    } else {
                                        result = result_pade;
                                        result1 = result_pade0;
                                        method_used = "pade (64-bit) - comparison mode";
                                    }
'''.replace(old_block, new_code)
'''

# Write the modified file
with open('/home/elhacedor/LibRPA-develop/driver/task_qsgw.cpp', 'w') as f:
    f.write(new_code)

print("GreenX AC comparison code has been added successfully!")
