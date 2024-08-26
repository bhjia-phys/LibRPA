#include <iostream>
#include "meanfield.h"

double calculate_total_electrons(const MeanField &mf) {
    double total_electrons = 0.0;

    // 遍历所有自旋通道
    for (int is = 0; is < mf.get_n_spins(); ++is) {
        // 遍历所有k点
        for (int ik = 0; ik < mf.get_n_kpoints(); ++ik) {
            // 遍历所有能带
            for (int ib = 0; ib < mf.get_n_bands(); ++ib) {
                // 累加每个本征态的占据权重
                total_electrons += mf.get_weight()[is](ik, ib);
            }
        }
    }

    return total_electrons;
}
