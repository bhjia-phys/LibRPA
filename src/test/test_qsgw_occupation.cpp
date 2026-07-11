#include "../qsgw/occupation.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using librpa_int::MeanField;
using librpa_int::qsgw::OccupationSettings;
using librpa_int::qsgw::update_qsgw_occupations;

namespace
{

void assert_close(const double actual, const double expected, const double tolerance = 1.0e-12)
{
    assert(std::abs(actual - expected) < tolerance);
}

double total_weight(const MeanField& meanfield)
{
    double result = 0.0;
    for (int spin = 0; spin < meanfield.get_n_spins(); ++spin)
    {
        for (int kpoint = 0; kpoint < meanfield.get_n_kpoints(); ++kpoint)
        {
            for (int band = 0; band < meanfield.get_n_bands(); ++band)
            {
                result += meanfield.get_weight()[spin](kpoint, band);
            }
        }
    }
    return result;
}

void test_global_filling_preserves_nonuniform_kpoint_weights()
{
    MeanField reference(1, 2, 2, 2, 1);
    reference.get_weight()[0].zero_out();
    reference.get_weight()[0](0, 0) = 1.5;
    reference.get_weight()[0](1, 0) = 0.5;

    MeanField live = reference;
    live.get_eigenvals()[0](0, 0) = -1.0;
    live.get_eigenvals()[0](0, 1) = 2.0;
    live.get_eigenvals()[0](1, 0) = -0.5;
    live.get_eigenvals()[0](1, 1) = 3.0;

    const std::vector<double> kpoint_weights{0.75, 0.25};
    const auto result = update_qsgw_occupations(
        live, reference, kpoint_weights, 2.0, OccupationSettings{});

    assert_close(live.get_weight()[0](0, 0), 1.5);
    assert_close(live.get_weight()[0](1, 0), 0.5);
    assert_close(live.get_weight()[0](0, 1), 0.0);
    assert_close(live.get_weight()[0](1, 1), 0.0);
    assert_close(total_weight(live), 2.0);
    assert_close(result.electron_count, 2.0);
    assert(result.chemical_potential > -0.5);
    assert(result.chemical_potential < 2.0);
    assert(!result.metallic);
}

void test_global_spin_filling_does_not_fill_one_electron_per_spin()
{
    MeanField reference(2, 1, 2, 2, 1);
    reference.get_weight()[0].zero_out();
    reference.get_weight()[1].zero_out();
    reference.get_weight()[0](0, 0) = 1.0;

    MeanField live = reference;
    live.get_eigenvals()[0](0, 0) = -1.0;
    live.get_eigenvals()[0](0, 1) = 10.0;
    live.get_eigenvals()[1](0, 0) = 0.0;
    live.get_eigenvals()[1](0, 1) = 20.0;

    const auto result = update_qsgw_occupations(
        live, reference, {1.0}, 1.0, OccupationSettings{});

    assert_close(live.get_weight()[0](0, 0), 1.0);
    assert_close(live.get_weight()[0](0, 1), 0.0);
    assert_close(live.get_weight()[1](0, 0), 0.0);
    assert_close(live.get_weight()[1](0, 1), 0.0);
    assert_close(total_weight(live), 1.0);
    assert_close(result.electron_count, 1.0);
    assert(result.chemical_potential > -1.0);
    assert(result.chemical_potential < 0.0);
}

} // namespace

int main()
{
    test_global_filling_preserves_nonuniform_kpoint_weights();
    test_global_spin_filling_does_not_fill_one_electron_per_spin();
    std::cout << "test_qsgw_occupation: all tests passed\n";
    return 0;
}
