#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../qsgw/hartree_route.h"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using librpa_int::Vector3_Order;
using librpa_int::qsgw::hartree_density_requires_symmetry_restore;
using librpa_int::qsgw::select_hartree_reader_route;

namespace
{

template <typename Function>
void assert_throws(Function&& function)
{
    bool threw = false;
    try
    {
        function();
    }
    catch (const std::exception&)
    {
        threw = true;
    }
    assert(threw);
}

void test_full_grid_uses_direct_density_with_or_without_gw_symmetry()
{
    const std::vector<Vector3_Order<double>> full_grid{
        {0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}};

    assert(!hartree_density_requires_symmetry_restore(
        full_grid, full_grid, true));
    assert(!hartree_density_requires_symmetry_restore(
        full_grid, full_grid, false));
}

void test_reduced_grid_requires_enabled_symmetry_restore()
{
    const std::vector<Vector3_Order<double>> full_grid{
        {0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}};

    assert(hartree_density_requires_symmetry_restore(
        {{0.0, 0.0, 0.0}}, full_grid, true));
    assert_throws([&] {
        (void)hartree_density_requires_symmetry_restore(
            {{0.0, 0.0, 0.0}}, full_grid, false);
    });
}

void test_scf_grid_must_be_a_subset_of_the_full_bvk_grid()
{
    const std::vector<Vector3_Order<double>> full_grid{
        {0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}};

    assert_throws([&] {
        (void)hartree_density_requires_symmetry_restore(
            {{0.0, 0.0, 0.0}, {0.25, 0.0, 0.0}},
            full_grid, true);
    });
}

void test_reader_route_keeps_ri_and_coulomb_in_the_same_auxiliary_space()
{
    const auto full = select_hartree_reader_route(
        false, "Cs_data", "Cs_shrinked_data");
    assert(!full.use_shrink_basis);
    assert(full.ri_prefix == "Cs_data");

    const auto shrink = select_hartree_reader_route(
        true, "Cs_data", "Cs_shrinked_data");
    assert(shrink.use_shrink_basis);
    assert(shrink.ri_prefix == "Cs_shrinked_data");

    assert_throws([] {
        (void)select_hartree_reader_route(true, "Cs_data", "Cs_data");
    });
    assert_throws([] {
        (void)select_hartree_reader_route(false, "", "Cs_shrinked_data");
    });
}

} // namespace

int main()
{
    test_full_grid_uses_direct_density_with_or_without_gw_symmetry();
    test_reduced_grid_requires_enabled_symmetry_restore();
    test_scf_grid_must_be_a_subset_of_the_full_bvk_grid();
    test_reader_route_keeps_ri_and_coulomb_in_the_same_auxiliary_space();
    std::cout << "test_qsgw_hartree_route: all tests passed\n";
    return 0;
}
