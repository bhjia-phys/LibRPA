#ifdef NDEBUG
#undef NDEBUG
#endif

#include <array>
#include <cassert>
#include <complex>
#include <iostream>
#include <vector>

#include "../api/compute_helper.h"
#include "../qsgw/band_bvk_remap.h"
#include "../utils/constants.h"

using librpa_int::atom_t;
using librpa_int::Atoms;
using librpa_int::Matrix3;
using librpa_int::PeriodicBoundaryData;
using librpa_int::Vector3;
using librpa_int::Vector3_Order;
using librpa_int::qsgw::build_legacy_band_bvk_remap;

namespace
{

using AtomMatrix = std::array<std::complex<double>, 4>;

AtomMatrix interpolate_displaced_atom(const std::vector<Vector3_Order<int>>& Rs,
                                      const std::vector<AtomMatrix>& cells,
                                      const librpa_int::AtomPairBvKRemap<atom_t>& remap,
                                      const Vector3_Order<double>& k)
{
    AtomMatrix result{};
    for (std::size_t ir = 0; ir != Rs.size(); ++ir)
    {
        const auto* images = remap.find_R_bvk({atom_t{1}, atom_t{1}}, Rs[ir]);
        std::complex<double> phase{};
        if (images == nullptr)
            phase = std::polar(1.0, librpa_int::TWO_PI * (k * Rs[ir]));
        else
        {
            for (const auto& image : *images)
                phase += std::polar(1.0, librpa_int::TWO_PI * (k * image));
            phase /= static_cast<double>(images->size());
        }
        for (std::size_t e = 0; e != result.size(); ++e) result[e] += phase * cells[ir][e];
    }
    return result;
}

void test_displaced_atom_preserves_adjoint_and_mesh()
{
    constexpr double a = 5.1315533365962613;
    const Matrix3 lattice(0.0, a, a, a, 0.0, a, a, a, 0.0);

    PeriodicBoundaryData pbc;
    pbc.set_latvec({0.0, a, a, a, 0.0, a, a, a, 0.0});
    pbc.set_period(4, 4, 4);

    Atoms atoms;
    atoms.set({14, 14},
              {Vector3<double>{0.0, 0.0, 0.0}, Vector3<double>{0.5 * a, 0.5 * a, 0.5 * a}},
              lattice);

    std::vector<AtomMatrix> cells(pbc.Rlist.size());
    const Vector3_Order<int> corner{1, 1, 1};
    const auto positive = static_cast<std::size_t>(pbc.get_R_index(corner));
    const auto negative = static_cast<std::size_t>(pbc.get_R_index(-corner));
    cells[positive] = AtomMatrix{{{0.3, 0.1}, {0.7, 0.2}, {-0.4, 0.6}, {0.1, -0.2}}};
    for (std::size_t i = 0; i != 2; ++i)
        for (std::size_t j = 0; j != 2; ++j)
            cells[negative][2 * i + j] = std::conj(cells[positive][2 * j + i]);
    const auto boundary = static_cast<std::size_t>(pbc.get_R_index(Vector3_Order<int>{-2, 0, 0}));
    cells[boundary] = AtomMatrix{{{0.8, 0.0}, {0.25, 0.17}, {0.25, -0.17}, {-0.3, 0.0}}};

    const std::array<Vector3_Order<double>, 2> off_mesh{Vector3_Order<double>{0.173, 0.271, -0.119},
                                                        Vector3_Order<double>{-0.31, 0.07, 0.43}};
    const auto mesh = librpa_int::build_uniform_kmesh_frac(pbc.period);
    const librpa_int::AtomPairBvKRemap<atom_t> identity;
    for (const int mode : {0, 1})
    {
        const auto shared = librpa_int::api::build_band_bvk_remap(atoms, pbc, mode);
        const auto qsgw = build_legacy_band_bvk_remap(atoms, pbc, mode);
        for (const auto& k : mesh)
        {
            const auto expected = interpolate_displaced_atom(pbc.Rlist, cells, identity, k);
            const auto actual = interpolate_displaced_atom(pbc.Rlist, cells, qsgw, k);
            for (std::size_t e = 0; e != actual.size(); ++e)
                assert(std::abs(actual[e] - expected[e]) < 5e-12);
        }
        for (const auto& k : off_mesh)
        {
            const auto actual = interpolate_displaced_atom(pbc.Rlist, cells, qsgw, k);
            const auto expected = interpolate_displaced_atom(pbc.Rlist, cells, shared, k);
            for (std::size_t i = 0; i != 2; ++i)
                for (std::size_t j = 0; j != 2; ++j)
                {
                    assert(std::abs(actual[2 * i + j] - expected[2 * i + j]) < 5e-12);
                    assert(std::abs(actual[2 * i + j] - std::conj(actual[2 * j + i])) < 5e-12);
                }
        }
    }
}

}  // namespace

int main()
{
    test_displaced_atom_preserves_adjoint_and_mesh();
    std::cout << "test_qsgw_band_bvk_remap: all tests passed\n";
    return 0;
}
