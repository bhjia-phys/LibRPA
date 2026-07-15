#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../qsgw/hartree_kernel.h"

#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

using librpa_int::ComplexMatrix;
using librpa_int::cplxdb;
using librpa_int::qsgw::HartreeCkMap;
using librpa_int::qsgw::HartreeDkMap;
using librpa_int::qsgw::HartreeKNormalization;
using librpa_int::qsgw::HartreeVqMap;
using librpa_int::qsgw::contract_hartree_full_grid;

namespace
{

using AtomSizes = std::map<int, int>;

void assert_close(const cplxdb actual, const cplxdb expected,
                  const double tolerance = 1.0e-12)
{
    assert(std::abs(actual - expected) < tolerance);
}

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

ComplexMatrix make_matrix(const int rows, const int columns,
                          const std::vector<cplxdb>& values)
{
    assert(static_cast<int>(values.size()) == rows * columns);
    ComplexMatrix result(rows, columns);
    for (int row = 0; row < rows; ++row)
    {
        for (int column = 0; column < columns; ++column)
        {
            result(row, column) = values[row * columns + column];
        }
    }
    return result;
}

HartreeDkMap direct_oracle(
    const HartreeCkMap& c_k, const HartreeVqMap& v_q0,
    const HartreeDkMap& weighted_density_k, const AtomSizes& n_ao,
    const AtomSizes& n_aux, const std::vector<int>& kpoints,
    const HartreeKNormalization normalization)
{
    std::map<int, std::vector<cplxdb>> m;
    for (const auto& [atom, size] : n_aux)
    {
        m[atom].assign(size, 0.0);
    }

    for (const auto& [v, n_v] : n_ao)
    {
        for (const auto& [u, n_u] : n_ao)
        {
            for (const int k : kpoints)
            {
                const auto& d_vu = weighted_density_k.at(v).at(u).at(k);
                const auto& c_uv = c_k.at(u).at(v).at(k);
                const auto& c_vu = c_k.at(v).at(u).at(k);
                for (int iv = 0; iv < n_v; ++iv)
                {
                    for (int iu = 0; iu < n_u; ++iu)
                    {
                        const auto density = d_vu(iv, iu);
                        for (int a = 0; a < n_aux.at(u); ++a)
                        {
                            m[u][a] += c_uv(a, iu * n_v + iv) * density;
                        }
                        for (int a = 0; a < n_aux.at(v); ++a)
                        {
                            m[v][a] += std::conj(c_vu(a, iv * n_u + iu)) * density;
                        }
                    }
                }
            }
        }
    }

    std::map<int, std::vector<cplxdb>> n;
    for (const auto& [mu, n_mu] : n_aux)
    {
        n[mu].assign(n_mu, 0.0);
        for (const auto& [nu, n_nu] : n_aux)
        {
            const auto& v_mu_nu = v_q0.at(mu).at(nu);
            for (int a_mu = 0; a_mu < n_mu; ++a_mu)
            {
                for (int a_nu = 0; a_nu < n_nu; ++a_nu)
                {
                    n[mu][a_mu] += v_mu_nu(a_mu, a_nu) * m[nu][a_nu];
                }
            }
        }
        if (normalization == HartreeKNormalization::legacy_extra_inverse_nk)
        {
            for (auto& value : n[mu])
            {
                value /= static_cast<double>(kpoints.size());
            }
        }
    }

    HartreeDkMap result;
    for (const auto& [s, n_s] : n_ao)
    {
        for (const auto& [t, n_t] : n_ao)
        {
            for (const int k : kpoints)
            {
                auto& h_st = result[s][t][k];
                h_st = ComplexMatrix(n_s, n_t);
                h_st.zero_out();
                const auto& c_st = c_k.at(s).at(t).at(k);
                const auto& c_ts = c_k.at(t).at(s).at(k);
                for (int is = 0; is < n_s; ++is)
                {
                    for (int it = 0; it < n_t; ++it)
                    {
                        for (int a = 0; a < n_aux.at(s); ++a)
                        {
                            h_st(is, it) += c_st(a, is * n_t + it) * n[s][a];
                        }
                        for (int a = 0; a < n_aux.at(t); ++a)
                        {
                            h_st(is, it) +=
                                std::conj(c_ts(a, it * n_s + is)) * n[t][a];
                        }
                    }
                }
            }
        }
    }
    return result;
}

void assert_same(const HartreeDkMap& actual, const HartreeDkMap& expected)
{
    assert(actual.size() == expected.size());
    for (const auto& [i, by_j] : expected)
    {
        for (const auto& [j, by_k] : by_j)
        {
            for (const auto& [k, matrix] : by_k)
            {
                const auto& got = actual.at(i).at(j).at(k);
                assert(got.nr == matrix.nr);
                assert(got.nc == matrix.nc);
                for (int row = 0; row < matrix.nr; ++row)
                {
                    for (int column = 0; column < matrix.nc; ++column)
                    {
                        assert_close(got(row, column), matrix(row, column));
                    }
                }
            }
        }
    }
}

void test_weighted_density_has_no_second_inverse_nk()
{
    const AtomSizes n_ao{{0, 1}};
    const AtomSizes n_aux{{0, 1}};
    const std::vector<int> kpoints{0, 1};
    HartreeCkMap c_k;
    HartreeDkMap density_k;
    for (const int k : kpoints)
    {
        c_k[0][0][k] = make_matrix(1, 1, {1.0});
        density_k[0][0][k] = make_matrix(1, 1, {0.5});
    }
    HartreeVqMap v_q0;
    v_q0[0][0] = make_matrix(1, 1, {3.0});

    const auto corrected = contract_hartree_full_grid(
        c_k, v_q0, density_k, n_ao, kpoints,
        HartreeKNormalization::weighted_occupations);
    const auto legacy = contract_hartree_full_grid(
        c_k, v_q0, density_k, n_ao, kpoints,
        HartreeKNormalization::legacy_extra_inverse_nk);

    assert_close(corrected.at(0).at(0).at(0)(0, 0), 12.0);
    assert_close(corrected.at(0).at(0).at(1)(0, 0), 12.0);
    assert_close(legacy.at(0).at(0).at(0)(0, 0), 6.0);
    assert_close(legacy.at(0).at(0).at(1)(0, 0), 6.0);
}

void test_complex_two_atom_contraction_matches_direct_oracle()
{
    const AtomSizes n_ao{{0, 2}, {1, 1}};
    const AtomSizes n_aux{{0, 1}, {1, 2}};
    const std::vector<int> kpoints{0};

    HartreeCkMap c_k;
    c_k[0][0][0] = make_matrix(
        1, 4, {{1.0, 0.2}, {0.5, -0.3}, {-0.2, 0.1}, {0.7, 0.0}});
    c_k[0][1][0] = make_matrix(1, 2, {{0.4, 0.1}, {-0.3, 0.5}});
    c_k[1][0][0] = make_matrix(
        2, 2, {{0.2, -0.4}, {0.6, 0.3}, {-0.1, 0.2}, {0.8, -0.1}});
    c_k[1][1][0] = make_matrix(2, 1, {{0.9, 0.2}, {-0.4, 0.6}});

    HartreeDkMap density_k;
    density_k[0][0][0] = make_matrix(
        2, 2, {{0.8, 0.0}, {0.1, 0.2}, {0.1, -0.2}, {0.4, 0.0}});
    density_k[0][1][0] = make_matrix(2, 1, {{0.2, 0.1}, {-0.1, 0.3}});
    density_k[1][0][0] = make_matrix(1, 2, {{0.2, -0.1}, {-0.1, -0.3}});
    density_k[1][1][0] = make_matrix(1, 1, {{0.6, 0.0}});

    HartreeVqMap v_q0;
    v_q0[0][0] = make_matrix(1, 1, {2.0});
    v_q0[0][1] = make_matrix(1, 2, {{0.3, 0.1}, {-0.2, 0.4}});
    v_q0[1][0] = make_matrix(2, 1, {{0.3, -0.1}, {-0.2, -0.4}});
    v_q0[1][1] = make_matrix(
        2, 2, {{1.5, 0.0}, {0.2, 0.1}, {0.2, -0.1}, {1.1, 0.0}});

    const auto expected = direct_oracle(
        c_k, v_q0, density_k, n_ao, n_aux, kpoints,
        HartreeKNormalization::weighted_occupations);
    const auto actual = contract_hartree_full_grid(
        c_k, v_q0, density_k, n_ao, kpoints,
        HartreeKNormalization::weighted_occupations);
    assert_same(actual, expected);
}

void test_zero_density_delta_produces_exactly_zero_hartree()
{
    const AtomSizes n_ao{{0, 1}};
    const std::vector<int> kpoints{0, 1};
    HartreeCkMap c_k;
    HartreeDkMap density_delta_k;
    for (const int k : kpoints)
    {
        c_k[0][0][k] = make_matrix(1, 1, {{0.7, 0.4}});
        density_delta_k[0][0][k] = make_matrix(1, 1, {0.0});
    }
    HartreeVqMap v_q0;
    v_q0[0][0] = make_matrix(1, 1, {1.0e12});

    const auto result = contract_hartree_full_grid(
        c_k, v_q0, density_delta_k, n_ao, kpoints,
        HartreeKNormalization::weighted_occupations);
    assert_close(result.at(0).at(0).at(0)(0, 0), 0.0);
    assert_close(result.at(0).at(0).at(1)(0, 0), 0.0);
}

void test_invalid_shapes_and_nonfinite_data_are_rejected()
{
    const AtomSizes n_ao{{0, 1}};
    const std::vector<int> kpoints{0};
    HartreeCkMap c_k;
    c_k[0][0][0] = make_matrix(1, 1, {1.0});
    HartreeDkMap density_k;
    density_k[0][0][0] = make_matrix(1, 1, {1.0});
    HartreeVqMap v_q0;
    v_q0[0][0] = make_matrix(1, 1, {1.0});

    auto wrong_c = c_k;
    wrong_c[0][0][0] = make_matrix(1, 2, {1.0, 2.0});
    assert_throws([&] {
        contract_hartree_full_grid(
            wrong_c, v_q0, density_k, n_ao, kpoints,
            HartreeKNormalization::weighted_occupations);
    });

    HartreeDkMap nonfinite_density;
    nonfinite_density[0][0][0] = density_k.at(0).at(0).at(0);
    nonfinite_density[0][0][0](0, 0) =
        std::numeric_limits<double>::quiet_NaN();
    assert(std::isfinite(density_k.at(0).at(0).at(0)(0, 0).real()));
    assert_throws([&] {
        contract_hartree_full_grid(
            c_k, v_q0, nonfinite_density, n_ao, kpoints,
            HartreeKNormalization::weighted_occupations);
    });

    assert_throws([&] {
        contract_hartree_full_grid(
            c_k, v_q0, density_k, n_ao, {},
            HartreeKNormalization::weighted_occupations);
    });
}

} // namespace

int main()
{
    test_weighted_density_has_no_second_inverse_nk();
    test_complex_two_atom_contraction_matches_direct_oracle();
    test_zero_density_delta_produces_exactly_zero_hartree();
    test_invalid_shapes_and_nonfinite_data_are_rejected();
    std::cout << "test_qsgw_hartree_kernel: all tests passed\n";
    return 0;
}
