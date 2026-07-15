#include "hartree_kernel.h"

#include <cmath>
#include <set>
#include <stdexcept>
#include <string>

namespace librpa_int
{
namespace qsgw
{
namespace
{

void require_finite_matrix(const ComplexMatrix& value,
                           const std::string& label)
{
    for (int index = 0; index < value.size; ++index)
    {
        if (!std::isfinite(value.c[index].real()) ||
            !std::isfinite(value.c[index].imag()))
        {
            throw std::invalid_argument(label + " contains non-finite data");
        }
    }
}

std::map<int, int> validate_and_infer_aux_sizes(
    const HartreeCkMap& c_k,
    const HartreeVqMap& v_q0,
    const HartreeDkMap& density_k,
    const std::map<int, int>& atom_ao_sizes,
    const std::vector<int>& kpoints)
{
    if (atom_ao_sizes.empty())
    {
        throw std::invalid_argument("QSGW Hartree AO-size map is empty");
    }
    if (kpoints.empty())
    {
        throw std::invalid_argument("QSGW Hartree k-point list is empty");
    }
    const std::set<int> unique_kpoints(kpoints.begin(), kpoints.end());
    if (unique_kpoints.size() != kpoints.size())
    {
        throw std::invalid_argument(
            "QSGW Hartree k-point list contains duplicates");
    }
    for (const int kpoint : kpoints)
    {
        if (kpoint < 0)
        {
            throw std::invalid_argument(
                "QSGW Hartree k-point indices must be non-negative");
        }
    }

    if (c_k.size() != atom_ao_sizes.size() ||
        density_k.size() != atom_ao_sizes.size() ||
        v_q0.size() != atom_ao_sizes.size())
    {
        throw std::invalid_argument(
            "QSGW Hartree atom maps do not cover the AO-size map");
    }

    std::map<int, int> atom_aux_sizes;
    for (const auto& [atom_i, ao_i] : atom_ao_sizes)
    {
        if (ao_i <= 0)
        {
            throw std::invalid_argument(
                "QSGW Hartree AO sizes must be positive");
        }
        const auto c_i = c_k.find(atom_i);
        const auto density_i = density_k.find(atom_i);
        const auto v_i = v_q0.find(atom_i);
        if (c_i == c_k.end() || density_i == density_k.end() ||
            v_i == v_q0.end() ||
            c_i->second.size() != atom_ao_sizes.size() ||
            density_i->second.size() != atom_ao_sizes.size() ||
            v_i->second.size() != atom_ao_sizes.size())
        {
            throw std::invalid_argument(
                "QSGW Hartree ordered atom-pair maps are incomplete");
        }

        int aux_i = -1;
        for (const auto& [atom_j, ao_j] : atom_ao_sizes)
        {
            const auto c_j = c_i->second.find(atom_j);
            const auto density_j = density_i->second.find(atom_j);
            if (c_j == c_i->second.end() ||
                density_j == density_i->second.end() ||
                c_j->second.size() != kpoints.size() ||
                density_j->second.size() != kpoints.size())
            {
                throw std::invalid_argument(
                    "QSGW Hartree k-resolved atom-pair maps are incomplete");
            }
            for (const int kpoint : kpoints)
            {
                const auto c_matrix_it = c_j->second.find(kpoint);
                const auto density_matrix_it = density_j->second.find(kpoint);
                if (c_matrix_it == c_j->second.end() ||
                    density_matrix_it == density_j->second.end())
                {
                    throw std::invalid_argument(
                        "QSGW Hartree k-resolved atom-pair maps are incomplete");
                }
                const ComplexMatrix& c_matrix = c_matrix_it->second;
                const ComplexMatrix& density_matrix = density_matrix_it->second;
                if (aux_i < 0)
                {
                    aux_i = c_matrix.nr;
                }
                if (aux_i <= 0 || c_matrix.nr != aux_i ||
                    c_matrix.nc != ao_i * ao_j)
                {
                    throw std::invalid_argument(
                        "QSGW Hartree RI coefficient has an invalid shape");
                }
                if (density_matrix.nr != ao_i || density_matrix.nc != ao_j)
                {
                    throw std::invalid_argument(
                        "QSGW Hartree density block has an invalid shape");
                }
                require_finite_matrix(c_matrix,
                                      "QSGW Hartree RI coefficient");
                require_finite_matrix(density_matrix,
                                      "QSGW Hartree density block");
            }
        }
        atom_aux_sizes[atom_i] = aux_i;
    }

    for (const auto& [atom_i, aux_i] : atom_aux_sizes)
    {
        for (const auto& [atom_j, aux_j] : atom_aux_sizes)
        {
            const auto row = v_q0.at(atom_i).find(atom_j);
            if (row == v_q0.at(atom_i).end() ||
                row->second.nr != aux_i || row->second.nc != aux_j)
            {
                throw std::invalid_argument(
                    "QSGW Hartree bare Coulomb block has an invalid shape");
            }
            require_finite_matrix(row->second,
                                  "QSGW Hartree bare Coulomb block");
        }
    }
    return atom_aux_sizes;
}

} // namespace

HartreeDkMap contract_hartree_full_grid(
    const HartreeCkMap& c_k,
    const HartreeVqMap& v_q0,
    const HartreeDkMap& weighted_density_k,
    const std::map<int, int>& atom_ao_sizes,
    const std::vector<int>& kpoints,
    const HartreeKNormalization normalization)
{
    const auto atom_aux_sizes = validate_and_infer_aux_sizes(
        c_k, v_q0, weighted_density_k, atom_ao_sizes, kpoints);

    std::map<int, std::vector<cplxdb>> density_aux;
    for (const auto& [atom, size] : atom_aux_sizes)
    {
        density_aux[atom].assign(size, 0.0);
    }

    for (const auto& [atom_v, ao_v] : atom_ao_sizes)
    {
        for (const auto& [atom_u, ao_u] : atom_ao_sizes)
        {
            for (const int kpoint : kpoints)
            {
                const ComplexMatrix& density_vu =
                    weighted_density_k.at(atom_v).at(atom_u).at(kpoint);
                const ComplexMatrix& c_uv =
                    c_k.at(atom_u).at(atom_v).at(kpoint);
                const ComplexMatrix& c_vu =
                    c_k.at(atom_v).at(atom_u).at(kpoint);
                for (int orbital_v = 0; orbital_v < ao_v; ++orbital_v)
                {
                    for (int orbital_u = 0; orbital_u < ao_u; ++orbital_u)
                    {
                        const cplxdb density =
                            density_vu(orbital_v, orbital_u);
                        for (int auxiliary = 0;
                             auxiliary < atom_aux_sizes.at(atom_u);
                             ++auxiliary)
                        {
                            density_aux[atom_u][auxiliary] +=
                                c_uv(auxiliary,
                                     orbital_u * ao_v + orbital_v) *
                                density;
                        }
                        for (int auxiliary = 0;
                             auxiliary < atom_aux_sizes.at(atom_v);
                             ++auxiliary)
                        {
                            density_aux[atom_v][auxiliary] +=
                                std::conj(c_vu(
                                    auxiliary,
                                    orbital_v * ao_u + orbital_u)) *
                                density;
                        }
                    }
                }
            }
        }
    }

    std::map<int, std::vector<cplxdb>> potential_aux;
    for (const auto& [atom_mu, aux_mu] : atom_aux_sizes)
    {
        potential_aux[atom_mu].assign(aux_mu, 0.0);
        for (const auto& [atom_nu, aux_nu] : atom_aux_sizes)
        {
            const ComplexMatrix& coulomb = v_q0.at(atom_mu).at(atom_nu);
            for (int mu = 0; mu < aux_mu; ++mu)
            {
                for (int nu = 0; nu < aux_nu; ++nu)
                {
                    potential_aux[atom_mu][mu] +=
                        coulomb(mu, nu) * density_aux[atom_nu][nu];
                }
            }
        }
        if (normalization ==
            HartreeKNormalization::legacy_extra_inverse_nk)
        {
            for (cplxdb& value : potential_aux[atom_mu])
            {
                value /= static_cast<double>(kpoints.size());
            }
        }
        else if (normalization !=
                 HartreeKNormalization::weighted_occupations)
        {
            throw std::invalid_argument(
                "QSGW Hartree k-point normalization mode is invalid");
        }
    }

    HartreeDkMap result;
    for (const auto& [atom_s, ao_s] : atom_ao_sizes)
    {
        for (const auto& [atom_t, ao_t] : atom_ao_sizes)
        {
            for (const int kpoint : kpoints)
            {
                ComplexMatrix h_st(ao_s, ao_t);
                const ComplexMatrix& c_st =
                    c_k.at(atom_s).at(atom_t).at(kpoint);
                const ComplexMatrix& c_ts =
                    c_k.at(atom_t).at(atom_s).at(kpoint);
                for (int orbital_s = 0; orbital_s < ao_s; ++orbital_s)
                {
                    for (int orbital_t = 0; orbital_t < ao_t; ++orbital_t)
                    {
                        for (int auxiliary = 0;
                             auxiliary < atom_aux_sizes.at(atom_s);
                             ++auxiliary)
                        {
                            h_st(orbital_s, orbital_t) +=
                                c_st(auxiliary,
                                     orbital_s * ao_t + orbital_t) *
                                potential_aux[atom_s][auxiliary];
                        }
                        for (int auxiliary = 0;
                             auxiliary < atom_aux_sizes.at(atom_t);
                             ++auxiliary)
                        {
                            h_st(orbital_s, orbital_t) +=
                                std::conj(c_ts(
                                    auxiliary,
                                    orbital_t * ao_s + orbital_s)) *
                                potential_aux[atom_t][auxiliary];
                        }
                    }
                }
                result[atom_s][atom_t][kpoint] = std::move(h_st);
            }
        }
    }
    return result;
}

} // namespace qsgw
} // namespace librpa_int
