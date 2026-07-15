#include "hartree_density.h"

#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

namespace librpa_int
{
namespace qsgw
{
namespace
{

constexpr double two_pi = 6.283185307179586476925286766559;

double dot(const Vector3_Order<double>& lhs,
           const Vector3_Order<int>& rhs)
{
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

cplxdb phase(const Vector3_Order<double>& kpoint,
             const Vector3_Order<int>& translation,
             const double sign)
{
    const double angle = sign * two_pi * dot(kpoint, translation);
    return {std::cos(angle), std::sin(angle)};
}

void require_finite(const ComplexMatrix& value, const std::string& label)
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

void require_complete_meanfield(const MeanField& meanfield,
                                const std::vector<Vector3_Order<double>>& kpoints,
                                const char* label)
{
    if (!meanfield.initialized() ||
        meanfield.get_n_kpoints() != static_cast<int>(kpoints.size()))
    {
        throw std::invalid_argument(
            std::string("QSGW ") + label +
            " mean field and k-point list are inconsistent");
    }
    for (const auto& kpoint : kpoints)
    {
        if (!std::isfinite(kpoint.x) || !std::isfinite(kpoint.y) ||
            !std::isfinite(kpoint.z))
        {
            throw std::invalid_argument(
                std::string("QSGW ") + label +
                " k-point list contains non-finite data");
        }
    }
    for (int spin = 0; spin < meanfield.get_n_spins(); ++spin)
    {
        for (int kpoint = 0; kpoint < meanfield.get_n_kpoints(); ++kpoint)
        {
            for (int band = 0; band < meanfield.get_n_bands(); ++band)
            {
                const double occupation =
                    meanfield.get_weight()[spin](kpoint, band);
                if (!std::isfinite(occupation))
                {
                    throw std::invalid_argument(
                        std::string("QSGW ") + label +
                        " occupation contains non-finite data");
                }
            }
            for (int spinor = 0; spinor < meanfield.get_n_spinor(); ++spinor)
            {
                const ComplexMatrix* block =
                    meanfield.find_wfc(spin, spinor, kpoint);
                if (block == nullptr ||
                    block->nr != meanfield.get_n_bands() ||
                    block->nc != meanfield.get_n_aos())
                {
                    throw std::invalid_argument(
                        std::string("QSGW ") + label +
                        " wavefunction map is incomplete or has an invalid shape");
                }
                require_finite(*block,
                               std::string("QSGW ") + label +
                                   " wavefunction");
            }
        }
    }
}

void require_same_meanfield_shape(const MeanField& lhs,
                                  const MeanField& rhs)
{
    if (lhs.get_n_spins() != rhs.get_n_spins() ||
        lhs.get_n_kpoints() != rhs.get_n_kpoints() ||
        lhs.get_n_bands() != rhs.get_n_bands() ||
        lhs.get_n_aos() != rhs.get_n_aos() ||
        lhs.get_n_spinor() != rhs.get_n_spinor())
    {
        throw std::invalid_argument(
            "QSGW live and reference density mean fields have different shapes");
    }
}

void require_unique_r_grid(const std::vector<Vector3_Order<int>>& r_grid)
{
    if (r_grid.empty())
    {
        throw std::invalid_argument("QSGW BvK real-space grid is empty");
    }
    const std::set<Vector3_Order<int>> unique(r_grid.begin(), r_grid.end());
    if (unique.size() != r_grid.size())
    {
        throw std::invalid_argument(
            "QSGW BvK real-space grid contains duplicates");
    }
}

std::map<int, int> infer_operator_atom_sizes(
    const PeriodicOperatorRMap& operator_r)
{
    if (operator_r.empty())
    {
        throw std::invalid_argument("QSGW periodic operator is empty");
    }
    std::map<int, int> atom_sizes;
    for (const auto& [atom_i, by_pair] : operator_r)
    {
        if (by_pair.empty())
        {
            throw std::invalid_argument(
                "QSGW periodic operator contains an empty atom row");
        }
        for (const auto& [pair, block] : by_pair)
        {
            const int atom_j = pair.first;
            if (block.nr <= 0 || block.nc <= 0)
            {
                throw std::invalid_argument(
                    "QSGW periodic operator contains an empty block");
            }
            const auto row_size = atom_sizes.emplace(atom_i, block.nr);
            if (!row_size.second && row_size.first->second != block.nr)
            {
                throw std::invalid_argument(
                    "QSGW periodic operator has inconsistent row block sizes");
            }
            const auto column_size = atom_sizes.emplace(atom_j, block.nc);
            if (!column_size.second && column_size.first->second != block.nc)
            {
                throw std::invalid_argument(
                    "QSGW periodic operator has inconsistent column block sizes");
            }
            require_finite(block, "QSGW periodic operator block");
        }
    }
    return atom_sizes;
}

ComplexMatrix total_density_k_unchecked(const MeanField& meanfield,
                                        const int kpoint)
{
    ComplexMatrix result(meanfield.get_n_aos(), meanfield.get_n_aos());
    for (int spin = 0; spin < meanfield.get_n_spins(); ++spin)
    {
        for (int spinor = 0; spinor < meanfield.get_n_spinor(); ++spinor)
        {
            const ComplexMatrix& wfc =
                meanfield.get_eigenvectors().at(spin).at(spinor).at(kpoint);
            for (int band = 0; band < meanfield.get_n_bands(); ++band)
            {
                const double occupation =
                    meanfield.get_weight()[spin](kpoint, band);
                for (int row = 0; row < meanfield.get_n_aos(); ++row)
                {
                    for (int column = 0; column < meanfield.get_n_aos();
                         ++column)
                    {
                        result(row, column) +=
                            occupation * wfc(band, row) *
                            std::conj(wfc(band, column));
                    }
                }
            }
        }
    }
    return result;
}

} // namespace

ComplexMatrix total_density_k(const MeanField& meanfield, const int kpoint)
{
    if (kpoint < 0 || kpoint >= meanfield.get_n_kpoints())
    {
        throw std::out_of_range("QSGW density k-point index is out of range");
    }
    const std::vector<Vector3_Order<double>> placeholder_kpoints(
        static_cast<std::size_t>(meanfield.get_n_kpoints()),
        Vector3_Order<double>{0.0, 0.0, 0.0});
    require_complete_meanfield(meanfield, placeholder_kpoints, "density");

    return total_density_k_unchecked(meanfield, kpoint);
}

DensityRMap build_total_density_rspace(
    const MeanField& meanfield,
    const std::vector<Vector3_Order<double>>& kpoints,
    const std::vector<Vector3_Order<int>>& r_grid)
{
    require_complete_meanfield(meanfield, kpoints, "density");
    require_unique_r_grid(r_grid);

    std::vector<ComplexMatrix> density_k;
    density_k.reserve(kpoints.size());
    for (int kpoint = 0; kpoint < meanfield.get_n_kpoints(); ++kpoint)
    {
        density_k.push_back(
            total_density_k_unchecked(meanfield, kpoint));
    }

    DensityRMap result;
    for (const Vector3_Order<int>& translation : r_grid)
    {
        ComplexMatrix density_r(meanfield.get_n_aos(), meanfield.get_n_aos());
        for (std::size_t kpoint = 0; kpoint < kpoints.size(); ++kpoint)
        {
            const cplxdb factor = phase(kpoints[kpoint], translation, -1.0);
            for (int row = 0; row < meanfield.get_n_aos(); ++row)
            {
                for (int column = 0; column < meanfield.get_n_aos(); ++column)
                {
                    density_r(row, column) +=
                        factor * density_k[kpoint](row, column);
                }
            }
        }
        result[translation] = std::move(density_r);
    }
    return result;
}

DensityRMap build_total_density_rspace_symmetry(
    const SymmetryContext& symmetry,
    const std::vector<SpeciesBasisLayout>& wfc_layouts,
    const MeanField& meanfield,
    const std::vector<Vector3_Order<double>>& ibz_kpoints,
    const std::vector<Vector3_Order<int>>& r_grid,
    const std::map<atom_t, std::size_t>& atom_ao_sizes)
{
    require_complete_meanfield(meanfield, ibz_kpoints, "symmetry density");
    require_unique_r_grid(r_grid);
    if (!can_restore_symmetry_kstar_meanfield(
            symmetry, wfc_layouts, meanfield, ibz_kpoints, atom_ao_sizes))
    {
        throw std::invalid_argument(
            "QSGW symmetry density requires a valid reduced k-star mean field");
    }

    const double total_density_scale =
        2.0 / static_cast<double>(meanfield.get_n_spins() *
                                  meanfield.get_n_spinor());
    DensityRMap result;
    for (const Vector3_Order<int>& translation : r_grid)
    {
        ComplexMatrix density_r(meanfield.get_n_aos(), meanfield.get_n_aos());
        for (int spin = 0; spin < meanfield.get_n_spins(); ++spin)
        {
            for (int spinor = 0; spinor < meanfield.get_n_spinor(); ++spinor)
            {
                const ComplexMatrix component =
                    get_symmetry_restored_dmat_cplx_R(
                        symmetry, wfc_layouts, meanfield, spin, spinor,
                        spinor, ibz_kpoints, translation, atom_ao_sizes);
                for (int row = 0; row < meanfield.get_n_aos(); ++row)
                {
                    for (int column = 0; column < meanfield.get_n_aos();
                         ++column)
                    {
                        density_r(row, column) +=
                            total_density_scale * component(row, column);
                    }
                }
            }
        }
        result[translation] = std::move(density_r);
    }
    return result;
}

DensityRMap build_total_density_delta_rspace(
    const MeanField& live,
    const MeanField& reference,
    const std::vector<Vector3_Order<double>>& kpoints,
    const std::vector<Vector3_Order<int>>& r_grid)
{
    require_same_meanfield_shape(live, reference);
    require_complete_meanfield(live, kpoints, "live density");
    require_complete_meanfield(reference, kpoints, "reference density");
    require_unique_r_grid(r_grid);

    if (&live == &reference)
    {
        DensityRMap zero;
        for (const Vector3_Order<int>& translation : r_grid)
        {
            zero[translation] =
                ComplexMatrix(live.get_n_aos(), live.get_n_aos());
        }
        return zero;
    }

    const DensityRMap live_density =
        build_total_density_rspace(live, kpoints, r_grid);
    const DensityRMap reference_density =
        build_total_density_rspace(reference, kpoints, r_grid);
    DensityRMap result;
    for (const Vector3_Order<int>& translation : r_grid)
    {
        ComplexMatrix delta(live.get_n_aos(), live.get_n_aos());
        for (int row = 0; row < live.get_n_aos(); ++row)
        {
            for (int column = 0; column < live.get_n_aos(); ++column)
            {
                delta(row, column) =
                    live_density.at(translation)(row, column) -
                    reference_density.at(translation)(row, column);
            }
        }
        result[translation] = std::move(delta);
    }
    return result;
}

DensityRMap build_total_density_delta_rspace_symmetry(
    const SymmetryContext& symmetry,
    const std::vector<SpeciesBasisLayout>& wfc_layouts,
    const MeanField& live,
    const MeanField& reference,
    const std::vector<Vector3_Order<double>>& ibz_kpoints,
    const std::vector<Vector3_Order<int>>& r_grid,
    const std::map<atom_t, std::size_t>& atom_ao_sizes)
{
    require_same_meanfield_shape(live, reference);
    require_complete_meanfield(live, ibz_kpoints,
                               "live symmetry density");
    require_complete_meanfield(reference, ibz_kpoints,
                               "reference symmetry density");
    require_unique_r_grid(r_grid);
    if (&live == &reference)
    {
        DensityRMap zero;
        for (const Vector3_Order<int>& translation : r_grid)
        {
            zero[translation] =
                ComplexMatrix(live.get_n_aos(), live.get_n_aos());
        }
        return zero;
    }

    const DensityRMap live_density = build_total_density_rspace_symmetry(
        symmetry, wfc_layouts, live, ibz_kpoints, r_grid, atom_ao_sizes);
    const DensityRMap reference_density = build_total_density_rspace_symmetry(
        symmetry, wfc_layouts, reference, ibz_kpoints, r_grid,
        atom_ao_sizes);
    DensityRMap result;
    for (const Vector3_Order<int>& translation : r_grid)
    {
        ComplexMatrix delta(live.get_n_aos(), live.get_n_aos());
        for (int row = 0; row < live.get_n_aos(); ++row)
        {
            for (int column = 0; column < live.get_n_aos(); ++column)
            {
                delta(row, column) =
                    live_density.at(translation)(row, column) -
                    reference_density.at(translation)(row, column);
            }
        }
        result[translation] = std::move(delta);
    }
    return result;
}

WeightedDensityKMap reconstruct_weighted_full_grid_density(
    const DensityRMap& density_r,
    const std::vector<Vector3_Order<double>>& full_kpoints)
{
    if (density_r.empty() || full_kpoints.empty() ||
        density_r.size() != full_kpoints.size())
    {
        throw std::invalid_argument(
            "QSGW density reconstruction requires equally sized non-empty R and k grids");
    }
    const int rows = density_r.begin()->second.nr;
    const int columns = density_r.begin()->second.nc;
    if (rows <= 0 || columns <= 0)
    {
        throw std::invalid_argument(
            "QSGW real-space density matrices must be non-empty");
    }
    for (const auto& [translation, value] : density_r)
    {
        (void)translation;
        if (value.nr != rows || value.nc != columns)
        {
            throw std::invalid_argument(
                "QSGW real-space density matrices have inconsistent shapes");
        }
        require_finite(value, "QSGW real-space density");
    }

    WeightedDensityKMap result;
    const double inverse_grid_size =
        1.0 / static_cast<double>(density_r.size());
    for (std::size_t kpoint = 0; kpoint < full_kpoints.size(); ++kpoint)
    {
        if (!std::isfinite(full_kpoints[kpoint].x) ||
            !std::isfinite(full_kpoints[kpoint].y) ||
            !std::isfinite(full_kpoints[kpoint].z))
        {
            throw std::invalid_argument(
                "QSGW full k grid contains non-finite data");
        }
        ComplexMatrix density_k(rows, columns);
        for (const auto& [translation, value] : density_r)
        {
            const cplxdb factor =
                inverse_grid_size *
                phase(full_kpoints[kpoint], translation, 1.0);
            for (int row = 0; row < rows; ++row)
            {
                for (int column = 0; column < columns; ++column)
                {
                    density_k(row, column) += factor * value(row, column);
                }
            }
        }
        result[static_cast<int>(kpoint)] = std::move(density_k);
    }
    return result;
}

void validate_canonical_bvk_grid(
    const Vector3_Order<int>& period,
    const std::vector<Vector3_Order<double>>& kpoints,
    const double tolerance)
{
    const std::array<int, 3> periods{period.x, period.y, period.z};
    if (period.x <= 0 || period.y <= 0 || period.z <= 0 ||
        !(tolerance > 0.0) || !std::isfinite(tolerance))
    {
        throw std::invalid_argument(
            "QSGW BvK period and tolerance must be positive");
    }
    const long long expected_size =
        static_cast<long long>(period.x) * period.y * period.z;
    if (expected_size != static_cast<long long>(kpoints.size()))
    {
        throw std::invalid_argument(
            "QSGW k grid does not contain one point per BvK cell");
    }

    const Vector3_Order<double>& reference = kpoints.front();
    const std::array<double, 3> reference_values{
        reference.x, reference.y, reference.z};
    std::set<std::array<int, 3>> grid_indices;
    for (const Vector3_Order<double>& kpoint : kpoints)
    {
        const std::array<double, 3> values{kpoint.x, kpoint.y, kpoint.z};
        std::array<int, 3> index{};
        for (int direction = 0; direction < 3; ++direction)
        {
            if (!std::isfinite(values[direction]))
            {
                throw std::invalid_argument(
                    "QSGW k grid contains non-finite data");
            }
            const double scaled_difference =
                periods[direction] *
                (values[direction] - reference_values[direction]);
            const long long nearest = std::llround(scaled_difference);
            if (std::abs(scaled_difference - nearest) >
                tolerance * periods[direction])
            {
                throw std::invalid_argument(
                    "QSGW k grid is not commensurate with the BvK period");
            }
            const long long modulo =
                ((nearest % periods[direction]) + periods[direction]) %
                periods[direction];
            index[direction] = static_cast<int>(modulo);
        }
        if (!grid_indices.insert(index).second)
        {
            throw std::invalid_argument(
                "QSGW k grid contains periodic duplicates");
        }
    }
}

SpinKMatrixMap project_periodic_operator_to_fixed_basis(
    const PeriodicOperatorRMap& operator_r,
    const MeanField& reference,
    const std::vector<Vector3_Order<double>>& kpoints)
{
    require_complete_meanfield(reference, kpoints, "projection reference");
    const std::map<int, int> atom_sizes =
        infer_operator_atom_sizes(operator_r);
    std::map<int, int> atom_offsets;
    int total_aos = 0;
    for (const auto& [atom, size] : atom_sizes)
    {
        atom_offsets[atom] = total_aos;
        total_aos += size;
    }
    if (total_aos != reference.get_n_aos())
    {
        throw std::invalid_argument(
            "QSGW periodic operator AO blocks do not match the reference mean field");
    }

    SpinKMatrixMap result;
    for (int kpoint = 0; kpoint < reference.get_n_kpoints(); ++kpoint)
    {
        ComplexMatrix operator_k(total_aos, total_aos);
        for (const auto& [atom_i, by_pair] : operator_r)
        {
            for (const auto& [pair, block] : by_pair)
            {
                const int atom_j = pair.first;
                const Vector3_Order<int>& translation = pair.second;
                const cplxdb factor =
                    phase(kpoints[static_cast<std::size_t>(kpoint)],
                          translation, 1.0);
                for (int row = 0; row < block.nr; ++row)
                {
                    for (int column = 0; column < block.nc; ++column)
                    {
                        operator_k(atom_offsets.at(atom_i) + row,
                                   atom_offsets.at(atom_j) + column) +=
                            factor * block(row, column);
                    }
                }
            }
        }

        for (int spin = 0; spin < reference.get_n_spins(); ++spin)
        {
            Matz projected(reference.get_n_bands(),
                           reference.get_n_bands(), MAJOR::ROW);
            for (int bra = 0; bra < reference.get_n_bands(); ++bra)
            {
                for (int ket = 0; ket < reference.get_n_bands(); ++ket)
                {
                    for (int spinor = 0;
                         spinor < reference.get_n_spinor(); ++spinor)
                    {
                        const ComplexMatrix& wfc =
                            reference.get_eigenvectors()
                                .at(spin)
                                .at(spinor)
                                .at(kpoint);
                        for (int row = 0; row < total_aos; ++row)
                        {
                            for (int column = 0; column < total_aos; ++column)
                            {
                                projected(bra, ket) +=
                                    std::conj(wfc(bra, row)) *
                                    operator_k(row, column) *
                                    wfc(ket, column);
                            }
                        }
                    }
                }
            }
            for (int bra = 0; bra < projected.nr(); ++bra)
            {
                for (int ket = bra; ket < projected.nc(); ++ket)
                {
                    const cplxdb value = 0.5 *
                        (projected(bra, ket) +
                         std::conj(projected(ket, bra)));
                    projected(bra, ket) = value;
                    projected(ket, bra) = std::conj(value);
                }
            }
            result[spin][kpoint] = std::move(projected);
        }
    }
    return result;
}

} // namespace qsgw
} // namespace librpa_int
