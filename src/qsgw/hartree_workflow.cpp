#include "hartree_workflow.h"

#include "hartree_dump.h"

#include "../utils/constants.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace librpa_int
{
namespace qsgw
{
namespace
{

bool finite_complex(const cplxdb value)
{
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}

cplxdb fourier_phase(const Vector3_Order<double>& kpoint,
                     const Vector3_Order<int>& translation,
                     const double sign)
{
    const double angle = sign * TWO_PI *
                         (kpoint.x * translation.x +
                          kpoint.y * translation.y +
                          kpoint.z * translation.z);
    return {std::cos(angle), std::sin(angle)};
}

bool is_periodic_zero(const Vector3_Order<double>& qpoint,
                      const double tolerance)
{
    return std::abs(qpoint.x - std::round(qpoint.x)) <= tolerance &&
           std::abs(qpoint.y - std::round(qpoint.y)) <= tolerance &&
           std::abs(qpoint.z - std::round(qpoint.z)) <= tolerance;
}

ComplexMatrix conjugate_transpose(const ComplexMatrix& input)
{
    ComplexMatrix output(input.nc, input.nr);
    for (int row = 0; row < input.nr; ++row)
    {
        for (int column = 0; column < input.nc; ++column)
        {
            output(column, row) = std::conj(input(row, column));
        }
    }
    return output;
}

void add_weighted_operator_block(
    PeriodicOperatorRMap& result,
    const int atom_i,
    const int atom_j,
    const Vector3_Order<int>& translation,
    const ComplexMatrix& block,
    const double weight)
{
    auto& target = result[atom_i][{atom_j, translation}];
    if (target.nr == 0 && target.nc == 0)
    {
        target = ComplexMatrix(block.nr, block.nc);
    }
    if (target.nr != block.nr || target.nc != block.nc)
    {
        throw std::invalid_argument(
            "QSGW Hartree BvK remap combines incompatible blocks");
    }
    for (int row = 0; row < block.nr; ++row)
    {
        for (int column = 0; column < block.nc; ++column)
        {
            target(row, column) += weight * block(row, column);
        }
    }
}

} // namespace

HartreeStaticData build_hartree_static_data(
    const Cs_LRI& coefficients,
    const atpair_k_cplx_mat_t& coulomb,
    const AtomicBasis& wavefunction_basis,
    const AtomicBasis& auxiliary_basis,
    const PeriodicBoundaryData& pbc,
    const AtomPairBvKRemap<atom_t>& bvk_remap,
    const HartreeKNormalization normalization,
    const double zero_tolerance)
{
    validate_canonical_bvk_grid(
        pbc.period, pbc.kfrac_list_full, zero_tolerance);
    if (pbc.Rlist.size() != pbc.kfrac_list_full.size())
    {
        throw std::invalid_argument(
            "QSGW Hartree full k and real-space grids have different sizes");
    }

    const Cs_LRI materialized = materialize_hartree_coefficients(
        coefficients, wavefunction_basis, auxiliary_basis);

    HartreeStaticData result;
    result.period = pbc.period;
    result.full_kpoints = pbc.kfrac_list_full;
    result.translations = pbc.Rlist;
    result.normalization = normalization;
    result.bvk_remap = bvk_remap;
    for (int atom = 0;
         atom < static_cast<int>(wavefunction_basis.n_atoms); ++atom)
    {
        result.atom_ao_sizes[atom] =
            static_cast<int>(wavefunction_basis.get_atom_nb(atom));
    }
    result.c_k = build_hartree_c_k(
        materialized, wavefunction_basis, auxiliary_basis,
        result.full_kpoints);
    result.v_q0 = build_hartree_v_q0(
        coulomb, auxiliary_basis, zero_tolerance);
    return result;
}

Cs_LRI materialize_hartree_coefficients(
    const Cs_LRI& coefficients,
    const AtomicBasis& wavefunction_basis,
    const AtomicBasis& auxiliary_basis)
{
    if (!wavefunction_basis.initialized() || !auxiliary_basis.initialized() ||
        wavefunction_basis.n_atoms != auxiliary_basis.n_atoms)
    {
        throw std::invalid_argument(
            "QSGW Hartree coefficient bases are incomplete");
    }
    if (!coefficients.data_IJR.empty())
    {
        return coefficients;
    }
    if (!coefficients.use_libri || coefficients.data_libri.empty())
    {
        throw std::invalid_argument(
            "QSGW Hartree has no materializable RI coefficients");
    }

    Cs_LRI result;
    result.use_libri = false;
    for (const auto& [atom_i, by_atom_j_translation] :
         coefficients.data_libri)
    {
        if (atom_i < 0 || atom_i >= wavefunction_basis.n_atoms)
        {
            throw std::invalid_argument(
                "QSGW Hartree RI tensor contains an invalid atom index");
        }
        const int ao_i = static_cast<int>(wavefunction_basis[atom_i]);
        const int aux_i = static_cast<int>(auxiliary_basis[atom_i]);
        for (const auto& [atom_translation, tensor] :
             by_atom_j_translation)
        {
            const int atom_j = atom_translation.first;
            if (atom_j < 0 || atom_j >= wavefunction_basis.n_atoms)
            {
                throw std::invalid_argument(
                    "QSGW Hartree RI tensor contains an invalid atom index");
            }
            const int ao_j = static_cast<int>(wavefunction_basis[atom_j]);
            if (tensor.shape.size() != 3 ||
                tensor.shape[0] != static_cast<std::size_t>(aux_i) ||
                tensor.shape[1] != static_cast<std::size_t>(ao_i) ||
                tensor.shape[2] != static_cast<std::size_t>(ao_j) ||
                tensor.data == nullptr)
            {
                throw std::invalid_argument(
                    "QSGW Hartree RI tensor has an invalid shape");
            }
            auto matrix_ptr = std::make_shared<matrix>(
                ao_i * ao_j, aux_i, true);
            for (int orbital_i = 0; orbital_i < ao_i; ++orbital_i)
            {
                for (int orbital_j = 0; orbital_j < ao_j; ++orbital_j)
                {
                    const int orbital = orbital_i * ao_j + orbital_j;
                    for (int auxiliary = 0; auxiliary < aux_i; ++auxiliary)
                    {
                        const double value = tensor(
                            static_cast<std::size_t>(auxiliary),
                            static_cast<std::size_t>(orbital_i),
                            static_cast<std::size_t>(orbital_j));
                        if (!std::isfinite(value))
                        {
                            throw std::invalid_argument(
                                "QSGW Hartree RI tensor contains non-finite data");
                        }
                        (*matrix_ptr)(orbital, auxiliary) = value;
                    }
                }
            }
            const auto& cell = atom_translation.second;
            result.data_IJR[atom_i][atom_j]
                            [{cell[0], cell[1], cell[2]}] =
                std::move(matrix_ptr);
        }
    }
    return result;
}

HartreeCkMap build_hartree_c_k(
    const Cs_LRI& coefficients,
    const AtomicBasis& wavefunction_basis,
    const AtomicBasis& auxiliary_basis,
    const std::vector<Vector3_Order<double>>& full_kpoints)
{
    if (coefficients.data_IJR.empty())
    {
        throw std::invalid_argument(
            "QSGW Hartree requires a complete real-space RI coefficient copy");
    }
    if (!wavefunction_basis.initialized() ||
        !auxiliary_basis.initialized() ||
        wavefunction_basis.n_atoms != auxiliary_basis.n_atoms ||
        full_kpoints.empty())
    {
        throw std::invalid_argument(
            "QSGW Hartree RI Fourier input is incomplete");
    }

    HartreeCkMap result;
    for (int atom_i = 0; atom_i < wavefunction_basis.n_atoms; ++atom_i)
    {
        const int ao_i = static_cast<int>(wavefunction_basis[atom_i]);
        const int aux_i = static_cast<int>(auxiliary_basis[atom_i]);
        for (int atom_j = 0; atom_j < wavefunction_basis.n_atoms; ++atom_j)
        {
            const int ao_j = static_cast<int>(wavefunction_basis[atom_j]);
            for (std::size_t kpoint = 0; kpoint < full_kpoints.size(); ++kpoint)
            {
                result[atom_i][atom_j][static_cast<int>(kpoint)] =
                    ComplexMatrix(aux_i, ao_i * ao_j);
            }
        }
    }

    for (const auto& [atom_i, by_atom_j] : coefficients.data_IJR)
    {
        if (atom_i < 0 || atom_i >= wavefunction_basis.n_atoms)
        {
            throw std::invalid_argument(
                "QSGW Hartree RI coefficient contains an invalid atom index");
        }
        const int ao_i = static_cast<int>(wavefunction_basis[atom_i]);
        const int aux_i = static_cast<int>(auxiliary_basis[atom_i]);
        for (const auto& [atom_j, by_translation] : by_atom_j)
        {
            if (atom_j < 0 || atom_j >= wavefunction_basis.n_atoms)
            {
                throw std::invalid_argument(
                    "QSGW Hartree RI coefficient contains an invalid atom index");
            }
            const int ao_j = static_cast<int>(wavefunction_basis[atom_j]);
            for (const auto& [translation, coefficient_ptr] : by_translation)
            {
                if (!coefficient_ptr ||
                    coefficient_ptr->nr != ao_i * ao_j ||
                    coefficient_ptr->nc != aux_i)
                {
                    throw std::invalid_argument(
                        "QSGW Hartree RI coefficient has an invalid shape");
                }
                for (std::size_t kpoint = 0;
                     kpoint < full_kpoints.size(); ++kpoint)
                {
                    const cplxdb phase = fourier_phase(
                        full_kpoints[kpoint], translation, 1.0);
                    ComplexMatrix& target =
                        result[atom_i][atom_j][static_cast<int>(kpoint)];
                    for (int orbital = 0; orbital < ao_i * ao_j;
                         ++orbital)
                    {
                        for (int auxiliary = 0; auxiliary < aux_i;
                             ++auxiliary)
                        {
                            const double value =
                                (*coefficient_ptr)(orbital, auxiliary);
                            if (!std::isfinite(value))
                            {
                                throw std::invalid_argument(
                                    "QSGW Hartree RI coefficient contains non-finite data");
                            }
                            target(auxiliary, orbital) += phase * value;
                        }
                    }
                }
            }
        }
    }
    return result;
}

HartreeVqMap build_hartree_v_q0(
    const atpair_k_cplx_mat_t& coulomb,
    const AtomicBasis& auxiliary_basis,
    const double zero_tolerance)
{
    if (!auxiliary_basis.initialized() || !(zero_tolerance > 0.0) ||
        !std::isfinite(zero_tolerance))
    {
        throw std::invalid_argument(
            "QSGW Hartree q=0 Coulomb input is invalid");
    }

    HartreeVqMap input_blocks;
    for (const auto& [atom_i, by_atom_j] : coulomb)
    {
        for (const auto& [atom_j, by_qpoint] : by_atom_j)
        {
            if (atom_i < 0 || atom_i >= auxiliary_basis.n_atoms ||
                atom_j < 0 || atom_j >= auxiliary_basis.n_atoms)
            {
                throw std::invalid_argument(
                    "QSGW Hartree Coulomb contains an invalid atom index");
            }
            const ComplexMatrix* q0 = nullptr;
            for (const auto& [qpoint, matrix_ptr] : by_qpoint)
            {
                if (is_periodic_zero(qpoint, zero_tolerance))
                {
                    if (q0 != nullptr || !matrix_ptr)
                    {
                        throw std::invalid_argument(
                            "QSGW Hartree Coulomb has ambiguous q=0 data");
                    }
                    q0 = matrix_ptr.get();
                }
            }
            if (q0 == nullptr ||
                q0->nr != static_cast<int>(auxiliary_basis[atom_i]) ||
                q0->nc != static_cast<int>(auxiliary_basis[atom_j]))
            {
                throw std::invalid_argument(
                    "QSGW Hartree Coulomb q=0 block is missing or malformed");
            }
            for (int index = 0; index < q0->size; ++index)
            {
                if (!finite_complex(q0->c[index]))
                {
                    throw std::invalid_argument(
                        "QSGW Hartree Coulomb contains non-finite data");
                }
            }
            input_blocks[atom_i][atom_j] = *q0;
        }
    }

    HartreeVqMap result;
    for (int atom_i = 0; atom_i < auxiliary_basis.n_atoms; ++atom_i)
    {
        for (int atom_j = atom_i; atom_j < auxiliary_basis.n_atoms; ++atom_j)
        {
            const auto direct_atom = input_blocks.find(atom_i);
            const auto reverse_atom = input_blocks.find(atom_j);
            const ComplexMatrix* direct =
                direct_atom != input_blocks.end() &&
                        direct_atom->second.count(atom_j) != 0
                    ? &direct_atom->second.at(atom_j)
                    : nullptr;
            const ComplexMatrix* reverse =
                reverse_atom != input_blocks.end() &&
                        reverse_atom->second.count(atom_i) != 0
                    ? &reverse_atom->second.at(atom_i)
                    : nullptr;
            if (direct == nullptr && reverse == nullptr)
            {
                throw std::invalid_argument(
                    "QSGW Hartree Coulomb does not cover all ordered atom pairs");
            }

            const int rows = static_cast<int>(auxiliary_basis[atom_i]);
            const int columns = static_cast<int>(auxiliary_basis[atom_j]);
            ComplexMatrix projected(rows, columns);
            for (int row = 0; row < rows; ++row)
            {
                for (int column = 0; column < columns; ++column)
                {
                    const cplxdb direct_value = direct == nullptr
                        ? std::conj((*reverse)(column, row))
                        : (*direct)(row, column);
                    const cplxdb reverse_value = reverse == nullptr
                        ? direct_value
                        : std::conj((*reverse)(column, row));
                    projected(row, column) =
                        0.5 * (direct_value + reverse_value);
                }
            }
            result[atom_i][atom_j] = projected;
            result[atom_j][atom_i] = conjugate_transpose(projected);
        }
    }
    return result;
}

HartreeDkMap split_weighted_density_by_atom(
    const WeightedDensityKMap& density_k,
    const std::map<int, int>& atom_ao_sizes)
{
    if (density_k.empty() || atom_ao_sizes.empty())
    {
        throw std::invalid_argument(
            "QSGW Hartree density split input is empty");
    }
    std::map<int, int> offsets;
    int total_aos = 0;
    for (const auto& [atom, size] : atom_ao_sizes)
    {
        if (size <= 0)
        {
            throw std::invalid_argument(
                "QSGW Hartree atom AO sizes must be positive");
        }
        offsets[atom] = total_aos;
        total_aos += size;
    }

    HartreeDkMap result;
    for (const auto& [kpoint, density] : density_k)
    {
        if (density.nr != total_aos || density.nc != total_aos)
        {
            throw std::invalid_argument(
                "QSGW Hartree density does not match atom AO sizes");
        }
        for (const auto& [atom_i, size_i] : atom_ao_sizes)
        {
            for (const auto& [atom_j, size_j] : atom_ao_sizes)
            {
                ComplexMatrix block(size_i, size_j);
                for (int row = 0; row < size_i; ++row)
                {
                    for (int column = 0; column < size_j; ++column)
                    {
                        const cplxdb value =
                            density(offsets.at(atom_i) + row,
                                    offsets.at(atom_j) + column);
                        if (!finite_complex(value))
                        {
                            throw std::invalid_argument(
                                "QSGW Hartree density contains non-finite data");
                        }
                        block(row, column) = value;
                    }
                }
                result[atom_i][atom_j][kpoint] = std::move(block);
            }
        }
    }
    return result;
}

PeriodicOperatorRMap inverse_fourier_hartree_operator(
    const HartreeDkMap& operator_k,
    const std::vector<Vector3_Order<double>>& full_kpoints,
    const std::vector<Vector3_Order<int>>& translations,
    const AtomPairBvKRemap<atom_t>* bvk_remap)
{
    if (operator_k.empty() || full_kpoints.empty() || translations.empty() ||
        full_kpoints.size() != translations.size())
    {
        throw std::invalid_argument(
            "QSGW Hartree inverse Fourier grids are incomplete");
    }
    const double inverse_grid_size =
        1.0 / static_cast<double>(full_kpoints.size());
    PeriodicOperatorRMap result;
    for (const auto& [atom_i, by_atom_j] : operator_k)
    {
        for (const auto& [atom_j, by_kpoint] : by_atom_j)
        {
            if (by_kpoint.size() != full_kpoints.size())
            {
                throw std::invalid_argument(
                    "QSGW Hartree operator does not cover the full k grid");
            }
            const ComplexMatrix& reference = by_kpoint.begin()->second;
            if (reference.nr <= 0 || reference.nc <= 0)
            {
                throw std::invalid_argument(
                    "QSGW Hartree operator matrix is empty");
            }
            for (const Vector3_Order<int>& translation : translations)
            {
                ComplexMatrix block(reference.nr, reference.nc);
                for (std::size_t kpoint = 0;
                     kpoint < full_kpoints.size(); ++kpoint)
                {
                    const auto matrix_it =
                        by_kpoint.find(static_cast<int>(kpoint));
                    if (matrix_it == by_kpoint.end() ||
                        matrix_it->second.nr != reference.nr ||
                        matrix_it->second.nc != reference.nc)
                    {
                        throw std::invalid_argument(
                            "QSGW Hartree operator k-grid layout is inconsistent");
                    }
                    const cplxdb factor =
                        inverse_grid_size *
                        fourier_phase(full_kpoints[kpoint], translation, -1.0);
                    for (int row = 0; row < reference.nr; ++row)
                    {
                        for (int column = 0; column < reference.nc; ++column)
                        {
                            const cplxdb value =
                                matrix_it->second(row, column);
                            if (!finite_complex(value))
                            {
                                throw std::invalid_argument(
                                    "QSGW Hartree operator contains non-finite data");
                            }
                            block(row, column) += factor * value;
                        }
                    }
                }
                const auto* remapped = bvk_remap == nullptr
                                           ? nullptr
                                           : bvk_remap->find_R_bvk(
                                                 {atom_i, atom_j}, translation);
                if (remapped == nullptr || remapped->empty())
                {
                    add_weighted_operator_block(
                        result, atom_i, atom_j, translation, block, 1.0);
                }
                else
                {
                    const double weight =
                        1.0 / static_cast<double>(remapped->size());
                    for (const Vector3_Order<int>& target : *remapped)
                    {
                        add_weighted_operator_block(
                            result, atom_i, atom_j, target, block, weight);
                    }
                }
            }
        }
    }
    return result;
}

PeriodicOperatorRMap build_hartree_delta_periodic_operator(
    const HartreeStaticData& static_data,
    const MeanField& live,
    const MeanField& reference,
    const std::vector<Vector3_Order<double>>& scf_kpoints,
    const HartreeSymmetryData* symmetry_data)
{
    validate_canonical_bvk_grid(
        static_data.period,
        static_data.full_kpoints, 1.0e-10);
    if (static_data.translations.size() !=
        static_data.full_kpoints.size())
    {
        throw std::invalid_argument(
            "QSGW Hartree full k and real-space grids have different sizes");
    }

    DensityRMap density_delta_r;
    if (symmetry_data == nullptr)
    {
        density_delta_r = build_total_density_delta_rspace(
            live, reference, scf_kpoints, static_data.translations);
    }
    else
    {
        if (symmetry_data->context == nullptr ||
            symmetry_data->wavefunction_layouts == nullptr ||
            symmetry_data->atom_ao_sizes == nullptr)
        {
            throw std::invalid_argument(
                "QSGW Hartree symmetry data is incomplete");
        }
        density_delta_r = build_total_density_delta_rspace_symmetry(
            *symmetry_data->context,
            *symmetry_data->wavefunction_layouts,
            live, reference, scf_kpoints, static_data.translations,
            *symmetry_data->atom_ao_sizes);
    }

    const WeightedDensityKMap density_delta_k =
        reconstruct_weighted_full_grid_density(
            density_delta_r, static_data.full_kpoints);
    const HartreeDkMap density_blocks =
        split_weighted_density_by_atom(
            density_delta_k, static_data.atom_ao_sizes);
    std::vector<int> kpoint_indices(static_data.full_kpoints.size());
    for (std::size_t index = 0; index < kpoint_indices.size(); ++index)
    {
        kpoint_indices[index] = static_cast<int>(index);
    }
    const HartreeDkMap hartree_k = contract_hartree_full_grid(
        static_data.c_k, static_data.v_q0, density_blocks,
        static_data.atom_ao_sizes, kpoint_indices,
        static_data.normalization);
    PeriodicOperatorRMap hartree_r = inverse_fourier_hartree_operator(
        hartree_k, static_data.full_kpoints,
        static_data.translations, &static_data.bvk_remap);
    maybe_dump_hartree_pipeline(
        static_data, density_delta_k, hartree_k, hartree_r);
    return hartree_r;
}

SpinKMatrixMap build_hartree_delta_fixed_basis(
    const HartreeStaticData& static_data,
    const MeanField& live,
    const MeanField& reference,
    const std::vector<Vector3_Order<double>>& scf_kpoints,
    const MeanField& target_reference,
    const std::vector<Vector3_Order<double>>& target_kpoints,
    const HartreeSymmetryData* symmetry_data)
{
    const PeriodicOperatorRMap hartree_r =
        build_hartree_delta_periodic_operator(
            static_data, live, reference, scf_kpoints, symmetry_data);
    return project_periodic_operator_to_fixed_basis(
        hartree_r, target_reference, target_kpoints);
}

} // namespace qsgw
} // namespace librpa_int
