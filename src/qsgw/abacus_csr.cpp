#include "abacus_csr.h"

#include <cmath>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace librpa_int
{
namespace qsgw
{
namespace
{

struct SparseBlock
{
    Vector3_Order<int> cell;
    std::vector<double> values_ry;
    std::vector<int> columns;
    std::vector<int> row_offsets;
};

void validate_options(const AbacusCsrOptions& options)
{
    if (!std::isfinite(options.zero_threshold_ry) ||
        options.zero_threshold_ry < 0.0 ||
        !std::isfinite(options.imaginary_tolerance_ha) ||
        options.imaginary_tolerance_ha < 0.0)
    {
        throw std::invalid_argument(
            "QSGW ABACUS CSR tolerances must be finite and non-negative");
    }
}

SparseBlock make_sparse_block(const Vector3_Order<int>& cell,
                              const Matz& matrix,
                              const int dimension,
                              const AbacusCsrOptions& options)
{
    if (matrix.nr() != dimension || matrix.nc() != dimension)
    {
        throw std::invalid_argument(
            "QSGW ABACUS CSR blocks have inconsistent dimensions");
    }

    SparseBlock result;
    result.cell = cell;
    result.row_offsets.reserve(static_cast<std::size_t>(dimension) + 1);
    result.row_offsets.push_back(0);
    for (int row = 0; row < dimension; ++row)
    {
        for (int column = 0; column < dimension; ++column)
        {
            const cplxdb value = matrix(row, column);
            if (!std::isfinite(value.real()) ||
                !std::isfinite(value.imag()))
            {
                throw std::invalid_argument(
                    "QSGW ABACUS CSR input contains non-finite data");
            }
            if (std::abs(value.imag()) > options.imaginary_tolerance_ha)
            {
                throw std::invalid_argument(
                    "QSGW ABACUS real CSR export would discard a significant imaginary component");
            }
            const double value_ry = 2.0 * value.real();
            if (std::abs(value_ry) > options.zero_threshold_ry)
            {
                result.values_ry.push_back(value_ry);
                result.columns.push_back(column);
            }
        }
        result.row_offsets.push_back(
            static_cast<int>(result.values_ry.size()));
    }
    return result;
}

} // namespace

void write_abacus_hamiltonian_csr(
    std::ostream& output,
    const RealSpaceMatrixMap& blocks,
    const AbacusCsrOptions& options)
{
    validate_options(options);
    if (!output.good())
        throw std::invalid_argument(
            "QSGW ABACUS CSR output stream is not writable");
    if (blocks.empty())
        throw std::invalid_argument(
            "QSGW ABACUS CSR export requires real-space blocks");

    const int dimension = blocks.begin()->second.nr();
    if (dimension <= 0 || blocks.begin()->second.nc() != dimension)
        throw std::invalid_argument(
            "QSGW ABACUS CSR matrices must be non-empty and square");

    std::vector<SparseBlock> sparse_blocks;
    sparse_blocks.reserve(blocks.size());
    for (const auto& [cell, matrix] : blocks)
    {
        SparseBlock sparse = make_sparse_block(
            cell, matrix, dimension, options);
        if (!sparse.values_ry.empty())
            sparse_blocks.push_back(std::move(sparse));
    }

    output << "STEP: 0\n"
           << "Matrix Dimension of H(R): " << dimension << '\n'
           << "Matrix number of H(R): " << sparse_blocks.size() << '\n';
    for (const SparseBlock& block : sparse_blocks)
    {
        output << block.cell.x << ' ' << block.cell.y << ' '
               << block.cell.z << ' ' << block.values_ry.size() << '\n';
        for (const double value : block.values_ry)
            output << ' ' << std::scientific << std::setprecision(16)
                   << value;
        output << '\n';
        for (const int column : block.columns) output << ' ' << column;
        output << '\n';
        for (const int offset : block.row_offsets) output << ' ' << offset;
        output << '\n';
    }
    if (!output.good())
        throw std::runtime_error("Failed to write QSGW ABACUS CSR output");
}

} // namespace qsgw
} // namespace librpa_int
