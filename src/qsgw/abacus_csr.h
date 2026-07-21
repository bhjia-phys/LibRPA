#pragma once

#include "matrix_map.h"

#include <iosfwd>

namespace librpa_int
{
namespace qsgw
{

struct AbacusCsrOptions
{
    double zero_threshold_ry = 1.0e-10;
    double imaginary_tolerance_ha = 1.0e-10;
};

// Write non-SOC AO H(R) in the real ABACUS CSR format. Input is in Hartree;
// serialized matrix values are converted to Rydberg.
void write_abacus_hamiltonian_csr(
    std::ostream& output,
    const RealSpaceMatrixMap& blocks,
    const AbacusCsrOptions& options = {});

} // namespace qsgw
} // namespace librpa_int
