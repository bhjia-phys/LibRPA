#pragma once

#include <string>

#include "../../src/core/qsgw/headwing_velocity.h"

namespace driver::qsgw
{
// Native ABACUS real-space CSR inputs: S dimensionless, r converted to Bohr.
// QSGW supplies its current H in memory; G0W0 reads the original DFT H below.
librpa_int::qsgw::HeadwingAoOperators read_headwing_ao_operators(const std::string& overlap_file,
                                                                 const std::string& position_file,
                                                                 double position_scale_bohr = 1.0);
// Original DFT H(R): retain all complex entries and independent R support.
// The driver supplies the explicit conversion from input energy to Hartree.
librpa_int::qsgw::LocalizedMatrix read_headwing_hamiltonian_csr(const std::string& filename,
                                                                double energy_scale_ha);
// Diagnostic model actually differentiated for one entering QSGW update.
// Full complex CSR values are in eV; no entry threshold is applied.
void write_headwing_hamiltonian_csr(const std::string& filename,
                                    const librpa_int::qsgw::LocalizedMatrix& blocks);
}  // namespace driver::qsgw
