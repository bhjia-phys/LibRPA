#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../qsgw/abacus_csr.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

using librpa_int::Matz;
using librpa_int::cplxdb;
using librpa_int::qsgw::AbacusCsrOptions;
using librpa_int::qsgw::RealSpaceMatrixMap;
using librpa_int::qsgw::write_abacus_hamiltonian_csr;

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

void test_writer_uses_abacus_real_csr_and_ry_units()
{
    Matz block(2, 2);
    block(0, 0) = 1.5;
    block(0, 1) = -0.25;
    block(1, 0) = 5.0e-12;
    block(1, 1) = 2.0;
    RealSpaceMatrixMap blocks;
    blocks[{0, 0, 0}] = block;

    std::ostringstream output;
    write_abacus_hamiltonian_csr(output, blocks);

    std::istringstream input(output.str());
    std::string line;
    std::getline(input, line);
    assert(line == "STEP: 0");
    std::getline(input, line);
    assert(line == "Matrix Dimension of H(R): 2");
    std::getline(input, line);
    assert(line == "Matrix number of H(R): 1");

    int rx = 0;
    int ry = 0;
    int rz = 0;
    int nonzero = 0;
    input >> rx >> ry >> rz >> nonzero;
    assert(rx == 0 && ry == 0 && rz == 0 && nonzero == 3);
    double v0 = 0.0;
    double v1 = 0.0;
    double v2 = 0.0;
    input >> v0 >> v1 >> v2;
    assert(std::abs(v0 - 3.0) < 1.0e-14);
    assert(std::abs(v1 + 0.5) < 1.0e-14);
    assert(std::abs(v2 - 4.0) < 1.0e-14);
    int c0 = -1;
    int c1 = -1;
    int c2 = -1;
    input >> c0 >> c1 >> c2;
    assert(c0 == 0 && c1 == 1 && c2 == 1);
    int p0 = -1;
    int p1 = -1;
    int p2 = -1;
    input >> p0 >> p1 >> p2;
    assert(p0 == 0 && p1 == 2 && p2 == 3);
}

void test_zero_blocks_are_omitted()
{
    RealSpaceMatrixMap blocks;
    blocks[{0, 0, 0}] = Matz(1, 1);
    std::ostringstream output;
    write_abacus_hamiltonian_csr(output, blocks);
    assert(output.str().find("Matrix number of H(R): 0") !=
           std::string::npos);
}

void test_invalid_or_lossy_output_is_rejected()
{
    RealSpaceMatrixMap empty;
    assert_throws([&] {
        std::ostringstream output;
        write_abacus_hamiltonian_csr(output, empty);
    });

    RealSpaceMatrixMap complex_blocks;
    complex_blocks[{0, 0, 0}] = Matz(1, 1);
    complex_blocks.at({0, 0, 0})(0, 0) = cplxdb(1.0, 1.0e-4);
    assert_throws([&] {
        std::ostringstream output;
        write_abacus_hamiltonian_csr(output, complex_blocks);
    });

    RealSpaceMatrixMap malformed;
    malformed[{0, 0, 0}] = Matz(1, 2);
    assert_throws([&] {
        std::ostringstream output;
        write_abacus_hamiltonian_csr(output, malformed);
    });

    RealSpaceMatrixMap nonfinite;
    nonfinite[{0, 0, 0}] = Matz(1, 1);
    nonfinite.at({0, 0, 0})(0, 0) =
        std::numeric_limits<double>::quiet_NaN();
    assert_throws([&] {
        std::ostringstream output;
        write_abacus_hamiltonian_csr(output, nonfinite);
    });

    AbacusCsrOptions invalid;
    invalid.zero_threshold_ry = -1.0;
    assert_throws([&] {
        std::ostringstream output;
        write_abacus_hamiltonian_csr(output, complex_blocks, invalid);
    });
}

} // namespace

int main()
{
    test_writer_uses_abacus_real_csr_and_ry_units();
    test_zero_blocks_are_omitted();
    test_invalid_or_lossy_output_is_rejected();
    return 0;
}
