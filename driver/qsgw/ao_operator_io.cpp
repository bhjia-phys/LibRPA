#include "ao_operator_io.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "../../src/utils/constants.h"

namespace driver::qsgw
{
namespace
{
using namespace librpa_int;

std::string line(std::istream& input)
{
    std::string result;
    while (std::getline(input, result))
    {
        const auto comment = result.find('#');
        if (comment != std::string::npos) result.erase(comment);
        if (result.find_first_not_of(" \t\r") != std::string::npos) return result;
    }
    throw std::runtime_error("Incomplete QSGW AO CSR input");
}

int last_integer(const std::string& text)
{
    std::istringstream input(text);
    std::string token;
    while (input >> token)
    {
    }
    std::size_t end = 0;
    const int value = std::stoi(token, &end);
    if (end != token.size()) throw std::runtime_error("Invalid QSGW AO CSR header");
    return value;
}

enum class CsrFormat
{
    Compact,
    Legacy,
    Modern,
};

std::tuple<int, int, CsrFormat> header(std::istream& input)
{
    std::string first = line(input);
    if (first.find("--- Ionic Step") != std::string::npos)
    {
        int n = -1, nr = -1;
        std::string text;
        while (std::getline(input, text))
        {
            if (text.find("--- Ionic Step") != std::string::npos)
                throw std::runtime_error("Multiple ionic steps in QSGW AO CSR input");
            if (text.find("number of localized basis") != std::string::npos)
            {
                if (!(std::istringstream(text) >> n))
                    throw std::runtime_error("Invalid QSGW AO CSR basis count");
            }
            if (text.find("number of Bravais lattice vector R") != std::string::npos)
            {
                if (!(std::istringstream(text) >> nr))
                    throw std::runtime_error("Invalid QSGW AO CSR R count");
            }
            if (text.find("CSR Format") != std::string::npos)
            {
                if (n <= 0 || nr < 0)
                    throw std::runtime_error("Invalid QSGW AO CSR dimensions");
                return {n, nr, CsrFormat::Modern};
            }
        }
        throw std::runtime_error("Missing CSR Format section in QSGW AO input");
    }
    const bool native = first.find("STEP:") != std::string::npos ||
                        first.find("Matrix Dimension") != std::string::npos;
    if (first.find("STEP:") != std::string::npos) first = line(input);
    const int n = last_integer(first);
    const int nr = last_integer(line(input));
    if (n <= 0 || nr < 0) throw std::runtime_error("Invalid QSGW AO CSR dimensions");
    return {n, nr, native ? CsrFormat::Legacy : CsrFormat::Compact};
}

void skip_comment_lines(std::istream& input)
{
    input >> std::ws;
    while (input.peek() == '#')
    {
        std::string ignored;
        std::getline(input, ignored);
        input >> std::ws;
    }
}

void csr_marker(std::istream& input, const std::string& expected)
{
    std::string text;
    while (std::getline(input, text))
    {
        const auto first = text.find_first_not_of(" \t\r");
        if (first == std::string::npos) continue;
        const auto last = text.find_last_not_of(" \t\r");
        if (text.substr(first, last - first + 1) != expected)
            throw std::runtime_error("Expected " + expected + " in QSGW AO CSR input");
        return;
    }
    throw std::runtime_error("Missing " + expected + " in QSGW AO CSR input");
}

Matz matrix(std::istream& input, const int n, const int nnz, const bool empty_has_pointers,
            const bool upper_only = true, const CsrFormat format = CsrFormat::Legacy)
{
    if (nnz < 0 || static_cast<long long>(nnz) > static_cast<long long>(n) * n)
        throw std::runtime_error("Invalid QSGW AO CSR nonzero count");
    Matz result(n, n, MAJOR::ROW);
    std::fill(result.ptr(), result.ptr() + result.size(), cplxdb{});
    if (nnz == 0 && !empty_has_pointers && format != CsrFormat::Modern) return result;
    std::vector<cplxdb> values(nnz);
    std::vector<int> columns(nnz), pointers(n + 1);
    if (format == CsrFormat::Modern) csr_marker(input, "# CSR values");
    for (auto& x : values)
        if (!(input >> x) || !std::isfinite(x.real()) || !std::isfinite(x.imag()))
            throw std::runtime_error("Invalid QSGW AO CSR value");
    if (format == CsrFormat::Modern) csr_marker(input, "# CSR column indices");
    for (auto& x : columns)
        if (!(input >> x) || x < 0 || x >= n)
            throw std::runtime_error("Invalid QSGW AO CSR column");
    if (format == CsrFormat::Modern) csr_marker(input, "# CSR row pointers");
    for (auto& x : pointers)
        if (!(input >> x) || x < 0 || x > nnz)
            throw std::runtime_error("Invalid QSGW AO CSR row pointer");
    if (pointers.front() != 0 || pointers.back() != nnz ||
        !std::is_sorted(pointers.begin(), pointers.end()))
        throw std::runtime_error("Invalid QSGW AO CSR row range");
    for (int row = 0; row < n; ++row)
    {
        int previous = -1;
        for (int j = pointers[row]; j < pointers[row + 1]; ++j)
        {
            if (columns[j] <= previous)
                throw std::runtime_error("QSGW AO CSR columns are not unique and ordered");
            previous = columns[j];
            if (!upper_only || columns[j] >= row) result(row, columns[j]) = values[j];
        }
    }
    return result;
}

}  // namespace

librpa_int::qsgw::HeadwingAoOperators read_headwing_ao_operators(const std::string& overlap_file,
                                                                 const std::string& position_file,
                                                                 const double position_scale_bohr)
{
    if (!(position_scale_bohr > 0.0) || !std::isfinite(position_scale_bohr))
        throw std::runtime_error("Invalid head/wing position-unit conversion");
    std::ifstream s(overlap_file), r(position_file);
    if (!s || !r) throw std::runtime_error("Cannot open QSGW head/wing S(R) or r(R) CSR input");
    const auto [n, ns, format_s] = header(s);
    const auto [nr, nr_blocks, format_r] = header(r);
    if (n != nr) throw std::runtime_error("QSGW head/wing S(R) and r(R) dimensions differ");
    librpa_int::qsgw::HeadwingAoOperators result;
    result.dimension = n;
    for (int i = 0; i < ns; ++i)
    {
        int x, y, z, nnz;
        skip_comment_lines(s);
        if (!(s >> x >> y >> z >> nnz)) throw std::runtime_error("Incomplete QSGW overlap R block");
        if (!result.overlap
                 .emplace(Vector3_Order<int>(x, y, z), matrix(s, n, nnz, false, true, format_s))
                 .second)
            throw std::runtime_error("Duplicate QSGW overlap R block");
    }
    for (int i = 0; i < nr_blocks; ++i)
    {
        int x, y, z;
        skip_comment_lines(r);
        if (!(r >> x >> y >> z)) throw std::runtime_error("Incomplete QSGW position R block");
        for (int a = 0; a < 3; ++a)
        {
            int nnz;
            if (!(r >> nnz)) throw std::runtime_error("Incomplete QSGW position nonzero count");
            if (!result.position[a]
                     .emplace(Vector3_Order<int>(x, y, z),
                              matrix(r, n, nnz, format_r == CsrFormat::Compact, true, format_r))
                     .second)
                throw std::runtime_error("Duplicate QSGW position R block");
        }
    }
    std::string trailing;
    skip_comment_lines(s);
    skip_comment_lines(r);
    if (s >> trailing || r >> trailing) throw std::runtime_error("Trailing QSGW AO CSR data");
    for (auto& component : result.position)
        for (auto& [cell, block] : component)
            for (std::size_t j = 0; j < block.size(); ++j) block.ptr()[j] *= position_scale_bohr;
    return result;
}

librpa_int::qsgw::LocalizedMatrix read_headwing_hamiltonian_csr(const std::string& filename,
                                                                const double energy_scale_ha)
{
    if (!(energy_scale_ha > 0.0) || !std::isfinite(energy_scale_ha))
        throw std::runtime_error("Invalid head/wing Hamiltonian-unit conversion");
    std::ifstream input(filename);
    if (!input) throw std::runtime_error("Cannot open head/wing H(R) CSR input: " + filename);
    const auto [n, nr, format] = header(input);
    librpa_int::qsgw::LocalizedMatrix result;
    for (int i = 0; i < nr; ++i)
    {
        int x, y, z, nnz;
        skip_comment_lines(input);
        if (!(input >> x >> y >> z >> nnz))
            throw std::runtime_error("Incomplete head/wing Hamiltonian R block");
        auto block = matrix(input, n, nnz, false, false, format);
        for (std::size_t j = 0; j < block.size(); ++j) block.ptr()[j] *= energy_scale_ha;
        if (!result.emplace(Vector3_Order<int>(x, y, z), std::move(block)).second)
            throw std::runtime_error("Duplicate head/wing Hamiltonian R block");
    }
    std::string trailing;
    skip_comment_lines(input);
    if (input >> trailing) throw std::runtime_error("Trailing head/wing Hamiltonian CSR data");
    return result;
}
void write_headwing_hamiltonian_csr(const std::string& filename,
                                    const librpa_int::qsgw::LocalizedMatrix& blocks)
{
    if (blocks.empty()) throw std::runtime_error("Empty QSGW head/wing Hamiltonian");
    const int n = blocks.begin()->second.nr();
    std::ofstream output(filename);
    if (!output) throw std::runtime_error("Cannot write QSGW head/wing Hamiltonian: " + filename);
    output << "STEP: 0\nMatrix Dimension of H(R): " << n
           << "\nMatrix number of H(R): " << blocks.size() << '\n'
           << std::setprecision(17) << std::scientific;
    for (const auto& [r, matrix] : blocks)
    {
        if (matrix.nr() != n || matrix.nc() != n)
            throw std::runtime_error("Invalid QSGW head/wing Hamiltonian shape");
        output << r.x << ' ' << r.y << ' ' << r.z << ' ' << n * n << '\n';
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) output << matrix(i, j) * librpa_int::HA2EV << ' ';
        output << '\n';
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) output << j << ' ';
        output << '\n';
        for (int i = 0; i <= n; ++i) output << i * n << ' ';
        output << '\n';
    }
    output.flush();
    if (!output) throw std::runtime_error("Cannot flush QSGW head/wing Hamiltonian: " + filename);
}
}  // namespace driver::qsgw
