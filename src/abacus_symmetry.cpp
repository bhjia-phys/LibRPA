/*!
 * @file abacus_symmetry.cpp
 * @brief Utilities for reading ABACUS symmetry sidecar files.
 */
#include "abacus_symmetry.h"

#include "constants.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace LIBRPA
{
namespace
{

constexpr double kAbacusSymmetryCoordTol = 1e-5;

std::string trim(const std::string& text)
{
    const auto begin = std::find_if_not(text.begin(), text.end(),
                                        [](unsigned char ch) { return std::isspace(ch) != 0; });
    if (begin == text.end())
    {
        return "";
    }
    const auto end = std::find_if_not(text.rbegin(), text.rend(),
                                      [](unsigned char ch) { return std::isspace(ch) != 0; })
                         .base();
    return std::string(begin, end);
}

std::string strip_comment(const std::string& text)
{
    const auto comment_pos = text.find('#');
    return trim(text.substr(0, comment_pos));
}

bool starts_with(const std::string& text, const std::string& prefix)
{
    return text.rfind(prefix, 0) == 0;
}

std::string parent_path(const std::string& file_path)
{
    const auto pos = file_path.find_last_of("/\\");
    if (pos == std::string::npos)
    {
        return ".";
    }
    if (pos == 0)
    {
        return file_path.substr(0, 1);
    }
    return file_path.substr(0, pos);
}

std::string base_name(const std::string& file_path)
{
    const auto pos = file_path.find_last_of("/\\");
    if (pos == std::string::npos)
    {
        return file_path;
    }
    return file_path.substr(pos + 1);
}

bool is_absolute_path(const std::string& file_path)
{
    if (file_path.empty())
    {
        return false;
    }
    if (file_path.front() == '/' || file_path.front() == '\\')
    {
        return true;
    }
    return file_path.size() > 1 && std::isalpha(static_cast<unsigned char>(file_path[0])) != 0
           && file_path[1] == ':';
}

std::string join_path(const std::string& dir_path, const std::string& file_name)
{
    if (dir_path.empty())
    {
        return file_name;
    }
    if (dir_path.back() == '/' || dir_path.back() == '\\')
    {
        return dir_path + file_name;
    }
    return dir_path + "/" + file_name;
}

bool file_exists(const std::string& file_path)
{
    std::ifstream ifs(file_path);
    return ifs.good();
}

std::vector<std::string> build_abacus_path_candidates(const std::string& dir_path)
{
    std::vector<std::string> dirs;
    auto append_unique = [&dirs](const std::string& dir) {
        if (!dir.empty() && std::find(dirs.begin(), dirs.end(), dir) == dirs.end())
        {
            dirs.push_back(dir);
        }
    };

    append_unique(dir_path);
    append_unique(join_path(dir_path, "OUT.ABACUS"));
    if (base_name(dir_path) == "OUT.ABACUS")
    {
        append_unique(parent_path(dir_path));
    }
    return dirs;
}

std::vector<std::string> split_fields(const std::string& line)
{
    std::vector<std::string> fields;
    std::istringstream iss(line);
    std::string field;
    while (iss >> field)
    {
        fields.push_back(field);
    }
    return fields;
}

bool is_section_header(const std::string& line)
{
    static const std::set<std::string> section_headers{
        "ATOMIC_SPECIES",
        "NUMERICAL_ORBITAL",
        "NUMERICAL_DESCRIPTOR",
        "LATTICE_CONSTANT",
        "LATTICE_PARAMETER",
        "LATTICE_VECTORS",
        "ATOMIC_POSITIONS",
        "NUMERICAL_DESCRIPTOR_VNA",
    };
    return section_headers.count(line) != 0;
}

std::vector<double> extract_doubles(const std::string& text)
{
    std::vector<double> values;
    const char* begin = text.c_str();
    char* end = nullptr;
    while (*begin != '\0')
    {
        const double value = std::strtod(begin, &end);
        if (end != begin)
        {
            values.push_back(value);
            begin = end;
        }
        else
        {
            ++begin;
        }
    }
    return values;
}

std::vector<long> extract_integers(const std::string& text)
{
    std::vector<long> values;
    const char* begin = text.c_str();
    char* end = nullptr;
    while (*begin != '\0')
    {
        const long value = std::strtol(begin, &end, 10);
        if (end != begin)
        {
            values.push_back(value);
            begin = end;
        }
        else
        {
            ++begin;
        }
    }
    return values;
}

bool is_integer_line(const std::string& text)
{
    const std::string stripped = trim(text);
    if (stripped.empty())
    {
        return false;
    }
    std::size_t start = (stripped.front() == '+' || stripped.front() == '-') ? 1 : 0;
    if (start == stripped.size())
    {
        return false;
    }
    return std::all_of(stripped.begin() + static_cast<std::ptrdiff_t>(start), stripped.end(),
                       [](unsigned char ch) { return std::isdigit(ch) != 0; });
}

bool starts_with_integer_token(const std::string& text)
{
    const std::string stripped = trim(text);
    if (stripped.empty())
    {
        return false;
    }

    std::size_t index = (stripped.front() == '+' || stripped.front() == '-') ? 1 : 0;
    if (index == stripped.size() || std::isdigit(static_cast<unsigned char>(stripped[index])) == 0)
    {
        return false;
    }
    while (index < stripped.size() && std::isdigit(static_cast<unsigned char>(stripped[index])) != 0)
    {
        ++index;
    }
    return index == stripped.size()
           || std::isspace(static_cast<unsigned char>(stripped[index])) != 0;
}

bool nearly_integer(const double value, const double tol = kAbacusSymmetryCoordTol)
{
    return std::abs(value - std::round(value)) < tol;
}

int shell_symbol_to_l(const char symbol)
{
    const std::string shells = "SPDFGHIJKLMNO";
    const auto pos = shells.find(static_cast<char>(std::toupper(static_cast<unsigned char>(symbol))));
    if (pos == std::string::npos)
    {
        return -1;
    }
    return static_cast<int>(pos);
}

int compute_nao_from_shell_counts(const std::vector<int>& shell_counts)
{
    int nao = 0;
    for (int l = 0; l < static_cast<int>(shell_counts.size()); ++l)
    {
        nao += shell_counts[l] * (2 * l + 1);
    }
    return nao;
}

std::vector<int> build_atom_offsets(const std::map<atom_t, size_t>& atom_nw)
{
    std::vector<int> offsets(atom_nw.size() + 1, 0);
    int running = 0;
    for (std::size_t atom = 0; atom < atom_nw.size(); ++atom)
    {
        const auto iter = atom_nw.find(atom);
        if (iter == atom_nw.end())
        {
            throw std::runtime_error("Atomic orbital counts are not contiguous in atom_nw");
        }
        offsets[atom] = running;
        running += static_cast<int>(iter->second);
    }
    offsets.back() = running;
    return offsets;
}

Vector3_Order<double> restrict_fractional_coordinate(const Vector3_Order<double>& vec,
                                                     const double tol = kAbacusSymmetryCoordTol)
{
    auto wrap = [tol](const double x) {
        double wrapped = std::fmod(x + 100.0 + tol, 1.0) - tol;
        if (std::abs(wrapped) < tol)
        {
            wrapped = 0.0;
        }
        return wrapped;
    };
    return {wrap(vec.x), wrap(vec.y), wrap(vec.z)};
}

Vector3_Order<double> multiply_row_vector(
    const Vector3_Order<double>& vec,
    const std::array<std::array<double, 3>, 3>& matrix)
{
    return {
        vec.x * matrix[0][0] + vec.y * matrix[1][0] + vec.z * matrix[2][0],
        vec.x * matrix[0][1] + vec.y * matrix[1][1] + vec.z * matrix[2][1],
        vec.x * matrix[0][2] + vec.y * matrix[1][2] + vec.z * matrix[2][2],
    };
}

std::array<std::array<double, 3>, 3> multiply_rotation_matrices(
    const std::array<std::array<double, 3>, 3>& lhs,
    const std::array<std::array<double, 3>, 3>& rhs)
{
    std::array<std::array<double, 3>, 3> product{{{{0.0, 0.0, 0.0}},
                                                   {{0.0, 0.0, 0.0}},
                                                   {{0.0, 0.0, 0.0}}}};
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            for (int k = 0; k < 3; ++k)
            {
                product[row][col] += lhs[row][k] * rhs[k][col];
            }
        }
    }
    return product;
}

bool is_identity_rotation(const std::array<std::array<double, 3>, 3>& matrix,
                          const double tol = 1e-8)
{
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            const double target = (row == col) ? 1.0 : 0.0;
            if (std::abs(matrix[row][col] - target) >= tol)
            {
                return false;
            }
        }
    }
    return true;
}

Vector3_Order<int> round_vec3_to_int(const Vector3_Order<double>& vec)
{
    return {static_cast<int>(std::lround(vec.x)),
            static_cast<int>(std::lround(vec.y)),
            static_cast<int>(std::lround(vec.z))};
}

bool is_nearly_integer_vec3(const Vector3_Order<double>& vec,
                            const double tol = kAbacusSymmetryCoordTol)
{
    return nearly_integer(vec.x, tol) && nearly_integer(vec.y, tol) && nearly_integer(vec.z, tol);
}

ComplexMatrix extract_atom_block(const ComplexMatrix& matrix,
                                 const atom_t atom_i,
                                 const atom_t atom_j,
                                 const std::map<atom_t, size_t>& atom_nw,
                                 const std::vector<int>& offsets)
{
    const int ni = static_cast<int>(atom_nw.at(atom_i));
    const int nj = static_cast<int>(atom_nw.at(atom_j));
    ComplexMatrix block(ni, nj);
    const int row0 = offsets[atom_i];
    const int col0 = offsets[atom_j];
    for (int i = 0; i < ni; ++i)
    {
        for (int j = 0; j < nj; ++j)
        {
            block(i, j) = matrix(row0 + i, col0 + j);
        }
    }
    return block;
}

void set_atom_block(ComplexMatrix& matrix,
                    const atom_t atom_i,
                    const atom_t atom_j,
                    const ComplexMatrix& block,
                    const std::vector<int>& offsets)
{
    const int row0 = offsets[atom_i];
    const int col0 = offsets[atom_j];
    for (int i = 0; i < block.nr; ++i)
    {
        for (int j = 0; j < block.nc; ++j)
        {
            matrix(row0 + i, col0 + j) = block(i, j);
        }
    }
}

struct AbacusRSpaceOperationInfo
{
    std::vector<atom_t> atom_map;
    std::vector<Vector3_Order<int>> return_lattice;
};

std::vector<AbacusRSpaceOperationInfo> build_rspace_operation_info(
    const AbacusSymmetryContext& ctx,
    const std::map<atom_t, std::array<double, 3>>& coord_frac)
{
    if (coord_frac.size() != ctx.atom_to_type.size())
    {
        throw std::runtime_error("Fractional coordinates and ABACUS atom mapping have inconsistent sizes");
    }

    std::vector<AbacusRSpaceOperationInfo> infos(ctx.rspace_operations.size());
    for (auto& info : infos)
    {
        info.atom_map.resize(coord_frac.size(), static_cast<atom_t>(-1));
        info.return_lattice.resize(coord_frac.size(), {0, 0, 0});
    }

    for (std::size_t isym = 0; isym < ctx.rspace_operations.size(); ++isym)
    {
        const auto& op = ctx.rspace_operations[isym];
        for (atom_t atom_from = 0; atom_from < coord_frac.size(); ++atom_from)
        {
            const auto& coord_from = coord_frac.at(atom_from);
            const Vector3_Order<double> coord_from_vec =
                restrict_fractional_coordinate({coord_from[0], coord_from[1], coord_from[2]});
            // Keep the unwrapped rotated position so that the integer return lattice is preserved
            // exactly as in the ABACUS irreducible-sector construction.
            const Vector3_Order<double> transformed =
                multiply_row_vector(coord_from_vec, op.rotation)
                + restrict_fractional_coordinate(op.translation);

            atom_t matched_atom = static_cast<atom_t>(-1);
            Vector3_Order<int> matched_return{0, 0, 0};
            for (atom_t atom_to = 0; atom_to < coord_frac.size(); ++atom_to)
            {
                if (ctx.atom_to_type.at(atom_from) != ctx.atom_to_type.at(atom_to))
                {
                    continue;
                }
                const auto& coord_to = coord_frac.at(atom_to);
                const Vector3_Order<double> coord_to_vec{coord_to[0], coord_to[1], coord_to[2]};
                const Vector3_Order<double> diff =
                    transformed - restrict_fractional_coordinate(coord_to_vec);
                if (!is_nearly_integer_vec3(diff))
                {
                    continue;
                }
                if (matched_atom != static_cast<atom_t>(-1))
                {
                    throw std::runtime_error("ABACUS real-space symmetry atom mapping is ambiguous");
                }
                matched_atom = atom_to;
                matched_return = round_vec3_to_int(diff);
            }

            if (matched_atom == static_cast<atom_t>(-1))
            {
                throw std::runtime_error("Failed to match ABACUS real-space symmetry atom mapping");
            }

            infos[isym].atom_map[atom_from] = matched_atom;
            infos[isym].return_lattice[atom_from] = matched_return;
        }
    }
    return infos;
}

std::vector<int> build_rspace_inverse_map(
    const AbacusSymmetryContext& ctx,
    const std::map<atom_t, std::array<double, 3>>& coord_frac)
{
    (void)coord_frac;
    std::vector<int> inverse_map(ctx.rspace_operations.size(), -1);
    for (std::size_t isym = 0; isym < ctx.rspace_operations.size(); ++isym)
    {
        for (std::size_t jsym = 0; jsym < ctx.rspace_operations.size(); ++jsym)
        {
            const auto composed_rotation = multiply_rotation_matrices(
                ctx.rspace_operations[isym].rotation, ctx.rspace_operations[jsym].rotation);
            const auto composed_translation =
                multiply_row_vector(ctx.rspace_operations[isym].translation,
                                    ctx.rspace_operations[jsym].rotation)
                + ctx.rspace_operations[jsym].translation;
            const bool is_inverse =
                is_identity_rotation(composed_rotation) && is_nearly_integer_vec3(composed_translation);

            if (is_inverse)
            {
                inverse_map[isym] = static_cast<int>(jsym);
                break;
            }
        }
        if (inverse_map[isym] < 0)
        {
            throw std::runtime_error("Failed to build inverse symmetry-operation map for ABACUS sidecars");
        }
    }
    return inverse_map;
}

Vector3_Order<int> rotate_rspace_vector(
    const Vector3_Order<int>& R,
    const AbacusRSpaceOperationInfo& op_info,
    const AbacusSymmetryOperation& op,
    const atom_t atom_from_i,
    const atom_t atom_from_j)
{
    const Vector3_Order<double> R_double{static_cast<double>(R.x), static_cast<double>(R.y),
                                         static_cast<double>(R.z)};
    const Vector3_Order<double> rotated_double =
        multiply_row_vector(R_double, op.rotation)
        + Vector3_Order<double>(static_cast<double>(op_info.return_lattice[atom_from_j].x),
                                static_cast<double>(op_info.return_lattice[atom_from_j].y),
                                static_cast<double>(op_info.return_lattice[atom_from_j].z))
        - Vector3_Order<double>(static_cast<double>(op_info.return_lattice[atom_from_i].x),
                                static_cast<double>(op_info.return_lattice[atom_from_i].y),
                                static_cast<double>(op_info.return_lattice[atom_from_i].z));
    if (!is_nearly_integer_vec3(rotated_double))
    {
        throw std::runtime_error("ABACUS real-space symmetry generated a non-integer lattice vector");
    }
    // Keep the raw rotated lattice vector returned by the ABACUS formula.
    // The caller is responsible for filtering against the explicit R list.
    return round_vec3_to_int(rotated_double);
}

Vector3_Order<double> parse_vec3_double(const std::string& line, const std::string& context)
{
    const auto values = extract_doubles(line);
    if (values.size() != 3)
    {
        throw std::runtime_error("Failed to parse 3-vector in " + context + ": " + line);
    }
    return {values[0], values[1], values[2]};
}

abacus_R_t parse_vec3_int(const std::string& line, const std::string& context)
{
    const auto values = extract_integers(line);
    if (values.size() < 3)
    {
        throw std::runtime_error("Failed to parse integer 3-vector in " + context + ": " + line);
    }
    return {static_cast<int>(values[values.size() - 3]),
            static_cast<int>(values[values.size() - 2]),
            static_cast<int>(values[values.size() - 1])};
}

ComplexMatrix parse_complex_row_matrix(const std::string& line,
                                       const int expected_count,
                                       const std::string& context)
{
    const auto values = extract_doubles(line);
    ComplexMatrix row(1, expected_count);
    if (static_cast<int>(values.size()) == expected_count)
    {
        for (int i = 0; i < expected_count; ++i)
        {
            row(0, i) = std::complex<double>(values[i], 0.0);
        }
    }
    else if (static_cast<int>(values.size()) == 2 * expected_count)
    {
        for (int i = 0; i < expected_count; ++i)
        {
            row(0, i) = std::complex<double>(values[2 * i], values[2 * i + 1]);
        }
    }
    else
    {
        throw std::runtime_error("Failed to parse complex row in " + context + ": " + line);
    }
    return row;
}

ComplexMatrix parse_shell_rotation(const std::vector<std::string>& lines,
                                   std::size_t& index,
                                   const int nm,
                                   const std::string& context)
{
    ComplexMatrix mat(nm, nm);
    for (int row = 0; row < nm; ++row)
    {
        while (index < lines.size() && trim(lines[index]).empty())
        {
            ++index;
        }
        if (index >= lines.size())
        {
            throw std::runtime_error("Unexpected end of file while reading " + context);
        }
        const auto parsed_row = parse_complex_row_matrix(lines[index], nm, context);
        for (int col = 0; col < nm; ++col)
        {
            mat(row, col) = parsed_row(0, col);
        }
        ++index;
    }
    return mat;
}

void load_irreducible_sector_file(const std::string& file_path,
                                  abacus_irreducible_sector_t& irreducible_sector)
{
    std::ifstream ifs(file_path);
    if (!ifs.good())
    {
        throw std::runtime_error("Failed to open " + file_path);
    }

    std::string line;
    while (std::getline(ifs, line))
    {
        if (trim(line).empty())
        {
            continue;
        }
        const auto values = extract_integers(line);
        if (values.size() != 5)
        {
            throw std::runtime_error("Failed to parse irreducible-sector line: " + line);
        }
        const atpair_t atom_pair{static_cast<atom_t>(values[0]), static_cast<atom_t>(values[1])};
        const abacus_R_t R{static_cast<int>(values[2]), static_cast<int>(values[3]),
                           static_cast<int>(values[4])};
        irreducible_sector[atom_pair].insert(R);
    }
}

void load_symrot_R_file(const std::string& file_path, AbacusSymmetryContext& ctx)
{
    std::ifstream ifs(file_path);
    if (!ifs.good())
    {
        throw std::runtime_error("Failed to open " + file_path);
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(ifs, line))
    {
        lines.push_back(line);
    }

    std::size_t index = 0;
    while (index < lines.size() && !starts_with(trim(lines[index]), "Lmax of AOs:"))
    {
        ++index;
    }
    if (index >= lines.size())
    {
        throw std::runtime_error("Missing AO Lmax header in " + file_path);
    }
    ctx.ao_lmax = static_cast<int>(extract_integers(lines[index]).front());
    ++index;

    while (index < lines.size() && !starts_with(trim(lines[index]), "Lmax of ABFs:"))
    {
        ++index;
    }
    if (index >= lines.size())
    {
        throw std::runtime_error("Missing ABF Lmax header in " + file_path);
    }
    ctx.abf_lmax = static_cast<int>(extract_integers(lines[index]).front());
    ++index;

    while (index < lines.size() && !is_integer_line(lines[index]))
    {
        ++index;
    }

    const int lmax = std::max(ctx.ao_lmax, ctx.abf_lmax);
    while (index < lines.size())
    {
        while (index < lines.size() && trim(lines[index]).empty())
        {
            ++index;
        }
        if (index >= lines.size())
        {
            break;
        }
        if (!is_integer_line(lines[index]))
        {
            throw std::runtime_error("Expected symmetry index in " + file_path + ": " + lines[index]);
        }

        AbacusSymmetryOperation op;
        op.isym = std::stoi(trim(lines[index]));
        ++index;

        for (int row = 0; row < 3; ++row)
        {
            while (index < lines.size() && trim(lines[index]).empty())
            {
                ++index;
            }
            if (index >= lines.size())
            {
                throw std::runtime_error("Unexpected end of file while reading rotation matrix");
            }
            const auto values = extract_doubles(lines[index]);
            if (values.size() != 3)
            {
                throw std::runtime_error("Failed to parse symmetry rotation matrix row: " + lines[index]);
            }
            for (int col = 0; col < 3; ++col)
            {
                op.rotation[row][col] = values[col];
            }
            ++index;
        }

        while (index < lines.size() && trim(lines[index]).empty())
        {
            ++index;
        }
        if (index >= lines.size())
        {
            throw std::runtime_error("Unexpected end of file while reading symmetry translation");
        }
        op.translation = parse_vec3_double(lines[index], file_path);
        ++index;

        for (int l = 0; l <= lmax; ++l)
        {
            const int nm = 2 * l + 1;
            op.shell_rotations[l] =
                parse_shell_rotation(lines, index, nm, "symrot_R l=" + std::to_string(l));
        }
        ctx.rspace_operations.push_back(std::move(op));
    }
}

void load_symrot_k_file(const std::string& file_path, AbacusSymmetryContext& ctx)
{
    std::ifstream ifs(file_path);
    if (!ifs.good())
    {
        throw std::runtime_error("Failed to open " + file_path);
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(ifs, line))
    {
        lines.push_back(line);
    }

    std::size_t index = 0;
    while (index < lines.size() && !starts_with(trim(lines[index]), "Star "))
    {
        ++index;
    }

    while (index < lines.size())
    {
        while (index < lines.size() && trim(lines[index]).empty())
        {
            ++index;
        }
        if (index >= lines.size())
        {
            break;
        }
        if (!starts_with(trim(lines[index]), "Star "))
        {
            throw std::runtime_error("Expected star header in " + file_path + ": " + lines[index]);
        }

        AbacusKStar star;
        const auto star_numbers = extract_integers(lines[index]);
        if (star_numbers.empty())
        {
            throw std::runtime_error("Failed to parse star index in " + lines[index]);
        }
        star.star_index = static_cast<int>(star_numbers.front()) - 1;
        const auto left = lines[index].find('(');
        const auto right = lines[index].find(')', left == std::string::npos ? 0 : left);
        if (left == std::string::npos || right == std::string::npos)
        {
            throw std::runtime_error("Failed to parse IBZ k-vector in " + lines[index]);
        }
        star.k_ibz = parse_vec3_double(lines[index].substr(left, right - left + 1), file_path);
        ++index;

        while (index < lines.size())
        {
            while (index < lines.size() && trim(lines[index]).empty())
            {
                ++index;
            }
            if (index >= lines.size() || starts_with(trim(lines[index]), "Star "))
            {
                break;
            }
            if (!starts_with_integer_token(lines[index]))
            {
                throw std::runtime_error("Expected symmetry index in " + file_path + ": " + lines[index]);
            }

            AbacusKStarMember member;
            member.isym = std::stoi(trim(lines[index]));
            ++index;

            while (index < lines.size() && trim(lines[index]).empty())
            {
                ++index;
            }
            if (index >= lines.size())
            {
                throw std::runtime_error("Unexpected end of file while reading k-star member");
            }
            member.k_bz = parse_vec3_double(lines[index], file_path);
            ++index;

            while (index < lines.size())
            {
                while (index < lines.size() && trim(lines[index]).empty())
                {
                    ++index;
                }
                if (index >= lines.size() || starts_with(trim(lines[index]), "Star ")
                    || starts_with_integer_token(lines[index]))
                {
                    break;
                }
                if (!starts_with(trim(lines[index]), "atom "))
                {
                    throw std::runtime_error("Expected atom header in " + file_path + ": " + lines[index]);
                }

                AbacusKAtomRotation atom_rotation;
                const auto values = extract_integers(lines[index]);
                if (values.size() < 4)
                {
                    throw std::runtime_error("Failed to parse atom symmetry header: " + lines[index]);
                }
                atom_rotation.atom_from = static_cast<int>(values[0]) - 1;
                atom_rotation.atom_to = static_cast<int>(values[1]) - 1;
                atom_rotation.atom_type = static_cast<int>(values[2]) - 1;
                atom_rotation.lmax = static_cast<int>(values[3]);
                ++index;

                for (int l = 0; l <= atom_rotation.lmax; ++l)
                {
                    const int nm = 2 * l + 1;
                    atom_rotation.shell_rotations[l] =
                        parse_shell_rotation(lines, index, nm, "symrot_k l=" + std::to_string(l));
                }
                member.atom_rotations.push_back(std::move(atom_rotation));
            }
            star.members.push_back(std::move(member));
        }

        ctx.kstars.push_back(std::move(star));
    }
}

std::string find_first_existing_file(const std::vector<std::string>& candidates)
{
    for (const auto& candidate : candidates)
    {
        if (!candidate.empty() && file_exists(candidate))
        {
            return candidate;
        }
    }
    return "";
}

std::string read_abacus_input_keyword(const std::string& input_file, const std::string& keyword)
{
    std::ifstream ifs(input_file);
    if (!ifs.good())
    {
        return "";
    }

    std::string line;
    while (std::getline(ifs, line))
    {
        const std::string cleaned = strip_comment(line);
        if (cleaned.empty())
        {
            continue;
        }
        const auto fields = split_fields(cleaned);
        if (!fields.empty() && fields.front() == keyword && fields.size() >= 2)
        {
            return fields[1];
        }
    }
    return "";
}

struct ParsedAbacusStru
{
    std::vector<std::string> species_labels;
    std::vector<std::string> orbital_files;
    std::map<atom_t, int> atom_to_type;
};

ParsedAbacusStru parse_abacus_stru_file(const std::string& stru_file)
{
    std::ifstream ifs(stru_file);
    if (!ifs.good())
    {
        throw std::runtime_error("Failed to open " + stru_file);
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(ifs, line))
    {
        const std::string cleaned = strip_comment(line);
        if (!cleaned.empty())
        {
            lines.push_back(cleaned);
        }
    }

    ParsedAbacusStru parsed;
    std::size_t index = 0;
    while (index < lines.size())
    {
        const std::string& current = lines[index];
        if (current == "ATOMIC_SPECIES")
        {
            ++index;
            while (index < lines.size() && !is_section_header(lines[index]))
            {
                const auto fields = split_fields(lines[index]);
                if (!fields.empty())
                {
                    parsed.species_labels.push_back(fields.front());
                }
                ++index;
            }
            continue;
        }
        if (current == "NUMERICAL_ORBITAL")
        {
            ++index;
            while (index < lines.size() && !is_section_header(lines[index]))
            {
                parsed.orbital_files.push_back(lines[index]);
                ++index;
            }
            continue;
        }
        if (current == "ATOMIC_POSITIONS")
        {
            ++index;
            if (index < lines.size())
            {
                ++index;
            }

            atom_t atom_index = 0;
            while (index < lines.size() && !is_section_header(lines[index]))
            {
                const auto species_fields = split_fields(lines[index]);
                if (species_fields.empty())
                {
                    ++index;
                    continue;
                }
                const std::string& species_label = species_fields.front();
                const auto type_iter =
                    std::find(parsed.species_labels.begin(), parsed.species_labels.end(), species_label);
                if (type_iter == parsed.species_labels.end())
                {
                    throw std::runtime_error("Failed to match atom type label " + species_label
                                             + " in " + stru_file);
                }
                const int atom_type =
                    static_cast<int>(std::distance(parsed.species_labels.begin(), type_iter));
                ++index;
                if (index >= lines.size())
                {
                    throw std::runtime_error("Unexpected end of file while reading magnetic moment block in "
                                             + stru_file);
                }
                ++index;
                if (index >= lines.size())
                {
                    throw std::runtime_error("Unexpected end of file while reading atom count in "
                                             + stru_file);
                }
                const auto count_fields = split_fields(lines[index]);
                if (count_fields.empty())
                {
                    throw std::runtime_error("Missing atom count in " + stru_file);
                }
                const int nat_this_type = std::stoi(count_fields.front());
                ++index;
                for (int i = 0; i < nat_this_type; ++i)
                {
                    if (index >= lines.size())
                    {
                        throw std::runtime_error("Unexpected end of file while reading atomic positions in "
                                                 + stru_file);
                    }
                    parsed.atom_to_type[atom_index++] = atom_type;
                    ++index;
                }
            }
            continue;
        }
        ++index;
    }

    return parsed;
}

AbacusAOTypeLayout parse_abacus_orbital_file(const std::string& orbital_file,
                                             const std::string& species_label)
{
    std::ifstream ifs(orbital_file);
    if (!ifs.good())
    {
        throw std::runtime_error("Failed to open orbital file " + orbital_file);
    }

    AbacusAOTypeLayout layout;
    layout.label = species_label;
    layout.orbital_file = orbital_file;

    int lmax = -1;
    std::string line;
    while (std::getline(ifs, line))
    {
        const std::string cleaned = strip_comment(line);
        if (cleaned.empty())
        {
            continue;
        }
        if (starts_with(cleaned, "Lmax"))
        {
            const auto values = extract_integers(cleaned);
            if (values.empty())
            {
                throw std::runtime_error("Failed to parse Lmax in orbital file " + orbital_file);
            }
            lmax = static_cast<int>(values.front());
            layout.shell_counts.resize(static_cast<std::size_t>(lmax + 1), 0);
            continue;
        }
        if (starts_with(cleaned, "Number of "))
        {
            const auto prefix_size = std::string("Number of ").size();
            const auto token = cleaned.substr(prefix_size);
            char shell_symbol = '\0';
            for (const char ch : token)
            {
                if (std::isalpha(static_cast<unsigned char>(ch)) != 0)
                {
                    shell_symbol = ch;
                    break;
                }
            }
            const int l = shell_symbol_to_l(shell_symbol);
            const auto values = extract_integers(cleaned);
            if (l < 0 || values.empty())
            {
                continue;
            }
            if (static_cast<int>(layout.shell_counts.size()) <= l)
            {
                layout.shell_counts.resize(static_cast<std::size_t>(l + 1), 0);
            }
            layout.shell_counts[static_cast<std::size_t>(l)] = static_cast<int>(values.back());
        }
        if (starts_with(cleaned, "SUMMARY"))
        {
            break;
        }
    }

    if (lmax >= 0 && static_cast<int>(layout.shell_counts.size()) < lmax + 1)
    {
        layout.shell_counts.resize(static_cast<std::size_t>(lmax + 1), 0);
    }
    layout.nao = compute_nao_from_shell_counts(layout.shell_counts);
    if (layout.nao <= 0)
    {
        throw std::runtime_error("Parsed zero AO functions from orbital file " + orbital_file);
    }
    return layout;
}

std::string resolve_abacus_file(const std::string& file_name,
                                const std::vector<std::string>& search_dirs)
{
    if (is_absolute_path(file_name))
    {
        return file_exists(file_name) ? file_name : "";
    }

    for (const auto& dir : search_dirs)
    {
        if (dir.empty())
        {
            continue;
        }
        const std::string candidate = join_path(dir, file_name);
        if (file_exists(candidate))
        {
            return candidate;
        }
    }
    return file_exists(file_name) ? file_name : "";
}

void try_load_abacus_ao_shell_layout(const std::string& dir_path,
                                     AbacusSymmetryContext& ctx,
                                     std::ostream* log)
{
    const auto candidate_dirs = build_abacus_path_candidates(dir_path);
    std::vector<std::string> stru_candidates;
    std::vector<std::string> input_candidates;
    for (const auto& dir : candidate_dirs)
    {
        stru_candidates.push_back(join_path(dir, "STRU"));
        input_candidates.push_back(join_path(dir, "INPUT"));
    }

    const std::string stru_file = find_first_existing_file(stru_candidates);
    if (stru_file.empty())
    {
        if (log != nullptr)
        {
            (*log) << "| AO shell layout        : unavailable (STRU not found)\n";
        }
        return;
    }

    const std::string input_file = find_first_existing_file(input_candidates);
    const std::string orbital_dir =
        input_file.empty() ? "" : read_abacus_input_keyword(input_file, "orbital_dir");

    try
    {
        const ParsedAbacusStru parsed = parse_abacus_stru_file(stru_file);
        if (parsed.species_labels.empty() || parsed.orbital_files.empty())
        {
            throw std::runtime_error("Failed to find ATOMIC_SPECIES / NUMERICAL_ORBITAL sections in "
                                     + stru_file);
        }
        if (parsed.species_labels.size() != parsed.orbital_files.size())
        {
            throw std::runtime_error("The number of NUMERICAL_ORBITAL entries does not match "
                                     "ATOMIC_SPECIES in " + stru_file);
        }

        std::vector<std::string> search_dirs;
        search_dirs.push_back(parent_path(stru_file));
        search_dirs.push_back(dir_path);
        if (!input_file.empty())
        {
            search_dirs.push_back(parent_path(input_file));
        }
        if (!orbital_dir.empty())
        {
            if (is_absolute_path(orbital_dir))
            {
                search_dirs.push_back(orbital_dir);
            }
            else
            {
                if (!input_file.empty())
                {
                    search_dirs.push_back(join_path(parent_path(input_file), orbital_dir));
                }
                search_dirs.push_back(join_path(parent_path(stru_file), orbital_dir));
                search_dirs.push_back(join_path(dir_path, orbital_dir));
            }
        }

        ctx.ao_type_layouts.clear();
        ctx.ao_type_layouts.reserve(parsed.species_labels.size());
        for (std::size_t itype = 0; itype < parsed.species_labels.size(); ++itype)
        {
            const std::string resolved_orbital =
                resolve_abacus_file(parsed.orbital_files[itype], search_dirs);
            if (resolved_orbital.empty())
            {
                throw std::runtime_error("Failed to resolve orbital file "
                                         + parsed.orbital_files[itype] + " for species "
                                         + parsed.species_labels[itype]);
            }
            ctx.ao_type_layouts.push_back(
                parse_abacus_orbital_file(resolved_orbital, parsed.species_labels[itype]));
        }

        ctx.atom_to_type = parsed.atom_to_type;
        ctx.ao_shell_layout_available = true;
        if (log != nullptr)
        {
            (*log) << "| AO shell layout        : loaded for " << ctx.ao_type_layouts.size()
                   << " atom types and " << ctx.atom_to_type.size() << " atoms\n";
            for (std::size_t itype = 0; itype < ctx.ao_type_layouts.size(); ++itype)
            {
                const auto& layout = ctx.ao_type_layouts[itype];
                (*log) << "|   type " << itype << " (" << layout.label << ")"
                       << " nao=" << layout.nao << " shell_counts=";
                for (std::size_t l = 0; l < layout.shell_counts.size(); ++l)
                {
                    if (l != 0)
                    {
                        (*log) << ",";
                    }
                    (*log) << layout.shell_counts[l];
                }
                (*log) << "\n";
            }
        }
    }
    catch (const std::exception& ex)
    {
        ctx.ao_type_layouts.clear();
        ctx.atom_to_type.clear();
        ctx.ao_shell_layout_available = false;
        if (log != nullptr)
        {
            (*log) << "| AO shell layout        : unavailable (" << ex.what() << ")\n";
        }
    }
}

} // namespace

AbacusSymmetryContext abacus_symmetry_ctx;

void AbacusSymmetryContext::clear()
{
    available = false;
    ao_shell_layout_available = false;
    ao_lmax = -1;
    abf_lmax = -1;
    irreducible_sector.clear();
    rspace_operations.clear();
    kstars.clear();
    ao_type_layouts.clear();
    atom_to_type.clear();
}

bool AbacusSymmetryContext::empty() const
{
    return irreducible_sector.empty() && rspace_operations.empty() && kstars.empty()
           && ao_type_layouts.empty() && atom_to_type.empty();
}

bool AbacusSymmetryContext::has_ao_shell_layout() const
{
    return ao_shell_layout_available && !ao_type_layouts.empty();
}

std::size_t AbacusSymmetryContext::count_irreducible_pairs() const
{
    return irreducible_sector.size();
}

std::size_t AbacusSymmetryContext::count_irreducible_blocks() const
{
    std::size_t count = 0;
    for (const auto& pair_Rs : irreducible_sector)
    {
        count += pair_Rs.second.size();
    }
    return count;
}

std::size_t AbacusSymmetryContext::count_kstar_members() const
{
    std::size_t count = 0;
    for (const auto& star : kstars)
    {
        count += star.members.size();
    }
    return count;
}

std::size_t AbacusSymmetryContext::count_atoms_with_layout() const
{
    return atom_to_type.size();
}

const AbacusAOTypeLayout& AbacusSymmetryContext::get_ao_type_layout(const int atom_type) const
{
    if (atom_type < 0 || atom_type >= static_cast<int>(ao_type_layouts.size()))
    {
        throw std::out_of_range("ABACUS atom type is out of range in AO shell layout");
    }
    return ao_type_layouts[static_cast<std::size_t>(atom_type)];
}

bool load_abacus_symmetry_context(const std::string& dir_path,
                                  AbacusSymmetryContext& ctx,
                                  std::ostream* log)
{
    const auto candidate_dirs = build_abacus_path_candidates(dir_path);
    std::string sidecar_dir;
    for (const auto& dir : candidate_dirs)
    {
        if (file_exists(join_path(dir, "irreducible_sector.txt"))
            || file_exists(join_path(dir, "symrot_R.txt"))
            || file_exists(join_path(dir, "symrot_k.txt")))
        {
            sidecar_dir = dir;
            break;
        }
    }

    const std::string irreducible_sector_file = join_path(sidecar_dir, "irreducible_sector.txt");
    const std::string symrot_R_file = join_path(sidecar_dir, "symrot_R.txt");
    const std::string symrot_k_file = join_path(sidecar_dir, "symrot_k.txt");

    const bool has_irreducible_sector = !sidecar_dir.empty() && file_exists(irreducible_sector_file);
    const bool has_symrot_R = !sidecar_dir.empty() && file_exists(symrot_R_file);
    const bool has_symrot_k = !sidecar_dir.empty() && file_exists(symrot_k_file);

    if (!has_irreducible_sector && !has_symrot_R && !has_symrot_k)
    {
        ctx.clear();
        return false;
    }

    if (!(has_irreducible_sector && has_symrot_R && has_symrot_k))
    {
        std::ostringstream oss;
        oss << "Incomplete ABACUS symmetry sidecar set near " << dir_path
            << ". Expected irreducible_sector.txt, symrot_R.txt and symrot_k.txt together.";
        throw std::runtime_error(oss.str());
    }

    ctx.clear();
    load_irreducible_sector_file(irreducible_sector_file, ctx.irreducible_sector);
    load_symrot_R_file(symrot_R_file, ctx);
    load_symrot_k_file(symrot_k_file, ctx);
    try_load_abacus_ao_shell_layout(dir_path, ctx, nullptr);
    ctx.available = true;

    if (log != nullptr)
    {
        (*log) << "Detected ABACUS symmetry sidecar files\n"
               << "| irreducible atom pairs : " << ctx.count_irreducible_pairs() << "\n"
               << "| irreducible {pair, R}  : " << ctx.count_irreducible_blocks() << "\n"
               << "| real-space operations  : " << ctx.rspace_operations.size() << "\n"
               << "| IBZ k-stars            : " << ctx.kstars.size() << "\n"
               << "| total star members     : " << ctx.count_kstar_members() << "\n"
               << "| AO / ABF lmax          : " << ctx.ao_lmax << " / " << ctx.abf_lmax << "\n";
        if (ctx.has_ao_shell_layout())
        {
            (*log) << "| AO shell layout        : loaded for " << ctx.ao_type_layouts.size()
                   << " atom types and " << ctx.atom_to_type.size() << " atoms\n";
            for (std::size_t itype = 0; itype < ctx.ao_type_layouts.size(); ++itype)
            {
                const auto& layout = ctx.ao_type_layouts[itype];
                (*log) << "|   type " << itype << " (" << layout.label << ")"
                       << " nao=" << layout.nao << " shell_counts=";
                for (std::size_t l = 0; l < layout.shell_counts.size(); ++l)
                {
                    if (l != 0)
                    {
                        (*log) << ",";
                    }
                    (*log) << layout.shell_counts[l];
                }
                (*log) << "\n";
            }
        }
        else
        {
            (*log) << "| AO shell layout        : unavailable\n";
        }
    }
    return true;
}

bool load_global_abacus_symmetry_context(const std::string& dir_path, std::ostream* log)
{
    return load_abacus_symmetry_context(dir_path, abacus_symmetry_ctx, log);
}

ComplexMatrix build_abacus_ao_rotation_matrix(const AbacusSymmetryContext& ctx,
                                              const int atom_type,
                                              const std::map<int, ComplexMatrix>& shell_rotations)
{
    const auto& layout = ctx.get_ao_type_layout(atom_type);
    ComplexMatrix rotation(layout.nao, layout.nao);

    int offset = 0;
    for (int l = 0; l < static_cast<int>(layout.shell_counts.size()); ++l)
    {
        const int shell_count = layout.shell_counts[static_cast<std::size_t>(l)];
        if (shell_count == 0)
        {
            continue;
        }

        const auto rotation_iter = shell_rotations.find(l);
        if (rotation_iter == shell_rotations.end())
        {
            throw std::runtime_error("Missing shell rotation block for l=" + std::to_string(l));
        }

        const ComplexMatrix& shell_rotation = rotation_iter->second;
        const int nm = 2 * l + 1;
        if (shell_rotation.nr != nm || shell_rotation.nc != nm)
        {
            throw std::runtime_error("Shell rotation block has incompatible shape for l="
                                     + std::to_string(l));
        }

        for (int ishell = 0; ishell < shell_count; ++ishell)
        {
            for (int row = 0; row < nm; ++row)
            {
                for (int col = 0; col < nm; ++col)
                {
                    rotation(offset + row, offset + col) = shell_rotation(row, col);
                }
            }
            offset += nm;
        }
    }

    if (offset != layout.nao)
    {
        throw std::runtime_error("Failed to assemble the full AO rotation matrix for atom type "
                                 + std::to_string(atom_type));
    }
    return rotation;
}

ComplexMatrix rotate_abacus_kspace_matrix(const AbacusSymmetryContext& ctx,
                                          const AbacusKStarMember& member,
                                          const ComplexMatrix& matrix_ibz,
                                          const std::map<atom_t, size_t>& atom_nw,
                                          const Vector3_Order<double>& k_ibz,
                                          const std::map<atom_t, std::array<double, 3>>& coord_frac_map,
                                          const bool use_time_reversal)
{
    // -------------------------------------------------------------------------
    // Rotate D(k_ibz) to D(k_bz) using the Bloch rotation matrix M from
    // symrot_k.txt.  M already includes the exp(2*pi*i * k_ibz . O_I) phase
    // (see ABACUS symmetry_rotation.cpp line 387-388).
    //
    // ABACUS col-major:  D^T(k_bz) = M† · D^T(k_ibz) · M
    // Row-major:         D(k_bz)   = M^T · D(k_ibz) · M*
    //
    // Block formula (M_I = M[S(I), I]):
    //   non-TRS:  D_bz[I, J] = M_I^T  · D_ibz[S(I), S(J)]  · conj(M_J)
    //   TRS:      D_bz[I, J] = M_I†   · conj(D_ibz[S(I), S(J)]) · M_J
    // -------------------------------------------------------------------------
    (void)k_ibz;         // phase already embedded in M from symrot_k.txt
    (void)coord_frac_map;
    if (!ctx.has_ao_shell_layout())
    {
        throw std::runtime_error("AO shell layout is required before rotating ABACUS k-space matrices");
    }

    const auto offsets = build_atom_offsets(atom_nw);
    const int nao_total = offsets.back();
    if (matrix_ibz.nr != nao_total || matrix_ibz.nc != nao_total)
    {
        throw std::runtime_error("The input matrix dimension is incompatible with the AO basis layout");
    }

    // Build atom permutation: rotations_by_from[I] gives the rotation entry for atom I.
    std::vector<const AbacusKAtomRotation*> rotations_by_from(atom_nw.size(), nullptr);
    std::vector<bool> visited_to(atom_nw.size(), false);
    for (const auto& atom_rotation : member.atom_rotations)
    {
        if (atom_rotation.atom_from < 0 || atom_rotation.atom_from >= static_cast<int>(atom_nw.size())
            || atom_rotation.atom_to < 0 || atom_rotation.atom_to >= static_cast<int>(atom_nw.size()))
        {
            throw std::runtime_error("ABACUS k-space atom mapping is out of range");
        }
        rotations_by_from[static_cast<std::size_t>(atom_rotation.atom_from)] = &atom_rotation;
        visited_to[static_cast<std::size_t>(atom_rotation.atom_to)] = true;
    }

    for (std::size_t atom = 0; atom < atom_nw.size(); ++atom)
    {
        if (rotations_by_from[atom] == nullptr)
        {
            throw std::runtime_error("ABACUS k-space atom rotations do not cover every atom");
        }
        if (!visited_to[atom])
        {
            throw std::runtime_error("ABACUS k-space atom mapping is not a full permutation");
        }
    }

    // Build the AO rotation blocks M_I for each atom.
    // symrot_k.txt exports the full Bloch orbital rotation matrix M which already
    // includes the exp(2*pi*i * k_ibz . O_I) phase factor (see ABACUS
    // symmetry_rotation.cpp line 387-388).  No phase correction is needed.
    ComplexMatrix rotated_matrix(nao_total, nao_total);

    std::vector<ComplexMatrix> atom_M_blocks(atom_nw.size());
    for (std::size_t atom = 0; atom < atom_nw.size(); ++atom)
    {
        const auto* atom_rotation = rotations_by_from[atom];
        atom_M_blocks[atom] = build_abacus_ao_rotation_matrix(ctx,
                                                               atom_rotation->atom_type,
                                                               atom_rotation->shell_rotations);
    }

    // Apply the block-level rotation formula.
    //
    // ABACUS col-major formula:  D^T(k_bz) = M† · D^T(k_ibz) · M
    // Row-major equivalent:      D(k_bz)   = M^T · D(k_ibz) · M*
    //
    // M[S(I), I] = (phase · T_l) is the per-atom block stored in symrot_k.txt
    // and returned by build_abacus_ao_rotation_matrix.
    //
    // Block formulas (M_I denotes M[S(I), I]):
    //   non-TRS:  D_bz[I, J] = M_I^T  · D_ibz[S(I), S(J)]  · conj(M_J)
    //   TRS:      D_bz[I, J] = M_I†   · D_ibz[S(I), S(J)]* · M_J
    //
    // Source indices: S(I) = atom_to,  destination indices: I = atom_from.
    for (std::size_t atom_i = 0; atom_i < atom_nw.size(); ++atom_i)
    {
        const auto* rot_i = rotations_by_from[atom_i];
        const auto& M_i = atom_M_blocks[atom_i];
        for (std::size_t atom_j = 0; atom_j < atom_nw.size(); ++atom_j)
        {
            const auto* rot_j = rotations_by_from[atom_j];
            const auto& M_j = atom_M_blocks[atom_j];
            // Read from D_ibz at the MAPPED atom positions S(I), S(J)
            const ComplexMatrix block_ibz =
                extract_atom_block(matrix_ibz,
                                   static_cast<atom_t>(rot_i->atom_to),
                                   static_cast<atom_t>(rot_j->atom_to),
                                   atom_nw, offsets);
            ComplexMatrix block_rotated;
            if (use_time_reversal)
            {
                // TRS: D_bz[I,J] = M_I† · conj(D_ibz[S(I),S(J)]) · M_J
                block_rotated = transpose(M_i, true) * conj(block_ibz) * M_j;
            }
            else
            {
                // Space group: D_bz[I,J] = M_I^T · D_ibz[S(I),S(J)] · conj(M_J)
                block_rotated = transpose(M_i, false) * block_ibz * conj(M_j);
            }
            // Write to D_bz at the ORIGINAL atom positions I, J
            set_atom_block(rotated_matrix,
                           static_cast<atom_t>(atom_i),
                           static_cast<atom_t>(atom_j),
                           block_rotated,
                           offsets);
        }
    }

    return rotated_matrix;
}

void build_abacus_rspace_sector_stars(const AbacusSymmetryContext& ctx,
                                      const std::map<atom_t, std::array<double, 3>>& coord_frac,
                                      const Vector3_Order<int>& period,
                                      const std::vector<Vector3_Order<int>>& Rlist,
                                      abacus_rspace_sector_stars_t& sector_stars,
                                      std::ostream* log)
{
    (void)period;
    if (!ctx.available || ctx.irreducible_sector.empty() || ctx.rspace_operations.empty())
    {
        throw std::runtime_error("ABACUS real-space symmetry metadata is incomplete");
    }
    if (ctx.atom_to_type.empty())
    {
        throw std::runtime_error("ABACUS atom-to-type mapping is unavailable for real-space symmetry");
    }

    const auto op_infos = build_rspace_operation_info(ctx, coord_frac);
    const auto inverse_map = build_rspace_inverse_map(ctx, coord_frac);

    sector_stars.clear();
    std::set<Vector3_Order<int>> Rset(Rlist.begin(), Rlist.end());
    using full_key_t = std::tuple<atom_t, atom_t, Vector3_Order<int>>;
    std::set<full_key_t> covered;

    for (const auto& pair_Rs : ctx.irreducible_sector)
    {
        const atpair_t& ir_pair = pair_Rs.first;
        for (const auto& ir_R_array : pair_Rs.second)
        {
            const Vector3_Order<int> ir_R{ir_R_array[0], ir_R_array[1], ir_R_array[2]};
            auto& star_members = sector_stars[ir_pair][ir_R];
            std::vector<std::string> candidate_debug;
            for (std::size_t isym = 0; isym < ctx.rspace_operations.size(); ++isym)
            {
                const int inv = inverse_map[isym];
                const auto& op_info = op_infos[static_cast<std::size_t>(inv)];
                const auto full_I = op_info.atom_map[ir_pair.first];
                const auto full_J = op_info.atom_map[ir_pair.second];
                const auto full_R = rotate_rspace_vector(ir_R,
                                                         op_info,
                                                         ctx.rspace_operations[static_cast<std::size_t>(inv)],
                                                         ir_pair.first,
                                                         ir_pair.second);
                const bool in_rset = Rset.count(full_R) != 0;
                const full_key_t full_key{full_I, full_J, full_R};
                const bool is_duplicate = covered.count(full_key) != 0;

                std::ostringstream oss;
                oss << "isym=" << isym << " inv=" << inv << " -> (" << full_I << ", " << full_J
                    << "), R=(" << full_R.x << ", " << full_R.y << ", " << full_R.z
                    << "), in_Rset=" << (in_rset ? "true" : "false")
                    << ", duplicate=" << (is_duplicate ? "true" : "false");
                candidate_debug.push_back(oss.str());

                if (!in_rset)
                {
                    continue;
                }

                if (covered.insert(full_key).second)
                {
                    star_members.push_back(
                        {static_cast<int>(isym), {full_I, full_J}, full_R});
                }
            }

            if (star_members.empty())
            {
                std::ostringstream oss;
                oss << "Failed to build a real-space symmetry star from ABACUS sidecars for "
                    << "irreducible pair (" << ir_pair.first << ", " << ir_pair.second << ")"
                    << " and R=(" << ir_R.x << ", " << ir_R.y << ", " << ir_R.z << ")";
                for (const auto& line : candidate_debug)
                {
                    oss << "\n  " << line;
                }
                throw std::runtime_error(oss.str());
            }
        }
    }

    if (log != nullptr)
    {
        std::size_t total_members = 0;
        for (const auto& pair_star : sector_stars)
        {
            for (const auto& R_star : pair_star.second)
            {
                total_members += R_star.second.size();
            }
        }
        (*log) << "| real-space sector stars: " << total_members << " full members restored from "
               << ctx.count_irreducible_blocks() << " irreducible blocks (" << covered.size()
               << " unique full {atom pair, R} blocks)\n";
    }
}

ComplexMatrix rotate_abacus_rspace_matrix(const AbacusSymmetryContext& ctx,
                                          const int isym,
                                          const atom_t atom_from_i,
                                          const atom_t atom_from_j,
                                          const ComplexMatrix& matrix_source)
{
    if (!ctx.has_ao_shell_layout())
    {
        throw std::runtime_error("AO shell layout is required before rotating ABACUS real-space matrices");
    }
    if (isym < 0 || isym >= static_cast<int>(ctx.rspace_operations.size()))
    {
        throw std::out_of_range("ABACUS real-space symmetry index is out of range");
    }

    const int type_i = ctx.atom_to_type.at(atom_from_i);
    const int type_j = ctx.atom_to_type.at(atom_from_j);
    const auto& op = ctx.rspace_operations[static_cast<std::size_t>(isym)];
    const ComplexMatrix T_i = build_abacus_ao_rotation_matrix(ctx, type_i, op.shell_rotations);
    const ComplexMatrix T_j = build_abacus_ao_rotation_matrix(ctx, type_j, op.shell_rotations);

    if (matrix_source.nr != T_i.nr || matrix_source.nc != T_j.nr)
    {
        throw std::runtime_error("ABACUS real-space rotation has incompatible AO dimensions");
    }

    return transpose(T_i, true) * matrix_source * T_j;
}

} // namespace LIBRPA
