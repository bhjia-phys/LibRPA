#include "hartree_dump.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace librpa_int
{
namespace qsgw
{
namespace
{

const char* normalization_name(const HartreeKNormalization normalization)
{
    switch (normalization)
    {
    case HartreeKNormalization::weighted_occupations:
        return "weighted_occupations";
    case HartreeKNormalization::legacy_extra_inverse_nk:
        return "legacy_extra_inverse_nk";
    }
    return "unknown";
}

void write_complex_value(std::ostream& output, const cplxdb& value)
{
    output << ' ' << std::setprecision(17) << value.real() << ' '
           << std::setprecision(17) << value.imag();
}

std::filesystem::path next_call_directory(
    const std::filesystem::path& dump_root)
{
    std::filesystem::create_directories(dump_root);
    if (!std::filesystem::is_directory(dump_root))
    {
        throw std::runtime_error(
            "QSGW Hartree dump root is not a directory: " +
            dump_root.string());
    }
    int call_count = 0;
    for (const auto& entry :
         std::filesystem::directory_iterator(dump_root))
    {
        if (entry.is_directory() &&
            entry.path().filename().string().rfind("call_", 0) == 0)
        {
            ++call_count;
        }
    }
    std::ostringstream name;
    name << "call_" << std::setw(3) << std::setfill('0') << (call_count + 1);
    const std::filesystem::path call_dir = dump_root / name.str();
    std::filesystem::create_directories(call_dir);
    return call_dir;
}

void write_manifest(const std::filesystem::path& call_dir,
                    const HartreeStaticData& static_data,
                    const WeightedDensityKMap& density_delta_k,
                    const HartreeDkMap& hartree_k,
                    const PeriodicOperatorRMap& hartree_r)
{
    std::ofstream output(call_dir / "manifest.txt");
    if (!output)
    {
        throw std::runtime_error("QSGW Hartree dump cannot write manifest");
    }
    output << "schema=qsgw_hartree_pipeline_dump_v1\n";
    output << "normalization=" << normalization_name(static_data.normalization)
           << '\n';
    output << "kpoint_count=" << static_data.full_kpoints.size() << '\n';
    output << "translation_count=" << static_data.translations.size() << '\n';
    output << "period=" << static_data.period.x << ' ' << static_data.period.y
           << ' ' << static_data.period.z << '\n';
    output << "atom_ao_sizes=";
    for (const auto& [atom, size] : static_data.atom_ao_sizes)
    {
        output << atom << ':' << size << ' ';
    }
    output << '\n';
    output << "density_delta_k_count=" << density_delta_k.size() << '\n';
    output << "hartree_k_atom_count=" << hartree_k.size() << '\n';
    output << "hartree_r_atom_count=" << hartree_r.size() << '\n';
    output << "density_delta_k_file=density_delta_k.txt\n";
    output << "density_delta_k_columns=kpoint row column real imag\n";
    output << "hartree_k_file=hartree_k.txt\n";
    output << "hartree_k_columns=atom_i atom_j kpoint row column real imag\n";
    output << "hartree_r_file=hartree_r.txt\n";
    output << "hartree_r_columns=atom_i atom_j R_x R_y R_z row column real "
              "imag\n";
}

void write_density_delta_k(const std::filesystem::path& call_dir,
                           const WeightedDensityKMap& density_delta_k)
{
    std::ofstream output(call_dir / "density_delta_k.txt");
    if (!output)
    {
        throw std::runtime_error(
            "QSGW Hartree dump cannot write density_delta_k");
    }
    output << "# kpoint row column real imag\n";
    for (const auto& [kpoint, matrix] : density_delta_k)
    {
        for (int row = 0; row < matrix.nr; ++row)
        {
            for (int column = 0; column < matrix.nc; ++column)
            {
                output << kpoint << ' ' << row << ' ' << column;
                write_complex_value(output, matrix(row, column));
                output << '\n';
            }
        }
    }
}

void write_hartree_k(const std::filesystem::path& call_dir,
                     const HartreeDkMap& hartree_k)
{
    std::ofstream output(call_dir / "hartree_k.txt");
    if (!output)
    {
        throw std::runtime_error("QSGW Hartree dump cannot write hartree_k");
    }
    output << "# atom_i atom_j kpoint row column real imag\n";
    for (const auto& [atom_i, by_atom_j] : hartree_k)
    {
        for (const auto& [atom_j, by_kpoint] : by_atom_j)
        {
            for (const auto& [kpoint, matrix] : by_kpoint)
            {
                for (int row = 0; row < matrix.nr; ++row)
                {
                    for (int column = 0; column < matrix.nc; ++column)
                    {
                        output << atom_i << ' ' << atom_j << ' ' << kpoint
                               << ' ' << row << ' ' << column;
                        write_complex_value(output, matrix(row, column));
                        output << '\n';
                    }
                }
            }
        }
    }
}

void write_hartree_r(const std::filesystem::path& call_dir,
                     const PeriodicOperatorRMap& hartree_r)
{
    std::ofstream output(call_dir / "hartree_r.txt");
    if (!output)
    {
        throw std::runtime_error("QSGW Hartree dump cannot write hartree_r");
    }
    output << "# atom_i atom_j R_x R_y R_z row column real imag\n";
    for (const auto& [atom_i, by_pair] : hartree_r)
    {
        for (const auto& [pair, matrix] : by_pair)
        {
            const int atom_j = pair.first;
            const Vector3_Order<int>& translation = pair.second;
            for (int row = 0; row < matrix.nr; ++row)
            {
                for (int column = 0; column < matrix.nc; ++column)
                {
                    output << atom_i << ' ' << atom_j << ' '
                           << translation.x << ' ' << translation.y << ' '
                           << translation.z << ' ' << row << ' ' << column;
                    write_complex_value(output, matrix(row, column));
                    output << '\n';
                }
            }
        }
    }
}

} // namespace

void maybe_dump_hartree_pipeline(
    const HartreeStaticData& static_data,
    const WeightedDensityKMap& density_delta_k,
    const HartreeDkMap& hartree_k,
    const PeriodicOperatorRMap& hartree_r)
{
    const char* dump_root_env = std::getenv("LIBRPA_QSGW_HARTREE_DUMP_DIR");
    if (dump_root_env == nullptr || dump_root_env[0] == '\0')
    {
        return;
    }
    const std::filesystem::path call_dir =
        next_call_directory(std::filesystem::path(dump_root_env));
    write_manifest(call_dir, static_data, density_delta_k, hartree_k,
                   hartree_r);
    write_density_delta_k(call_dir, density_delta_k);
    write_hartree_k(call_dir, hartree_k);
    write_hartree_r(call_dir, hartree_r);
}

} // namespace qsgw
} // namespace librpa_int
