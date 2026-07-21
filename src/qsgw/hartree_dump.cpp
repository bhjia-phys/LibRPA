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

std::size_t count_bvk_remap_sources(const HartreeStaticData& static_data)
{
    std::size_t count = 0;
    for (const auto& [atom_pair, by_translation] :
         static_data.bvk_remap.data())
    {
        static_cast<void>(atom_pair);
        count += by_translation.size();
    }
    return count;
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
    output << "full_kpoints_file=full_kpoints.txt\n";
    output << "full_kpoints_columns=index kx ky kz\n";
    output << "translations_file=translations.txt\n";
    output << "translations_columns=index R_x R_y R_z\n";
    output << "bvk_remap_source_count="
           << count_bvk_remap_sources(static_data) << '\n';
    output << "bvk_remap_file=bvk_remap.txt\n";
    output << "bvk_remap_columns=atom_i atom_j source_R_x source_R_y "
              "source_R_z target_index target_count target_R_x target_R_y "
              "target_R_z\n";
}

void write_full_kpoints(const std::filesystem::path& call_dir,
                        const HartreeStaticData& static_data)
{
    std::ofstream output(call_dir / "full_kpoints.txt");
    if (!output)
    {
        throw std::runtime_error(
            "QSGW Hartree dump cannot write full k-point grid");
    }
    output << "# index kx ky kz\n";
    for (std::size_t index = 0; index < static_data.full_kpoints.size();
         ++index)
    {
        const auto& kpoint = static_data.full_kpoints[index];
        output << index << ' ' << std::setprecision(17) << kpoint.x << ' '
               << kpoint.y << ' ' << kpoint.z << '\n';
    }
}

void write_translations(const std::filesystem::path& call_dir,
                        const HartreeStaticData& static_data)
{
    std::ofstream output(call_dir / "translations.txt");
    if (!output)
    {
        throw std::runtime_error(
            "QSGW Hartree dump cannot write BvK translations");
    }
    output << "# index R_x R_y R_z\n";
    for (std::size_t index = 0; index < static_data.translations.size();
         ++index)
    {
        const auto& translation = static_data.translations[index];
        output << index << ' ' << translation.x << ' ' << translation.y
               << ' ' << translation.z << '\n';
    }
}

void write_bvk_remap(const std::filesystem::path& call_dir,
                     const HartreeStaticData& static_data)
{
    std::ofstream output(call_dir / "bvk_remap.txt");
    if (!output)
    {
        throw std::runtime_error(
            "QSGW Hartree dump cannot write atom-pair BvK remap");
    }
    output << "# atom_i atom_j source_R_x source_R_y source_R_z "
              "target_index target_count target_R_x target_R_y target_R_z\n";
    for (const auto& [atom_pair, by_translation] :
         static_data.bvk_remap.data())
    {
        for (const auto& [source, targets] : by_translation)
        {
            for (std::size_t index = 0; index < targets.size(); ++index)
            {
                const auto& target = targets[index];
                output << atom_pair.first << ' ' << atom_pair.second << ' '
                       << source.x << ' ' << source.y << ' ' << source.z
                       << ' ' << index << ' ' << targets.size() << ' '
                       << target.x << ' ' << target.y << ' ' << target.z
                       << '\n';
            }
        }
    }
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
    write_full_kpoints(call_dir, static_data);
    write_translations(call_dir, static_data);
    write_bvk_remap(call_dir, static_data);
    write_density_delta_k(call_dir, density_delta_k);
    write_hartree_k(call_dir, hartree_k);
    write_hartree_r(call_dir, hartree_r);
}

} // namespace qsgw
} // namespace librpa_int
