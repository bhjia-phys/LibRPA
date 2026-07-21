#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../qsgw/hartree_dump.h"

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using librpa_int::ComplexMatrix;
using librpa_int::Vector3_Order;
using librpa_int::cplxdb;
using librpa_int::qsgw::HartreeDkMap;
using librpa_int::qsgw::HartreeKNormalization;
using librpa_int::qsgw::HartreeStaticData;
using librpa_int::qsgw::PeriodicOperatorRMap;
using librpa_int::qsgw::WeightedDensityKMap;
using librpa_int::qsgw::maybe_dump_hartree_pipeline;

namespace
{

const std::string dump_root = "test_qsgw_hartree_dump.tmp";

HartreeStaticData make_static_data()
{
    HartreeStaticData static_data;
    static_data.normalization = HartreeKNormalization::legacy_extra_inverse_nk;
    static_data.atom_ao_sizes = {{0, 2}, {1, 2}};
    static_data.period = Vector3_Order<int>(2, 1, 1);
    static_data.full_kpoints = {Vector3_Order<double>(0.0, 0.0, 0.0),
                                Vector3_Order<double>(0.5, 0.0, 0.0)};
    static_data.translations = {Vector3_Order<int>(0, 0, 0),
                                Vector3_Order<int>(1, 0, 0)};
    const std::map<librpa_int::atom_t, librpa_int::Vector3<double>>
        coordinates{{0, {0.1, 0.0, 0.0}}, {1, {0.9, 0.0, 0.0}}};
    static_data.bvk_remap =
        librpa_int::AtomPairBvKRemap<librpa_int::atom_t>(
            coordinates, static_data.translations,
            Vector3_Order<int>(3, 1, 1), librpa_int::Matrix3{}, 0);
    return static_data;
}

WeightedDensityKMap make_density()
{
    WeightedDensityKMap density;
    for (int kpoint = 0; kpoint < 2; ++kpoint)
    {
        ComplexMatrix matrix(4, 4);
        for (int index = 0; index < 16; ++index)
        {
            matrix.c[index] = cplxdb(0.01 * (index + 1 + kpoint),
                                     -0.02 * (index + 1));
        }
        density[kpoint] = matrix;
    }
    return density;
}

HartreeDkMap make_hartree_k()
{
    HartreeDkMap hartree;
    for (int atom_i = 0; atom_i < 2; ++atom_i)
    {
        for (int atom_j = 0; atom_j < 2; ++atom_j)
        {
            for (int kpoint = 0; kpoint < 2; ++kpoint)
            {
                ComplexMatrix block(2, 2);
                for (int index = 0; index < 4; ++index)
                {
                    block.c[index] = cplxdb(
                        0.1 * (index + 1 + atom_i + atom_j + kpoint), 0.0);
                }
                hartree[atom_i][atom_j][kpoint] = block;
            }
        }
    }
    return hartree;
}

PeriodicOperatorRMap make_hartree_r()
{
    PeriodicOperatorRMap hartree;
    for (int atom_i = 0; atom_i < 2; ++atom_i)
    {
        for (int atom_j = 0; atom_j < 2; ++atom_j)
        {
            for (const auto& translation :
                 {Vector3_Order<int>(0, 0, 0), Vector3_Order<int>(1, 0, 0)})
            {
                ComplexMatrix block(2, 2);
                for (int index = 0; index < 4; ++index)
                {
                    block.c[index] =
                        cplxdb(0.05 * (index + 1 + atom_i), 0.01 * index);
                }
                hartree[atom_i][{atom_j, translation}] = block;
            }
        }
    }
    return hartree;
}

std::string read_file(const std::filesystem::path& path)
{
    std::ifstream input(path);
    assert(input.good());
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

long count_data_rows(const std::string& text)
{
    long count = 0;
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line))
    {
        if (!line.empty() && line[0] != '#')
        {
            ++count;
        }
    }
    return count;
}

void test_noop_without_env()
{
    std::filesystem::remove_all(dump_root);
#ifdef _WIN32
    _putenv("LIBRPA_QSGW_HARTREE_DUMP_DIR=");
#else
    unsetenv("LIBRPA_QSGW_HARTREE_DUMP_DIR");
#endif
    maybe_dump_hartree_pipeline(make_static_data(), make_density(),
                                make_hartree_k(), make_hartree_r());
    assert(!std::filesystem::exists(dump_root));
}

void test_dump_schema_and_roundtrip()
{
    std::filesystem::remove_all(dump_root);
#ifdef _WIN32
    _putenv((std::string("LIBRPA_QSGW_HARTREE_DUMP_DIR=") + dump_root).c_str());
#else
    setenv("LIBRPA_QSGW_HARTREE_DUMP_DIR", dump_root.c_str(), 1);
#endif
    const WeightedDensityKMap density = make_density();
    maybe_dump_hartree_pipeline(make_static_data(), density,
                                make_hartree_k(), make_hartree_r());

    const std::filesystem::path call_dir =
        std::filesystem::path(dump_root) / "call_001";
    assert(std::filesystem::is_directory(call_dir));

    const std::string manifest = read_file(call_dir / "manifest.txt");
    assert(manifest.find("schema=qsgw_hartree_pipeline_dump_v1\n") !=
           std::string::npos);
    assert(manifest.find("normalization=legacy_extra_inverse_nk\n") !=
           std::string::npos);
    assert(manifest.find("kpoint_count=2\n") != std::string::npos);
    assert(manifest.find("atom_ao_sizes=0:2 1:2 \n") != std::string::npos);
    assert(manifest.find("full_kpoints_file=full_kpoints.txt\n") !=
           std::string::npos);
    assert(manifest.find("translations_file=translations.txt\n") !=
           std::string::npos);
    assert(manifest.find("bvk_remap_source_count=1\n") !=
           std::string::npos);
    assert(manifest.find("bvk_remap_file=bvk_remap.txt\n") !=
           std::string::npos);

    const std::string kpoints_text =
        read_file(call_dir / "full_kpoints.txt");
    assert(kpoints_text.find("# index kx ky kz\n") == 0);
    assert(count_data_rows(kpoints_text) == 2);
    assert(kpoints_text.find("1 0.5 0 0\n") != std::string::npos);

    const std::string translations_text =
        read_file(call_dir / "translations.txt");
    assert(translations_text.find("# index R_x R_y R_z\n") == 0);
    assert(count_data_rows(translations_text) == 2);
    assert(translations_text.find("1 1 0 0\n") != std::string::npos);

    const std::string remap_text = read_file(call_dir / "bvk_remap.txt");
    assert(remap_text.find("# atom_i atom_j source_R_x source_R_y source_R_z "
                           "target_index target_count target_R_x target_R_y "
                           "target_R_z\n") == 0);
    assert(count_data_rows(remap_text) == 1);
    assert(remap_text.find("0 1 1 0 0 0 1 -2 0 0\n") !=
           std::string::npos);

    const std::string density_text = read_file(call_dir / "density_delta_k.txt");
    assert(density_text.find("# kpoint row column real imag\n") == 0);
    assert(count_data_rows(density_text) == 2 * 16);

    const std::string hartree_k_text = read_file(call_dir / "hartree_k.txt");
    assert(count_data_rows(hartree_k_text) == 2 * 2 * 2 * 4);

    const std::string hartree_r_text = read_file(call_dir / "hartree_r.txt");
    assert(count_data_rows(hartree_r_text) == 2 * 2 * 2 * 4);

    // exact 17-digit roundtrip of one density element
    {
        std::istringstream lines(density_text);
        std::string line;
        bool checked = false;
        while (std::getline(lines, line))
        {
            if (line.empty() || line[0] == '#')
            {
                continue;
            }
            std::istringstream fields(line);
            int kpoint, row, column;
            double real, imag;
            fields >> kpoint >> row >> column >> real >> imag;
            if (kpoint == 0 && row == 0 && column == 0)
            {
                assert(std::abs(real - density.at(0)(0, 0).real()) < 1.0e-17);
                assert(std::abs(imag - density.at(0)(0, 0).imag()) < 1.0e-17);
                checked = true;
            }
        }
        assert(checked);
    }

    // a second dump appends call_002 without touching call_001
    maybe_dump_hartree_pipeline(make_static_data(), density,
                                make_hartree_k(), make_hartree_r());
    assert(std::filesystem::is_directory(
        std::filesystem::path(dump_root) / "call_002"));
    assert(std::filesystem::is_directory(call_dir));

    std::filesystem::remove_all(dump_root);
#ifdef _WIN32
    _putenv("LIBRPA_QSGW_HARTREE_DUMP_DIR=");
#else
    unsetenv("LIBRPA_QSGW_HARTREE_DUMP_DIR");
#endif
}

} // namespace

int main()
{
    test_noop_without_env();
    test_dump_schema_and_roundtrip();
    std::cout << "test_qsgw_hartree_dump: all tests passed\n";
    return 0;
}
