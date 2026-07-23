#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../qsgw/input_contract.h"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

using librpa_int::qsgw::BandUpdateMode;
using librpa_int::qsgw::HartreeUpdateMode;
using librpa_int::qsgw::HeadwingGridMode;
using librpa_int::qsgw::HeadwingUpdateMode;
using librpa_int::qsgw::IndependentHeadwingPaths;
using librpa_int::qsgw::QsgwInputContract;
using librpa_int::qsgw::QsgwProducer;
using librpa_int::qsgw::resolve_band_reference_paths;
using librpa_int::qsgw::resolve_independent_headwing_paths;
using librpa_int::qsgw::resolve_same_grid_velocity_paths;
using librpa_int::qsgw::validate_band_reference_binding;
using librpa_int::qsgw::validate_hartree_input_binding;
using librpa_int::qsgw::validate_independent_headwing_binding;
using librpa_int::qsgw::validate_qsgw_execution_modes;
using librpa_int::qsgw::validate_scf_input_binding;

namespace
{

constexpr const char* abc_sha256 =
    "ba7816bf8f01cfea414140de5dae2223"
    "b00361a396177a9cb410ff61f20015ad";

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

std::string abacus_independent_contract()
{
    std::ostringstream text;
    text << "# librpa-qsgw-input-contract-v1\n"
         << "producer\tabacus\n"
         << "internal_energy_units\thartree\n"
         << "mf0_basis\tstate_coefficients_in_nao\n"
         << "mf0_gauge\tproducer_state\n"
         << "n_spins\t1\n"
         << "n_bands\t8\n"
         << "n_aos\t8\n"
         << "n_scf_kpoints\t10\n"
         << "n_headwing_kpoints\t64\n"
         << "n_band_kpoints\t0\n"
         << "headwing_grid\tindependent_full\n"
         << "headwing_update\tlive_ao_fourier\n"
         << "hartree_update\toff\n"
         << "band_update\toff\n"
         << "role\tsha256\tfile\n";
    for (const char* role : {
             "mf0_eigenvalues", "mf0_wavefunctions", "scf_kpoints",
             "vxc_scf_manifest", "reader_static", "headwing_mf0_eigenvalues",
             "headwing_mf0_wavefunctions", "headwing_kpoints",
             "headwing_velocity_mf0"})
    {
        text << role << '\t' << abc_sha256 << '\t' << role << ".dat\n";
    }
    return text.str();
}

std::string aims_same_grid_contract()
{
    std::ostringstream text;
    text << "# librpa-qsgw-input-contract-v1\n"
         << "producer\tfhi-aims\n"
         << "internal_energy_units\thartree\n"
         << "mf0_basis\tstate_coefficients_in_nao\n"
         << "mf0_gauge\tproducer_state\n"
         << "n_spins\t1\n"
         << "n_bands\t12\n"
         << "n_aos\t12\n"
         << "n_scf_kpoints\t4\n"
         << "n_headwing_kpoints\t4\n"
         << "n_band_kpoints\t3\n"
         << "headwing_grid\tscf\n"
         << "headwing_update\tfixed_basis_rotation\n"
         << "hartree_update\tdelta_density\n"
         << "band_update\tfixed_basis_rotation\n"
         << "role\tsha256\tfile\n";
    for (const char* role : {
             "mf0_eigenvalues", "mf0_wavefunctions", "scf_kpoints",
             "vxc_scf_manifest", "reader_static", "velocity_mf0",
             "hartree_ri_coefficients", "hartree_coulomb",
             "hartree_aux_basis"})
    {
        text << role << '\t' << abc_sha256 << '\t' << role << ".dat\n";
    }
    text << "band_kpoints\t" << abc_sha256
         << "\tband_kpath_info\n"
         << "vxc_band_manifest\t" << abc_sha256
         << "\tqsgw_vxc_band.manifest\n";
    for (int kpoint = 1; kpoint <= 3; ++kpoint)
    {
        std::ostringstream index;
        index << std::setw(5) << std::setfill('0') << kpoint;
        text << "band_mf0_eigenvalues\t" << abc_sha256
             << "\tband_KS_eigenvalue_k_" << index.str() << ".txt\n"
             << "band_mf0_wavefunctions\t" << abc_sha256
             << "\tband_KS_eigenvector_k_" << index.str() << ".txt\n";
    }
    return text.str();
}

std::string abacus_bound_independent_contract()
{
    std::ostringstream text;
    text << "# librpa-qsgw-input-contract-v1\n"
         << "producer\tabacus\n"
         << "internal_energy_units\thartree\n"
         << "mf0_basis\tstate_coefficients_in_nao\n"
         << "mf0_gauge\tproducer_state\n"
         << "n_spins\t1\n"
         << "n_bands\t2\n"
         << "n_aos\t2\n"
         << "n_scf_kpoints\t2\n"
         << "n_headwing_kpoints\t3\n"
         << "n_band_kpoints\t0\n"
         << "headwing_grid\tindependent_full\n"
         << "headwing_update\tlive_ao_fourier\n"
         << "hartree_update\toff\n"
         << "band_update\toff\n"
         << "role\tsha256\tfile\n";
    for (const char* role : {
             "mf0_eigenvalues", "mf0_wavefunctions", "scf_kpoints",
             "vxc_scf_manifest", "reader_static"})
    {
        text << role << '\t' << abc_sha256 << '\t' << role << ".dat\n";
    }
    text << "headwing_kpoints\t" << abc_sha256
         << "\tpyatb_librpa_df/k_path_info\n"
         << "headwing_mf0_eigenvalues\t" << abc_sha256
         << "\tpyatb_librpa_df/band_out\n";
    for (int kpoint = 0; kpoint < 3; ++kpoint)
    {
        text << "headwing_mf0_wavefunctions\t" << abc_sha256
             << "\tpyatb_librpa_df/KS_eigenvector_" << kpoint
             << ".dat\n";
    }
    text << "headwing_velocity_mf0\t" << abc_sha256
         << "\tpyatb_librpa_df/velocity_matrix\n";
    return text.str();
}

void test_abacus_independent_grid_requires_live_operator_update()
{
    std::istringstream input(abacus_independent_contract());
    const QsgwInputContract contract =
        QsgwInputContract::parse(input, "abacus-contract");
    assert(contract.producer() == QsgwProducer::Abacus);
    assert(contract.headwing_grid() == HeadwingGridMode::IndependentFullGrid);
    assert(contract.headwing_update() == HeadwingUpdateMode::LiveAoFourier);
    assert(contract.hartree_update() == HartreeUpdateMode::Off);
    assert(contract.n_scf_kpoints() == 10);
    assert(contract.n_headwing_kpoints() == 64);

    std::string frozen = abacus_independent_contract();
    const auto position = frozen.find("headwing_update\tlive_ao_fourier");
    frozen.replace(position,
                   std::string("headwing_update\tlive_ao_fourier").size(),
                   "headwing_update\tfixed_basis_rotation");
    std::istringstream frozen_input(frozen);
    assert_throws([&] {
        (void)QsgwInputContract::parse(frozen_input, "frozen-full-grid");
    });
}

void test_aims_same_grid_hartree_and_band_contract()
{
    std::istringstream input(aims_same_grid_contract());
    const QsgwInputContract contract =
        QsgwInputContract::parse(input, "aims-contract");
    assert(contract.producer() == QsgwProducer::FhiAims);
    assert(contract.headwing_grid() == HeadwingGridMode::ScfGrid);
    assert(contract.headwing_update() ==
           HeadwingUpdateMode::FixedBasisRotation);
    assert(contract.hartree_update() == HartreeUpdateMode::DeltaDensity);
    assert(contract.band_update() == BandUpdateMode::FixedBasisRotation);
    assert(contract.files("hartree_ri_coefficients").size() == 1);
    assert(contract.files("hartree_coulomb").size() == 1);

    std::string obsolete = aims_same_grid_contract();
    const auto position =
        obsolete.find("band_update\tfixed_basis_rotation");
    obsolete.replace(
        position,
        std::string("band_update\tfixed_basis_rotation").size(),
        "band_update\toperator_fourier");
    std::istringstream obsolete_input(obsolete);
    assert_throws([&] {
        (void)QsgwInputContract::parse(
            obsolete_input, "obsolete-band-operator-fourier");
    });
}

void test_execution_modes_must_match_the_input_contract()
{
    std::istringstream input(aims_same_grid_contract());
    const QsgwInputContract contract =
        QsgwInputContract::parse(input, "aims-execution-contract");

    validate_qsgw_execution_modes(
        contract, HeadwingGridMode::ScfGrid, true, true);
    assert_throws([&] {
        validate_qsgw_execution_modes(
            contract, HeadwingGridMode::Disabled, true, true);
    });
    assert_throws([&] {
        validate_qsgw_execution_modes(
            contract, HeadwingGridMode::ScfGrid, false, true);
    });
    assert_throws([&] {
        validate_qsgw_execution_modes(
            contract, HeadwingGridMode::ScfGrid, true, false);
    });

    std::istringstream independent_input(abacus_independent_contract());
    const QsgwInputContract independent = QsgwInputContract::parse(
        independent_input, "independent-execution-contract");
    validate_qsgw_execution_modes(
        independent, HeadwingGridMode::IndependentFullGrid, false, false);
    assert_throws([&] {
        validate_qsgw_execution_modes(
            independent, HeadwingGridMode::ScfGrid, false, false);
    });
}

void test_hartree_binding_requires_the_exact_reader_file_sets()
{
    const std::filesystem::path root =
        std::filesystem::absolute("test_qsgw_hartree_binding.tmp")
            .lexically_normal();
    std::istringstream input(aims_same_grid_contract());
    const QsgwInputContract contract =
        QsgwInputContract::parse(input, "hartree-binding-contract");

    validate_hartree_input_binding(
        contract, root.string(),
        {root / "hartree_ri_coefficients.dat"},
        {root / "hartree_coulomb.dat"},
        {root / "hartree_aux_basis.dat"});

    assert_throws([&] {
        validate_hartree_input_binding(
            contract, root.string(),
            {root / "hartree_ri_coefficients.dat"},
            {root / "wrong_coulomb.dat"},
            {root / "hartree_aux_basis.dat"});
    });
    assert_throws([&] {
        validate_hartree_input_binding(
            contract, root.string(),
            {root / "hartree_ri_coefficients.dat",
             root / "extra_ri_coefficients.dat"},
            {root / "hartree_coulomb.dat"},
            {root / "hartree_aux_basis.dat"});
    });
}

void test_hartree_binding_accepts_auxiliary_basis_inferred_from_ri_files()
{
    const std::filesystem::path root =
        std::filesystem::absolute("test_qsgw_hartree_inferred_basis.tmp")
            .lexically_normal();
    std::string text = aims_same_grid_contract();
    const std::string ri_row =
        std::string("hartree_ri_coefficients\t") + abc_sha256 +
        "\thartree_ri_coefficients.dat\n";
    text.replace(text.find(ri_row), ri_row.size(),
                 ri_row + std::string("hartree_ri_coefficients\t") +
                     abc_sha256 + "\thartree_ri_coefficients_2.dat\n");
    const std::string aux_row =
        std::string("hartree_aux_basis\t") + abc_sha256 +
        "\thartree_aux_basis.dat\n";
    text.replace(text.find(aux_row), aux_row.size(),
                 std::string("hartree_aux_basis\t") + abc_sha256 +
                     "\thartree_ri_coefficients.dat\n" +
                     "hartree_aux_basis\t" + abc_sha256 +
                     "\thartree_ri_coefficients_2.dat\n");

    std::istringstream input(text);
    const QsgwInputContract contract =
        QsgwInputContract::parse(input, "inferred-basis-contract");
    const std::vector<std::filesystem::path> ri_files{
        root / "hartree_ri_coefficients.dat",
        root / "hartree_ri_coefficients_2.dat"};
    validate_hartree_input_binding(
        contract, root.string(), ri_files,
        {root / "hartree_coulomb.dat"}, ri_files);
}

void test_contract_rejects_missing_roles_bad_dimensions_and_unsafe_paths()
{
    std::string missing = abacus_independent_contract();
    const auto begin = missing.find("headwing_velocity_mf0\t");
    missing.erase(begin, missing.find('\n', begin) - begin + 1);
    std::istringstream missing_input(missing);
    assert_throws([&] {
        (void)QsgwInputContract::parse(missing_input, "missing-role");
    });

    std::string dimensions = abacus_independent_contract();
    const auto dimension = dimensions.find("n_bands\t8");
    dimensions.replace(dimension, std::string("n_bands\t8").size(),
                       "n_bands\t0");
    std::istringstream dimension_input(dimensions);
    assert_throws([&] {
        (void)QsgwInputContract::parse(dimension_input, "bad-dimension");
    });

    std::string unsafe = abacus_independent_contract();
    const auto path = unsafe.find("mf0_eigenvalues.dat");
    unsafe.replace(path, std::string("mf0_eigenvalues.dat").size(),
                   "../mf0_eigenvalues.dat");
    std::istringstream unsafe_input(unsafe);
    assert_throws([&] {
        (void)QsgwInputContract::parse(unsafe_input, "unsafe-path");
    });
}

void test_contract_verifies_every_static_input_hash()
{
    const std::string path = "test_qsgw_input_contract.tmp";
    {
        std::ofstream output(path, std::ios::binary);
        output << "abc";
    }
    std::string text = abacus_independent_contract();
    std::size_t position = 0;
    while ((position = text.find(".dat", position)) != std::string::npos)
    {
        const std::size_t line_begin = text.rfind('\t', position);
        text.replace(line_begin + 1, position + 4 - line_begin - 1, path);
        position = line_begin + 1 + path.size();
    }
    std::istringstream input(text);
    const QsgwInputContract contract =
        QsgwInputContract::parse(input, "hash-contract");
    contract.validate_file_hashes(".");
    {
        std::ofstream output(path, std::ios::binary | std::ios::app);
        output << "changed";
    }
    assert_throws([&] { contract.validate_file_hashes("."); });
    std::remove(path.c_str());
}

void test_scf_binding_requires_the_exact_reader_file_sets()
{
    const std::filesystem::path root =
        std::filesystem::absolute("test_qsgw_scf_binding.tmp")
            .lexically_normal();
    std::istringstream input(abacus_independent_contract());
    const QsgwInputContract contract =
        QsgwInputContract::parse(input, "scf-binding-contract");

    validate_scf_input_binding(
        contract, root.string(), root / "mf0_eigenvalues.dat",
        {root / "mf0_wavefunctions.dat"}, root / "scf_kpoints.dat",
        {root / "reader_static.dat"});

    assert_throws([&] {
        validate_scf_input_binding(
            contract, root.string(), root / "mf0_eigenvalues.dat",
            {root / "wrong_wavefunctions.dat"}, root / "scf_kpoints.dat",
            {root / "reader_static.dat"});
    });
    assert_throws([&] {
        validate_scf_input_binding(
            contract, root.string(), root / "mf0_eigenvalues.dat",
            {root / "mf0_wavefunctions.dat"}, root / "scf_kpoints.dat",
            {root / "reader_static.dat", root / "unbound_static.dat"});
    });
}

void test_same_grid_velocity_paths_follow_reader_precedence()
{
    const std::filesystem::path root =
        "test_qsgw_same_grid_velocity_paths.tmp";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "pyatb_librpa_df");
    {
        std::ofstream output(root / "velocity_matrix");
        output << "root";
    }
    {
        std::ofstream output(root / "pyatb_librpa_df" / "velocity_matrix");
        output << "pyatb";
    }
    {
        std::ofstream output(root / "pyatb_librpa_df" / "k_path_info");
        output << "4 4 1 3\n";
    }
    for (const char* name : {
             "band_out", "KS_eigenvector_0.dat",
             "KS_eigenvector_1.dat", "KS_eigenvector_2.dat"})
    {
        std::ofstream output(root / "pyatb_librpa_df" / name);
        output << name;
    }

    const auto abacus =
        resolve_same_grid_velocity_paths(QsgwProducer::Abacus,
                                         root.string(), 2);
    assert(abacus.size() == 6);
    const std::filesystem::path pyatb =
        std::filesystem::absolute(root / "pyatb_librpa_df")
            .lexically_normal();
    assert(abacus[0] == pyatb / "k_path_info");
    assert(abacus[1] == pyatb / "band_out");
    assert(abacus[2] == pyatb / "KS_eigenvector_0.dat");
    assert(abacus[3] == pyatb / "KS_eigenvector_1.dat");
    assert(abacus[4] == pyatb / "KS_eigenvector_2.dat");
    assert(abacus[5] == pyatb / "velocity_matrix");

    std::filesystem::remove(root / "pyatb_librpa_df" / "velocity_matrix");
    const auto abacus_root =
        resolve_same_grid_velocity_paths(QsgwProducer::Abacus,
                                         root.string(), 4);
    assert(abacus_root.size() == 1);
    assert(abacus_root.front() ==
           std::filesystem::absolute(root / "velocity_matrix")
               .lexically_normal());

    const auto aims =
        resolve_same_grid_velocity_paths(QsgwProducer::FhiAims,
                                         root.string(), 2);
    assert(aims.size() == 2);
    assert(aims.front().filename() == "mommat_ks_kpt_000001.dat");
    assert(aims.back().filename() == "mommat_ks_kpt_000002.dat");

    std::filesystem::remove_all(root);
}

void test_band_reference_paths_match_the_driver_reader_names()
{
    const std::filesystem::path root =
        "test_qsgw_band_reference_paths.tmp";
    const auto paths = resolve_band_reference_paths(
        root.string(), "custom_band_kpath_info", 3);
    const std::filesystem::path absolute_root =
        std::filesystem::absolute(root).lexically_normal();

    assert(paths.kpoints == absolute_root / "custom_band_kpath_info");
    assert(paths.eigenvalues.size() == 3);
    assert(paths.wavefunctions.size() == 3);
    assert(paths.eigenvalues.front() ==
           absolute_root / "band_KS_eigenvalue_k_00001.txt");
    assert(paths.eigenvalues.back() ==
           absolute_root / "band_KS_eigenvalue_k_00003.txt");
    assert(paths.wavefunctions.front() ==
           absolute_root / "band_KS_eigenvector_k_00001.txt");
    assert(paths.wavefunctions.back() ==
           absolute_root / "band_KS_eigenvector_k_00003.txt");

    assert_throws([&] {
        (void)resolve_band_reference_paths(root.string(),
                                           "custom_band_kpath_info", 0);
    });
}

void test_band_reference_binding_requires_the_exact_reader_file_set()
{
    const std::filesystem::path root =
        "test_qsgw_band_reference_binding.tmp";
    std::istringstream input(aims_same_grid_contract());
    const QsgwInputContract contract =
        QsgwInputContract::parse(input, "bound-band-contract");
    validate_band_reference_binding(
        contract, root.string(), root.string(), "band_kpath_info");

    std::string missing = aims_same_grid_contract();
    const std::string missing_line =
        std::string("band_mf0_wavefunctions\t") + abc_sha256 +
        "\tband_KS_eigenvector_k_00002.txt\n";
    missing.erase(missing.find(missing_line), missing_line.size());
    std::istringstream missing_input(missing);
    const QsgwInputContract missing_contract =
        QsgwInputContract::parse(missing_input, "missing-band-reader-file");
    assert_throws([&] {
        validate_band_reference_binding(
            missing_contract, root.string(), root.string(),
            "band_kpath_info");
    });

    std::string extra = aims_same_grid_contract();
    extra += std::string("band_mf0_eigenvalues\t") + abc_sha256 +
             "\tband_KS_eigenvalue_k_00004.txt\n";
    std::istringstream extra_input(extra);
    const QsgwInputContract extra_contract =
        QsgwInputContract::parse(extra_input, "extra-band-reader-file");
    assert_throws([&] {
        validate_band_reference_binding(
            extra_contract, root.string(), root.string(),
            "band_kpath_info");
    });
}

void test_independent_headwing_binding_matches_exact_reader_files()
{
    const std::filesystem::path root =
        "test_qsgw_independent_headwing_binding.tmp";
    const IndependentHeadwingPaths paths =
        resolve_independent_headwing_paths(root.string(), 3);
    const std::filesystem::path expected_directory =
        std::filesystem::absolute(root / "pyatb_librpa_df")
            .lexically_normal();
    assert(paths.directory == expected_directory);
    assert(paths.kpoints == expected_directory / "k_path_info");
    assert(paths.eigenvalues == expected_directory / "band_out");
    assert(paths.wavefunctions.size() == 3);
    assert(paths.wavefunctions.front() ==
           expected_directory / "KS_eigenvector_0.dat");
    assert(paths.wavefunctions.back() ==
           expected_directory / "KS_eigenvector_2.dat");
    assert(paths.velocity == expected_directory / "velocity_matrix");

    std::istringstream input(abacus_bound_independent_contract());
    const QsgwInputContract contract = QsgwInputContract::parse(
        input, "bound-independent-headwing-contract");
    validate_independent_headwing_binding(
        contract, root.string(), root.string());

    std::string missing = abacus_bound_independent_contract();
    const std::string missing_line =
        std::string("headwing_mf0_wavefunctions\t") + abc_sha256 +
        "\tpyatb_librpa_df/KS_eigenvector_1.dat\n";
    missing.erase(missing.find(missing_line), missing_line.size());
    std::istringstream missing_input(missing);
    const QsgwInputContract missing_contract = QsgwInputContract::parse(
        missing_input, "missing-independent-headwing-file");
    assert_throws([&] {
        validate_independent_headwing_binding(
            missing_contract, root.string(), root.string());
    });

    std::string wrong = abacus_bound_independent_contract();
    const auto wrong_path = wrong.find(
        "pyatb_librpa_df/velocity_matrix");
    wrong.replace(
        wrong_path,
        std::string("pyatb_librpa_df/velocity_matrix").size(),
        "velocity_matrix");
    std::istringstream wrong_input(wrong);
    const QsgwInputContract wrong_contract = QsgwInputContract::parse(
        wrong_input, "wrong-independent-headwing-file");
    assert_throws([&] {
        validate_independent_headwing_binding(
            wrong_contract, root.string(), root.string());
    });

    assert_throws([&] {
        (void)resolve_independent_headwing_paths(root.string(), 0);
    });
}

} // namespace

int main()
{
    test_abacus_independent_grid_requires_live_operator_update();
    test_aims_same_grid_hartree_and_band_contract();
    test_execution_modes_must_match_the_input_contract();
    test_hartree_binding_requires_the_exact_reader_file_sets();
    test_hartree_binding_accepts_auxiliary_basis_inferred_from_ri_files();
    test_contract_rejects_missing_roles_bad_dimensions_and_unsafe_paths();
    test_contract_verifies_every_static_input_hash();
    test_scf_binding_requires_the_exact_reader_file_sets();
    test_same_grid_velocity_paths_follow_reader_precedence();
    test_band_reference_paths_match_the_driver_reader_names();
    test_band_reference_binding_requires_the_exact_reader_file_set();
    test_independent_headwing_binding_matches_exact_reader_files();
    std::cout << "test_qsgw_input_contract: all tests passed\n";
    return 0;
}
