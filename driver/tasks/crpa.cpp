#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "../../src/io/fs.h"
#include "../../src/io/global_io.h"
#include "../../src/mpi/global_mpi.h"
#include "../../src/utils/constants.h"
#include "../driver.h"
#include "../reader_nao_overlap.h"
#include "../task.h"
#include "librpa_crpa.h"

namespace
{

std::ofstream output_file(const std::string& name)
{
    std::ofstream output(librpa_int::join_path(driver::opts.output_dir, name));
    if (!output) throw std::runtime_error("cannot create cRPA output " + name);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output << std::setprecision(16);
    return output;
}

std::array<double, 3> orbital_averages(const double* packed, int n)
{
    const auto real = [=](int a, int b, int c, int d)
    { return packed[2 * ((a * n + b) * n * n + c * n + d)] * librpa_int::HA2EV; };
    std::array<double, 3> averages{};
    for (int m = 0; m < n; ++m)
    {
        averages[0] += real(m, m, m, m) / n;
        for (int other = 0; other < n; ++other)
            if (other != m)
            {
                averages[1] += real(m, m, other, other) / (n * (n - 1));
                averages[2] += real(m, other, other, m) / (n * (n - 1));
            }
    }
    return averages;
}

void write_result(const LibrpaCrpaResult* result)
{
    auto metadata = output_file("crpa_metadata.txt");
    metadata << "Original KS poles; P_d selects A on both occupied and empty branches.\n"
             << "Output Phi uses independent B window and separate physical spin frames.\n"
             << "First radial shell; eg/t2g labels refer to global Cartesian axes.\n"
             << "Native positive minimax nodes only; no direct U(0) was calculated.\n"
             << "Interaction and frequency columns use eV; indices are zero based.\n"
             << "U_intra=mean U_mm,mm; U_inter=mean_(m!=n) U_mm,nn; J=mean U_mn,nm.\n"
             << driver::driver_params.format();
    auto windows = output_file("crpa_windows.dat");
    windows << "# window spin k count original_KS_band_indices_zero_based\n";
    for (int response : {1, 0})
        for (int spin = 0; spin < driver::n_spins; ++spin)
            for (int k = 0; k < driver::n_kpoints; ++k)
            {
                int count = 0;
                const int* bands = nullptr;
                librpa_crpa_result_bands(result, response, spin, k, &count, &bands);
                windows << (response ? 'A' : 'B') << ' ' << spin << ' ' << k << ' ' << count;
                for (int i = 0; i < count; ++i) windows << ' ' << bands[i];
                windows << '\n';
            }
    auto tensors = output_file("crpa_tensors.dat");
    tensors << "# atom spin_left spin_right nu_eV a b c d V_re V_im U_re U_im W_re W_im\n";
    auto summary = output_file("crpa_summary.dat");
    summary << "# atom spin_left spin_right n_orbitals nu_eV V_intra V_inter J_bare "
               "U_intra U_inter J_U W_intra W_inter J_W\n";
    for (int record = 0; record < librpa_crpa_result_size(result); ++record)
    {
        LibrpaCrpaTensor tensor;
        librpa_crpa_result_tensor(result, record, &tensor);
        const int n = tensor.n_orbitals;
        const double frequency = tensor.frequency_ha * librpa_int::HA2EV;
        summary << tensor.atom_index << ' ' << tensor.spin_left << ' ' << tensor.spin_right << ' '
                << n << ' ' << frequency;
        for (const auto* values :
             {tensor.bare_ri, tensor.partially_screened_ri, tensor.fully_screened_ri})
            for (const double average : orbital_averages(values, n)) summary << ' ' << average;
        summary << '\n';
        for (int a = 0; a < n; ++a)
            for (int b = 0; b < n; ++b)
                for (int c = 0; c < n; ++c)
                    for (int d = 0; d < n; ++d)
                    {
                        const int index = 2 * ((a * n + b) * n * n + c * n + d);
                        tensors << tensor.atom_index << ' ' << tensor.spin_left << ' '
                                << tensor.spin_right << ' ' << frequency << ' ' << a << ' ' << b
                                << ' ' << c << ' ' << d;
                        for (const auto* values : {tensor.bare_ri, tensor.partially_screened_ri,
                                                   tensor.fully_screened_ri})
                            tensors << ' ' << values[index] * librpa_int::HA2EV << ' '
                                    << values[index + 1] * librpa_int::HA2EV;
                        tensors << '\n';
                    }
    }
}

}  // namespace

void driver::task_crpa_u()
{
    using librpa_int::global::mpi_comm_global_h;
    LibrpaCrpaInput input;
    librpa_init_crpa_input(&input);
    std::vector<std::string> species;
    std::vector<const char*> labels;
    std::vector<double> overlap_packed;
    std::string error;
    try
    {
        std::string species_text = driver_params.crpa_species_labels;
        std::replace(species_text.begin(), species_text.end(), ',', ' ');
        std::istringstream stream(species_text);
        std::string label;
        while (stream >> label) species.push_back(label);
        for (const auto& name : species) labels.push_back(name.c_str());
        const auto filename =
            librpa_int::is_absolute_path(driver_params.crpa_overlap_file)
                ? driver_params.crpa_overlap_file
                : librpa_int::join_path(driver_params.input_dir, driver_params.crpa_overlap_file);
        const auto overlap = librpa_driver::read_abacus_nao_overlap_csr(filename);
        std::vector<double> kfrac(3 * n_kpoints);
        librpa_get_crpa_kgrid(h.get_c_handler(), n_kpoints, kfrac.data());
        std::vector<librpa_int::Vector3_Order<double>> kpoints;
        for (int k = 0; k < n_kpoints; ++k)
            kpoints.push_back({kfrac[3 * k], kfrac[3 * k + 1], kfrac[3 * k + 2]});
        const auto overlap_k = librpa_driver::fourier_nao_overlap(overlap, kpoints);
        if (overlap.dimension != n_basis_ao)
            throw std::runtime_error("cRPA overlap and loaded AO dimensions differ");
        for (const auto& matrix : overlap_k)
            for (int i = 0; i < matrix.size; ++i)
            {
                overlap_packed.push_back(matrix.c[i].real());
                overlap_packed.push_back(matrix.c[i].imag());
            }
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
    }
    const int failed = !error.empty();
    int any_failed = 0;
    mpi_comm_global_h.allreduce(&failed, &any_failed, 1, MPI_MAX);
    if (any_failed)
        throw std::runtime_error("cRPA native input: " +
                                 (error.empty() ? "another MPI rank failed" : error));

    input.n_species = static_cast<int>(labels.size());
    input.species_labels = labels.data();
    input.correlated_species = driver_params.crpa_correlated_species.c_str();
    input.ligand_species = driver_params.crpa_ligand_species.c_str();
    input.parent_orbitals = driver_params.crpa_parent_orbitals.c_str();
    input.output_orbitals = driver_params.crpa_output_orbitals.c_str();
    input.n_response_edges = static_cast<int>(driver_params.crpa_response_windows_ha.size());
    input.response_windows_ha = driver_params.crpa_response_windows_ha.data();
    input.n_orbital_edges = static_cast<int>(driver_params.crpa_orbital_windows_ha.size());
    input.orbital_windows_ha = driver_params.crpa_orbital_windows_ha.data();
    input.n_response_bands = static_cast<int>(driver_params.crpa_response_bands.size());
    input.response_bands = driver_params.crpa_response_bands.data();
    input.n_orbital_bands = static_cast<int>(driver_params.crpa_orbital_bands.size());
    input.orbital_bands = driver_params.crpa_orbital_bands.data();
    input.n_kpoints = n_kpoints;
    input.n_aos = n_basis_ao;
    input.overlap_k_ri = overlap_packed.data();
    input.residual_tol = driver_params.crpa_residual_tol;
    std::unique_ptr<LibrpaCrpaResult, decltype(&librpa_delete_crpa_result)> result(
        librpa_compute_crpa_window(h.get_c_handler(), &opts, &input), librpa_delete_crpa_result);
    if (mpi_comm_global_h.is_root())
    {
        write_result(result.get());
        librpa_int::global::lib_printf("cRPA v/U/W tensors and window audit written to %s\n",
                                       opts.output_dir);
    }
}
