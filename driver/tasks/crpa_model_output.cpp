#include "crpa_model_output.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <vector>

#include "../../src/io/npz_writer.h"
#include "../../src/utils/constants.h"
#include "../driver.h"

namespace driver
{
namespace
{
std::string quoted(const std::string& value)
{
    std::string output = "\"";
    for (char c : value)
    {
        if (c == '\\' || c == '"') output += '\\';
        if (c == '\n')
            output += "\\n";
        else
            output += c;
    }
    return output + "\"";
}

void write_sites(librpa_int::NpzWriter& writer, const LibrpaCrpaResult* result,
                 const LibrpaCrpaModelInfo& info, bool parent)
{
    const int count = parent ? info.n_parent_sites : info.n_output_sites;
    std::vector<int> sites;
    std::ostringstream labels;
    labels << '[';
    bool first = true;
    for (int site = 0; site < count; ++site)
    {
        int atom, start, n;
        librpa_crpa_model_site(result, parent, site, &atom, &start, &n);
        sites.insert(sites.end(), {atom, start, n});
        for (int orbital = 0; orbital < n; ++orbital)
        {
            if (!first) labels << ',';
            first = false;
            labels << quoted(librpa_crpa_model_label(result, parent, site, orbital));
        }
    }
    labels << ']';
    const std::string prefix = parent ? "parent" : "output";
    writer.array(prefix + "_sites", "<i4", {static_cast<std::size_t>(count), 3}, sites.data(), 4);
    writer.text(prefix + "_labels_json", labels.str());
}
}  // namespace

void write_crpa_model(const LibrpaCrpaResult* result, const std::string& directory, int state_index,
                      int gw_update, bool final_state)
{
    LibrpaCrpaModelInfo info;
    librpa_crpa_model_info(result, &info);
    const auto size = [](int n) { return static_cast<std::size_t>(n); };
    if (!std::isfinite(driver_params.crpa_fit_tol_ev) || driver_params.crpa_fit_tol_ev <= 0 ||
        !std::isfinite(driver_params.crpa_eta_ev) || driver_params.crpa_eta_ev <= 0)
        throw std::runtime_error("cRPA fit tolerance and broadening must be positive");
    std::vector<double> queries;
    if (driver_params.crpa_output_static) queries.insert(queries.end(), {0, 0});
    for (double frequency : driver_params.crpa_real_frequencies_ev)
        queries.insert(queries.end(), {frequency / librpa_int::HA2EV,
                                       driver_params.crpa_eta_ev / librpa_int::HA2EV});
    std::unique_ptr<LibrpaCrpaContinuedResult, decltype(&librpa_delete_crpa_continued_result)>
        continued(nullptr, librpa_delete_crpa_continued_result);
    if (!queries.empty())
        continued.reset(librpa_continue_crpa(result, queries.size() / 2, queries.data(), -1,
                                             driver_params.crpa_fit_tol_ev / librpa_int::HA2EV));
    std::ofstream local(std::filesystem::path(directory) / "crpa_local_model.dat");
    local.precision(17);
    local << "# parent spin atom a b Hloc_re_eV Hloc_im_eV density_re density_im\n";
    for (bool parent : {true, false})
        for (int site = 0; site < (parent ? info.n_parent_sites : info.n_output_sites); ++site)
        {
            int atom, start, n;
            librpa_crpa_model_site(result, parent, site, &atom, &start, &n);
            std::vector<double> h(2 * n * n), density(h.size());
            for (int spin = 0; spin < info.n_spins; ++spin)
            {
                librpa_crpa_model_local(result, parent, spin, site, h.data(), density.data());
                for (int a = 0; a < n; ++a)
                    for (int b = 0; b < n; ++b)
                    {
                        const int i = 2 * (a * n + b);
                        local << parent << ' ' << spin << ' ' << atom << ' ' << a << ' ' << b << ' '
                              << h[i] * librpa_int::HA2EV << ' ' << h[i + 1] * librpa_int::HA2EV
                              << ' ' << density[i] << ' ' << density[i + 1] << '\n';
                    }
            }
        }
    if (!local) throw std::runtime_error("cannot write local cRPA model");
    if (continued)
    {
        std::ofstream summary(std::filesystem::path(directory) / "crpa_continuation_summary.dat");
        summary.precision(17);
        summary
            << "# Native Pade of U-v in z=i*nu; z=0 is extrapolated, not a response node.\n"
            << "# Numerical diagnostics only; no causality or material-convergence certificate.\n"
            << "# atom spin_left spin_right omega_eV eta_eV U_intra U_inter J_U max_spread_eV "
               "source_residual_eV holdout_residual_eV flagged_entries status_bits\n";
        for (int tensor = 0; tensor < librpa_crpa_continued_size(continued.get()); ++tensor)
            for (int query = 0; query < queries.size() / 2; ++query)
            {
                LibrpaCrpaContinuedTensor view;
                librpa_crpa_continued_tensor(continued.get(), tensor, query, &view);
                const int n = view.n_orbitals, entries = n * n * n * n;
                int flags = 0, flagged = 0;
                double spread = 0;
                for (int i = 0; i < entries; ++i)
                {
                    flags |= view.status[i];
                    flagged += view.status[i] != 0;
                    spread = std::max(spread, view.subset_spread_ha[i]);
                }
                const auto at = [&](int a, int b, int c, int d)
                { return view.values_ha_ri[2 * (((a * n + b) * n + c) * n + d)]; };
                double intra = 0, inter = 0, exchange = 0;
                for (int a = 0; a < n; ++a)
                {
                    intra += at(a, a, a, a) / n;
                    for (int b = 0; b < n; ++b)
                        if (a != b)
                        {
                            inter += at(a, a, b, b) / (n * (n - 1));
                            exchange += at(a, b, b, a) / (n * (n - 1));
                        }
                }
                summary << view.atom_index << ' ' << view.spin_left << ' ' << view.spin_right << ' '
                        << queries[2 * query] * librpa_int::HA2EV << ' '
                        << queries[2 * query + 1] * librpa_int::HA2EV << ' '
                        << intra * librpa_int::HA2EV << ' ' << inter * librpa_int::HA2EV << ' '
                        << exchange * librpa_int::HA2EV << ' ' << spread * librpa_int::HA2EV << ' '
                        << view.max_source_residual_ha * librpa_int::HA2EV << ' '
                        << view.max_holdout_residual_ha * librpa_int::HA2EV << ' ' << flagged << ' '
                        << flags << '\n';
            }
        if (!summary) throw std::runtime_error("cannot write cRPA continuation summary");
    }
    if (!driver_params.crpa_output_model) return;
    const auto static_path =
        std::filesystem::absolute(std::filesystem::path(opts.output_dir) / "crpa_static.npz");
    static std::set<std::string> written_static;
    if (written_static.count(static_path.string()) == 0)
    {
        librpa_int::NpzWriter archive(static_path.string());
        archive.array("lattice_bohr", "<f8", {3, 3}, info.lattice_bohr, 8);
        archive.array("positions_bohr", "<f8", {size(info.n_atoms), 3}, info.positions_bohr, 8);
        archive.array("atom_types", "<i4", {size(info.n_atoms)}, info.atom_types, 4);
        archive.array("ao_counts", "<i4", {size(info.n_atoms)}, info.ao_counts, 4);
        archive.array("shell_offsets", "<i4", {size(info.n_atoms + 1)}, info.shell_offsets, 4);
        archive.array("shell_l", "<i4", {size(info.n_shells)}, info.shell_l, 4);
        archive.array("basis_convention", "<i4", {5}, info.basis_convention, 4);
        archive.array("k_fractional", "<f8", {size(info.n_kpoints), 3}, info.k_fractional, 8);
        archive.array("k_weights", "<f8", {size(info.n_kpoints)}, info.k_weights, 8);
        std::vector<double> overlap;
        if (driver_params.crpa_output_raw)
        {
            for (int k = 0; k < info.n_kpoints; ++k)
            {
                LibrpaCrpaModelK view;
                librpa_crpa_model_k(result, 0, k, &view);
                overlap.insert(overlap.end(), view.overlap_ri,
                               view.overlap_ri + 2 * size(info.n_aos) * info.n_aos);
            }
            archive.array("S", "<c16", {size(info.n_kpoints), size(info.n_aos), size(info.n_aos)},
                          overlap.data(), 16);
        }
        std::ostringstream species;
        species << '[';
        auto species_text = driver_params.crpa_species_labels;
        std::replace(species_text.begin(), species_text.end(), ',', ' ');
        std::istringstream input(species_text);
        std::string label;
        bool first = true;
        while (input >> label)
        {
            if (!first) species << ',';
            first = false;
            species << quoted(label);
        }
        species << ']';
        archive.text("species_json", species.str());
        archive.finish();
        written_static.insert(static_path.string());
    }
    librpa_int::NpzWriter archive((std::filesystem::path(directory) / "crpa_model.npz").string());
    std::ostringstream metadata;
    metadata
        << "{\"format_version\":1,\"energy_unit\":\"Ha\",\"frequency_axis\":\"positive_imaginary\","
        << "\"density_order\":\"ab,cd\",\"spin_frames\":\""
        << (info.n_spinor == 2 ? "joint_spinor"
                               : (info.n_spins == 2 ? "separate_collinear" : "spin_degenerate"))
        << "\","
        << "\"n_spinor\":" << info.n_spinor << ",\"fermi_ha\":";
    metadata.precision(17);
    metadata << info.fermi_ha << ",\"state_index\":" << state_index
             << ",\"gw_update\":" << gw_update
             << ",\"final_state\":" << (final_state ? "true" : "false") << ",\"static_file\":"
             << quoted(std::filesystem::relative(static_path, std::filesystem::absolute(directory))
                           .generic_string())
             << ",\"parent_orbitals\":" << quoted(driver_params.crpa_parent_orbitals)
             << ",\"output_orbitals\":" << quoted(driver_params.crpa_output_orbitals)
             << ",\"raw_saved\":" << (driver_params.crpa_output_raw ? "true" : "false")
             << ",\"static_extrapolated\":" << (driver_params.crpa_output_static ? "true" : "false")
             << ",\"fit_tol_ev\":" << driver_params.crpa_fit_tol_ev
             << ",\"ha_to_ev\":" << librpa_int::HA2EV << '}';
    archive.text("metadata_json", metadata.str());
    archive.array("output_columns", "<i4", {size(info.n_output)}, info.output_columns, 4);
    write_sites(archive, result, info, true);
    write_sites(archive, result, info, false);
    for (bool parent : {true, false})
        for (int site = 0; site < (parent ? info.n_parent_sites : info.n_output_sites); ++site)
        {
            int atom, start, n;
            librpa_crpa_model_site(result, parent, site, &atom, &start, &n);
            std::vector<double> h, density;
            h.resize(2 * size(info.n_spins) * n * n);
            density.resize(h.size());
            for (int spin = 0; spin < info.n_spins; ++spin)
                librpa_crpa_model_local(result, parent, spin, site,
                                        h.data() + 2 * size(spin) * n * n,
                                        density.data() + 2 * size(spin) * n * n);
            const auto prefix = std::string(parent ? "parent_" : "output_");
            const auto suffix = "_" + std::to_string(site);
            archive.array(prefix + "Hlocal" + suffix, "<c16",
                          {size(info.n_spins), size(n), size(n)}, h.data(), 16);
            archive.array(prefix + "density_local" + suffix, "<c16",
                          {size(info.n_spins), size(n), size(n)}, density.data(), 16);
        }

    const auto stack = [&](const std::string& name, std::vector<std::size_t> dimensions,
                           const double* LibrpaCrpaModelK::* field, bool complex)
    {
        std::size_t elements = 1;
        for (auto dim : dimensions) elements *= dim;
        const auto doubles = elements * (complex ? 2 : 1);
        std::vector<double> data;
        data.reserve(size(info.n_spins) * info.n_kpoints * doubles);
        for (int spin = 0; spin < info.n_spins; ++spin)
            for (int k = 0; k < info.n_kpoints; ++k)
            {
                LibrpaCrpaModelK view;
                librpa_crpa_model_k(result, spin, k, &view);
                const auto* values = view.*field;
                data.insert(data.end(), values, values + doubles);
            }
        dimensions.insert(dimensions.begin(), {size(info.n_spins), size(info.n_kpoints)});
        archive.array(name, complex ? "<c16" : "<f8", dimensions, data.data(), complex ? 16 : 8);
    };
    stack("E", {size(info.n_bands)}, &LibrpaCrpaModelK::energies_ha, false);
    stack("occupations", {size(info.n_bands)}, &LibrpaCrpaModelK::occupations, false);
    if (driver_params.crpa_output_raw)
        stack("C", {size(info.n_aos * info.n_spinor), size(info.n_bands)},
              &LibrpaCrpaModelK::coefficients_ri, true);
    for (bool parent : {true, false})
    {
        const auto n = size(parent ? info.n_parent : info.n_output);
        const std::string prefix = parent ? "parent_" : "output_";
        if (driver_params.crpa_output_raw)
            stack(prefix + "Phi", {size(info.n_aos * info.n_spinor), n},
                  parent ? &LibrpaCrpaModelK::parent_phi_ri : &LibrpaCrpaModelK::output_phi_ri,
                  true);
        stack(prefix + "P", {n, size(info.n_bands)},
              parent ? &LibrpaCrpaModelK::parent_projector_ri
                     : &LibrpaCrpaModelK::output_projector_ri,
              true);
        stack(prefix + "H", {n, n},
              parent ? &LibrpaCrpaModelK::parent_h_ha_ri : &LibrpaCrpaModelK::output_h_ha_ri, true);
        stack(prefix + "density", {n, n},
              parent ? &LibrpaCrpaModelK::parent_density_ri : &LibrpaCrpaModelK::output_density_ri,
              true);
    }
    for (bool response : {true, false})
    {
        std::vector<int> offsets{0}, bands;
        for (int spin = 0; spin < info.n_spins; ++spin)
            for (int k = 0; k < info.n_kpoints; ++k)
            {
                int count;
                const int* selected;
                librpa_crpa_result_bands(result, response, spin, k, &count, &selected);
                if (count) bands.insert(bands.end(), selected, selected + count);
                offsets.push_back(bands.size());
            }
        const std::string prefix = response ? "A_" : "B_";
        archive.array(prefix + "offsets", "<i4", {offsets.size()}, offsets.data(), 4);
        archive.array(prefix + "bands", "<i4", {bands.size()}, bands.data(), 4);
    }
    const int records = librpa_crpa_result_size(result);
    LibrpaCrpaTensor first;
    librpa_crpa_result_tensor(result, 0, &first);
    int frequencies = 0;
    std::vector<double> nodes;
    for (; frequencies < records; ++frequencies)
    {
        LibrpaCrpaTensor tensor;
        librpa_crpa_result_tensor(result, frequencies, &tensor);
        if (tensor.atom_index != first.atom_index || tensor.spin_left != first.spin_left ||
            tensor.spin_right != first.spin_right)
            break;
        nodes.push_back(tensor.frequency_ha);
    }
    archive.array("nu", "<f8", {nodes.size()}, nodes.data(), 8);
    std::vector<int> tensor_sites;
    for (int record = 0; record < records; record += frequencies)
    {
        LibrpaCrpaTensor tensor;
        librpa_crpa_result_tensor(result, record, &tensor);
        const auto n = size(tensor.n_orbitals), entries = n * n * n * n;
        int site = 0;
        for (; site < info.n_output_sites; ++site)
        {
            int atom, start, count;
            librpa_crpa_model_site(result, false, site, &atom, &start, &count);
            if (atom == tensor.atom_index) break;
        }
        const auto key = std::to_string(site) + "_" + std::to_string(tensor.spin_left) + "_" +
                         std::to_string(tensor.spin_right);
        tensor_sites.insert(tensor_sites.end(), {site, tensor.spin_left, tensor.spin_right});
        archive.array("v_" + key, "<c16", {n, n, n, n}, tensor.bare_ri, 16);
        for (bool full : {false, true})
        {
            if (full && !tensor.fully_screened_ri) continue;
            std::vector<double> values;
            values.reserve(2 * entries * frequencies);
            for (int iw = 0; iw < frequencies; ++iw)
            {
                librpa_crpa_result_tensor(result, record + iw, &tensor);
                const auto* data = full ? tensor.fully_screened_ri : tensor.partially_screened_ri;
                values.insert(values.end(), data, data + 2 * entries);
            }
            archive.array((full ? "W_" : "U_") + key, "<c16", {nodes.size(), n, n, n, n},
                          values.data(), 16);
        }
    }
    archive.array("tensor_sites", "<i4", {tensor_sites.size() / 3, 3}, tensor_sites.data(), 4);
    if (continued)
    {
        archive.array("continued_z", "<c16", {queries.size() / 2}, queries.data(), 16);
        for (int tensor = 0; tensor < librpa_crpa_continued_size(continued.get()); ++tensor)
        {
            const auto* indices = tensor_sites.data() + 3 * tensor;
            const auto key = std::to_string(indices[0]) + "_" + std::to_string(indices[1]) + "_" +
                             std::to_string(indices[2]);
            std::vector<double> values, spreads;
            std::vector<int> status;
            std::size_t n = 0;
            for (int query = 0; query < queries.size() / 2; ++query)
            {
                LibrpaCrpaContinuedTensor view;
                librpa_crpa_continued_tensor(continued.get(), tensor, query, &view);
                n = view.n_orbitals;
                const auto entries = n * n * n * n;
                values.insert(values.end(), view.values_ha_ri, view.values_ha_ri + 2 * entries);
                spreads.insert(spreads.end(), view.subset_spread_ha,
                               view.subset_spread_ha + entries);
                status.insert(status.end(), view.status, view.status + entries);
            }
            const std::vector<std::size_t> dimensions{queries.size() / 2, n, n, n, n};
            archive.array("continued_U_" + key, "<c16", dimensions, values.data(), 16);
            archive.array("continued_status_" + key, "<i4", dimensions, status.data(), 4);
            archive.array("continued_spread_" + key, "<f8", dimensions, spreads.data(), 8);
        }
    }
    archive.finish();
}
}  // namespace driver
