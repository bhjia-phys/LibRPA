#include "checkpoint.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace librpa_int
{
namespace qsgw
{
namespace
{

template <typename T>
void put(std::ostream& output, const T& value)
{
    output.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T get(std::istream& input)
{
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) throw std::runtime_error("Truncated QSGW checkpoint");
    return value;
}

double get_real(std::istream& input)
{
    const double value = get<double>(input);
    if (!std::isfinite(value)) throw std::runtime_error("Non-finite QSGW checkpoint value");
    return value;
}

void put_complex(std::ostream& output, const cplxdb value)
{
    put(output, value.real());
    put(output, value.imag());
}

cplxdb get_complex(std::istream& input)
{
    const double real = get_real(input);
    const double imaginary = get_real(input);
    return {real, imaginary};
}

void require_equal(const double saved, const double expected)
{
    if (!std::isfinite(expected) || std::abs(saved - expected) > 1.0e-12)
        throw std::runtime_error(
            "QSGW checkpoint reference or update settings differ from the input");
}

void write_meanfield(std::ostream& output, const MeanField& meanfield)
{
    for (const int count :
         {meanfield.get_n_spins(), meanfield.get_n_kpoints(), meanfield.get_n_bands(),
          meanfield.get_n_aos(), meanfield.get_n_spinor()})
        put(output, static_cast<std::int32_t>(count));
    put(output, meanfield.get_efermi());
    for (int spin = 0; spin < meanfield.get_n_spins(); ++spin)
    {
        for (int k = 0; k < meanfield.get_n_kpoints(); ++k)
            for (int band = 0; band < meanfield.get_n_bands(); ++band)
            {
                put(output, meanfield.get_eigenvals()[spin](k, band));
                put(output, meanfield.get_weight()[spin](k, band));
            }
        for (int spinor = 0; spinor < meanfield.get_n_spinor(); ++spinor)
            for (int k = 0; k < meanfield.get_n_kpoints(); ++k)
            {
                const auto& wfc = meanfield.get_eigenvectors().at(spin).at(spinor).at(k);
                if (wfc.nr != meanfield.get_n_bands() || wfc.nc != meanfield.get_n_aos())
                    throw std::runtime_error("QSGW checkpoint requires replicated wavefunctions");
                for (int i = 0; i < wfc.size; ++i) put_complex(output, wfc.c[i]);
            }
    }
}

// Shapes are checked against native inputs before any checkpoint allocation.
MeanField read_meanfield(std::istream& input, const MeanField& reference)
{
    for (const int count :
         {reference.get_n_spins(), reference.get_n_kpoints(), reference.get_n_bands(),
          reference.get_n_aos(), reference.get_n_spinor()})
        if (get<std::int32_t>(input) != count)
            throw std::runtime_error("QSGW checkpoint mean-field dimensions differ from the input");
    MeanField result = reference;
    result.get_efermi() = get_real(input);
    for (int spin = 0; spin < result.get_n_spins(); ++spin)
    {
        for (int k = 0; k < result.get_n_kpoints(); ++k)
            for (int band = 0; band < result.get_n_bands(); ++band)
            {
                result.get_eigenvals()[spin](k, band) = get_real(input);
                result.get_weight()[spin](k, band) = get_real(input);
            }
        for (int spinor = 0; spinor < result.get_n_spinor(); ++spinor)
            for (int k = 0; k < result.get_n_kpoints(); ++k)
            {
                auto& wfc = result.get_eigenvectors().at(spin).at(spinor).at(k);
                for (int i = 0; i < wfc.size; ++i) wfc.c[i] = get_complex(input);
            }
    }
    return result;
}

void compare_meanfield(const MeanField& saved, const MeanField& reference)
{
    require_equal(saved.get_efermi(), reference.get_efermi());
    for (int spin = 0; spin < reference.get_n_spins(); ++spin)
    {
        for (int k = 0; k < reference.get_n_kpoints(); ++k)
            for (int band = 0; band < reference.get_n_bands(); ++band)
            {
                require_equal(saved.get_eigenvals()[spin](k, band),
                              reference.get_eigenvals()[spin](k, band));
                require_equal(saved.get_weight()[spin](k, band),
                              reference.get_weight()[spin](k, band));
            }
        for (int spinor = 0; spinor < reference.get_n_spinor(); ++spinor)
            for (int k = 0; k < reference.get_n_kpoints(); ++k)
            {
                const auto& expected = reference.get_eigenvectors().at(spin).at(spinor).at(k);
                const auto& actual = saved.get_eigenvectors().at(spin).at(spinor).at(k);
                for (int i = 0; i < expected.size; ++i)
                {
                    require_equal(actual.c[i].real(), expected.c[i].real());
                    require_equal(actual.c[i].imag(), expected.c[i].imag());
                }
            }
    }
}

void write_matrices(std::ostream& output, const SpinKMatrixMap& matrices,
                    const MeanField& reference)
{
    const int n = reference.get_n_bands();
    for (int spin = 0; spin < reference.get_n_spins(); ++spin)
        for (int k = 0; k < reference.get_n_kpoints(); ++k)
        {
            const auto& matrix = matrices.at(spin).at(k);
            if (matrix.nr() != n || matrix.nc() != n)
                throw std::runtime_error("Invalid QSGW checkpoint Hamiltonian dimensions");
            for (int row = 0; row < n; ++row)
                for (int col = 0; col < n; ++col) put_complex(output, matrix(row, col));
        }
}

SpinKMatrixMap read_matrices(std::istream& input, const MeanField& reference)
{
    SpinKMatrixMap result;
    const int n = reference.get_n_bands();
    for (int spin = 0; spin < reference.get_n_spins(); ++spin)
        for (int k = 0; k < reference.get_n_kpoints(); ++k)
        {
            Matz matrix(n, n, MAJOR::ROW);
            for (int row = 0; row < n; ++row)
                for (int col = 0; col < n; ++col) matrix(row, col) = get_complex(input);
            result[spin].emplace(k, std::move(matrix));
        }
    return result;
}

void write_context(std::ostream& output, const CheckpointContext& context)
{
    put(output, std::int32_t(context.band_reference != nullptr));
    for (const int value :
         {context.cut_mode, context.unoccupied_keep, context.nfreq, context.head_mode})
        put(output, static_cast<std::int32_t>(value));
    put(output, context.cut_shift_ha);
    put(output, context.mixing_beta);
    put(output, std::int32_t(0));
    put(output, std::int32_t(0));
    for (const auto* points : {&context.kpoints, &context.band_kpoints})
    {
        put(output, static_cast<std::int32_t>(points->size()));
        for (const auto& point : *points)
            for (const double value : {point.x, point.y, point.z}) put(output, value);
    }
    put(output, static_cast<std::int32_t>(context.k_weights.size()));
    for (const double weight : context.k_weights) put(output, weight);
    write_meanfield(output, context.reference);
    write_matrices(output, context.dft_vxc, context.reference);
    if (context.band_reference)
    {
        write_meanfield(output, *context.band_reference);
        write_matrices(output, context.dft_vxc_band, *context.band_reference);
    }
}

void check_context(std::istream& input, const CheckpointContext& context,
                   const std::uint32_t version)
{
    if (get<std::int32_t>(input) != (context.band_reference != nullptr))
        throw std::runtime_error("QSGW checkpoint task differs from the input");
    for (const int value :
         {context.cut_mode, context.unoccupied_keep, context.nfreq, context.head_mode})
        require_equal(get<std::int32_t>(input), value);
    require_equal(get_real(input), context.cut_shift_ha);
    require_equal(get_real(input), context.mixing_beta);
    // Fixed-mode files retain compatibility with version 1--3 checkpoints.
    // Mixer history belongs to the separate automatic-mixing implementation.
    if (version >= 2 && get<std::int32_t>(input) != 0)
        throw std::runtime_error("Automatic-mixing checkpoints require a mixer-aware build");
    if (version >= 3 && get<std::int32_t>(input) != 0)
        throw std::runtime_error("QSGW checkpoint requires an unsupported mixing metric");
    for (const auto* points : {&context.kpoints, &context.band_kpoints})
    {
        if (get<std::int32_t>(input) != static_cast<int>(points->size()))
            throw std::runtime_error("QSGW checkpoint k-point count differs from the input");
        for (const auto& point : *points)
            for (const double value : {point.x, point.y, point.z})
                require_equal(get_real(input), value);
    }
    if (get<std::int32_t>(input) != static_cast<int>(context.k_weights.size()))
        throw std::runtime_error("QSGW checkpoint k-weight count differs from the input");
    for (const double weight : context.k_weights) require_equal(get_real(input), weight);
    const auto check_channel = [&](const MeanField& reference, const SpinKMatrixMap& vxc)
    {
        compare_meanfield(read_meanfield(input, reference), reference);
        const auto saved_vxc = read_matrices(input, reference);
        for (const auto& [spin, by_k] : saved_vxc)
            for (const auto& [k, matrix] : by_k)
                for (int row = 0; row < matrix.nr(); ++row)
                    for (int col = 0; col < matrix.nc(); ++col)
                    {
                        const auto expected = vxc.at(spin).at(k)(row, col);
                        require_equal(matrix(row, col).real(), expected.real());
                        require_equal(matrix(row, col).imag(), expected.imag());
                    }
    };
    check_channel(context.reference, context.dft_vxc);
    if (context.band_reference) check_channel(*context.band_reference, context.dft_vxc_band);
}

}  // namespace

void write_qsgw_checkpoint(const std::string& filename, const CheckpointContext& context,
                           const IterationSummary& summary, const MeanField& live,
                           const MeanField* band_live, const SpinKMatrixMap& hamiltonian,
                           const SpinKMatrixMap& band_hamiltonian)
{
    const std::string temporary = filename + ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("Cannot write QSGW checkpoint: " + temporary);
    output.write("LIBRPA_QSGW", 11);
    put(output, std::uint32_t(3));           // format version, including coefficient metric
    put(output, std::uint32_t(0x01020304));  // reject incompatible byte order
    write_context(output, context);
    put(output, std::int32_t(summary.iteration));
    for (const double value :
         {summary.maximum_eigenvalue_change_ev, summary.residual_l2_ha, summary.residual_max_ha,
          summary.fermi_energy_ev, summary.gap_ev, summary.electron_count, summary.beta})
        put(output, value);
    put(output, std::int32_t(summary.converged));
    write_meanfield(output, live);
    write_matrices(output, hamiltonian, context.reference);
    if (context.band_reference)
    {
        write_meanfield(output, *band_live);
        write_matrices(output, band_hamiltonian, *context.band_reference);
    }
    output.flush();
    if (!output) throw std::runtime_error("Failed to flush QSGW checkpoint: " + temporary);
    output.close();
    if (!output) throw std::runtime_error("Failed to close QSGW checkpoint: " + temporary);
    // Keep the last complete iteration if the process dies during a later write.
    std::filesystem::rename(temporary, filename);
}

IterationSummary read_qsgw_checkpoint(const std::string& filename, const CheckpointContext& context,
                                      MeanField& live, MeanField* band_live,
                                      SpinKMatrixMap& hamiltonian, SpinKMatrixMap& band_hamiltonian)
{
    std::ifstream input(filename, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read QSGW checkpoint: " + filename);
    char magic[11];
    input.read(magic, sizeof(magic));
    if (!input || std::string(magic, sizeof(magic)) != "LIBRPA_QSGW")
        throw std::runtime_error("Unsupported QSGW checkpoint format");
    const auto version = get<std::uint32_t>(input);
    if ((version < 1 || version > 3) || get<std::uint32_t>(input) != 0x01020304)
        throw std::runtime_error("Unsupported QSGW checkpoint format or byte order");
    check_context(input, context, version);
    IterationSummary summary;
    summary.iteration = get<std::int32_t>(input);
    if (summary.iteration < 1) throw std::runtime_error("Invalid QSGW checkpoint iteration");
    summary.maximum_eigenvalue_change_ev = get_real(input);
    summary.residual_l2_ha = get_real(input);
    summary.residual_max_ha = get_real(input);
    summary.fermi_energy_ev = get_real(input);
    summary.gap_ev = get_real(input);
    summary.electron_count = get_real(input);
    summary.beta = get_real(input);
    const int converged = get<std::int32_t>(input);
    if (converged != 0 && converged != 1)
        throw std::runtime_error("Invalid QSGW checkpoint convergence flag");
    summary.converged = converged != 0;
    MeanField restored = read_meanfield(input, context.reference);
    SpinKMatrixMap restored_h = read_matrices(input, context.reference);
    MeanField restored_band;
    SpinKMatrixMap restored_band_h;
    if (context.band_reference)
    {
        restored_band = read_meanfield(input, *context.band_reference);
        restored_band_h = read_matrices(input, *context.band_reference);
    }
    if (input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Unexpected trailing QSGW checkpoint data");
    live = std::move(restored);
    hamiltonian = std::move(restored_h);
    if (context.band_reference)
    {
        *band_live = std::move(restored_band);
        band_hamiltonian = std::move(restored_band_h);
    }
    return summary;
}

void broadcast_checkpoint_meanfield(MeanField& live, const MpiCommHandler& communicator)
{
    communicator.bcast(&live.get_efermi(), 1, 0);
    for (int spin = 0; spin < live.get_n_spins(); ++spin)
    {
        auto& eigenvalues = live.get_eigenvals()[spin];
        auto& occupations = live.get_weight()[spin];
        communicator.bcast(eigenvalues.c, eigenvalues.nr * eigenvalues.nc, 0);
        communicator.bcast(occupations.c, occupations.nr * occupations.nc, 0);
        for (int spinor = 0; spinor < live.get_n_spinor(); ++spinor)
            for (int k = 0; k < live.get_n_kpoints(); ++k)
            {
                auto& wfc = live.get_eigenvectors().at(spin).at(spinor).at(k);
                communicator.bcast(wfc.c, wfc.size, 0);
            }
    }
}

}  // namespace qsgw
}  // namespace librpa_int
