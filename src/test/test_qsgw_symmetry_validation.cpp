#include "../core/meanfield_mpi.h"
#include "../core/qsgw/projection_target.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{

using namespace librpa_int;
using qsgw::validate_live_symmetry_state;

void require(const bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void require_rejection(Function function, const std::string& diagnostic = {})
{
    try
    {
        function();
    }
    catch (const std::exception& error)
    {
        require(diagnostic.empty() || std::string(error.what()).find(diagnostic) != std::string::npos,
                "unexpected rejection: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("invalid live symmetry state was accepted");
}

struct InversionModel
{
    PeriodicBoundaryData pbc;
    AtomicBasis basis{std::vector<std::size_t>{1, 1}};
    SymmetryContext symmetry;

    InversionModel()
    {
        pbc.set_latvec({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
        basis.set_l_shells({{0}, {0}});
        symmetry.set_crystal_structure(
            pbc.latvec, pbc.G, {{0, 0}, {1, 0}},
            {{0, {0.0, 0.0, 0.0}}, {1, {0.25, 0.25, 0.25}}});
        symmetry.basis_convention = {-1, 0, LIBRPA_ANGULAR_ORDER_NATURAL,
                                     LIBRPA_RSH_COEFF_1_M, LIBRPA_RSH_COEFF_1_M};
        SymmetryOperation inversion;
        inversion.rotation = {-1, 0, 0, 0, -1, 0, 0, 0, -1};
        inversion.translation = {0.25, 0.25, 0.25};
        inversion.use_row_convention = true;
        symmetry.set_rspace_operations({SpaceGroupSymOp::IDENTITY, inversion});
        symmetry.set_available();
    }

    SymmetryContext explicit_spin_symmetry() const
    {
        auto result = symmetry;
        SymmetrySpinOperation identity, inversion;
        inversion.spatial_id = 1;
        result.set_symmetry_spin_operations({identity, inversion}, false);
        return result;
    }

    MeanField meanfield(const double angle, const int components = 1,
                        const int spins = 1) const
    {
        MeanField result(spins, 1, 2 * components, 2, components);
        const double c = std::cos(angle), s = std::sin(angle);
        for (int spin = 0; spin < spins; ++spin)
        {
            for (int band = 0; band < 2 * components; ++band)
            {
                result.get_eigenvals()[spin](0, band) = band % 2 == 0 ? -1.0 : 1.0;
                result.get_weight()[spin](0, band) = band % 2 == 0 ? 1.0 : 0.0;
            }
            for (int component = 0; component < components; ++component)
            {
                ComplexMatrix coefficients(2 * components, 2);
                coefficients.zero_out();
                coefficients(2 * component, 0) = c;
                coefficients(2 * component, 1) = s;
                coefficients(2 * component + 1, 0) = -s;
                coefficients(2 * component + 1, 1) = c;
                result.get_eigenvectors()[spin][component][0] = std::move(coefficients);
            }
        }
        return result;
    }
};

void test_legacy_and_strict_numerical_frames()
{
    const InversionModel model;
    const double broken_angle = 0.31;
    const auto broken = model.meanfield(broken_angle);
    // In this orthonormal two-s-orbital model inversion exchanges the atoms.
    // H = C^T diag(-1,+1) C has H00 != H11 at this angle, independently
    // establishing that the fixture violates A^T H A = H.
    const double h00 = -std::cos(2.0 * broken_angle);
    const double h11 = std::cos(2.0 * broken_angle);
    require(std::abs(h00 - h11) > 0.1, "the negative control is inversion covariant");
    require_rejection(
        [&] { validate_symmetry_band_weights(broken, model.basis, model.pbc, model.symmetry); },
        "covariance failed");

    // The actual Task helper preserves the historical scalar full-grid path
    // without rotating or replacing its finite, complete E/C/f inputs.
    const auto before = broken;
    validate_live_symmetry_state(broken, model.basis, model.pbc, model.symmetry, false);
    for (int band = 0; band < 2; ++band)
    {
        require(broken.get_eigenvals()[0](0, band) == before.get_eigenvals()[0](0, band) &&
                    broken.get_weight()[0](0, band) == before.get_weight()[0](0, band),
                "validation changed E/f");
        for (int ao = 0; ao < 2; ++ao)
            require((*broken.find_wfc(0, 0, 0))(band, ao) ==
                        (*before.find_wfc(0, 0, 0))(band, ao),
                    "validation changed C");
    }

    const auto explicit_spin = model.explicit_spin_symmetry();
    require_rejection(
        [&] { validate_live_symmetry_state(broken, model.basis, model.pbc, explicit_spin, false); },
        "covariance failed");
    // The current mesh is full after expansion; the retained source provenance
    // must still select strict validation.
    require_rejection(
        [&] { validate_live_symmetry_state(broken, model.basis, model.pbc, model.symmetry, true); },
        "covariance failed");

    const auto covariant = model.meanfield(std::acos(-1.0) / 4.0);
    validate_symmetry_band_weights(covariant, model.basis, model.pbc, model.symmetry);
    validate_live_symmetry_state(covariant, model.basis, model.pbc, explicit_spin, false);
    validate_live_symmetry_state(covariant, model.basis, model.pbc, model.symmetry, true);

    // Spinor storage remains strict even for an implicit ordinary/grey table.
    const auto broken_spinor = model.meanfield(broken_angle, 2);
    require_rejection(
        [&]
        {
            validate_live_symmetry_state(broken_spinor, model.basis, model.pbc, model.symmetry,
                                         false);
        },
        "covariance failed");
    const auto covariant_spinor = model.meanfield(std::acos(-1.0) / 4.0, 2);
    validate_live_symmetry_state(covariant_spinor, model.basis, model.pbc, model.symmetry, false);

    // Two-channel collinear states are outside the compatibility exception too.
    const auto broken_collinear = model.meanfield(broken_angle, 1, 2);
    require_rejection(
        [&]
        {
            validate_live_symmetry_state(broken_collinear, model.basis, model.pbc, explicit_spin,
                                         false);
        },
        "covariance failed");
    require_rejection(
        [&]
        {
            validate_live_symmetry_state(broken_collinear, model.basis, model.pbc, model.symmetry,
                                         false);
        },
        "collinear spin channels");
}

void test_legacy_frame_validation_is_not_skipped()
{
    const InversionModel model;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (int damaged = 0; damaged < 3; ++damaged)
    {
        auto state = model.meanfield(0.31);
        if (damaged == 0) state.get_eigenvectors()[0][0][0](0, 0) = {nan, 0.0};
        if (damaged == 1) state.get_eigenvals()[0](0, 0) = nan;
        if (damaged == 2) state.get_weight()[0](0, 0) = nan;
        require_rejection(
            [&]
            {
                validate_live_symmetry_state(state, model.basis, model.pbc, model.symmetry, false);
            },
            "non-finite");
    }

    auto incomplete = model.meanfield(0.31);
    incomplete.get_eigenvectors()[0][0].clear();
    require_rejection(
        [&]
        {
            validate_live_symmetry_state(incomplete, model.basis, model.pbc, model.symmetry, false);
        },
        "incomplete");
    auto malformed = model.meanfield(0.31);
    malformed.get_eigenvals()[0].create(1, 1);
    require_rejection(
        [&]
        {
            validate_live_symmetry_state(malformed, model.basis, model.pbc, model.symmetry, false);
        },
        "malformed");

    // A reduced source that has not been expanded cannot qualify merely because
    // expanded_from_ibz is false.
    auto reduced_grid = model.pbc;
    reduced_grid.set_period(2, 1, 1);
    reduced_grid.kfrac_list = model.pbc.kfrac_list;
    reduced_grid.klist = model.pbc.klist;
    require_rejection(
        [&]
        {
            validate_live_symmetry_state(model.meanfield(0.31), model.basis, reduced_grid,
                                         model.symmetry, false);
        });

    // Rectangular frames stay outside the legacy exception. Master supports
    // a retained frame only when its span is closed under the operations.
    MeanField rectangular(1, 1, 1, 2);
    ComplexMatrix coefficients(1, 2);
    coefficients.zero_out();
    coefficients(0, 0) = 1.0;
    rectangular.get_eigenvectors()[0][0][0] = std::move(coefficients);
    rectangular.get_eigenvals()[0](0, 0) = -1.0;
    rectangular.get_weight()[0](0, 0) = 1.0;
    require_rejection(
        [&]
        {
            validate_live_symmetry_state(rectangular, model.basis, model.pbc, model.symmetry, false);
        },
        "covariance failed");

    // Inversion exchanges the two AOs, so (1,1)/sqrt(2) spans a closed
    // one-dimensional retained space. This is a numerical positive control
    // for the retained-frame route inherited from master.
    rectangular.get_eigenvectors()[0][0][0](0, 0) = std::sqrt(0.5);
    rectangular.get_eigenvectors()[0][0][0](0, 1) = std::sqrt(0.5);
    validate_live_symmetry_state(rectangular, model.basis, model.pbc, model.symmetry, false);
}

} // namespace

int main()
{
    try
    {
        test_legacy_and_strict_numerical_frames();
        test_legacy_frame_validation_is_not_skipped();
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "QSGW live symmetry validation controls passed\n";
    return 0;
}
