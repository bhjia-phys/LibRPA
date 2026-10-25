#include "localized_operator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace librpa_int::localized_operator
{
constexpr double two_pi = 6.283185307179586476925286766559;

Matz zero_matrix(const int n)
{
    Matz result(n, n, MAJOR::ROW);
    std::fill(result.ptr(), result.ptr() + result.size(), cplxdb{});
    return result;
}

double maximum_entry(const Matz& matrix)
{
    double result = 0.0;
    for (std::size_t i = 0; i < matrix.size(); ++i)
    {
        if (!std::isfinite(matrix.ptr()[i].real()) || !std::isfinite(matrix.ptr()[i].imag()))
            throw std::runtime_error("Localized operator contains non-finite matrix data");
        result = std::max(result, std::abs(matrix.ptr()[i]));
    }
    return result;
}

cplxdb phase(const Vector3_Order<double>& k, const Vector3_Order<int>& r)
{
    return std::polar(1.0, two_pi * (k.x * r.x + k.y * r.y + k.z * r.z));
}

Matz fourier(const LocalizedMatrix& blocks, const Vector3_Order<double>& k, const int n,
             const Matrix3& lattice, const int derivative)
{
    Matz result = zero_matrix(n);
    for (const auto& [r, matrix] : blocks)
    {
        if (matrix.nr() != n || matrix.nc() != n)
            throw std::invalid_argument("Localized operator basis dimensions differ");
        cplxdb weight = phase(k, r);
        if (derivative >= 0)
        {
            const std::array<double, 3> distance{
                r.x * lattice.e11 + r.y * lattice.e21 + r.z * lattice.e31,
                r.x * lattice.e12 + r.y * lattice.e22 + r.z * lattice.e32,
                r.x * lattice.e13 + r.y * lattice.e23 + r.z * lattice.e33};
            weight *= cplxdb(0.0, distance.at(derivative));
        }
        maximum_entry(matrix);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) result(i, j) += weight * matrix(i, j);
    }
    return result;
}

// S/r use the audited PyATB upper-triangle storage and completion convention.
Matz complete_hermitian_upper(Matz upper)
{
    for (int i = 0; i < upper.nr(); ++i)
    {
        upper(i, i) = upper(i, i).real();
        for (int j = i + 1; j < upper.nc(); ++j) upper(j, i) = std::conj(upper(i, j));
    }
    return upper;
}

std::vector<Vector3_Order<int>> shortest_periodic_images(const Vector3_Order<int>& residue,
                                                         const Vector3_Order<int>& period,
                                                         const Vector3_Order<double>& center_from,
                                                         const Vector3_Order<double>& center_to,
                                                         const Matrix3& lattice)
{
    const std::array<int, 3> r{residue.x, residue.y, residue.z};
    const std::array<int, 3> n{period.x, period.y, period.z};
    const std::array<double, 3> d{r[0] + center_to.x - center_from.x,
                                  r[1] + center_to.y - center_from.y,
                                  r[2] + center_to.z - center_from.z};
    if (!std::isfinite(lattice.Det()) || lattice.Det() == 0.0)
        throw std::invalid_argument("Periodic images require a nonsingular finite lattice");
    const Matrix3 inverse = lattice.Inverse();
    const std::array<double, 3> reciprocal_norm{std::hypot(inverse.e11, inverse.e21, inverse.e31),
                                                std::hypot(inverse.e12, inverse.e22, inverse.e32),
                                                std::hypot(inverse.e13, inverse.e23, inverse.e33)};
    const auto distance_squared = [&](const std::array<int, 3>& shift)
    {
        const Vector3<double> fractional(d[0] + n[0] * double(shift[0]),
                                         d[1] + n[1] * double(shift[1]),
                                         d[2] + n[2] * double(shift[2]));
        const double value = (fractional * lattice).norm2();
        if (!std::isfinite(value))
            throw std::invalid_argument("Periodic image distance is not finite");
        return value;
    };
    const auto checked_integer = [](const double value)
    {
        // Leave loop increment and integer R construction away from overflow.
        if (!std::isfinite(value) || value < std::numeric_limits<int>::min() + 1.0 ||
            value > std::numeric_limits<int>::max() - 1.0)
            throw std::invalid_argument("Periodic image search exceeds integer range");
        return static_cast<int>(value);
    };
    std::array<int, 3> initial{}, lower{}, upper{};
    for (int a = 0; a < 3; ++a)
    {
        if (n[a] <= 0 || !std::isfinite(d[a]) || !std::isfinite(reciprocal_norm[a]))
            throw std::invalid_argument(
                "Periodic images require positive periods and finite centers");
        initial[a] = checked_integer(std::round(-d[a] / n[a]));
    }
    const double initial_distance = distance_squared(initial);
    const double radius = std::sqrt(initial_distance + 1e-10 * std::max(1.0, initial_distance));
    for (int a = 0; a < 3; ++a)
    {
        const double extent = radius * reciprocal_norm[a] / n[a];
        const double center = -d[a] / n[a];
        lower[a] = checked_integer(std::ceil(center - extent - 1e-12));
        upper[a] = checked_integer(std::floor(center + extent + 1e-12));
    }
    double best = initial_distance;
    for (int x = lower[0]; x <= upper[0]; ++x)
        for (int y = lower[1]; y <= upper[1]; ++y)
            for (int z = lower[2]; z <= upper[2]; ++z)
                best = std::min(best, distance_squared({x, y, z}));
    // Determine the minimum first: incremental tie selection can depend on
    // enumeration order and fail the atom-pair reversal identity.
    const double tolerance = 1e-10 * std::max(1.0, best);
    std::vector<Vector3_Order<int>> result;
    for (int x = lower[0]; x <= upper[0]; ++x)
        for (int y = lower[1]; y <= upper[1]; ++y)
            for (int z = lower[2]; z <= upper[2]; ++z)
                if (distance_squared({x, y, z}) <= best + tolerance)
                    result.emplace_back(checked_integer(r[0] + double(n[0]) * x),
                                        checked_integer(r[1] + double(n[1]) * y),
                                        checked_integer(r[2] + double(n[2]) * z));
    if (result.empty()) throw std::runtime_error("Periodic image search found no nearest image");
    return result;
}

}  // namespace librpa_int::localized_operator
