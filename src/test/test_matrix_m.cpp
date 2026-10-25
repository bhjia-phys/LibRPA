#include "../math/matrix_m.h"
#include "testutils.h"

using namespace std;
using namespace librpa_int;

static void test_constuctor()
{
    Matd m1(2, 3, MAJOR::COL);
    m1.zero_out();
    m1(0, 1) = 1.0;
    m1(1, 0) = 2.0;
    printf("Matrix (set column-major)\n%s", str(m1).c_str());
    auto det = get_determinant(m1);
    printf("Determinant = %f\n", det);
    assert(fequal(det, -2.0));

    m1.swap_to_row_major();
    printf("Matrix (swap to row-major)\n%s", str(m1).c_str());
    det = get_determinant(m1);
    printf("Determinant = %f\n", det);
    assert(fequal(det, -2.0));
}

static void test_minmax()
{
    int ir = 0, ic = 0;
    Matd m1({{1.0, 2.0}, {3.0, -4.0}}, MAJOR::ROW);
    assert(fequal(m1.max(), 3.0));
    assert(fequal(m1.min(), -4.0));
    m1.argmin(ir, ic);
    assert( ir == 1 && ic == 1);
    m1.argmax(ir, ic);
    assert( ir == 1 && ic == 0);

    // complex matrix
    Matz mc1({{1.0, 2.0}, {3.0, -4.0}}, MAJOR::ROW);
    assert(fequal(mc1.max(), 4.0));
    assert(fequal(mc1.min(), 1.0));
    mc1.argmin(ir, ic);
    assert( ir == 0 && ic == 0);
    mc1.argmax(ir, ic);
    assert( ir == 1 && ic == 1);
}

static void test_power_hemat()
{
    cout << "Run test_power_hemat" << endl;
    const int n = 8;
    auto hemat = random_he_selected_ev<double>(n, {1.0, 2.25, 4.0, 9.0}, MAJOR::ROW);
    cout << "hemat" << endl << hemat;
    const auto hemat_back = hemat.copy();
    auto sqrt_hemat = power_hemat(hemat, 0.5, false, false);
    cout << "hemat after power_hemat" << endl << hemat;
    assert(hemat == hemat_back);
    assert(hemat == sqrt_hemat * sqrt_hemat);
    hemat.swap_to_col_major();
    sqrt_hemat = power_hemat(hemat, 0.5, false, false);
    assert(hemat == sqrt_hemat * sqrt_hemat);
}

static void test_multiply()
{
    Matd m1({{1.0, 2.0}, {2.0, -1.0}}, MAJOR::ROW);
    Matd m2({{2.0, 0.0, -1.0}, {1.0, -1.0, 0.0}}, MAJOR::ROW);
    printf("Mat1\n%sMat2\n%s", str(m1).c_str(), str(m2).c_str());
    auto m1_times_m2 = m1 * m2;
    printf("Mat1*Mat2\n%s", str(m1_times_m2).c_str());
    Matd m3({{4.0, -2.0, -1.0}, {3.0, 1.0, -2.0}}, MAJOR::ROW);
    assert(fequal_array(6, m1_times_m2.ptr(), m3.ptr()));

    // same result should be obtained with column-major
    m1.swap_to_col_major();
    m2.swap_to_col_major();
    m1_times_m2 = m1 * m2;
    m3.swap_to_col_major();
    assert(fequal_array(6, m1_times_m2.ptr(), m3.ptr()));
}

static void test_conjugate()
{
    Matz m({{{1.0, -2.0}, {2.0, -1.0}}, {{2.0, 1.0}, -1.0}});
    Matz m_conj_ref({{{1.0, 2.0}, {2.0, 1.0}}, {{2.0, -1.0}, -1.0}});
    m.conj();
    assert(m == m_conj_ref);
}

template <typename T>
static void check_inverse(const matrix_m<T> &m, const matrix_m<T> &expected,
                          const char *label)
{
    const auto original = m.copy();
    // Exercise resize as well as the preservation of the requested layout.
    matrix_m<T> result(1, 1, m.major());
    inverse(m, result);
    assert(result.major() == m.major());
    assert(result.nr() == m.nr() && result.nc() == m.nc());
    double reference_error = 0.0, left_residual = 0.0, right_residual = 0.0;
    for (int i = 0; i < m.nr(); ++i)
        for (int j = 0; j < m.nc(); ++j)
        {
            assert(m(i, j) == original(i, j));
            reference_error = std::max(reference_error, std::abs(result(i, j) - expected(i, j)));
            T left = 0, right = 0;
            // Independent scalar products: do not verify inverse via the
            // library's matrix multiplication or a second LAPACK inverse.
            for (int k = 0; k < m.nc(); ++k)
            {
                left += m(i, k) * result(k, j);
                right += result(i, k) * m(k, j);
            }
            const T identity = (i == j) ? T(1) : T(0);
            left_residual = std::max(left_residual, std::abs(left - identity));
            right_residual = std::max(right_residual, std::abs(right - identity));
        }
    printf("inverse %s %s: reference=%.3e left=%.3e right=%.3e\n", label,
           m.is_row_major() ? "row" : "column", reference_error,
           left_residual, right_residual);
    assert(reference_error < 1e-12);
    assert(left_residual < 1e-12 && right_residual < 1e-12);
}

static void test_inverse()
{
    for (const auto major : {MAJOR::ROW, MAJOR::COL})
    {
        // A=P*U, where P exchanges rows 0 and 2. The analytic inverse is
        // U^{-1}*P. These nonsymmetric matrices require an LU row pivot.
        Matd real({{0.0, 0.0, 5.0}, {0.0, 4.0, 2.0}, {2.0, 1.0, 3.0}}, major);
        Matd real_inverse({{-0.25, -0.125, 0.5}, {-0.1, 0.25, 0.0},
                           {0.2, 0.0, 0.0}}, major);
        check_inverse(real, real_inverse, "double-3x3-pivot");
        // Non-Hermitian complex entries detect unintended conjugation.
        Matz complex({{0.0, 0.0, {2.0, -1.0}}, {0.0, 2.0, {1.0, 1.0}},
                      {{1.0, 1.0}, {2.0, -1.0}, {0.0, -1.0}}}, major);
        Matz complex_inverse({{{0.6, 0.3}, {-0.25, 0.75}, {0.5, -0.5}},
                              {{-0.1, -0.3}, 0.5, 0.0},
                              {{0.4, 0.2}, 0.0, 0.0}}, major);
        check_inverse(complex, complex_inverse, "complex-3x3-pivot");
        check_inverse(Matd({{4.0}}, major), Matd({{0.25}}, major), "double-1x1");
        check_inverse(Matz({{{1.0, 2.0}}}, major),
                      Matz({{{0.2, -0.4}}}, major), "complex-1x1");
    }
}

int main (int argc, char *argv[])
{
    test_constuctor();
    test_multiply();
    test_minmax();
    test_power_hemat();
    test_conjugate();
    test_inverse();
    return 0;
}
