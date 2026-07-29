#pragma once
// ============================================================
//  expm_pade.h — Matrix exponential, scaling-and-squaring Padé
//
//  Higham (2005), "The scaling and squaring method for the
//  matrix exponential revisited", SIAM J. Matrix Anal. Appl.
//  26(4):1179-1193.  Same algorithm family as MATLAB's expm.
//
//  Hand-rolled rather than <unsupported/Eigen/MatrixFunctions>
//  so the cost is predictable and the code is auditable — for
//  a Teensy timing benchmark you want to know exactly which
//  branch you're paying for.
//
//  Precision dispatch
//  ------------------
//  The theta_m thresholds are the largest ||A||_1 for which the
//  [m/m] Pade approximant has backward error at or below the
//  unit roundoff u of the working precision.  They therefore
//  depend on u and MUST be swapped when Scalar changes.
//
//    double (u = 2^-53):  m in {3,5,7,9,13}, scale to theta_13
//    float  (u = 2^-24):  m in {3,5,7},      scale to theta_7
//
//  Single precision stops at m=7 because there is no accuracy
//  left to buy above it — theta_9/theta_13 are derived from a
//  backward-error target ~9 orders tighter than float roundoff.
//  (Same degree sets as Eigen's MatrixExponential.h.)
//
//  Cost at Ts <= 0.05 for the augmented 18x18:
//    double -> m = 7 or 9, s = 0  (3-4 GEMMs + one 18x18 LU)
//    float  -> m = 3 or 5, s = 0  (1-2 GEMMs + one 18x18 LU)
// ============================================================

// https://sci-hub.st/10.1137/1.9780898717778.ch10
// https://www.cis.upenn.edu/~cis6100/higham_matrix_exponential_siam_2004.pdf

#include <Eigen/Dense>
#include <cmath>
#include <type_traits>

// ------------------------------------------------------------
//  Degree-selection thresholds, per precision.
// ------------------------------------------------------------
template <typename S>
struct expm_pade_traits;

template <>
struct expm_pade_traits<double> {
    // Higham (2005), Table 2.3, u = 2^-53.
    static constexpr double th3  = 1.495585217958292e-2;
    static constexpr double th5  = 2.539398330063230e-1;
    static constexpr double th7  = 9.504178996162932e-1;
    static constexpr double th9  = 2.097847961257068e+0;
    static constexpr double th13 = 5.371920351148152e+0;
    static constexpr int    max_degree = 13;
    static constexpr double th_top     = th13;   // scaling target
};

template <>
struct expm_pade_traits<float> {
    // u = 2^-24.  Degrees 9 and 13 are unused at this precision.
    static constexpr float th3 = 4.258730016922831e-1f;
    static constexpr float th5 = 1.880152677804762e+0f;
    static constexpr float th7 = 3.925724783138660e+0f;
    static constexpr int   max_degree = 7;
    static constexpr float th_top     = th7;     // scaling target
};

// ------------------------------------------------------------
//  Padé numerator/denominator builders.
//    U = odd part * A,  V = even part,  r_m = (V-U)^{-1}(V+U)
// ------------------------------------------------------------
namespace expm_detail {

template <typename MatT>
inline void pade3(const MatT& A, const MatT& I, MatT& U, MatT& V) {
    using S = typename MatT::Scalar;
    const MatT A2 = A * A;
    U = A * (A2 + S(60.0) * I);
    V = S(12.0) * A2 + S(120.0) * I;
}

template <typename MatT>
inline void pade5(const MatT& A, const MatT& I, MatT& U, MatT& V) {
    using S = typename MatT::Scalar;
    const MatT A2 = A * A;
    const MatT A4 = A2 * A2;
    U = A * (A4 + S(420.0) * A2 + S(15120.0) * I);
    V = S(30.0) * A4 + S(3360.0) * A2 + S(30240.0) * I;
}

template <typename MatT>
inline void pade7(const MatT& A, const MatT& I, MatT& U, MatT& V) {
    using S = typename MatT::Scalar;
    const MatT A2 = A * A;
    const MatT A4 = A2 * A2;
    const MatT A6 = A4 * A2;
    U = A * (A6 + S(1512.0) * A4 + S(277200.0) * A2 + S(8648640.0) * I);
    V = S(56.0) * A6 + S(25200.0) * A4 + S(1995840.0) * A2
      + S(17297280.0) * I;
}

// --- double-only below this line ---

template <typename MatT>
inline void pade9(const MatT& A, const MatT& I, MatT& U, MatT& V) {
    using S = typename MatT::Scalar;
    const MatT A2 = A * A;
    const MatT A4 = A2 * A2;
    const MatT A6 = A4 * A2;
    const MatT A8 = A6 * A2;
    U = A * (A8 + S(3960.0) * A6 + S(2162160.0) * A4
           + S(302702400.0) * A2 + S(8821612800.0) * I);
    V = S(90.0) * A8 + S(110880.0) * A6 + S(30270240.0) * A4
      + S(2075673600.0) * A2 + S(17643225600.0) * I;
}

template <typename MatT>
inline void pade13(const MatT& A, const MatT& I, MatT& U, MatT& V) {
    using S = typename MatT::Scalar;
    const MatT A2 = A * A;
    const MatT A4 = A2 * A2;
    const MatT A6 = A4 * A2;

    U = A * ( A6 * (A6 + S(16380.0) * A4 + S(40840800.0) * A2)
            + S(33522128640.0)       * A6
            + S(10559470521600.0)    * A4
            + S(1187353796428800.0)  * A2
            + S(32382376266240000.0) * I );

    V =       A6 * (S(182.0) * A6 + S(960960.0) * A4
                    + S(1323241920.0) * A2)
            + S(670442572800.0)      * A6
            + S(129060195264000.0)   * A4
            + S(7771770303897600.0)  * A2
            + S(64764752532480000.0) * I;
}

}  // namespace expm_detail

// ------------------------------------------------------------
template <typename MatT>
MatT expm_pade(const MatT& Ain)
{
    using S  = typename MatT::Scalar;
    using Th = expm_pade_traits<S>;
    static_assert(MatT::RowsAtCompileTime > 0,
                  "expm_pade requires a fixed-size matrix");

    const MatT I = MatT::Identity();

    // Matrix 1-norm = max absolute column sum.
    // (NB: Eigen's lpNorm<1>() is the ENTRYWISE sum — not this.)
    const S nA = Ain.cwiseAbs().colwise().sum().maxCoeff();

    MatT A = Ain;
    MatT U, V;
    int  s = 0;

    if (nA <= Th::th3) {
        expm_detail::pade3(A, I, U, V);

    } else if (nA <= Th::th5) {
        expm_detail::pade5(A, I, U, V);

    } else if constexpr (Th::max_degree == 7) {
        // ---- single precision: degree 7 is the top ----
        if (nA > Th::th_top) {
            s = static_cast<int>(std::ceil(std::log2(double(nA / Th::th_top))));
            if (s < 0) s = 0;
            A = Ain * std::ldexp(S(1), -s);   // exact, no pow()
        }
        expm_detail::pade7(A, I, U, V);

    } else {
        // ---- double precision: 7, 9, then 13 with scaling ----
        if (nA <= Th::th7) {
            expm_detail::pade7(A, I, U, V);
        } else if (nA <= Th::th9) {
            expm_detail::pade9(A, I, U, V);
        } else {
            if (nA > Th::th_top) {
                s = static_cast<int>(std::ceil(std::log2(double(nA / Th::th_top))));
                if (s < 0) s = 0;
                A = Ain * std::ldexp(S(1), -s);
            }
            expm_detail::pade13(A, I, U, V);
        }
    }

    // r_m(A) = (V - U)^{-1} (V + U)
    MatT X = (V - U).partialPivLu().solve(V + U);

    // Undo the scaling by repeated squaring.
    for (int i = 0; i < s; ++i) X = (X * X).eval();

    return X;
}