#pragma once
// ============================================================
//  expm_pade.h — Matrix exponential, scaling-and-squaring Padé
//
//  Higham (2005), "The scaling and squaring method for the
//  matrix exponential revisited", SIAM J. Matrix Anal. Appl.
//  26(4):1179-1193.  Same algorithm family as MATLAB's expm.
//
//  Hand-rolled rather than <unsupported/Eigen/MatrixFunctions>
//  so the cost is predictable, the code is auditable, and the
//  exactness can be tuned — machine precision by default, but
//  this is is likely not needed in practice: a dial for
//  optimisation later, as required. And for this Teensy timing
//  benchmark you want to know exactly which branch you're
//  paying for.
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
// ============================================================

// doi:10.1137/1.9780898717778.ch10
// https://www.cis.upenn.edu/~cis6100/higham_matrix_exponential_siam_2004.pdf

#include <Eigen/Dense>
#include <cmath>
#include <type_traits>
#include "mm_kernels.h"

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

// Dense Eigen matrix: N diagonal adds instead of N² fill + N² scaled add.
template <typename MatT>
inline void diag_add(MatT& X, typename MatT::Scalar s) {
    X.diagonal().array() += s;
}

template <typename MatT>
inline void pade3(const MatT& A, MatT& U, MatT& V) {
    using S = typename MatT::Scalar;
    const MatT A2 = A * A;
    MatT t = A2;
    diag_add(t, S(60.0));
    U = A * t;
    V = S(12.0) * A2;
    diag_add(V, S(120.0));
}

template <typename MatT>
inline void pade5(const MatT& A, MatT& U, MatT& V) {
    using S = typename MatT::Scalar;
    const MatT A2 = A * A;
    const MatT A4 = A2 * A2;
    MatT t = A4 + S(420.0) * A2;
    diag_add(t, S(15120.0));
    U = A * t;
    V = S(30.0) * A4 + S(3360.0) * A2;
    diag_add(V, S(30240.0));
}

template <typename MatT>
inline void pade7(const MatT& A, MatT& U, MatT& V) {
    using S = typename MatT::Scalar;
    const MatT A2 = A * A;
    const MatT A4 = A2 * A2;
    const MatT A6 = A4 * A2;
    MatT t = A6 + S(1512.0) * A4 + S(277200.0) * A2;
    diag_add(t, S(8648640.0));
    U = A * t;
    V = S(56.0) * A6 + S(25200.0) * A4 + S(1995840.0) * A2;
    diag_add(V, S(17297280.0));
}

// --- double-only below this line ---

template <typename MatT>
inline void pade9(const MatT& A, MatT& U, MatT& V) {
    using S = typename MatT::Scalar;
    const MatT A2 = A * A;
    const MatT A4 = A2 * A2;
    const MatT A6 = A4 * A2;
    const MatT A8 = A6 * A2;
    MatT t = A8 + S(3960.0) * A6 + S(2162160.0) * A4
           + S(302702400.0) * A2;
    diag_add(t, S(8821612800.0));
    U = A * t;
    V = S(90.0) * A8 + S(110880.0) * A6 + S(30270240.0) * A4
      + S(2075673600.0) * A2;
    diag_add(V, S(17643225600.0));
}

template <typename MatT>
inline void pade13(const MatT& A, MatT& U, MatT& V) {
    using S = typename MatT::Scalar;
    const MatT A2 = A * A;
    const MatT A4 = A2 * A2;
    const MatT A6 = A4 * A2;

    MatT t = A6 * (A6 + S(16380.0) * A4 + S(40840800.0) * A2)
           + S(33522128640.0)       * A6
           + S(10559470521600.0)    * A4
           + S(1187353796428800.0)  * A2;
    diag_add(t, S(32382376266240000.0));
    U = A * t;

    V = A6 * (S(182.0) * A6 + S(960960.0) * A4
             + S(1323241920.0) * A2)
      + S(670442572800.0)      * A6
      + S(129060195264000.0)   * A4
      + S(7771770303897600.0)  * A2;
    diag_add(V, S(64764752532480000.0));
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

    const S nA = Ain.cwiseAbs().colwise().sum().maxCoeff();

    MatT A = Ain;
    MatT U, V;
    int  s = 0;

    if (nA <= Th::th3) {
        expm_detail::pade3(A, U, V);

    } else if (nA <= Th::th5) {
        expm_detail::pade5(A, U, V);

    } else if constexpr (Th::max_degree == 7) {
        if (nA > Th::th_top) {
            s = static_cast<int>(std::ceil(std::log2(double(nA / Th::th_top))));
            if (s < 0) s = 0;
            A = Ain * std::ldexp(S(1), -s);
        }
        expm_detail::pade7(A, U, V);

    } else {
        if (nA <= Th::th7) {
            expm_detail::pade7(A, U, V);
        } else if (nA <= Th::th9) {
            expm_detail::pade9(A, U, V);
        } else {
            if (nA > Th::th_top) {
                s = static_cast<int>(std::ceil(std::log2(double(nA / Th::th_top))));
                if (s < 0) s = 0;
                A = Ain * std::ldexp(S(1), -s);
            }
            expm_detail::pade13(A, U, V);
        }
    }

    MatT X = (V - U).partialPivLu().solve(V + U);
    for (int i = 0; i < s; ++i) X = (X * X).eval();
    return X;
}

#include <algorithm>   // std::max

// ============================================================
//  Block-structured Van Loan path.
//
//  For  F = [A B; 0 0]  every power keeps the shape
//
//    Z = [ P    Q  ]     bottom-left structurally zero,
//        [ 0   c*I ]     bottom-right always a multiple of I
//
//  because F^k = [A^k  A^(k-1) B; 0  0] for k >= 1.  That set is
//  closed under +, -, scalar*, and *, so the Pade builders above
//  run verbatim on the triple (P,Q,c) and never touch a zero
//  block.  The coefficient tables are shared, not forked — only
//  the driver differs (norm, solve, squaring).
//
//  Per matrix product:  n^2(n+m) instead of (n+m)^3.
//  NX=12, NU=6 -> 2592 vs 5832 multiply-adds.
// ============================================================

namespace expm_detail {

template <typename S_, int N, int M>
struct BlockUT {
    using Scalar = S_;
    static constexpr int RowsAtCompileTime = N + M;   // for expm_pade's assert

    Eigen::Matrix<Scalar, N, N> P;
    Eigen::Matrix<Scalar, N, M> Q;
    Scalar                      c;

    static BlockUT Identity() {
        return { Eigen::Matrix<Scalar, N, N>::Identity(),
                 Eigen::Matrix<Scalar, N, M>::Zero(),
                 Scalar(1) };
    }
};

// (P1,Q1,c1) * (P2,Q2,c2) = (P1 P2,  P1 Q2 + c2 Q1,  c1 c2)
template <typename S, int N, int M>
inline BlockUT<S,N,M> operator*(const BlockUT<S,N,M>& a,
                                const BlockUT<S,N,M>& b) {
    return { a.P * b.P, a.P * b.Q + b.c * a.Q, a.c * b.c };
}

// Kernel-backed product for the concrete Teensy case.
// (P1,Q1,c1)*(P2,Q2,c2) = (P1 P2,  P1 Q2 + c2 Q1,  c1 c2)
template <typename S>
inline BlockUT<S,12,6> operator*(const BlockUT<S,12,6>& a,
                                 const BlockUT<S,12,6>& b) {
    BlockUT<S,12,6> r;
    mm12  (r.P.data(), a.P.data(), b.P.data());
    mm12x6(r.Q.data(), a.P.data(), b.Q.data());
    r.Q.noalias() += b.c * a.Q;
    r.c = a.c * b.c;
    return r;
}

template <typename S, int N, int M>
inline BlockUT<S,N,M> operator+(const BlockUT<S,N,M>& a,
                                const BlockUT<S,N,M>& b) {
    return { a.P + b.P, a.Q + b.Q, a.c + b.c };
}

template <typename S, int N, int M>
inline BlockUT<S,N,M> operator-(const BlockUT<S,N,M>& a,
                                const BlockUT<S,N,M>& b) {
    return { a.P - b.P, a.Q - b.Q, a.c - b.c };
}

template <typename S, int N, int M>
inline BlockUT<S,N,M> operator*(const S& k, const BlockUT<S,N,M>& a) {
    return { k * a.P, k * a.Q, k * a.c };
}

template <typename S, int N, int M>
inline void diag_add(BlockUT<S,N,M>& X, S s) {
    X.P.diagonal().array() += s;
    X.c += s;
}

}  // namespace expm_detail


// ------------------------------------------------------------
//  expm([Ac Bc; 0 0] * Ts) -> (Ad, Bd), without ever forming
//  the (N+M) x (N+M) matrix.
//
//  MUST be a template: the degree ladder names th9/th13, which
//  do not exist in the float traits.  Only a dependent Th lets
//  if-constexpr discard that branch un-instantiated.
// ------------------------------------------------------------
template <typename S, int N, int M>
void expm_pade_vanloan(const Eigen::Matrix<S,N,N>& Ac,
                       const Eigen::Matrix<S,N,M>& Bc,
                       const S                     Ts,
                       Eigen::Matrix<S,N,N>&       Ad,
                       Eigen::Matrix<S,N,M>&       Bd)
{
    using Th  = expm_pade_traits<S>;
    using Blk = expm_detail::BlockUT<S,N,M>;
    static_assert(N > 0 && M > 0, "expm_pade_vanloan needs non-empty blocks");

    Blk       F { Ac * Ts, Bc * Ts, S(0) };

    // ||F||_1 = max column sum.  Bottom block row is zero, so this
    // is just the larger of the two blocks' max column sums.
    const S nA = std::max(F.P.cwiseAbs().colwise().sum().maxCoeff(),
                          F.Q.cwiseAbs().colwise().sum().maxCoeff());

    Blk U, V;
    int s = 0;

    auto rescale = [&](S theta) {
        s = static_cast<int>(std::ceil(std::log2(double(nA / theta))));
        if (s < 0) s = 0;
        const S f = std::ldexp(S(1), -s);
        F.P *= f;
        F.Q *= f;
    };

    if (nA <= Th::th3) {
        expm_detail::pade3(F, U, V);

    } else if (nA <= Th::th5) {
        expm_detail::pade5(F, U, V);

    } else if constexpr (Th::max_degree == 7) {
        // ---- single precision: degree 7 is the top ----
        if (nA > Th::th_top) rescale(Th::th_top);
        expm_detail::pade7(F, U, V);

    } else {
        // ---- double precision: 7, 9, then 13 with scaling ----
        if (nA <= Th::th7) {
            expm_detail::pade7(F, U, V);
        } else if (nA <= Th::th9) {
            expm_detail::pade9(F, U, V);
        } else {
            if (nA > Th::th_top) rescale(Th::th_top);
            expm_detail::pade13(F, U, V);
        }
    }

    // r_m(F) = (V - U)^{-1}(V + U).
    //   U.c == 0 always — every term carries F to an odd power >= 1 —
    //   so W.c == Y.c == the Pade constant term, which is never zero.
    //   Solving W X = Y then forces X.c = 1 and leaves
    //     X.P = W.P^{-1} Y.P
    //     X.Q = W.P^{-1} (Y.Q - W.Q)
    //   One N x N factorisation, N + M right-hand sides, instead of
    //   an (N+M) x (N+M) one.
    const Blk  W  = V - U;
    const Blk  Y  = V + U;
    const auto lu = W.P.partialPivLu();

    Blk X;
    X.P = lu.solve(Y.P);
    X.Q = lu.solve(Y.Q - W.Q);
    X.c = S(1);

    // Undo the scaling.  X^2 = (X.P^2, X.P X.Q + X.Q, 1).
    // Safe: operator* builds a full temporary before the assignment.
    for (int i = 0; i < s; ++i) X = X * X;

    Ad = X.P;
    Bd = X.Q;
}