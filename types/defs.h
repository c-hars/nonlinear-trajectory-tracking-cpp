#pragma once
// ============================================================
//  types/defs.h — Dimensions, scalar type, Eigen aliases
//
//  Port of compute_u_SDDRE_v3 + iterative_dare to C++/Eigen
//  targeting Teensy 4.1 (Cortex-M7, hardware double FPU).
// ============================================================

#include <Eigen/Dense>
#include <limits>

// ---- Compile-time dimensions --------------------------------
constexpr int NX = 12;   // states
constexpr int NU = 6;    // inputs  (== n_rotors)
constexpr int NY = 6;    // outputs (rows of C)

// ---- Scalar type --------------------------------------------
#ifndef SDDRE_USE_FLOAT
  #define SDDRE_USE_FLOAT 0
#endif

#if SDDRE_USE_FLOAT
  using Scalar = float;
#else
  using Scalar = double;
#endif
constexpr Scalar SCALAR_EPS = std::numeric_limits<Scalar>::epsilon();

// ---- Precision-dependent numerical tolerances ---------------

template <typename S> struct sddre_tol;

template <> struct sddre_tol<double> {
    static constexpr double dlyap = 1e-14;           // ~45 * eps(double)
    static constexpr int dlyap_max_doublings = 55;   // practical limit for any Schur-stable matrix; covers rho up to 1 - 2^-48 (16x double precision)
    static constexpr double dare  = 1e-4;
};

template <> struct sddre_tol<float> {
    static constexpr float dlyap = 5e-6f;            // ~42 * eps(float)
    static constexpr int dlyap_max_doublings = 25;   // practical limit for any Schur-stable matrix; covers rho up to 1 - 2^-19 (16x single precision)
    static constexpr float dare  = 1e-4f;
};

using Tol = sddre_tol<Scalar>;

// ---- Fixed-size Eigen types ---------------------------------
using MatNX   = Eigen::Matrix<Scalar, NX, NX>;
using MatNU   = Eigen::Matrix<Scalar, NU, NU>;
using MatNY   = Eigen::Matrix<Scalar, NY, NY>;
using MatNXNU = Eigen::Matrix<Scalar, NX, NU>;
using MatNUNX = Eigen::Matrix<Scalar, NU, NX>;
using MatNYNX = Eigen::Matrix<Scalar, NY, NX>;
using MatNXNY = Eigen::Matrix<Scalar, NX, NY>;
using Mat3    = Eigen::Matrix<Scalar, 3, 3>;
using VecNX   = Eigen::Matrix<Scalar, NX, 1>;
using VecNU   = Eigen::Matrix<Scalar, NU, 1>;
using VecNY   = Eigen::Matrix<Scalar, NY, 1>;

// Augmented (NX+NU) x (NX+NU) for the Van Loan c2d
constexpr int NA = NX + NU;
using MatAug = Eigen::Matrix<Scalar, NA, NA>;

// Wide RHS for the SDA two-right-hand-side solve
using MatNX2 = Eigen::Matrix<Scalar, NX, 2 * NX>;

// ============================================================
//  In-place symmetrisation.
//
//  *** Why not:  P = (P + P.transpose()) * 0.5  ?  ***
//  That aliases: Eigen assigns coefficient-wise, so P(i,j) is
//  overwritten before P(j,i) reads it, and the result is wrong.
//  (Eigen only auto-detects the bare  m = m.transpose()  case,
//  not compound expressions like this one.)
// 
//  The safe Eigen code would be:
//    P = ((P + P.transpose()) * 0.5).eval(),
//  which forces a temporary.
// 
//  This loop is cheaper in any case,
//    n(n-1)/2 = 66 ops for n=12,
//  and no temporary.
// ============================================================
template <typename MatT>
inline void symmetrise(MatT& X) {
    const int n = static_cast<int>(X.rows());
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            const Scalar v = Scalar(0.5) * (X(i, j) + X(j, i));
            X(i, j) = v;
            X(j, i) = v;
        }
}
