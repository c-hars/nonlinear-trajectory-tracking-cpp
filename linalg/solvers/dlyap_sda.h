#pragma once
// ============================================================
//  dlyap_sda.h — solves the discrete-time Lyapunov equation
// 
//       A*X*A' - X + Q = 0
// 
// 
//  Cold solver (no X0 supplied). Preconditions are unchecked by design (hot-loop solver).
// 
//  Uses the Smith doubling algorithm (SDA). Each iteration computes:
//       dX = P*X*P', X <- X + dX, P <- P^2
// 
//  Convergence requires a Schur-stable A; divergence shows up as reaching max_doublings, providing a stability test.
//  Stopping test is the geometric tail extrapolation (same as the MATLAB version).
// 
//  Port of dlyap_sda.m
// 
//  (doi:10.1137/0116017, doi:10.1002/gamm.202000018)
// ============================================================

#include "types/defs.h"
#include "linalg/solvers/solver_types.h"

#if SDOPT_TEENSY_BUILD
  #include "platform/mm_kernels.h"
  static_assert(NX == 12, "mm12 kernels are hard-coded for 12x12");
  static_assert(std::is_same<Scalar, float>::value ||
                std::is_same<Scalar, double>::value,
                "mm_kernels has tile shapes for float and double only");
  static_assert(!(MatNX::Flags & Eigen::RowMajorBit), "mm12 kernels assume column-major");
#endif

#include <algorithm>
#include <type_traits>

inline MatNX dlyap_sda(
    const MatNX& A, const MatNX& Q,
    DlyapInfo& info,
    Scalar tol = Tol::dlyap,
    int    max_doublings = 60)
{
    info = DlyapInfo{};

    MatNX X = Q;
    MatNX P = A;
    Scalar ninc      = 0;
    Scalar ninc_prev = 0;

    MatNX T, inc;
#if SDOPT_TEENSY_BUILD
    MatNX Psq;
#endif

    int j = 0;
    while (j < max_doublings) {
        ++j;

        // Since X is symmetric, so is the increment P X P'
        // Compute T = P X, then inc = T P'
#if SDOPT_TEENSY_BUILD
        // mm12_abt_sym computes only the lower triangle of T P' (the upper triangle is mirrored)
        mm12(T.data(), P.data(), X.data());            // T   = P X
        mm12_abt_sym(inc.data(), T.data(), P.data());  // inc = T P'
#else
        T = P * X;
        inc.noalias() = T * P.transpose();
#endif
        X += inc;
        ninc = inc.norm();
        const Scalar nX = X.norm();

        // Convergence test (bypassed on first iteration since ninc_prev == 0)
        if (ninc_prev > Scalar(0)) {
            const Scalar r = ninc / ninc_prev;
            if (r < Scalar(1) && ninc * r / (Scalar(1) - r) <= tol * nX) {
                info.converged = true;
                info.is_stable = true;
                break;
            }
        }

        ninc_prev = ninc;

        // P <- P^2
#if SDOPT_TEENSY_BUILD
        mm12(Psq.data(), P.data(), P.data());
        P = Psq;
#else
        P = (P * P).eval();
#endif

    }

    info.doublings = j;
    info.rel_increment = ninc / std::max(X.norm(), SCALAR_EPS);
    return X;
}
