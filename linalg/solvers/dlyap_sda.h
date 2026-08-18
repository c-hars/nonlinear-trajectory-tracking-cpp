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

#if USE_TEENSY_KERNELS
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

#if USE_TEENSY_KERNELS
    MatNX T, inc, Psq;                        // hoisted scratch for kernels
#endif

    int j = 1;
    for (j = 1; j <= max_doublings; ++j) {
        // X is symmetric, so inc = P X P' is too. Form T = P X in
        // full, then only the lower triangle of T P', and mirror —
        // saves roughly half of the second product per doubling.

#if USE_TEENSY_KERNELS
        mm12(T.data(), P.data(), X.data());            // T   = P X
        mm12_abt_sym(inc.data(), T.data(), P.data());  // inc = T P', symmetric
#else
        const MatNX T = P * X;
        MatNX inc;
        inc.noalias() = T * P.transpose();
#endif
        
        X += inc;
        ninc            = inc.norm();
        const Scalar nX = X.norm();

        if (j > 1 && ninc_prev > Scalar(0)) {
            const Scalar r = ninc / ninc_prev;
            if (r < Scalar(1) && ninc * r / (Scalar(1) - r) <= tol * nX) {
                info.converged = true;
                info.is_stable = true;
                break;
            }
        }

        ninc_prev = ninc;

#if USE_TEENSY_KERNELS
        mm12(Psq.data(), P.data(), P.data());      // no-alias temp
        P = Psq;
#else
        P = (P * P).eval();
#endif
    }

    // The loop's behaviour: didn't converge within max_doublings -> declared unstable. If the loop hits max_doublings, then info.converged == false and info.is_stable == false (defaults, declared in solver_types.h)
    // Covers both genuine divergence and rho too close to 1 to resolve a solution (non-uniqueness condition not satisfied - the same failure mode as MATLAB's dlyap()).

    info.doublings     = j;
    info.rel_increment = ninc / std::max(X.norm(), SCALAR_EPS);
    return X;
}
