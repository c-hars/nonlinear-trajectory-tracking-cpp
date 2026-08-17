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
#include "platform/mm_kernels.h"

#include <algorithm>
#include <type_traits>

static_assert(NX == 12, "mm12 kernels are hard-coded for 12x12");
static_assert(std::is_same<Scalar, float>::value ||
              std::is_same<Scalar, double>::value,
              "mm_kernels has tile shapes for float and double only");
static_assert(!(MatNX::Flags & Eigen::RowMajorBit), "mm12 kernels assume column-major");

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

    MatNX T, inc, Psq;                        // (‡) hoisted scratch
    int j = 1;
    for (j = 1; j <= max_doublings; ++j) {
        // X is symmetric, so inc = P X P' is too. Form T = P X in
        // full, then only the lower triangle of T P', and mirror —
        // saves roughly half of the second product per doubling.
        // const MatNX T = P * X; // (†)
        // MatNX inc; // (†)
        // inc.noalias() = T * P.transpose(); // (†)
        mm12(T.data(), P.data(), X.data());        // (‡) T   = P X
        // mm12_abt(inc.data(), T.data(), P.data());  // (‡) inc = T P'
        mm12_abt_sym(inc.data(), T.data(), P.data());  // (‡, v2) inc = T P', symmetric
        
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
        // P = (P * P).eval(); // (†)
        mm12(Psq.data(), P.data(), P.data());      // (‡) no-alias temp
        P = Psq;                                   // (‡) replaces P = (P*P).eval()
    }

    // The loop's behaviour: didn't converge within max_doublings -> declared unstable. If the loop hits max_doublings, then info.converged == false and info.is_stable == false (defaults, declared in solver_types.h)
    // Covers both genuine divergence and rho too close to 1 to resolve a solution (non-uniqueness condition not satisfied - the same failure mode as MATLAB's dlyap()).
    // Note: setting info.diverged is redundant (covered by !converged, and info.diverged is not referenced anywhere else in the code) so has not been carried over from the previous commits.

    info.doublings     = j;
    info.rel_increment = ninc / std::max(X.norm(), SCALAR_EPS);
    return X;
}
