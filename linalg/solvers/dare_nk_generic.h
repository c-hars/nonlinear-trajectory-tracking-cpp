#pragma once
// ============================================================
//  dare_nk_generic.h — Solve the DARE iteratively from P0
//
//  Clean Eigen-centric version: returns P only. The caller
//  computes the gain from P as needed. No LLT passthrough.
//
//    NK      — Default. Newton-Kleinman via dlyap (quadratic
//              convergence; P0 must yield a stabilising gain
//              K0). 
//    Riccati — direct DARE recursion (linear convergence; 
//              globally stable from any PSD P0).
//
//  Port of iterative_dare.m
// ============================================================

#include "types/defs.h"
#include "linalg/solvers/solver_types.h"
#include "linalg/solvers/dlyap_sda.h"
#include "linalg/compute_dare_residual.h"

#include <limits>

inline MatNX dare_nk(
    const MatNX& A, const MatNXNU& B,
    const MatNX& Q, const MatNU& R,
    const MatNX& P0,
    const DARESolverOpts& opts,
    DARESolverInfo& info)
{
    info = DARESolverInfo{};

    // --- max_iters == 0: pass-through -----------------------
    if (opts.max_iters == 0) {
        info.tol_achieved = std::numeric_limits<Scalar>::quiet_NaN();
        return P0;
    }

    MatNX P = P0;

    // Inline gain computation — K = (R + B'PB) \ (B'PA)
    auto compute_K = [&](const MatNX& Pk) -> MatNUNX {
        return (R + B.transpose() * Pk * B).llt().solve(B.transpose() * Pk * A);
    };

    switch (opts.method) {

    // ========================================================
    //  Riccati iteration
    //    P_{k+1} = A'P_k A - A'P_k B K_k + Q
    // ========================================================
    case DARESolverMethod::Riccati: {
        const int min_it = opts.min_iters * 25;
        const int max_it = opts.max_iters * 25;

        for (int i = 1; i <= max_it; ++i) {
            const MatNUNX K = compute_K(P);
            P = (A.transpose() * P * A
               - A.transpose() * P * B * K + Q).eval();
            symmetrise(P);

            if (opts.early_break && (i >= min_it) &&
                ((i - min_it) % opts.riccati_check_every == 0))
            {
                const Scalar res = compute_dare_residual(A, B, Q, R, P);
                if (res < opts.tolerance) {
                    info.solver_iterations = i;
                    info.tol_achieved      = res;
                    info.solve_success     = true;
                    return P;
                }
            }
            info.solver_iterations = i;
        }
        break;
    }

    // ========================================================
    //  Newton-Kleinman
    //    K_k = (R + B'P_k B)^{-1} B'P_k A
    //    A_K = A - B K_k
    //    P_{k+1} solves   A_K' P A_K - P + (Q + K'RK) = 0
    //
    //  dlyap_sda solves  M X M' - X + N = 0 (observability
    //  convention, matching the MATLAB-native dlyap), so the
    //  call passes M = A_K'
    // ========================================================
    case DARESolverMethod::NK: {
        for (int i = 1; i <= opts.max_iters; ++i) {
            const MatNUNX K  = compute_K(P);
            const MatNX   AK = A - B * K;
            MatNX         Qk = Q + K.transpose() * R * K;
            symmetrise(Qk);

            DlyapInfo dinfo;
            P = dlyap_sda(AK.transpose(), Qk, dinfo,
                          opts.dlyap_sda_tolerance, opts.dlyap_sda_max_doublings);

            if (i == 1 && !dinfo.is_stable) {
                // Non-converged initial gain, K0 was not in stability basin: NK will diverge.
                // Signal the caller to fall back to a cold SDA solve.
                info.unstable_k0       = true;
                info.solver_iterations = i;
                info.tol_achieved      = compute_dare_residual(A, B, Q, R, P);
                info.solve_success     = false;
                return P;
            }

            if (opts.early_break && (i >= opts.min_iters)) {
                const Scalar res = compute_dare_residual(A, B, Q, R, P);
                if (res < opts.tolerance) {
                    info.solver_iterations = i;
                    info.tol_achieved      = res;
                    info.solve_success     = true;
                    return P;
                }
            }
            info.solver_iterations = i;
        }
        break;
    }

    default:
        break;
    }

    // Only reached when the loop exhausted without early-break return.
    info.tol_achieved  = compute_dare_residual(A, B, Q, R, P);
    info.solve_success = (info.tol_achieved < opts.tolerance);
    return P;
}
