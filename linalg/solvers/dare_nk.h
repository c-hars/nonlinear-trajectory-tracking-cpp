#pragma once
// ============================================================
//  linalg/solvers/dare_nk.h — Solve the DARE iteratively
//                                     from P0 (warm started)
//
//    NK      — Newton-Kleinman via dlyap (quadratic convergence;
//              P0 must yield a stabilising gain K0).
//    Riccati — direct DARE recursion (linear convergence;
//              globally stable from any PSD P0).
//
//  Port of iterative_dare.m
// ============================================================

#include "types/defs.h"
#include "linalg/solvers/solver_types.h"
#include "linalg/solvers/dlyap_sda.h"
#include "linalg/compute_dare_gain.h"
#include "linalg/compute_dare_residual.h"

#include <limits>

template <typename RW>
inline MatNX dare_nk(
    const MatNX& A, const MatNXNU& B,
    const MatNX& Q, const RW& R,
    const MatNX& P0,
    const DARESolverOpts& opts,
    MatNUNX& K_out,
    Eigen::LDLT<MatNU>& S_ldlt,
    DARESolverInfo& info)
{
    info = DARESolverInfo{};

    // --- max_iters == 0: pass-through -----------------------
    if (opts.max_iters == 0) {
        K_out = compute_dare_gain(B, R, P0, A, S_ldlt);
        info.tol_achieved = std::numeric_limits<Scalar>::quiet_NaN();
        return P0;
    }

    // --- Early exit if P0 already solves the DARE -----------
    //  NOT SUPPORTED. Only reachable with min_iters == 0,
    //  which is never recommended.

    MatNX P = P0;

    switch (opts.method) {

    // ========================================================
    //  Riccati iteration
    //    P_{k+1} = A'P_k A - A'P_k B K_k + Q
    // ========================================================
    case DARESolverMethod::Riccati: {
        const int min_it = opts.min_iters * 25;
        const int max_it = opts.max_iters * 25;

        for (int i = 1; i <= max_it; ++i) {
            const MatNUNX K = compute_dare_gain(B, R, P, A);
            P = (A.transpose() * P * A
               - A.transpose() * P * B * K + Q).eval();
            symmetrise(P);

            if (opts.early_break && (i >= min_it) &&
                ((i - min_it) % opts.riccati_check_every == 0))
            {
                // Gain from the updated P, kept for return.
                K_out = compute_dare_gain(B, R, P, A, S_ldlt);
                const Scalar res = compute_dare_residual(A, B, Q, R, P, K_out);
                if (res < opts.tolerance) {
                    info.solver_iterations = i;
                    info.tol_achieved      = res;
                    info.solve_success     = true;
                    return P;
                }
            }
            info.solver_iterations = i;
        }
        // Loop exhausted — refresh the gain for the final P.
        K_out = compute_dare_gain(B, R, P, A, S_ldlt);
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
        MatNU   S;                                       // R + B'PB, explicit
        MatNUNX K = compute_dare_gain(B, R, P, A, S_ldlt, S); // gain of incoming P0

        for (int i = 1; i <= opts.max_iters; ++i) {
            const MatNX AK = A - B * K;

            // Qk = Q + K'RK
            // Open questions:
            //  - is Q exactly symmetric (symmetrised at build)
            //  - is K'RK is exactly symmetric
            // ... or really, where does symmetrise() need to be.
            MatNX Qk = Q;
            R.add_KtRK(Qk, K);
            symmetrise(Qk);
            

            DlyapInfo dinfo;
            P = dlyap_sda(AK.transpose(), Qk, dinfo, opts.dlyap_sda_tolerance, opts.dlyap_sda_max_doublings);

            if (i == 1 && !dinfo.is_stable) {
                // Non-converged initial gain, K0 was not in stability basin: NK will diverge. Signal the caller to fall back to a cold SDA solve.
                // K_out still made consistent with the returned P (contract), though the SDA fallback discards both.
                info.unstable_k0       = true;
                info.solver_iterations = i;
                K_out                  = compute_dare_gain(B, R, P, A, S_ldlt, S);
                info.tol_achieved      = compute_dare_residual(A, B, Q, R, P, K_out);
                info.solve_success     = false;
                return P;
            }

            // Gain of the new P — needed next iteration or as the returned gain either way, and it makes the increment test nearly free.
            const MatNUNX K_new = compute_dare_gain(B, R, P, A, S_ldlt, S);

            if (opts.early_break && (i >= opts.min_iters)) {
                #if USE_INCREMENT_PROXY
                    // Newton-increment identity
                    const MatNUNX dK  = K_new - K;
                    const Scalar  res = (dK.transpose() * (S * dK)).norm();
                    if (res < opts.tolerance) {
                        info.solver_iterations = i;
                        info.tol_achieved      = res;
                        info.tol_is_estimate   = true;
                        info.solve_success     = true;
                        K_out                  = K_new;
                        return P;
                    }
                #else
                    // Explicit DARE residual
                    const Scalar res = compute_dare_residual(A, B, Q, R, P, K_new);
                    if (res < opts.tolerance) {
                        info.solver_iterations = i;
                        info.tol_achieved      = res;
                        info.tol_is_estimate   = false;
                        info.solve_success     = true;
                        K_out                  = K_new;
                        return P;
                    }
                #endif
            }
            K = K_new;
            info.solver_iterations = i;
        }
        K_out = K;
        break;
    }

    default:
        K_out = compute_dare_gain(B, R, P, A, S_ldlt);
        break;
    }

    // Only reached when the loop's exhausted without early-break return.
    // K_out matches P here on every path, so the true residual costs no extra gain computation.
    if (opts.early_break) {
        info.tol_achieved  = compute_dare_residual(A, B, Q, R, P, K_out);
        info.solve_success = (info.tol_achieved < opts.tolerance);
    } else {
        info.tol_achieved = std::numeric_limits<Scalar>::quiet_NaN();
        info.solve_success = false;
    }
    return P;
}
