#pragma once
// ============================================================
//  dare_solvers.h — DARE / Stein solvers
//
//  Ports of:
//    dlyap_fast_c.m   — Smith doubling, geometric tail test
//    dare_sda.m       — structure-preserving doubling (DARE)
//    iterative_dare.m — Newton-Kleinman + Riccati iteration
// ============================================================

#include "sddre_types.h"

// ============================================================
//  compute_gain
//    K = (R + B'PB)^{-1} B'PA
// ============================================================
inline MatNUNX compute_gain(
    const MatNXNU& B, const MatNU& R,
    const MatNX& P, const MatNX& A)
{
    const MatNUNX BtP = B.transpose() * P; // B is 12x6, Bt is 6x12, BtP is 6x12 = NuNX
    const MatNU   S   = R + BtP * B;   // 6x6 SPD
    const MatNUNX rhs = BtP * A;       // 6x12
    return S.ldlt().solve(rhs);
}

// ============================================================
//  compute_dare_residual
//
//  *** ASSUMPTION — compute_dare_residual.m was not supplied. ***
//  Implemented here as the relative Frobenius residual of the
//  DARE in closed-loop (Lyapunov) form:
//
//    || A_cl' P A_cl + K'RK + Q - P ||_F / max(||P||_F, 1)
//
//  If your MATLAB version normalises differently (e.g. by
//  ||Q||_F, as dlyap_fast_c/dlyap_schur_real do for their own
//  residuals), change the denominator here — the 1e-4 tolerance
//  is calibrated against YOUR definition, so a mismatch will
//  shift iteration counts and therefore the timings.
// ============================================================
inline Scalar compute_dare_residual(
    const MatNX& A, const MatNXNU& B,
    const MatNX& Q, const MatNU& R,
    const MatNX& P, const MatNUNX& K)
{
    const MatNX A_cl  = A - B * K;
    const MatNX P_rhs = A_cl.transpose() * P * A_cl
                      + K.transpose() * R * K + Q;
    return (P_rhs - P).norm() / std::max(P.norm(), Scalar(1));
}

// 5-arg overload: recomputes K from the supplied P.
inline Scalar compute_dare_residual(
    const MatNX& A, const MatNXNU& B,
    const MatNX& Q, const MatNU& R,
    const MatNX& P)
{
    return compute_dare_residual(A, B, Q, R, P, compute_gain(B, R, P, A));
}

// ============================================================
//  dlyap_fast_c  —  Smith doubling for the Stein equation
//
//      A X A' - X + Q = 0
//
//  (Same orientation as native dlyap, so NK calls it with A_K'.)
//
//  Accumulates  X = sum_k A^k Q (A^k)'  by repeated squaring.
//  Stopping test is the geometric tail extrapolation from the
//  MATLAB version: with r = ninc/ninc_prev < 1, the remaining
//  tail is bounded by ninc*r/(1-r), which is compared against
//  tol*||X||_F. Divergence shows up as Inf/Inf = NaN in r.
//
//  is_stable is only asserted on convergence — running out of
//  doublings leaves it false, which is what feeds the NK
//  unstable-K0 fallback. Deliberate, and matches MATLAB.
// ============================================================
inline MatNX dlyap_fast_c(
    const MatNX& A, const MatNX& Q,
    DlyapInfo& info,
    Scalar tol = Tol::dlyap,
    int    max_doublings = 60)
{
    info = DlyapInfo{};

    MatNX X = Q;
    MatNX P = A;

    Scalar ninc      = std::numeric_limits<Scalar>::infinity();
    Scalar ninc_prev = std::numeric_limits<Scalar>::quiet_NaN();

    int j = 1;
    for (j = 1; j <= max_doublings; ++j) {
        const MatNX inc = P * X * P.transpose();
        X += inc;

        ninc            = inc.norm();
        const Scalar nX = X.norm();
        const Scalar r  = ninc / ninc_prev;   // NaN on the first pass

        if (r < Scalar(1)) {                  // NaN < 1 is false -> skips j==1
            if (ninc * r / (Scalar(1) - r) <= tol * nX) {
                info.converged = true;
                info.is_stable = true;
                break;
            }
        } else if (j > 1 && std::isnan(r)) {
            info.diverged  = true;
            info.is_stable = false;
            break;
        }

        ninc_prev = ninc;
        P = (P * P).eval();
    }

    // Q is symmetric by construction in the NK call (Q + K'RK),
    // so symmetrise unconditionally — no ishermitian() check.
    symmetrise(X);

    info.doublings     = j;
    info.rel_increment = ninc / std::max(X.norm(), SCALAR_EPS);
    return X;
}

// ============================================================
//  dare_sda  —  Structure-preserving doubling for the DARE
//
//      P = A'PA - A'PB(R + B'PB)^{-1}B'PA + Q
//
//  Doubles the invariant subspace of the symplectic pencil:
//      Mk      = I + G_k H_k
//      A_{k+1} = A_k (Mk \ A_k)
//      G_{k+1} = G_k + A_k (Mk \ G_k) A_k'
//      H_{k+1} = H_k + A_k' H_k (Mk \ A_k)
//
//  H increases monotonically to P; A_k -> 0 as rho(A_cl)^(2^k).
//  Reduces exactly to Smith doubling when G = 0.
//  Requires A nonsingular — free for A = expm(Ac*Ts).
//
//  Cold by construction: no P0, no warm start.
// ============================================================
inline MatNX dare_sda(
    const MatNX& A, const MatNXNU& B,
    const MatNX& Q, const MatNU& R,
    Scalar tolerance,
    int    min_doublings,
    int    max_doublings,
    DARESolverInfo& info)
{
    MatNX Ak = A;
    // MatNX G  = B * R.ldlt().solve(B.transpose());  // old  // B R^{-1} B'
    MatNUNX R_inv_Bt = R.ldlt().solve(B.transpose());
    MatNX G = B * R_inv_Bt; // B R^{-1} B'
    symmetrise(G);
    MatNX H  = Q;

    int i = 1;
    for (i = 1; i <= max_doublings; ++i) {
        const MatNX Mk = MatNX::Identity() + G * H;

        // One factorisation, two right-hand sides: [A_k, G_k]
        MatNX2 rhs;
        rhs.template leftCols<NX>()  = Ak;
        rhs.template rightCols<NX>() = G;
        const MatNX2 Ssol = Mk.partialPivLu().solve(rhs);

        const MatNX Yk = Ssol.template leftCols<NX>();   // (I + GH) \ A_k
        const MatNX Zk = Ssol.template rightCols<NX>();  // (I + GH) \ G_k

        // H and G both consume the OLD A_k — update A_k last.
        // .eval() forces a temporary: H and G appear on both sides.
        // H += (Ak.transpose() * H * Yk).eval();   symmetrise(H); // old
        MatNX AkTH = Ak.transpose() * H;
        H += AkTH * Yk;                          symmetrise(H);
        G += (Ak * Zk * Ak.transpose()).eval();  symmetrise(G);
        Ak = (Ak * Yk).eval();

        if (i >= min_doublings) {
            // norm(Ak) is the algorithm's native monitor and nearly
            // free. It also catches the case where further doublings
            // cannot progress, which the residual test alone misses.
            if (Ak.norm() < SCALAR_EPS) break;
            if (compute_dare_residual(A, B, Q, R, H) < tolerance) break;
        }
    }

    const MatNX P = H;
    info.solver_iterations = i;
    info.tol_achieved      = compute_dare_residual(A, B, Q, R, P);
    info.solve_success     = (info.tol_achieved < tolerance);
    return P;
}

// ============================================================
//  iterative_dare  —  solve the DARE iteratively
//
//    NK      — Newton-Kleinman via dlyap (quadratic convergence;
//              P0 must yield a stabilising gain).
//    Riccati — direct DARE recursion (linear convergence;
//              globally stable from any PSD P0).
// ============================================================
inline MatNX iterative_dare(
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

    // --- Early exit if P0 already solves the DARE -----------
    //  Only reachable with min_iters == 0. Pulls in EigenSolver,
    //  which is heavy on Teensy; #define SDDRE_NO_EARLY_EXIT to
    //  drop it from the build entirely.
#ifndef SDDRE_NO_EARLY_EXIT
    if (opts.early_break && opts.min_iters == 0) {
        const MatNUNX K   = compute_gain(B, R, P0, A);
        const Scalar  res = compute_dare_residual(A, B, Q, R, P0, K);
        if (res < opts.tolerance) {
            const MatNX A_cl = A - B * K;
            Eigen::EigenSolver<MatNX> eig(A_cl, /*computeEigenvectors=*/false);
            bool stable = true;
            for (int i = 0; i < NX; ++i)
                if (std::abs(eig.eigenvalues()(i)) >= Scalar(1)) {
                    stable = false;
                    break;
                }
            if (stable) {
                info.tol_achieved      = res;
                info.solve_success     = true;
                info.solver_iterations = 0;
                return P0;
            }
        }
    }
#endif

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
            const MatNUNX K = compute_gain(B, R, P, A);
            P = (A.transpose() * P * A
               - A.transpose() * P * B * K + Q).eval();
            symmetrise(P);

            if (opts.early_break && (i >= min_it) &&
                ((i - min_it) % opts.riccati_check_every == 0))
            {
                // 5-arg: K recomputed from the UPDATED P (matches MATLAB)
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
    //  dlyap_fast_c solves  M X M' - X + N = 0,  so the call
    //  passes M = A_K'. Getting this transpose wrong silently
    //  solves the WRONG Stein equation.
    // ========================================================
    case DARESolverMethod::NK: {
        for (int i = 1; i <= opts.max_iters; ++i) {
            const MatNUNX K  = compute_gain(B, R, P, A);
            const MatNX   AK = A - B * K;
            MatNX         Qk = Q + K.transpose() * R * K;
            symmetrise(Qk);

            DlyapInfo dinfo;
            P = dlyap_fast_c(AK.transpose(), Qk, dinfo,
                             opts.dlyap_tolerance, opts.dlyap_max_doublings);

            if (i == 1 && !dinfo.is_stable) {
                // Unstable / non-converged initial gain: NK will diverge.
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

    info.tol_achieved  = compute_dare_residual(A, B, Q, R, P);
    info.solve_success = (info.tol_achieved < opts.tolerance);
    return P;
}
