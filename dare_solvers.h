#pragma once
// ============================================================
//  dare_solvers.h — DARE / Stein solvers
//
//  Ports of:
//    dlyap_fast_c.m   — Smith doubling DLYAP solver
//    dare_sda.m       — structure-preserving doubling (DARE)
//    iterative_dare.m — Newton-Kleinman + Riccati iteration
//
//  C6: every function taking an input weight is templated on
//  RW = RWeight<Mode> rather than taking MatNU. All arithmetic
//  involving R goes through the RWeight operations, so the
//  scalar and dense modes share one body.
// ============================================================

#include "sddre_types.h"
#include "mm_kernels.h"

// ============================================================
//  compute_gain
//    K = (R + B'PB)^{-1} B'PA
//  5-arg form also returns the LDLT of S = R + B'PB so the
//  caller can reuse the factorisation for the feedforward
//  solve instead of forming and factorising S a second time.
// ============================================================
// 6-arg form additionally returns S itself, which the NK
// Newton-increment test needs explicitly (LDLT alone doesn't
// hand S back cheaply).
template <typename RW>
inline MatNUNX compute_gain(
    const MatNXNU& B, const RW& R,
    const MatNX& P, const MatNX& A,
    Eigen::LDLT<MatNU>& S_ldlt, MatNU& S_out)
{
    const MatNUNX BtP = B.transpose() * P;   // 6x12
    R.set_R_plus(S_out, BtP * B);            // 6x6 SPD
    S_ldlt.compute(S_out);
    return S_ldlt.solve(BtP * A);
}

template <typename RW>
inline MatNUNX compute_gain(
    const MatNXNU& B, const RW& R,
    const MatNX& P, const MatNX& A,
    Eigen::LDLT<MatNU>& S_ldlt)
{
    MatNU S;
    return compute_gain(B, R, P, A, S_ldlt, S);
}

template <typename RW>
inline MatNUNX compute_gain(
    const MatNXNU& B, const RW& R,
    const MatNX& P, const MatNX& A)
{
    Eigen::LDLT<MatNU> S_ldlt;
    return compute_gain(B, R, P, A, S_ldlt);
}

// ============================================================
//  compute_dare_residual
//
//  *** NOTE ***
//  Implemented here as the relative Frobenius residual of the
//  DARE in closed-loop (Lyapunov) form:
//
//    || A_cl' P A_cl + K'RK + Q - P ||_F / max(||P||_F, 1)
//
//  i.e. normalised according to condition number: MATLAB
//  version does not yet have this: the mismatch may
//  shift iteration counts - keep in mind re SIL validation.
// ============================================================
template <typename RW>
inline Scalar compute_dare_residual(
    const MatNX& A, const MatNXNU& B,
    const MatNX& Q, const RW& R,
    const MatNX& P, const MatNUNX& K)
{
    const MatNX A_cl = A - B * K;

    MatNX P_rhs = Q;
    R.add_KtRK(P_rhs, K);
    P_rhs.noalias() += A_cl.transpose() * P * A_cl;

    return (P_rhs - P).norm() / std::max(P.norm(), Scalar(1));
}

// 5-arg overload: recomputes K from the supplied P.
template <typename RW>
inline Scalar compute_dare_residual(
    const MatNX& A, const MatNXNU& B,
    const MatNX& Q, const RW& R,
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
//  tol*||X||_F. Divergence shows up as reaching max_doublings.
//
//  is_stable is only asserted on convergence — running out of
//  doublings leaves it false, which is what feeds the NK
//  unstable-K0 fallback. Deliberate, and matches MATLAB.
// ============================================================

static_assert(NX == 12, "mm12 kernels are hard-coded for 12x12");
static_assert(std::is_same<Scalar, double>::value, "mm12 kernels assume double");
static_assert(!(MatNX::Flags & Eigen::RowMajorBit), "mm12 kernels assume column-major");

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
        // P = (P * P).eval(); // (†)
        mm12(Psq.data(), P.data(), P.data());      // (‡) no-alias temp
        P = Psq;                                   // (‡) replaces P = (P*P).eval()
    }

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
//
//  C6: R enters exactly once, in the initial G = B R^-1 B'.
//  Scalar mode turns that from an LDLT plus a 6-RHS solve into
//  a scaled outer product. Real, but this path is cold — it
//  runs at k == 1 and on the NK fallback only.
// ============================================================
template <typename RW>
inline MatNX dare_sda(
    const MatNX& A, const MatNXNU& B,
    const MatNX& Q, const RW& R,
    Scalar tolerance,
    int    min_doublings,
    int    max_doublings,
    DARESolverInfo& info)
{
    MatNX Ak = A;
    MatNX G  = R.B_Rinv_Bt(B);   // B R^{-1} B'
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
//  iterative_dare  —  solve the DARE iteratively from P0
//
//    NK      — Newton-Kleinman via dlyap (quadratic convergence;
//              P0 must yield a stabilising gain).
//    Riccati — direct DARE recursion (linear convergence;
//              globally stable from any PSD P0).
//
//  C1 — the NK convergence test uses the Newton-increment
//  identity instead of the full DARE residual. For the exact
//  Stein solve (completion of squares on the DARE),
//
//      R(P_{i+1}) = -(K_{i+1}-K_i)' (R + B'P_{i+1}B) (K_{i+1}-K_i)
//
//  so ||dK' S dK||_F / max(||P||_F, 1) IS the relative residual,
//  exact up to the dlyap tolerance — and K_{i+1}, S are needed
//  for the next iteration anyway, so the check is nearly free
//  (no A_cl, no A_cl'PA_cl). tol_is_estimate is set on this
//  path; enable SDDREOpts::post_residual_check (C3) to get the
//  directly-evaluated residual back as a health signal.
//
//  C2 — returns the gain of the final P in K_out and its
//  S = R + B'PB factorisation in S_ldlt, so the caller doesn't
//  recompute either.
//
//  C6 — the increment test still needs S explicitly, and S is
//  a full 6x6 in both modes (B'PB is dense), so that line is
//  unchanged. The savings here are in Qk and in compute_gain.
// ============================================================
template <typename RW>
inline MatNX iterative_dare(
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
        K_out = compute_gain(B, R, P0, A, S_ldlt);
        info.tol_achieved = std::numeric_limits<Scalar>::quiet_NaN();
        return P0;
    }

    // --- Early exit if P0 already solves the DARE -----------
    //  Only reachable with min_iters == 0. Pulls in EigenSolver,
    //  which is heavy on Teensy; #define SDDRE_NO_EARLY_EXIT to
    //  drop it from the build entirely.
#ifndef SDDRE_NO_EARLY_EXIT
    if (opts.early_break && opts.min_iters == 0) {
        const MatNUNX K   = compute_gain(B, R, P0, A, S_ldlt);
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
                K_out                  = K;
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
                // Gain from the UPDATED P (matches MATLAB), kept for return.
                K_out = compute_gain(B, R, P, A, S_ldlt);
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
        K_out = compute_gain(B, R, P, A, S_ldlt);
        break;
    }

    // ========================================================
    //  Newton-Kleinman
    //    K_k = (R + B'P_k B)^{-1} B'P_k A
    //    A_K = A - B K_k
    //    P_{k+1} solves   A_K' P A_K - P + (Q + K'RK) = 0
    //
    //  dlyap_fast_c solves  M X M' - X + N = 0 (observability
    //  convention, matching the MATLAB- native dlyap), so the
    //  call passes M = A_K'
    // ========================================================
    case DARESolverMethod::NK: {
        MatNU   S;                                       // R + B'PB, explicit
        MatNUNX K = compute_gain(B, R, P, A, S_ldlt, S); // gain of incoming P0

        for (int i = 1; i <= opts.max_iters; ++i) {
            const MatNX AK = A - B * K;

            // Qk = Q + K'RK. Q is exactly symmetric (symmetrised at
            // build); K'RK is symmetric in exact arithmetic but not
            // bit-exact from a general gemm, so symmetrise stays.
            MatNX Qk = Q;
            R.add_KtRK(Qk, K);
            symmetrise(Qk);
            

            DlyapInfo dinfo;
            P = dlyap_fast_c(AK.transpose(), Qk, dinfo,
                             opts.dlyap_tolerance, opts.dlyap_max_doublings);

            if (i == 1 && !dinfo.is_stable) {
                // Non-converged initial gain, K0 was not in stability basin: NK will diverge.
                // Signal the caller to fall back to a cold SDA solve.
                // K_out still made consistent with the returned P (contract),
                // though the SDA fallback discards both.
                info.unstable_k0       = true;
                info.solver_iterations = i;
                K_out                  = compute_gain(B, R, P, A, S_ldlt, S);
                info.tol_achieved      = compute_dare_residual(A, B, Q, R, P, K_out);
                info.solve_success     = false;
                return P;
            }

            // Gain of the NEW P — needed next iteration or as the returned
            // gain either way, and it makes the increment test nearly free.
            const MatNUNX K_new = compute_gain(B, R, P, A, S_ldlt, S);

            if (opts.early_break && (i >= opts.min_iters)) {
                // C1: Newton-increment identity (see header comment).
                const MatNUNX dK  = K_new - K;
                const Scalar  res = (dK.transpose() * (S * dK)).norm()
                                  / std::max(P.norm(), Scalar(1));
                if (res < opts.tolerance) {
                    info.solver_iterations = i;
                    info.tol_achieved      = res;
                    info.tol_is_estimate   = true;
                    info.solve_success     = true;
                    K_out                  = K_new;
                    return P;
                }
            }
            K = K_new;
            info.solver_iterations = i;
        }
        K_out = K;   // == gain of the final P; S_ldlt already matches
        break;
    }

    default:
        K_out = compute_gain(B, R, P, A, S_ldlt);
        break;
    }

    // Loop exhausted without early-break return. K_out matches P here on
    // every path, so the true residual costs no extra gain computation.
    if (opts.early_break) {
        info.tol_achieved  = compute_dare_residual(A, B, Q, R, P, K_out);
        info.solve_success = (info.tol_achieved < opts.tolerance);
    } else {
        info.tol_achieved = std::numeric_limits<Scalar>::quiet_NaN();
        info.solve_success = false;
    }
    return P;
}