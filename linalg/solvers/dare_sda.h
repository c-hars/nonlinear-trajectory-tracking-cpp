#pragma once
// ============================================================
//  linalg/solvers/dare_sda.h — Structure-preserving doubling
//                               for the DARE
//
//      P = A'PA - A'PB(R + B'PB)^{-1}B'PA + Q
//
//  Doubles the invariant subspace of the symplectic pencil:
//      Mk      = I + G_k H_k
//      A_{k+1} = A_k (Mk \ A_k)
//      G_{k+1} = G_k + A_k (Mk \ G_k) A_k'
//      H_{k+1} = H_k + A_k' H_k (Mk \ A_k)
//
//  Cold solver (no P0 supplied).
//
//  Port of dare_sda.m
// ============================================================

#include "types/defs.h"
#include "linalg/solvers/solver_types.h"
#include "linalg/compute_dare_residual.h"

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
