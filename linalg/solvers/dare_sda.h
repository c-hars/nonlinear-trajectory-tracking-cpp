#pragma once
// ============================================================
//  dare_sda.h — solves the DARE
// 
//       P = A'PA - A'PB(R + B'PB)^{-1}B'PA + Q
// 
// 
//  Cold solver (no P0 supplied). Preconditions are unchecked by design (hot-loop solver).
// 
//  Uses the structure-preserving doubling algorithm (SDA):
//       Wk      = I + G_k*H_k
//       A_{k+1} = A_k*(Wk\A_k)
//       G_{k+1} = G_k + A_k*(Wk\G_k)*A_k'
//       H_{k+1} = H_k + A_k'*H_k*(Wk\A_k)
// 
//  Port of dare_sda.m
// 
//  (doi:10.1080/00207170410001714988, doi:10.1002/gamm.202000018)
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

        MatNX AkTH = Ak.transpose() * H;
        H += AkTH * Yk;
        G += (Ak * Zk * Ak.transpose()).eval();
        symmetrise(H);
        symmetrise(G);
        Ak = (Ak * Yk).eval();

        if (i >= min_doublings) {
            if (Ak.norm() < SCALAR_EPS) break; // catches the case where further doublings cannot progress
            if (compute_dare_residual(A, B, Q, R, H) < tolerance) break;
        }
    }

    const MatNX P = H;
    info.solver_iterations = i;
    info.tol_achieved      = compute_dare_residual(A, B, Q, R, P);
    info.solve_success     = (info.tol_achieved < tolerance);
    return P;
}
