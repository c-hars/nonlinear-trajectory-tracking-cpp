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

#if SDOPT_TEENSY_BUILD
template <typename RW>
inline MatNX dare_sda(
    const MatNX& A, const MatNXNU& B,
    const MatNX& Q, const RW& R,
#else
inline MatNX dare_sda(
    const MatNX& A, const MatNXNU& B,
    const MatNX& Q, const MatNU& R,
#endif
    Scalar tolerance,
    int    min_doublings,
    int    max_doublings,
    DARESolverInfo& info)
{
    info = DARESolverInfo{};

    MatNX Ak = A;

#if SDOPT_TEENSY_BUILD
    MatNX G = R.B_Rinv_Bt(B);
#else
    MatNX G = B * R.llt().solve(B.transpose());   // B R^{-1} B'
#endif
    symmetrise(G);
    MatNX H  = Q;

    Scalar res = 0; // sentinel
    bool res_valid = false;
    int i;
    for (i = 1; i <= max_doublings; ++i) {
        
        // Solve for Yk, Zk
        const MatNX Wk = MatNX::Identity() + G * H;
        MatNX2 rhs;
        rhs.template leftCols<NX>()  = Ak;
        rhs.template rightCols<NX>() = G;
        const MatNX2 S = Wk.partialPivLu().solve(rhs); // one factorisation, two right-hand sides
        const MatNX Yk = S.template leftCols<NX>();    // Yk = (I + GH) \ A_k
        const MatNX Zk = S.template rightCols<NX>();   // Zk = (I + GH) \ G_k

        H += (Ak.transpose() * H * Yk).eval();
        G += (Ak * Zk * Ak.transpose()).eval();
        symmetrise(H);
        symmetrise(G);
        Ak = (Ak * Yk).eval();

        if (i >= min_doublings) {
            if (Ak.norm() < SCALAR_EPS) break; // catches the case where further doublings cannot progress
            res = compute_dare_residual(A, B, Q, R, H);
            if (res < tolerance) {res_valid = true; break;}
        }
    }

    if (!res_valid) {
        res = compute_dare_residual(A, B, Q, R, H);
        res_valid = true;
    }

    info.solver_iterations = i;
    info.tol_achieved      = res;
    info.solve_success     = (res < tolerance);
    return H; // == P (the cost-to-go matrix, just a different notation)
}
