#pragma once
// ============================================================
//  compute_dare_gain.h — DARE gain computation
//
//    K = (R + B'PB)^{-1} B'PA
//
//  3-arg, 5-arg and 6-arg overloads.
//  5-arg also returns the LLT of S = R + B'PB so the
//  caller can reuse the factorisation for the feedforward
//  solve instead of forming and factorising S a second time.
//  6-arg additionally returns S itself, which the NK
//  Newton-increment test needs.
// ============================================================

#include "types/defs.h"

// 6-arg form: returns K, S_llt and S_out.
template <typename RW>
inline MatNUNX compute_dare_gain(
    const MatNXNU& B, const RW& R,
    const MatNX& P, const MatNX& A,
    Eigen::LLT<MatNU>& S_llt, MatNU& S_out)
{
    const MatNUNX BtP = B.transpose() * P;   // 6x12
    R.set_R_plus(S_out, BtP * B);            // 6x6 SPD
    S_llt.compute(S_out);
    return S_llt.solve(BtP * A);
}

// 5-arg form: returns K and S_llt.
template <typename RW>
inline MatNUNX compute_dare_gain(
    const MatNXNU& B, const RW& R,
    const MatNX& P, const MatNX& A,
    Eigen::LLT<MatNU>& S_llt)
{
    MatNU S;
    return compute_dare_gain(B, R, P, A, S_llt, S);
}

// 3-arg form: returns K only.
template <typename RW>
inline MatNUNX compute_dare_gain(
    const MatNXNU& B, const RW& R,
    const MatNX& P, const MatNX& A)
{
    Eigen::LLT<MatNU> S_llt;
    return compute_dare_gain(B, R, P, A, S_llt);
}
