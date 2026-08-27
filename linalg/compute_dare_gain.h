#pragma once
// ============================================================
//  compute_dare_gain.h — DARE gain computation
//
//    K = (R + B'PB)^{-1} B'PA
//
//  4-arg, 5-arg and 6-arg overloads.
//  4-arg form returns K.
//  5-arg also returns the LLT of S = R + B'PB, so the caller
//  can reuse the factorisation instead of forming and
//  factorising S a second time (e.g. for the feedforward part).
//  6-arg additionally returns S itself (e.g. for the NK
//  Newton-increment test).
// 
//  Templated on RType; callers with a raw MatNU R cannot use
//  this directly.
// 
// ============================================================

#include "types/defs.h"

// 6-arg form: returns K, S_llt and S_out.
template <typename RW>
inline MatNUNX compute_dare_gain(
    const MatNXNU& B, const RW& R,
    const MatNX& P, const MatNX& A,
    Eigen::LLT<MatNU>& S_llt, MatNU& S_out)
{
    const MatNUNX BtP = B.transpose() * P;
    R.set_R_plus(S_out, BtP * B); // S = R + B'PB
    S_llt.compute(S_out);
    return S_llt.solve(BtP * A); // K = inv(R+B'PB)*B'PA
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

// 4-arg form: returns K only.
template <typename RW>
inline MatNUNX compute_dare_gain(
    const MatNXNU& B, const RW& R,
    const MatNX& P, const MatNX& A)
{
    Eigen::LLT<MatNU> S_llt;
    return compute_dare_gain(B, R, P, A, S_llt);
}
