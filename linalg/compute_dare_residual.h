#pragma once
// ============================================================
//  linalg/compute_dare_residual.h
// 
//  Residual of the DARE:
//
//    || A_cl' P A_cl - P + Q + K'RK ||_F
// 
//  Unnormalised (same as the MATLAB reference).
//  Lyapunov form (expands to same numerator as Riccati form, has better numerics).
// 
//  NB: only valid for K computed from the same P; the 6-arg form
//  trusts the implementer to only use a valid K.
// 
//  Future work should consider enabling the normalised form for generality.
//  One option: normalise via P (a cheap proxy for full normalisation).
//  But full normalisation is preferable (and not that expensive).
//  Normalised residual: num/den where
//      num = norm(A_cl'*P*A_cl - P + Q + K'*R*K) as above, and
//      den = norm(K'*(R+B'*P*B)*K)) + norm(P) + norm(Q)
//  See doi:10.1002/nla.251 and doi:10.11650/twjm/1500405875.
//
// ============================================================

#include "types/defs.h"
#include "linalg/compute_dare_gain.h"

// 7-arg form: uses a supplied K and S.
template <typename RW>
inline Scalar compute_dare_residual(
    const MatNX& A, const MatNXNU& B,
    const MatNX& Q, const RW& R,
    const MatNX& P, const MatNUNX& K,
    const MatNU& S)
{
    const MatNX A_cl = A - B * K;

    // A_cl'*P*A_cl + Q + K'*R*K for the numerator
    MatNX P_rhs = Q;
    R.add_KtRK(P_rhs, K);
    P_rhs.noalias() += A_cl.transpose() * P * A_cl;
    
    // // K'*(R + B'PB)*K for the denominator (‡)
    // // (†) is largely the same compute as unnormalised; (‡) comes with a ~3% compute penalty
    // // The LDLT of S is also computed elsewhere and known; passed in to speed things up

    // return (P_rhs - P).norm();
    // return (P_rhs - P).norm() / P.norm(); // (†)
    // return (P_rhs - P).norm() / P.norm() / (KtRplusBtPBK.norm() + P.norm() + Q.norm()); // (‡) full DNRes-normalised
    return (P_rhs - P).norm() / P.norm() / ((K.transpose() * S * K).norm() + P.norm() + Q.norm()); // (‡) full DNRes-normalised
}

// 6-arg form: uses a supplied K.
template <typename RW>
inline Scalar compute_dare_residual(
    const MatNX& A, const MatNXNU& B,
    const MatNX& Q, const RW& R,
    const MatNX& P, const MatNUNX& K)
{

    MatNU S;
    R.set_R_plus(S, (B.transpose() * P * B).eval());

    return compute_dare_residual(A,B,Q,R,P,K,S);
}

// 5-arg overload: recomputes K from the supplied P.
template <typename RW>
inline Scalar compute_dare_residual(
    const MatNX& A, const MatNXNU& B,
    const MatNX& Q, const RW& R,
    const MatNX& P)
{
    return compute_dare_residual(A, B, Q, R, P, compute_dare_gain(B, R, P, A));
}
