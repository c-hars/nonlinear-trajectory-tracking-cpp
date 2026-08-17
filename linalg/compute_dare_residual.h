#pragma once
// ============================================================
//  linalg/compute_dare_residual.h
// 
//  Residual of the DARE:
//
//    || A_cl' P A_cl - P + Q + K'RK ||_F
// 
//  Unnormalised: same as the MATLAB reference.
//  Lyapunov form: expands to same numerator as Riccati form, has better numerics.
//  Only valid for K computed from the same P: the 6-arg form trusts the implementer to only use a valid K.
// 
//  Future work: consider using the normalised form (†) or (‡) for better generality.
//    (‡): normalised residual = num/den where
//           num = norm(A_cl'*P*A_cl - P + Q + K'*R*K) as above, and
//           den = norm(P) + norm(Q) + norm(K'*(R+B'*P*B)*K))
//         which costs about 3% extra compute.
//    (†) uses the same numerator as (‡) but with
//          den = norm(P)
//        only. Much cheaper – effectively the same compute as unnormalised – and a fair approximation.
//  See doi:10.1002/nla.251 and doi:10.11650/twjm/1500405875 for more info.
//  Note: if switching to normalised, the Newton increment test in dare_nk.h also needs normalisation.
//  
//
// ============================================================

#include "types/defs.h"
#include "linalg/compute_dare_gain.h"

// 6-arg form: uses a supplied K and S.
template <typename RW>
inline Scalar compute_dare_residual(
    const MatNX& A, const MatNXNU& B,
    const MatNX& Q, const RW& R,
    const MatNX& P, const MatNUNX& K)
{
    const MatNX A_cl = A - B * K;

    // A_cl'*P*A_cl + Q + K'*R*K for the numerator
    MatNX P_rhs = Q;
    R.add_KtRK(P_rhs, K);
    P_rhs.noalias() += A_cl.transpose() * P * A_cl;
    
    // // // K'*(R + B'PB)*K, for the denominator (‡)
    // MatNU S;
    // R.set_R_plus(S, (B.transpose() * P * B).eval());
    // const MatNX KtRplusBtPBK = K.transpose() * S * K;

    return (P_rhs - P).norm();
    // return (P_rhs - P).norm() / P.norm(); // (†)
    // return (P_rhs - P).norm() / P.norm() / (KtRplusBtPBK.norm() + P.norm() + Q.norm()); // (‡) full DNRes-normalised
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
