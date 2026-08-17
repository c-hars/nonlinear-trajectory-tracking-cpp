#pragma once
// ============================================================
//  linalg/compute_dare_residual.h
// 
//  Residual of the DARE in (Lyapunov) form:
//
//    || A_cl' P A_cl + K'RK + Q - P ||_F / ||P||_F
//
//  i.e. normalised according to condition number: MATLAB
//  version does not yet have this: keep in mind re SIL validation.
// 
//  Normalising via P is a cheap proxy for full normalisation
//  (see doi:10.1002/nla.251 and doi:10.11650/twjm/1500405875).
// 
//  NB: only valid for K computed from the same P. A robust
//  version recomputes K from from a given P; usage here trusts
//  the implementer to only use a fresh K from the same P.
//
//  Port of compute_dare_residual.m
// ============================================================

#include "types/defs.h"
#include "linalg/compute_dare_gain.h"

// 6-arg form: uses a supplied K.
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

    return (P_rhs - P).norm() / P.norm();
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
