#pragma once
// ============================================================
//  c2d_zoh.h — ZOH discretisation
//
//  ZOH discretisation via the matrix exponential (Van Loan).
//  Uses a structure-exploiting Pade solver for ~2.3x speedup
//  over a structure-disregarding baseline (see expm_pade.h
//  for details).
// 
//  MATLAB reference code (c2d_zoh_expm.m):
//      M = expm([Ac Bc; 0 0] * Ts)
//      Ad = M(1:n, 1:n),  Bd = M(1:n, n+1:end)
// 
// doi:10.1109/tac.1978.1101743
// ============================================================

#include "types/defs.h"
#include "linalg/solvers/expm_pade.h"

inline void c2d_zoh_expm(
    const MatNX& Ac, const MatNXNU& Bc, Scalar Ts,
    MatNX& Ad, MatNXNU& Bd)
{
    expm_pade_vanloan(Ac, Bc, Ts, Ad, Bd);
}
