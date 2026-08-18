#pragma once
// ============================================================
//  c2d_zoh.h — ZOH discretisation via Van Loan's method
//              (doi:10.1109/tac.1978.1101743)
//
//  Implements ZOH discretisation but via a reduced 12x12 solve,
//  documented in expm_pade.h. ~2.3x faster than the
//  structure-disregarding baseline.
// 
//  MATLAB reference code (c2d_zoh_expm.m):
//      M = expm([Ac Bc; 0 0] * Ts)
//      Ad = M(1:n, 1:n),  Bd = M(1:n, n+1:end)
// 
// ============================================================

#include "types/defs.h"
#include "linalg/solvers/expm_pade.h"

inline void c2d_zoh_expm(
    const MatNX& Ac, const MatNXNU& Bc, Scalar Ts,
    MatNX& Ad, MatNXNU& Bd)
{
    expm_pade_vanloan(Ac, Bc, Ts, Ad, Bd);
}
