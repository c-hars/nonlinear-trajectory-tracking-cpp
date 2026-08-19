#pragma once
// ============================================================
//  sdopt_opts.h — Controller-level option / info structs
//
//  Shared by both the generic and Teensy controller variants.
// ============================================================

#include "linalg/solvers/solver_types.h"

struct SDOPTOpts {
    Scalar preview_horizon             = 2.0;   // [s]
    bool   use_full_fh_mpc_at_terminal = false;
    bool   always_use_full_fh_mpc      = false;
    bool   post_residual_check         = false;  // Teensy path only; ignored by generic
    DARESolverOpts dare;
};

struct SDOPTSolveInfo {
    Scalar         time_sdc_discretize_us = 0;
    Scalar         time_dare_us           = 0;
    Scalar         time_feedforward_us    = 0;
    DARESolverInfo dare_info;
}; 
