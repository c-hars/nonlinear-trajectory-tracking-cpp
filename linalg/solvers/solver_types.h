#pragma once
// ============================================================
//  solver_types.h — solver enums, options, info
// ============================================================

#include "types/defs.h"

#include <limits>

// ---- Enums / option structs ---------------------------------

enum class DARESolverMethod { NK, Riccati, SDA };

struct DARESolverOpts {
    DARESolverMethod method = DARESolverMethod::NK;
    int    min_iters           = 1;       // recommended default
    int    max_iters           = 10;      // conservative default
    bool   early_break         = true;
    Scalar tolerance           = Tol::dare;   // DARE residual threshold
    int    riccati_check_every = 25;     // Riccati branch only

    // --- dlyap_sda (Smith doubling) parameters ---
    Scalar dlyap_sda_tolerance     = Tol::dlyap;
    int    dlyap_sda_max_doublings = Tol::dlyap_max_doublings;

    // --- dare_sda parameters ---
    int    dare_sda_min_doublings   = 5;
    int    dare_sda_max_doublings   = 40;
};

struct DARESolverInfo {
    int    solver_iterations = 0;
    Scalar tol_achieved      = std::numeric_limits<Scalar>::quiet_NaN();
    bool   solve_success     = false;
    bool   unstable_k0       = false;  // NK: initial gain destabilising
    bool   used_sda_fallback = false;
    bool   tol_is_estimate   = false;  // true when tol_achieved came from the Newton-increment (rather than the full DARE residual)
};

// ---- dlyap info ---------------------------------------------
struct DlyapInfo {
    bool   converged     = false; // Do not modify - critical for solver behaviour
    bool   is_stable     = false; // Do not modify - critical for solver behaviour
    int    doublings     = 0;
    Scalar rel_increment = -1; // Sentinel - never possible from genuine solver behaviour
};
