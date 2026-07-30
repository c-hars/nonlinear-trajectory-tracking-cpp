#pragma once
// ============================================================
//  sddre_types.h — Dimensions, type aliases, config structs
//
//  Port of compute_u_SDDRE_v3 + iterative_dare to C++/Eigen
//  targeting Teensy 4.1 (Cortex-M7, hardware double FPU).
// ============================================================

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <algorithm>

// ---- Compile-time dimensions --------------------------------
constexpr int NX = 12;   // states
constexpr int NU = 6;    // inputs  (== n_rotors)
constexpr int NY = 6;    // outputs (rows of C)

// ---- Scalar type --------------------------------------------
#ifndef SDDRE_USE_FLOAT
  #define SDDRE_USE_FLOAT 1
#endif

#if SDDRE_USE_FLOAT
  using Scalar = float;
#else
  using Scalar = double;
#endif
constexpr Scalar SCALAR_EPS = std::numeric_limits<Scalar>::epsilon();

// ---- Precision-dependent numerical tolerances ---------------
//  Kept together deliberately: every constant here is tied to the
//  unit roundoff of Scalar, and is wrong if blindly copied across
//  precisions.

template <typename S> struct sddre_tol;

template <> struct sddre_tol<double> {
    static constexpr double dlyap = 1e-14;   // ~45 * eps(double)
    static constexpr double dare  = 1e-4;
};

template <> struct sddre_tol<float> {
    static constexpr float dlyap = 5e-6f; // ~42 * eps(float)
    static constexpr float dare  = 1e-4f;
};

using Tol = sddre_tol<Scalar>;

// ---- Fixed-size Eigen types ---------------------------------
//  All stack-allocated — no heap in the hot path.
using MatNX   = Eigen::Matrix<Scalar, NX, NX>;
using MatNU   = Eigen::Matrix<Scalar, NU, NU>;
using MatNY   = Eigen::Matrix<Scalar, NY, NY>;
using MatNXNU = Eigen::Matrix<Scalar, NX, NU>;
using MatNUNX = Eigen::Matrix<Scalar, NU, NX>;
using MatNYNX = Eigen::Matrix<Scalar, NY, NX>;
using MatNXNY = Eigen::Matrix<Scalar, NX, NY>;
using Mat3    = Eigen::Matrix<Scalar, 3, 3>;
using VecNX   = Eigen::Matrix<Scalar, NX, 1>;
using VecNU   = Eigen::Matrix<Scalar, NU, 1>;
using VecNY   = Eigen::Matrix<Scalar, NY, 1>;

// Augmented (NX+NU) x (NX+NU) for the Van Loan c2d
constexpr int NA = NX + NU;
using MatAug = Eigen::Matrix<Scalar, NA, NA>;

// Wide RHS for the SDA two-right-hand-side solve
using MatNX2 = Eigen::Matrix<Scalar, NX, 2 * NX>;

// ============================================================
//  In-place symmetrisation.
//
//  *** DO NOT write  P = (P + P.transpose()) * 0.5;  ***
//  That aliases: Eigen assigns coefficient-wise, so P(i,j) is
//  overwritten before P(j,i) reads it, and the result is wrong.
//  (Eigen only auto-detects the bare  m = m.transpose()  case,
//  not compound expressions like this one.)
//
//  This loop is also cheaper — n(n-1)/2 = 66 ops for n=12,
//  and no temporary.
// ============================================================
template <typename MatT>
inline void symmetrise(MatT& X) {
    const int n = static_cast<int>(X.rows());
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            const Scalar v = Scalar(0.5) * (X(i, j) + X(j, i));
            X(i, j) = v;
            X(j, i) = v;
        }
}

// ---- Enums / option structs ---------------------------------

enum class DARESolverMethod { NK, Riccati, SDA };

struct DARESolverOpts {
    DARESolverMethod method = DARESolverMethod::NK;
    int    min_iters           = 1;       // recommended default
    int    max_iters           = 10;      // conservative default
    bool   early_break         = true;
    Scalar tolerance           = Tol::dare;   // DARE residual threshold
    int    riccati_check_every = 25;     // Riccati branch only

    // --- dlyap_fast_c (Smith doubling) parameters ---
    Scalar dlyap_tolerance     = Tol::dlyap;
    int    dlyap_max_doublings = 60;     // covers rho up to 1 - 2^52 (double precision). TODO LATER: single/double-dependent max_doublings

    // --- dare_sda parameters ---
    int    sda_min_doublings   = 1;
    int    sda_max_doublings   = 40;
};

struct DARESolverInfo {
    int    solver_iterations = 0;
    Scalar tol_achieved      = std::numeric_limits<Scalar>::quiet_NaN();
    bool   solve_success     = false;
    bool   unstable_k0       = false;  // NK: initial gain destabilising
    bool   used_sda_fallback = false;
};

struct SDDREOpts {
    Scalar preview_horizon             = 2.0;  // s (inf -> constant-ref approx)
    bool   use_full_fh_mpc_at_terminal = false;
    bool   always_use_full_fh_mpc      = false;
    DARESolverOpts dare;
};

struct SDDRESolveInfo {
    Scalar         time_sdc_discretize_us = 0;
    Scalar         time_dare_us           = 0;
    Scalar         time_feedforward_us    = 0;
    DARESolverInfo dare_info;
};

// ============================================================
//  QuadParams — hexacopter plant parameters
//
//  Mirrors the fields the MATLAB SDC functions read off qp.
// ============================================================
struct QuadParams {
    Scalar Ts    = 0.01;      // sample period            [s]
    Scalar m     = 5.0;       // mass                     [kg]
    Scalar I_xx  = 0.008;     // inertia                  [kg m^2]
    Scalar I_yy  = 0.009;
    Scalar I_zz  = 0.015;
    Scalar kF    = 0.000015;    // thrust coefficient
    Scalar kM    = 0.00000015;  // drag / reaction-torque coefficient

    VecNU  nominal_omegas = VecNU::Zero();  // hover rotor speeds
    VecNU  max_du         = VecNU::Zero();  // upper clamp on delta_u
    VecNU  x              = VecNU::Zero();  // rotor x arm positions   [m]
    VecNU  y              = VecNU::Zero();  // rotor y arm positions   [m]
    VecNU  dirs           = VecNU::Zero();  // spin directions, +/-1
    VecNU  enabled        = VecNU::Ones();  // 1 = healthy, 0 = failed

    // Convex blend in the quaternion-kinematics SDC factorisation.
    //   alpha = 1 -> Omega_w / (q0/2)I split
    //   alpha = 0 -> plain Xi_q split
    Scalar quat_blend_alpha = 1.0;

    Scalar g = 9.81;
};

// ---- dlyap info ---------------------------------------------
struct DlyapInfo {
    bool   converged     = false;
    bool   diverged      = false;
    bool   is_stable     = false;
    int    doublings     = 0;
    Scalar rel_increment = std::numeric_limits<Scalar>::quiet_NaN();
};
