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
#include <type_traits>
#include <cassert>

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
    static constexpr int dlyap_max_doublings = 60; // covers rho up to 1 - 2^52 (double precision)
    static constexpr double dare  = 1e-4;
};

template <> struct sddre_tol<float> {
    static constexpr float dlyap = 5e-6f; // ~42 * eps(float)
    static constexpr int dlyap_max_doublings = 30;  // TODO: verify coverage up to 1 - 2^23 (single precision)
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

// ============================================================
//  C6 — input-weight structure exploitation
//
//  The controller's R is (here and typically) a scalar multiple of identity, r*I. Change 6 (C6) allows that structure to be exploited.
//  The code should support both the dense and sparse extremes – full PSD R, to scalar multiple identity r*I. In between cases (e.g. diagonal or block structures) are neglected to keep the code simple + readily toggleable between; we only need to test the two extreme cases, the others lie within. Plus: r*I offers the most speedup, and is probably the most common choice (especially in aerospace where plants likely have symmetric actuator arrangements).
// 
// The weight matrix R is wrapped in a type parameterised on how much structure it is allowed to assume:
//
//    RMode::Scalar — stores r alone. K'RK becomes r*(K'K),
//                    R + B'PB becomes a diagonal add, and
//                    B R^-1 B' becomes a scaled outer product
//                    with no factorisation at all.
//    RMode::Dense  — stores the full 6x6 and does exactly the
//                    R arithmetic the pre-C6 code did.
//
//  One textual body per operation, selected by if constexpr.
//
//  NOT necessarily bit-identical across modes, especially in single-precision: r*(K'K) scales before the summation, while K'*(R*K) scales after, so the two can differ at the last bit or so – the precision difference is not a concern, but could manifest behaviourally which is a concern: near the DARE early-break threshold that could move a Newton-Kleinman iteration count by one: expected, and the reason total iteration counts are now reported/monitored in test_main.cpp.
// ============================================================
enum class RMode { Scalar, Dense };

#ifndef SDDRE_R_MODE_DEFAULT
  #define SDDRE_R_MODE_DEFAULT RMode::Scalar
#endif

inline const char* r_mode_name(RMode m) {
    return (m == RMode::Scalar) ? "scalar" : "dense";
}

template <RMode Mode>
struct RWeight {
    static constexpr RMode mode      = Mode;
    static constexpr bool  is_scalar = (Mode == RMode::Scalar);

    using Storage = std::conditional_t<is_scalar, Scalar, MatNU>;
    Storage R_;

    RWeight() { set(Scalar(1)); }

    // ---- Setters ----------------------------------------
    //  set(r) is the honest one — it says what the weight is.
    void set(Scalar r) {
        if constexpr (is_scalar) {
            R_ = r;
        } else {
            R_.setZero();
            R_.diagonal().setConstant(r);
        }
    }

    // set(Rm) accepts a full matrix so existing call sites and MATLAB-side exports keep working.
    // In scalar mode it keeps only r and asserts the input really was r*I.
    // Asserts compile out under NDEBUG. Set in the PlatformIO build; make sure the desktop g++ line keeps -DNDEBUG too, or this check runs on every call.
    void set(const MatNU& Rm) {
        if constexpr (is_scalar) {
            const Scalar r = Rm(0, 0);
            assert(r > Scalar(0) &&
                   "RMode::Scalar requires R = r*I with r > 0");
            assert((Rm - r * MatNU::Identity()).cwiseAbs().maxCoeff()
                       <= Scalar(8) * SCALAR_EPS * std::abs(r) &&
                   "RMode::Scalar requires R = r*I");
            R_ = r;
        } else {
            R_ = Rm;
        }
    }

    // ---- Accessors ---------------------------------------
    Scalar scalar() const {
        if constexpr (is_scalar) return R_;
        else                     return R_(0, 0);
    }

    MatNU dense() const {
        if constexpr (is_scalar) return R_ * MatNU::Identity();
        else                     return R_;
    }

    // ---- Hot-path operations ------------------------------

    // S <- R + M
    // Scalar mode touches 6 diagonal entries instead of 36.
    template <typename Derived>
    void set_R_plus(MatNU& S, const Eigen::MatrixBase<Derived>& M) const {
        if constexpr (is_scalar) {
            S = M;
            S.diagonal().array() += R_;
        } else {
            S = R_ + M;
        }
    }

    //  X <- X + K'RK.
    //  Scalar: one 12x6 * 6x12 gemm with alpha = r (864 madds).
    //  Dense:  a 12x6 temporary then the same gemm (1296 madds).
    //
    //  Since K'RK is symmetric, one possible speedup stands out: selfadjointView<Lower>().rankUpdate(K.transpose(), r) would compute only the lower triangle: 468 madds, about 45% off.
    //  Tried and rejected: measurably SLOWER on Cortex-M7. The saved arithmetic does not survive the consequences: the full matrix is needed, so the lower triangle has to be mirrored into the upper before use, and on this unit the load/store cost trumps the saving in arithmetic. Same result as the triangular-view shortcut in the DARE residual, for the same reason. Both are recorded as measured negatives.
    void add_KtRK(MatNX& X, const MatNUNX& K) const {
        if constexpr (is_scalar) X.noalias() += R_ * (K.transpose() * K);
        else                     X.noalias() += K.transpose() * R_ * K;
    }

    //  G = B R^{-1} B'  (the SDA ctrb-gramian block).
    //
    //  Scalar mode deletes the 6x6 LDLT factorisation, and its 12-right-hand-side solve.
    //  But, cold path only: this runs at k == 1 and on the Newton-Kleinman fallback, so it never shows up as an improvement in median compute.
    MatNX B_Rinv_Bt(const MatNXNU& B) const {
        MatNX G;
        if constexpr (is_scalar) {
            G.noalias() = (Scalar(1) / R_) * (B * B.transpose());
        } else {
            const MatNUNX Rinv_Bt = R_.ldlt().solve(B.transpose());
            G.noalias() = B * Rinv_Bt;
        }
        return G;
    }
};

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
    int    dlyap_max_doublings = Tol::dlyap_max_doublings;

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
    bool   tol_is_estimate   = false;  // true when tol_achieved came from the
                                       // NK Newton-increment identity rather
                                       // than the full DARE residual
};

struct SDDREOpts {
    Scalar preview_horizon             = 2.0;  // s (inf -> constant-ref approx)
    bool   use_full_fh_mpc_at_terminal = false;
    bool   always_use_full_fh_mpc      = false;

    // C3: after the DARE solve, evaluate the TRUE relative residual once in
    // compute_u (reusing the solver's gain) and overwrite tol_achieved /
    // solve_success with it. Restores the health signal that the NK
    // Newton-increment test no longer provides in-solver.
    bool   post_residual_check         = false;

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
    bool   converged     = false; // Do not modify - critical for solver behaviour
    bool   is_stable     = false; // Do not modify - critical for solver behaviour
    int    doublings     = 0;
    Scalar rel_increment = -1; // Sentinel - never possible from genuine solver behaviour
};