#pragma once
// ============================================================
//  sdopt_controller_generic.h — SDOPT trajectory-tracking controller
//
//  Mutually exclusive with sdopt_controller_teensy.h — both
//  define SDOPTController. This is the clean + generic version,
//  most faithful to the MATLAB reference.
// 
//  Ported from compute_u_SDDRE_v3.m. Step indices remain 1-based
//  to keep the implementation diffable against MATLAB.
// ============================================================

#ifdef SDOPT_CONTROLLER_TEENSY_H_
  #error "sdopt_controller_generic.h and sdopt_controller_teensy.h cannot both be included in the same translation unit"
#endif
#define SDOPT_CONTROLLER_GENERIC_H_

#include "types/defs.h"
#include "linalg/solvers/solver_types.h"
#include "linalg/solvers/dare_sda.h"
#include "linalg/solvers/dare_nk_generic.h"
#include "linalg/solvers/c2d_zoh.h"
#include "linalg/compute_dare_residual.h"
#include "plant/sdc_model.h"
#include "platform/timing.h"
#include "controllers/sdopt_opts.h"

#include <vector>
#include <algorithm>
#include <cmath>

// ============================================================
//  SDOPTController
// ============================================================
class SDOPTController {
public:
    // ---- Weight matrices (externally set) --------------------
    MatNYNX C   = MatNYNX::Zero();
    MatNY   Qy  = MatNY::Identity();
    MatNU   R   = MatNU::Identity();
    MatNY   Qyf = MatNY::Identity();

    // ---- Reference trajectory --------------------------------
    //  Column-major: r_data[col * NY + row], col in [0, r_len).
    const Scalar* r_data = nullptr;
    int           r_len  = 0;

    // ---- Options / parameters --------------------------------
    SDOPTOpts  opts;
    QuadParams qp;

    void set_R(Scalar r) {
        R.setZero();
        R.diagonal().setConstant(r);
    }

    void reset() {
        P_ss.setZero();
        K_ss.setZero();
        wex_valid_ = false;
        rebuild_weight_cache();
    }

    // =========================================================
    //  compute_u
    // =========================================================
    VecNU compute_u(const VecNX& xk, int k, const VecNU& uk_prev,
                    SDOPTSolveInfo& info)
    {
        // ------------------------------------------------
        //  1. SDC matrices + ZOH discretisation
        // ------------------------------------------------
        sdopt_tick_t t0 = sdopt_ticks();

        const MatNX Ac = get_A_sdc_quaternion(xk, qp);
        const VecNU uk_clamped = uk_prev.cwiseMax(-qp.nominal_omegas)
                                        .cwiseMin(qp.max_du);
        const MatNXNU Bc = get_B_sdc(uk_clamped, qp);
        MatNX   A;
        MatNXNU B;
        c2d_zoh_expm(Ac, Bc, qp.Ts, A, B);

        info.time_sdc_discretise_us = sdopt_elapsed_us(t0);

        // ------------------------------------------------
        //  2. Solve DARE
        // ------------------------------------------------
        t0 = sdopt_ticks();

        DARESolverOpts dare_opts = opts.dare;
        if (k == 1) dare_opts.method = DARESolverMethod::SDA;  // cold init

        DARESolverInfo dare_info;

        if (dare_opts.method == DARESolverMethod::SDA) {
            P_ss = dare_sda(A, B, Q, R, dare_opts.tolerance,
                             dare_opts.dare_sda_min_doublings,
                             dare_opts.dare_sda_max_doublings, dare_info);
            K_ss = (R + B.transpose() * P_ss * B).llt().solve(B.transpose() * P_ss * A);

        } else {
            P_ss = dare_nk(A, B, Q, R, P_ss, dare_opts, dare_info);
            K_ss = (R + B.transpose() * P_ss * B).llt().solve(B.transpose() * P_ss * A);

            // NK fallback: cold SDA when the warm start K0 is destabilising
            if (dare_opts.method == DARESolverMethod::NK &&
                !dare_info.solve_success && dare_info.unstable_k0)
            {
                DARESolverInfo sda_info;
                P_ss = dare_sda(A, B, Q, R, dare_opts.tolerance,
                                 dare_opts.dare_sda_min_doublings,
                                 dare_opts.dare_sda_max_doublings, sda_info);
                K_ss = (R + B.transpose() * P_ss * B).llt().solve(B.transpose() * P_ss * A);

                dare_info.tol_achieved      = sda_info.tol_achieved;
                dare_info.solver_iterations = sda_info.solver_iterations;
                dare_info.solve_success     = sda_info.solve_success;
                dare_info.used_sda_fallback = true;
            }
        }

        info.dare_info    = dare_info;
        info.time_dare_us = sdopt_elapsed_us(t0);

        // ------------------------------------------------
        //  3. Feedforward
        // ------------------------------------------------
        t0 = sdopt_ticks();

        const MatNX A_cl = A - B * K_ss;
        const MatNU S = R + B.transpose() * P_ss * B;

        VecNU u;

        if (std::isinf(opts.preview_horizon)) {
            // ---- Constant-reference approximation ----------
            const Eigen::Map<const VecNY> r_k(r_data + (k - 1) * NY);
            const MatNX I_minus_Ft = MatNX::Identity() - A_cl.transpose();
            const VecNX s = I_minus_Ft.partialPivLu().solve(CtQy * r_k);
            u = -K_ss * xk + S.llt().solve(B.transpose() * s);

        } else {
            const int M = static_cast<int>(std::round(opts.preview_horizon / qp.Ts));
            const bool approaching_terminal = (k + M >= r_len);

            if (!(opts.always_use_full_fh_mpc ||
                  (opts.use_full_fh_mpc_at_terminal && approaching_terminal)))
            {
                // ---- Preview costate sweep ---
                rebuild_wex_if_needed(CtQy, M);

                const MatNX F = A_cl.transpose();

                VecNX v = (MatNX::Identity() - F).partialPivLu().solve(wex_col(k + M));

                for (int j = M - 1; j >= 1; --j)
                    v = F * v + wex_col(k + j);

                u = -K_ss * xk + S.llt().solve(B.transpose() * v);

            } else {
                // ---- Full finite-horizon LQT recursion ----
                MatNX P_term = Qf;

                const int M_cl = std::min(M, r_len - k);

                const Eigen::Map<const VecNY> r_end(r_data + (k + M_cl - 1) * NY);
                VecNX v = CtQyf * r_end;

                for (int j = M_cl - 1; j >= 1; --j) {
                    const MatNUNX K_j    = (R + B.transpose() * P_term * B).llt().solve(B.transpose() * P_term * A);
                    const MatNX   A_cl_j = A - B * K_j;

                    P_term = (Q + K_j.transpose() * R * K_j
                            + A_cl_j.transpose() * P_term * A_cl_j).eval();
                    symmetrise(P_term);

                    const Eigen::Map<const VecNY> r_j(r_data + (k + j - 1) * NY);
                    v = A_cl_j.transpose() * v + CtQy * r_j;
                }

                const MatNUNX Kk     = (R + B.transpose() * P_term * B).llt().solve(B.transpose() * P_term * A);
                const MatNU   S_term = R + B.transpose() * P_term * B;
                u = -Kk * xk + S_term.llt().solve(B.transpose() * v);
            }
        }

        info.time_feedforward_us = sdopt_elapsed_us(t0);
        return u;
    }

private:
    // ---- Riccati solution and gain --------------------------
    MatNX   P_ss = MatNX::Zero();
    MatNUNX K_ss = MatNUNX::Zero();

    // ---- Constant products ----------------------------------
    MatNXNY CtQy   = MatNXNY::Zero();  // C' * Qy
    MatNXNY CtQyf  = MatNXNY::Zero();  // C' * Qyf
    MatNX   Q  = MatNX::Zero();        // C' * Qy * C
    MatNX   Qf = MatNX::Zero();        // C' * Qyf * C

    void rebuild_weight_cache() {
        CtQy   = C.transpose() * Qy;
        CtQyf  = C.transpose() * Qyf;
        Q      = C.transpose() * Qy * C;
        Qf     = C.transpose() * Qyf * C;
        symmetrise(Q);
        symmetrise(Qf);
        wex_valid_ = false;  // Wex depends on CtQy
    }

    // ---- Wex cache ------------------------------------------
    //  Wex = C'* Qy * r
    //  NB: duplicated in sdopt_controller_teensy.h — keep in sync.
    bool                wex_valid_ = false;
    int                 wex_N_     = 0;
    int                 wex_M_     = 0;
    std::vector<Scalar> wex_buf_;

    void rebuild_wex_if_needed(const MatNXNY& CtQy, int M) {
        const int N = r_len;
        if (wex_valid_ && wex_N_ == N && wex_M_ == M) return;

        const int n_cols = N + M + 1;
        wex_buf_.resize(static_cast<size_t>(NX) * n_cols);

        for (int c = 0; c < n_cols; ++c) {
            const int r_col = std::min(c, N - 1);   // hold last ref past the end
            const Eigen::Map<const VecNY> r_j(r_data + r_col * NY);
            Eigen::Map<VecNX> w_j(wex_buf_.data() + static_cast<size_t>(c) * NX);
            w_j.noalias() = CtQy * r_j;
        }

        wex_N_     = N;
        wex_M_     = M;
        wex_valid_ = true;
    }

    // 1-based MATLAB column -> 0-based buffer index
    VecNX wex_col(int matlab_col) const {
        return Eigen::Map<const VecNX>(
            wex_buf_.data() + static_cast<size_t>(matlab_col - 1) * NX);
    }
};
