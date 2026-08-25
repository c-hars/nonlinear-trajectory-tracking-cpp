#pragma once
// ============================================================
//  sdopt_controller_teensy.h — SDOPT trajectory-tracking controller
//
//  Mutually exclusive with sdopt_controller_generic.h — both
//  define SDOPTController.
// 
//  This is the Teensy-optimised version: symmetry exploitation,
//  custom matrix multiply kernels, structure exploitation
//  on R, LLT decompositions passed around throughout.
// 
//  Ported from compute_u_SDDRE_v3.m. Step indices remain 1-based
//  to keep the implementation diffable against MATLAB.
// ============================================================

#ifdef SDOPT_CONTROLLER_GENERIC_H_
  #error "sdopt_controller_teensy.h and sdopt_controller_generic.h cannot both be included in the same translation unit"
#endif
#define SDOPT_CONTROLLER_TEENSY_H_

#include "types/defs.h"
#include "types/r_weight.h"
#include "linalg/solvers/solver_types.h"
#include "linalg/solvers/dare_sda.h"
#include "linalg/solvers/dare_nk_teensy.h"
#include "linalg/solvers/c2d_zoh.h"
#include "linalg/compute_dare_gain.h"
#include "linalg/compute_dare_residual.h"
#include "plant/sdc_model.h"
#include "platform/timing.h"
#include "platform/mm_kernels.h"
#include "controllers/sdopt_opts.h"

#include <vector>
#include <algorithm>
#include <cmath>

// ============================================================
//  SDOPTControllerT
// ============================================================
template <RMode RM = SDOPT_R_MODE_DEFAULT>
class SDOPTControllerT {
public:
    static constexpr RMode r_mode = RM;
    using RType = RWeight<RM>;

    // ---- Weight matrices (externally set) --------------------
    MatNYNX C   = MatNYNX::Zero();
    MatNY   Qy  = MatNY::Identity();
    RType   R;
    MatNY   Qyf = MatNY::Identity();

    // ---- Reference trajectory --------------------------------
    //  Column-major: r_data[col * NY + row], col in [0, r_len).
    const Scalar* r_data = nullptr;
    int           r_len  = 0;

    // ---- Options / parameters --------------------------------
    SDOPTOpts  opts;
    QuadParams qp;

    void set_R(Scalar r) {
        R.set(r);
    }

    void reset() {
        P_ss.setZero();
        K_ss.setZero();
        wex_valid_ = false;
        rebuild_weight_cache();
    }

    // =========================================================
    //  compute_u   (k is 1-based, matching MATLAB)
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

        Eigen::LLT<MatNU> S_llt;   // S = R + B'P_ss B (reused in step 3)

        DARESolverOpts dare_opts = opts.dare;
        if (k == 1) dare_opts.method = DARESolverMethod::SDA;  // cold init

        DARESolverInfo dare_info;

        if (dare_opts.method == DARESolverMethod::SDA) {
            P_ss = dare_sda(A, B, Q, R, dare_opts.tolerance,
                             dare_opts.dare_sda_min_doublings,
                             dare_opts.dare_sda_max_doublings, dare_info);
            K_ss = compute_dare_gain(B, R, P_ss, A, S_llt);

        } else {
            // Note: P, K and S factorisation all come back from the solver
            P_ss = dare_nk(A, B, Q, R, P_ss, dare_opts, 
                           K_ss, S_llt, dare_info);

            // NK fallback: cold SDA when the warm start K0 is destabilising
            if (dare_opts.method == DARESolverMethod::NK && !dare_info.solve_success && dare_info.unstable_k0) {
                DARESolverInfo sda_info;
                P_ss = dare_sda(A, B, Q, R, dare_opts.tolerance,
                                 dare_opts.dare_sda_min_doublings,
                                 dare_opts.dare_sda_max_doublings, sda_info);
                K_ss = compute_dare_gain(B, R, P_ss, A, S_llt);

                dare_info.tol_achieved      = sda_info.tol_achieved;
                dare_info.solver_iterations = sda_info.solver_iterations;
                dare_info.solve_success     = sda_info.solve_success;
                dare_info.used_sda_fallback = true;
                dare_info.tol_is_estimate   = false;
            }
        }

        // Optional true-residual health check.
        // Overwrites any increment-based proxy with a direct DARE residual evaluation.
        if (opts.post_residual_check) {
            const Scalar res = compute_dare_residual(A, B, Q, R, P_ss, K_ss);
            dare_info.tol_achieved    = res;
            dare_info.solve_success   = (res < dare_opts.tolerance);
            dare_info.tol_is_estimate = false;
        }

        info.dare_info    = dare_info;
        info.time_dare_us = sdopt_elapsed_us(t0);

        // ------------------------------------------------
        //  3. Feedforward
        // ------------------------------------------------
        t0 = sdopt_ticks();

        const MatNX  A_cl = A - B * K_ss;

        VecNU u;

        if (std::isinf(opts.preview_horizon)) {
            // ---- Constant-reference approximation ----------
            const Eigen::Map<const VecNY> r_k(r_data + (k - 1) * NY);
            const MatNX I_minus_Ft = MatNX::Identity() - A_cl.transpose();
            const VecNX s = I_minus_Ft.partialPivLu().solve(CtQy * r_k);
            u = -K_ss * xk + S_llt.solve(B.transpose() * s);

        } else {
            const int M = static_cast<int>(std::round(opts.preview_horizon / qp.Ts));
            const bool approaching_terminal = (k + M >= r_len);

            if (!(opts.always_use_full_fh_mpc ||
                  (opts.use_full_fh_mpc_at_terminal && approaching_terminal)))
            {
                // ---- Preview costate sweep ---
                rebuild_wex_if_needed(CtQy, M);

                const MatNX F = A_cl.transpose();

                // Raw algorithm:
                //   v = (I - F)^-1 * wex(k+M)
                //   for j = M-1 down to 1:  v = F*v + wex(k+j)
                //   u = -K_ss * xk + S^-1 B' v

                VecNX v_a = (MatNX::Identity() - F).partialPivLu().solve(wex_col(k + M));
                VecNX v_b;
                Scalar* v_cur = v_a.data();
                Scalar* v_nxt = v_b.data();

                for (int j = M - 1; j >= 1; --j) {
                    mv12_fma(v_nxt, F.data(), v_cur, wex_ptr(k + j));
                    std::swap(v_cur, v_nxt);
                }
                const Eigen::Map<const VecNX> v(v_cur);

                u = -K_ss * xk + S_llt.solve(B.transpose() * v);

            } else {
                // ---- Full finite-horizon LQT recursion ----
                MatNX P_term = Qf;

                const int M_cl = std::min(M, r_len - k);

                const Eigen::Map<const VecNY> r_end(r_data + (k + M_cl - 1) * NY);
                VecNX v = CtQyf * r_end;

                for (int j = M_cl - 1; j >= 1; --j) {
                    const MatNUNX K_j    = compute_dare_gain(B, R, P_term, A);
                    const MatNX   A_cl_j = A - B * K_j;

                    // P_j    = Q + K_j'R K_j + A_cl_j' P_{j+1} A_cl_j
                    // Computing and adding K'RK is accelerated via RType
                    MatNX P_next = Q;
                    R.add_KtRK(P_next, K_j);                                  // P_next = P_j = Q + K_j'R K_j (P_next receives the sum)
                    P_next.noalias() += A_cl_j.transpose() * P_term * A_cl_j; // P_next = P_j = Q + K_j'R K_j + A_cl_j' P_{j+1} A_cl_j
                    P_term = P_next;                                          // P_term = P_j (ready as P_{j+1} for the next pass down)
                    symmetrise(P_term);

                    const Eigen::Map<const VecNY> r_j(r_data + (k + j - 1) * NY);
                    v = A_cl_j.transpose() * v + CtQy * r_j;
                }

                Eigen::LLT<MatNU> S_term_llt;
                const MatNUNX Kk = compute_dare_gain(B, R, P_term, A, S_term_llt);
                u = -Kk * xk + S_term_llt.solve(B.transpose() * v);
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
    MatNXNY CtQy  = MatNXNY::Zero();   // C' * Qy
    MatNXNY CtQyf = MatNXNY::Zero();   // C' * Qyf
    MatNX   Q     = MatNX::Zero();     // C' * Qy * C
    MatNX   Qf    = MatNX::Zero();     // C' * Qyf * C

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
    //  NB: duplicated in sdopt_controller_generic.h — keep in sync.
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

    // 1-based MATLAB column -> 0-based buffer index.
    VecNX wex_col(int matlab_col) const {
        return Eigen::Map<const VecNX>(
            wex_buf_.data() + static_cast<size_t>(matlab_col - 1) * NX);
    }

    const Scalar* wex_ptr(int matlab_col) const {
        return wex_buf_.data() + static_cast<size_t>(matlab_col - 1) * NX;
    }
};

// Default instantiation — scalar R unless overridden at build time.
using SDOPTController = SDOPTControllerT<>;