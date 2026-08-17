#pragma once
// ============================================================
//  controllers/sdopt_controller.h — SDOPT trajectory-tracking
//                                    controller
//
//  Templated on RMode: Scalar (R = rI, default) exploits the
//  diagonal structure; Dense keeps full R for comparison.
//  Both instantiations can coexist in the same binary.
//
//  Ported from compute_u_SDDRE_v3.m — step indices remain
//  1-based to keep the two implementations diffable.
// ============================================================

#include "types/defs.h"
#include "types/r_weight.h"
#include "linalg/solvers/solver_types.h"
#include "linalg/solvers/dare_sda.h"
#include "linalg/solvers/dare_nk.h"
#include "linalg/solvers/c2d_zoh.h"
#include "linalg/compute_dare_gain.h"
#include "plant/sdc_model.h"
#include "platform/timing.h"
#include "platform/mm_kernels.h"

#include <vector>
#include <algorithm>
#include <cmath>

// ============================================================
//  Controller-level option / info structs
// ============================================================

struct SDOPTOpts {
    Scalar preview_horizon             = 2.0;   // [s]
    bool   use_full_fh_mpc_at_terminal = false;
    bool   always_use_full_fh_mpc      = false;
    bool   post_residual_check         = false; // true -> evaluates the actual residual after the DARE solve (which uses the Newton-increment proxy throughout)
    DARESolverOpts dare;
};

struct SDOPTSolveInfo {
    Scalar         time_sdc_discretize_us = 0;
    Scalar         time_dare_us           = 0;
    Scalar         time_feedforward_us    = 0;
    DARESolverInfo dare_info;
};

// ============================================================
//  SDOPTControllerT
// ============================================================
template <RMode RM = SDDRE_R_MODE_DEFAULT>
class SDOPTControllerT {
public:
    static constexpr RMode r_mode = RM;
    using RType = RWeight<RM>;

    // ---- Weight matrices (set once before first call) -------
    MatNYNX C   = MatNYNX::Zero();   // 6x12 output selection
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

    void reset() {
        P_ss_.setZero();
        K_ss_.setZero();
        wex_valid_ = false;
        rebuild_weight_cache();
    }

    // Call this if C, Qy or Qyf change without a full reset().
    void rebuild_weight_cache() {
        CtQy_  = C.transpose() * Qy;
        CtQyf_ = C.transpose() * Qyf;
        Q_     = CtQy_ * C;
        symmetrise(Q_);
        wex_valid_ = false;      // Wex depends on CtQy_
    }

    // =========================================================
    //  compute_u   (k is 1-based, matching MATLAB)
    // =========================================================
    VecNU compute_u(Scalar /*tk*/, const VecNX& xk, int k,
                    const VecNU& uk_prev,
                    SDOPTSolveInfo& info)
    {
        // ------------------------------------------------
        //  1. SDC matrices + ZOH discretisation
        // ------------------------------------------------
        sddre_tick_t t0 = sddre_ticks();

        const MatNX Ac = get_A_sdc_quaternion(xk, qp);

        const VecNU uk_clamped = uk_prev.cwiseMax(-qp.nominal_omegas)
                                        .cwiseMin(qp.max_du);
        const MatNXNU Bc = get_B_sdc(uk_clamped, qp);

        MatNX   A;
        MatNXNU B;
        c2d_zoh_expm(Ac, Bc, qp.Ts, A, B);

        info.time_sdc_discretize_us = _sddre_elapsed_us(t0);

        // ------------------------------------------------
        //  2. Solve DARE
        // ------------------------------------------------
        t0 = sddre_ticks();

        Eigen::LDLT<MatNU> S_ldlt;   // S = R + B'P_ss B (reused in step 3)

        DARESolverOpts dare_opts = opts.dare;
        if (k == 1) dare_opts.method = DARESolverMethod::SDA;  // cold init

        DARESolverInfo dare_info;

        if (dare_opts.method == DARESolverMethod::SDA) {
            P_ss_ = dare_sda(A, B, Q_, R, dare_opts.tolerance,
                             dare_opts.dare_sda_min_doublings,
                             dare_opts.dare_sda_max_doublings, dare_info);
            K_ss_ = compute_dare_gain(B, R, P_ss_, A, S_ldlt);

        } else {
            // Note: P, K and S factorisation all come back from the solver
            P_ss_ = dare_nk(A, B, Q_, R, P_ss_, dare_opts,
                                   K_ss_, S_ldlt, dare_info);

            // NK fallback: cold SDA when the warm start K0 is destabilising
            if (dare_opts.method == DARESolverMethod::NK && !dare_info.solve_success && dare_info.unstable_k0) {
                DARESolverInfo sda_info;
                P_ss_ = dare_sda(A, B, Q_, R, dare_opts.tolerance,
                                 dare_opts.dare_sda_min_doublings,
                                 dare_opts.dare_sda_max_doublings, sda_info);
                K_ss_ = compute_dare_gain(B, R, P_ss_, A, S_ldlt);

                dare_info.tol_achieved      = sda_info.tol_achieved;
                dare_info.solver_iterations = sda_info.solver_iterations;
                dare_info.solve_success     = sda_info.solve_success;
                dare_info.used_sda_fallback = true;
                dare_info.tol_is_estimate   = false;
            }
        }

        // Optional true-residual health check.
        // Overwrites the increment-based proxy with a direct DARE residual evaluation; this is what SIL must compare.
        if (opts.post_residual_check) {
            const Scalar res = compute_dare_residual(A, B, Q_, R, P_ss_, K_ss_);
            dare_info.tol_achieved    = res;
            dare_info.solve_success   = (res < dare_opts.tolerance);
            dare_info.tol_is_estimate = false;
        }

        info.dare_info    = dare_info;
        info.time_dare_us = _sddre_elapsed_us(t0);

        // ------------------------------------------------
        //  3. Feedforward
        // ------------------------------------------------
        t0 = sddre_ticks();

        const MatNX   A_cl = A - B * K_ss_;

        VecNU u;

        if (std::isinf(opts.preview_horizon)) {
            // ---- Constant-reference approximation ----------
            const Eigen::Map<const VecNY> r_k(r_data + (k - 1) * NY);
            const MatNX I_minus_Ft = MatNX::Identity() - A_cl.transpose();
            const VecNX s = I_minus_Ft.partialPivLu().solve(CtQy_ * r_k);
            u = -K_ss_ * xk + S_ldlt.solve(B.transpose() * s);

        } else {
            const int M = static_cast<int>(
                std::round(opts.preview_horizon / qp.Ts));
            const bool approaching_terminal = (k + M >= r_len);

            if (!(opts.always_use_full_fh_mpc ||
                  (opts.use_full_fh_mpc_at_terminal && approaching_terminal)))
            {
                // ---- Preview costate sweep (d = 1 branch) ---
                rebuild_wex_if_needed(CtQy_, M);

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

                u = -K_ss_ * xk + S_ldlt.solve(B.transpose() * v);

            } else {
                // ---- Terminal: full finite-horizon LQT recursion ----
                MatNX P_term = CtQyf_ * C;
                symmetrise(P_term);

                const int M_cl = std::min(M, r_len - k);

                const Eigen::Map<const VecNY> r_end(
                    r_data + (k + M_cl - 1) * NY);
                VecNX v = CtQyf_ * r_end;

                for (int j = M_cl - 1; j >= 1; --j) {
                    const MatNUNX K_j    = compute_dare_gain(B, R, P_term, A);
                    const MatNX   A_cl_j = A - B * K_j;

                    // P_j    = Q + K_j'R K_j + A_cl_j' P_{j+1} A_cl_j
                    //   Old code: P_term = (Q_ + K_j.transpose() * R * K_j + A_cl_j.transpose() * P_term * A_cl_j).eval();
                    //   Now: uses adaptive path, according to R matrice's RType.
                    MatNX P_next = Q_;                                        // P_next == Q
                    R.add_KtRK(P_next, K_j);                                  // P_next += K_j'R K_j (P_next receives the sum)
                    P_next.noalias() += A_cl_j.transpose() * P_term * A_cl_j; // P_next += A_cl_j' P_{j+1} A_cl_j
                    P_term = P_next;                                          // P_term == P_j (ready as P_{j+1} for the next pass down)
                    symmetrise(P_term);

                    const Eigen::Map<const VecNY> r_j(
                        r_data + (k + j - 1) * NY);
                    v = A_cl_j.transpose() * v + CtQy_ * r_j;
                }

                Eigen::LDLT<MatNU> S_term_ldlt;
                const MatNUNX Kk = compute_dare_gain(B, R, P_term, A, S_term_ldlt);
                u = -Kk * xk + S_term_ldlt.solve(B.transpose() * v);
            }
        }

        info.time_feedforward_us = _sddre_elapsed_us(t0);
        return u;
    }

private:
    // ---- Persistent state (MATLAB `persistent P_ss K_ss`) ---
    MatNX   P_ss_ = MatNX::Zero();      // Riccati matrix
    MatNUNX K_ss_ = MatNUNX::Zero();    // Riccati gain

    // ---- Cached constant weight products --------------------
    //  Depend only on C, Qy, Qyf. Rebuilt by reset().
    MatNX   Q_     = MatNX::Zero();     // C' * Qy * C
    MatNXNY CtQy_  = MatNXNY::Zero();   // C' * Qy
    MatNXNY CtQyf_ = MatNXNY::Zero();   // C' * Qyf

    // ---- Wex cache ------------------------------------------
    //  Wex = C'*Qy * rpad,  rpad = [r_, repmat(r_(:,end), 1, M+1)]
    //  Column-major, MATLAB 1-based columns 1 .. N+M+1.
    bool                wex_valid_ = false;
    int                 wex_N_     = 0;
    int                 wex_M_     = 0;
    std::vector<Scalar> wex_buf_;

    void rebuild_wex_if_needed(const MatNXNY& CtQy, int M) {
        const int N = r_len;
        if (wex_valid_ && wex_N_ == N && wex_M_ == M) return;

        const int n_cols = N + M + 1;          // d = 1
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
    // Open for debate Eigen::Map<const VecNX> instead of VecNX. No measurable performance increase, yet usage consequences.
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
