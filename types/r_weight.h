#pragma once
// ============================================================
//  types/r_weight.h — Structure exploitation of R
//
//  R is often r*I; this wrapper is parameterised on how much
//  structure R may assume:
//
//    RMode::Dense  — stores the full 6×6 and uses standard arithmetic.
//    RMode::Scalar — stores r alone and uses structure-exploiting
//                    arithmetic.
//
//  Only the two structural extremes (Scalar and Dense) are supported —
//  intermediate structures (e.g. diagonal, block-diagonal) possible, 
//  but performance differences are all bracketed by the two extremes.
// 
// ============================================================

#include "types/defs.h"

#include <cmath>
#include <type_traits>

enum class RMode { Scalar, Dense };

#ifndef SDOPT_R_MODE_DEFAULT
  #define SDOPT_R_MODE_DEFAULT RMode::Scalar
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
    void set(Scalar r) {
        if constexpr (is_scalar) {
            R_ = r;
        } else {
            R_.setZero();
            R_.diagonal().setConstant(r);
        }
    }

    void set(const MatNU& Rm) {
        static_assert(!is_scalar, "set(MatNU) requires RMode::Dense");
        R_ = Rm;
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
    template <typename Derived>
    void set_R_plus(MatNU& S, const Eigen::MatrixBase<Derived>& M) const {
        if constexpr (is_scalar) {
            S = M;
            S.diagonal().array() += R_;
        } else {
            S = R_ + M;
        }
    }

    //  X <- X + K'RK
    //  Note: symmetric product; rankUpdate (lower-tri only) was tried earlier — slower on Cortex-M7 at the time (mirror-copy cost from the temporary dominated), but worth revisiting now: mm12_abt_sym handles the symmetric product efficiently.
    void add_KtRK(MatNX& X, const MatNUNX& K) const {
        if constexpr (is_scalar) X.noalias() += R_ * (K.transpose() * K);
        else                     X.noalias() += K.transpose() * R_ * K;
    }

    //  G = B R^{-1} B'
    //  Cold path only (dare_sda)
    MatNX B_Rinv_Bt(const MatNXNU& B) const {
        MatNX G;
        if constexpr (is_scalar) {
            G.noalias() = (Scalar(1) / R_) * (B * B.transpose());
        } else {
            const MatNUNX Rinv_Bt = R_.llt().solve(B.transpose());
            G.noalias() = B * Rinv_Bt;
        }
        return G;
    }
};
