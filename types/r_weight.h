#pragma once
// ============================================================
//  types/r_weight.h — Structure exploitation of R
//
//  R is often r*I; this wrapper is parameterised on how much
//  structure R may assume:
//
//    RMode::Scalar — stores r alone. Eliminates factorisations
//                    and replaces matrix ops with scaled products.
//    RMode::Dense  — stores the full 6×6; general-case arithmetic.
//
//  Only the two structural extremes (Scalar and Dense) are supported —
//  these bracket any intermediate structures (e.g. diagonal, block-diagonal).
//
//  NB: modes are not necessarily bit-identical (e.g. r*(K'K)
//  vs K'*(R*K) – these can differ at the last bit).
// ============================================================

#include "types/defs.h"

#include <cmath>
#include <type_traits>
#include <cassert>

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
    //  Scalar: one gemm with alpha = r.  Dense: temporary + gemm.
    //  rankUpdate (lower-tri only) was tried — slower on Cortex-M7; mirror-copy cost dominates.
    void add_KtRK(MatNX& X, const MatNUNX& K) const {
        if constexpr (is_scalar) X.noalias() += R_ * (K.transpose() * K);
        else                     X.noalias() += K.transpose() * R_ * K;
    }

    //  G = B R^{-1} B'
    //  Scalar mode replaces the LLT factorisation + solve with one scalar division.
    //  Cold path only (k == 1 and Newton-Kleinman fallback).
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
