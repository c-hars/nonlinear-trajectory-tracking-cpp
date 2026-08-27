#pragma once
// ============================================================
//  plant/sdc_model.h — SDC parameterisation
//
//  Ports of:
//    get_A_matrix_SDRE_QuaternionAttitude.m
//    get_B_matrix_SDRE.m
// ============================================================

#include "types/defs.h"
#include "plant/quad_params.h"

#include <cmath>
#include <algorithm>

// ============================================================
//  get_A_sdc_quaternion
//
//  State (0-based here, 1-based in MATLAB):
//    [0:2]  pos      
//    [3:5]  vel      
//    [6:8]  q1,q2,q3
//    [9:11] wx,wy,wz 
// ============================================================
inline MatNX get_A_sdc_quaternion(const VecNX& x, const QuadParams& qp)
{
    const Scalar q1 = x(6), q2 = x(7), q3 = x(8);

    // q0 extraction (via the unit norm constraint)
    // NOTE: clamped at 0 so a slightly over-unity vector part can't produce NaN. No warnings as the singularity is approached however (unlike MATLAB).
    const Scalar q0sq = Scalar(1) - q1 * q1 - q2 * q2 - q3 * q3;
    const Scalar q0   = std::sqrt(std::max(q0sq, Scalar(0)));

    const Scalar wx = x(9), wy = x(10), wz = x(11);
    const Scalar g  = qp.g;

    MatNX A = MatNX::Zero();

    // ---- Position kinematics: body -> inertial DCM ----------
    Mat3 C_Ib;
    C_Ib << (q0*q0 + q1*q1 - q2*q2 - q3*q3),
            2*(q1*q2 - q0*q3),
            2*(q1*q3 + q0*q2),

            2*(q1*q2 + q0*q3),
            (q0*q0 - q1*q1 + q2*q2 - q3*q3),
            2*(q2*q3 - q0*q1),

            2*(q1*q3 - q0*q2),
            2*(q2*q3 + q0*q1),
            (q0*q0 - q1*q1 - q2*q2 + q3*q3);
    A.block<3,3>(0, 3) = C_Ib;

    // ---- Coriolis ------------------------------------------
    // Deliberately omitted (same treatment as all SDC
    // factorisations):  A.block<3,3>(3,3) = -skew(w);

    // ---- Gravity residual, "symmetric split" factorisation --
    //  Exploits 1 - q0^2 = q1^2 + q2^2 + q3^2 so the affine
    //  term folds into the SDC form.
    Mat3 Ag;
    Ag <<     -q3,  2*q0,    -q1,
            -2*q0,   -q3,    -q2,
             2*q1,  2*q2,      0;
    A.block<3,3>(3, 6) = g * Ag;

    // ---- Quaternion kinematics ------------------------------
    Mat3 Omega_w;
    Omega_w <<    0,   wz,  -wy,
                -wz,    0,   wx,
                 wy,  -wx,    0;
    Mat3 Xi_q;
    Xi_q <<  q0, -q3,  q2,
             q3,  q0, -q1,
            -q2,  q1,  q0;

    const Scalar alpha = qp.quat_blend_alpha;
    A.block<3,3>(6, 6) = alpha * Scalar(0.5) * Omega_w;
    A.block<3,3>(6, 9) = (Scalar(1) - alpha) * Scalar(0.5) * Xi_q
                       + alpha * Scalar(0.5) * q0 * Mat3::Identity();

    // ---- Gyroscopic coupling --------------------------------
    A(9, 10)  = -x(11) * (qp.I_zz - qp.I_yy) / qp.I_xx;
    A(10, 11) =  -x(9) * (qp.I_xx - qp.I_zz) / qp.I_yy;
    A(11, 9)  = -x(10) * (qp.I_yy - qp.I_xx) / qp.I_zz;

    return A;
}

// ============================================================
//  get_B_sdc  —  SDC mixer
//
//  Uses the SDC-factorised thrust  (2*w_nom + du) so that B(du)*du
//  reproduces the quadratic  kF*w^2  term exactly.
// ============================================================
inline MatNXNU get_B_sdc(const VecNU& delta_u, const QuadParams& qp)
{
    MatNXNU Bc = MatNXNU::Zero();

    // Effective per-rotor term, gated by the health mask.
    const VecNU w = (Scalar(2) * qp.nominal_omegas + delta_u).cwiseProduct(qp.enabled);

    for (int j = 0; j < NU; ++j) {
        const Scalar f = qp.kF * w(j);
        Bc(5,  j) =  f / qp.m;
        Bc(9,  j) =  f * qp.y(j) / qp.I_xx;                // wx_dot = tau_x / I_xx
        Bc(10, j) = -f * qp.x(j) / qp.I_yy;                // wy_dot = tau_y / I_yy
        Bc(11, j) = -qp.kM * w(j) * qp.dirs(j) / qp.I_zz;  // wz_dot = tau_z / I_zz
    }
    return Bc;
}
