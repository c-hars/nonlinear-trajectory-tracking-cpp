#pragma once
// ============================================================
//  sddre_model.h — SDC parameterisation + ZOH discretisation
//
//  Ports of:
//    get_A_matrix_SDRE_QuaternionAttitude.m
//    get_B_matrix_SDRE.m
//    c2d_zoh_expm.m
// ============================================================

#include "sddre_types.h"
#include "expm_pade.h"

// ============================================================
//  get_A_sdc_quaternion
//
//  State (0-based here, 1-based in MATLAB):
//    [0:2]  pos          (MATLAB 1:3)
//    [3:5]  vel          (MATLAB 4:6)
//    [6:8]  q1,q2,q3     (MATLAB 7:9)    q0 recovered from norm
//    [9:11] wx,wy,wz     (MATLAB 10:12)
// ============================================================
inline MatNX get_A_sdc_quaternion(const VecNX& x, const QuadParams& qp)
{
    const Scalar q1 = x(6), q2 = x(7), q3 = x(8);

    // q0 from the norm constraint. Clamped at 0 so a slightly
    // over-unity vector part can't produce NaN — MATLAB would
    // go complex here and warn below 0.1.
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
    // Deliberately omitted (same treatment as the Euler
    // formulation):  A.block<3,3>(3,3) = -skew(w);

    // ---- Gravity residual, "symmetric split" factorisation --
    //  Exploits 1 - q0^2 = q1^2 + q2^2 + q3^2 so the affine
    //  term folds into the SDC form.
    Mat3 Ag;
    Ag <<     -q3,  2*q0,   -q1,
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
//  Uses the SDC-factorised thrust  (2*w_nom + du)  rather than
//  the true nonlinear  w = w_nom + du,  so that B(du)*du
//  reproduces the quadratic  kF*w^2  term exactly.
// ============================================================
inline MatNXNU get_B_sdc(const VecNU& delta_u, const QuadParams& qp)
{
    MatNXNU Bc = MatNXNU::Zero();

    // Effective per-rotor term, gated by the health mask.
    const VecNU w = (Scalar(2) * qp.nominal_omegas + delta_u)
                        .cwiseProduct(qp.enabled);

    for (int j = 0; j < NU; ++j) {
        const Scalar f = qp.kF * w(j);
        Bc(5,  j) =  f              / qp.m;      // vel_z <- F_t / m
        Bc(9,  j) =  f * qp.y(j)    / qp.I_xx;   // wx_dot <- tau_x / I_xx
        Bc(10, j) = -f * qp.x(j)    / qp.I_yy;   // wy_dot <- tau_y / I_yy
        Bc(11, j) = -qp.kM * w(j) * qp.dirs(j) / qp.I_zz;  // wz_dot
    }
    return Bc;
}

// ============================================================
//  c2d_zoh_expm  —  ZOH discretisation via Van Loan's method
//                   (doi:10.1109/tac.1978.1101743)
//
//    M = expm([Ac Bc; 0 0] * Ts)
//    Ad = M(1:n, 1:n),  Bd = M(1:n, n+1:end)
//
//  The augmented matrix would be 18x18 here.  Its bottom NU rows
//  are zero, so every power keeps the form [P Q; 0 cI] and the
//  exponential comes out as [Ad Bd; 0 I].  expm_pade_vanloan
//  carries only the (P,Q,c) triple through the same Pade
//  builders, so the zero blocks are never multiplied and the
//  final solve is 12x12 rather than 18x18.  ~2.3x.
// ============================================================
inline void c2d_zoh_expm(
    const MatNX& Ac, const MatNXNU& Bc, Scalar Ts,
    MatNX& Ad, MatNXNU& Bd)
{
    expm_pade_vanloan(Ac, Bc, Ts, Ad, Bd);
}

// // ============================================================
// //  c2d_zoh_expm  —  ZOH discretisation via Van Loan's method
// //                   (doi:10.1109/tac.1978.1101743)
// //
// //    M = expm([Ac Bc; 0 0] * Ts)
// //    Ad = M(1:n, 1:n),  Bd = M(1:n, n+1:end)
// //
// //  The augmented matrix is 18x18 here. Its bottom NU rows are
// //  zero, so the exponential has the structure [Ad Bd; 0 I] —
// //  exploitable, but the general expm keeps this auditable and
// //  the cost is already modest.
// // ============================================================
// inline void c2d_zoh_expm(
//     const MatNX& Ac, const MatNXNU& Bc, Scalar Ts,
//     MatNX& Ad, MatNXNU& Bd)
// {
//     MatAug M = MatAug::Zero();
//     M.topLeftCorner<NX, NX>()  = Ac * Ts;
//     M.topRightCorner<NX, NU>() = Bc * Ts;

//     const MatAug E = expm_pade(M);

//     Ad = E.topLeftCorner<NX, NX>();
//     Bd = E.topRightCorner<NX, NU>();
// }
