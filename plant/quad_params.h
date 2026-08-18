#pragma once
// ============================================================
//  plant/quad_params.h — hexacopter plant parameters
//
//  Ported from load_copter_params.m – C++ version of MATLAB's "qp" struct.
//  Values are defined exactly as per the MATLAB code (for identical numerical precision).
// ============================================================

#include "types/defs.h"

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
