#pragma once
// ============================================================
//  expm_pade.h — Matrix exponential, scaling-and-squaring Padé
//
//  Higham (2005), "The scaling and squaring method for the
//  matrix exponential revisited", SIAM J. Matrix Anal. Appl.
//  26(4):1179-1193.  Same algorithm family as MATLAB's expm.
//
//  Hand-rolled rather than <unsupported/Eigen/MatrixFunctions>
//  so the cost is predictable and the code is auditable — for
//  a Teensy timing benchmark you want to know exactly which
//  branch you're paying for.
//
//  For Ac*Ts with Ts <= 0.05 the 1-norm of the augmented 18x18
//  lands in the m=7 or m=9 branch with s=0 (no squaring), i.e.
//  3-4 GEMMs + one 18x18 LU solve. Deterministic.
// ============================================================

#include <Eigen/Dense>
#include <cmath>

template <typename MatT>
MatT expm_pade(const MatT& Ain)
{
    using S = typename MatT::Scalar;
    static_assert(MatT::RowsAtCompileTime > 0,
                  "expm_pade requires a fixed-size matrix");

    const MatT I = MatT::Identity();

    // Matrix 1-norm = max absolute column sum.
    // (NB: Eigen's lpNorm<1>() is the ENTRYWISE sum — not this.)
    const S nA = Ain.cwiseAbs().colwise().sum().maxCoeff();

    // // Higham's degree-selection thresholds for double precision.
    const S th3  = S(1.495585217958292e-2);
    const S th5  = S(2.539398330063230e-1);
    const S th7  = S(9.504178996162932e-1);
    const S th9  = S(2.097847961257068e+0);
    const S th13 = S(5.371920351148152e+0);

    // Higham's degree-selection thresholds adapted for single precision?


    MatT A = Ain;
    MatT U, V;
    int  s = 0;

    if (nA <= th3) {
        const MatT A2 = A * A;
        U = A * (A2 + S(60.0) * I);
        V = S(12.0) * A2 + S(120.0) * I;

    } else if (nA <= th5) {
        const MatT A2 = A * A;
        const MatT A4 = A2 * A2;
        U = A * (A4 + S(420.0) * A2 + S(15120.0) * I);
        V = S(30.0) * A4 + S(3360.0) * A2 + S(30240.0) * I;

    } else if (nA <= th7) {
        const MatT A2 = A * A;
        const MatT A4 = A2 * A2;
        const MatT A6 = A4 * A2;
        U = A * (A6 + S(1512.0) * A4 + S(277200.0) * A2 + S(8648640.0) * I);
        V = S(56.0) * A6 + S(25200.0) * A4 + S(1995840.0) * A2
          + S(17297280.0) * I;

    } else if (nA <= th9) {
        const MatT A2 = A * A;
        const MatT A4 = A2 * A2;
        const MatT A6 = A4 * A2;
        const MatT A8 = A6 * A2;
        U = A * (A8 + S(3960.0) * A6 + S(2162160.0) * A4
               + S(302702400.0) * A2 + S(8821612800.0) * I);
        V = S(90.0) * A8 + S(110880.0) * A6 + S(30270240.0) * A4
          + S(2075673600.0) * A2 + S(17643225600.0) * I;

    } else {
        // Degree 13 with scaling.
        if (nA > th13) {
            s = static_cast<int>(std::ceil(std::log2(double(nA / th13))));
            if (s < 0) s = 0;
            A = Ain * std::pow(S(2), -s);
        }
        const MatT A2 = A * A;
        const MatT A4 = A2 * A2;
        const MatT A6 = A4 * A2;

        U = A * ( A6 * (A6 + S(16380.0) * A4 + S(40840800.0) * A2)
                + S(33522128640.0)      * A6
                + S(10559470521600.0)   * A4
                + S(1187353796428800.0) * A2
                + S(32382376266240000.0) * I );

        V =       A6 * (S(182.0) * A6 + S(960960.0) * A4
                        + S(1323241920.0) * A2)
                + S(670442572800.0)      * A6
                + S(129060195264000.0)   * A4
                + S(7771770303897600.0)  * A2
                + S(64764752532480000.0) * I;
    }

    // r_m(A) = (V - U)^{-1} (V + U)
    MatT X = (V - U).partialPivLu().solve(V + U);

    // Undo the scaling by repeated squaring.
    for (int i = 0; i < s; ++i) X = (X * X).eval();

    return X;
}
