// C = A * B. All 12x12, column-major, C must not alias A or B.
static inline void mm12(double* __restrict C,
                        const double* __restrict A,
                        const double* __restrict B)
{
    for (int j = 0; j < 12; j += 4) {
        const double* Bj = B + 12*j;
        for (int i = 0; i < 12; i += 2) {
            double c00=0, c01=0, c02=0, c03=0;
            double c10=0, c11=0, c12=0, c13=0;
            const double* Ai = A + i;
            for (int k = 0; k < 12; ++k) {
                const double a0 = Ai[12*k],  a1 = Ai[12*k + 1];
                const double b0 = Bj[k],      b1 = Bj[k + 12];
                const double b2 = Bj[k + 24], b3 = Bj[k + 36];
                c00 += a0*b0; c01 += a0*b1; c02 += a0*b2; c03 += a0*b3;
                c10 += a1*b0; c11 += a1*b1; c12 += a1*b2; c13 += a1*b3;
            }
            double* Cij = C + i + 12*j;
            Cij[0]  = c00; Cij[1]  = c10;
            Cij[12] = c01; Cij[13] = c11;
            Cij[24] = c02; Cij[25] = c12;
            Cij[36] = c03; Cij[37] = c13;
        }
    }
}

// C = A * B'. All 12x12, column-major, C must not alias A or B.
static inline void mm12_abt(double* __restrict C,
                            const double* __restrict A,
                            const double* __restrict B)
{
    for (int j = 0; j < 12; j += 4) {
        const double* Bj = B + j;              // B(j..j+3, k) are contiguous
        for (int i = 0; i < 12; i += 2) {
            double c00=0, c01=0, c02=0, c03=0;
            double c10=0, c11=0, c12=0, c13=0;
            const double* Ai = A + i;
            for (int k = 0; k < 12; ++k) {
                const double a0 = Ai[12*k],     a1 = Ai[12*k + 1];
                const double b0 = Bj[12*k],     b1 = Bj[12*k + 1];
                const double b2 = Bj[12*k + 2], b3 = Bj[12*k + 3];
                c00 += a0*b0; c01 += a0*b1; c02 += a0*b2; c03 += a0*b3;
                c10 += a1*b0; c11 += a1*b1; c12 += a1*b2; c13 += a1*b3;
            }
            double* Cij = C + i + 12*j;
            Cij[0]  = c00; Cij[1]  = c10;
            Cij[12] = c01; Cij[13] = c11;
            Cij[24] = c02; Cij[25] = c12;
            Cij[36] = c03; Cij[37] = c13;
        }
    }
}

// C = A * B', valid ONLY when the result is symmetric
// (here: A = P X with X symmetric, B = P).
// Computes the 2x4 blocks touching the lower triangle (i >= j),
// then mirrors lower -> upper. C must not alias A or B.
static inline void mm12_abt_sym(double* __restrict C,
                                const double* __restrict A,
                                const double* __restrict B)
{
    for (int j = 0; j < 12; j += 4) {
        const double* Bj = B + j;
        for (int i = j; i < 12; i += 2) {
            double c00=0, c01=0, c02=0, c03=0;
            double c10=0, c11=0, c12=0, c13=0;
            const double* Ai = A + i;
            for (int k = 0; k < 12; ++k) {
                const double a0 = Ai[12*k],     a1 = Ai[12*k + 1];
                const double b0 = Bj[12*k],     b1 = Bj[12*k + 1];
                const double b2 = Bj[12*k + 2], b3 = Bj[12*k + 3];
                c00 += a0*b0; c01 += a0*b1; c02 += a0*b2; c03 += a0*b3;
                c10 += a1*b0; c11 += a1*b1; c12 += a1*b2; c13 += a1*b3;
            }
            double* Cij = C + i + 12*j;
            Cij[0]  = c00; Cij[1]  = c10;
            Cij[12] = c01; Cij[13] = c11;
            Cij[24] = c02; Cij[25] = c12;
            Cij[36] = c03; Cij[37] = c13;
        }
    }
    // Mirror lower -> upper. Overwrites the few upper entries the
    // diagonal blocks did compute — enforces exact symmetry.
    for (int c = 1; c < 12; ++c)
        for (int r = 0; r < c; ++r)
            C[r + 12*c] = C[c + 12*r];
}

// C = A * B. A 12x12, B and C 12x6, column-major, C must not alias A or B.
static inline void mm12x6(double* __restrict C,
                          const double* __restrict A,
                          const double* __restrict B)
{
    for (int j = 0; j < 6; j += 3) {
        const double* Bj = B + 12*j;
        for (int i = 0; i < 12; i += 2) {
            double c00=0, c01=0, c02=0;
            double c10=0, c11=0, c12=0;
            const double* Ai = A + i;
            for (int k = 0; k < 12; ++k) {
                const double a0 = Ai[12*k], a1 = Ai[12*k + 1];
                const double b0 = Bj[k], b1 = Bj[k + 12], b2 = Bj[k + 24];
                c00 += a0*b0; c01 += a0*b1; c02 += a0*b2;
                c10 += a1*b0; c11 += a1*b1; c12 += a1*b2;
            }
            double* Cij = C + i + 12*j;
            Cij[0]  = c00; Cij[1]  = c10;
            Cij[12] = c01; Cij[13] = c11;
            Cij[24] = c02; Cij[25] = c12;
        }
    }
}