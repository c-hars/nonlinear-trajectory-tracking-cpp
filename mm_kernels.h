#pragma once
// ============================================================
//  mm_kernels.h — hand-tiled small matrix kernels
//
//  Column-major throughout. Precision dispatch is by plain
//  function overloading on double* / float*, so call sites are
//  identical in either build and there is no template body for
//  the compiler to (fail to) optimise.
//
//  Every accumulator is a named local scalar, written out by
//  hand. A first attempt used one templated body with a local
//  c[MR][NR] array, relying on -O3 scalar replacement to put it
//  in registers; arm-none-eabi-gcc did NOT, and
//  every accumulator update became a stack load/store pair
//  (~2x on every stage). Hence: no accumulator arrays, ever.
//
//  Tile shapes — set by the FPv5-D16 register file, which has
//  16 double registers aliasing 32 single registers:
//
//    double: 2x4 accumulator block (8 regs) + operands.
//    float:  4x4 accumulator block (16 regs) + operands —
//            twice the reuse per load, and FP32 FMA issues
//            faster on the M7 on top of that.
//
//  mm12x6: double 2x3, float 4x3 (6 columns, so NR = 3).
// ============================================================

// ============================================================
//  double kernels — 2x4 tiles
// ============================================================

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

// y = F*x + w. F 12x12 column-major; x, y, w 12-vectors.
// y must not alias x. Accumulators start from w — the add is free.
static inline void mv12_fma(double* __restrict y,
                            const double* __restrict F,
                            const double* __restrict x,
                            const double* __restrict w)
{
    double a0=w[0], a1=w[1], a2 =w[2],  a3 =w[3],
           a4=w[4], a5=w[5], a6 =w[6],  a7 =w[7],
           a8=w[8], a9=w[9], a10=w[10], a11=w[11];
    for (int k = 0; k < 12; ++k) {
        const double xk = x[k];
        const double* Fk = F + 12*k;      // column k
        a0  += Fk[0]*xk;  a1  += Fk[1]*xk;  a2  += Fk[2]*xk;
        a3  += Fk[3]*xk;  a4  += Fk[4]*xk;  a5  += Fk[5]*xk;
        a6  += Fk[6]*xk;  a7  += Fk[7]*xk;  a8  += Fk[8]*xk;
        a9  += Fk[9]*xk;  a10 += Fk[10]*xk; a11 += Fk[11]*xk;
    }
    y[0]=a0; y[1]=a1; y[2]=a2;  y[3]=a3;
    y[4]=a4; y[5]=a5; y[6]=a6;  y[7]=a7;
    y[8]=a8; y[9]=a9; y[10]=a10; y[11]=a11;
}

// ============================================================
//  float kernels — 4x4 tiles
//
//  16 accumulators + 4 a + 4 b operands = 24 of the 32 S
//  registers; the k loop does 16 FMAs per 8 loads instead of
//  the double kernel's 8 per 6.
// ============================================================

// C = A * B. All 12x12, column-major, C must not alias A or B.
static inline void mm12(float* __restrict C,
                        const float* __restrict A,
                        const float* __restrict B)
{
    for (int j = 0; j < 12; j += 4) {
        const float* Bj = B + 12*j;
        for (int i = 0; i < 12; i += 4) {
            float c00=0, c01=0, c02=0, c03=0;
            float c10=0, c11=0, c12=0, c13=0;
            float c20=0, c21=0, c22=0, c23=0;
            float c30=0, c31=0, c32=0, c33=0;
            const float* Ai = A + i;
            for (int k = 0; k < 12; ++k) {
                const float a0 = Ai[12*k],     a1 = Ai[12*k + 1];
                const float a2 = Ai[12*k + 2], a3 = Ai[12*k + 3];
                const float b0 = Bj[k],      b1 = Bj[k + 12];
                const float b2 = Bj[k + 24], b3 = Bj[k + 36];
                c00 += a0*b0; c01 += a0*b1; c02 += a0*b2; c03 += a0*b3;
                c10 += a1*b0; c11 += a1*b1; c12 += a1*b2; c13 += a1*b3;
                c20 += a2*b0; c21 += a2*b1; c22 += a2*b2; c23 += a2*b3;
                c30 += a3*b0; c31 += a3*b1; c32 += a3*b2; c33 += a3*b3;
            }
            float* Cij = C + i + 12*j;
            Cij[0]  = c00; Cij[1]  = c10; Cij[2]  = c20; Cij[3]  = c30;
            Cij[12] = c01; Cij[13] = c11; Cij[14] = c21; Cij[15] = c31;
            Cij[24] = c02; Cij[25] = c12; Cij[26] = c22; Cij[27] = c32;
            Cij[36] = c03; Cij[37] = c13; Cij[38] = c23; Cij[39] = c33;
        }
    }
}

// C = A * B'. All 12x12, column-major, C must not alias A or B.
static inline void mm12_abt(float* __restrict C,
                            const float* __restrict A,
                            const float* __restrict B)
{
    for (int j = 0; j < 12; j += 4) {
        const float* Bj = B + j;               // B(j..j+3, k) are contiguous
        for (int i = 0; i < 12; i += 4) {
            float c00=0, c01=0, c02=0, c03=0;
            float c10=0, c11=0, c12=0, c13=0;
            float c20=0, c21=0, c22=0, c23=0;
            float c30=0, c31=0, c32=0, c33=0;
            const float* Ai = A + i;
            for (int k = 0; k < 12; ++k) {
                const float a0 = Ai[12*k],     a1 = Ai[12*k + 1];
                const float a2 = Ai[12*k + 2], a3 = Ai[12*k + 3];
                const float b0 = Bj[12*k],     b1 = Bj[12*k + 1];
                const float b2 = Bj[12*k + 2], b3 = Bj[12*k + 3];
                c00 += a0*b0; c01 += a0*b1; c02 += a0*b2; c03 += a0*b3;
                c10 += a1*b0; c11 += a1*b1; c12 += a1*b2; c13 += a1*b3;
                c20 += a2*b0; c21 += a2*b1; c22 += a2*b2; c23 += a2*b3;
                c30 += a3*b0; c31 += a3*b1; c32 += a3*b2; c33 += a3*b3;
            }
            float* Cij = C + i + 12*j;
            Cij[0]  = c00; Cij[1]  = c10; Cij[2]  = c20; Cij[3]  = c30;
            Cij[12] = c01; Cij[13] = c11; Cij[14] = c21; Cij[15] = c31;
            Cij[24] = c02; Cij[25] = c12; Cij[26] = c22; Cij[27] = c32;
            Cij[36] = c03; Cij[37] = c13; Cij[38] = c23; Cij[39] = c33;
        }
    }
}

// C = A * B', valid ONLY when the result is symmetric
// (here: A = P X with X symmetric, B = P).
// 4x4 tiles on the same 4-grid in i and j, so i >= j covers the
// lower triangle exactly; mirror lower -> upper afterwards.
// C must not alias A or B.
static inline void mm12_abt_sym(float* __restrict C,
                                const float* __restrict A,
                                const float* __restrict B)
{
    for (int j = 0; j < 12; j += 4) {
        const float* Bj = B + j;
        for (int i = j; i < 12; i += 4) {
            float c00=0, c01=0, c02=0, c03=0;
            float c10=0, c11=0, c12=0, c13=0;
            float c20=0, c21=0, c22=0, c23=0;
            float c30=0, c31=0, c32=0, c33=0;
            const float* Ai = A + i;
            for (int k = 0; k < 12; ++k) {
                const float a0 = Ai[12*k],     a1 = Ai[12*k + 1];
                const float a2 = Ai[12*k + 2], a3 = Ai[12*k + 3];
                const float b0 = Bj[12*k],     b1 = Bj[12*k + 1];
                const float b2 = Bj[12*k + 2], b3 = Bj[12*k + 3];
                c00 += a0*b0; c01 += a0*b1; c02 += a0*b2; c03 += a0*b3;
                c10 += a1*b0; c11 += a1*b1; c12 += a1*b2; c13 += a1*b3;
                c20 += a2*b0; c21 += a2*b1; c22 += a2*b2; c23 += a2*b3;
                c30 += a3*b0; c31 += a3*b1; c32 += a3*b2; c33 += a3*b3;
            }
            float* Cij = C + i + 12*j;
            Cij[0]  = c00; Cij[1]  = c10; Cij[2]  = c20; Cij[3]  = c30;
            Cij[12] = c01; Cij[13] = c11; Cij[14] = c21; Cij[15] = c31;
            Cij[24] = c02; Cij[25] = c12; Cij[26] = c22; Cij[27] = c32;
            Cij[36] = c03; Cij[37] = c13; Cij[38] = c23; Cij[39] = c33;
        }
    }
    // Mirror lower -> upper. Overwrites the few upper entries the
    // diagonal blocks did compute — enforces exact symmetry.
    for (int c = 1; c < 12; ++c)
        for (int r = 0; r < c; ++r)
            C[r + 12*c] = C[c + 12*r];
}

// C = A * B. A 12x12, B and C 12x6, column-major, C must not alias A or B.
static inline void mm12x6(float* __restrict C,
                          const float* __restrict A,
                          const float* __restrict B)
{
    for (int j = 0; j < 6; j += 3) {
        const float* Bj = B + 12*j;
        for (int i = 0; i < 12; i += 4) {
            float c00=0, c01=0, c02=0;
            float c10=0, c11=0, c12=0;
            float c20=0, c21=0, c22=0;
            float c30=0, c31=0, c32=0;
            const float* Ai = A + i;
            for (int k = 0; k < 12; ++k) {
                const float a0 = Ai[12*k],     a1 = Ai[12*k + 1];
                const float a2 = Ai[12*k + 2], a3 = Ai[12*k + 3];
                const float b0 = Bj[k], b1 = Bj[k + 12], b2 = Bj[k + 24];
                c00 += a0*b0; c01 += a0*b1; c02 += a0*b2;
                c10 += a1*b0; c11 += a1*b1; c12 += a1*b2;
                c20 += a2*b0; c21 += a2*b1; c22 += a2*b2;
                c30 += a3*b0; c31 += a3*b1; c32 += a3*b2;
            }
            float* Cij = C + i + 12*j;
            Cij[0]  = c00; Cij[1]  = c10; Cij[2]  = c20; Cij[3]  = c30;
            Cij[12] = c01; Cij[13] = c11; Cij[14] = c21; Cij[15] = c31;
            Cij[24] = c02; Cij[25] = c12; Cij[26] = c22; Cij[27] = c32;
        }
    }
}

// y = F*x + w. F 12x12 column-major; x, y, w 12-vectors.
// y must not alias x. Accumulators start from w — the add is free.
// 12 accumulators + operands fits the 32 S registers easily.
static inline void mv12_fma(float* __restrict y,
                            const float* __restrict F,
                            const float* __restrict x,
                            const float* __restrict w)
{
    float a0=w[0], a1=w[1], a2 =w[2],  a3 =w[3],
          a4=w[4], a5=w[5], a6 =w[6],  a7 =w[7],
          a8=w[8], a9=w[9], a10=w[10], a11=w[11];
    for (int k = 0; k < 12; ++k) {
        const float xk = x[k];
        const float* Fk = F + 12*k;       // column k
        a0  += Fk[0]*xk;  a1  += Fk[1]*xk;  a2  += Fk[2]*xk;
        a3  += Fk[3]*xk;  a4  += Fk[4]*xk;  a5  += Fk[5]*xk;
        a6  += Fk[6]*xk;  a7  += Fk[7]*xk;  a8  += Fk[8]*xk;
        a9  += Fk[9]*xk;  a10 += Fk[10]*xk; a11 += Fk[11]*xk;
    }
    y[0]=a0; y[1]=a1; y[2]=a2;  y[3]=a3;
    y[4]=a4; y[5]=a5; y[6]=a6;  y[7]=a7;
    y[8]=a8; y[9]=a9; y[10]=a10; y[11]=a11;
}