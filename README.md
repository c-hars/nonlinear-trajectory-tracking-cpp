# nonlinear-trajectory-tracking-cpp

C++ implementation of the **SD-OPT**  controller, optimised for execution on embedded hardware (a Teensy 4.1). The algorithm itself is developed and documented in the companion repository [`nonlinear-trajectory-tracking`](https://github.com/c-hars/nonlinear-trajectory-tracking).

SD-OPT (state-dependent optimal preview tracking) solves a warm-started state-dependent Riccati equation at each timestep, with a cold solve for the first iteration. For the associated Stein and Riccati equations, doubling algorithms are used as the embedded-friendly dedicated solvers — fast, lightweight, and numerically robust.

## Build

Eigen (5.0.0) for linear algebra. PlatformIO for Teensy deployment.

The Makefile uses `-MMD -MP` for automatic header dependency tracking. Set `EIGEN_DIR` to your Eigen install path (defaults to `$(HOME)/eigen`). PlatformIO reads `EIGEN_DIR` from the environment via `${sysenv.EIGEN_DIR}`.

Three important compile-time switches live in `types/defs.h` (toggled there directly, or via `-D` flags):
- `SDOPT_USE_FLOAT` — `1` for single precision, `0` for double (default: `1`)
- `SDOPT_TEENSY_BUILD` — `1` enables the Teensy-optimised controller (hand-tiled kernels, `RWeight` templating, ...)  (default: `0`)
- `NK_USE_INCREMENT_PROXY` — `1` uses a cheaper Newton-increment stopping test instead of the full DARE residual test (default: `0`)

## Key results
- Control outputs: accuracy diffchecked and validated against MATLAB reference — under both single/double precision, and generic/Teensy-optimised builds
- Verified across regimes: actuator commands match the MATLAB reference code, including at the extreme (point-of-failure) trajectories
    - Riccati residuals are within the specified 1e-4 threshold; convergence sequence matches the `dlyap`/`idare`-based reference
    - NK iteration stats and other diagnostics also cross-verified.
- Consistent sub-millisecond compute profile established, on (Teensy) hardware.
    


*Hexacopter trajectory tracking: 12 state system with 6 control inputs, tracking 6 output references. Loop rate 100Hz, preview horizon 2.0s (200 samples).*

| Platform | Precision | Median (µs) | p95 (µs) | Budget util. |
|---|---|---:|---:|---:|
| Teensy 4.1 (600 MHz Cortex-M7) | float | **956** | **957** | 9.6 % |
| Teensy 4.1 (600 MHz Cortex-M7) | double | 2 369 | 2 535 | 23.7 % |
| Desktop | double | 19 | 23 | 0.2 % |



---

## Appendix: demo runs
```
=== SDDRE Timing Test (Teensy 4.1, float precision) ===
NX=12 NU=6 NY=6  N=501

   k,   sdc,  dare,    ff, total, it,     res, ok, fb
   1,   131,  2055,   740,  2926,  8, 3.0e-06,  1,  0  u = [ -415.1  -531.6  -480.9  -273.8   -94.8  -182.9]
   2,   130,  1048,   287,  1466,  2, 4.5e-06,  1,  0  u = [ -408.6  -506.3  -461.5  -317.0  -182.8  -250.1]
   3,   131,   541,   287,   959,  1, 7.5e-05,  1,  0  u = [ -369.5  -450.0  -418.0  -356.2  -259.5  -302.0]
   4,   130,   542,   286,   958,  1, 2.7e-05,  1,  0  u = [ -346.4  -393.8  -367.9  -351.7  -300.3  -329.5]
   5,   130,   540,   286,   957,  1, 1.7e-05,  1,  0  u = [ -323.5  -354.8  -332.0  -339.6  -307.3  -331.1]
   6,   130,   540,   286,   957,  1, 1.6e-05,  1,  0  u = [ -303.3  -330.8  -308.6  -323.1  -295.2  -318.0]
   7,   130,   540,   286,   957,  1, 1.7e-05,  1,  0  u = [ -284.7  -312.9  -290.6  -304.2  -275.6  -298.4]
   8,   130,   540,   286,   957,  1, 1.7e-05,  1,  0  u = [ -266.2  -294.9  -272.9  -284.2  -255.0  -277.6]
   9,   131,   540,   286,   957,  1, 1.8e-05,  1,  0  u = [ -247.2  -275.0  -254.0  -264.3  -236.0  -257.6]
  10,   131,   540,   286,   957,  1, 1.8e-05,  1,  0  u = [ -227.8  -253.6  -234.0  -244.6  -218.5  -238.5]
  20,   129,   540,   286,   956,  1, 1.5e-05,  1,  0  u = [  -36.9   -34.8   -34.9   -51.1   -53.1   -53.0]
  40,   129,   540,   286,   955,  1, 5.3e-07,  1,  0  u = [  115.5   140.9   116.7   134.7   109.5   133.5]
  60,   129,   540,   286,   956,  1, 3.7e-07,  1,  0  u = [  111.3   131.1   111.1   129.2   109.4   129.5]
  80,   129,   540,   286,   956,  1, 1.0e-06,  1,  0  u = [   84.6   103.5    85.2   100.9    82.1   100.4]
 100,   129,   540,   285,   956,  1, 2.3e-06,  1,  0  u = [   53.8    74.4    56.3    72.1    51.5    69.5]
 120,   129,   541,   286,   956,  1, 3.5e-06,  1,  0  u = [   67.0    74.1    71.4    76.1    68.9    71.6]
 140,   129,   541,   286,   956,  1, 2.4e-06,  1,  0  u = [  149.9   124.3   151.8   130.0   155.8   128.1]
 160,   129,   541,   285,   957,  1, 6.5e-07,  1,  0  u = [  235.2   200.3   234.5   203.4   238.4   204.1]
 180,   130,   540,   286,   956,  1, 1.5e-07,  1,  0  u = [  274.4   256.5   273.8   258.1   276.0   258.7]
 200,   129,   540,   286,   955,  1, 2.5e-07,  1,  0  u = [  273.9   270.1   273.9   271.8   275.6   271.7]
 220,   129,   541,   286,   956,  1, 7.0e-07,  1,  0  u = [  244.1   245.4   244.7   247.7   246.3   247.1]
 240,   129,   540,   286,   955,  1, 1.5e-06,  1,  0  u = [  211.6   213.8   212.1   215.6   213.4   215.0]
 260,   129,   541,   286,   956,  1, 1.7e-06,  1,  0  u = [  214.1   216.8   214.0   216.0   213.2   216.1]
 280,   129,   541,   286,   956,  1, 1.0e-06,  1,  0  u = [  252.6   253.6   252.2   250.9   249.9   251.3]
 300,   129,   541,   286,   956,  1, 2.8e-07,  1,  0  u = [  282.9   279.4   282.8   277.3   280.7   277.3]
 320,   129,   540,   286,   955,  1, 1.5e-07,  1,  0  u = [  272.4   264.9   272.8   263.1   270.6   262.7]
 340,   129,   539,   286,   954,  1, 4.6e-07,  1,  0  u = [  216.8   199.8   217.3   197.4   214.4   196.9]
 360,   129,   540,   286,   955,  1, 1.6e-06,  1,  0  u = [  136.7   107.0   136.4   102.9   132.4   103.1]
 380,   129,   541,   286,   956,  1, 3.5e-06,  1,  0  u = [   84.5    54.9    81.1    50.9    80.4    54.1]
 400,   129,   540,   286,   957,  1, 3.1e-06,  1,  0  u = [   84.1    92.3    79.1    93.8    85.7    99.0]
 420,   129,   540,   286,   957,  1, 1.2e-06,  1,  0  u = [   93.1   114.5    90.3   117.7    96.3   120.7]
 440,   129,   540,   286,   955,  1, 1.1e-06,  1,  0  u = [   54.9    48.5    52.2    52.3    58.7    55.0]
 460,   129,   541,   286,   956,  1, 3.7e-06,  1,  0  u = [  -25.1   -84.7   -28.6   -76.8   -16.5   -73.6]
 480,   129,   541,   286,   956,  1, 8.8e-06,  1,  0  u = [  -60.2  -110.0   -51.5  -100.1   -49.5  -108.1]
 500,   129,   541,   286,   957,  1, 4.1e-06,  1,  0  u = [   10.4    76.9    25.2    72.2     6.1    56.4]

--- Timing (us) --- sdc    dare      ff   total
  median:           129     540     286     956
  p95:              130     541     286     957
  max:              131    2055     740    2926

--- Health ---
  budget @ 100 Hz : 10000 us
  median util     : 9.6 %
  worst-case util : 29.3 %
  SDA fallbacks   : 0 / 501
  DARE failures   : 0 / 501
  worst residual  : 7.46e-05

Done.
```

```
=== SDDRE Timing Test (desktop, double precision) ===
NX=12 NU=6 NY=6  N=501

   k,   sdc,  dare,    ff, total, it,     res, ok, fb
   1,    42,    80,    33,   158,  8, 3.0e-06,  1,  0  u = [ -415.1  -531.6  -480.9  -273.8   -94.8  -182.9]
   2,     7,    46,    11,    64,  2, 4.5e-06,  1,  0  u = [ -408.6  -506.3  -461.5  -317.0  -182.8  -250.1]
   3,     8,    24,    10,    42,  1, 7.5e-05,  1,  0  u = [ -369.4  -450.0  -418.0  -356.2  -259.5  -302.0]
   4,     7,    22,    10,    39,  1, 2.7e-05,  1,  0  u = [ -346.4  -393.8  -367.9  -351.7  -300.3  -329.5]
   5,     7,    21,    10,    38,  1, 1.7e-05,  1,  0  u = [ -323.5  -354.8  -332.0  -339.6  -307.3  -331.1]
   6,     6,    20,    10,    36,  1, 1.6e-05,  1,  0  u = [ -303.3  -330.8  -308.5  -323.1  -295.2  -318.0]
   7,     7,    21,    10,    38,  1, 1.7e-05,  1,  0  u = [ -284.7  -312.9  -290.6  -304.2  -275.6  -298.4]
   8,     6,    20,    10,    36,  1, 1.7e-05,  1,  0  u = [ -266.2  -294.9  -272.9  -284.2  -255.0  -277.6]
   9,     6,    19,    10,    35,  1, 1.8e-05,  1,  0  u = [ -247.2  -275.0  -254.0  -264.3  -236.0  -257.6]
  10,     6,    22,     9,    37,  1, 1.8e-05,  1,  0  u = [ -227.8  -253.6  -234.0  -244.6  -218.5  -238.5]
  20,     5,    21,     9,    35,  1, 1.5e-05,  1,  0  u = [  -36.9   -34.8   -34.9   -51.1   -53.1   -53.0]
  40,     5,    20,     9,    34,  1, 5.1e-07,  1,  0  u = [  115.5   140.9   116.7   134.7   109.5   133.5]
  60,     6,    20,     9,    35,  1, 3.6e-07,  1,  0  u = [  111.3   131.1   111.1   129.2   109.4   129.5]
  80,     5,    20,     9,    34,  1, 1.0e-06,  1,  0  u = [   84.6   103.5    85.2   100.9    82.1   100.4]
 100,     5,    19,     9,    33,  1, 2.3e-06,  1,  0  u = [   53.8    74.4    56.3    72.1    51.5    69.5]
 120,     5,    19,     9,    33,  1, 3.5e-06,  1,  0  u = [   67.0    74.1    71.4    76.1    68.9    71.6]
 140,     5,    20,     9,    34,  1, 2.5e-06,  1,  0  u = [  149.9   124.3   151.8   130.0   155.8   128.1]
 160,     4,    19,     9,    33,  1, 6.6e-07,  1,  0  u = [  235.2   200.3   234.5   203.4   238.4   204.1]
 180,     4,    19,     9,    32,  1, 1.0e-07,  1,  0  u = [  274.4   256.5   273.8   258.1   276.0   258.7]
 200,     5,    19,     9,    33,  1, 2.1e-07,  1,  0  u = [  273.9   270.1   273.9   271.8   275.6   271.7]
 220,     5,    19,     9,    33,  1, 7.1e-07,  1,  0  u = [  244.1   245.4   244.7   247.7   246.3   247.1]
 240,     5,    19,     9,    33,  1, 1.5e-06,  1,  0  u = [  211.6   213.8   212.1   215.6   213.4   215.0]
 260,     5,    19,     9,    33,  1, 1.8e-06,  1,  0  u = [  214.1   216.8   214.0   216.0   213.2   216.1]
 280,     5,    19,     9,    33,  1, 9.6e-07,  1,  0  u = [  252.6   253.6   252.2   250.9   249.9   251.3]
 300,     5,    20,     9,    34,  1, 2.5e-07,  1,  0  u = [  282.9   279.4   282.8   277.3   280.7   277.3]
 320,     5,    19,     9,    33,  1, 1.2e-07,  1,  0  u = [  272.4   264.9   272.8   263.1   270.6   262.7]
 340,     5,    19,     9,    33,  1, 4.9e-07,  1,  0  u = [  216.8   199.8   217.3   197.4   214.4   196.9]
 360,     5,    20,     9,    34,  1, 1.6e-06,  1,  0  u = [  136.7   107.0   136.4   102.9   132.4   103.1]
 380,     5,    19,     9,    33,  1, 3.5e-06,  1,  0  u = [   84.5    54.9    81.1    50.9    80.4    54.1]
 400,     5,    19,     9,    33,  1, 3.1e-06,  1,  0  u = [   84.2    92.3    79.1    93.8    85.7    99.0]
 420,     5,    20,     9,    35,  1, 1.2e-06,  1,  0  u = [   93.1   114.5    90.3   117.7    96.3   120.7]
 440,     5,    21,     9,    35,  1, 1.1e-06,  1,  0  u = [   54.9    48.5    52.2    52.3    58.7    55.0]
 460,     5,    19,     9,    33,  1, 3.6e-06,  1,  0  u = [  -25.1   -84.7   -28.6   -76.8   -16.5   -73.6]
 480,     5,    19,     9,    33,  1, 8.8e-06,  1,  0  u = [  -60.2  -110.0   -51.6  -100.1   -49.5  -108.1]
 500,     6,    19,     9,    34,  1, 4.1e-06,  1,  0  u = [   10.4    76.9    25.2    72.2     6.1    56.4]

--- Timing (us) --- sdc    dare      ff   total
  median:             5      19       9      33
  p95:                6      21      10      37
  max:               42      80      33     158

--- Health ---
  budget @ 100 Hz : 10000 us
  median util     : 0.3 %
  worst-case util : 1.6 %
  SDA fallbacks   : 0 / 501
  DARE failures   : 0 / 501
  worst residual  : 7.46e-05

Done.
```