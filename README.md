# nonlinear-trajectory-tracking-cpp

C++ implementation of the **SD-OPT**  controller, optimised for execution on embedded hardware (Teensy 4.x). The algorithm itself is developed and documented in the companion repository [`nonlinear-trajectory-tracking`](https://github.com/c-hars/nonlinear-trajectory-tracking).

SD-OPT ("state-dependent optimal preview tracking") solves a state-dependent Riccati equation at each timestep; a cold solve for the first iteration, with subsequent solves warm-started from the previous solution. The warm solver is iterative, meaning real time iteration (RTI) schemes can be used – for consistent per-step compute – otherwise solves are completed within some specified tolerance (DARE residual).

Solving Stein and Riccati equations is a key part of the SD-OPT algorithm; *doubling algorithms* are used as the embedded-friendly solvers here – fast, lightweight, numerically robust. Note that these solver implementations are not specific to the SD-OPT algorithm, and are welcome to be reused in other C++ applications.

## Build

Eigen (5.0.0) for linear algebra, PlatformIO for Teensy deployment.

The install path for Eigen (`EIGEN_DIR`) needs to be set manually in the `Makefile`, though the default `~/eigen` may already work. PlatformIO reads `EIGEN_DIR` from the environment via `${sysenv.EIGEN_DIR}` – see `platformio.ini`.

Three build switches live in `types/defs.h`:
- `SDOPT_USE_FLOAT` — `1` for single precision, `0` for double
- `SDOPT_TEENSY_BUILD` — `0` for the generic, Eigen-centric controller (best for desktop PCs), `1` for the Teensy-optimised controller (structure exploitation, matrix multiply kernels tailored to the Cortex-M7, ...).
- `NK_USE_INCREMENT_PROXY` — `0` for a direct DARE residual evaluation in the iterative solve stopping test; `1` uses a cheaper Newton-increment proxy.
Toggle them there directly, or via `-D` flags.

## Key results
- **Accuracy**: validated against MATLAB reference – under both single/double precision, and generic/Teensy-optimised builds
  - when solve-to-tolerance is used, Riccati **residuals** are indeed within the specified 1e-4 threshold. The **convergence sequence** also matches the MATLAB-based reference. **Control inputs** match against MATLAB's output, when using either the doubling-based solvers or MATLAB's `dlyap`/`dare` solvers.
  - **under single/double precision**; no practical difference, and single precision is much faster, unlocking sub-ms execution. The numerical robustness is largely coming from sticking to Riccati theory / closed-form solutions, and using the closed-loop matrix over the prediction horizon. In contrast, nonlinear MPC struggles with this setup – it's a challenging optimisation problem (unstable plant / long prediction horizon / coarse sampling rate).
  - **verified across regimes**, including at the extreme (point-of-failure) trajectories.
- **Compute**: median/p95 execution is sub-millisecond on the embedded target (Teensy 4.1), with constant per-step timing in RTI mode. There is also further headroom for optimisation – the stats here shouldn't be read as the absolute floor. The focus here was balanced between optimisation and maintaining readability/auditability/generality.
> Note 1: this *does not* immediately indicate kHz-rate control is possible – these results use 100Hz loop rates, and scaling to 1000Hz also increases the number of iterations in the preview recursion by 10x. Hitting kHz-rate control would be best achieved by algorithm changes, not just further optimisation of the C++ / this implementation; namely, decimation of the preview grid. The preview grid doesn't necessarily need to get finer just because the feedback runs faster. Higher-rate feedback is useful, while a finer preview grid tends to just bloat the compute time if the relevant plant/reference dynamics were already adequately captured at the coarser resolution. How far the preview can be decimated without affecting tracking is something to validate for the particular system.
>
> Those algorithm changes are currently implemented and validated, but not yet in the online repositories. Note, the decimation is not a new concept – it's much like move blocking in MPC.

> Note 2: this analysis also does not cover the terminal mode of SD-OPT ("shrinking horizon MPC"), which has a different compute profile. It's unlikely to be used in most tracking applications – the assumption that the world ends at the end of the maneuver is pretty niche :) An online tracking controller would generally not use this – just a fixed preview horizon, which is what this repo assumes. Terminal mode also has a higher, non-consistent compute, which is especially problematic at high sampling rates. The workload roughly looks like a strong spike upon entry, then a linear ramp toward zero at the end of the maneuver. Addressable; algorithm changes are implemented/validated but similarly not yet online.


### Timing numbers

Measured on hardware – desktop timing also included for comparison. Problem setup:
- Tracking: 5s figure 8 maneuver – near threshold of feasibility (4.5s)
- Control: 100Hz, preview horizon 2.0s (200 samples)
- System: NX = 12 (12 state system), NU = 6 (6 control inputs), NY = 6 (6 output references to track). Open loop unstable.

| Platform | Precision | Controller | DARE mode | Min (µs) | Median (µs) | p95 (µs) | Median budget util. |
|---|---|---|---|---:|---:|---:|---:|
| Teensy 4.1 (600 MHz Cortex-M7) | float | Teensy-optimised | RTI, 1 DARE iteration | **572** | **573** | **574** | 5.73% |
| Teensy 4.1 (600 MHz Cortex-M7) | float | Teensy-optimised | DARE to tolerance (1e-4) | **572** | **802** | **803** | 8.02% |
| Desktop | double | Generic | DARE to tolerance (1e-4) | **26** | **38** | **44** | 0.38% |

Cold-start initialisation takes approximately 2.56 ms on Teensy; subsequent steps sit between 0.57 ms and 0.80 ms, depending on the configuration (see appendix).

---

## Appendix: test run outputs

```
=== SD-OPT Timing Test (Teensy 4.1, float precision) ===
- Teensy-optimised controller
- R mode: scalar
- RTI: maximum of 1 DARE iteration per control step


   k,     sdc,    dare,      ff,   total, it,      res, ok, fb
   1,   93.70, 1858.27,  606.79, 2559.03,  9, 8.13e-06, 1, 0, u = [ -415.3  -532.1  -481.4  -274.0   -94.5  -182.7]
   2,   94.61,  259.20,  219.88,  573.96,  1, 1.68e-02, 0, 0, u = [ -378.5  -475.1  -444.3  -311.3  -165.9  -224.2]
   3,   94.67,  259.15,  219.89,  573.97,  1, 2.50e-02, 0, 0, u = [ -373.6  -463.6  -426.4  -354.7  -238.5  -291.9]
   4,   94.63,  259.21,  219.77,  573.86,  1, 5.53e-03, 0, 0, u = [ -354.0  -408.8  -380.1  -349.8  -284.9  -321.0]
   5,   94.62,  259.21,  219.80,  573.88,  1, 2.19e-03, 0, 0, u = [ -331.2  -367.7  -343.2  -341.4  -302.3  -329.3]
   6,   94.62,  259.29,  219.72,  573.88,  1, 2.32e-03, 0, 0, u = [ -309.2  -338.8  -315.6  -326.7  -296.2  -320.4]
   7,   94.62,  259.21,  219.92,  574.00,  1, 2.72e-03, 0, 0, u = [ -289.4  -318.3  -295.3  -308.7  -279.3  -302.9]
   8,   94.61,  259.21,  219.71,  573.78,  1, 3.09e-03, 0, 0, u = [ -270.5  -299.7  -277.1  -289.1  -259.5  -282.7]
   9,   94.64,  259.25,  219.78,  573.93,  1, 3.27e-03, 0, 0, u = [ -251.4  -279.7  -258.0  -268.9  -240.0  -262.3]
  10,   94.64,  259.18,  219.76,  573.83,  1, 3.30e-03, 0, 0, u = [ -231.6  -258.1  -237.8  -248.7  -221.8  -242.6]
  20,   93.41,  259.18,  219.84,  572.68,  1, 2.73e-03, 0, 0, u = [  -36.9   -34.9   -34.8   -51.4   -53.3   -53.5]
  40,   93.41,  258.97,  219.62,  572.25,  1, 9.86e-05, 1, 0, u = [  115.6   141.0   116.8   134.9   109.7   133.7]
  60,   93.46,  258.98,  219.64,  572.34,  1, 7.33e-05, 1, 0, u = [  111.2   131.0   111.0   129.1   109.4   129.4]
  80,   93.41,  259.18,  219.82,  572.67,  1, 1.99e-04, 0, 0, u = [   84.5   103.4    85.1   100.8    82.0   100.2]
 100,   93.46,  259.16,  219.72,  572.60,  1, 4.21e-04, 0, 0, u = [   53.8    74.4    56.4    72.1    51.5    69.4]
 120,   93.41,  259.17,  219.79,  572.64,  1, 6.15e-04, 0, 0, u = [   67.2    74.3    71.6    76.1    69.0    71.6]
 140,   93.46,  259.16,  219.65,  572.53,  1, 4.34e-04, 0, 0, u = [  150.7   124.9   152.6   130.6   156.4   128.7]
 160,   93.41,  259.18,  219.66,  572.51,  1, 1.15e-04, 0, 0, u = [  235.8   201.1   235.1   204.1   239.0   204.8]
 180,   93.46,  258.96,  219.65,  572.33,  1, 1.82e-05, 1, 0, u = [  274.6   256.9   274.1   258.5   276.2   259.0]
 200,   93.41,  259.00,  219.58,  572.25,  1, 3.63e-05, 1, 0, u = [  273.7   270.0   273.8   271.7   275.4   271.6]
 220,   93.43,  259.23,  219.67,  572.59,  1, 1.20e-04, 0, 0, u = [  243.8   245.1   244.3   247.4   246.1   246.8]
 240,   93.41,  259.16,  219.64,  572.47,  1, 2.45e-04, 0, 0, u = [  211.4   213.6   211.9   215.5   213.3   214.9]
 260,   93.41,  259.23,  219.65,  572.55,  1, 2.90e-04, 0, 0, u = [  214.2   216.9   214.1   216.2   213.5   216.3]
 280,   93.41,  259.16,  219.66,  572.49,  1, 1.61e-04, 0, 0, u = [  253.0   253.9   252.6   251.3   250.4   251.7]
 300,   93.41,  259.05,  219.57,  572.29,  1, 4.26e-05, 1, 0, u = [  282.9   279.4   282.9   277.4   280.9   277.5]
 320,   93.41,  258.98,  219.58,  572.23,  1, 2.02e-05, 1, 0, u = [  272.1   264.5   272.5   262.8   270.3   262.4]
 340,   93.41,  259.05,  219.61,  572.33,  1, 8.56e-05, 1, 0, u = [  216.4   199.5   217.0   197.0   213.9   196.4]
 360,   93.41,  259.16,  219.69,  572.52,  1, 2.88e-04, 0, 0, u = [  136.6   107.0   136.4   102.8   132.2   102.9]
 380,   93.41,  259.23,  219.64,  572.54,  1, 6.10e-04, 0, 0, u = [   84.5    55.0    81.2    50.9    80.2    54.0]
 400,   93.41,  259.14,  219.61,  572.42,  1, 5.73e-04, 0, 0, u = [   84.2    92.6    79.2    94.0    85.6    99.1]
 420,   93.41,  259.23,  219.73,  572.62,  1, 2.21e-04, 0, 0, u = [   93.2   114.5    90.3   117.7    96.3   120.7]
 440,   93.41,  259.16,  219.70,  572.53,  1, 2.15e-04, 0, 0, u = [   55.3    48.7    52.3    52.6    59.2    55.5]
 460,   93.41,  259.23,  219.84,  572.74,  1, 7.02e-04, 0, 0, u = [  -24.5   -83.9   -28.1   -75.9   -15.8   -72.5]
 480,   93.41,  259.18,  219.87,  572.71,  1, 1.60e-03, 0, 0, u = [  -60.2  -110.3   -51.6  -100.4   -49.4  -108.3]
 500,   93.41,  259.21,  219.65,  572.53,  1, 7.51e-04, 0, 0, u = [   10.5    77.3    25.3    72.5     6.0    56.7]

--- Timing (us) ---  sdc     dare       ff    total
  median:         93.407  259.158  219.693  572.548
  p95:            94.615  259.228  219.910  573.783
  max:            94.687 1858.270  606.793 2559.028

--- Health ---
  budget @ 100 Hz : 10000 us
  median util     : 5.73 %
  worst-case util : 25.59 %
  NK iterations   : 500 total
  SDA fallbacks   : 0 / 501
  DARE failures   : 365 / 501 <- NB: expected. "RTI, capped at 1 iteration, fails to (consistently) hit the requested 1e-4 tolerance" – it's the tradeoff of fixed iteration schemes.
  worst residual  : 2.50e-02

Done.
```



```
=== SD-OPT Timing Test (Teensy 4.1, float precision) ===
- Same as above, except no RTI: solve the DARE to tolerance (1e-4)

=== R mode: scalar ===
   k,     sdc,    dare,      ff,   total, it,      res, ok, fb
   1,   93.70, 1858.23,  606.24, 2558.44,  9, 8.13e-06, 1, 0, u = [ -415.3  -532.1  -481.4  -274.0   -94.5  -182.7]
   2,   94.61,  717.36,  219.82, 1032.06,  3, 1.66e-07, 1, 0, u = [ -390.0  -475.3  -437.5  -298.6  -165.4  -232.2]
   3,   94.61,  717.38,  219.87, 1032.12,  3, 1.16e-07, 1, 0, u = [ -382.0  -451.3  -417.0  -331.5  -239.0  -288.3]
   4,   94.62,  488.20,  219.81,  802.89,  2, 5.70e-05, 1, 0, u = [ -357.4  -405.7  -377.5  -343.5  -285.6  -321.1]
   5,   94.62,  488.20,  219.81,  802.89,  2, 3.33e-06, 1, 0, u = [ -331.1  -365.9  -341.2  -339.8  -302.5  -329.6]
   6,   94.61,  488.23,  219.81,  802.91,  2, 9.98e-07, 1, 0, u = [ -308.4  -337.2  -314.3  -326.4  -296.7  -320.6]
   7,   94.61,  488.25,  219.77,  802.89,  2, 1.22e-06, 1, 0, u = [ -288.1  -316.1  -293.6  -308.0  -279.6  -302.6]
   8,   94.64,  488.22,  219.77,  802.89,  2, 1.53e-06, 1, 0, u = [ -268.8  -296.8  -274.9  -287.8  -259.4  -281.8]
   9,   94.61,  488.15,  219.81,  802.83,  2, 1.72e-06, 1, 0, u = [ -249.3  -276.5  -255.6  -267.5  -239.9  -261.4]
  10,   94.62,  488.21,  219.77,  802.86,  2, 1.78e-06, 1, 0, u = [ -229.6  -254.8  -235.2  -247.3  -221.8  -241.8]
  20,   93.41,  488.15,  219.91,  801.72,  2, 1.67e-06, 1, 0, u = [  -36.3   -32.0   -32.7   -50.4   -54.6   -53.9]
  40,   93.41,  258.96,  219.58,  572.21,  1, 9.90e-05, 1, 0, u = [  115.6   141.0   116.8   134.9   109.7   133.7]
  60,   93.41,  259.03,  219.61,  572.31,  1, 7.33e-05, 1, 0, u = [  111.2   131.0   111.0   129.1   109.4   129.4]
  80,   93.41,  488.22,  219.90,  801.78,  2, 8.82e-09, 1, 0, u = [   84.4   103.3    85.1   100.8    82.0   100.2]
 100,   93.41,  488.22,  219.72,  801.61,  2, 4.07e-08, 1, 0, u = [   53.5    74.1    56.1    72.0    51.5    69.4]
 120,   93.41,  488.22,  219.74,  801.62,  2, 8.90e-08, 1, 0, u = [   66.7    73.7    71.0    75.8    68.8    71.4]
 140,   93.41,  488.20,  219.67,  801.53,  2, 4.53e-08, 1, 0, u = [  150.4   124.6   152.2   130.1   156.0   128.3]
 160,   93.41,  488.22,  219.59,  801.47,  2, 3.66e-09, 1, 0, u = [  235.8   201.0   235.1   204.0   238.8   204.7]
 180,   93.41,  259.03,  219.59,  572.29,  1, 1.82e-05, 1, 0, u = [  274.6   256.9   274.1   258.5   276.2   259.0]
 200,   93.41,  258.98,  219.58,  572.23,  1, 3.63e-05, 1, 0, u = [  273.7   270.0   273.8   271.7   275.4   271.6]
 220,   93.41,  488.16,  219.75,  801.58,  2, 3.08e-09, 1, 0, u = [  243.7   245.1   244.2   247.2   245.9   246.7]
 240,   93.41,  488.20,  219.70,  801.57,  2, 1.36e-08, 1, 0, u = [  211.3   213.5   211.7   215.1   212.9   214.7]
 260,   93.41,  488.22,  219.67,  801.55,  2, 2.00e-08, 1, 0, u = [  214.1   216.8   213.9   215.8   213.1   216.0]
 280,   93.41,  488.22,  219.71,  801.59,  2, 5.80e-09, 1, 0, u = [  252.9   253.8   252.4   251.1   250.3   251.6]
 300,   93.41,  258.93,  219.59,  572.18,  1, 4.26e-05, 1, 0, u = [  282.9   279.4   282.9   277.4   280.9   277.5]
 320,   93.41,  258.95,  219.61,  572.23,  1, 2.02e-05, 1, 0, u = [  272.1   264.5   272.5   262.8   270.3   262.4]
 340,   93.41,  258.93,  219.63,  572.23,  1, 8.56e-05, 1, 0, u = [  216.4   199.5   217.0   197.0   213.9   196.4]
 360,   93.41,  488.20,  219.71,  801.58,  2, 2.02e-08, 1, 0, u = [  136.7   107.0   136.4   102.8   132.3   103.1]
 380,   93.41,  488.22,  219.76,  801.65,  2, 9.19e-08, 1, 0, u = [   84.6    54.9    81.1    51.1    80.7    54.5]
 400,   93.41,  488.22,  219.68,  801.57,  2, 8.39e-08, 1, 0, u = [   84.3    92.5    79.2    94.3    86.1    99.5]
 420,   93.41,  488.21,  219.79,  801.67,  2, 1.46e-08, 1, 0, u = [   93.3   114.5    90.4   117.8    96.5   120.8]
 440,   93.41,  488.22,  219.73,  801.62,  2, 9.79e-09, 1, 0, u = [   55.4    48.8    52.4    52.7    59.4    55.7]
 460,   93.41,  488.21,  219.88,  801.76,  2, 1.15e-07, 1, 0, u = [  -23.9   -83.6   -27.8   -75.2   -14.9   -71.6]
 480,   93.41,  488.21,  219.90,  801.78,  2, 6.57e-07, 1, 0, u = [  -59.0  -108.8   -50.0   -98.5   -47.8  -106.9]
 500,   93.46,  488.17,  219.70,  801.59,  2, 2.80e-07, 1, 0, u = [   10.8    78.2    26.3    73.4     6.4    57.0]

--- Timing (us) ---  sdc     dare       ff    total
  median:         93.407  488.163  219.708  801.567
  p95:            94.613  488.230  219.903  802.795
  max:            94.662 1858.230  606.245 2558.443

--- Health ---
  budget @ 100 Hz : 10000 us
  median util     : 8.02 %
  worst-case util : 25.58 %
  NK iterations   : 867 total
  SDA fallbacks   : 0 / 501
  DARE failures   : 0 / 501
  worst residual  : 9.91e-05

Done.

```



```
=== SD-OPT Timing Test (desktop, double precision) ===
- Generic controller
- R mode: dense
- No RTI: solve the DARE to tolerance (1e-4)

   k,     sdc,    dare,      ff,   total, it,      res, ok, fb
   1,    5.70,   46.60,   17.10,   69.80,  9, 7.68e-09, 1, 0, u = [ -415.3  -532.1  -481.4  -274.0   -94.5  -182.7]
   2,    7.00,   39.10,   10.90,   57.20,  3, 1.66e-07, 1, 0, u = [ -390.0  -475.3  -437.5  -298.6  -165.4  -232.2]
   3,    6.60,   40.40,   11.60,   58.80,  3, 1.14e-07, 1, 0, u = [ -382.0  -451.3  -417.0  -331.5  -239.0  -288.3]
   4,    6.00,   27.90,   10.30,   44.60,  2, 5.71e-05, 1, 0, u = [ -357.4  -405.7  -377.5  -343.5  -285.6  -321.1]
   5,    6.00,   26.90,   11.10,   44.30,  2, 3.31e-06, 1, 0, u = [ -331.1  -365.9  -341.2  -339.8  -302.5  -329.6]
   6,    7.50,   28.60,   11.00,   47.40,  2, 9.92e-07, 1, 0, u = [ -308.4  -337.2  -314.3  -326.4  -296.7  -320.6]
   7,    6.80,   27.00,   11.00,   45.00,  2, 1.22e-06, 1, 0, u = [ -288.1  -316.1  -293.6  -308.0  -279.6  -302.6]
   8,    6.40,   26.90,   10.90,   44.90,  2, 1.53e-06, 1, 0, u = [ -268.7  -296.8  -274.9  -287.8  -259.4  -281.8]
   9,    7.00,   27.40,   11.10,   46.00,  2, 1.72e-06, 1, 0, u = [ -249.3  -276.5  -255.6  -267.5  -239.9  -261.4]
  10,    6.50,   26.50,   10.90,   44.20,  2, 1.78e-06, 1, 0, u = [ -229.5  -254.8  -235.2  -247.4  -221.8  -241.8]
  20,    4.60,   25.50,    9.40,   39.50,  2, 1.67e-06, 1, 0, u = [  -36.3   -32.0   -32.7   -50.4   -54.6   -53.9]
  40,    4.40,   13.50,    9.20,   27.40,  1, 9.90e-05, 1, 0, u = [  115.6   141.0   116.8   134.9   109.7   133.7]
  60,    4.50,   13.70,    9.30,   27.70,  1, 7.33e-05, 1, 0, u = [  111.2   131.0   111.0   129.1   109.4   129.4]
  80,    4.50,   26.30,    9.30,   40.20,  2, 8.60e-09, 1, 0, u = [   84.4   103.3    85.0   100.8    82.0   100.2]
 100,    3.80,   24.20,    9.20,   37.40,  2, 4.09e-08, 1, 0, u = [   53.5    74.1    56.1    72.0    51.5    69.4]
 120,    4.50,   24.20,    9.20,   38.10,  2, 9.11e-08, 1, 0, u = [   66.7    73.7    71.0    75.8    68.8    71.4]
 140,    3.80,   24.20,    9.20,   37.30,  2, 4.58e-08, 1, 0, u = [  150.4   124.6   152.2   130.1   156.0   128.3]
 160,    3.80,   24.20,    9.20,   37.40,  2, 3.74e-09, 1, 0, u = [  235.8   201.0   235.0   204.0   238.8   204.7]
 180,    3.70,   12.60,    9.10,   25.80,  1, 1.82e-05, 1, 0, u = [  274.6   256.9   274.1   258.4   276.2   259.0]
 200,    4.40,   12.60,    9.40,   26.60,  1, 3.63e-05, 1, 0, u = [  273.7   270.0   273.8   271.7   275.4   271.6]
 220,    3.80,   24.30,    9.30,   37.50,  2, 3.23e-09, 1, 0, u = [  243.7   245.1   244.2   247.2   245.9   246.7]
 240,    3.80,   24.50,    9.30,   37.90,  2, 1.39e-08, 1, 0, u = [  211.3   213.5   211.7   215.1   212.9   214.7]
 260,    3.90,   24.20,    9.30,   37.50,  2, 1.96e-08, 1, 0, u = [  214.1   216.8   213.9   215.8   213.1   216.0]
 280,    3.80,   24.20,    9.20,   37.40,  2, 5.86e-09, 1, 0, u = [  252.9   253.8   252.5   251.1   250.3   251.6]
 300,    3.80,   12.60,    9.20,   25.80,  1, 4.26e-05, 1, 0, u = [  283.0   279.4   282.9   277.4   280.9   277.5]
 320,    3.80,   12.70,    9.20,   25.90,  1, 2.02e-05, 1, 0, u = [  272.1   264.5   272.5   262.8   270.3   262.4]
 340,    3.80,   12.80,    9.30,   26.00,  1, 8.56e-05, 1, 0, u = [  216.4   199.5   217.0   197.0   213.9   196.4]
 360,    3.80,   24.20,    9.30,   37.60,  2, 2.01e-08, 1, 0, u = [  136.7   107.0   136.4   102.8   132.3   103.1]
 380,    4.70,   24.80,    9.30,   39.10,  2, 9.36e-08, 1, 0, u = [   84.6    54.9    81.1    51.1    80.7    54.5]
 400,    3.80,   24.30,    9.20,   37.50,  2, 8.37e-08, 1, 0, u = [   84.3    92.5    79.2    94.3    86.1    99.5]
 420,    4.50,   26.00,    9.20,   39.90,  2, 1.49e-08, 1, 0, u = [   93.3   114.5    90.4   117.8    96.5   120.9]
 440,    4.50,   25.90,    9.20,   39.80,  2, 1.03e-08, 1, 0, u = [   55.4    48.8    52.4    52.7    59.4    55.7]
 460,    4.50,   24.20,    9.30,   38.10,  2, 1.15e-07, 1, 0, u = [  -23.9   -83.6   -27.8   -75.2   -14.9   -71.6]
 480,    4.50,   24.30,    9.30,   38.20,  2, 6.57e-07, 1, 0, u = [  -59.0  -108.8   -50.0   -98.5   -47.8  -106.9]
 500,    4.50,   24.30,    9.20,   38.10,  2, 2.82e-07, 1, 0, u = [   10.8    78.2    26.3    73.4     6.4    57.0]

--- Timing (us) ---  sdc     dare       ff    total
  median:          4.400   24.300    9.300   38.000
  p95:             6.400   26.900   10.700   44.100
  max:             9.500   46.600   17.100   69.800

--- Health ---
  budget @ 100 Hz : 10000 us
  median util     : 0.38 %
  worst-case util : 0.70 %
  NK iterations   : 867 total
  SDA fallbacks   : 0 / 501
  DARE failures   : 0 / 501
  worst residual  : 9.91e-05

Done.
```