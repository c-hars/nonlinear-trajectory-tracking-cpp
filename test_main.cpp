// ============================================================
//  test_main.cpp — SDDRE timing test harness
//
//  Dual-target: Teensy 4.1 (Arduino) or desktop (g++/clang++).
//
//  Desktop:
//    g++ -std=c++17 -O2 -DNDEBUG -DEIGEN_NO_DEBUG -D_USE_MATH_DEFINES -I"C:\Users\conor\eigen" test_main.cpp -o test_sddre.exe
//
//  Teensy / PlatformIO:
//    build_flags in platformio.ini
//    (EIGEN_NO_DEBUG / NDEBUG matter a lot — assertions in the fixed-size paths are otherwise a large fraction of runtime.)
//
//  C6: the harness runs the same trajectory twice, once with
//  RMode::Dense (pre-C6 arithmetic) and once with RMode::Scalar,
//  in one process. Both instantiations exist in one binary
//  deliberately — comparing across separate builds or sessions
//  is not trustworthy at this effect size.
//
//  Trajectory data — exported from MATLAB, row-major.
//
// ============================================================


#include "sddre_controller.h"

#ifdef ARDUINO
  #include <Arduino.h>
#endif

#ifndef ARDUINO
  #include <cstdio>
  #include <cstring>
  #include <algorithm>
  #include <vector>
#endif

constexpr int N_STEPS = 501;
static Scalar x_traj[N_STEPS * NX];
static Scalar u_traj[N_STEPS * NU];
static Scalar r_traj[N_STEPS * NY];

// Steps discarded before timing starts. First call pays for cold
// caches and the SDA cold init; neither is steady-state cost.
constexpr int N_WARMUP = 20;

#ifdef ARDUINO
  #include "trajectory_data.h"   // x_traj_data[] etc. stay const double

  template <typename Src, typename Dst>
  static void copy_narrow(const Src* src, Dst* dst, size_t n) {
      for (size_t i = 0; i < n; ++i) dst[i] = static_cast<Dst>(src[i]);
  }

  static bool load_trajectory_data() {
      copy_narrow(x_traj_data, x_traj, size_t(N_STEPS) * NX);
      copy_narrow(u_traj_data, u_traj, size_t(N_STEPS) * NU);
      copy_narrow(r_traj_data, r_traj, size_t(N_STEPS) * NY);
      return true;
  }
#endif

#ifndef ARDUINO
  static bool load_bin(const char* path, Scalar* buf, size_t n) {
      FILE* f = std::fopen(path, "rb");
      if (!f) { std::printf("ERROR: cannot open %s\n", path); return false; }
      // MATLAB export is always float64 — stage and narrow.
      std::vector<double> tmp(n);
      size_t got = std::fread(tmp.data(), sizeof(double), n, f);
      std::fclose(f);
      if (got != n) {
          std::printf("ERROR: %s — expected %zu doubles, got %zu\n", path, n, got);
          return false;
      }
      for (size_t i = 0; i < n; ++i) buf[i] = static_cast<Scalar>(tmp[i]);
      return true;
  }

  static bool load_trajectory_data() {
      return load_bin("x_traj.bin", x_traj, N_STEPS * NX)
          && load_bin("u_traj.bin", u_traj, N_STEPS * NU)
          && load_bin("r_traj.bin", r_traj, N_STEPS * NY);
  }
#endif

// ============================================================
//  Hexacopter parameters
//   6 arms at 60 deg spacing, alternating spin directions.
// ============================================================
static void setup_params(QuadParams& qp) {
    qp.Ts    = 0.01;
    qp.m     = 5.0;
    qp.I_xx  = 0.008;
    qp.I_yy  = 0.009;
    qp.I_zz  = 0.015;
    qp.kF    = 0.000015;
    qp.kM    = 0.00000015;
    qp.g     = 9.81;

    const double L = 0.20;
    for (int j = 0; j < NU; ++j) {
        const double phi = j * (M_PI / 3.0);
        qp.x(j)    = L * std::cos(phi + M_PI / 6.0);
        qp.y(j)    = L * std::sin(phi + M_PI / 6.0);
        qp.dirs(j) = (j % 2 == 0) ? -1.0 : 1.0;
    }
    qp.enabled.setOnes();

    // Hover: 6 * kF * w^2 = m*g
    const double w_hover = std::sqrt(qp.m * qp.g / (NU * qp.kF));
    qp.nominal_omegas.setConstant(w_hover);

    const double max_omega = 10000.0 * M_PI / 30.0;   // 10k RPM -> rad/s
    qp.max_du.setConstant(max_omega - w_hover);

    qp.quat_blend_alpha = 1.0;
}

// ============================================================
//  Dummy trajectory.
//  Timing is valid with dummy data; correctness is not.
//  Replaced with actual MATLAB export data for SIL validation.
// ============================================================
[[maybe_unused]] static void generate_dummy_data(const QuadParams& qp) {
    std::memset(x_traj, 0, sizeof(x_traj));
    std::memset(u_traj, 0, sizeof(u_traj));
    std::memset(r_traj, 0, sizeof(r_traj));

    for (int k = 0; k < N_STEPS; ++k) {
        const Scalar t = k * qp.Ts;
        Scalar* xk = x_traj + k * NX;

        xk[0]  =  1.0  * std::sin(0.6 * t);  // pos
        xk[1]  =  1.0  * std::sin(1.2 * t);
        xk[2]  =  0.5  * std::sin(0.4 * t);
        xk[3]  =  0.6  * std::cos(0.6 * t);  // vel
        xk[4]  =  1.2  * std::cos(1.2 * t);
        xk[5]  =  0.2  * std::cos(0.4 * t);
        xk[6]  =  0.12 * std::sin(1.0 * t);  // q1,q2,q3 (~14 deg tilt)
        xk[7]  =  0.12 * std::cos(1.0 * t);
        xk[8]  =  0.05 * std::sin(0.5 * t);
        xk[9]  =  0.3  * std::cos(1.0 * t);  // body rates
        xk[10] = -0.3  * std::sin(1.0 * t);
        xk[11] =  0.1  * std::cos(0.5 * t);

        Scalar* uk = u_traj + k * NU;
        for (int j = 0; j < NU; ++j)
            uk[j] = 5.0 * std::sin(0.8 * t + j);

        Scalar* rk = r_traj + k * NY;
        rk[0] = std::sin(0.6 * t);
        rk[1] = std::sin(1.2 * t);
        rk[2] = 0.5 * std::sin(0.4 * t);
    }
}

template <RMode RM>
static void setup_controller(SDDREControllerT<RM>& ctrl) {
    // sel = [1,2,3,9,10,11] (MATLAB) -> 0-based state cols 0,1,2,8,9,10
    ctrl.C.setZero();
    const int sel[NY] = {0, 1, 2, 8, 9, 10};
    for (int i = 0; i < NY; ++i) ctrl.C(i, sel[i]) = 1.0;

    ctrl.Qy.setZero();  ctrl.Qy.diagonal() << 1.0, 1.0, 1.0, 5.2525, 0.0131, 0.0131;
    ctrl.Qyf.setZero(); ctrl.Qyf.diagonal() << 47.0291, 46.9286, 43.9393, 119.3914, 0.0217, 0.0234;

    // C6: R = r*I, stated as such. In RMode::Dense this fills the
    // 6x6; in RMode::Scalar only r is kept.
    ctrl.R.set(Scalar(1.048e-6));

    ctrl.r_data = r_traj;
    ctrl.r_len  = N_STEPS;

    ctrl.opts.preview_horizon             = 2.0;
    ctrl.opts.use_full_fh_mpc_at_terminal = false;
    ctrl.opts.always_use_full_fh_mpc      = false;
    ctrl.opts.post_residual_check         = false;  // C3: true residual as health signal (costs ~one residual eval per step)

    ctrl.opts.dare.method    = DARESolverMethod::NK;
    ctrl.opts.dare.min_iters = 1;
    ctrl.opts.dare.max_iters = 10;
    ctrl.opts.dare.tolerance = 1e-4;

    setup_params(ctrl.qp);
    ctrl.reset();
}

#ifdef ARDUINO
  #define PRINT(...) Serial.printf(__VA_ARGS__)
#else
  #define PRINT(...) std::printf(__VA_ARGS__)
#endif

static Scalar pct(std::vector<Scalar> v, double p) {
    if (v.empty()) return 0;
    size_t idx = static_cast<size_t>(p / 100.0 * (v.size() - 1));
    std::nth_element(v.begin(), v.begin() + idx, v.end());
    return v[idx];
}

// ============================================================
//  One full pass over the trajectory.
//
//  Returns the median total time so the caller can report the
//  scalar-vs-dense difference directly. Iteration counts are
//  summed as well: they are the implementation-independent
//  quantity, and C6 is allowed to move them by a step or two
//  (see the bit-exactness note in sddre_types.h).
// ============================================================
struct RunResult {
    Scalar med_sdc = 0, med_dare = 0, med_ff = 0, med_tot = 0;
    Scalar p95_tot = 0, max_tot = 0;
    long   total_iters = 0;
    int    n_fallback = 0, n_fail = 0;
    Scalar worst_res = 0;
};

template <RMode RM>
static RunResult run_pass(bool verbose)
{
    // Static rather than stack-local: the controller carries ~5.5 kB of
    // fixed-size matrices, and the Wex cache is ~67 kB on the heap per
    // instantiation (12 x (N + M + 1) doubles). Two instantiations is
    // ~135 kB, comfortable in Teensy 4.1's RAM2.
    static SDDREControllerT<RM> ctrl;
    setup_controller(ctrl);

    // ---- Warm-up: discarded ----------------------------------
    for (int k = 1; k <= N_WARMUP; ++k) {
        const Eigen::Map<const VecNX> xk(x_traj + (k - 1) * NX);
        const Eigen::Map<const VecNU> uk(u_traj + (k - 1) * NU);
        SDDRESolveInfo winfo;
        volatile Scalar sink = ctrl.compute_u(k * ctrl.qp.Ts, xk, k, uk, winfo)(0);
        (void)sink;
    }
    ctrl.reset();   // back to the cold P_ss so k == 1 takes the SDA path

    std::vector<Scalar> t_sdc, t_dare, t_ff, t_tot;
    t_sdc.reserve(N_STEPS); t_dare.reserve(N_STEPS);
    t_ff.reserve(N_STEPS);  t_tot.reserve(N_STEPS);

    RunResult res;

    if (verbose)
        PRINT("   k,     sdc,    dare,      ff,   total, it,     res, ok, fb\n");

    for (int k = 1; k <= N_STEPS; ++k) {
        const Eigen::Map<const VecNX> xk(x_traj + (k - 1) * NX);
        const Eigen::Map<const VecNU> uk(u_traj + (k - 1) * NU);

        const sddre_tick_t t0 = sddre_ticks();
        SDDRESolveInfo info;
        const VecNU u = ctrl.compute_u(k * ctrl.qp.Ts, xk, k, uk, info);
        const Scalar total = _sddre_elapsed_us(t0);

        t_sdc.push_back(info.time_sdc_discretize_us);
        t_dare.push_back(info.time_dare_us);
        t_ff.push_back(info.time_feedforward_us);
        t_tot.push_back(total);

        res.total_iters += info.dare_info.solver_iterations;
        if (info.dare_info.used_sda_fallback) ++res.n_fallback;
        if (!info.dare_info.solve_success)    ++res.n_fail;
        if (info.dare_info.tol_achieved > res.worst_res)
            res.worst_res = info.dare_info.tol_achieved;

        // Print every step for the first 10, then every 20th
        if (verbose && (k <= 10 || k % 20 == 0)) {
            PRINT("%4d, %7.2f, %7.2f, %7.2f, %7.2f, %2d, %7.1e,  %d,  %d  ",
                  k, info.time_sdc_discretize_us, info.time_dare_us,
                  info.time_feedforward_us, total,
                  info.dare_info.solver_iterations,
                  info.dare_info.tol_achieved,
                  info.dare_info.solve_success ? 1 : 0,
                  info.dare_info.used_sda_fallback ? 1 : 0);
            PRINT("u = [%7.1f %7.1f %7.1f %7.1f %7.1f %7.1f]\n",
                  u(0), u(1), u(2), u(3), u(4), u(5));
        }
        (void)u;
    }

    res.med_sdc  = pct(t_sdc,  50);
    res.med_dare = pct(t_dare, 50);
    res.med_ff   = pct(t_ff,   50);
    res.med_tot  = pct(t_tot,  50);
    res.p95_tot  = pct(t_tot,  95);
    res.max_tot  = pct(t_tot, 100);

    if (verbose) {
        PRINT("\n--- Timing (us) ---%10s%9s%9s%9s\n", "sdc", "dare", "ff", "total");
        PRINT("  median: %14.3f%9.3f%9.3f%9.3f\n",
              res.med_sdc, res.med_dare, res.med_ff, res.med_tot);
        PRINT("  p95:    %14.3f%9.3f%9.3f%9.3f\n",
              pct(t_sdc,95), pct(t_dare,95), pct(t_ff,95), res.p95_tot);
        PRINT("  max:    %14.3f%9.3f%9.3f%9.3f\n",
              pct(t_sdc,100), pct(t_dare,100), pct(t_ff,100), res.max_tot);

        PRINT("\n--- Health ---\n");
        PRINT("  budget @ %.0f Hz : %.0f us\n", 1.0/ctrl.qp.Ts, 1e6*ctrl.qp.Ts);
        PRINT("  median util     : %.2f %%\n",
              100.0 * res.med_tot / (1e6*ctrl.qp.Ts));
        PRINT("  worst-case util : %.2f %%\n",
              100.0 * res.max_tot / (1e6*ctrl.qp.Ts));
        PRINT("  NK iterations   : %ld total\n", res.total_iters);
        PRINT("  SDA fallbacks   : %d / %d\n", res.n_fallback, N_STEPS);
        PRINT("  DARE failures   : %d / %d\n", res.n_fail, N_STEPS);
        PRINT("  worst residual  : %.2e\n", res.worst_res);
    }

    return res;
}

static void run_timing_test() {
    if (!load_trajectory_data()) {
        PRINT("Trajectory load failed — aborting.\n");
        return;
    }

    PRINT("clock resolution : %.5f us/tick\n", (double)sddre_tick_us());
    PRINT("x[0] = [%.4f %.4f %.4f ... %.4f %.4f %.4f]\n",
        x_traj[0], x_traj[1], x_traj[2],
        x_traj[NX-3], x_traj[NX-2], x_traj[NX-1]);
    PRINT("u[0] = [%.4f %.4f %.4f %.4f %.4f %.4f]\n\n",
        u_traj[0], u_traj[1], u_traj[2],
        u_traj[3], u_traj[4], u_traj[5]);

    PRINT("=== R mode: dense (pre-C6) ===\n");
    const RunResult dense = run_pass<RMode::Dense>(true);

    PRINT("\n\n=== R mode: scalar (C6) ===\n");
    const RunResult scal = run_pass<RMode::Scalar>(true);

    // Second dense pass: session drift within one process. If this
    // differs from the first by more than the scalar-vs-dense gap,
    // the gap is not measurable here and should not be reported.
    PRINT("\n\n=== R mode: dense (repeat, drift check) ===\n");
    const RunResult dense2 = run_pass<RMode::Dense>(false);

    auto rel = [](Scalar a, Scalar b) {
        return (b > 0) ? 100.0 * (double(a) - double(b)) / double(b) : 0.0;
    };

    PRINT("\n\n=== C6 summary (median us) ===\n");
    PRINT("%-10s%10s%10s%10s%10s\n", "stage", "dense", "scalar", "delta", "%");
    PRINT("%-10s%10.3f%10.3f%10.3f%10.2f\n", "sdc",
          dense.med_sdc, scal.med_sdc, scal.med_sdc - dense.med_sdc,
          rel(scal.med_sdc, dense.med_sdc));
    PRINT("%-10s%10.3f%10.3f%10.3f%10.2f\n", "dare",
          dense.med_dare, scal.med_dare, scal.med_dare - dense.med_dare,
          rel(scal.med_dare, dense.med_dare));
    PRINT("%-10s%10.3f%10.3f%10.3f%10.2f\n", "ff",
          dense.med_ff, scal.med_ff, scal.med_ff - dense.med_ff,
          rel(scal.med_ff, dense.med_ff));
    PRINT("%-10s%10.3f%10.3f%10.3f%10.2f\n", "total",
          dense.med_tot, scal.med_tot, scal.med_tot - dense.med_tot,
          rel(scal.med_tot, dense.med_tot));

    PRINT("\ndrift (dense pass 2 vs pass 1, total): %+.2f %%\n",
          rel(dense2.med_tot, dense.med_tot));
    PRINT("NK iterations  dense %ld  scalar %ld  (delta %+ld)\n",
          dense.total_iters, scal.total_iters,
          scal.total_iters - dense.total_iters);
    PRINT("worst residual dense %.2e  scalar %.2e\n",
          dense.worst_res, scal.worst_res);
    PRINT("\nInterpretation: the scalar-vs-dense gap is only meaningful if\n"
          "it exceeds the drift figure above.\n");
}

#ifdef ARDUINO

void setup() {
    Serial.begin(115200);
    while (!Serial) {}
    delay(500);
    sddre_timing_init();
    PRINT("\n=== SDDRE Timing Test (Teensy 4.1) ===\n");
    PRINT("NX=%d NU=%d NY=%d  N=%d\n\n", NX, NU, NY, N_STEPS);
    run_timing_test();
    PRINT("\nDone.\n");
}

void loop() {}

#else

int main() {
    sddre_timing_init();
    std::printf("\n=== SDDRE Timing Test (desktop) ===\n");
    std::printf("NX=%d NU=%d NY=%d  N=%d\n\n", NX, NU, NY, N_STEPS);
    run_timing_test();
    std::printf("\nDone.\n");
    return 0;
}

#endif