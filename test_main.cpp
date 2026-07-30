// ============================================================
//  test_main.cpp — SDDRE timing test harness
//
//  Dual-target: Teensy 4.1 (Arduino) or desktop (g++/clang++).
//
//  Desktop:
//    g++ -std=c++17 -O2 -D_USE_MATH_DEFINES -I"C:\Users\conor\eigen" test_main.cpp -o test_sddre.exe
//
//  Teensy / PlatformIO:
//    build_flags in platformio.ini
//    (EIGEN_NO_DEBUG / NDEBUG matter a lot — assertions in the fixed-size paths are otherwise a large fraction of runtime.)
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

#ifdef ARDUINO
  #include "trajectory_data.h"   // x_traj_data[] etc. stay const double

  template <typename Src, typename Dst>
  static void copy_narrow(const Src* src, Dst* dst, size_t n) {
      for (size_t i = 0; i < n; ++i) dst[i] = static_cast<Dst>(src[i]);
  }

  static void load_trajectory_data() {
      copy_narrow(x_traj_data, x_traj, size_t(N_STEPS) * NX);
      copy_narrow(u_traj_data, u_traj, size_t(N_STEPS) * NU);
      copy_narrow(r_traj_data, r_traj, size_t(N_STEPS) * NY);
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
static void generate_dummy_data(const QuadParams& qp) {
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

static SDDREController ctrl;

static void setup_controller() {
    // sel = [1,2,3,9,10,11] (MATLAB) -> 0-based state cols 0,1,2,8,9,10
    ctrl.C.setZero();
    const int sel[NY] = {0, 1, 2, 8, 9, 10};
    for (int i = 0; i < NY; ++i) ctrl.C(i, sel[i]) = 1.0;

    ctrl.Qy.setZero();  ctrl.Qy.diagonal() << 1.0, 1.0, 1.0, 5.2525, 0.0131, 0.0131;
    ctrl.Qyf.setZero(); ctrl.Qyf.diagonal() << 47.0291, 46.9286, 43.9393, 119.3914, 0.0217, 0.0234;
    ctrl.R.setIdentity();   ctrl.R  *= 1.048 * 1e-6;

    ctrl.r_data = r_traj;
    ctrl.r_len  = N_STEPS;

    ctrl.opts.preview_horizon             = 2.0;
    ctrl.opts.use_full_fh_mpc_at_terminal = false;
    ctrl.opts.always_use_full_fh_mpc      = false;

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

static void run_timing_test() {
    setup_controller();

    // generate_dummy_data(ctrl.qp);

    #ifdef ARDUINO
        load_trajectory_data();
    #else
        if (!load_bin("x_traj.bin", x_traj, N_STEPS * NX) ||
            !load_bin("u_traj.bin", u_traj, N_STEPS * NU) ||
            !load_bin("r_traj.bin", r_traj, N_STEPS * NY)) {
            PRINT("Trajectory load failed — aborting.\n");
            return;
        }
    #endif

    PRINT("x[0] = [%.4f %.4f %.4f ... %.4f %.4f %.4f]\n",
        x_traj[0], x_traj[1], x_traj[2],
        x_traj[NX-3], x_traj[NX-2], x_traj[NX-1]);
    PRINT("u[0] = [%.4f %.4f %.4f %.4f %.4f %.4f]\n",
        u_traj[0], u_traj[1], u_traj[2],
        u_traj[3], u_traj[4], u_traj[5]);

    std::vector<Scalar> t_sdc, t_dare, t_ff, t_tot;
    t_sdc.reserve(N_STEPS); t_dare.reserve(N_STEPS);
    t_ff.reserve(N_STEPS);  t_tot.reserve(N_STEPS);

    int n_fallback = 0, n_fail = 0;
    Scalar worst_res = 0;

    PRINT("   k,   sdc,  dare,    ff, total, it,   res, ok, fb\n");

    for (int k = 1; k <= N_STEPS; ++k) {
        const Eigen::Map<const VecNX> xk(x_traj + (k - 1) * NX);
        const Eigen::Map<const VecNU> uk(u_traj + (k - 1) * NU);

        const uint32_t t0 = _sddre_micros();
        SDDRESolveInfo info;
        const VecNU u = ctrl.compute_u(k * ctrl.qp.Ts, xk, k, uk, info);
        const Scalar total = _sddre_elapsed_us(t0);

        t_sdc.push_back(info.time_sdc_discretize_us);
        t_dare.push_back(info.time_dare_us);
        t_ff.push_back(info.time_feedforward_us);
        t_tot.push_back(total);

        if (info.dare_info.used_sda_fallback) ++n_fallback;
        if (!info.dare_info.solve_success)    ++n_fail;
        if (info.dare_info.tol_achieved > worst_res)
            worst_res = info.dare_info.tol_achieved;

        // Print every step for the first 10, then every 20th
        if (k <= 10 || k % 20 == 0) {
            PRINT("%4d, %5.0f, %5.0f, %5.0f, %5.0f, %2d, %7.1e,  %d,  %d  ",
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

    PRINT("\n--- Timing (us) ---%8s%8s%8s%8s\n", "sdc", "dare", "ff", "total");
    PRINT("  median: %13.0f%8.0f%8.0f%8.0f\n",
          pct(t_sdc,50), pct(t_dare,50), pct(t_ff,50), pct(t_tot,50));
    PRINT("  p95:    %13.0f%8.0f%8.0f%8.0f\n",
          pct(t_sdc,95), pct(t_dare,95), pct(t_ff,95), pct(t_tot,95));
    PRINT("  max:    %13.0f%8.0f%8.0f%8.0f\n",
          pct(t_sdc,100), pct(t_dare,100), pct(t_ff,100), pct(t_tot,100));

    PRINT("\n--- Health ---\n");
    PRINT("  budget @ %.0f Hz : %.0f us\n", 1.0/ctrl.qp.Ts, 1e6*ctrl.qp.Ts);
    PRINT("  median util     : %.1f %%\n",
          100.0 * pct(t_tot,50) / (1e6*ctrl.qp.Ts));
    PRINT("  worst-case util : %.1f %%\n",
          100.0 * pct(t_tot,100) / (1e6*ctrl.qp.Ts));
    PRINT("  SDA fallbacks   : %d / %d\n", n_fallback, N_STEPS);
    PRINT("  DARE failures   : %d / %d\n", n_fail, N_STEPS);
    PRINT("  worst residual  : %.2e\n", worst_res);
}

#ifdef ARDUINO

void setup() {
    Serial.begin(115200);
    while (!Serial) {}
    delay(500);
    PRINT("\n=== SDDRE Timing Test (Teensy 4.1) ===\n");
    PRINT("NX=%d NU=%d NY=%d  N=%d\n\n", NX, NU, NY, N_STEPS);
    run_timing_test();
    PRINT("\nDone.\n");
}

void loop() {}

#else

int main() {
    std::printf("\n=== SDDRE Timing Test (desktop) ===\n");
    std::printf("NX=%d NU=%d NY=%d  N=%d\n\n", NX, NU, NY, N_STEPS);
    run_timing_test();
    std::printf("\nDone.\n");
    return 0;
}

#endif
