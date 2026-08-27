#pragma once
// ============================================================
//  platform/timing.h — platform timing
//
//    Teensy 4.1 — DWT cycle counter, 1 tick = 1 CPU cycle,
//                 ~1.67 ns at 600 MHz. Wraps every ~7.2 s;
//                 unsigned subtraction handles that correctly
//                 for any interval shorter than the period.
//    Desktop    — steady_clock in nanoseconds. Actual
//                 granularity is platform-dependent (~100 ns
//                 via QueryPerformanceCounter on Windows).
//
//  Reported times are in (fractional) microseconds.
// ============================================================

#include "types/defs.h"
#include <stdint.h>

#ifdef ARDUINO
  // Cortex-M7 DWT cycle counter.
  // Registers are addressed directly to avoid pulling Arduino.h into Eigen's include path.
  // DWT block base is 0xE0001000, the System Control Space (SCS) is 0xE000E000.
  #define SDOPT_DWT_CYCCNT (*(volatile uint32_t*)0xE0001004) // DWT +0x004 | Cycle Count
  #define SDOPT_DWT_CTRL   (*(volatile uint32_t*)0xE0001000) // DWT +0x000 | DWT Control
  #define SDOPT_DWT_LAR    (*(volatile uint32_t*)0xE0001FB0) // DWT +0xFB0 | CoreSight Lock Access Register
  #define SDOPT_DEMCR      (*(volatile uint32_t*)0xE000EDFC) // SCS +0xDFC | Debug Exception and Monitor Control Register

  // Default core clock - overridden by F_CPU if defined (so an overclocked build scales correctly)
  #ifndef SDOPT_CPU_HZ
    #ifdef F_CPU
      #define SDOPT_CPU_HZ (F_CPU)
    #else
      #define SDOPT_CPU_HZ 600000000u
    #endif
  #endif

  using sdopt_tick_t = uint32_t;

  inline void sdopt_timing_init() {
      SDOPT_DEMCR    |= (1u << 24);   // TRCENA – enables DWT hardware
      SDOPT_DWT_LAR   = 0xC5ACCE55u;  // CoreSight unlock key – permits writes to DWT registers
      SDOPT_DWT_CYCCNT = 0;           // zero the cycle counter
      SDOPT_DWT_CTRL |= 1u;           // CYCCNTENA – start the cycle counter
  }

  inline sdopt_tick_t sdopt_ticks() { return SDOPT_DWT_CYCCNT; }

  inline Scalar sdopt_ticks_to_us(sdopt_tick_t d) {
      return static_cast<Scalar>(d) *
             (Scalar(1e6) / Scalar(SDOPT_CPU_HZ));
  }

  inline Scalar sdopt_tick_us() { return sdopt_ticks_to_us(1); }

#else
  #include <chrono>

  using sdopt_tick_t = uint64_t;

  inline void sdopt_timing_init() {}

  inline sdopt_tick_t sdopt_ticks() {
      using namespace std::chrono;
      static const auto t0 = steady_clock::now();
      return static_cast<uint64_t>(
          duration_cast<nanoseconds>(steady_clock::now() - t0).count());
  }

  inline Scalar sdopt_ticks_to_us(sdopt_tick_t d) {
      return static_cast<Scalar>(d) * Scalar(1e-3);
  }

  inline Scalar sdopt_tick_us() {
      using P = std::chrono::steady_clock::period;
      return Scalar(1e6) * Scalar(P::num) / Scalar(P::den);
  }
#endif

inline Scalar sdopt_elapsed_us(sdopt_tick_t start) {
    return sdopt_ticks_to_us(sdopt_ticks() - start);
}
