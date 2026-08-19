#pragma once
// ============================================================
//  platform/timing.h — platform timing
//
//    Teensy 4.1 — DWT cycle counter, 1 tick = 1 CPU cycle
//                 (1.67 ns at 600 MHz). Wraps every ~7.2 s;
//                 unsigned subtraction handles that correctly
//                 for any interval shorter than the period.
//    Desktop    — steady_clock in nanoseconds. Actual
//                 granularity is platform-dependent (~100 ns
//                 via QueryPerformanceCounter on Windows),
//                 still ~10x better than micros().
//
//  Reported times stay in microseconds, now fractional.
// ============================================================

#include "types/defs.h"
#include <stdint.h>

#ifdef ARDUINO
  // Cortex-M7 DWT cycle counter
  // - Registers addressed directly to avoid pulling Arduino.h into Eigen's include path
  //   - All four live in the ARMv7-M private peripheral bus at 0xE0000000
  // - DWT block base is 0xE0001000, the System Control Space (SCS) is 0xE000E000.
  #define SDOPT_DWT_CYCCNT (*(volatile uint32_t*)0xE0001004) // +0x004 cycle count, free-running, wraps at 2^32 (~7.16 s at 600 MHz)
  #define SDOPT_DWT_CTRL   (*(volatile uint32_t*)0xE0001000) // +0x000 DWT control; bit 0 = CYCCNTENA (Cycle Counter Enable)
  #define SDOPT_DWT_LAR    (*(volatile uint32_t*)0xE0001FB0) // +0xFB0 CoreSight Lock Access Register
  #define SDOPT_DEMCR      (*(volatile uint32_t*)0xE000EDFC) // SCS +0xDFC Debug Exception and Monitor Control

  // Default core clock - overridden by F_CPU if defined (so an overclocked build scales correctly)
  #ifndef SDOPT_CPU_HZ
    #ifdef F_CPU
      #define SDOPT_CPU_HZ (F_CPU)
    #else
      #define SDOPT_CPU_HZ 600000000u
    #endif
  #endif

  using sdopt_tick_t = uint32_t;   // one tick = one CPU cycle

  inline void sdopt_timing_init() {
      SDOPT_DEMCR    |= (1u << 24);        // TRCENA, Trace Enable (gates power and clock; without it the DWT registers read back as zero)
      SDOPT_DWT_LAR   = 0xC5ACCE55u;       // CoreSight unlock key (Cortex-M7)
      SDOPT_DWT_CYCCNT = 0;                // start from a known point
      SDOPT_DWT_CTRL |= 1u;                // cycle counter enable (CYCCNT advances one per CPU cycle while set, frozen while clear)
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
      // Nominal period of steady_clock, in microseconds. The real
      // granularity is usually coarser than this.
      using P = std::chrono::steady_clock::period;
      return Scalar(1e6) * Scalar(P::num) / Scalar(P::den);
  }
#endif

inline Scalar sdopt_elapsed_us(sdopt_tick_t start) {
    return sdopt_ticks_to_us(sdopt_ticks() - start);
}
