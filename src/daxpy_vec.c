#include <stdio.h>
#include <stdlib.h>
#include <riscv_vector.h>
#ifndef N
#define N 4096
#endif
#ifndef REPS
#define REPS 50
#endif
static double x[N], y[N];
int main(void) {
  for (int i = 0; i < N; i++) { x[i] = i * 0.5; y[i] = 1.0; }
  const double a = 2.0;
  for (int r = 0; r < REPS; r++) {
    size_t n = N, i = 0;
    while (n > 0) {
      size_t vl = __riscv_vsetvl_e64m1(n);
      vfloat64m1_t vx = __riscv_vle64_v_f64m1(&x[i], vl);
      vfloat64m1_t vy = __riscv_vle64_v_f64m1(&y[i], vl);
      vy = __riscv_vfmacc_vf_f64m1(vy, a, vx, vl);
      __riscv_vse64_v_f64m1(&y[i], vy, vl);
      i += vl; n -= vl;
    }
  }
  double s = 0; for (int i = 0; i < N; i++) s += y[i];
  printf("checksum %.1f\n", s);
  return 0;
}
