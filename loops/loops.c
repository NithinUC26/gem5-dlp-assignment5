// Loop-level parallelism demo: independent loop, reduction, loop-carried dependence, loop interchange.
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
#define NV (1<<23)
#define M 512
#define TRIALS 5
static float a[NV], b[NV], c[NV];
static double refd[NV];
static float A[M][M], B[M][M], C[M][M];
static double dsum; static float off[64];
typedef void (*fn)(void);
static double best(fn f) { double bt = 1e30; for (int t = 0; t < TRIALS; t++) { double t0 = now(); f(); double d = now() - t0; if (d < bt) bt = d; } return bt; }

static void l1_serial(void) { for (int i = 0; i < NV; i++) a[i] = sqrtf(b[i]) * c[i] + sinf(b[i]); }
static void l1_par(void) {
  #pragma omp parallel for
  for (int i = 0; i < NV; i++) a[i] = sqrtf(b[i]) * c[i] + sinf(b[i]);
}
static void l2_serial(void) { double s = 0; for (int i = 0; i < NV; i++) s += (double)b[i] * c[i]; dsum = s; }
static void l2_par(void) {
  double s = 0;
  #pragma omp parallel for reduction(+:s)
  for (int i = 0; i < NV; i++) s += (double)b[i] * c[i];
  dsum = s;
}
static void l3_serial(void) { a[0] = b[0]; for (int i = 1; i < NV; i++) a[i] = a[i-1] + b[i]; }
static void l3_naive(void) {            // WRONG: ignores the loop-carried dependence
  a[0] = b[0];
  #pragma omp parallel for
  for (int i = 1; i < NV; i++) a[i] = a[i-1] + b[i];
}
static void l3_scan(void) {             // correct two-pass parallel prefix sum
  #pragma omp parallel
  {
    int nt = omp_get_num_threads(), id = omp_get_thread_num();
    int lo = (int)((long)NV * id / nt), hi = (int)((long)NV * (id + 1) / nt);
    float run = 0; for (int i = lo; i < hi; i++) { run += b[i]; a[i] = run; }
    off[id] = run;
    #pragma omp barrier
    float add = 0; for (int k = 0; k < id; k++) add += off[k];
    for (int i = lo; i < hi; i++) a[i] += add;
  }
}
static void l4_ijk(void) {
  memset(C, 0, sizeof C);
  for (int i = 0; i < M; i++) for (int j = 0; j < M; j++) { float s = 0; for (int k = 0; k < M; k++) s += A[i][k] * B[k][j]; C[i][j] = s; }
}
static void l4_ikj(void) {
  memset(C, 0, sizeof C);
  for (int i = 0; i < M; i++) for (int k = 0; k < M; k++) { float aik = A[i][k]; for (int j = 0; j < M; j++) C[i][j] += aik * B[k][j]; }
}
static void l4_ikj_par(void) {
  memset(C, 0, sizeof C);
  #pragma omp parallel for
  for (int i = 0; i < M; i++) for (int k = 0; k < M; k++) { float aik = A[i][k]; for (int j = 0; j < M; j++) C[i][j] += aik * B[k][j]; }
}

int main(void) {
  for (int i = 0; i < NV; i++) { b[i] = (i % 1000) * 0.001f; c[i] = 1.0f + (i % 7) * 0.01f; }
  printf("threads available: %d\n", omp_get_max_threads());
  double s, p;
  s = best(l1_serial); float r1 = a[12345]; p = best(l1_par);
  printf("L1 independent loop   serial %.4f s  parallel %.4f s  speedup %.2f\n", s, p, s / p);
  printf("   same result as serial: %s\n", a[12345] == r1 ? "yes" : "NO");
  s = best(l2_serial); double rs = dsum; p = best(l2_par);
  printf("L2 sum reduction        serial %.4f s  parallel %.4f s  speedup %.2f\n", s, p, s / p);
  printf("   relative difference in the sum: %.1e\n", fabs(dsum - rs) / rs);
  { double r = 0; for (int i = 0; i < NV; i++) { r += b[i]; refd[i] = r; } }   // double-precision reference
  s = best(l3_serial);
  double e_ser = 0, e_naive = 0, e_scan = 0; long bad_naive = 0, bad_scan = 0;
  for (int i = 1; i < NV; i++) { double d = fabs(a[i] - refd[i]) / fabs(refd[i]); if (d > e_ser) e_ser = d; }
  memset(a, 0, sizeof a); l3_naive();                                          // zeroed so stale values cannot hide the race
  for (int i = 1; i < NV; i++) { double d = fabs(a[i] - refd[i]) / fabs(refd[i]); if (d > e_naive) e_naive = d; if (d > 1e-3) bad_naive++; }
  memset(a, 0, sizeof a); l3_scan();
  for (int i = 1; i < NV; i++) { double d = fabs(a[i] - refd[i]) / fabs(refd[i]); if (d > e_scan) e_scan = d; if (d > 1e-3) bad_scan++; }
  p = best(l3_scan);
  printf("L3 prefix sum (carried) serial %.4f s  two-pass %.4f s  speedup %.2f\n", s, p, s / p);
  printf("   max relative error vs double-precision reference:\n");
  printf("     serial %.1e | two-pass %.1e (%ld wrong elements)\n", e_ser, e_scan, bad_scan);
  printf("     naive parallel %.1e (%ld of %d elements wrong)\n", e_naive, bad_naive, NV);
  for (int i = 0; i < M; i++) for (int j = 0; j < M; j++) { A[i][j] = (i + j) % 10 * 0.1f; B[i][j] = (i - j) % 10 * 0.1f; }
  double t1 = best(l4_ijk); float cij = C[7][9];
  double t2 = best(l4_ikj); float cik = C[7][9];
  double t3 = best(l4_ikj_par);
  printf("L4 matmul %dx%d         ijk %.4f s  ikj %.4f s (x%.2f)\n", M, M, t1, t2, t1 / t2);
  printf("   ikj + parallel %.4f s (x%.2f vs ijk)\n", t3, t1 / t3);
  printf("   C[7][9]: ijk %.3f  ikj %.3f\n", cij, cik);
  return 0;
}
