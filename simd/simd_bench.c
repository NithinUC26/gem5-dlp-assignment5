// SAXPY (y = a*x + y) and dot product: scalar vs SSE (128-bit), AVX2+FMA (256-bit), AVX-512 (512-bit).
#define _POSIX_C_SOURCE 200809L
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

__attribute__((optimize("no-tree-vectorize")))
void saxpy_scalar(float a, const float *x, float *y, size_t n) { for (size_t i = 0; i < n; i++) y[i] = a * x[i] + y[i]; }

__attribute__((target("sse2")))
void saxpy_sse(float a, const float *x, float *y, size_t n) {
  __m128 va = _mm_set1_ps(a); size_t i = 0;
  for (; i + 4 <= n; i += 4) _mm_storeu_ps(y + i, _mm_add_ps(_mm_mul_ps(va, _mm_loadu_ps(x + i)), _mm_loadu_ps(y + i)));
  for (; i < n; i++) y[i] = a * x[i] + y[i];
}
__attribute__((target("avx2,fma")))
void saxpy_avx2(float a, const float *x, float *y, size_t n) {
  __m256 va = _mm256_set1_ps(a); size_t i = 0;
  for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
  for (; i < n; i++) y[i] = a * x[i] + y[i];
}
__attribute__((target("avx512f")))
void saxpy_avx512(float a, const float *x, float *y, size_t n) {
  __m512 va = _mm512_set1_ps(a); size_t i = 0;
  for (; i + 16 <= n; i += 16) _mm512_storeu_ps(y + i, _mm512_fmadd_ps(va, _mm512_loadu_ps(x + i), _mm512_loadu_ps(y + i)));
  for (; i < n; i++) y[i] = a * x[i] + y[i];
}

__attribute__((optimize("no-tree-vectorize")))
float dot_scalar(const float *x, const float *y, size_t n) { float s = 0; for (size_t i = 0; i < n; i++) s += x[i] * y[i]; return s; }
__attribute__((optimize("no-tree-vectorize")))
float dot_scalar4(const float *x, const float *y, size_t n) {
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0; size_t i = 0;
  for (; i + 4 <= n; i += 4) { s0 += x[i]*y[i]; s1 += x[i+1]*y[i+1]; s2 += x[i+2]*y[i+2]; s3 += x[i+3]*y[i+3]; }
  float s = (s0 + s1) + (s2 + s3); for (; i < n; i++) s += x[i] * y[i]; return s;
}
__attribute__((target("sse2")))
float dot_sse(const float *x, const float *y, size_t n) {
  __m128 s0 = _mm_setzero_ps(), s1 = s0, s2 = s0, s3 = s0; size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_loadu_ps(x+i),    _mm_loadu_ps(y+i)));
    s1 = _mm_add_ps(s1, _mm_mul_ps(_mm_loadu_ps(x+i+4),  _mm_loadu_ps(y+i+4)));
    s2 = _mm_add_ps(s2, _mm_mul_ps(_mm_loadu_ps(x+i+8),  _mm_loadu_ps(y+i+8)));
    s3 = _mm_add_ps(s3, _mm_mul_ps(_mm_loadu_ps(x+i+12), _mm_loadu_ps(y+i+12)));
  }
  s0 = _mm_add_ps(_mm_add_ps(s0, s1), _mm_add_ps(s2, s3));
  float t[4]; _mm_storeu_ps(t, s0); float s = t[0] + t[1] + t[2] + t[3];
  for (; i < n; i++) s += x[i] * y[i]; return s;
}
__attribute__((target("avx2,fma")))
float dot_avx2(const float *x, const float *y, size_t n) {
  __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0; size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i),    _mm256_loadu_ps(y+i),    s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i+8),  _mm256_loadu_ps(y+i+8),  s1);
    s2 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i+16), _mm256_loadu_ps(y+i+16), s2);
    s3 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i+24), _mm256_loadu_ps(y+i+24), s3);
  }
  s0 = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
  float t[8]; _mm256_storeu_ps(t, s0); float s = 0; for (int k = 0; k < 8; k++) s += t[k];
  for (; i < n; i++) s += x[i] * y[i]; return s;
}
__attribute__((target("avx512f")))
float dot_avx512(const float *x, const float *y, size_t n) {
  __m512 s0 = _mm512_setzero_ps(), s1 = s0, s2 = s0, s3 = s0; size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    s0 = _mm512_fmadd_ps(_mm512_loadu_ps(x+i),    _mm512_loadu_ps(y+i),    s0);
    s1 = _mm512_fmadd_ps(_mm512_loadu_ps(x+i+16), _mm512_loadu_ps(y+i+16), s1);
    s2 = _mm512_fmadd_ps(_mm512_loadu_ps(x+i+32), _mm512_loadu_ps(y+i+32), s2);
    s3 = _mm512_fmadd_ps(_mm512_loadu_ps(x+i+48), _mm512_loadu_ps(y+i+48), s3);
  }
  s0 = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
  float s = _mm512_reduce_add_ps(s0);
  for (; i < n; i++) s += x[i] * y[i]; return s;
}

typedef void (*saxpy_fn)(float, const float *, float *, size_t);
typedef float (*dot_fn)(const float *, const float *, size_t);

static volatile float sink;
static double time_saxpy(saxpy_fn f, const float *x, float *y, size_t n, int reps) {
  double best = 1e30;
  for (int trial = 0; trial < 9; trial++) {
    double t0 = now();
    for (int r = 0; r < reps; r++) f(1.0000001f, x, y, n);
    double t = (now() - t0) / reps; if (t < best) best = t;
  }
  sink = y[0]; return best;
}
static double time_dot(dot_fn f, const float *x, const float *y, size_t n, int reps) {
  double best = 1e30; float acc = 0;
  for (int trial = 0; trial < 9; trial++) {
    double t0 = now();
    for (int r = 0; r < reps; r++) acc += f(x, y, n);
    double t = (now() - t0) / reps; if (t < best) best = t;
  }
  sink = acc; return best;
}

int main(void) {
  size_t sizes[] = {2048, 32768, 262144, 16777216};
  const char *where[] = {"L1 (16 KiB)", "L2 (256 KiB)", "L2/L3 (2 MiB)", "DRAM (128 MiB)"};
  const char *names[] = {"scalar", "SSE", "AVX2+FMA", "AVX-512"};
  const char *dnames[] = {"scalar (1 accumulator)", "scalar (4 accumulators)", "SSE", "AVX2+FMA", "AVX-512"};
  saxpy_fn sf[] = {saxpy_scalar, saxpy_sse, saxpy_avx2, saxpy_avx512};
  dot_fn df[] = {dot_scalar, dot_scalar4, dot_sse, dot_avx2, dot_avx512};
  printf("kernel,size_floats,working_set,variant,ns_per_element,speedup_vs_scalar,max_abs_err_or_rel_err\n");
  for (int si = 0; si < 4; si++) {
    size_t n = sizes[si];
    float *x = aligned_alloc(64, n * 4), *y = aligned_alloc(64, n * 4), *y0 = aligned_alloc(64, n * 4), *yr = aligned_alloc(64, n * 4);
    srand(1); for (size_t i = 0; i < n; i++) { x[i] = (float)rand() / RAND_MAX; y0[i] = (float)rand() / RAND_MAX; }
    int reps = (int)(2e8 / n); if (reps < 1) reps = 1;
    // SAXPY
    double base = 0;
    for (int v = 0; v < 4; v++) {
      memcpy(y, y0, n * 4); sf[v](2.5f, x, y, n);               // correctness on identical input
      if (v == 0) memcpy(yr, y, n * 4);
      double err = 0; for (size_t i = 0; i < n; i++) { double d = fabs((double)y[i] - yr[i]); if (d > err) err = d; }
      memcpy(y, y0, n * 4);
      double t = time_saxpy(sf[v], x, y, n, reps) / n * 1e9; if (v == 0) base = t;
      printf("saxpy,%zu,%s,%s,%.4f,%.2f,%.2e\n", n, where[si], names[v], t, base / t, err);
    }
    // dot product (error measured against a double-precision reference)
    double dbase = 0, ref = 0;
    for (size_t i = 0; i < n; i++) ref += (double)x[i] * (double)y0[i];
    for (int v = 0; v < 5; v++) {
      float r = df[v](x, y0, n);
      double rel = fabs((double)r - ref) / fabs(ref);
      double t = time_dot(df[v], x, y0, n, reps) / n * 1e9; if (v == 0) dbase = t;
      printf("dot,%zu,%s,%s,%.4f,%.2f,%.2e\n", n, where[si], dnames[v], t, dbase / t, rel);
    }
    free(x); free(y); free(y0); free(yr);
  }
  return 0;
}
