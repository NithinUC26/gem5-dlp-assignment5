#include <stdio.h>
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
  for (int r = 0; r < REPS; r++)
    for (int i = 0; i < N; i++) y[i] = a * x[i] + y[i];
  double s = 0; for (int i = 0; i < N; i++) s += y[i];
  printf("checksum %.1f\n", s);
  return 0;
}
