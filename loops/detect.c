void independent(float *restrict a, const float *restrict b, const float *restrict c, int n) {
  for (int i = 0; i < n; i++) a[i] = b[i] * c[i] + 1.0f;
}
void carried(float *a, const float *b, int n) {
  for (int i = 1; i < n; i++) a[i] = a[i - 1] + b[i];
}
void maybe_alias(float *a, const float *b, int n) {
  for (int i = 0; i < n; i++) a[i] = b[i] * 2.0f;
}
