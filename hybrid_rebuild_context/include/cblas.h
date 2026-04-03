#pragma once

#include <stdexcept>

enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112 };
enum CBLAS_UPLO { CblasUpper = 121, CblasLower = 122 };

inline float cblas_sdot(const int n, const float *x, const int incx, const float *y, const int incy) {
  float acc = 0.0f;
  for (int i = 0; i < n; ++i) {
    acc += x[i * incx] * y[i * incy];
  }
  return acc;
}

inline void cblas_sgemm(const CBLAS_ORDER order, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
                        const int m, const int n, const int k, const float alpha, const float *a, const int lda,
                        const float *b, const int ldb, const float beta, float *c, const int ldc) {
  if (order != CblasRowMajor) {
    throw std::runtime_error("fallback cblas_sgemm only supports row-major order");
  }

  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float acc = 0.0f;
      for (int depth = 0; depth < k; ++depth) {
        const float a_value = (transa == CblasNoTrans) ? a[row * lda + depth] : a[depth * lda + row];
        const float b_value = (transb == CblasNoTrans) ? b[depth * ldb + col] : b[col * ldb + depth];
        acc += a_value * b_value;
      }
      c[row * ldc + col] = alpha * acc + beta * c[row * ldc + col];
    }
  }
}

inline void cblas_ssyrk(const CBLAS_ORDER order, const CBLAS_UPLO uplo, const CBLAS_TRANSPOSE trans, const int n,
                        const int k, const float alpha, const float *a, const int lda, const float beta, float *c,
                        const int ldc) {
  if (order != CblasRowMajor) {
    throw std::runtime_error("fallback cblas_ssyrk only supports row-major order");
  }
  if (uplo != CblasUpper) {
    throw std::runtime_error("fallback cblas_ssyrk only supports upper-triangular output");
  }

  for (int row = 0; row < n; ++row) {
    for (int col = row; col < n; ++col) {
      float acc = 0.0f;
      for (int depth = 0; depth < k; ++depth) {
        const float row_value = (trans == CblasNoTrans) ? a[row * lda + depth] : a[depth * lda + row];
        const float col_value = (trans == CblasNoTrans) ? a[col * lda + depth] : a[depth * lda + col];
        acc += row_value * col_value;
      }
      c[row * ldc + col] = alpha * acc + beta * c[row * ldc + col];
    }
  }
}
