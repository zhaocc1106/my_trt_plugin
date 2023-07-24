#pragma once

#include <cuda_runtime.h>

namespace zcc {
namespace plugin {
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
  int width;
  int height;
  float* elements;
} Matrix;

#define BLOCK_SIZE 16

/**
 * The cuda kernel function of matrix inner product.
 * @param a: Matrix a.
 * @param b: Matrix b.
 * @param c: The matrix to save ab.
 */
__global__ void MatInnerProdKernel(Matrix a, Matrix b, Matrix c);

/**
 * Matrix inner product in gpu. Matrix dimensions are assumed to be multiples of BLOCK_SIZE
 * @param a: Matrix a.
 * @param b: Matrix b.
 * @param c: The matrix to save ab.
 */
void MatInnerProdInGpu(const Matrix& a, const Matrix& b, Matrix& c, cudaStream_t& stream);

} // namespace plugin
} // namespace zcc