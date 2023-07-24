#include <cmath>
#include <cstdio>

#include "mat_mul.h"

namespace zcc {
namespace plugin {

/**
 * The cuda kernel function of matrix inner product.
 * @param a: Matrix a.
 * @param b: Matrix b.
 * @param c: The matrix to save ab.
 */
__global__ void MatInnerProdKernel(Matrix a, Matrix b, Matrix c) {
  /* Each thread computes one element of matrix c. */
  float c_element = 0;
  int row = blockIdx.y * blockDim.y + threadIdx.y; // The row num of element.
  int col = blockIdx.x * blockDim.x + threadIdx.x; // The col num of element.
  if (row >= c.height || col >= c.width) {
    return;
  }
  for (int i = 0; i < a.width; i++) {
    c_element += a.elements[row * a.width + i] * b.elements[i * b.width + col];
  }
  // printf("row: %d, col: %d, c_element: %f\n", row, col, c_element);
  c.elements[row * c.width + col] = c_element;
}

/**
 * Matrix inner product in gpu.
 * @param a: Matrix a.
 * @param b: Matrix b.
 * @param c: The matrix to save ab.
 */
void MatInnerProdInGpu(const Matrix& a, const Matrix& b, Matrix& c, cudaStream_t& stream) {
  /* Invoke kernel function. */
  dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dim_grid(std::ceil(float(c.width) / float(dim_block.x)), std::ceil(float(c.height) / float(dim_block.y)));
  // printf("dim_block: (%d, %d), dim_grid: (%d, %d)\n", dim_block.x, dim_block.y, dim_grid.x, dim_grid.y);
  MatInnerProdKernel<<<dim_grid, dim_block, 0, stream>>>(a, b, c);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("MatInnerProdInGpu failed: %s\n", cudaGetErrorString(err));
  }
}

} // namespace plugin
} // namespace zcc