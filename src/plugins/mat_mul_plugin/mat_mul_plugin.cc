#include "mat_mul_plugin.h"

#include <iostream>
#include <memory>

#define USE_CUBLAS 0
#define USE_CUBLASLT 1

#if USE_CUBLAS
#include <cublas_v2.h>
#elif USE_CUBLASLT
#include <cublasLt.h>
#else
#include "mat_mul.h"
#endif

namespace zcc {
namespace plugin {

namespace {
static const char* PLUGIN_VERSION{"1"};
static const char* PLUGIN_NAME{"MatMul"};
} // namespace

MatMulPluginDynamic::MatMulPluginDynamic(const std::string& name) : PluginDynamicBase(name) {
  std::cout << "MatMulPluginDynamic constructor." << std::endl;
}

MatMulPluginDynamic::MatMulPluginDynamic(const std::string name, const void* data, size_t length)
    : PluginDynamicBase(name) {
  std::cout << "MatMulPluginDynamic deserialize constructor." << std::endl;
}

nvinfer1::IPluginV2DynamicExt* MatMulPluginDynamic::clone() const PLUGIN_NOEXCEPT {
  std::cout << "MatMulPluginDynamic clone." << std::endl;
  MatMulPluginDynamic* plugin = new MatMulPluginDynamic(layer_name_);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::DimsExprs MatMulPluginDynamic::getOutputDimensions(int output_index,
                                                             const nvinfer1::DimsExprs* inputs,
                                                             int nb_inputs,
                                                             nvinfer1::IExprBuilder& expr_builder) PLUGIN_NOEXCEPT {
  std::cout << "MatMulPluginDynamic getOutputDimensions" << std::endl;
  assert(nb_inputs == 2);
  assert(output_index == 0);

  nvinfer1::DimsExprs output(inputs[0]);
  output.d[0] = inputs[0].d[0];
  output.d[1] = inputs[1].d[1];

  return output;
}

bool MatMulPluginDynamic::supportsFormatCombination(int pos,
                                                    const nvinfer1::PluginTensorDesc* inOut,
                                                    int nbInputs,
                                                    int nbOutputs) noexcept {
  std::cout << "MatMulPluginDynamic supportsFormatCombination." << std::endl;
  assert(0 <= pos && pos < 3);
  const auto* in = inOut;
  const auto* out = inOut + nbInputs;

  switch (pos) {
    case 0:
      std::cout << "pos 0, in[0].type: " << int(in[0].type) << ", in[0].format: " << int(in[0].format) << std::endl;
      return in[0].type == nvinfer1::DataType::kFLOAT && in[0].format == nvinfer1::TensorFormat::kLINEAR;
    case 1:
      std::cout << "pos 1, in[1].type: " << int(in[1].type) << ", in[1].format: " << int(in[1].format) << std::endl;
      return in[1].type == nvinfer1::DataType::kFLOAT && in[1].format == nvinfer1::TensorFormat::kLINEAR;
    case 2:
      return out[0].type == nvinfer1::DataType::kFLOAT && out[0].format == nvinfer1::TensorFormat::kLINEAR;
  }
  return false;
}

void MatMulPluginDynamic::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                                          int nb_inputs,
                                          const nvinfer1::DynamicPluginTensorDesc* out,
                                          int nb_outputs) PLUGIN_NOEXCEPT {
  std::cout << "MatMulPluginDynamic configurePlugin." << std::endl;
  assert(nb_inputs == 2);
  assert(nb_outputs == 1);
}

size_t MatMulPluginDynamic::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                                             int nb_inputs,
                                             const nvinfer1::PluginTensorDesc* outputs,
                                             int nb_outputs) const PLUGIN_NOEXCEPT {
  std::cout << "MatMulPluginDynamic getWorkspaceSize." << std::endl;
  return 0;
}

int MatMulPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                 const nvinfer1::PluginTensorDesc* outputDesc,
                                 const void* const* inputs,
                                 void* const* outputs,
                                 void* workspace,
                                 cudaStream_t stream) noexcept {
  std::cout << "MatMulPluginDynamic enqueue." << std::endl;
  const auto& input1_dims = inputDesc[0].dims;
  const auto& input2_dims = inputDesc[1].dims;

  const float* input1 = static_cast<const float*>(inputs[0]);
  const float* input2 = static_cast<const float*>(inputs[1]);
  float* output = static_cast<float*>(outputs[0]);

#if USE_CUBLAS // Use cublas to calc.
  const int m = input1_dims.d[0];
  const int k = input1_dims.d[1];
  const int n = input2_dims.d[1];
  const int lda = k;
  const int ldb = n;
  const int ldc = n;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  // std::cout << "m: " << m << ", k: " << k << ", n: " << n << ", lda: " << lda << ", ldb: " << ldb << ", ldc: " << ldc
  //           << ", alpha: " << alpha << ", beta: " << beta << std::endl;
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  cublasSetStream(cublas_handle, stream);

  // 由于cublasSgemm的参数是按照列优先的，host端传入的inp1和inp2是按行优先存储的，则传入cublasSgemm的参数相当于是inp1的转置和inp2的转置
  // 则最终计算结果需要计算出output ^ T，转为host端后则是output转置的转置，即output
  // 所以实际传入参数顺序应为inp2转置，inp1转置，output转置
  cublasSgemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, input2, CUDA_R_32F, ldb, input1, CUDA_R_32F,
                lda, &beta, output, CUDA_R_32F, ldc);
  cublasDestroy(cublas_handle);
#elif USE_CUBLASLT
  const int m = input1_dims.d[0];
  const int k = input1_dims.d[1];
  const int n = input2_dims.d[1];
  const int lda = k;
  const int ldb = n;
  const int ldc = n;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasLtHandle_t cublaslt_handle;
  cublasLtCreate(&cublaslt_handle);
  cublasLtMatmulDesc_t matmul_desc;
  cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasLtMatrixLayout_t input1_desc, input2_desc, output_desc;
  cublasLtMatrixLayoutCreate(&input1_desc, CUDA_R_32F, k, m, lda);
  cublasLtMatrixLayoutCreate(&input2_desc, CUDA_R_32F, n, k, ldb);
  cublasLtMatrixLayoutCreate(&output_desc, CUDA_R_32F, n, m, ldc);

  auto status = cublasLtMatmul(cublaslt_handle, matmul_desc, &alpha, input2, input2_desc, input1, input1_desc, &beta,
                               output, output_desc, output, output_desc, nullptr, workspace, 0, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "cublasLtMatmul failed, status: " << status << std::endl;
  }

  cublasLtMatrixLayoutDestroy(input1_desc);
  cublasLtMatrixLayoutDestroy(input2_desc);
  cublasLtMatrixLayoutDestroy(output_desc);
  cublasLtMatmulDescDestroy(matmul_desc);
  cublasLtDestroy(cublaslt_handle);
#else // Use my matmul kernel to calc.
  const int m = input1_dims.d[0];
  const int k = input1_dims.d[1];
  const int n = input2_dims.d[1];
  Matrix input1_matrix = {k, m, const_cast<float*>(input1)};
  Matrix input2_matrix = {n, k, const_cast<float*>(input2)};
  Matrix output_matrix = {n, m, output};
  MatInnerProdInGpu(input1_matrix, input2_matrix, output_matrix, stream);
#endif

#if 0
  // print input1 and input2
  float* input1_host = new float[m * k];
  float* input2_host = new float[k * n];
  cudaMemcpy(input1_host, input1, sizeof(float) * m * k, cudaMemcpyDeviceToHost);
  cudaMemcpy(input2_host, input2, sizeof(float) * k * n, cudaMemcpyDeviceToHost);
  std::cout << "input1_host: " << std::endl;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      std::cout << input1_host[i * k + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "input2_host: " << std::endl;
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < n; ++j) {
      std::cout << input2_host[i * n + j] << " ";
    }
    std::cout << std::endl;
  }

  // print output
  float* output_host = new float[m * n];
  cudaMemcpy(output_host, output, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
  std::cout << "output_host: " << std::endl;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      std::cout << output_host[i * n + j] << " ";
    }
    std::cout << std::endl;
  }
#endif
  return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType MatMulPluginDynamic::getOutputDataType(int index,
                                                          const nvinfer1::DataType* input_types,
                                                          int nb_inputs) const PLUGIN_NOEXCEPT {
  std::cout << "MatMulPluginDynamic getOutputDataType." << std::endl;
  assert(index == 0);
  assert(nb_inputs == 2);
  return nvinfer1::DataType::kFLOAT;
}

// IPluginV2 Methods
const char* MatMulPluginDynamic::getPluginType() const PLUGIN_NOEXCEPT {
  std::cout << "MatMulPluginDynamic getPluginType." << std::endl;
  return PLUGIN_NAME;
}

const char* MatMulPluginDynamic::getPluginVersion() const PLUGIN_NOEXCEPT {
  std::cout << "MatMulPluginDynamic getPluginVersion." << std::endl;
  return PLUGIN_VERSION;
}

int MatMulPluginDynamic::getNbOutputs() const PLUGIN_NOEXCEPT {
  std::cout << "MatMulPluginDynamic getNbOutputs." << std::endl;
  return 1;
}

size_t MatMulPluginDynamic::getSerializationSize() const PLUGIN_NOEXCEPT {
  std::cout << "MatMulPluginDynamic getSerializationSize." << std::endl;
  return 0;
}

void MatMulPluginDynamic::serialize(void* buffer) const noexcept {
  std::cout << "MatMulPluginDynamic serialize." << std::endl;
}

MatMulPluginDynamicCreator::MatMulPluginDynamicCreator() {
  plugin_attributes_.clear();
  fc_.nbFields = 0;
  fc_.fields = nullptr;
}

const char* MatMulPluginDynamicCreator::getPluginName() const PLUGIN_NOEXCEPT {
  std::cout << "MatMulPluginDynamicCreator getPluginName." << std::endl;
  return PLUGIN_NAME;
}

const char* MatMulPluginDynamicCreator::getPluginVersion() const PLUGIN_NOEXCEPT {
  std::cout << "MatMulPluginDynamicCreator getPluginVersion." << std::endl;
  return PLUGIN_VERSION;
}

nvinfer1::IPluginV2* MatMulPluginDynamicCreator::createPlugin(const char* name,
                                                              const nvinfer1::PluginFieldCollection* fc) noexcept {
  std::cout << "MatMulPluginDynamicCreator createPlugin." << std::endl;
  fc_ = *fc;
  auto* plugin = new MatMulPluginDynamic(name);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2* MatMulPluginDynamicCreator::deserializePlugin(const char* name,
                                                                   const void* serial_data,
                                                                   size_t serial_length) noexcept {
  std::cout << "MatMulPluginDynamicCreator deserializePlugin." << std::endl;
  auto* plugin = new MatMulPluginDynamic(name);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

} // namespace plugin
} // namespace zcc