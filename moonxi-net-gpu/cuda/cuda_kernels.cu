// cuda_kernels.cu — Compiled by nvcc into shared library cuda_kernels.dll
// Contains CUDA kernels and cuBLAS wrappers called from the C stub.
// Functions are exported for dynamic loading via LoadLibrary/GetProcAddress.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

// Mark functions as shared library exports
#ifdef _WIN32
#define CUDA_EXPORT __declspec(dllexport)
#else
#define CUDA_EXPORT __attribute__((visibility("default")))
#endif

// ============================================================================
// CUDA initialization helper (cuDNN 9.21 requires active CUDA context)
// ============================================================================

static void ensure_cuda_init(void) {
    static int cuda_initialized = 0;
    if (!cuda_initialized) {
        cudaFree(0);
        cuda_initialized = 1;
    }
}

// ============================================================================
// Shape-keyed caches for cuDNN contexts (avoid repeated cudnnFind* overhead)
// ============================================================================

// Conv2d cache: key = "n:c:h:w:out_c:kh:kw:pad_h:pad_w:stride_h:stride_w:dilation_h:dilation_w"
static std::unordered_map<std::string, void*>* conv_cache = nullptr;

// BN cache: key = "n:c:h:w"
static std::unordered_map<std::string, void*>* bn_cache = nullptr;

// Pool cache: key = "n:c:h:w:kh:kw:pad_h:pad_w:stride_h:stride_w:pool_type"
static std::unordered_map<std::string, void*>* pool_cache = nullptr;

static std::string make_conv_key(int n, int c, int h, int w, int out_c, int kh, int kw,
                                  int pad_h, int pad_w, int stride_h, int stride_w,
                                  int dilation_h, int dilation_w) {
    char buf[256];
    snprintf(buf, sizeof(buf), "conv:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",
             n, c, h, w, out_c, kh, kw, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
    return std::string(buf);
}

static std::string make_bn_key(int n, int c, int h, int w) {
    char buf[128];
    snprintf(buf, sizeof(buf), "bn:%d:%d:%d:%d", n, c, h, w);
    return std::string(buf);
}

static std::string make_pool_key(int n, int c, int h, int w, int kh, int kw,
                                  int pad_h, int pad_w, int stride_h, int stride_w, int pool_type) {
    char buf[256];
    snprintf(buf, sizeof(buf), "pool:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",
             n, c, h, w, kh, kw, pad_h, pad_w, stride_h, stride_w, pool_type);
    return std::string(buf);
}

// ============================================================================
// Transpose and elementwise kernels
// ============================================================================

// NCHW -> NHWC transpose
// NCHW flat index: n*C*H*W + c*H*W + h*W + w
// NHWC flat index: n*H*W*C + h*W*C + w*C + c
__global__ void transpose_nchw_to_nhwc_kernel(
    const float* src, float* dst,
    int N, int C, int H, int W
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * C * H * W;
  if (idx < total) {
    int n = idx / (C * H * W);
    int rem = idx % (C * H * W);
    int c = rem / (H * W);
    rem = rem % (H * W);
    int h = rem / W;
    int w = rem % W;

    int nhwc_idx = n * H * W * C + h * W * C + w * C + c;
    dst[nhwc_idx] = src[idx];
  }
}

// NHWC -> NCHW transpose (inverse)
// NHWC flat index: n*H*W*C + h*W*C + w*C + c
// NCHW flat index: n*C*H*W + c*H*W + h*W + w
__global__ void transpose_nhwc_to_nchw_kernel(
    const float* src, float* dst,
    int N, int C, int H, int W
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * C * H * W;
  if (idx < total) {
    int n = idx / (C * H * W);
    int rem = idx % (C * H * W);
    int c = rem / (H * W);
    rem = rem % (H * W);
    int h = rem / W;
    int w = rem % W;

    int nhwc_idx = n * H * W * C + h * W * C + w * C + c;
    dst[idx] = src[nhwc_idx];
  }
}

// Bias add for NCHW tensor: data[n*C*HW + c*HW + hw] += bias[c]
__global__ void bias_add_nchw_kernel(
    float* data, const float* bias, int N, int C, int HW
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * C * HW;
  if (idx < total) {
    int rem = idx % (C * HW);
    int c = rem / HW;
    // data[n*C*HW + c*HW + hw] += bias[c]
    data[idx] += bias[c];
  }
}

// Simple elementwise add: out[i] = a[i] + b[i]
__global__ void elementwise_add_kernel(
    const float* a, const float* b, float* out, int n
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

// In-place elementwise add: dst[i] += src[i]
__global__ void elementwise_add_into_kernel(
    const float* src, float* dst, int n
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] += src[i];
  }
}

extern "C" {

// ============================================================================
// CUDA Runtime tests
// ============================================================================

CUDA_EXPORT int cuda_get_device_count() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    return -1;
  }
  return count;
}

CUDA_EXPORT int cuda_get_device_name(char* buf, int buf_len) {
  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, 0);
  if (err != cudaSuccess) {
    return -1;
  }
  snprintf(buf, buf_len, "%s", prop.name);
  return 0;
}

CUDA_EXPORT int cuda_get_compute_capability(int* major, int* minor) {
  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, 0);
  if (err != cudaSuccess) {
    return -1;
  }
  *major = prop.major;
  *minor = prop.minor;
  return 0;
}

CUDA_EXPORT void* cuda_alloc(int64_t size) {
  void* ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, (size_t)size);
  if (err != cudaSuccess) {
    return nullptr;
  }
  return ptr;
}

CUDA_EXPORT void cuda_free_device(void* ptr) {
  cudaFree(ptr);
}

CUDA_EXPORT int cuda_memset_zero(void* dst, int64_t count) {
  cudaError_t err = cudaMemset(dst, 0, count * sizeof(float));
  return (err == cudaSuccess) ? 0 : -1;
}

CUDA_EXPORT int cuda_copy_h2d(void* dst, const float* src, int64_t count) {
  cudaError_t err = cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyHostToDevice);
  return (err == cudaSuccess) ? 0 : -1;
}

// Raw byte copy host→device (for int arrays, pointer arrays, etc.)
CUDA_EXPORT int cuda_raw_h2d(void* dst, const void* src, int64_t bytes) {
  cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
  return (err == cudaSuccess) ? 0 : -1;
}

CUDA_EXPORT int cuda_copy_ptrs_h2d(void* dst, void** src, int32_t count) {
  size_t bytes = (size_t)count * sizeof(void*);
  cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
  return (err == cudaSuccess) ? 0 : -1;
}

CUDA_EXPORT int cuda_copy_d2h(float* dst, const void* src, int64_t count) {
  cudaError_t err = cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyDeviceToHost);
  return (err == cudaSuccess) ? 0 : -1;
}

// Copy from host array at offset to GPU
CUDA_EXPORT int cuda_copy_h2d_offset(void* dst, const void* src_base, int offset_elements, int count_elements) {
    const float* src = (const float*)src_base + offset_elements;
    cudaError_t err = cudaMemcpy(dst, src, count_elements * sizeof(float), cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Vector add kernel
// ============================================================================

__global__ void vec_add_kernel(const float* a, const float* b, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

CUDA_EXPORT int cuda_vec_add(const float* a_dev, const float* b_dev, float* out_dev, int n) {
  int block = 256;
  int grid = (n + block - 1) / block;
  vec_add_kernel<<<grid, block>>>(a_dev, b_dev, out_dev, n);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return -1;
  return 0;
}

// ============================================================================
// cuBLAS tests
// ============================================================================

CUDA_EXPORT void* cublas_create_handle() {
  cublasHandle_t handle;
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return nullptr;
  }
  return (void*)handle;
}

CUDA_EXPORT void cublas_destroy_handle(void* handle) {
  if (handle) {
    cublasDestroy((cublasHandle_t)handle);
  }
}

// y = alpha * A * x + beta * y  (sgemv/sgemv)
// A is m x n (row-major), x is n, y is m
CUDA_EXPORT int cublas_sgemv(void* handle,
                 int m, int n,
                 const float* alpha,
                 const float* d_A,   // device ptr, column-major (we transpose)
                 const float* d_x,   // device ptr
                 const float* beta,
                 float* d_y) {        // device ptr
  // cuBLAS assumes column-major. For row-major A (m x n),
  // treating it as column-major gives A^T. So we use CUBLAS_OP_T.
  cublasStatus_t status = cublasSgemv(
    (cublasHandle_t)handle,
    CUBLAS_OP_T,    // transpose because we store row-major
    n, m,           // rows/cols of the column-major view
    alpha,
    d_A, n,         // lda = n (column-major leading dimension)
    d_x, 1,
    beta,
    d_y, 1
  );
return (status == CUBLAS_STATUS_SUCCESS) ? 0 : (int)status;
}

// C = alpha * A * B + beta * C  (sgemm)
// A is m x k (row-major), B is k x n (row-major), C is m x n (row-major)
// cuBLAS column-major: C = alpha * B^T * A^T + beta * C^T  (swap A/B, use OP_N)
CUDA_EXPORT int cublas_sgemm(void* handle,
                   int m, int n, int k,
                   const float* alpha,
                   const float* d_A,   // device ptr, m x k row-major
                   const float* d_B,   // device ptr, k x n row-major
                   const float* beta,
                   float* d_C) {        // device ptr, m x n row-major
  cublasStatus_t status = cublasGemmEx(
    (cublasHandle_t)handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    n, m, k,
    alpha,
    d_B, CUDA_R_32F, n,
    d_A, CUDA_R_32F, k,
    beta,
    d_C, CUDA_R_32F, n,
    CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP
  );
  return (status == CUBLAS_STATUS_SUCCESS) ? 0 : (int)status;
}

// ============================================================================
// cuDNN Handle
// ============================================================================

CUDA_EXPORT void* cudnn_create_handle() {
  cudnnHandle_t handle;
  cudnnStatus_t status = cudnnCreate(&handle);
  if (status != CUDNN_STATUS_SUCCESS) {
    return nullptr;
  }
  return (void*)handle;
}

CUDA_EXPORT void cudnn_destroy_handle(void* handle) {
  if (handle) {
    cudnnDestroy((cudnnHandle_t)handle);
  }
}

// ============================================================================
// Conv2d Context
// ============================================================================

typedef struct {
  cudnnTensorDescriptor_t input_desc;
  cudnnFilterDescriptor_t filter_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnTensorDescriptor_t output_desc;
  cudnnConvolutionFwdAlgo_t fwd_algo;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
  void* workspace;
  size_t workspace_size;
  int out_n, out_c, out_h, out_w;
} ConvContext;

static int calc_output_size(int input, int filter, int pad, int stride, int dilation) {
  return ((input + 2 * pad - dilation * (filter - 1) - 1) / stride) + 1;
}

CUDA_EXPORT void* conv2d_create(
    int n, int c, int h, int w,
    int out_c, int kh, int kw,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w
) {
  ensure_cuda_init();

  // Check cache first
  if (!conv_cache) conv_cache = new std::unordered_map<std::string, void*>();
  std::string key = make_conv_key(n, c, h, w, out_c, kh, kw, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
  auto it = conv_cache->find(key);
  if (it != conv_cache->end()) {
    return it->second;
  }

  ConvContext* ctx = (ConvContext*)malloc(sizeof(ConvContext));
  if (!ctx) return nullptr;

  memset(ctx, 0, sizeof(ConvContext));

  // Compute output dimensions
  ctx->out_n = n;
  ctx->out_c = out_c;
  ctx->out_h = calc_output_size(h, kh, pad_h, stride_h, dilation_h);
  ctx->out_w = calc_output_size(w, kw, pad_w, stride_w, dilation_w);

  // Create descriptors
  cudnnCreateTensorDescriptor(&ctx->input_desc);
  cudnnCreateFilterDescriptor(&ctx->filter_desc);
  cudnnCreateConvolutionDescriptor(&ctx->conv_desc);
  cudnnCreateTensorDescriptor(&ctx->output_desc);

  // Set input descriptor: NCHW
  cudnnSetTensor4dDescriptor(ctx->input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

  // Set filter descriptor: NCHW (out_c, c, kh, kw)
  cudnnSetFilter4dDescriptor(ctx->filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_c, c, kh, kw);

  // Set convolution descriptor
  cudnnSetConvolution2dDescriptor(
    ctx->conv_desc,
    pad_h, pad_w,
    stride_h, stride_w,
    dilation_h, dilation_w,
    CUDNN_CROSS_CORRELATION,
    CUDNN_DATA_FLOAT
  );

  // Enable Tensor Core acceleration for f32 convolutions
  cudnnSetConvolutionMathType(ctx->conv_desc, CUDNN_TENSOR_OP_MATH);

  // Set output descriptor: NCHW
  cudnnSetTensor4dDescriptor(ctx->output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                            ctx->out_n, ctx->out_c, ctx->out_h, ctx->out_w);

  // Create a real cuDNN handle for algorithm selection
  cudnnHandle_t find_handle;
  cudnnStatus_t st = cudnnCreate(&find_handle);
  if (st != CUDNN_STATUS_SUCCESS) {
    // Cannot create handle — use default algorithms
    ctx->fwd_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    ctx->bwd_data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    ctx->bwd_filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
    ctx->workspace_size = 0;
    ctx->workspace = nullptr;
    // Store in cache before returning
    (*conv_cache)[key] = (void*)ctx;
    return (void*)ctx;
  }

  int returned_algo_count = 0;
  cudnnConvolutionFwdAlgoPerf_t fwd_results[4];
  st = cudnnFindConvolutionForwardAlgorithm(
    find_handle, ctx->input_desc, ctx->filter_desc,
    ctx->conv_desc, ctx->output_desc,
    4, &returned_algo_count, fwd_results
  );
  ctx->fwd_algo = (st == CUDNN_STATUS_SUCCESS && returned_algo_count > 0)
    ? fwd_results[0].algo : CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

  // Find backward data algorithm
  cudnnConvolutionBwdDataAlgoPerf_t bwd_data_results[4];
  st = cudnnFindConvolutionBackwardDataAlgorithm(
    find_handle, ctx->filter_desc, ctx->output_desc,
    ctx->conv_desc, ctx->input_desc,
    4, &returned_algo_count, bwd_data_results
  );
  ctx->bwd_data_algo = (st == CUDNN_STATUS_SUCCESS && returned_algo_count > 0)
    ? bwd_data_results[0].algo : CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;

  // Find backward filter algorithm
  cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_results[4];
  st = cudnnFindConvolutionBackwardFilterAlgorithm(
    find_handle, ctx->input_desc, ctx->output_desc,
    ctx->conv_desc, ctx->filter_desc,
    4, &returned_algo_count, bwd_filter_results
  );
  ctx->bwd_filter_algo = (st == CUDNN_STATUS_SUCCESS && returned_algo_count > 0)
    ? bwd_filter_results[0].algo : CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;

  // Allocate workspace
  size_t fwd_ws_size = 0, bwd_data_ws_size = 0, bwd_filter_ws_size = 0;
  cudnnGetConvolutionForwardWorkspaceSize(find_handle, ctx->input_desc, ctx->filter_desc,
                                           ctx->conv_desc, ctx->output_desc, ctx->fwd_algo,
                                           &fwd_ws_size);
  cudnnGetConvolutionBackwardDataWorkspaceSize(find_handle, ctx->filter_desc, ctx->output_desc,
                                                  ctx->conv_desc, ctx->input_desc, ctx->bwd_data_algo,
                                                  &bwd_data_ws_size);
  cudnnGetConvolutionBackwardFilterWorkspaceSize(find_handle, ctx->input_desc, ctx->output_desc,
                                                    ctx->conv_desc, ctx->filter_desc, ctx->bwd_filter_algo,
                                                    &bwd_filter_ws_size);

  cudnnDestroy(find_handle);

  ctx->workspace_size = (fwd_ws_size > bwd_data_ws_size) ? fwd_ws_size : bwd_data_ws_size;
  if (bwd_filter_ws_size > ctx->workspace_size) ctx->workspace_size = bwd_filter_ws_size;

  if (ctx->workspace_size > 0) {
    cudaMalloc(&ctx->workspace, ctx->workspace_size);
  }

  // Store in cache before returning
  (*conv_cache)[key] = (void*)ctx;
  return (void*)ctx;
}

CUDA_EXPORT void conv2d_destroy(void* ctx_ptr) {
  if (!ctx_ptr) return;
  ConvContext* ctx = (ConvContext*)ctx_ptr;
  if (ctx->workspace) cudaFree(ctx->workspace);
  cudnnDestroyTensorDescriptor(ctx->input_desc);
  cudnnDestroyFilterDescriptor(ctx->filter_desc);
  cudnnDestroyConvolutionDescriptor(ctx->conv_desc);
  cudnnDestroyTensorDescriptor(ctx->output_desc);
  free(ctx);
}

CUDA_EXPORT int conv2d_forward(void* ctx_ptr, void* cudnn_handle, void* input, void* weight, void* bias, void* output) {
  ConvContext* ctx = (ConvContext*)ctx_ptr;
  cudnnHandle_t handle = (cudnnHandle_t)cudnn_handle;
  float alpha = 1.0f, beta = 0.0f;

  cudnnConvolutionForward(
    handle, &alpha,
    ctx->input_desc, input,
    ctx->filter_desc, weight,
    ctx->conv_desc, ctx->fwd_algo,
    ctx->workspace, ctx->workspace_size,
    &beta,
    ctx->output_desc, output
  );

  if (bias) {
    int block = 256;
    int total = ctx->out_n * ctx->out_c * ctx->out_h * ctx->out_w;
    int grid = (total + block - 1) / block;
    bias_add_nchw_kernel<<<grid, block>>>((float*)output, (const float*)bias,
                                           ctx->out_n, ctx->out_c, ctx->out_h * ctx->out_w);
  }

  return CUDNN_STATUS_SUCCESS;
}

CUDA_EXPORT int conv2d_forward_bias_relu(void* ctx_ptr, void* cudnn_handle, void* input, void* weight, void* bias, void* output) {
  ConvContext* ctx = (ConvContext*)ctx_ptr;
  cudnnHandle_t handle = (cudnnHandle_t)cudnn_handle;
  float alpha1 = 1.0f, alpha2 = 0.0f;

  cudnnTensorDescriptor_t bias_desc;
  cudnnCreateTensorDescriptor(&bias_desc);
  cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, ctx->out_c, 1, 1);

  cudnnActivationDescriptor_t act_desc;
  cudnnCreateActivationDescriptor(&act_desc);
  cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0);

  cudnnStatus_t st = cudnnConvolutionBiasActivationForward(
    handle,
    &alpha1,
    ctx->input_desc, input,
    ctx->filter_desc, weight,
    ctx->conv_desc, ctx->fwd_algo,
    ctx->workspace, ctx->workspace_size,
    &alpha2,
    ctx->output_desc, output,
    bias_desc, bias,
    act_desc,
    ctx->output_desc, output
  );

  cudnnDestroyActivationDescriptor(act_desc);
  cudnnDestroyTensorDescriptor(bias_desc);
  return (int)st;
}

CUDA_EXPORT int conv2d_backward_data(void* ctx_ptr, void* cudnn_handle, void* grad_output, void* weight, void* grad_input) {
  ConvContext* ctx = (ConvContext*)ctx_ptr;
  cudnnHandle_t handle = (cudnnHandle_t)cudnn_handle;
  float alpha = 1.0f, beta = 0.0f;

  cudnnConvolutionBackwardData(
    handle, &alpha,
    ctx->filter_desc, weight,
    ctx->output_desc, grad_output,
    ctx->conv_desc, ctx->bwd_data_algo,
    ctx->workspace, ctx->workspace_size,
    &beta,
    ctx->input_desc, grad_input
  );

  return CUDNN_STATUS_SUCCESS;
}

CUDA_EXPORT int conv2d_backward_filter(void* ctx_ptr, void* cudnn_handle, void* input, void* grad_output, void* grad_weight) {
  ConvContext* ctx = (ConvContext*)ctx_ptr;
  cudnnHandle_t handle = (cudnnHandle_t)cudnn_handle;
  float alpha = 1.0f, beta = 0.0f;

  cudnnConvolutionBackwardFilter(
    handle, &alpha,
    ctx->input_desc, input,
    ctx->output_desc, grad_output,
    ctx->conv_desc, ctx->bwd_filter_algo,
    ctx->workspace, ctx->workspace_size,
    &beta,
    ctx->filter_desc, grad_weight
  );

  return CUDNN_STATUS_SUCCESS;
}

// Bias backward: db[c] = sum_{n,h,w} dy[n,c,h,w]
// Uses shared-memory reduction per channel, avoids cuDNN descriptor overhead.
__global__ void bias_bwd_kernel(const float* __restrict__ grad_output,
                                float* __restrict__ grad_bias,
                                int batch, int channels, int hw) {
  // One block per channel
  int c = blockIdx.x;
  if (c >= channels) return;

  int total = batch * hw;  // number of elements per channel
  float sum = 0.0f;
  // Stride loop over (batch × hw) elements for this channel
  for (int i = threadIdx.x; i < total; i += blockDim.x) {
    int n = i / hw;
    int s = i % hw;
    sum += grad_output[n * channels * hw + c * hw + s];
  }

  // Warp-level reduce + shared memory
  __shared__ float sbuf[256];
  sbuf[threadIdx.x] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sbuf[threadIdx.x] += sbuf[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    grad_bias[c] = sbuf[0];
  }
}

CUDA_EXPORT int conv2d_backward_bias(void* cudnn_handle, void* grad_output, void* grad_bias, int batch, int out_c, int out_h, int out_w) {
  int hw = out_h * out_w;
  int block = 256;
  bias_bwd_kernel<<<out_c, block>>>((const float*)grad_output, (float*)grad_bias, batch, out_c, hw);
  return 0;
}

// ============================================================================
// BatchNorm Context
// ============================================================================

typedef struct {
  cudnnTensorDescriptor_t input_desc;
  cudnnTensorDescriptor_t param_desc;  // (1, C, 1, 1) for BN scale/bias/mean/var
  int n, c, h, w;
} BNContext;

CUDA_EXPORT void* batchnorm_create(int n, int c, int h, int w) {
  ensure_cuda_init();

  // Check cache first
  if (!bn_cache) bn_cache = new std::unordered_map<std::string, void*>();
  std::string key = make_bn_key(n, c, h, w);
  auto it = bn_cache->find(key);
  if (it != bn_cache->end()) {
    return it->second;
  }

  BNContext* ctx = (BNContext*)malloc(sizeof(BNContext));
  if (!ctx) return nullptr;

  ctx->n = n;
  ctx->c = c;
  ctx->h = h;
  ctx->w = w;

  cudnnCreateTensorDescriptor(&ctx->input_desc);
  cudnnSetTensor4dDescriptor(ctx->input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

  // BN param descriptor must be (1, C, 1, 1) per cuDNN spec
  cudnnCreateTensorDescriptor(&ctx->param_desc);
  cudnnSetTensor4dDescriptor(ctx->param_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, c, 1, 1);

  // Store in cache before returning
  (*bn_cache)[key] = (void*)ctx;
  return (void*)ctx;
}

CUDA_EXPORT void batchnorm_destroy(void* ctx_ptr) {
  if (!ctx_ptr) return;
  BNContext* ctx = (BNContext*)ctx_ptr;
  cudnnDestroyTensorDescriptor(ctx->input_desc);
  cudnnDestroyTensorDescriptor(ctx->param_desc);
  free(ctx);
}

CUDA_EXPORT int batchnorm_forward(void* ctx_ptr, void* cudnn_handle,
                                   void* input, void* output,
                                   void* bn_scale, void* bn_bias,
                                   void* bn_running_mean, void* bn_running_var,
                                   void* bn_save_mean, void* bn_save_inv_var,
                                   float momentum, float epsilon, int is_training) {
  BNContext* ctx = (BNContext*)ctx_ptr;
  cudnnHandle_t handle = (cudnnHandle_t)cudnn_handle;

  float one = 1.0f, zero = 0.0f;
  float inv_momentum = 1.0f - momentum;

  if (is_training) {
    cudnnBatchNormalizationForwardTraining(
      handle,
      CUDNN_BATCHNORM_SPATIAL,
      &one, &zero,
      ctx->input_desc, input,
      ctx->input_desc, output,
      ctx->param_desc, bn_scale, bn_bias,
      momentum,
      bn_running_mean, bn_running_var,
      epsilon,
      bn_save_mean, bn_save_inv_var
    );
  } else {
    cudnnBatchNormalizationForwardInference(
      handle, CUDNN_BATCHNORM_SPATIAL,
      &one, &zero,
      ctx->input_desc, input,
      ctx->input_desc, output,
      ctx->param_desc, bn_scale, bn_bias,
      bn_running_mean, bn_running_var,
      epsilon
    );
  }

  return CUDNN_STATUS_SUCCESS;
}

// Custom BN inference kernel — bypasses cuDNN for deterministic results.
// Formula: output = gamma * (input - running_mean) / sqrt(running_var + eps) + beta
__global__ void batchnorm_inference_kernel(
    const float* input, float* output,
    const float* gamma, const float* beta,
    const float* running_mean, const float* running_var,
    float eps, int N, int C, int HW
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * C * HW;
  if (idx >= total) return;
  int c = (idx / HW) % C;
  float inv_std = 1.0f / sqrtf(running_var[c] + eps);
  output[idx] = gamma[c] * (input[idx] - running_mean[c]) * inv_std + beta[c];
}

CUDA_EXPORT int cuda_batchnorm_inference(
    const float* input, float* output,
    const float* gamma, const float* beta,
    const float* running_mean, const float* running_var,
    float eps, int N, int C, int HW
) {
  int total = N * C * HW;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  batchnorm_inference_kernel<<<blocks, threads>>>(
      input, output, gamma, beta, running_mean, running_var, eps, N, C, HW);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return -1;
  return 0;
}

CUDA_EXPORT int batchnorm_backward(void* ctx_ptr, void* cudnn_handle,
                                    void* input, void* grad_output, void* grad_input,
                                    void* bn_scale, void* grad_bn_scale, void* grad_bn_bias,
                                    void* bn_save_mean, void* bn_save_inv_var,
                                    float epsilon) {
  BNContext* ctx = (BNContext*)ctx_ptr;
  cudnnHandle_t handle = (cudnnHandle_t)cudnn_handle;

  float one = 1.0f, zero = 0.0f;

  cudnnBatchNormalizationBackward(
    handle, CUDNN_BATCHNORM_SPATIAL,
    &one, &zero,          // alphaDataDiff, betaDataDiff
    &one, &zero,          // alphaParamDiff, betaParamDiff
    ctx->input_desc, input,       // xDesc, x
    ctx->input_desc, grad_output, // dyDesc, dy
    ctx->input_desc, grad_input,  // dxDesc, dx
    ctx->param_desc, bn_scale,    // paramDesc, bnScale
    grad_bn_scale,                // dBnScaleResult
    grad_bn_bias,                 // dBnBiasResult
    epsilon,
    bn_save_mean, bn_save_inv_var // savedMean, savedInvVariance
  );

  return CUDNN_STATUS_SUCCESS;
}

// ============================================================================
// ReLU via cuDNN activation
// ============================================================================

// Cached activation descriptor — avoids create/destroy per call
static cudnnActivationDescriptor_t get_relu_act_desc() {
  static cudnnActivationDescriptor_t desc = nullptr;
  if (!desc) {
    cudnnCreateActivationDescriptor(&desc);
    cudnnSetActivationDescriptor(desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);
  }
  return desc;
}

CUDA_EXPORT int cudnn_relu_forward(void* cudnn_handle, int64_t n_elements, void* input, void* output) {
  ensure_cuda_init();
  cudnnHandle_t handle = (cudnnHandle_t)cudnn_handle;

  cudnnActivationDescriptor_t act_desc = get_relu_act_desc();

  cudnnTensorDescriptor_t tensor_desc;
  cudnnCreateTensorDescriptor(&tensor_desc);
  cudnnSetTensor4dDescriptor(tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, (int)n_elements, 1);

  float alpha = 1.0f, beta = 0.0f;
  cudnnActivationForward(handle, act_desc, &alpha, tensor_desc, input, &beta, tensor_desc, output);

  cudnnDestroyTensorDescriptor(tensor_desc);
  return CUDNN_STATUS_SUCCESS;
}

CUDA_EXPORT int cudnn_relu_backward(void* cudnn_handle, int64_t n_elements, void* input, void* grad_output, void* grad_input) {
  ensure_cuda_init();
  cudnnHandle_t handle = (cudnnHandle_t)cudnn_handle;

  cudnnActivationDescriptor_t act_desc = get_relu_act_desc();

  cudnnTensorDescriptor_t tensor_desc;
  cudnnCreateTensorDescriptor(&tensor_desc);
  cudnnSetTensor4dDescriptor(tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, (int)n_elements, 1);

  float alpha = 1.0f, beta = 0.0f;
  cudnnActivationBackward(handle, act_desc, &alpha,
                          tensor_desc, input, // input for relu mask
                          tensor_desc, grad_output,
                          tensor_desc, input, // original input
                          &beta, tensor_desc, grad_input);

  cudnnDestroyTensorDescriptor(tensor_desc);
  return CUDNN_STATUS_SUCCESS;
}

// ============================================================================
// Pooling
// ============================================================================

typedef struct {
  cudnnTensorDescriptor_t input_desc;
  cudnnTensorDescriptor_t output_desc;
  cudnnPoolingDescriptor_t pool_desc;
  int n, c, out_h, out_w;
} PoolContext;

CUDA_EXPORT void* pool2d_create(int n, int c, int h, int w, int kh, int kw,
                                  int pad_h, int pad_w, int stride_h, int stride_w, int pool_type) {
  ensure_cuda_init();

  // Check cache first
  if (!pool_cache) pool_cache = new std::unordered_map<std::string, void*>();
  std::string key = make_pool_key(n, c, h, w, kh, kw, pad_h, pad_w, stride_h, stride_w, pool_type);
  auto it = pool_cache->find(key);
  if (it != pool_cache->end()) {
    return it->second;
  }

  PoolContext* ctx = (PoolContext*)malloc(sizeof(PoolContext));
  if (!ctx) return nullptr;

  ctx->n = n;
  ctx->c = c;

  cudnnCreateTensorDescriptor(&ctx->input_desc);
  cudnnCreateTensorDescriptor(&ctx->output_desc);
  cudnnCreatePoolingDescriptor(&ctx->pool_desc);

  cudnnSetTensor4dDescriptor(ctx->input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

  cudnnPoolingMode_t mode = (pool_type == 0) ? CUDNN_POOLING_MAX : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  cudnnSetPooling2dDescriptor(ctx->pool_desc, mode, CUDNN_NOT_PROPAGATE_NAN, kh, kw, pad_h, pad_w, stride_h, stride_w);

  // Compute output size
  ctx->out_h = ((h + 2 * pad_h - kh) / stride_h) + 1;
  ctx->out_w = ((w + 2 * pad_w - kw) / stride_w) + 1;

  cudnnSetTensor4dDescriptor(ctx->output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, ctx->out_h, ctx->out_w);

  // Store in cache before returning
  (*pool_cache)[key] = (void*)ctx;
  return (void*)ctx;
}

CUDA_EXPORT void pool2d_destroy(void* ctx_ptr) {
  if (!ctx_ptr) return;
  PoolContext* ctx = (PoolContext*)ctx_ptr;
  cudnnDestroyTensorDescriptor(ctx->input_desc);
  cudnnDestroyTensorDescriptor(ctx->output_desc);
  cudnnDestroyPoolingDescriptor(ctx->pool_desc);
  free(ctx);
}

CUDA_EXPORT int pool2d_forward(void* ctx_ptr, void* cudnn_handle, void* input, void* output) {
  PoolContext* ctx = (PoolContext*)ctx_ptr;
  cudnnHandle_t handle = (cudnnHandle_t)cudnn_handle;

  float alpha = 1.0f, beta = 0.0f;
  cudnnPoolingForward(handle, ctx->pool_desc, &alpha, ctx->input_desc, input, &beta, ctx->output_desc, output);

  return CUDNN_STATUS_SUCCESS;
}

CUDA_EXPORT int pool2d_backward(void* ctx_ptr, void* cudnn_handle, void* input, void* grad_output, void* grad_input) {
  PoolContext* ctx = (PoolContext*)ctx_ptr;
  cudnnHandle_t handle = (cudnnHandle_t)cudnn_handle;

  float alpha = 1.0f, beta = 0.0f;
  // Allocate a temporary output tensor for the backward pass reference
  // pool2d_backward needs the forward output as `y` but we only have input
  // We pass input as a placeholder — cuDNN only needs it for avg pool size info
  cudnnPoolingBackward(handle, ctx->pool_desc, &alpha,
                       ctx->output_desc, grad_output,
                       ctx->output_desc, grad_output,
                       ctx->input_desc, input,
                       &beta, ctx->input_desc, grad_input);

  return CUDNN_STATUS_SUCCESS;
}

// ============================================================================
// GPU Memory Pool
// ============================================================================

#define GPU_ALIGNMENT 256

typedef struct {
  void* base_ptr;
  int64_t offset;
  int64_t capacity;
  int64_t peak;
} Pool;

static int64_t align_up(int64_t size, int64_t alignment) {
  int64_t rem = size % alignment;
  if (rem == 0) return size;
  return size + (alignment - rem);
}

CUDA_EXPORT void* pool_create(int64_t capacity) {
  Pool* pool = (Pool*)malloc(sizeof(Pool));
  if (!pool) return (void*)0;

  cudaError_t err = cudaMalloc(&pool->base_ptr, (size_t)capacity);
  if (err != cudaSuccess) {
    free(pool);
    return (void*)0;
  }

  pool->offset = 0;
  pool->capacity = capacity;
  pool->peak = 0;
  return (void*)pool;
}

CUDA_EXPORT void pool_destroy(void* pool_ptr) {
  if (!pool_ptr) return;
  Pool* pool = (Pool*)pool_ptr;
  if (pool->base_ptr) {
    cudaFree(pool->base_ptr);
    pool->base_ptr = (void*)0;
  }
  free(pool);
}

CUDA_EXPORT void* pool_alloc(void* pool_ptr, int64_t size, int64_t alignment) {
  if (!pool_ptr || size <= 0) return (void*)0;
  Pool* pool = (Pool*)pool_ptr;

  int64_t aligned_size = align_up(size, alignment > GPU_ALIGNMENT ? alignment : GPU_ALIGNMENT);
  if (pool->offset + aligned_size > pool->capacity) {
    return (void*)0; // out of memory
  }

  void* result = (char*)pool->base_ptr + pool->offset;
  pool->offset += aligned_size;

  if (pool->offset > pool->peak) {
    pool->peak = pool->offset;
  }

  return result;
}

CUDA_EXPORT void pool_reset(void* pool_ptr) {
  if (!pool_ptr) return;
  Pool* pool = (Pool*)pool_ptr;
  pool->offset = 0;
}

CUDA_EXPORT int64_t pool_used(void* pool_ptr) {
  if (!pool_ptr) return 0;
  Pool* pool = (Pool*)pool_ptr;
  return pool->offset;
}

CUDA_EXPORT int64_t pool_peak(void* pool_ptr) {
  if (!pool_ptr) return 0;
  Pool* pool = (Pool*)pool_ptr;
  return pool->peak;
}

CUDA_EXPORT int64_t pool_capacity(void* pool_ptr) {
  if (!pool_ptr) return 0;
  Pool* pool = (Pool*)pool_ptr;
  return pool->capacity;
}

// ============================================================================
// Transpose and elementwise kernel wrappers
// ============================================================================

CUDA_EXPORT int cuda_transpose_nchw_to_nhwc(
    float* src, float* dst, int N, int C, int H, int W
) {
  int block = 256;
  int total = N * C * H * W;
  int grid = (total + block - 1) / block;
  transpose_nchw_to_nhwc_kernel<<<grid, block>>>(src, dst, N, C, H, W);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return -1;
  return 0;
}

CUDA_EXPORT int cuda_transpose_nhwc_to_nchw(
    float* src, float* dst, int N, int C, int H, int W
) {
  int block = 256;
  int total = N * C * H * W;
  int grid = (total + block - 1) / block;
  transpose_nhwc_to_nchw_kernel<<<grid, block>>>(src, dst, N, C, H, W);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return -1;
  return 0;
}

CUDA_EXPORT int cuda_bias_add_nchw(
    float* data, float* bias, int N, int C, int HW
) {
  int block = 256;
  int total = N * C * HW;
  int grid = (total + block - 1) / block;
  bias_add_nchw_kernel<<<grid, block>>>(data, bias, N, C, HW);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return -1;
  return 0;
}

CUDA_EXPORT int cuda_elementwise_add(
    float* a, float* b, float* out, int n
) {
  int block = 256;
  int grid = (n + block - 1) / block;
  elementwise_add_kernel<<<grid, block>>>(a, b, out, n);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return -1;
  return 0;
}

CUDA_EXPORT int cuda_elementwise_add_into(
    float* dst, const float* src, int n
) {
  int block = 256;
  int grid = (n + block - 1) / block;
  elementwise_add_into_kernel<<<grid, block>>>(src, dst, n);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return -1;
  return 0;
}

// ============================================================================
// Reduce Sum Spatial — reduces [N, C, H, W] → [C] by summing over N, H, W
// ============================================================================

__global__ void reduce_sum_spatial_kernel(
    const float* input, float* output,
    int N, int C, int H, int W
) {
  // Each block handles one channel c
  int c = blockIdx.x;
  if (c >= C) return;

  float sum = 0.0f;
  // Cooperatively sum all N*H*W elements for this channel
  for (int n = 0; n < N; n++) {
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        int idx = n * C * H * W + c * H * W + h * W + w;
        sum += input[idx];
      }
    }
  }
  output[c] = sum;
}

CUDA_EXPORT int cuda_reduce_sum_spatial(
    const float* input, float* output,
    int N, int C, int H, int W
) {
  int grid = C;
  int block = 1;
  reduce_sum_spatial_kernel<<<grid, block>>>(input, output, N, C, H, W);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return -1;
  return 0;
}

// ============================================================================
// Adaptive Avg Pool2d Forward — [N, C, H, W] → [N, C, 1, 1]
// ============================================================================

__global__ void adaptive_avg_pool2d_forward_kernel(
    const float* input, float* output,
    int N, int C, int H, int W
) {
  int n = blockIdx.x;
  int c = blockIdx.y;
  if (n >= N || c >= C) return;

  float sum = 0.0f;
  for (int h = 0; h < H; h++) {
    for (int w = 0; w < W; w++) {
      int idx = n * C * H * W + c * H * W + h * W + w;
      sum += input[idx];
    }
  }
  int out_idx = n * C + c;
  output[out_idx] = sum / (H * W);
}

CUDA_EXPORT int cuda_adaptive_avg_pool2d_forward(
    const float* input, float* output,
    int N, int C, int H, int W
) {
  dim3 grid(N, C);
  dim3 block(1);
  adaptive_avg_pool2d_forward_kernel<<<grid, block>>>(input, output, N, C, H, W);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return -1;
  return 0;
}

// ============================================================================
// Adaptive Avg Pool2d Backward — [N, C, 1, 1] → [N, C, H, W]
// ============================================================================

__global__ void adaptive_avg_pool2d_backward_kernel(
    const float* dy, float* d_input,
    int N, int C, int H, int W
) {
  int n = blockIdx.x;
  int c = blockIdx.y;
  int h = threadIdx.x;
  int w = threadIdx.y;
  if (n >= N || c >= C || h >= H || w >= W) return;

  int dy_idx = n * C + c;
  float grad = dy[dy_idx] / (H * W);
  int dx_idx = n * C * H * W + c * H * W + h * W + w;
  d_input[dx_idx] = grad;
}

CUDA_EXPORT int cuda_adaptive_avg_pool2d_backward(
    const float* dy, float* d_input,
    int N, int C, int H, int W
) {
  dim3 grid(N, C);
  // Each thread handles one (h, w) spatial position
  dim3 block(
      min((unsigned int)H, 32u),
      min((unsigned int)W, 32u)
  );
  adaptive_avg_pool2d_backward_kernel<<<grid, block>>>(dy, d_input, N, C, H, W);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return -1;
  return 0;
}

// ============================================================================
// Softmax CE Forward — fused softmax + cross-entropy loss
// ============================================================================

__global__ void softmax_ce_forward_kernel(
    const float* logits, const float* targets,
    float* loss_output, float* softmax_output,
    int batch, int num_classes
) {
  int b = blockIdx.x;
  if (b >= batch) return;

  // Find max value for numerical stability
  float max_val = logits[b * num_classes];
  for (int j = 1; j < num_classes; j++) {
    float val = logits[b * num_classes + j];
    if (val > max_val) max_val = val;
  }

  // Compute softmax and sum exp(logits - max)
  float sum_exp = 0.0f;
  for (int j = 0; j < num_classes; j++) {
    float exp_val = expf(logits[b * num_classes + j] - max_val);
    softmax_output[b * num_classes + j] = exp_val;
    sum_exp += exp_val;
  }

  // Compute cross-entropy loss contribution
  float loss_contrib = 0.0f;
  for (int j = 0; j < num_classes; j++) {
    softmax_output[b * num_classes + j] /= sum_exp;
    float sm = softmax_output[b * num_classes + j];
    float target = targets[b * num_classes + j];
    if (target > 0.0f) {
      loss_contrib -= target * logf(sm);
    }
  }

  // Atomically accumulate mean loss into scalar output
  atomicAdd(loss_output, loss_contrib / (float)batch);
}

CUDA_EXPORT int cuda_softmax_ce_forward(
    const float* logits, const float* targets,
    float* loss_output, float* softmax_output,
    int batch, int num_classes
) {
  // Zero-initialize scalar loss accumulator
  cudaMemset(loss_output, 0, sizeof(float));

  int grid = batch;
  int block = 1;
  softmax_ce_forward_kernel<<<grid, block>>>(logits, targets, loss_output, softmax_output, batch, num_classes);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return -1;

  return 0;
}

// ============================================================================
// Softmax CE Backward — compute d_logits from softmax + targets
// ============================================================================

__global__ void softmax_ce_backward_kernel(
    const float* logits, const float* targets,
    float* output,
    int batch, int num_classes
) {
  int b = blockIdx.x;
  if (b >= batch) return;

  // Compute softmax of logits for this batch element
  float max_val = logits[b * num_classes];
  for (int j = 1; j < num_classes; j++) {
    float val = logits[b * num_classes + j];
    if (val > max_val) max_val = val;
  }
  float sum_exp = 0.0f;
  for (int j = 0; j < num_classes; j++) {
    sum_exp += expf(logits[b * num_classes + j] - max_val);
  }

  // d_logits[b][j] = (softmax[j] - targets[j]) / batch
  for (int j = threadIdx.x; j < num_classes; j += blockDim.x) {
    int idx = b * num_classes + j;
    float sm = expf(logits[idx] - max_val) / sum_exp;
    output[idx] = (sm - targets[idx]) / (float)batch;
  }
}

CUDA_EXPORT int cuda_softmax_ce_backward(
    const float* logits, const float* targets,
    float* output,
    int batch, int num_classes
) {
  int grid = batch;
  int threads = num_classes < 256 ? num_classes : 256;
  softmax_ce_backward_kernel<<<grid, threads>>>(logits, targets, output, batch, num_classes);
  cudaError_t err = cudaGetLastError();
  return (err == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Softmax CE Forward with integer class labels — no one-hot needed
// labels: float array of shape [batch], each value is an integer class index
// ============================================================================

__global__ void softmax_ce_forward_labels_kernel(
    const float* logits, const float* labels,
    float* loss_output,
    int batch, int num_classes
) {
  int b = blockIdx.x;
  if (b >= batch) return;

  // Find max value for numerical stability
  float max_val = logits[b * num_classes];
  for (int j = 1; j < num_classes; j++) {
    float val = logits[b * num_classes + j];
    if (val > max_val) max_val = val;
  }

  // Compute sum of exp(logits - max)
  float sum_exp = 0.0f;
  for (int j = 0; j < num_classes; j++) {
    sum_exp += expf(logits[b * num_classes + j] - max_val);
  }

  // Cross-entropy loss = -log(softmax[class_idx])
  int class_idx = (int)labels[b];
  float log_softmax = logits[b * num_classes + class_idx] - max_val - logf(sum_exp);
  float loss_contrib = -log_softmax;

  // Atomically accumulate mean loss into scalar output
  atomicAdd(loss_output, loss_contrib / (float)batch);
}

CUDA_EXPORT int cuda_softmax_ce_forward_labels(
    const float* logits, const float* labels,
    float* loss_output,
    int batch, int num_classes
) {
  cudaMemset(loss_output, 0, sizeof(float));

  softmax_ce_forward_labels_kernel<<<batch, 1>>>(
    logits, labels, loss_output, batch, num_classes
  );
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return -1;

  return 0;
}

// ============================================================================
// Softmax CE Backward with integer class labels
// d_logits[b][j] = (softmax[j] - 1{j == class_idx}) / batch
// ============================================================================

__global__ void softmax_ce_backward_labels_kernel(
    const float* logits, const float* labels,
    float* output,
    int batch, int num_classes
) {
  int b = blockIdx.x;
  if (b >= batch) return;

  // Compute softmax of logits for this batch element
  float max_val = logits[b * num_classes];
  for (int j = 1; j < num_classes; j++) {
    float val = logits[b * num_classes + j];
    if (val > max_val) max_val = val;
  }
  float sum_exp = 0.0f;
  for (int j = 0; j < num_classes; j++) {
    sum_exp += expf(logits[b * num_classes + j] - max_val);
  }

  int class_idx = (int)labels[b];
  // d_logits[b][j] = (softmax[j] - indicator(j == class_idx)) / batch
  for (int j = threadIdx.x; j < num_classes; j += blockDim.x) {
    int idx = b * num_classes + j;
    float sm = expf(logits[idx] - max_val) / sum_exp;
    float indicator = (j == class_idx) ? 1.0f : 0.0f;
    output[idx] = (sm - indicator) / (float)batch;
  }
}

CUDA_EXPORT int cuda_softmax_ce_backward_labels(
    const float* logits, const float* labels,
    float* output,
    int batch, int num_classes
) {
  int threads = num_classes < 256 ? num_classes : 256;
  softmax_ce_backward_labels_kernel<<<batch, threads>>>(
    logits, labels, output, batch, num_classes
  );
  cudaError_t err = cudaGetLastError();
  return (err == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// Reduce Sum Leading — sum over first N leading dimensions
// Input: [d0, d1, ..., d_{n_lead-1}, d_n, ..., d_last]
// Output: [d_n, ..., d_last]
// Each output element is the sum of lead_stride input elements
// ============================================================================

__global__ void reduce_sum_leading_kernel(
    const float* input, float* output,
    int output_numel, int lead_stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_numel) return;
    float sum = 0.0f;
    for (int k = 0; k < lead_stride; k++) {
        sum += input[k * output_numel + idx];
    }
    output[idx] = sum;
}

CUDA_EXPORT int cuda_reduce_sum_leading(
    const float* input, float* output,
    int output_numel, int lead_stride
) {
    int threads = 256;
    int blocks = (output_numel + threads - 1) / threads;
    reduce_sum_leading_kernel<<<blocks, threads>>>(input, output, output_numel, lead_stride);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    return 0;
}

// ============================================================================
// Reduce Sum Dim — sum over a specific dimension where target has size 1
// Input: [..., dim_size, ...] -> Output: [..., 1, ...]
// ============================================================================

__global__ void reduce_sum_dim_kernel(
    const float* input, float* output,
    int output_numel, int dim_size, int stride_before, int stride_after
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_numel) return;
    int b = idx / stride_after;
    int a = idx % stride_after;
    float sum = 0.0f;
    for (int k = 0; k < dim_size; k++) {
        sum += input[b * stride_before + k * stride_after + a];
    }
    output[idx] = sum;
}

CUDA_EXPORT int cuda_reduce_sum_dim(
    const float* input, float* output,
    int output_numel, int dim_size, int stride_before, int stride_after
) {
    int threads = 256;
    int blocks = (output_numel + threads - 1) / threads;
    reduce_sum_dim_kernel<<<blocks, threads>>>(input, output, output_numel, dim_size, stride_before, stride_after);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    return 0;
}

// ============================================================================
// General Matrix Multiply — wraps cuBLAS sgemm for row-major tensors
// ============================================================================
//
// Convention:
//   transpose_a=false: A is stored as [M×K] in row-major
//   transpose_a=true:  A is stored as [K×M] in row-major (use A^T = [M×K])
//   transpose_b=false: B is stored as [K×N] in row-major
//   transpose_b=true:  B is stored as [N×K] in row-major (use B^T = [K×N])
//   C is always [M×N] in row-major
//
// Row-major C stored in memory = Column-major C^T stored in memory.
// So we compute C_col = C_row^T = op_B(B)^T @ op_A(A)^T.
//
// Using the cuBLAS operand-swap trick (pass B first, A second):
//   C_col = B_col_op @ A_col_op
// where:
//   B_col = B_row^T, and we apply cuBLAS op to get op_B(B)^T
//   A_col = A_row^T, and we apply cuBLAS op to get op_A(A)^T
//
// cuBLAS ops:
//   transpose_X=false: op_X(X) = X, op_X(X)^T = X^T = X_col → cuBLAS op = N
//   transpose_X=true:  op_X(X) = X^T, op_X(X)^T = X → need X_col^T → cuBLAS op = T

CUDA_EXPORT int cuda_matmul(
    void* cublas_handle,
    float* a, float* b, float* c,
    int M, int K, int N,
    int transpose_a, int transpose_b,
    float alpha, float beta
) {
  cublasHandle_t handle = (cublasHandle_t)cublas_handle;

  // cuBLAS ops: pass B first, A second (swap trick for row-major)
  cublasOperation_t cublas_op_b = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t cublas_op_a = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;

  // Leading dimensions for column-major view:
  // B stored [K×N] (no transpose) or [N×K] (transpose)
  //   → B_col has N rows (no transpose) or K rows (transpose)
  // A stored [M×K] (no transpose) or [K×M] (transpose)
  //   → A_col has K rows (no transpose) or M rows (transpose)
  int lda_b = transpose_b ? K : N;
  int lda_a = transpose_a ? M : K;

  cublasStatus_t status = cublasGemmEx(
    handle,
    cublas_op_b, cublas_op_a,
    N, M, K,
    &alpha,
    b, CUDA_R_32F, lda_b,
    a, CUDA_R_32F, lda_a,
    &beta,
    c, CUDA_R_32F, N,
    CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP
  );

  return (status == CUBLAS_STATUS_SUCCESS) ? 0 : (int)status;
}

// In-place accumulate: dst[i] += src[i] for i in 0..n
__global__ void accumulate_inplace_kernel(float* dst, const float* src, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] += src[idx];
  }
}

CUDA_EXPORT int cuda_accumulate_inplace(float* dst, const float* src, int n) {
  int block = 256;
  int grid = (n + block - 1) / block;
  accumulate_inplace_kernel<<<grid, block>>>(dst, src, n);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return -1;
  return 0;
}

// ============================================================================
// GPU Optimizer Kernels
// ============================================================================

// Fused SGD + momentum step with gradient clipping:
//   velocity[i] = momentum * velocity[i] + clip_coef * grad[i] + weight_decay * param[i]
//   param[i]    = param[i] - lr * velocity[i]
__global__ void sgd_momentum_step_kernel(
    float* param, const float* grad, float* velocity,
    int n, float lr, float momentum, float weight_decay, float clip_coef
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = grad[idx] * clip_coef;
        float v = momentum * velocity[idx] + g + weight_decay * param[idx];
        velocity[idx] = v;
        param[idx] = param[idx] - lr * v;
    }
}

CUDA_EXPORT int cuda_sgd_momentum_step(
    void* param, void* grad, void* velocity,
    int n, float lr, float momentum, float weight_decay, float clip_coef
) {
    int block = 256;
    int grid = (n + block - 1) / block;
    sgd_momentum_step_kernel<<<grid, block>>>(
        (float*)param, (const float*)grad, (float*)velocity,
        n, lr, momentum, weight_decay, clip_coef
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    return 0;
}

// SGD step without momentum:
//   param[i] = param[i] - lr * (grad[i] + weight_decay * param[i])
__global__ void sgd_step_kernel(
    float* param, const float* grad,
    int n, float lr, float weight_decay
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        param[idx] = param[idx] - lr * (grad[idx] + weight_decay * param[idx]);
    }
}

CUDA_EXPORT int cuda_sgd_step(
    void* param, void* grad,
    int n, float lr, float weight_decay
) {
    int block = 256;
    int grid = (n + block - 1) / block;
    sgd_step_kernel<<<grid, block>>>(
        (float*)param, (const float*)grad,
        n, lr, weight_decay
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    return 0;
}

// Squared L2 norm reduction: dst[0] = sum(src[i]^2)
// Uses atomicAdd for simplicity — sufficient for optimizer use case
__global__ void norm_sq_kernel(const float* src, int n, float* dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    // Each thread sums a stride
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += src[i] * src[i];
    }
    atomicAdd(dst, sum);
}

CUDA_EXPORT int cuda_norm_sq(void* src, int n, void* dst) {
    int block = 256;
    int grid = min((n + block - 1) / block, 1024);
    norm_sq_kernel<<<grid, block>>>((const float*)src, n, (float*)dst);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    return 0;
}

// Scale GPU buffer in-place: data[i] *= scale
__global__ void scale_inplace_kernel(float* data, int n, float scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    data[idx] *= scale;
}

CUDA_EXPORT int cuda_scale_inplace(void* data, int n, float scale) {
  int block = 256;
  int grid = (n + block - 1) / block;
  scale_inplace_kernel<<<grid, block>>>((float*)data, n, scale);
  return 0;
}

// Fill GPU buffer with a scalar value: out[i] = value
__global__ void fill_kernel(float* out, int n, float value) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = value;
}

CUDA_EXPORT int cuda_fill(void* out, int n, float value) {
  int block = 256;
  int grid = (n + block - 1) / block;
  fill_kernel<<<grid, block>>>((float*)out, n, value);
  return 0;
}

// Read a single f32 from GPU to CPU
CUDA_EXPORT float cuda_read_scalar(void* src) {
    float val;
    cudaMemcpy(&val, src, sizeof(float), cudaMemcpyDeviceToHost);
    return val;
}

// In-place axpy: y[i] += alpha * x[i]
__global__ void axpy_kernel(int n, float alpha, const float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] += alpha * x[idx];
    }
}

CUDA_EXPORT int cuda_axpy(int n, float alpha, void* x, void* y) {
    int block = 256;
    int grid = (n + block - 1) / block;
    axpy_kernel<<<grid, block>>>(n, alpha, (const float*)x, (float*)y);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    return 0;
}

// ============================================================================
// Elementwise GPU kernels (eliminate host roundtrips)
// ============================================================================

// out[i] = a[i] - b[i]
__global__ void sub_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

CUDA_EXPORT int cuda_sub(const float* a, const float* b, float* out, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    sub_kernel<<<grid, block>>>(a, b, out, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    return 0;
}

// out[i] = a[i] * b[i]
__global__ void mul_elem_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

CUDA_EXPORT int cuda_mul_elem(const float* a, const float* b, float* out, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    mul_elem_kernel<<<grid, block>>>(a, b, out, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    return 0;
}

// out[i] = src[i] * src[i]
__global__ void square_kernel(const float* src, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = src[idx];
        out[idx] = v * v;
    }
}

CUDA_EXPORT int cuda_square(const float* src, float* out, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    square_kernel<<<grid, block>>>(src, out, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    return 0;
}

// out[i] = sqrt(src[i])
__global__ void sqrt_kernel(const float* src, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sqrtf(src[idx]);
    }
}

CUDA_EXPORT int cuda_sqrt(const float* src, float* out, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    sqrt_kernel<<<grid, block>>>(src, out, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    return 0;
}

// out[i] = a[i] / b[i]
__global__ void div_elem_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / b[idx];
    }
}

CUDA_EXPORT int cuda_div_elem(const float* a, const float* b, float* out, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    div_elem_kernel<<<grid, block>>>(a, b, out, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    return 0;
}

// out[i] = src[i] * scale  (out-of-place version of scale_inplace)
__global__ void scale_kernel(const float* src, float* out, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = src[idx] * scale;
    }
}

CUDA_EXPORT int cuda_scale_out(const float* src, float* out, int n, float scale) {
    int block = 256;
    int grid = (n + block - 1) / block;
    scale_kernel<<<grid, block>>>(src, out, n, scale);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    return 0;
}

// 2D matrix transpose: out[j*n + i] = in[i*m + j]  for 0<=i<n, 0<=j<m
__global__ void transpose_2d_kernel(const float* in, float* out, int n, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * m;
    if (idx < total) {
        int i = idx / m;
        int j = idx % m;
        out[j * n + i] = in[idx];
    }
}

CUDA_EXPORT int cuda_transpose_2d(const float* in, float* out, int n, int m) {
    int block = 256;
    int total = n * m;
    int grid = (total + block - 1) / block;
    transpose_2d_kernel<<<grid, block>>>(in, out, n, m);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    return 0;
}

// ============================================================================
// Cache management for shape-keyed context caching
// ============================================================================

CUDA_EXPORT void conv2d_cache_clear(void) {
  if (!conv_cache) return;
  for (auto& kv : *conv_cache) {
    conv2d_destroy(kv.second);
  }
  delete conv_cache;
  conv_cache = nullptr;
}

CUDA_EXPORT void batchnorm_cache_clear(void) {
  if (!bn_cache) return;
  for (auto& kv : *bn_cache) {
    batchnorm_destroy(kv.second);
  }
  delete bn_cache;
  bn_cache = nullptr;
}

CUDA_EXPORT void pool2d_cache_clear(void) {
  if (!pool_cache) return;
  for (auto& kv : *pool_cache) {
    pool2d_destroy(kv.second);
  }
  delete pool_cache;
  pool_cache = nullptr;
}

// ============================================================================
// Multi-tensor fused kernels (for optimizer — process all params in 1 launch)
// ============================================================================

// Multi-tensor squared L2 norm: dst[0] += sum(t[k][i]^2) for all tensors k
// Each block handles one tensor.
// tensors: host array of (float* ptr, int n) pairs
__global__ void multi_tensor_norm_sq_kernel(
    float** tensor_ptrs, int* tensor_sizes, int num_tensors, float* dst
) {
    int k = blockIdx.x; // one block per tensor
    if (k >= num_tensors) return;
    const float* src = tensor_ptrs[k];
    int n = tensor_sizes[k];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sum += src[i] * src[i];
    }
    // Reduce within block
    __shared__ float sdata[256];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(dst, sdata[0]);
    }
}

CUDA_EXPORT int cuda_multi_tensor_norm_sq(
    float** tensor_ptrs, int* tensor_sizes, int num_tensors, float* dst
) {
    if (num_tensors == 0) return 0;
    size_t ptrs_bytes = num_tensors * sizeof(float*);
    float** d_ptrs;
    cudaMalloc(&d_ptrs, ptrs_bytes);
    cudaMemcpy(d_ptrs, tensor_ptrs, ptrs_bytes, cudaMemcpyHostToDevice);
    int* d_sizes;
    cudaMalloc(&d_sizes, num_tensors * sizeof(int));
    cudaMemcpy(d_sizes, tensor_sizes, num_tensors * sizeof(int), cudaMemcpyHostToDevice);
    int block = 256;
    multi_tensor_norm_sq_kernel<<<num_tensors, block>>>(
        d_ptrs, d_sizes, num_tensors, dst
    );
    cudaError_t err = cudaGetLastError();
    cudaFree(d_ptrs);
    cudaFree(d_sizes);
    if (err != cudaSuccess) return -1;
    return 0;
}

// Pure GPU norm_sq — all pointer arrays already on device.
CUDA_EXPORT int cuda_multi_tensor_norm_sq_gpu_only(
    float** d_ptrs, int* d_sizes, int num_tensors, float* dst
) {
    if (num_tensors == 0) return 0;
    int block = 256;
    multi_tensor_norm_sq_kernel<<<num_tensors, block>>>(
        d_ptrs, d_sizes, num_tensors, dst
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    return 0;
}

// Multi-tensor norm_sq with multi-block per tensor for better GPU occupancy.
// d_block_offsets: device array of prefix-sum of ceil(size/TPB) for each tensor.
// total_blocks: sum of blocks across all tensors.
__global__ void mt_norm_sq_kernel(
    float** tensor_ptrs, int* tensor_sizes, int* d_block_offsets,
    int num_tensors, float* dst
) {
    int flat_block = blockIdx.x;
    // Binary search for tensor index
    int lo = 0, hi = num_tensors - 1, k = 0;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (flat_block >= d_block_offsets[mid]) { k = mid; lo = mid + 1; }
        else { hi = mid - 1; }
    }
    int n = tensor_sizes[k];
    int local_block = flat_block - d_block_offsets[k];
    int chunk_start = local_block * blockDim.x;
    if (chunk_start >= n) return;
    int chunk_end = chunk_start + blockDim.x;
    if (chunk_end > n) chunk_end = n;
    const float* src = tensor_ptrs[k];
    float sum = 0.0f;
    for (int i = chunk_start + threadIdx.x; i < chunk_end; i += blockDim.x) {
        sum += src[i] * src[i];
    }
    __shared__ float sdata[256];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(dst, sdata[0]);
}

// Multi-tensor SGD with multi-block per tensor.
__global__ void mt_sgd_momentum_kernel(
    float** param_ptrs, float** grad_ptrs, float** velocity_ptrs,
    int* tensor_sizes, int* d_block_offsets,
    int num_tensors,
    float lr, float momentum, float weight_decay, float clip_coef
) {
    int flat_block = blockIdx.x;
    int lo = 0, hi = num_tensors - 1, k = 0;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (flat_block >= d_block_offsets[mid]) { k = mid; lo = mid + 1; }
        else { hi = mid - 1; }
    }
    float* param = param_ptrs[k];
    const float* grad = grad_ptrs[k];
    float* vel = velocity_ptrs[k];
    int n = tensor_sizes[k];
    int local_block = flat_block - d_block_offsets[k];
    int chunk_start = local_block * blockDim.x;
    if (chunk_start >= n) return;
    int chunk_end = chunk_start + blockDim.x;
    if (chunk_end > n) chunk_end = n;
    for (int i = chunk_start + threadIdx.x; i < chunk_end; i += blockDim.x) {
        float g = grad[i] * clip_coef;
        float v = momentum * vel[i] + g + weight_decay * param[i];
        vel[i] = v;
        param[i] = param[i] - lr * v;
    }
}

// Launch multi-block-per-tensor norm_sq. Host computes block_offsets and total_blocks.
CUDA_EXPORT int cuda_mt_norm_sq_gpu_only(
    float** d_ptrs, int* d_sizes, int* d_block_offsets,
    int num_tensors, int total_blocks, float* dst
) {
    if (num_tensors == 0 || total_blocks == 0) return 0;
    mt_norm_sq_kernel<<<total_blocks, 256>>>(d_ptrs, d_sizes, d_block_offsets, num_tensors, dst);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    return 0;
}

// Multi-block SGD kernel that reads norm from GPU memory — no D2H sync needed.
__global__ void mt_sgd_momentum_autoclip_kernel(
    float** param_ptrs, float** grad_ptrs, float** velocity_ptrs,
    int* tensor_sizes, int* d_block_offsets,
    int num_tensors,
    float lr, float momentum, float weight_decay,
    const float* norm_buf, float max_grad_norm
) {
    float norm_sq = norm_buf[0];
    float clip_coef = (norm_sq > max_grad_norm * max_grad_norm)
        ? max_grad_norm / sqrtf(norm_sq) : 1.0f;

    int flat_block = blockIdx.x;
    int lo = 0, hi = num_tensors - 1, k = 0;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (flat_block >= d_block_offsets[mid]) { k = mid; lo = mid + 1; }
        else { hi = mid - 1; }
    }
    float* param = param_ptrs[k];
    const float* grad = grad_ptrs[k];
    float* vel = velocity_ptrs[k];
    int n = tensor_sizes[k];
    int local_block = flat_block - d_block_offsets[k];
    int chunk_start = local_block * blockDim.x;
    if (chunk_start >= n) return;
    int chunk_end = chunk_start + blockDim.x;
    if (chunk_end > n) chunk_end = n;
    for (int i = chunk_start + threadIdx.x; i < chunk_end; i += blockDim.x) {
        float g = grad[i] * clip_coef;
        float v = momentum * vel[i] + g + weight_decay * param[i];
        vel[i] = v;
        param[i] = param[i] - lr * v;
    }
}

CUDA_EXPORT int cuda_mt_sgd_momentum_gpu_only(
    float** d_param_ptrs, float** d_grad_ptrs, float** d_vel_ptrs,
    int* d_sizes, int* d_block_offsets,
    int num_tensors, int total_blocks,
    float lr, float momentum, float weight_decay, float clip_coef
) {
    if (num_tensors == 0 || total_blocks == 0) return 0;
    mt_sgd_momentum_kernel<<<total_blocks, 256>>>(
        d_param_ptrs, d_grad_ptrs, d_vel_ptrs,
        d_sizes, d_block_offsets, num_tensors,
        lr, momentum, weight_decay, clip_coef
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    return 0;
}

CUDA_EXPORT int cuda_mt_sgd_momentum_autoclip(
    float** d_param_ptrs, float** d_grad_ptrs, float** d_vel_ptrs,
    int* d_sizes, int* d_block_offsets,
    int num_tensors, int total_blocks,
    float lr, float momentum, float weight_decay,
    float* norm_buf, float max_grad_norm
) {
    if (num_tensors == 0 || total_blocks == 0) return 0;
    mt_sgd_momentum_autoclip_kernel<<<total_blocks, 256>>>(
        d_param_ptrs, d_grad_ptrs, d_vel_ptrs,
        d_sizes, d_block_offsets, num_tensors,
        lr, momentum, weight_decay, norm_buf, max_grad_norm
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    return 0;
}

// Multi-tensor fused SGD momentum step:
//   for each tensor k:
//     g = grad[k][i] * clip_coef
//     v = momentum * velocity[k][i] + g + weight_decay * param[k][i]
//     velocity[k][i] = v
//     param[k][i] = param[k][i] - lr * v
// Each block handles one tensor.
__global__ void multi_tensor_sgd_momentum_kernel(
    float** param_ptrs, float** grad_ptrs, float** velocity_ptrs,
    int* tensor_sizes, int num_tensors,
    float lr, float momentum, float weight_decay, float clip_coef
) {
    int k = blockIdx.x; // one block per tensor
    if (k >= num_tensors) return;
    float* param = param_ptrs[k];
    const float* grad = grad_ptrs[k];
    float* vel = velocity_ptrs[k];
    int n = tensor_sizes[k];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float g = grad[i] * clip_coef;
        float v = momentum * vel[i] + g + weight_decay * param[i];
        vel[i] = v;
        param[i] = param[i] - lr * v;
    }
}

CUDA_EXPORT int cuda_multi_tensor_sgd_momentum_step(
    float** param_ptrs, float** grad_ptrs, float** velocity_ptrs,
    int* tensor_sizes, int num_tensors,
    float lr, float momentum, float weight_decay, float clip_coef
) {
    if (num_tensors == 0) return 0;
    size_t ptrs_bytes = num_tensors * sizeof(float*);
    float** d_params; float** d_grads; float** d_vels;
    int* d_sizes;
    cudaMalloc(&d_params, ptrs_bytes);
    cudaMemcpy(d_params, param_ptrs, ptrs_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_grads, ptrs_bytes);
    cudaMemcpy(d_grads, grad_ptrs, ptrs_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_vels, ptrs_bytes);
    cudaMemcpy(d_vels, velocity_ptrs, ptrs_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_sizes, num_tensors * sizeof(int));
    cudaMemcpy(d_sizes, tensor_sizes, num_tensors * sizeof(int), cudaMemcpyHostToDevice);
    int block = 256;
    multi_tensor_sgd_momentum_kernel<<<num_tensors, block>>>(
        d_params, d_grads, d_vels, d_sizes, num_tensors,
        lr, momentum, weight_decay, clip_coef
    );
    cudaError_t err = cudaGetLastError();
    cudaFree(d_params); cudaFree(d_grads); cudaFree(d_vels); cudaFree(d_sizes);
    if (err != cudaSuccess) return -1;
    return 0;
}

CUDA_EXPORT int cuda_multi_tensor_sgd_momentum_step_prealloc(
    float** param_ptrs, float** grad_ptrs, float** velocity_ptrs,
    int* tensor_sizes, int num_tensors,
    float lr, float momentum, float weight_decay, float clip_coef,
    float** d_params, float** d_grads, float** d_vels, int* d_sizes
) {
    if (num_tensors == 0) return 0;
    size_t ptrs_bytes = num_tensors * sizeof(float*);
    size_t sizes_bytes = num_tensors * sizeof(int);
    cudaMemcpy(d_params, param_ptrs, ptrs_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_grads, grad_ptrs, ptrs_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vels, velocity_ptrs, ptrs_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sizes, tensor_sizes, sizes_bytes, cudaMemcpyHostToDevice);
    int block = 256;
    multi_tensor_sgd_momentum_kernel<<<num_tensors, block>>>(
        d_params, d_grads, d_vels, d_sizes, num_tensors,
        lr, momentum, weight_decay, clip_coef
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    return 0;
}

// Pure GPU kernel launch — no h2d copies at all.
// All pointer arrays must already be populated on device.
CUDA_EXPORT int cuda_multi_tensor_sgd_momentum_step_gpu_only(
    float** d_params, float** d_grads, float** d_vels, int* d_sizes,
    int num_tensors,
    float lr, float momentum, float weight_decay, float clip_coef
) {
    if (num_tensors == 0) return 0;
    int block = 256;
    multi_tensor_sgd_momentum_kernel<<<num_tensors, block>>>(
        d_params, d_grads, d_vels, d_sizes, num_tensors,
        lr, momentum, weight_decay, clip_coef
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    return 0;
}

// Contiguous SGD momentum kernel: all params/grads/vels packed into 3 flat buffers.
// One kernel launch covers every element — no load imbalance.
__global__ void contiguous_sgd_momentum_kernel(
    float* params, const float* grads, float* vels,
    int total_elements,
    float lr, float momentum, float weight_decay, float clip_coef
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total_elements;
         i += blockDim.x * gridDim.x) {
        float g = grads[i] * clip_coef;
        float v = momentum * vels[i] + g + weight_decay * params[i];
        vels[i] = v;
        params[i] = params[i] - lr * v;
    }
}

CUDA_EXPORT int cuda_contiguous_sgd_momentum_step(
    float* params, const float* grads, float* vels,
    int total_elements,
    float lr, float momentum, float weight_decay, float clip_coef
) {
    if (total_elements == 0) return 0;
    int block = 256;
    int grid = (total_elements + block - 1) / block;
    contiguous_sgd_momentum_kernel<<<grid, block>>>(
        params, grads, vels, total_elements,
        lr, momentum, weight_decay, clip_coef
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    return 0;
}

// Better design: gather with precomputed offsets
__global__ void gather_offset_kernel(
    float** d_ptrs, const int* d_offsets, const int* d_sizes,
    int num_tensors, float* dst
) {
    int k = blockIdx.x;
    if (k >= num_tensors) return;
    float* src = d_ptrs[k];
    int off = d_offsets[k];
    int n = d_sizes[k];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        dst[off + i] = src[i];
    }
}

// Scatter: copy flat buffer back to scattered tensors.
__global__ void scatter_offset_kernel(
    float* src, float** d_ptrs, const int* d_offsets, const int* d_sizes,
    int num_tensors
) {
    int k = blockIdx.x;
    if (k >= num_tensors) return;
    float* dst = d_ptrs[k];
    int off = d_offsets[k];
    int n = d_sizes[k];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = src[off + i];
    }
}

// Single kernel: gather all grads into flat buffer, compute total norm_sq, SGD+momentum on flat buffer, scatter params back.
// This is the "M3 fused" approach — one kernel does everything.
// But it needs to run sequentially: first gather, then compute, then scatter.
// Actually, gather and scatter need to be separate launches because of data dependencies.
// Let's keep them separate for clarity.

// Contiguous norm_sq: compute sum of squares over a flat buffer, reduce to single float.
// Uses atomicAdd for simplicity (fast enough for a single output value).
__global__ void contiguous_norm_sq_kernel(
    const float* data, int n, float* out
) {
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        sum += data[i] * data[i];
    }
    // Block-level reduction
    __shared__ float sbuf[256];
    sbuf[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sbuf[threadIdx.x] += sbuf[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, sbuf[0]);
}

CUDA_EXPORT int cuda_gather(
    float** d_src_ptrs, const int* d_offsets, const int* d_sizes,
    int num_tensors, float* dst
) {
    if (num_tensors == 0) return 0;
    gather_offset_kernel<<<num_tensors, 256>>>(d_src_ptrs, d_offsets, d_sizes, num_tensors, dst);
    cudaError_t err = cudaGetLastError();
    return (err != cudaSuccess) ? -1 : 0;
}

CUDA_EXPORT int cuda_scatter(
    float* src, float** d_dst_ptrs, const int* d_offsets, const int* d_sizes,
    int num_tensors
) {
    if (num_tensors == 0) return 0;
    scatter_offset_kernel<<<num_tensors, 256>>>(src, d_dst_ptrs, d_offsets, d_sizes, num_tensors);
    cudaError_t err = cudaGetLastError();
    return (err != cudaSuccess) ? -1 : 0;
}

CUDA_EXPORT int cuda_contiguous_norm_sq(
    const float* data, int n, float* out
) {
    if (n == 0) return 0;
    int block = 256;
    int grid = (n + block - 1) / block;
    if (grid > 65535) grid = 65535;
    contiguous_norm_sq_kernel<<<grid, block>>>(data, n, out);
    cudaError_t err = cudaGetLastError();
    return (err != cudaSuccess) ? -1 : 0;
}

CUDA_EXPORT int conv2d_cache_size(void) {
  return conv_cache ? (int)conv_cache->size() : 0;
}

CUDA_EXPORT int batchnorm_cache_size(void) {
  return bn_cache ? (int)bn_cache->size() : 0;
}

CUDA_EXPORT int pool2d_cache_size(void) {
  return pool_cache ? (int)pool_cache->size() : 0;
}

CUDA_EXPORT void* cuda_event_create(void) {
  cudaEvent_t e;
  cudaError_t err = cudaEventCreate(&e);
  if (err != cudaSuccess) return nullptr;
  return (void*)e;
}

CUDA_EXPORT int cuda_event_record(void* stream, void* event) {
  cudaError_t err = cudaEventRecord((cudaEvent_t)event, (cudaStream_t)stream);
  return (err != cudaSuccess) ? -1 : 0;
}

CUDA_EXPORT int cuda_event_elapsed_ms(void* start, void* end, float* ms) {
  cudaError_t err = cudaEventElapsedTime(ms, (cudaEvent_t)start, (cudaEvent_t)end);
  return (err != cudaSuccess) ? -1 : 0;
}

CUDA_EXPORT int cuda_event_synchronize(void* event) {
  cudaError_t err = cudaEventSynchronize((cudaEvent_t)event);
  return (err != cudaSuccess) ? -1 : 0;
}

// GPU-side batch gather: gather N samples from a large flat buffer by index.
// src: flat GPU buffer [num_total_samples * stride]
// indices: GPU int32 array [batch_size] — sample indices
// dst: flat GPU buffer [batch_size * stride] — output
// stride: elements per sample (e.g. 3072 for CIFAR images)
// batch_size: number of samples to gather
// Each thread handles one float element of the output.
__global__ void batch_gather_kernel(
    const float* src, const int* indices, float* dst,
    int stride, int batch_size, int total_elements
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total_elements;
         i += blockDim.x * gridDim.x) {
        int sample = i / stride;
        int offset = i % stride;
        int src_idx = indices[sample] * stride + offset;
        dst[i] = src[src_idx];
    }
}

CUDA_EXPORT int cuda_batch_gather(
    const float* src, const int* indices, float* dst,
    int stride, int batch_size
) {
    int total_elements = batch_size * stride;
    if (total_elements == 0) return 0;
    int block = 256;
    int grid = (total_elements + block - 1) / block;
    if (grid > 65535) grid = 65535;
    batch_gather_kernel<<<grid, block>>>(src, indices, dst, stride, batch_size, total_elements);
    cudaError_t err = cudaGetLastError();
    return (err != cudaSuccess) ? -1 : 0;
}

// Argmax along a dimension for a 2D tensor [outer, inner].
// Each thread block handles one row (one "outer" index).
// Output is a float array of [outer] with the argmax index stored as float.
__global__ void argmax_kernel(
    const float* input, float* output,
    int outer, int inner
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= outer) return;

    const float* row_ptr = input + row * inner;
    float best_val = row_ptr[0];
    int32_t best_idx = 0;

    // Each thread scans a subset of the row, then reduce in shared memory
    for (int j = 1 + threadIdx.x; j < inner; j += blockDim.x) {
        float val = row_ptr[j];
        if (val > best_val) {
            best_val = val;
            best_idx = j;
        }
    }

    // Shared memory reduction: one result per row
    __shared__ float s_val[256];
    __shared__ int32_t s_idx[256];

    s_val[threadIdx.x * blockDim.y + threadIdx.y] = best_val;
    s_idx[threadIdx.x * blockDim.y + threadIdx.y] = best_idx;
    __syncthreads();

    // Reduce: thread 0 of each row picks the best
    if (threadIdx.x == 0) {
        float my_val = s_val[threadIdx.y];
        int32_t my_idx = s_idx[threadIdx.y];
        for (int t = 1; t < blockDim.x; t++) {
            float other_val = s_val[t * blockDim.y + threadIdx.y];
            int32_t other_idx = s_idx[t * blockDim.y + threadIdx.y];
            if (other_val > my_val) {
                my_val = other_val;
                my_idx = other_idx;
            }
        }
        output[row] = (float)my_idx;
    }
}

CUDA_EXPORT int cuda_argmax(
    const float* input, float* output,
    int outer, int inner
) {
    // Each block handles blockDim.y rows, blockDim.x threads per row
    int threads_per_row = 256;
    int rows_per_block = 1;
    dim3 block(threads_per_row, rows_per_block);
    int grid = (outer + rows_per_block - 1) / rows_per_block;
    if (grid > 65535) grid = 65535;
    argmax_kernel<<<grid, block>>>(input, output, outer, inner);
    cudaError_t err = cudaGetLastError();
    return (err != cudaSuccess) ? -1 : 0;
}

// ═══════════════════════════════════════════════════════════════
// Fused Adam kernel with autoclip
//   m = beta1 * m + (1-beta1) * g
//   v = beta2 * v + (1-beta2) * g^2
//   p = p - lr * (m / (1-beta1^t)) / (sqrt(v / (1-beta2^t)) + eps)
//   With optional weight decay: g += wd * p before update
// ═══════════════════════════════════════════════════════════════

__global__ void mt_adam_autoclip_kernel(
    float** param_ptrs, float** grad_ptrs,
    float** m_ptrs, float** v_ptrs,
    int* tensor_sizes, int* d_block_offsets,
    int num_tensors,
    float lr, float beta1, float beta2,
    float one_minus_beta1, float one_minus_beta2,
    float bias_correction1, float bias_correction2,
    float eps, float weight_decay,
    const float* norm_buf, float max_grad_norm
) {
    float norm_sq = norm_buf[0];
    float clip_coef = (norm_sq > max_grad_norm * max_grad_norm)
        ? max_grad_norm / sqrtf(norm_sq) : 1.0f;

    int flat_block = blockIdx.x;
    int lo = 0, hi = num_tensors - 1, k = 0;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (flat_block >= d_block_offsets[mid]) { k = mid; lo = mid + 1; }
        else { hi = mid - 1; }
    }
    float* param = param_ptrs[k];
    const float* grad = grad_ptrs[k];
    float* m = m_ptrs[k];
    float* v = v_ptrs[k];
    int n = tensor_sizes[k];
    int local_block = flat_block - d_block_offsets[k];
    int chunk_start = local_block * blockDim.x;
    if (chunk_start >= n) return;
    int chunk_end = chunk_start + blockDim.x;
    if (chunk_end > n) chunk_end = n;

    float step_size = lr * bias_correction1;

    for (int i = chunk_start + threadIdx.x; i < chunk_end; i += blockDim.x) {
        float g = grad[i] * clip_coef;
        if (weight_decay > 0.0f) g = g + weight_decay * param[i];

        float mi = beta1 * m[i] + one_minus_beta1 * g;
        m[i] = mi;
        float vi = beta2 * v[i] + one_minus_beta2 * g * g;
        v[i] = vi;

        float v_hat = vi * bias_correction2;
        float denom = sqrtf(v_hat) + eps;
        param[i] = param[i] - step_size * mi / denom;
    }
}

CUDA_EXPORT int cuda_mt_adam_autoclip(
    float** d_param_ptrs, float** d_grad_ptrs,
    float** d_m_ptrs, float** d_v_ptrs,
    int* d_sizes, int* d_block_offsets,
    int num_tensors, int total_blocks,
    float lr, float beta1, float beta2,
    float one_minus_beta1, float one_minus_beta2,
    float bias_correction1, float bias_correction2,
    float eps, float weight_decay,
    float* norm_buf, float max_grad_norm
) {
    if (num_tensors == 0 || total_blocks == 0) return 0;
    mt_adam_autoclip_kernel<<<total_blocks, 256>>>(
        d_param_ptrs, d_grad_ptrs, d_m_ptrs, d_v_ptrs,
        d_sizes, d_block_offsets, num_tensors,
        lr, beta1, beta2, one_minus_beta1, one_minus_beta2,
        bias_correction1, bias_correction2, eps, weight_decay,
        norm_buf, max_grad_norm
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    return 0;
}

// ═══════════════════════════════════════════════════════════════
// Fused RMSprop kernel with autoclip and optional momentum
//   v = alpha * v + (1-alpha) * g^2
//   if momentum: buf = momentum * buf + g / (sqrt(v) + eps); p -= lr * buf
//   else:        p -= lr * g / (sqrt(v) + eps)
//   With optional weight decay: g += wd * p before update
// ═══════════════════════════════════════════════════════════════

__global__ void mt_rmsprop_autoclip_kernel(
    float** param_ptrs, float** grad_ptrs,
    float** v_ptrs, float** buf_ptrs,
    int* tensor_sizes, int* d_block_offsets,
    int num_tensors,
    float lr, float alpha, float one_minus_alpha,
    float eps, float weight_decay, float momentum, int has_momentum,
    const float* norm_buf, float max_grad_norm
) {
    float norm_sq = norm_buf[0];
    float clip_coef = (norm_sq > max_grad_norm * max_grad_norm)
        ? max_grad_norm / sqrtf(norm_sq) : 1.0f;

    int flat_block = blockIdx.x;
    int lo = 0, hi = num_tensors - 1, k = 0;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (flat_block >= d_block_offsets[mid]) { k = mid; lo = mid + 1; }
        else { hi = mid - 1; }
    }
    float* param = param_ptrs[k];
    const float* grad = grad_ptrs[k];
    float* v = v_ptrs[k];
    float* buf = buf_ptrs[k];
    int n = tensor_sizes[k];
    int local_block = flat_block - d_block_offsets[k];
    int chunk_start = local_block * blockDim.x;
    if (chunk_start >= n) return;
    int chunk_end = chunk_start + blockDim.x;
    if (chunk_end > n) chunk_end = n;

    for (int i = chunk_start + threadIdx.x; i < chunk_end; i += blockDim.x) {
        float g = grad[i] * clip_coef;
        if (weight_decay > 0.0f) g = g + weight_decay * param[i];

        float vi = alpha * v[i] + one_minus_alpha * g * g;
        v[i] = vi;
        float denom = sqrtf(vi) + eps;

        if (has_momentum) {
            float b = momentum * buf[i] + g / denom;
            buf[i] = b;
            param[i] = param[i] - lr * b;
        } else {
            param[i] = param[i] - lr * g / denom;
        }
    }
}

CUDA_EXPORT int cuda_mt_rmsprop_autoclip(
    float** d_param_ptrs, float** d_grad_ptrs,
    float** d_v_ptrs, float** d_buf_ptrs,
    int* d_sizes, int* d_block_offsets,
    int num_tensors, int total_blocks,
    float lr, float alpha, float one_minus_alpha,
    float eps, float weight_decay, float momentum, int has_momentum,
    float* norm_buf, float max_grad_norm
) {
    if (num_tensors == 0 || total_blocks == 0) return 0;
    mt_rmsprop_autoclip_kernel<<<total_blocks, 256>>>(
        d_param_ptrs, d_grad_ptrs, d_v_ptrs, d_buf_ptrs,
        d_sizes, d_block_offsets, num_tensors,
        lr, alpha, one_minus_alpha,
        eps, weight_decay, momentum, has_momentum,
        norm_buf, max_grad_norm
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    return 0;
}

} // extern "C"
