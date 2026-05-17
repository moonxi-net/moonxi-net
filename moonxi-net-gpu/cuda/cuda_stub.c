// cuda_stub.c — C stub compiled by MoonBit's TCC
// Linux: statically linked, calls CUDA functions directly

#include <moonbit.h>
#include <stdio.h>
#include <time.h>

// CUDA runtime declarations (avoid including cuda_runtime_api.h which TCC can't find)
typedef enum {
  cudaSuccess = 0
} cudaError_t;
extern cudaError_t cudaDeviceSynchronize(void);
extern cudaError_t cudaGetLastError(void);
extern cudaError_t cudaPeekAtLastError(void);
extern const char* cudaGetErrorString(cudaError_t);

// ============================================================================
// Linux: extern declarations for statically linked CUDA functions
// ============================================================================

extern int    cuda_get_device_count(void);
extern int    cuda_get_device_name(char* buf, int buf_len);
extern int    cuda_get_compute_capability(int* major, int* minor);
extern void*  cuda_alloc(int64_t size);
extern void   cuda_free_device(void* ptr);
extern int    cuda_memset_zero(void* dst, int64_t count);
extern int    cuda_copy_h2d(void* dst, const float* src, int64_t count);
extern int    cuda_raw_h2d(void* dst, const void* src, int64_t bytes);
extern int cuda_copy_ptrs_h2d(void* dst, void** src, int32_t count);
extern void* cuda_event_create(void);
extern int cuda_event_record(void* stream, void* event);
extern int cuda_event_elapsed_ms(void* start, void* end, float* ms);
extern int cuda_event_synchronize(void* event);
extern int    cuda_copy_d2h(float* dst, const void* src, int64_t count);
extern int    cuda_copy_h2d_offset(void* dst, const void* src_base, int offset_elements, int count_elements);
extern int    cuda_vec_add(const float* a, const float* b, float* out, int n);
extern int    cuda_check_error(void);
extern void*  cublas_create_handle(void);
extern void   cublas_destroy_handle(void* handle);
extern int    cublas_sgemv(void* handle, int m, int n, const float* alpha,
                           const float* d_A, const float* d_x,
                           const float* beta, float* d_y);
extern int    cublas_sgemm(void* handle, int m, int n, int k,
                           const float* alpha, const float* d_A,
                           const float* d_B, const float* beta, float* d_C);
extern void*  cudnn_create_handle(void);
extern void   cudnn_destroy_handle(void* handle);
extern void*  conv2d_create(int n, int c, int h, int w, int out_c, int kh, int kw,
                            int pad_h, int pad_w, int stride_h, int stride_w,
                            int dilation_h, int dilation_w);
extern void   conv2d_destroy(void* ctx);
extern int    conv2d_forward(void* ctx, void* cudnn_handle, void* input, void* weight, void* bias, void* output);
extern int    conv2d_forward_bias_relu(void* ctx, void* cudnn_handle, void* input, void* weight, void* bias, void* output);
extern int    conv2d_backward_data(void* ctx, void* cudnn_handle, void* grad_output, void* weight, void* grad_input);
extern int    conv2d_backward_filter(void* ctx, void* cudnn_handle, void* input, void* grad_output, void* grad_weight);
extern int    conv2d_backward_bias(void* cudnn_handle, void* grad_output, void* grad_bias, int batch, int out_c, int out_h, int out_w);
extern void*  batchnorm_create(int n, int c, int h, int w);
extern void   batchnorm_destroy(void* ctx);
extern int    batchnorm_forward(void* ctx, void* cudnn_handle, void* input, void* output,
                                void* bn_scale, void* bn_bias,
                                void* bn_running_mean, void* bn_running_var,
                                void* bn_save_mean, void* bn_save_inv_var,
                                float momentum, float epsilon, int is_training);
extern int    cuda_batchnorm_inference(const float* input, float* output,
                                        const float* gamma, const float* beta,
                                        const float* running_mean, const float* running_var,
                                        float eps, int n, int c, int hw);
extern int    batchnorm_backward(void* ctx, void* cudnn_handle,
                                 void* input, void* grad_output, void* grad_input,
                                 void* bn_scale, void* grad_bn_scale, void* grad_bn_bias,
                                 void* bn_save_mean, void* bn_save_inv_var,
                                 float epsilon);
extern int    cudnn_relu_forward(void* cudnn_handle, int64_t n_elements, void* input, void* output);
extern int    cudnn_relu_backward(void* cudnn_handle, int64_t n_elements, void* input, void* grad_output, void* grad_input);
extern void*  pool2d_create(int n, int c, int h, int w, int kh, int kw,
                             int pad_h, int pad_w, int stride_h, int stride_w, int pool_type);
extern void   pool2d_destroy(void* ctx);
extern int    pool2d_forward(void* ctx, void* cudnn_handle, void* input, void* output);
extern int    pool2d_backward(void* ctx, void* cudnn_handle, void* input, void* grad_output, void* grad_input);
extern void*  pool_create(int64_t capacity);
extern void   pool_destroy(void* pool);
extern void*  pool_alloc(void* pool, int64_t size, int64_t alignment);
extern void   pool_reset(void* pool);
extern int64_t pool_used(void* pool);
extern int64_t pool_peak(void* pool);
extern int64_t pool_capacity(void* pool);
extern int cuda_transpose_nchw_to_nhwc(float* src, float* dst, int N, int C, int H, int W);
extern int cuda_transpose_nhwc_to_nchw(float* src, float* dst, int N, int C, int H, int W);
extern int cuda_bias_add_nchw(float* data, float* bias, int N, int C, int HW);
extern int cuda_elementwise_add(float* a, float* b, float* out, int n);
extern int cuda_elementwise_add_into(float* dst, const float* src, int n);
extern int cuda_multi_tensor_norm_sq(float** tensor_ptrs, int* tensor_sizes, int num_tensors, float* dst);
extern int cuda_multi_tensor_sgd_momentum_step(
    float** param_ptrs, float** grad_ptrs, float** velocity_ptrs,
    int* tensor_sizes, int num_tensors,
    float lr, float momentum, float weight_decay, float clip_coef
);
extern int cuda_multi_tensor_sgd_momentum_step_prealloc(
    float** param_ptrs, float** grad_ptrs, float** velocity_ptrs,
    int* tensor_sizes, int num_tensors,
    float lr, float momentum, float weight_decay, float clip_coef,
    void* d_param_ptrs, void* d_grad_ptrs, void* d_vel_ptrs, void* d_sizes
);
extern int cuda_multi_tensor_sgd_momentum_step_gpu_only(
    void* d_param_ptrs, void* d_grad_ptrs, void* d_vel_ptrs, void* d_sizes,
    int num_tensors,
    float lr, float momentum, float weight_decay, float clip_coef
);
extern int cuda_multi_tensor_norm_sq_gpu_only(
    void* d_ptrs, void* d_sizes, int num_tensors, void* dst
);
extern int cuda_mt_norm_sq_gpu_only(
    void* d_ptrs, void* d_sizes, void* d_block_offsets,
    int num_tensors, int total_blocks, void* dst
);
extern int cuda_mt_sgd_momentum_gpu_only(
    void* d_param_ptrs, void* d_grad_ptrs, void* d_vel_ptrs,
    void* d_sizes, void* d_block_offsets,
    int num_tensors, int total_blocks,
    float lr, float momentum, float weight_decay, float clip_coef
);
extern int cuda_mt_sgd_momentum_autoclip(
    void* d_param_ptrs, void* d_grad_ptrs, void* d_vel_ptrs,
    void* d_sizes, void* d_block_offsets,
    int num_tensors, int total_blocks,
    float lr, float momentum, float weight_decay,
    void* norm_buf, float max_grad_norm
);
extern int cuda_mt_adam_autoclip(
    void* d_param_ptrs, void* d_grad_ptrs,
    void* d_m_ptrs, void* d_v_ptrs,
    void* d_sizes, void* d_block_offsets,
    int num_tensors, int total_blocks,
    float lr, float beta1, float beta2,
    float one_minus_beta1, float one_minus_beta2,
    float bias_correction1, float bias_correction2,
    float eps, float weight_decay,
    void* norm_buf, float max_grad_norm
);
extern int cuda_mt_rmsprop_autoclip(
    void* d_param_ptrs, void* d_grad_ptrs,
    void* d_v_ptrs, void* d_buf_ptrs,
    void* d_sizes, void* d_block_offsets,
    int num_tensors, int total_blocks,
    float lr, float alpha, float one_minus_alpha,
    float eps, float weight_decay, float momentum, int has_momentum,
    void* norm_buf, float max_grad_norm
);
extern int cuda_contiguous_sgd_momentum_step(
    float* params, const float* grads, float* vels,
    int total_elements,
    float lr, float momentum, float weight_decay, float clip_coef
);
extern int cuda_gather(
    float** d_src_ptrs, const int* d_offsets, const int* d_sizes,
    int num_tensors, float* dst
);
extern int cuda_scatter(
    float* src, float** d_dst_ptrs, const int* d_offsets, const int* d_sizes,
    int num_tensors
);
extern int cuda_contiguous_norm_sq(
    const float* data, int n, float* out
);
extern int cuda_reduce_sum_spatial(const float* input, float* output, int N, int C, int H, int W);
extern int cuda_reduce_sum_leading(const float* input, float* output, int output_numel, int lead_stride);
extern int cuda_reduce_sum_dim(const float* input, float* output, int output_numel, int dim_size, int stride_before, int stride_after);
extern int cuda_adaptive_avg_pool2d_forward(const float* input, float* output, int N, int C, int H, int W);
extern int cuda_adaptive_avg_pool2d_backward(const float* dy, float* d_input, int N, int C, int H, int W);
extern int cuda_softmax_ce_forward(const float* logits, const float* targets, float* loss_output, float* softmax_output, int batch, int num_classes);
extern int cuda_softmax_ce_backward(const float* logits, const float* targets, float* output, int batch, int num_classes);
extern int cuda_softmax_ce_forward_labels(const float* logits, const float* labels, float* loss_output, int batch, int num_classes);
extern int cuda_softmax_ce_backward_labels(const float* logits, const float* labels, float* output, int batch, int num_classes);
extern int cuda_matmul(void* cublas_handle, float* a, float* b, float* c, int M, int K, int N, int transpose_a, int transpose_b, float alpha, float beta);
extern int cuda_accumulate_inplace(float* dst, const float* src, int n);
extern int cuda_sub(const float* a, const float* b, float* out, int n);
extern int cuda_mul_elem(const float* a, const float* b, float* out, int n);
extern int cuda_square(const float* src, float* out, int n);
extern int cuda_sqrt(const float* src, float* out, int n);
extern int cuda_div_elem(const float* a, const float* b, float* out, int n);
extern int cuda_fill(float* out, int n, float value);
extern int cuda_scale_out(const float* src, float* out, int n, float scale);
extern int cuda_transpose_2d(const float* in, float* out, int n, int m);
extern int    cuda_sgd_momentum_step(void* param, void* grad, void* velocity, int n, float lr, float momentum, float weight_decay, float clip_coef);
extern int    cuda_sgd_step(void* param, void* grad, int n, float lr, float weight_decay);
extern int    cuda_norm_sq(void* src, int n, void* dst);
extern int    cuda_scale_inplace(void* data, int n, float scale);
extern float cuda_read_scalar(void* src);
extern int    cuda_axpy(int n, float alpha, void* x, void* y);
extern void   conv2d_cache_clear(void);
extern int    conv2d_cache_size(void);
extern void   batchnorm_cache_clear(void);
extern int    batchnorm_cache_size(void);
extern void   pool2d_cache_clear(void);
extern int    pool2d_cache_size(void);

// ============================================================================
// CudaPtr — opaque GPU memory pointer (kept for backward compat if needed)
// ============================================================================

typedef struct {
  void* ptr;
} MoonbitCudaPtr;

static void moonbit_noop_finalize(void* obj) {
  (void)obj;
}

static void cuda_ptr_finalize(void* obj) {
  MoonbitCudaPtr* p = (MoonbitCudaPtr*)obj;
  if (p->ptr) {
    cuda_free_device(p->ptr);
    p->ptr = NULL;
  }
}

MOONBIT_FFI_EXPORT
void* moonbit_cuda_ptr_raw(MoonbitCudaPtr* p) {
  return p->ptr;
}

// ============================================================================
// Alloc — returns GPU pointer as void*
// ============================================================================

MOONBIT_FFI_EXPORT
void* moonbit_cuda_alloc(int64_t size) {
  return cuda_alloc(size);
}

// ============================================================================
// CublasHandle — returns void*
// ============================================================================

typedef struct {
  void* handle;
} MoonbitCublasHandle;

MOONBIT_FFI_EXPORT
void* moonbit_cublas_create() {
  return cublas_create_handle();
}

MOONBIT_FFI_EXPORT
void moonbit_cublas_destroy(void* handle) {
  if (handle) {
    cublas_destroy_handle(handle);
  }
}

// ============================================================================
// GPU Memory Pool — uses void* for pool handle
// ============================================================================

MOONBIT_FFI_EXPORT
void* moonbit_pool_create(int64_t capacity) {
  return pool_create(capacity);
}

MOONBIT_FFI_EXPORT
void* moonbit_pool_alloc(void* pool, int64_t size) {
  if (!pool) return NULL;
  return pool_alloc(pool, size, 256);
}

MOONBIT_FFI_EXPORT
void moonbit_pool_reset(void* pool) {
  if (!pool) return;
  // No cudaDeviceSynchronize() needed — all ops are on the default stream,
  // so CUDA guarantees ordering: next batch's kernels automatically wait
  // for previous batch's kernels to complete before using recycled memory.
  pool_reset(pool);
}

MOONBIT_FFI_EXPORT
int64_t moonbit_pool_used(void* pool) {
  if (!pool) return 0;
  return pool_used(pool);
}

MOONBIT_FFI_EXPORT
int64_t moonbit_pool_peak(void* pool) {
  if (!pool) return 0;
  return pool_peak(pool);
}

MOONBIT_FFI_EXPORT
int64_t moonbit_pool_capacity(void* pool) {
  if (!pool) return 0;
  return pool_capacity(pool);
}

// ============================================================================
// CUDA Runtime info functions
// ============================================================================

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_device_count() {
  return cuda_get_device_count();
}

MOONBIT_FFI_EXPORT
moonbit_bytes_t moonbit_cuda_device_name() {
  char buf[256] = {0};
  cuda_get_device_name(buf, sizeof(buf));
  int len = 0;
  while (buf[len] && len < 255) len++;
  moonbit_bytes_t bytes = moonbit_make_bytes(len, 0);
  for (int i = 0; i < len; i++) {
    bytes[i] = (uint8_t)buf[i];
  }
  return bytes;
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_compute_capability_major() {
  int major = 0, minor = 0;
  cuda_get_compute_capability(&major, &minor);
  return major;
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_compute_capability_minor() {
  int major = 0, minor = 0;
  cuda_get_compute_capability(&major, &minor);
  return minor;
}

// ============================================================================
// Memory operations — all data pointers are void*
// ============================================================================

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_copy_h2d(void* dst, float* src, int64_t count) {
  return cuda_copy_h2d(dst, src, count);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_raw_h2d(void* dst, void* src, int64_t bytes) {
  return cuda_raw_h2d(dst, src, bytes);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_copy_ptrs_h2d(void* dst, void** src, int32_t count) {
  return cuda_copy_ptrs_h2d(dst, src, count);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_copy_d2h(float* dst, void* src, int64_t count) {
  return cuda_copy_d2h(dst, src, count);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_copy_h2d_offset(void* dst, void* src_base, int32_t offset_elements, int32_t count_elements) {
    return cuda_copy_h2d_offset(dst, src_base, offset_elements, count_elements);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_memset_zero(void* dst, int64_t count) {
    return (int32_t)cuda_memset_zero(dst, count);
}

// ============================================================================
// CUDA Error Check
// ============================================================================

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_check_error() {
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    return (int32_t)err;
  }
  err = cudaGetLastError();
  return (int32_t)err;
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_peek_error() {
  cudaError_t err = cudaPeekAtLastError();
  return (int32_t)err;
}

// ============================================================================
// Vector add
// ============================================================================

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_vec_add(void* a, void* b, void* out, int32_t n) {
  return cuda_vec_add(
    (const float*)a,
    (const float*)b,
    (float*)out,
    n
  );
}

// ============================================================================
// cuBLAS operations — handle is now void*
// ============================================================================

MOONBIT_FFI_EXPORT
int32_t moonbit_cublas_sgemv(
  void* handle,
  int32_t m, int32_t n,
  float alpha,
  void* d_A,
  void* d_x,
  float beta,
  void* d_y
) {
  return cublas_sgemv(
    handle, m, n, &alpha,
    (const float*)d_A,
    (const float*)d_x,
    &beta,
    (float*)d_y
  );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cublas_sgemm(
  void* handle,
  int32_t m, int32_t n, int32_t k,
  float alpha,
  void* d_A,
  void* d_B,
  float beta,
  void* d_C
) {
  return cublas_sgemm(
    handle, m, n, k, &alpha,
    (const float*)d_A,
    (const float*)d_B,
    &beta,
    (float*)d_C
  );
}

// ============================================================================
// CudnnHandle — returns void*
// ============================================================================

typedef struct {
  void* handle;
} MoonbitCudnnHandle;

MOONBIT_FFI_EXPORT
void* moonbit_cudnn_create() {
  return cudnn_create_handle();
}

MOONBIT_FFI_EXPORT
void moonbit_cudnn_destroy(void* handle) {
  if (handle) {
    cudnn_destroy_handle(handle);
  }
}

// ============================================================================
// Conv2dContext — returns void*
// ============================================================================

typedef struct {
  void* ctx;
} MoonbitConv2dContext;

MOONBIT_FFI_EXPORT
void* moonbit_conv2d_create(
  int n, int c, int h, int w,
  int out_c, int kh, int kw,
  int pad_h, int pad_w,
  int stride_h, int stride_w,
  int dilation_h, int dilation_w
) {
  return conv2d_create(n, c, h, w, out_c, kh, kw, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_conv2d_forward(
  void* ctx,
  void* cudnn_handle,
  void* input,
  void* weight,
  void* bias,
  void* output
) {
  return conv2d_forward(ctx, cudnn_handle, input, weight, bias, output);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_conv2d_forward_bias_relu(
  void* ctx,
  void* cudnn_handle,
  void* input,
  void* weight,
  void* bias,
  void* output
) {
  return conv2d_forward_bias_relu(ctx, cudnn_handle, input, weight, bias, output);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_conv2d_backward_data(
  void* ctx,
  void* cudnn_handle,
  void* grad_output,
  void* weight,
  void* grad_input
) {
  return conv2d_backward_data(ctx, cudnn_handle, grad_output, weight, grad_input);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_conv2d_backward_filter(
  void* ctx,
  void* cudnn_handle,
  void* input,
  void* grad_output,
  void* grad_weight
) {
  return conv2d_backward_filter(ctx, cudnn_handle, input, grad_output, grad_weight);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_conv2d_backward_bias(
  void* cudnn_handle,
  void* grad_output,
  void* grad_bias,
  int batch, int out_c, int out_h, int out_w
) {
  return conv2d_backward_bias(cudnn_handle, grad_output, grad_bias, batch, out_c, out_h, out_w);
}

MOONBIT_FFI_EXPORT
void moonbit_conv2d_destroy(void* ctx) {
  if (!ctx) return;
  conv2d_destroy(ctx);
}

// ============================================================================
// BatchNormContext — returns void*
// ============================================================================

typedef struct {
  void* ctx;
} MoonbitBatchNormContext;

MOONBIT_FFI_EXPORT
void* moonbit_batchnorm_create(int n, int c, int h, int w) {
  return batchnorm_create(n, c, h, w);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_batchnorm_forward(
  void* ctx,
  void* cudnn_handle,
  void* input,
  void* output,
  void* bn_scale,
  void* bn_bias,
  void* bn_running_mean,
  void* bn_running_var,
  void* bn_save_mean,
  void* bn_save_inv_var,
  float momentum,
  float epsilon,
  int is_training
) {
  return batchnorm_forward(ctx, cudnn_handle,
                           input,
                           output,
                           bn_scale,
                           bn_bias,
                           bn_running_mean,
                           bn_running_var,
                           bn_save_mean,
                           bn_save_inv_var,
                            momentum, epsilon, is_training);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_batchnorm_inference(
  void* input,
  void* output,
  void* gamma,
  void* beta,
  void* running_mean,
  void* running_var,
  float epsilon,
  int n,
  int c,
  int hw
) {
  return cuda_batchnorm_inference(input, output, gamma, beta,
                                   running_mean, running_var,
                                   epsilon, n, c, hw);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_batchnorm_backward(
  void* ctx,
  void* cudnn_handle,
  void* input,
  void* grad_output,
  void* grad_input,
  void* bn_scale,
  void* grad_bn_scale,
  void* grad_bn_bias,
  void* bn_save_mean,
  void* bn_save_inv_var,
  float epsilon
) {
  return batchnorm_backward(ctx, cudnn_handle,
                            input,
                            grad_output,
                            grad_input,
                            bn_scale,
                            grad_bn_scale,
                            grad_bn_bias,
                            bn_save_mean,
                            bn_save_inv_var,
                            epsilon);
}

MOONBIT_FFI_EXPORT
void moonbit_batchnorm_destroy(void* ctx) {
  if (!ctx) return;
  batchnorm_destroy(ctx);
}

// ============================================================================
// ReLU via cuDNN activation — handle is now void*
// ============================================================================

MOONBIT_FFI_EXPORT
int32_t moonbit_cudnn_relu_forward(
  void* handle,
  int64_t n_elements,
  void* input,
  void* output
) {
  return cudnn_relu_forward(handle, n_elements, input, output);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cudnn_relu_backward(
  void* handle,
  int64_t n_elements,
  void* input,
  void* grad_output,
  void* grad_input
) {
  return cudnn_relu_backward(handle, n_elements, input, grad_output, grad_input);
}

// ============================================================================
// Pool2dContext — returns void*
// ============================================================================

typedef struct {
  void* ctx;
} MoonbitPool2dContext;

MOONBIT_FFI_EXPORT
void* moonbit_pool2d_create(
  int n, int c, int h, int w,
  int kh, int kw,
  int pad_h, int pad_w,
  int stride_h, int stride_w,
  int pool_type
) {
  return pool2d_create(n, c, h, w, kh, kw, pad_h, pad_w, stride_h, stride_w, pool_type);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_pool2d_forward(
  void* ctx,
  void* cudnn_handle,
  void* input,
  void* output
) {
  return pool2d_forward(ctx, cudnn_handle, input, output);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_pool2d_backward(
  void* ctx,
  void* cudnn_handle,
  void* input,
  void* grad_output,
  void* grad_input
) {
  return pool2d_backward(ctx, cudnn_handle, input, grad_output, grad_input);
}

MOONBIT_FFI_EXPORT
void moonbit_pool2d_destroy(void* ctx) {
  if (!ctx) return;
  pool2d_destroy(ctx);
}

// ============================================================================
// Cache management
// ============================================================================

MOONBIT_FFI_EXPORT
void moonbit_conv2d_cache_clear(void) {
  conv2d_cache_clear();
}

MOONBIT_FFI_EXPORT
int32_t moonbit_conv2d_cache_size(void) {
  return conv2d_cache_size();
}

MOONBIT_FFI_EXPORT
void moonbit_batchnorm_cache_clear(void) {
  batchnorm_cache_clear();
}

MOONBIT_FFI_EXPORT
int32_t moonbit_batchnorm_cache_size(void) {
  return batchnorm_cache_size();
}

MOONBIT_FFI_EXPORT
void moonbit_pool2d_cache_clear(void) {
  pool2d_cache_clear();
}

MOONBIT_FFI_EXPORT
int32_t moonbit_pool2d_cache_size(void) {
  return pool2d_cache_size();
}

// ============================================================================
// Transpose and elementwise kernels
// ============================================================================

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_transpose_nchw_to_nhwc(
  void* src,
  void* dst,
  int32_t N, int32_t C, int32_t H, int32_t W
) {
  return cuda_transpose_nchw_to_nhwc(
    (float*)src,
    (float*)dst,
    N, C, H, W
  );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_transpose_nhwc_to_nchw(
  void* src,
  void* dst,
  int32_t N, int32_t C, int32_t H, int32_t W
) {
  return cuda_transpose_nhwc_to_nchw(
    (float*)src,
    (float*)dst,
    N, C, H, W
  );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_bias_add_nchw(
  void* data,
  void* bias,
  int32_t N, int32_t C, int32_t HW
) {
  return cuda_bias_add_nchw(
    (float*)data,
    (float*)bias,
    N, C, HW
  );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_elementwise_add(
  void* a,
  void* b,
  void* out,
  int32_t n
) {
  if (!a || !b || !out) return -1;
  return cuda_elementwise_add(
    (float*)a,
    (float*)b,
    (float*)out,
    n
  );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_elementwise_add_into(
  void* dst,
  void* src,
  int32_t n
) {
  if (!dst || !src) return -1;
  return cuda_elementwise_add_into(
    (float*)dst,
    (const float*)src,
    n
  );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_reduce_sum_spatial(
  void* input,
  void* output,
  int32_t N, int32_t C, int32_t H, int32_t W
) {
  return cuda_reduce_sum_spatial(
    (const float*)input,
    (float*)output,
    N, C, H, W
  );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_reduce_sum_leading(
  void* input,
  void* output,
  int32_t output_numel,
  int32_t lead_stride
) {
  return cuda_reduce_sum_leading(
    (const float*)input,
    (float*)output,
    output_numel,
    lead_stride
  );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_reduce_sum_dim(
  void* input,
  void* output,
  int32_t output_numel,
  int32_t dim_size,
  int32_t stride_before,
  int32_t stride_after
) {
  return cuda_reduce_sum_dim(
    (const float*)input,
    (float*)output,
    output_numel,
    dim_size,
    stride_before,
    stride_after
  );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_adaptive_avg_pool2d_forward(
  void* input,
  void* output,
  int32_t N, int32_t C, int32_t H, int32_t W
) {
  return cuda_adaptive_avg_pool2d_forward(
    (const float*)input,
    (float*)output,
    N, C, H, W
  );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_adaptive_avg_pool2d_backward(
  void* dy,
  void* d_input,
  int32_t N, int32_t C, int32_t H, int32_t W
) {
  return cuda_adaptive_avg_pool2d_backward(
    (float*)dy,
    (float*)d_input,
    N, C, H, W
  );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_softmax_ce_forward(
  void* logits,
  void* targets,
  void* loss_output,
  void* softmax_output,
  int32_t batch,
  int32_t num_classes
) {
  return cuda_softmax_ce_forward(
    (const float*)logits,
    (const float*)targets,
    (float*)loss_output,
    (float*)softmax_output,
    batch, num_classes
  );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_softmax_ce_backward(
  void* logits,
  void* targets,
  void* output,
  int32_t batch,
  int32_t num_classes
) {
  return cuda_softmax_ce_backward(
    (const float*)logits,
    (const float*)targets,
    (float*)output,
    batch, num_classes
  );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_softmax_ce_forward_labels(
  void* logits,
  void* labels,
  void* loss_output,
  int32_t batch,
  int32_t num_classes
) {
  return cuda_softmax_ce_forward_labels(
    (const float*)logits,
    (const float*)labels,
    (float*)loss_output,
    batch, num_classes
  );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_softmax_ce_backward_labels(
  void* logits,
  void* labels,
  void* output,
  int32_t batch,
  int32_t num_classes
) {
  return cuda_softmax_ce_backward_labels(
    (const float*)logits,
    (const float*)labels,
    (float*)output,
    batch, num_classes
  );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_matmul(
  void* handle,
  void* a,
  void* b,
  void* c,
  int32_t M, int32_t K, int32_t N,
  int32_t transpose_a,
  int32_t transpose_b,
  float alpha,
  float beta
) {
  return cuda_matmul(
    handle,
    (float*)a,
    (float*)b,
    (float*)c,
    M, K, N,
    transpose_a, transpose_b,
    alpha, beta
  );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_accumulate_inplace(
  void* dst,
  void* src,
  int32_t n
) {
  if (!dst || !src) return -1;
  return cuda_accumulate_inplace((float*)dst, (const float*)src, n);
}

// ============================================================================
// GPU Optimizer Kernels
// ============================================================================

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_sgd_momentum_step(
    void* param, void* grad, void* velocity,
    int32_t n, float lr, float momentum, float weight_decay, float clip_coef
) {
    return cuda_sgd_momentum_step(param, grad, velocity, n, lr, momentum, weight_decay, clip_coef);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_sgd_step(
    void* param, void* grad,
    int32_t n, float lr, float weight_decay
) {
    return cuda_sgd_step(param, grad, n, lr, weight_decay);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_norm_sq(void* src, int32_t n, void* dst) {
    return cuda_norm_sq(src, n, dst);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_scale_inplace(void* data, int32_t n, float scale) {
  return cuda_scale_inplace(data, n, scale);
}

int32_t moonbit_cuda_fill(void* out, int32_t n, float value) {
  return cuda_fill(out, n, value);
}

MOONBIT_FFI_EXPORT
float moonbit_cuda_read_scalar(void* src) {
    return cuda_read_scalar(src);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_axpy(int32_t n, float alpha, void* x, void* y) {
    return cuda_axpy(n, alpha, x, y);
}

// === Elementwise GPU kernels ===

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_sub(void* a, void* b, void* out, int32_t n) {
    return cuda_sub((const float*)a, (const float*)b, (float*)out, n);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_mul_elem(void* a, void* b, void* out, int32_t n) {
    return cuda_mul_elem((const float*)a, (const float*)b, (float*)out, n);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_square(void* src, void* out, int32_t n) {
    return cuda_square((const float*)src, (float*)out, n);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_sqrt(void* src, void* out, int32_t n) {
    return cuda_sqrt((const float*)src, (float*)out, n);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_div_elem(void* a, void* b, void* out, int32_t n) {
    return cuda_div_elem((const float*)a, (const float*)b, (float*)out, n);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_scale_out(void* src, void* out, int32_t n, float scale) {
    return cuda_scale_out((const float*)src, (float*)out, n, scale);
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_transpose_2d(void* in, void* out, int32_t n, int32_t m) {
    return cuda_transpose_2d((const float*)in, (float*)out, n, m);
}

// === Timer Utility ===

MOONBIT_FFI_EXPORT
double moonbit_get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

MOONBIT_FFI_EXPORT
void moonbit_cuda_sync(void) {
  cudaDeviceSynchronize();
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_multi_tensor_norm_sq(
    void** tensor_ptrs, int32_t* tensor_sizes, int32_t num_tensors, void* dst
) {
    return cuda_multi_tensor_norm_sq(
        (float**)tensor_ptrs, tensor_sizes, num_tensors, (float*)dst
    );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_multi_tensor_norm_sq_gpu_only(
    void* d_ptrs, void* d_sizes, int32_t num_tensors, void* dst
) {
    return cuda_multi_tensor_norm_sq_gpu_only(
        (float**)d_ptrs, (int*)d_sizes, num_tensors, (float*)dst
    );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_mt_norm_sq_gpu_only(
    void* d_ptrs, void* d_sizes, void* d_block_offsets,
    int32_t num_tensors, int32_t total_blocks, void* dst
) {
    return cuda_mt_norm_sq_gpu_only(
        (float**)d_ptrs, (int*)d_sizes, (int*)d_block_offsets,
        num_tensors, total_blocks, (float*)dst
    );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_mt_sgd_momentum_gpu_only(
    void* d_param_ptrs, void* d_grad_ptrs, void* d_vel_ptrs,
    void* d_sizes, void* d_block_offsets,
    int32_t num_tensors, int32_t total_blocks,
    float lr, float momentum, float weight_decay, float clip_coef
) {
    return cuda_mt_sgd_momentum_gpu_only(
        (float**)d_param_ptrs, (float**)d_grad_ptrs, (float**)d_vel_ptrs,
        (int*)d_sizes, (int*)d_block_offsets,
        num_tensors, total_blocks,
        lr, momentum, weight_decay, clip_coef
    );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_mt_sgd_momentum_autoclip(
    void* d_param_ptrs, void* d_grad_ptrs, void* d_vel_ptrs,
    void* d_sizes, void* d_block_offsets,
    int32_t num_tensors, int32_t total_blocks,
    float lr, float momentum, float weight_decay,
    void* norm_buf, float max_grad_norm
) {
    return cuda_mt_sgd_momentum_autoclip(
        (float**)d_param_ptrs, (float**)d_grad_ptrs, (float**)d_vel_ptrs,
        (int*)d_sizes, (int*)d_block_offsets,
        num_tensors, total_blocks,
        lr, momentum, weight_decay,
        (float*)norm_buf, max_grad_norm
    );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_multi_tensor_sgd_momentum_step(
    void** param_ptrs, void** grad_ptrs, void** velocity_ptrs,
    int32_t* tensor_sizes, int32_t num_tensors,
    float lr, float momentum, float weight_decay, float clip_coef
) {
    return cuda_multi_tensor_sgd_momentum_step(
        (float**)param_ptrs, (float**)grad_ptrs, (float**)velocity_ptrs,
        tensor_sizes, num_tensors,
        lr, momentum, weight_decay, clip_coef
    );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_multi_tensor_sgd_momentum_step_prealloc(
    void** param_ptrs, void** grad_ptrs, void** velocity_ptrs,
    int32_t* tensor_sizes, int32_t num_tensors,
    float lr, float momentum, float weight_decay, float clip_coef,
    void* d_param_ptrs, void* d_grad_ptrs, void* d_vel_ptrs, void* d_sizes
) {
    return cuda_multi_tensor_sgd_momentum_step_prealloc(
        (float**)param_ptrs, (float**)grad_ptrs, (float**)velocity_ptrs,
        tensor_sizes, num_tensors,
        lr, momentum, weight_decay, clip_coef,
        (float**)d_param_ptrs, (float**)d_grad_ptrs, (float**)d_vel_ptrs, (int*)d_sizes
    );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_multi_tensor_sgd_momentum_step_gpu_only(
    void* d_param_ptrs, void* d_grad_ptrs, void* d_vel_ptrs, void* d_sizes,
    int32_t num_tensors,
    float lr, float momentum, float weight_decay, float clip_coef
) {
    return cuda_multi_tensor_sgd_momentum_step_gpu_only(
        (float**)d_param_ptrs, (float**)d_grad_ptrs, (float**)d_vel_ptrs, (int*)d_sizes,
        num_tensors,
        lr, momentum, weight_decay, clip_coef
    );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_contiguous_sgd_momentum_step(
    void* params, void* grads, void* vels,
    int32_t total_elements,
    float lr, float momentum, float weight_decay, float clip_coef
) {
    return cuda_contiguous_sgd_momentum_step(
        (float*)params, (const float*)grads, (float*)vels,
        total_elements, lr, momentum, weight_decay, clip_coef
    );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_gather(
    void** d_src_ptrs, void* d_offsets, void* d_sizes,
    int32_t num_tensors, void* dst
) {
    return cuda_gather(
        (float**)d_src_ptrs, (const int*)d_offsets, (const int*)d_sizes,
        num_tensors, (float*)dst
    );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_scatter(
    void* src, void** d_dst_ptrs, void* d_offsets, void* d_sizes,
    int32_t num_tensors
) {
    return cuda_scatter(
        (float*)src, (float**)d_dst_ptrs, (const int*)d_offsets, (const int*)d_sizes,
        num_tensors
    );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_contiguous_norm_sq(
    void* data, int32_t n, void* out
) {
    return cuda_contiguous_norm_sq((const float*)data, n, (float*)out);
}

MOONBIT_FFI_EXPORT
float moonbit_cuda_test_ptr_array_read(void** ptrs, int32_t n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float val;
        cuda_copy_d2h(&val, (const void*)ptrs[i], 1);
        sum += val;
    }
    return sum;
}

MOONBIT_FFI_EXPORT
void* moonbit_cuda_event_create(void) {
    return cuda_event_create();
}

MOONBIT_FFI_EXPORT
void* moonbit_cuda_null_buffer(void) {
    return NULL;
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_event_record(void* stream, void* event) {
    return cuda_event_record(stream, event);
}

MOONBIT_FFI_EXPORT
float moonbit_cuda_event_elapsed_ms(void* start, void* end) {
    float ms = 0.0f;
    int32_t err = cuda_event_elapsed_ms(start, end, &ms);
    if (err != 0) return -1.0f;
    return ms;
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_event_synchronize(void* event) {
    return cuda_event_synchronize(event);
}

extern int cuda_batch_gather(
    const float* src, const int* indices, float* dst,
    int stride, int batch_size
);

extern int cuda_argmax(
    const float* input, float* output,
    int outer, int inner
);

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_batch_gather(
    void* src, void* indices, void* dst,
    int32_t stride, int32_t batch_size
) {
    return cuda_batch_gather(
        (const float*)src, (const int*)indices, (float*)dst,
        stride, batch_size
    );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_argmax(
    void* input, void* output,
    int32_t outer, int32_t inner
) {
    return cuda_argmax(
        (const float*)input, (float*)output,
        outer, inner
    );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_mt_adam_autoclip(
    void* d_param_ptrs, void* d_grad_ptrs,
    void* d_m_ptrs, void* d_v_ptrs,
    void* d_sizes, void* d_block_offsets,
    int32_t num_tensors, int32_t total_blocks,
    float lr, float beta1, float beta2,
    float one_minus_beta1, float one_minus_beta2,
    float bias_correction1, float bias_correction2,
    float eps, float weight_decay,
    void* norm_buf, float max_grad_norm
) {
    return cuda_mt_adam_autoclip(
        (float**)d_param_ptrs, (float**)d_grad_ptrs,
        (float**)d_m_ptrs, (float**)d_v_ptrs,
        (int*)d_sizes, (int*)d_block_offsets,
        num_tensors, total_blocks,
        lr, beta1, beta2, one_minus_beta1, one_minus_beta2,
        bias_correction1, bias_correction2, eps, weight_decay,
        (float*)norm_buf, max_grad_norm
    );
}

MOONBIT_FFI_EXPORT
int32_t moonbit_cuda_mt_rmsprop_autoclip(
    void* d_param_ptrs, void* d_grad_ptrs,
    void* d_v_ptrs, void* d_buf_ptrs,
    void* d_sizes, void* d_block_offsets,
    int32_t num_tensors, int32_t total_blocks,
    float lr, float alpha, float one_minus_alpha,
    float eps, float weight_decay, float momentum, int32_t has_momentum,
    void* norm_buf, float max_grad_norm
) {
    return cuda_mt_rmsprop_autoclip(
        (float**)d_param_ptrs, (float**)d_grad_ptrs,
        (float**)d_v_ptrs, (float**)d_buf_ptrs,
        (int*)d_sizes, (int*)d_block_offsets,
        num_tensors, total_blocks,
        lr, alpha, one_minus_alpha,
        eps, weight_decay, momentum, has_momentum,
        (float*)norm_buf, max_grad_norm
    );
}