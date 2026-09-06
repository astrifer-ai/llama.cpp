#include "common.cuh"

#define CUDA_SIGMOID_SCALE_BLOCK_SIZE 256

void ggml_cuda_op_scale_sigmoid_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * src);
