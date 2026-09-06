#include "sigmoidscale.cuh"
#include "unary.cuh"

// dst[i] = s_out * sigmoid(s_in * x[i])
//
// Fuses GGML_OP_SCALE + GGML_UNARY_OP_SIGMOID + GGML_OP_SCALE. qwen4exp's build_hc_combine
// writes its scatter weight as 2*sigmoid(inject/hc), and inject is [hc, n_tokens] -- 16
// floats at 4-user decode. Three kernels of ~1.0 us each, twice per layer, 96 times per
// decode step, is 288 dispatches carrying ~1.0 ms of launch overhead for ~0 work.
//
// ===========================================================================================
// THE ARITHMETIC CONTRACT
//
// This must be BIT-IDENTICAL to the chain it replaces. The chain rounds three times: the
// first scale, the sigmoid, the second scale. This kernel performs the same three operations
// in the same order on the same f32 values, so the rounding structure already matches -- but
// only if the compiler does not fuse any of them. There is no multiply-add here for
// -ffp-contract=fast to contract (multiply, unary, multiply), which is why this op is safer
// than GGML_OP_MUL_COLLAPSE was; the pragma below is a guard, not a fix, and it must stay.
// See the contraction note in mulcollapse.cu for what happens when that guard is missing:
// 42.9 % of elements differed and test-backend-ops, a tolerance gate, passed anyway.
//
// The sigmoid itself comes from ggml_cuda_op_sigmoid_single() in unary.cuh, which is also
// what the unfused GGML_UNARY_OP_SIGMOID path calls. Do not re-spell the expression here:
// one definition is what keeps the two paths bit-identical.
// ===========================================================================================
#if defined(__clang__)
#  define GGML_SIGMOID_SCALE_NO_CONTRACT _Pragma("clang fp contract(off)")
#else
#  error "sigmoidscale.cu needs a compiler with #pragma clang fp contract(off): without it an " \
         "optimiser may contract the scales and silently break bit-identity with the unfused chain"
#endif

static __global__ void scale_sigmoid_scale_f32(
        const float * __restrict__ x,
        float       * __restrict__ dst,
        const float   s_in,
        const float   s_out,
        const int     k) {
    GGML_SIGMOID_SCALE_NO_CONTRACT

    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = s_out * ggml_cuda_op_sigmoid_single(s_in * x[i]);
}

// fused GGML_OP_SCALE + GGML_UNARY_OP_SIGMOID + GGML_OP_SCALE
void ggml_cuda_op_scale_sigmoid_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * src) {
    const ggml_tensor * src0 = src->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_nelements(src0) == ggml_nelements(dst));

    float s_in;
    float s_out;
    memcpy(&s_in,  (const float *) src->op_params + 0, sizeof(float));
    memcpy(&s_out, (const float *) dst->op_params + 0, sizeof(float));

    const int k = ggml_nelements(src0);
    const int num_blocks = (k + CUDA_SIGMOID_SCALE_BLOCK_SIZE - 1) / CUDA_SIGMOID_SCALE_BLOCK_SIZE;

    scale_sigmoid_scale_f32<<<num_blocks, CUDA_SIGMOID_SCALE_BLOCK_SIZE, 0, ctx.stream()>>>(
        (const float *) src0->data, (float *) dst->data, s_in, s_out, k);
}
