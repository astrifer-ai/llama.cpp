#include "mulcollapse.cuh"

// dst[i0, 0, i2, i3] = scale * sum_i1 a[i0, i1, i2, i3] * b[i0, i1, i2, i3]
//
// Replaces the mean-collapse in qwen4exp's hierarchical controller, which the graph
// otherwise expresses as mul + cont + (ne1-1) x add + scale -- six dispatches, all of
// them shorter than the ~2.26 us launch cost on this hardware, twice per layer.
//
// Consecutive threads take consecutive i0, so both loads are fully coalesced. The i1
// loop runs ascending to keep the summation order of the add-chain it replaces, which
// makes the result bit-identical rather than merely close.
static __global__ void mul_collapse_f32(
        const float * __restrict__ a,
        const float * __restrict__ b,
        float       * __restrict__ dst,
        const int64_t ne0,
        const int64_t ne1,
        const float   scale) {

    const int64_t i0  = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t row = blockIdx.y;                 // flattened (i2, i3)

    if (i0 >= ne0) {
        return;
    }

    const int64_t base = row*ne0*ne1 + i0;

    float acc = 0.0f;
    for (int64_t i1 = 0; i1 < ne1; ++i1) {
        const int64_t idx = base + i1*ne0;
        acc += a[idx] * b[idx];
    }

    dst[row*ne0 + i0] = scale*acc;
}

void ggml_cuda_op_mul_collapse(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    const int64_t ne0   = src0->ne[0];
    const int64_t ne1   = src0->ne[1];
    const int64_t nrows = src0->ne[2]*src0->ne[3];

    GGML_ASSERT(dst->ne[0] == ne0 && dst->ne[1] == 1);
    GGML_ASSERT(nrows <= INT_MAX);

    const int block = 256;
    const dim3 grid((ne0 + block - 1)/block, (unsigned) nrows, 1);

    mul_collapse_f32<<<grid, block, 0, ctx.stream()>>>(
        (const float *) src0->data, (const float *) src1->data, (float *) dst->data,
        ne0, ne1, scale);
}
