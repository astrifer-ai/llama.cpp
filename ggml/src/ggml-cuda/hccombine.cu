#include "hccombine.cuh"

// dst[i0,i1,i2] = residual[i0,i1,i2] + b[i0,i2] * (s_out * sigmoid(s_in * w[i1,i2]))
//
// Fuses qwen4exp's build_hc_combine, which the graph otherwise expresses as
// scale + sigmoid + scale + repeat + mul + add -- six nodes, twice per layer.
// The gate depends only on (i1, i2), so each block computes it once into shared
// memory and then streams the residual with fully coalesced accesses along i0.
static __global__ void hc_combine_f32(
        const float * __restrict__ residual,
        const float * __restrict__ b,
        const float * __restrict__ w,
        float       * __restrict__ dst,
        const int64_t ne0,
        const int64_t ne1,
        const float   s_in,
        const float   s_out) {

    const int64_t i2 = blockIdx.y;

    // One block owns a slice of i0 for ALL i1, so b[i0] is loaded once into a register
    // and reused ne1 times. The earlier version put i1 in the grid, which re-read b from
    // memory once per i1: fine while b stayed in cache, but a measured loss at 5 users x
    // 64K context, where the KV cache evicts it between reads. The gates are ne1 values
    // shared by the whole block, so one warp computes them into shared memory.
    extern __shared__ float s_gate[];
    for (int64_t i1 = threadIdx.x; i1 < ne1; i1 += blockDim.x) {
        s_gate[i1] = s_out / (1.0f + expf(-s_in*w[i2*ne1 + i1]));
    }
    __syncthreads();

    for (int64_t i0 = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
         i0 < ne0;
         i0 += (int64_t) gridDim.x*blockDim.x) {

        const float bv = b[i2*ne0 + i0];

        for (int64_t i1 = 0; i1 < ne1; ++i1) {
            const int64_t idx = (i2*ne1 + i1)*ne0 + i0;
            dst[idx] = residual[idx] + bv*s_gate[i1];
        }
    }
}

void ggml_cuda_op_hc_combine(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * res = dst->src[0];
    const ggml_tensor * b   = dst->src[1];
    const ggml_tensor * w   = dst->src[2];

    GGML_ASSERT(res->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type   == GGML_TYPE_F32);
    GGML_ASSERT(w->type   == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(res));
    GGML_ASSERT(ggml_is_contiguous(b));
    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(ggml_is_contiguous(dst));

    float params[2];
    memcpy(params, dst->op_params, sizeof(params));

    const int64_t ne0 = res->ne[0];
    const int64_t ne1 = res->ne[1];
    const int64_t ne2 = res->ne[2];

    GGML_ASSERT(res->ne[3] == 1);
    GGML_ASSERT(ggml_nelements(b) == ne0*ne2);
    GGML_ASSERT(ggml_nelements(w) == ne1*ne2);
    GGML_ASSERT(ne2 <= 65535);
    GGML_ASSERT(ne1*sizeof(float) <= 48*1024);

    const int block  = 256;
    const int nblk_x = (int) std::min<int64_t>((ne0 + block - 1)/block, 256);

    const dim3 grid(nblk_x, (unsigned) ne2, 1);
    const size_t shmem = ne1*sizeof(float);

    hc_combine_f32<<<grid, block, shmem, ctx.stream()>>>(
        (const float *) res->data, (const float *) b->data, (const float *) w->data,
        (float *) dst->data, ne0, ne1, params[0], params[1]);
}
