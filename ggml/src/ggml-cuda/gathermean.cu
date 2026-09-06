#include "gathermean.cuh"
#include "dequantize.cuh"
#include "convert.cuh"

// dst[i0, b, i2, i3] = scale * sum_{j < n_group} a[i0, ids[b*n_group + j, i2, i3], i2, i3]
//
// Replaces qwen4exp's QSA block mean-pool, which the graph otherwise expresses as
//   get_rows + r x cont + (r-1) x add + scale
// -- eight dispatches per attention layer, twelve layers, every decode step, and every one of
// them a full pass over the indexer KV cache. Measured at 32K x 4 users on the shipping model:
// 933 us per attention layer, 11.2 ms/step, 11.8 % of the step. It is O(n_kv), so it is worth
// almost nothing at 1K and it is one of the two largest terms at the operating point.
//
// The chain materialises `members` -- idx_dim x n_kv x n_stream floats -- and then reads it
// four more times. This reads the cache once and writes only the pooled result: at 32K that is
// 35 MB instead of 406 MB per step.
//
// ===========================================================================================
// THE ARITHMETIC CONTRACT -- READ THIS BEFORE TOUCHING THE REDUCTION
//
// Bit-identical to the chain, not merely close. Three things make it so, and all three matter:
//
//  1. THE SAME DEQUANTIZER. The chain's ggml_get_rows calls dequantize_kernel(row, ib, iqs, v)
//     and writes v.x to i00 = iybs+iqs and v.y to iybs+iqs+y_offset. This calls the identical
//     function with the identical (ib, iqs) and accumulates into the same two slots. Do not
//     re-derive the dequantisation here; a second spelling is a second rounding.
//
//  2. THE ACCUMULATOR STARTS AT THE FIRST MEMBER, NOT AT ZERO. The chain builds the sum as
//     `pooled = slice_0` and only then adds slice_1..slice_{r-1}. Starting from 0.0f would add
//     a leading `0.0f + x`, which is exact for every x EXCEPT x = -0.0: 0.0f + (-0.0f) is
//     +0.0f. An all-negative-zero block would then differ in the sign bit.
//
//  3. NO CONTRACTION. #pragma clang fp contract(off) keeps each add separate. There is no
//     multiply-add in this reduction for the optimiser to contract today, but the final
//     scale sits next to the adds and the guard costs nothing. See mulcollapse.cu for what
//     happened on this project when that guard was missing: 42.9 % of elements differed and
//     test-backend-ops, a tolerance gate, passed it anyway.
//
// Verify any change with an exact uint32 comparison against the unfused chain, on a harness
// whose resolution has been demonstrated on a known-different implementation. Not with
// test-backend-ops.
// ===========================================================================================
#if defined(__clang__) && !defined(__NVCC__)
#  define GGML_GATHER_MEAN_NO_CONTRACT _Pragma("clang fp contract(off)")
#else
#  error "gathermean.cu needs a compiler with #pragma clang fp contract(off): without it an " \
         "optimiser may contract the reduction and silently break bit-identity with the " \
         "unfused get_rows + cont + add + scale chain"
#endif

// ---------------------------------------------------------------------------------------
// WHY THE FINAL SCALE ADDS A ZERO
//
// The chain ends in GGML_OP_SCALE. The CUDA scale kernel is unconditional:
//     dst[i] = scale*x[i] + bias        (scale.cu)
// with bias = 0. When the summed members are exactly -0.0 that yields -0.0 + 0.0 = +0.0.
// A bare `scale*acc` keeps -0.0, and the two differ in the sign bit.
//
// -0.0 members are common in q4_0 and impossible in q8_0, which is exactly the split the GPU
// comparison showed. q4_0's quantiser sets d = max/-8 (ggml-quants.c), so d < 0 whenever the
// block's largest-magnitude value is positive -- about half of all blocks -- and the nibble
// for zero is 8, so dequantize_q4_0 computes (8 - 8.0f)*d = 0.0f * d = -0.0. q8_0's scale is
// amax/127 >= 0, so its zero level always dequantises to +0.0.
//
// The rate follows: the sum is -0.0 only if EVERY member is -0.0, so r=2 showed 0.10 % and
// r=4 showed ~1e-6, matching (P(nibble==8) * P(d<0))^r.
//
// The CPU backend takes a different branch -- ggml_compute_forward_scale_f32 skips the bias
// add entirely when bias == 0 and calls ggml_vec_scale_f32, a bare multiply, which keeps
// -0.0. So the two ggml backends already disagree about the sign of zero for GGML_OP_SCALE
// itself. This op inherits that; it does not introduce it. Each backend's gather_mean matches
// its own backend's chain, which is what the harness tests and what "replaces the chain"
// has to mean.
// ---------------------------------------------------------------------------------------

template<int qk, int qr, dequantize_kernel_t dequantize_kernel>
static __global__ void k_gather_mean_q(
        const void * __restrict__ src0, const int32_t * __restrict__ ids, float * __restrict__ dst,
        const int64_t ne00, const int n_group, const float scale,
        const int64_t ne1, const uint3 ne2_fdv,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {
    GGML_GATHER_MEAN_NO_CONTRACT

    // same decomposition k_get_rows uses: the divider is built over the INNER dimension, so
    // dm.x = z / ne2 lands in [0, ne1) and dm.y = z % ne2 lands in [0, ne2). Building it over
    // the outer dimension instead gives each index the other's range -- silently, and only on
    // shapes with more than one stream.
    for (int64_t z = blockIdx.z; z < ne1*(int64_t) ne2_fdv.z; z += gridDim.z) {
        const uint2 dm  = fast_div_modulo((uint32_t) z, ne2_fdv);
        const int   i11 = dm.x;                 // ids ne1 == src0 ne2
        const int   i12 = dm.y;                 // ids ne2 == src0 ne3
        const int   b   = blockIdx.x;           // output group

        for (int64_t i00 = 2*(blockIdx.y*(int64_t) blockDim.x + threadIdx.x);
             i00 < ne00;
             i00 += 2*(int64_t) gridDim.y*blockDim.x) {

            const int ib   =  i00/qk;           // block index
            const int iqs  = (i00%qk)/qr;       // quant index
            const int iybs =  i00 - i00%qk;     // dst block start index
            const int y_offset = qr == 1 ? 1 : qk/2;

            // the accumulator starts at member 0, exactly as the add chain does -- see (2) above
            float2 v;
            {
                const int i01 = ids[(int64_t)(b*n_group)*s10 + i11*s11 + i12*s12];
                dequantize_kernel((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03, ib, iqs, v);
            }
            float acc_x = v.x;
            float acc_y = v.y;

            for (int j = 1; j < n_group; ++j) {
                const int i01 = ids[(int64_t)(b*n_group + j)*s10 + i11*s11 + i12*s12];
                dequantize_kernel((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03, ib, iqs, v);
                acc_x += v.x;
                acc_y += v.y;
            }

            // + 0.0f mirrors the CUDA scale kernel's bias add; see the note above. Exact for
            // every other value: round(s*a) + 0.0f == round(s*a).
            float * dst_row = dst + b*s1 + i11*s2 + i12*s3;
            dst_row[iybs + iqs]            = scale*acc_x + 0.0f;
            dst_row[iybs + iqs + y_offset] = scale*acc_y + 0.0f;
        }
    }
}

template<typename src0_t>
static __global__ void k_gather_mean_float(
        const src0_t * __restrict__ src0, const int32_t * __restrict__ ids, float * __restrict__ dst,
        const int64_t ne00, const int n_group, const float scale,
        const int64_t ne1, const uint3 ne2_fdv,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {
    GGML_GATHER_MEAN_NO_CONTRACT

    for (int64_t z = blockIdx.z; z < ne1*(int64_t) ne2_fdv.z; z += gridDim.z) {
        const uint2 dm  = fast_div_modulo((uint32_t) z, ne2_fdv);
        const int   i11 = dm.x;
        const int   i12 = dm.y;
        const int   b   = blockIdx.x;

        for (int64_t i00 = blockIdx.y*(int64_t) blockDim.x + threadIdx.x;
             i00 < ne00;
             i00 += (int64_t) gridDim.y*blockDim.x) {

            const int i01_0 = ids[(int64_t)(b*n_group)*s10 + i11*s11 + i12*s12];
            const src0_t * row0 = (const src0_t *)((const char *) src0 + i01_0*nb01 + i11*nb02 + i12*nb03);
            float acc = ggml_cuda_cast<float>(row0[i00]);

            for (int j = 1; j < n_group; ++j) {
                const int i01 = ids[(int64_t)(b*n_group + j)*s10 + i11*s11 + i12*s12];
                const src0_t * row = (const src0_t *)((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);
                acc += ggml_cuda_cast<float>(row[i00]);
            }

            dst[b*s1 + i11*s2 + i12*s3 + i00] = scale*acc + 0.0f;   // see the note above
        }
    }
}

void ggml_cuda_op_gather_mean(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * ids  = dst->src[1];

    GGML_ASSERT(ids->type  == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ids->ne[3] == 1);

    const int   n_group = ggml_get_op_params_i32(dst, 0);
    const float scale   = ggml_get_op_params_f32(dst, 1);
    GGML_ASSERT(n_group >= 1);
    GGML_ASSERT(ids->ne[0] == dst->ne[1]*n_group);

    const int64_t ne00 = src0->ne[0];
    const int64_t n_out = dst->ne[1];
    const uint3   ne2_fdv = init_fastdiv_values((uint32_t) ids->ne[2]);

    // dst element strides
    const size_t s1 = dst->nb[1]/sizeof(float);
    const size_t s2 = dst->nb[2]/sizeof(float);
    const size_t s3 = dst->nb[3]/sizeof(float);
    // ids element strides
    const size_t s10 = ids->nb[0]/sizeof(int32_t);
    const size_t s11 = ids->nb[1]/sizeof(int32_t);
    const size_t s12 = ids->nb[2]/sizeof(int32_t);

    // ne00 is the indexer key width -- 128 on this model -- so a fixed 256-thread block would
    // leave three quarters of its waves idle at the i00 bound and cut the workgroups a CU can
    // hold. Size the block to the work instead, rounded up to a wave.
    const int64_t work_q = (ne00 + 1)/2;      // the quantised path handles two i00 per thread
    const int64_t work_f = ne00;
    auto fit = [](int64_t work) {
        int b = 32;
        while (b < work && b < CUDA_GATHER_MEAN_BLOCK_SIZE) { b *= 2; }
        return b;
    };
    const int block_q = fit(work_q);
    const int block_f = fit(work_f);
    const int64_t nblk_y_q = (work_q + block_q - 1)/block_q;
    const int64_t nblk_y_f = (work_f + block_f - 1)/block_f;
    const int64_t nz       = ids->ne[1]*ids->ne[2];

    GGML_ASSERT(n_out <= 2147483647 && nz <= 65535);

    const int32_t * ids_d = (const int32_t *) ids->data;
    float         * dst_d = (float *) dst->data;
    cudaStream_t st = ctx.stream();

#define LAUNCH_Q(qk_, qr_, dq_)                                                            \
    do {                                                                                   \
        const dim3 grid((unsigned) n_out, (unsigned) nblk_y_q, (unsigned) nz);             \
        k_gather_mean_q<qk_, qr_, dq_><<<grid, block_q, 0, st>>>(                           \
            src0->data, ids_d, dst_d, ne00, n_group, scale, ids->ne[1], ne2_fdv,           \
            s1, s2, s3, src0->nb[1], src0->nb[2], src0->nb[3], s10, s11, s12);             \
    } while (0)

#define LAUNCH_F(type_)                                                                    \
    do {                                                                                   \
        const dim3 grid((unsigned) n_out, (unsigned) nblk_y_f, (unsigned) nz);             \
        k_gather_mean_float<type_><<<grid, block_f, 0, st>>>(                               \
            (const type_ *) src0->data, ids_d, dst_d, ne00, n_group, scale, ids->ne[1],    \
            ne2_fdv, s1, s2, s3, src0->nb[1], src0->nb[2], src0->nb[3], s10, s11, s12);    \
    } while (0)

    switch (src0->type) {
        case GGML_TYPE_F32:    LAUNCH_F(float);                            break;
        case GGML_TYPE_F16:    LAUNCH_F(half);                             break;
        case GGML_TYPE_BF16:   LAUNCH_F(nv_bfloat16);                      break;
        case GGML_TYPE_Q4_0:   LAUNCH_Q(QK4_0, QR4_0, dequantize_q4_0);    break;
        case GGML_TYPE_Q4_1:   LAUNCH_Q(QK4_1, QR4_1, dequantize_q4_1);    break;
        case GGML_TYPE_Q5_0:   LAUNCH_Q(QK5_0, QR5_0, dequantize_q5_0);    break;
        case GGML_TYPE_Q5_1:   LAUNCH_Q(QK5_1, QR5_1, dequantize_q5_1);    break;
        case GGML_TYPE_Q8_0:   LAUNCH_Q(QK8_0, QR8_0, dequantize_q8_0);    break;
        default:
            // IQ4_NL and the k-quants use the super-block dequantiser, which has a different
            // signature and its own kernel in getrows.cu. ggml_gather_mean_supported() keeps
            // the graph off this path for those types rather than aborting at run time.
            GGML_ABORT("ggml_gather_mean: unsupported src0 type %s", ggml_type_name(src0->type));
    }

#undef LAUNCH_Q
#undef LAUNCH_F
}
