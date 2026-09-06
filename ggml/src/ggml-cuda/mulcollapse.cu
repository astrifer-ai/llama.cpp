#include "mulcollapse.cuh"

// dst[i0, 0, i2, i3] = scale * sum_i1 a[i0, i1, i2, i3] * b[i0, i1, i2, i3]
//
// Replaces the mean-collapse in qwen4exp's hierarchical controller, which the graph
// otherwise expresses as mul + cont + (ne1-1) x add + scale -- six dispatches, all of
// them shorter than the ~2.26 us launch cost on this hardware, twice per layer.
//
// Consecutive threads take consecutive i0, so both loads are fully coalesced.
//
// ===========================================================================================
// THE ARITHMETIC CONTRACT -- READ THIS BEFORE TOUCHING THE REDUCTION
//
// This op must be BIT-IDENTICAL to the chain it replaces, not merely close, and running the
// i1 loop ascending is NOT on its own enough to achieve that.
//
// The chain evaluates ggml_mul first, which materialises every product in an f32 buffer --
// i.e. it rounds each product to f32 before anything is summed. A kernel that accumulates
// inline (acc += a[idx]*b[idx]) under -ffp-contract=fast, which is the HIP default, gets that
// multiply-add contracted into a single v_fmac_f32. The FMA carries the product at full
// internal precision and never performs that rounding, so the result drifts from the chain.
// Measured, before this was guarded: 42.9% of output elements differed at the [2560,4,1,1]
// decode shape, max bit-pattern delta 5862. test-backend-ops did not catch it and cannot --
// it is a tolerance gate, not an equality gate. Use an exact uint32 comparison against the
// unfused chain if you change anything here.
//
// Two formulations below produce that contract. THEY MUST STAY ARITHMETICALLY IDENTICAL TO
// EACH OTHER AND TO THE CHAIN. Both were verified by exact bit comparison on gfx1201 and
// gfx1151; do not edit one without re-verifying the other the same way.
//
//   ROLLED  (clang/HIP): #pragma clang fp contract(off) keeps the multiply and the add as
//           separate f32 operations. 66 instructions, 9 VGPRs, no scratch, free. Correct for
//           any ne1, so it needs no shape bound.
//
//   ARRAY   (opt-in, any other compiler): materialise the products in a fixed-size register
//           array and sum it ascending. It does not depend on a pragma being honoured, but its
//           non-contraction is EMPIRICAL rather than guaranteed -- -ffp-contract=fast may
//           legally fuse across the array store, and we have only confirmed that it does not on
//           clang/AMDGPU and g++/x86. The compile-time bound also forces an 8-way unroll (238
//           instructions, 14 VGPRs, still no scratch) costing ~0.175 us/call, about 1.5-2% of
//           this fusion's own gain.
//
// Because the ARRAY branch is only empirically safe, a compiler that has neither is a BUILD
// ERROR by default rather than a silent fallback: an unverified toolchain must not quietly
// produce different arithmetic. Define GGML_MUL_COLLAPSE_ALLOW_ARRAY_FALLBACK to opt in to the
// ARRAY branch, and only after checking it on your target with an exact uint32 comparison.
//
// The compiler test deliberately excludes __NVCC__ as well as testing __clang__: under
// `nvcc -ccbin clang` the host pass defines __clang__ but the device pass does not, so a bare
// __clang__ test would compile the ARRAY kernel while the host launcher skipped the bound
// assert that branch requires. __NVCC__ is defined in both nvcc passes, so excluding it keeps
// host and device on the same branch.
// ===========================================================================================

// Build the ARRAY branch on a compiler that would otherwise take the ROLLED one with
//   -DMUL_COLLAPSE_ROLLED=0 -DGGML_MUL_COLLAPSE_ALLOW_ARRAY_FALLBACK
// That is how you check the two branches still agree: build both, and run the same exact uint32
// comparison against the unfused chain on each.
#if !defined(MUL_COLLAPSE_ROLLED)
#  if defined(__clang__) && !defined(__NVCC__)
#    define MUL_COLLAPSE_ROLLED 1
#  else
#    define MUL_COLLAPSE_ROLLED 0
#  endif
#endif

// ===========================================================================================
// THE INVARIANT IS BIT-AGREEMENT WITH THE REFERENCE, NOT "CONTRACTION OFF"
//
// Read the guard below as being about THIS kernel and its reference, not as a project rule.
// Whether contraction must be suppressed depends entirely on what the unfused graph does:
//
//   * ggml_mul (this kernel's reference) writes every product to an f32 buffer, so the chain
//     rounds where a contracted FMA would not -- contraction OFF is what matches.
//   * rms_norm_f32 (norm.cu) computes `tmp += xi*xi` inside its own reduction, so a fused
//     version of it forcing contraction off would INTRODUCE a divergence.
//
// The same applies to every other carrier of bit-identity. Seeding a pooling accumulator at
// 0.0f flips the sign of an all-negative-zero q4_0 block (see gathermean.cu), while seeding
// an rms_norm accumulator at 0.0f is safe because xi*xi is never negative. Two kernels, two
// different traps, one discipline: derive the hazard from the reference each time.
//
// And some kernels carry no arithmetic at all: reshaping the launch of get_rows (a pure copy)
// or rope (elementwise, no cross-thread reduction) cannot change an output bit for any shape,
// so none of this machinery is engaged there.
// ===========================================================================================
#if !MUL_COLLAPSE_ROLLED
#  if !defined(GGML_MUL_COLLAPSE_ALLOW_ARRAY_FALLBACK)
#    error "mul_collapse_f32 must reproduce the roundings ITS OWN reference performs, and on \
this compiler this file has no mechanism to make it do so. THIS IS NOT A PROJECT-WIDE RULE THAT \
FP CONTRACTION MUST BE OFF. The invariant is bit-agreement with whatever the unfused graph \
actually does; the pragma is only the instrument. Here the reference is ggml_mul, which \
materialises every product into an f32 buffer, so the chain performs a product rounding that a \
contracted FMA would skip -- for THIS kernel, contraction off is what matches (measured: 42.9% \
of elements differ at the [2560,4,1,1] decode shape with it on). A reference that itself \
contracts needs the opposite setting: rms_norm_f32 in norm.cu computes `tmp += xi*xi` in its own \
reduction, so a fused version of THAT forcing contraction off would INTRODUCE the divergence \
rather than prevent it. Derive the hazard from the reference each time; do not carry this \
kernel\'s fix forward as a habit. test-backend-ops will NOT catch either error -- it is a \
tolerance gate. Either add a contraction-disabling mechanism for this compiler above, or define \
GGML_MUL_COLLAPSE_ALLOW_ARRAY_FALLBACK to use the materialised-product form after verifying it \
on your target with an exact uint32 comparison against the unfused chain."
#  endif
// Largest hyper-connection count the ARRAY branch supports. Qwen3.8-Flash-Next uses 4.
#  define MUL_COLLAPSE_MAX_NE1 8
#  pragma message("mul_collapse_f32: this compiler has no `#pragma clang fp contract(off)`, so " \
                  "GGML_OP_MUL_COLLAPSE is building its portable materialised-product fallback. " \
                  "That branch MUST stay bit-identical to the rolled branch and to the unfused " \
                  "mul+add+scale chain -- verify with an exact uint32 comparison, NOT with " \
                  "test-backend-ops, which is a tolerance gate and will pass a contracted kernel.")
#endif

static __global__ void mul_collapse_f32(
        const float * __restrict__ a,
        const float * __restrict__ b,
        float       * __restrict__ dst,
        const int64_t ne0,
        const int64_t ne1,
        const float   scale) {
#if MUL_COLLAPSE_ROLLED
#pragma clang fp contract(off)
#endif

    const int64_t i0  = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t row = blockIdx.y;                 // flattened (i2, i3)

    if (i0 >= ne0) {
        return;
    }

    const int64_t base = row*ne0*ne1 + i0;

#if MUL_COLLAPSE_ROLLED

    // contract(off) above is what makes this bit-identical: each a*b is rounded to f32 in its
    // own instruction, exactly as ggml_mul rounds it, then the ascending adds reproduce the
    // add-chain. Correct for any ne1.
    float acc = 0.0f;
    for (int64_t i1 = 0; i1 < ne1; ++i1) {
        const int64_t idx = base + i1*ne0;
        acc += a[idx] * b[idx];
    }

#else

    // No usable contract pragma here, so force the rounding structurally instead: each product
    // is stored to its own f32 slot before any of them is summed. The bound is a compile-time
    // constant so both loops fully unroll and prod[] stays in registers rather than spilling.
    const int n1 = (int) ne1;                       // host asserts n1 <= MUL_COLLAPSE_MAX_NE1

    float prod[MUL_COLLAPSE_MAX_NE1];
#pragma unroll
    for (int i1 = 0; i1 < MUL_COLLAPSE_MAX_NE1; ++i1) {
        prod[i1] = 0.0f;
        if (i1 < n1) {
            const int64_t idx = base + (int64_t) i1*ne0;
            prod[i1] = a[idx] * b[idx];             // rounded to f32 here, as ggml_mul would
        }
    }

    float acc = prod[0];                            // the add-chain starts at the first product
#pragma unroll
    for (int i1 = 1; i1 < MUL_COLLAPSE_MAX_NE1; ++i1) {
        if (i1 < n1) {
            acc += prod[i1];                        // ascending: one f32 add per element
        }
    }

#endif

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

#if !MUL_COLLAPSE_ROLLED
    // The ARRAY branch holds the products in a fixed-size register array, so a larger ne1 has
    // no supported path. Fail loudly rather than quietly changing the arithmetic. The ROLLED
    // branch has no such limit, which is why this assert is branch-local.
    GGML_ASSERT(ne1 >= 1 && ne1 <= MUL_COLLAPSE_MAX_NE1);
#endif

    const int block = 256;
    const dim3 grid((ne0 + block - 1)/block, (unsigned) nrows, 1);

    mul_collapse_f32<<<grid, block, 0, ctx.stream()>>>(
        (const float *) src0->data, (const float *) src1->data, (float *) dst->data,
        ne0, ne1, scale);
}
