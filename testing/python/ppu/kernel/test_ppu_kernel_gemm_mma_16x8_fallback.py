"""PPU0015 GEMM MMA 16x8 fallback regression tests.

Reproduces the T.gemm failure when a warp's N slice is not divisible by 16.
On PPU0015, ``tilelang/ppu/intrinsics/macro/mma_macro_generator.py``
(``PPUTensorCoreIntrinEmitter::_initialize_micro_size``) originally asserted
``warp_col_tiles % 16 == 0`` and raised for N slices such as 8 or 24.

The fallback selects the MMA atom (cutlass3 SM80 atoms) by the warp N slice:
- ``warp_col_tiles % 16 == 0`` -> m16n16k* atoms (no fallback)
- ``warp_col_tiles % 8 == 0``  -> m16n8* atoms:
  - fp16 x fp16 (fp16 or fp32 accum): m16n8k16
  - bf16 x bf16 (fp32 accum):          m16n8k16
  - tf32 x tf32 (fp32 accum):          m16n8k8
  - int8 x int8 (int32 accum):         m16n8k32

Plain fp32/fp8/fp4 cases keep the 16x16 atoms (no fallback) and are not
covered here. PPU0010 does not support the fallback (it errors out on n8
tiles directly), so no PPU0010 cases are included.

All cases must match the torch reference once the fallback is implemented.
They fail on the pre-fallback code, which is exactly the purpose of this
repro. ``warp_col_tiles = block_N // n_warp``; with block_M = 32 and the
default Square warp policy all cases below get n_warp = 1, so
warp_col_tiles equals block_N.
"""

import tilelang
import tilelang.testing
import tilelang.language as T


def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((K, N), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_K, block_N), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, False, False)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages=0,
    num_threads=128,
):
    program = matmul(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        in_dtype,
        out_dtype,
        accum_dtype,
        num_stages,
        num_threads,
    )

    kernel = tilelang.compile(program, out_idx=[2])
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if in_dtype in (T.tfloat32, T.float32):
            # Convert float32 to tfloat32 because tfloat32 mma cannot truncate
            # float32 automatically, -0x1000 means
            A = (A.view(torch.int32) - 0x1000).view(torch.float32)
            B = (B.view(torch.int32) - 0x1000).view(torch.float32)
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# fp16 x fp16, fp32 accumulation (m16n8k16)
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_f16f16f32_n8():
    # block_N = 8 -> the single warp owns an 8-wide N slice
    # (warp_col_tiles = 8), not divisible by 16 but divisible by 8:
    # must fall back to the m16n8k16 atom.
    run_gemm(
        128,
        64,
        128,
        32,
        8,
        32,
        T.float16,
        T.float32,
        T.float32,
        0,
        128,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_f16f16f32_n24():
    # block_N = 24 -> warp_col_tiles = 24 (16 + 8 mixed), not divisible by 16
    # but divisible by 8: must fall back to the m16n8k16 atom.
    run_gemm(
        128,
        48,
        128,
        32,
        24,
        32,
        T.float16,
        T.float32,
        T.float32,
        0,
        128,
    )


# ---------------------------------------------------------------------------
# fp16 x fp16, fp16 accumulation (m16n8k16)
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_f16f16f16_n8():
    # warp_col_tiles = 8 -> m16n8k16 atom with fp16 accumulator.
    run_gemm(
        128,
        64,
        128,
        32,
        8,
        32,
        T.float16,
        T.float16,
        T.float16,
        0,
        128,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_f16f16f16_n24():
    # warp_col_tiles = 24 (16 + 8 mixed) -> m16n8k16 atom with fp16 accumulator.
    run_gemm(
        128,
        48,
        128,
        32,
        24,
        32,
        T.float16,
        T.float16,
        T.float16,
        0,
        128,
    )


# ---------------------------------------------------------------------------
# bf16 x bf16, fp32 accumulation (m16n8k16)
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_bf16bf16f32_n8():
    # warp_col_tiles = 8 -> m16n8k16 atom with bf16 operands.
    run_gemm(
        128,
        64,
        128,
        32,
        8,
        32,
        T.bfloat16,
        T.float32,
        T.float32,
        0,
        128,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_bf16bf16f32_n24():
    # warp_col_tiles = 24 (16 + 8 mixed) -> m16n8k16 atom with bf16 operands.
    run_gemm(
        128,
        48,
        128,
        32,
        24,
        32,
        T.bfloat16,
        T.float32,
        T.float32,
        0,
        128,
    )


# ---------------------------------------------------------------------------
# tf32 x tf32, fp32 accumulation (m16n8k8)
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_tf32f32f32_n8():
    # warp_col_tiles = 8 -> m16n8k8 atom (tf32 k_dim = 8, block_K = 32 is a
    # multiple of 8).
    run_gemm(
        128,
        64,
        128,
        32,
        8,
        32,
        T.tfloat32,
        T.float32,
        T.float32,
        0,
        128,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_tf32f32f32_n24():
    # warp_col_tiles = 24 (16 + 8 mixed) -> m16n8k8 atom.
    run_gemm(
        128,
        48,
        128,
        32,
        24,
        32,
        T.tfloat32,
        T.float32,
        T.float32,
        0,
        128,
    )


# ---------------------------------------------------------------------------
# int8 x int8, int32 accumulation (m16n8k32)
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_i8i8i32_n8():
    # warp_col_tiles = 8 -> m16n8k32 atom (int8 k_dim = 32, block_K = 64 is a
    # multiple of 32).
    run_gemm(
        128,
        64,
        128,
        32,
        8,
        64,
        T.int8,
        T.int32,
        T.int32,
        0,
        128,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_i8i8i32_n24():
    # warp_col_tiles = 24 (16 + 8 mixed) -> m16n8k32 atom.
    run_gemm(
        128,
        48,
        128,
        32,
        24,
        64,
        T.int8,
        T.int32,
        T.int32,
        0,
        128,
    )


if __name__ == "__main__":
    tilelang.testing.main()
