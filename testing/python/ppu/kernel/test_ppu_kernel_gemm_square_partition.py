"""PPU0015 Square warp-policy partition regression tests.

With the Square warp policy, a warp partition candidate is legal only if
every warp covers whole MMA tiles on both axes: the N slice must stay a
multiple of ``k_n_per_warp`` and the M slice a multiple of 16 rows. Illegal
candidates (e.g. ``warp_row_tiles`` of 24 for M = 48 with 2 row warps) are
skipped and the search falls back to a legal partition such as (1, 1).

Cases below use fp16 x fp16 operands with fp32 accumulation. The generated
kernel source is checked for the expected MMA atom shape so each case
proves which atom path was taken instead of silently degrading.
"""

import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang.transform import PassConfigKey


def matmul(M, N, K, block_M, block_N, block_K, num_stages, threads):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.float16)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, False, False, policy=T.GemmWarpPolicy.Square)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_square_gemm(M, N, K, block_M, block_N, block_K, force_n8, mma_m, mma_n):
    """Compile one Square-policy GEMM and check it against torch.

    The generated source must contain an ``mma_sync<..., mma_m, mma_n, ...>``
    atom and must not contain the other atom shape, pinning which MMA path
    the partition actually selected."""

    program = matmul(M, N, K, block_M, block_N, block_K, 0, 128)

    pass_configs = {}
    if force_n8:
        pass_configs[PassConfigKey.TL_PPU_FORCE_GEMM_N8] = True

    kernel = tilelang.compile(program, out_idx=[2], pass_configs=pass_configs)

    source = kernel.get_kernel_source()
    expected = f"{mma_m}, {mma_n}, 16"
    other_n = 8 if mma_n == 16 else 16
    other = f"{mma_m}, {other_n}, 16"
    assert expected in source, f"expected mma_sync atom <{expected}> not found in generated kernel source"
    assert other not in source, f"unexpected mma_sync atom <{other}> in generated kernel source"

    def ref_program(A, B):
        import torch

        return torch.matmul(A.to(torch.float32), B.to(torch.float32))

    kernel.get_profiler().assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_square_partition_m48_n16_default_matches_torch():
    # Square candidates that split M = 48 into a non-multiple-of-16 warp
    # row slice (e.g. 2 row warps -> 24 rows each) are illegal; the search
    # must fall back to (1, 1) and keep the m16n16 atom.
    run_square_gemm(
        48,
        16,
        64,
        48,
        16,
        64,
        force_n8=False,
        mma_m=16,
        mma_n=16,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_square_partition_m48_n16_forced_n8_matches_torch():
    # With TL_PPU_FORCE_GEMM_N8 and M = 48, the (2, 2) candidate (24 rows
    # per warp) is filtered out by the M-side divisibility rule and the
    # legal (1, 1) partition runs the m16n8 fallback atom.
    run_square_gemm(
        48,
        16,
        64,
        48,
        16,
        64,
        force_n8=True,
        mma_m=16,
        mma_n=8,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_square_partition_m48_n32_default_matches_torch():
    # M = 48 with N = 32: the unfiltered search would pick (2, 2) with
    # warp_row_tiles = 24; the M-side rule must fall back to (1, 1) so the
    # default (non-n8) path also compiles and matches torch.
    run_square_gemm(
        48,
        32,
        64,
        48,
        32,
        64,
        force_n8=False,
        mma_m=16,
        mma_n=16,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_square_partition_m64_n16_control_matches_torch():
    # Control: M = 64 is divisible by every warp row slice, so the M-side
    # rule leaves the (4, 1) partition and the m16n16 atom unchanged.
    run_square_gemm(
        64,
        16,
        64,
        64,
        16,
        64,
        force_n8=False,
        mma_m=16,
        mma_n=16,
    )


if __name__ == "__main__":
    tilelang.testing.main()
