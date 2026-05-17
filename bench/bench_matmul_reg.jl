# Register-tiled GEMM experiment.
#
# Hypothesis: BLAS-style matmul on CPU works because the innermost kernel is
# register-sized — a small (MR, 1) × (1, NR) outer-product accumulated over K,
# with MR×NR scalars sitting in vector registers across the K loop.
#
# Today cuTileCPU's matmul kernel uses 64×64×64 tiles. The vector.contract
# unrolls that into ~4 K straight-line FMAs; clang -O2 takes ~35 s and
# delivers ~270 GFLOPS (~19 % of OpenBLAS at 1024³).
#
# Try the BLAS-style register tile: (16, 1) × (1, 16) → outer-product
# accumulated over K. Each K-iter loads 16 F32 from A, 16 F32 from B, does
# 256 FMAs, accumulates into (16, 16) tile. Grid: M/16 × N/16 blocks of K
# iterations each.

using cuTile
const ct = cuTile
using cuTileCPU
using LinearAlgebra, Printf

# Register-tile matmul. RM, RN are the register tile sizes; K is the
# unblocked reduction dim. Inner body: one outer product per K iteration.
function matmul_reg_kernel(A::ct.TileArray{T,2}, B::ct.TileArray{T,2},
                           C::ct.TileArray{T,2},
                           RM::Int, RN::Int, K::Int) where {T}
    bid_m = ct.bid(1)
    bid_n = ct.bid(2)
    acc = zeros(T, (RM, RN))
    for k in 1:K
        a_col = ct.load(A; index=(bid_m, k), shape=(RM, 1))   # (RM, 1)
        b_row = ct.load(B; index=(k, bid_n), shape=(1, RN))   # (1, RN)
        acc = muladd(a_col, b_row, acc)                        # rank-1 update
    end
    ct.store(C; index=(bid_m, bid_n), tile=acc)
    return
end

# Reference: existing single-level tile matmul (matches what's in tests).
function matmul_tile_kernel(A::ct.TileArray{T,2}, B::ct.TileArray{T,2},
                            C::ct.TileArray{T,2},
                            BM::Int, BN::Int, BK::Int) where {T}
    bid_m = ct.bid(1)
    bid_n = ct.bid(2)
    acc = zeros(T, (BM, BN))
    for k in 1:cld(size(A, 2), BK)
        a = ct.load(A; index=(bid_m, k),     shape=(BM, BK))
        b = ct.load(B; index=(k,     bid_n), shape=(BK, BN))
        acc = muladd(a, b, acc)
    end
    ct.store(C; index=(bid_m, bid_n), tile=acc)
    return
end

const FLUSH = Vector{UInt8}(undef, 256 * 1024 * 1024); fill!(FLUSH, 0x01)
function flush_caches!()
    s = zero(UInt64)
    @inbounds for i in 1:64:length(FLUSH); s += FLUSH[i]; end
    return s
end

function time_min(f; samples=8, warmup=2)
    for _ in 1:warmup; f(); end
    best = typemax(UInt64)
    for _ in 1:samples
        flush_caches!()
        t0 = time_ns(); f(); t1 = time_ns()
        Δ = t1 - t0
        Δ < best && (best = Δ)
    end
    return Float64(best)
end

ngflops(t_ns, M, N, K) = 2.0 * M * N * K / t_ns

function bench(M, N, K)
    println("\n=== M=N=K=$M F32 ===")
    A = cuTileCPU.aligned_array(Float32, M, K; alignment=128); copyto!(A, rand(Float32, M, K))
    B = cuTileCPU.aligned_array(Float32, K, N; alignment=128); copyto!(B, rand(Float32, K, N))
    C = cuTileCPU.aligned_array(Float32, M, N; alignment=128); fill!(C, 0f0)

    # --- Reference for correctness ---
    A_h, B_h = collect(A), collect(B)
    C_ref = A_h * B_h

    # --- Register tile: 16×16, K unblocked ---
    RM, RN = 16, 16
    @assert M % RM == 0 && N % RN == 0
    fill!(C, 0f0)
    cuTileCPU.parallel_for(matmul_reg_kernel,
        (A, B, C, ct.Constant(RM), ct.Constant(RN), ct.Constant(K));
        blocks = (M ÷ RM, N ÷ RN))
    rerr = maximum(abs, C .- C_ref) / maximum(abs, C_ref)
    @printf("  register tile (RM=%d RN=%d):  correctness rel err %.2e\n",
            RM, RN, rerr)

    if rerr < 1e-3
        t_reg = time_min(() -> cuTileCPU.parallel_for(matmul_reg_kernel,
            (A, B, C, ct.Constant(RM), ct.Constant(RN), ct.Constant(K));
            blocks = (M ÷ RM, N ÷ RN)))
    else
        t_reg = NaN
        println("  ⚠ skipping reg-tile timing — incorrect output")
    end

    # --- 64-tile reference ---
    BM = BN = BK = 64
    fill!(C, 0f0)
    cuTileCPU.parallel_for(matmul_tile_kernel,
        (A, B, C, ct.Constant(BM), ct.Constant(BN), ct.Constant(BK));
        blocks = (M ÷ BM, N ÷ BN))
    @assert isapprox(C, C_ref; rtol=1e-3) "64-tile matmul gives wrong answer"
    t_64 = time_min(() -> cuTileCPU.parallel_for(matmul_tile_kernel,
        (A, B, C, ct.Constant(BM), ct.Constant(BN), ct.Constant(BK));
        blocks = (M ÷ BM, N ÷ BN)))

    # --- BLAS ---
    Cm = Matrix{Float32}(undef, M, N); fill!(Cm, 0f0)
    mul!(Cm, A_h, B_h)
    t_blas = time_min(() -> mul!(Cm, A_h, B_h))

    flops = 2.0 * M * N * K
    @printf("  register tile %dx%d:       %8.1f μs  %8.1f GFLOPS  (%.0f%% of BLAS)\n",
            RM, RN, t_reg/1e3, ngflops(t_reg, M, N, K), 100 * (flops/t_reg) / (flops/t_blas))
    @printf("  tile 64×64×64:           %8.1f μs  %8.1f GFLOPS  (%.0f%% of BLAS)\n",
            t_64/1e3, ngflops(t_64, M, N, K), 100 * (flops/t_64) / (flops/t_blas))
    @printf("  OpenBLAS:                %8.1f μs  %8.1f GFLOPS  (reference)\n",
            t_blas/1e3, ngflops(t_blas, M, N, K))
end

function bench_tile_sweep(M, N, K)
    println("\n=== Tile-size sweep at M=N=K=$M F32 ===")
    A = cuTileCPU.aligned_array(Float32, M, K; alignment=128); copyto!(A, rand(Float32, M, K))
    B = cuTileCPU.aligned_array(Float32, K, N; alignment=128); copyto!(B, rand(Float32, K, N))
    C = cuTileCPU.aligned_array(Float32, M, N; alignment=128); fill!(C, 0f0)
    A_h, B_h = collect(A), collect(B)
    C_ref = A_h * B_h
    flops = 2.0 * M * N * K

    Cm = Matrix{Float32}(undef, M, N); fill!(Cm, 0f0)
    mul!(Cm, A_h, B_h)
    t_blas = time_min(() -> mul!(Cm, A_h, B_h))
    @printf("  OpenBLAS:                  %8.1f μs  %8.1f GFLOPS  (reference)\n",
            t_blas/1e3, ngflops(t_blas, M, N, K))

    # Capped at 64. Tile=128 takes ~minutes to compile; tile=256 hangs clang
    # (confirmed earlier — vector.contract outerproduct unrolls to ~16M FMAs
    # straight-line at 256³).
    for tile in (8, 16, 32, 64)
        if M % tile != 0; continue; end
        fill!(C, 0f0)
        cuTileCPU.parallel_for(matmul_tile_kernel,
            (A, B, C, ct.Constant(tile), ct.Constant(tile), ct.Constant(tile));
            blocks = (M ÷ tile, N ÷ tile))
        if !isapprox(C, C_ref; rtol=1e-3)
            @printf("  tile %3dx%3dx%3d:        WRONG (correctness failure)\n",
                    tile, tile, tile)
            continue
        end
        t = time_min(() -> cuTileCPU.parallel_for(matmul_tile_kernel,
            (A, B, C, ct.Constant(tile), ct.Constant(tile), ct.Constant(tile));
            blocks = (M ÷ tile, N ÷ tile)))
        @printf("  tile %3dx%3dx%3d:        %8.1f μs  %8.1f GFLOPS  (%.0f%% of BLAS)\n",
                tile, tile, tile, t/1e3, ngflops(t, M, N, K),
                100 * (flops/t) / (flops/t_blas))
    end

    # Register-tile (rank-1 outer product over K).
    for (RM, RN) in ((16, 16), (32, 32))
        if M % RM != 0 || N % RN != 0; continue; end
        fill!(C, 0f0)
        cuTileCPU.parallel_for(matmul_reg_kernel,
            (A, B, C, ct.Constant(RM), ct.Constant(RN), ct.Constant(K));
            blocks = (M ÷ RM, N ÷ RN))
        if !isapprox(C, C_ref; rtol=1e-3)
            @printf("  reg %dx%d (K-unblocked):    WRONG\n", RM, RN)
            continue
        end
        t = time_min(() -> cuTileCPU.parallel_for(matmul_reg_kernel,
            (A, B, C, ct.Constant(RM), ct.Constant(RN), ct.Constant(K));
            blocks = (M ÷ RM, N ÷ RN)))
        @printf("  reg %dx%d (K=%d steps):     %8.1f μs  %8.1f GFLOPS  (%.0f%% of BLAS)\n",
                RM, RN, K, t/1e3, ngflops(t, M, N, K),
                100 * (flops/t) / (flops/t_blas))
    end
end

function main()
    # Fix BLAS thread count to match Julia thread count for an apples-to-apples
    # comparison. Without this, BLAS sometimes auto-picks half (NUMA awareness).
    BLAS.set_num_threads(Threads.nthreads())
    println("Julia threads: $(Threads.nthreads()), BLAS threads: $(BLAS.get_num_threads())")
    bench_tile_sweep(256, 256, 256)
    bench_tile_sweep(1024, 1024, 1024)
end

main()
