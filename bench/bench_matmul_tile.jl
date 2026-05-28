# Focused matmul tile-size sweep. Vary BM=BN=BK ∈ {16, 32, 64, 128} on a
# fixed M=N=K=1024 problem. Default OpenMP parallel mode.

using cuTile
const ct = cuTile
using MLIRKernels
using Printf, LinearAlgebra

function matmul_kernel(A::ct.TileArray{T,2}, B::ct.TileArray{T,2}, C::ct.TileArray{T,2},
                       BM::Int, BN::Int, BK::Int) where {T}
    bid_m = ct.bid(1)
    bid_n = ct.bid(2)
    acc = zeros(T, (BM, BN))
    for k in 1:cld(size(A, 2), BK)
        a = ct.load(A; index=(bid_m, k),  shape=(BM, BK))
        b = ct.load(B; index=(k, bid_n),  shape=(BK, BN))
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

function bench_tile(M, N, K, tile)
    A = MLIRKernels.aligned_array(Float32, M, K; alignment=128); copyto!(A, rand(Float32, M, K))
    B = MLIRKernels.aligned_array(Float32, K, N; alignment=128); copyto!(B, rand(Float32, K, N))
    C = MLIRKernels.aligned_array(Float32, M, N; alignment=128); fill!(C, 0f0)
    # warm + cache compile
    t_compile_start = time_ns()
    MLIRKernels.parallel_for(matmul_kernel,
        (A, B, C, ct.Constant(tile), ct.Constant(tile), ct.Constant(tile));
        blocks = (M ÷ tile, N ÷ tile))
    t_compile = (time_ns() - t_compile_start) / 1e9
    t = time_min(() -> MLIRKernels.parallel_for(matmul_kernel,
        (A, B, C, ct.Constant(tile), ct.Constant(tile), ct.Constant(tile));
        blocks = (M ÷ tile, N ÷ tile)))
    return t, t_compile
end

function main()
    M = N = K = 1024
    println("Problem: M=N=K=$M, F32, default OpenMP parallel")
    println("Julia threads: $(Threads.nthreads()), BLAS threads: $(BLAS.get_num_threads())")
    println()
    @printf("  %-12s %12s %14s %18s %10s\n",
            "tile", "n_blocks", "run μs", "GFLOPS", "compile s")
    @printf("  %-12s %12s %14s %18s %10s\n",
            "----", "--------", "------", "------", "---------")
    for tile in (16, 32, 64, 128)
        n_blocks = (M ÷ tile) * (N ÷ tile)
        t, tcomp = bench_tile(M, N, K, tile)
        @printf("  BM=BN=BK=%-4d %12d %14.1f %18.1f %10.1f\n",
                tile, n_blocks, t/1e3, ngflops(t, M, N, K), tcomp)
    end
    println()

    # BLAS reference
    A = MLIRKernels.aligned_array(Float32, M, K; alignment=128); copyto!(A, rand(Float32, M, K))
    B = MLIRKernels.aligned_array(Float32, K, N; alignment=128); copyto!(B, rand(Float32, K, N))
    C = MLIRKernels.aligned_array(Float32, M, N; alignment=128); fill!(C, 0f0)
    t_blas = time_min(() -> mul!(C, A, B))
    @printf("  %-12s %12s %14.1f %18.1f\n",
            "OpenBLAS", "—", t_blas/1e3, ngflops(t_blas, M, N, K))
end

main()
