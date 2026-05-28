# Matmul GFLOPS comparison: MLIRKernels vs Julia's BLAS (OpenBLAS / MKL by default)
# vs Julia hand-rolled triple loop.

using cuTile
const ct = cuTile
using MLIRKernels
using LinearAlgebra, BenchmarkTools, Printf

function matmul_kernel(A::ct.TileArray{T,2}, B::ct.TileArray{T,2}, C::ct.TileArray{T,2},
                       BM::Int, BN::Int, BK::Int) where {T}
    bid_m = ct.bid(1)
    bid_n = ct.bid(2)
    acc = zeros(T, (BM, BN))
    for k in 1:cld(size(A, 2), BK)
        a = ct.load(A; index=(bid_m, k), shape=(BM, BK))
        b = ct.load(B; index=(k, bid_n), shape=(BK, BN))
        acc = muladd(a, b, acc)
    end
    ct.store(C; index=(bid_m, bid_n), tile=acc)
    return
end

# Cache-flush scratch — same trick as bench_vadd.
const FLUSH = Vector{UInt8}(undef, 512 * 1024 * 1024); fill!(FLUSH, 0x01)
function flush_caches!()
    s = zero(UInt64)
    @inbounds for i in 1:64:length(FLUSH); s += FLUSH[i]; end
    return s
end

function time_min(f; samples::Int=20, warmup::Int=3)
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

function report(name::String, t_ns::Float64, M::Int, N::Int, K::Int)
    flops = 2.0 * M * N * K
    gflops = flops / t_ns
    @printf("  %-32s %9.1f μs  %8.1f GFLOPS\n", name, t_ns/1e3, gflops)
end

function bench_matmul(M::Int, N::Int, K::Int; BM::Int=64, BN::Int=64, BK::Int=64)
    println("\n=== M=$M N=$N K=$K  (tile BM=$BM BN=$BN BK=$BK) ===")
    @assert M % BM == 0 && N % BN == 0 && K % BK == 0

    A = MLIRKernels.aligned_array(Float32, M, K; alignment=128)
    B = MLIRKernels.aligned_array(Float32, K, N; alignment=128)
    C = MLIRKernels.aligned_array(Float32, M, N; alignment=128)
    copyto!(A, rand(Float32, M, K))
    copyto!(B, rand(Float32, K, N))
    fill!(C, 0f0)

    # Compile once (warm cache).
    MLIRKernels.parallel_for(matmul_kernel,
                           (A, B, C, ct.Constant(BM), ct.Constant(BN), ct.Constant(BK));
                           blocks = (M ÷ BM, N ÷ BN))
    expected = Array(A) * Array(B)
    @assert isapprox(C, expected; rtol=1f-3) "matmul correctness check failed"

    t_ct = time_min(() -> MLIRKernels.parallel_for(matmul_kernel,
                          (A, B, C, ct.Constant(BM), ct.Constant(BN), ct.Constant(BK));
                          blocks = (M ÷ BM, N ÷ BN)))
    fill!(C, 0f0)

    # Julia BLAS (mul!) — typically OpenBLAS on stock Julia.
    t_blas = time_min(() -> mul!(C, A, B))
    fill!(C, 0f0)

    # Naive triple-loop, multi-threaded.
    function triple_loop!(C, A, B)
        Threads.@threads for n in 1:size(C, 2)
            @inbounds for m in 1:size(C, 1)
                s = 0f0
                @simd for k in 1:size(A, 2)
                    s = muladd(A[m, k], B[k, n], s)
                end
                C[m, n] = s
            end
        end
    end
    t_naive = time_min(() -> triple_loop!(C, A, B))

    report("MLIRKernels @parallel_for", t_ct,    M, N, K)
    report("Julia BLAS (mul!)",       t_blas,  M, N, K)
    report("Julia naive triple-loop", t_naive, M, N, K)
end

function main()
    println("Julia threads: $(Threads.nthreads())")
    println("LinearAlgebra BLAS: $(BLAS.get_config())")
    # Sweep tile sizes on a fixed 2048³ shape to see how much tuning helps.
    println("\n--- tile-size sweep on 2048³ ---")
    for tile in (64, 128, 256)
        bench_matmul(2048, 2048, 2048; BM=tile, BN=tile, BK=tile)
    end
end

main()
