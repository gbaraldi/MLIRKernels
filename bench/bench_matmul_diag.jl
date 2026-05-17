# Matmul perf diagnostic — check the thread count we're using, whether
# OpenBLAS is using the same count, and how cuTileCPU matmul scales with
# tile size + thread count.

using cuTile
const ct = cuTile
using cuTileCPU
using LinearAlgebra, Printf

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

const FLUSH = Vector{UInt8}(undef, 512 * 1024 * 1024); fill!(FLUSH, 0x01)
function flush_caches!()
    s = zero(UInt64)
    @inbounds for i in 1:64:length(FLUSH); s += FLUSH[i]; end
    return s
end

function time_min(f; samples=10, warmup=3)
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

# Get OpenMP's view of the thread count. ccall library expression must be a
# const at parse time, so we bind to a const module-level alias.
const LIBOMP = cuTileCPU.LIBOMP
omp_max_threads() = Int(ccall((:omp_get_max_threads, LIBOMP), Cint, ()))
omp_num_procs() = Int(ccall((:omp_get_num_procs, LIBOMP), Cint, ()))
omp_set_threads!(n::Int) = ccall((:omp_set_num_threads, LIBOMP), Cvoid, (Cint,), Cint(n))

function bench_one(M, N, K; BM=64, BN=64, BK=64, label="")
    A = cuTileCPU.aligned_array(Float32, M, K; alignment=128); copyto!(A, rand(Float32, M, K))
    B = cuTileCPU.aligned_array(Float32, K, N; alignment=128); copyto!(B, rand(Float32, K, N))
    C = cuTileCPU.aligned_array(Float32, M, N; alignment=128); fill!(C, 0f0)

    # warm
    cuTileCPU.parallel_for(matmul_kernel,
        (A, B, C, ct.Constant(BM), ct.Constant(BN), ct.Constant(BK));
        blocks = (M ÷ BM, N ÷ BN))

    t = time_min(() -> cuTileCPU.parallel_for(matmul_kernel,
        (A, B, C, ct.Constant(BM), ct.Constant(BN), ct.Constant(BK));
        blocks = (M ÷ BM, N ÷ BN)))
    @printf("  %-40s %8.1f μs  %8.1f GFLOPS\n",
            label, t/1e3, ngflops(t, M, N, K))
end

function main()
    max_thr = omp_max_threads()
    nprocs = omp_num_procs()
    println("OpenMP: max_threads = $max_thr, num_procs = $nprocs")
    println("Julia threads: $(Threads.nthreads())")
    println("BLAS num threads (Julia): $(BLAS.get_num_threads())")
    println()

    M = N = K = 1024

    # 1. cuTileCPU at different tile sizes. Cap at 128 — at 256 the
    # `vector.contract` outerproduct lowering produces enough unrolled LLVM
    # IR (~O(N^3) FMA insts per kernel body) that clang -O2 hangs.
    println("=== cuTileCPU matmul @ M=N=K=$M, default OpenMP threads ===")
    for tile in (32, 64, 128)
        if M % tile == 0
            bench_one(M, N, K; BM=tile, BN=tile, BK=tile, label="BM=BN=BK=$tile")
        end
    end
    println()

    # 2. cuTileCPU with restricted thread counts
    println("=== cuTileCPU matmul @ M=N=K=$M, BM=BN=BK=64, thread sweep ===")
    saved = omp_max_threads()
    try
        for nthr in (1, 2, 4, 8, 16, 32, 64)
            nthr > nprocs && continue
            omp_set_threads!(nthr)
            bench_one(M, N, K; BM=64, BN=64, BK=64, label="threads=$nthr")
        end
    finally
        omp_set_threads!(saved)
    end
    println()

    # 3. OpenBLAS with restricted thread counts
    println("=== OpenBLAS matmul @ M=N=K=$M, thread sweep ===")
    A = cuTileCPU.aligned_array(Float32, M, K; alignment=128); copyto!(A, rand(Float32, M, K))
    B = cuTileCPU.aligned_array(Float32, K, N; alignment=128); copyto!(B, rand(Float32, K, N))
    C = cuTileCPU.aligned_array(Float32, M, N; alignment=128); fill!(C, 0f0)
    saved_blas = BLAS.get_num_threads()
    try
        for nthr in (1, 2, 4, 8, 16, 32, 64)
            nthr > nprocs && continue
            BLAS.set_num_threads(nthr)
            t = time_min(() -> mul!(C, A, B))
            @printf("  %-40s %8.1f μs  %8.1f GFLOPS\n",
                    "threads=$nthr", t/1e3, ngflops(t, M, N, K))
        end
    finally
        BLAS.set_num_threads(saved_blas)
    end
end

main()
