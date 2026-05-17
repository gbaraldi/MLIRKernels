# Batched matmul perf: cuTileCPU vs Julia BLAS (strided-batched gemm via mul!
# in a loop) vs naive triple-loop per batch.

using cuTile
const ct = cuTile
using cuTileCPU
using LinearAlgebra, Printf

function bmm_kernel(A::ct.TileArray{T,3}, B::ct.TileArray{T,3}, C::ct.TileArray{T,3},
                    BM::Int, BN::Int, BK::Int, BS::Int) where {T}
    bid_b = ct.bid(1)
    bid_m = ct.bid(2)
    bid_n = ct.bid(3)
    acc = zeros(T, (BS, BM, BN))
    for k in 1:cld(size(A, 2), BK)
        a = ct.load(A; index=(bid_b, bid_m, k),     shape=(BS, BM, BK))
        b = ct.load(B; index=(bid_b, k,     bid_n), shape=(BS, BK, BN))
        acc = muladd(a, b, acc)
    end
    ct.store(C; index=(bid_b, bid_m, bid_n), tile=acc)
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

function bench_bmm(batch::Int, M::Int, N::Int, K::Int;
                   BS::Int=2, BM::Int=32, BN::Int=32, BK::Int=32)
    println("\n=== BS=$batch  M=$M N=$N K=$K  (tile BS=$BS BM=$BM BN=$BN BK=$BK) ===")
    @assert batch % BS == 0 && M % BM == 0 && N % BN == 0 && K % BK == 0

    A = cuTileCPU.aligned_array(Float32, batch, M, K; alignment=128)
    B = cuTileCPU.aligned_array(Float32, batch, K, N; alignment=128)
    C = cuTileCPU.aligned_array(Float32, batch, M, N; alignment=128)
    copyto!(A, rand(Float32, batch, M, K))
    copyto!(B, rand(Float32, batch, K, N))
    fill!(C, 0f0)

    # Warm cache
    cuTileCPU.parallel_for(bmm_kernel,
        (A, B, C, ct.Constant(BM), ct.Constant(BN), ct.Constant(BK), ct.Constant(BS));
        blocks = (batch ÷ BS, M ÷ BM, N ÷ BN))

    t_ct = time_min(() -> cuTileCPU.parallel_for(bmm_kernel,
        (A, B, C, ct.Constant(BM), ct.Constant(BN), ct.Constant(BK), ct.Constant(BS));
        blocks = (batch ÷ BS, M ÷ BM, N ÷ BN)))

    # BLAS: loop over batch, mul! per slice. (Julia's stdlib has no batched gemm
    # API; we lean on the C-major view C[:,:,b] = A[:,:,b] * B[:,:,b] which is
    # what mul! does without copies.)
    t_blas = time_min(() -> begin
        for b in 1:batch
            mul!(view(C, :, :, b), view(A, :, :, b), view(B, :, :, b))
        end
    end)

    flops = 2.0 * batch * M * N * K
    @printf("  cuTileCPU @parallel_for         %9.1f μs  %8.1f GFLOPS\n",
            t_ct/1e3, flops / t_ct)
    @printf("  Julia BLAS loop (mul! per slc)  %9.1f μs  %8.1f GFLOPS\n",
            t_blas/1e3, flops / t_blas)
end

function main()
    println("Julia threads: $(Threads.nthreads())")
    bench_bmm(8,   128, 128, 128; BS=2, BM=32, BN=32, BK=32)
    bench_bmm(16,  256, 256, 256; BS=2, BM=32, BN=32, BK=32)
    bench_bmm(32,  512, 512, 512; BS=2, BM=32, BN=32, BK=32)
end

main()
