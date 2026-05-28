# Compare SPMD-mode vadd against tile-based vadd at the same DRAM-scale
# workloads. Both lower to vector<16xf32> + vector.transfer_read/write on
# the same f32 data, so any perf delta is from the wrapper paths
# (alignment hints, KernelState seed, etc.) — not the inner loop.

using cuTile
const ct = cuTile
using MLIRKernels
using Printf

# Tile-based (cuTile-style) vadd — uses TileArray + ct.bid + ct.load/store.
# This is the canonical cuTile shape with ArraySpec alignment hints.
function vadd_tile(a::ct.TileArray{T,1}, b::ct.TileArray{T,1},
                   c::ct.TileArray{T,1}, tile::Int) where {T}
    bid = ct.bid(1)
    ta = ct.load(a; index=bid, shape=(tile,))
    tb = ct.load(b; index=bid, shape=(tile,))
    ct.store(c; index=bid, tile=ta + tb)
    return
end

# SPMD-style vadd — plain scalar Julia, trailing `i::Int` lane index.
# No alignment hints, no KernelState seed, no Tile types.
function vadd_spmd(a::Vector{Float32}, b::Vector{Float32},
                  c::Vector{Float32}, i::Int)
    @inbounds c[i] = a[i] + b[i]
    return
end

# Cache flush scratch — 512 MB walked between samples to evict the test data
# from any CPU LLC. Same harness as bench_vadd.jl.
const FLUSH = Vector{UInt8}(undef, 512 * 1024 * 1024); fill!(FLUSH, 0x01)
function flush_caches!()
    s = zero(UInt64)
    @inbounds for i in 1:64:length(FLUSH); s += FLUSH[i]; end
    return s
end

function time_min(f; samples=20, warmup=3)
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

function gbps(t_ns, n, T)
    bytes = 3 * n * sizeof(T)   # 2 loads + 1 store per element
    return bytes / t_ns
end

function bench(n)
    @assert n % 16 == 0
    println("\n=== n = $n  ($(n * sizeof(Float32) / 1e6) MB / array) ===")

    # Tile-based path — aligned buffers, ct.Constant(16) tile size.
    a_t = MLIRKernels.aligned_array(Float32, n; alignment=128)
    b_t = MLIRKernels.aligned_array(Float32, n; alignment=128)
    c_t = MLIRKernels.aligned_array(Float32, n; alignment=128)
    copyto!(a_t, rand(Float32, n))
    copyto!(b_t, rand(Float32, n))

    MLIRKernels.@parallel_for blocks = cld(n, 16) vadd_tile(a_t, b_t, c_t, ct.Constant(16))
    @assert c_t ≈ a_t .+ b_t

    t_tile = time_min(() -> MLIRKernels.@parallel_for blocks = cld(n, 16) vadd_tile(
        a_t, b_t, c_t, ct.Constant(16)))

    # SPMD path, unaligned — plain Vector{Float32}, alignment=16 default.
    a_s = rand(Float32, n); b_s = rand(Float32, n); c_s = zeros(Float32, n)
    k_spmd = MLIRKernels.spmd_function(vadd_spmd,
        (Vector{Float32}, Vector{Float32}, Vector{Float32}, Int);
        lane_width=16)
    k_spmd(a_s, b_s, c_s, 0; blocks = cld(n, 16))
    @assert c_s ≈ a_s .+ b_s
    t_spmd = time_min(() -> k_spmd(a_s, b_s, c_s, 0; blocks = cld(n, 16)))

    # SPMD aligned — same kernel, alignment=128, aligned host buffers.
    a_a = MLIRKernels.aligned_array(Float32, n; alignment=128); copyto!(a_a, a_s)
    b_a = MLIRKernels.aligned_array(Float32, n; alignment=128); copyto!(b_a, b_s)
    c_a = MLIRKernels.aligned_array(Float32, n; alignment=128); fill!(c_a, 0f0)
    k_spmd_aligned = MLIRKernels.spmd_function(vadd_spmd,
        (Vector{Float32}, Vector{Float32}, Vector{Float32}, Int);
        lane_width=16, alignment=128)
    k_spmd_aligned(a_a, b_a, c_a, 0; blocks = cld(n, 16))
    @assert c_a ≈ a_a .+ b_a
    t_spmd_a = time_min(() -> k_spmd_aligned(a_a, b_a, c_a, 0; blocks = cld(n, 16)))

    # Reference: plain Julia broadcast.
    c_b = similar(c_s)
    t_bcast = time_min(() -> (c_b .= a_s .+ b_s))

    @printf("  tile-based:      %8.1f μs  %7.1f GB/s\n", t_tile/1e3,   gbps(t_tile,   n, Float32))
    @printf("  SPMD (16-byte):  %8.1f μs  %7.1f GB/s\n", t_spmd/1e3,   gbps(t_spmd,   n, Float32))
    @printf("  SPMD (128-byte): %8.1f μs  %7.1f GB/s\n", t_spmd_a/1e3, gbps(t_spmd_a, n, Float32))
    @printf("  Julia bcast:     %8.1f μs  %7.1f GB/s  (reference)\n",  t_bcast/1e3,  gbps(t_bcast,  n, Float32))
    @printf("  SPMD-aligned / tile: %.2fx\n", t_tile / t_spmd_a)
end

function main()
    println("Julia threads: $(Threads.nthreads())")
    println("Tile / lane width: 16  (vector<16xf32>)")

    # L2-fit through DRAM scale.
    for n in (1 << 12, 1 << 16, 1 << 20, 1 << 24, 1 << 26, 1 << 28)
        bench(n)
    end
end

main()
