# Compare cuTileCPU's compiled vadd against the Julia broadcast baseline.
#
# vadd is memory-bandwidth-bound on CPU: ~3 × n × sizeof(T) bytes touched
# (2 loads + 1 store per element). Throughput in GB/s is the headline metric.
#
# Run: julia -t auto --project=cuTileCPU/bench cuTileCPU/bench/bench_vadd.jl

using cuTile
const ct = cuTile
using cuTileCPU
using BenchmarkTools, Statistics, Printf

function vadd_kernel(a::ct.TileArray{T,1}, b::ct.TileArray{T,1},
                     c::ct.TileArray{T,1}, tile::Int) where {T}
    bid = ct.bid(1)
    a_tile = ct.load(a; index=bid, shape=(tile,))
    b_tile = ct.load(b; index=bid, shape=(tile,))
    ct.store(c; index=bid, tile=a_tile + b_tile)
    return
end

# Native Julia broadcast baseline. Threaded loop is the closest fair compare
# to cuTileCPU's OpenMP grid-parallel — single-threaded broadcast is
# memory-bandwidth limited for one core, the threaded form for all cores.
function julia_broadcast!(c, a, b)
    @. c = a + b
end

function julia_threaded!(c, a, b)
    Threads.@threads :static for i in eachindex(a)
        @inbounds c[i] = a[i] + b[i]
    end
end

function report(name::String, t_ns::Float64, n::Int, T::DataType)
    bytes = 3 * n * sizeof(T)
    gbps = bytes / t_ns
    @printf("  %-28s %8.1f μs  %7.1f GB/s\n", name, t_ns/1e3, gbps)
end

# Cache-flush scratch. Sized at ~512 MB so it overflows any plausible LLC on
# this 64-core box (TR/EPYC LLCs top out around 256–512 MB). We touch every
# 64-byte line and use the result so the compiler doesn't elide it.
const FLUSH_BYTES = 512 * 1024 * 1024
const FLUSH = Vector{UInt8}(undef, FLUSH_BYTES)
fill!(FLUSH, 0x01)
function flush_caches!()
    s = zero(UInt64)
    @inbounds for i in 1:64:length(FLUSH)
        s += FLUSH[i]
    end
    return s
end

# Time a single invocation `f()` with cache flush between samples. Returns the
# minimum runtime in nanoseconds, like @belapsed but with a flush hook.
function time_min(f; samples::Int=20, warmup::Int=3)
    for _ in 1:warmup
        f()
    end
    best = typemax(UInt64)
    for _ in 1:samples
        flush_caches!()
        # Use ccall(:jl_clock_now, …) or time_ns(); time_ns() is fine here.
        t0 = time_ns()
        f()
        t1 = time_ns()
        Δ = t1 - t0
        Δ < best && (best = Δ)
    end
    return Float64(best)
end

function bench_size(n::Int; tile::Int = 16, alignment::Int = 128, samples::Int=20)
    arr_mb = n * sizeof(Float32) / 1e6
    touched_mb = 3 * arr_mb
    println("\n=== n = $n  ($arr_mb MB / array, $touched_mb MB touched / call) ===")

    a = cuTileCPU.aligned_array(Float32, n; alignment)
    b = cuTileCPU.aligned_array(Float32, n; alignment)
    c = cuTileCPU.aligned_array(Float32, n; alignment)
    fill!(a, 1f0); fill!(b, 2f0); fill!(c, 0f0)

    # Compile + cache (so the first benchmark sample isn't dominated by clang).
    cuTileCPU.parallel_for(vadd_kernel, (a, b, c, ct.Constant(tile));
                            blocks = n ÷ tile)
    @assert all(==(3f0), c)

    t_ct = time_min(() -> cuTileCPU.parallel_for(vadd_kernel,
                                                  (a, b, c, ct.Constant(tile));
                                                  blocks = n ÷ tile); samples)
    t_bcast = time_min(() -> julia_broadcast!(c, a, b); samples)
    t_thr   = time_min(() -> julia_threaded!(c, a, b); samples)

    report("cuTileCPU @parallel_for", t_ct,    n, Float32)
    report("Julia broadcast (.= .+)", t_bcast, n, Float32)
    report("Julia Threads.@threads",  t_thr,   n, Float32)
end

function main()
    println("Julia threads: $(Threads.nthreads())")
    println("Tile size: 16  (vector<16xf32>)")
    println("Flush scratch: $(FLUSH_BYTES ÷ (1024*1024)) MB walked between samples")
    # Span from L1/L2 fit through anything aggregate-LLC could plausibly hold.
    # 2^28 Float32 = 1 GB per array, 3 GB touched per call: definitively DRAM.
    # Tiny grid: nblocks < nthreads ⇒ exercises the omp_set_num_threads fix.
    # n=256, tile=16 ⇒ 16 blocks vs 64 worker pool.
    for n in (256, 4096, 1 << 16, 1 << 20, 1 << 24, 1 << 26, 1 << 28)
        bench_size(n)
    end
end

main()
