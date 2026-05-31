# GPU SIMT path (MLIRCUDABackend): KA @kernels compiled through the MLIR gpu
# dialect → PTX and run on the device. Scalar-per-thread, so N-D @index +
# A[i,j] + reduction accumulators work with no uniform/varying harmonization.
# Kernels are defined at top level; only execution is guarded on a functional
# CUDA device (skips otherwise).
using CUDA, LLVM, KernelAbstractions, Atomix
using KernelAbstractions: @localmem, @synchronize, @uniform, @groupsize, @private, get_backend
using KernelAbstractions.Extras: @unroll

const MEXT = Base.get_extension(MLIRKernels, :MLIRCUDAExt)
const GPUB = MEXT.MLIRCUDABackend
const MLIRArray = MEXT.MLIRArray

# Pure (no GPU): the default-workgroup greedy fill. Mirrors CUDA's
# `threads_to_workgroupsize` distribution (fill dim 1, spill into higher dims),
# capped per-dim by extent and the NVIDIA block limits (x,y ≤ 1024, z ≤ 64).
@testset "default workgroupsize (greedy fill)" begin
    wg = MEXT._default_wgsize
    @test wg((1024,)) == (256,)                  # 1-D fills dim 1 up to the budget
    @test wg((100,)) == (100,)                   # capped by the extent
    @test wg((1, 1_000_000)) == (1, 256)         # leading singleton must NOT collapse to (1,1)
    @test wg((16, 16)) == (16, 16)               # square block, 256 lanes
    @test wg((1, 1, 1000)) == (1, 1, 64)         # z-dim capped at 64 (would be 256 uncapped)
    @test prod(wg((1, 50_000))) > 1              # never the degenerate one-thread block
    @test wg((300,); maxthreads=1024) == (300,)  # honours an occupancy-derived budget
    @test wg((4096,); maxthreads=1024) == (1024,)
end

@kernel function _g_vadd!(c, @Const(a), @Const(b))
    i = @index(Global, Linear)
    @inbounds c[i] = a[i] + b[i]
end
@kernel function _g_transpose!(a, @Const(b))
    i, j = @index(Global, NTuple)
    @inbounds a[i, j] = b[j, i]
end
@kernel function _g_matmul!(out, @Const(a), @Const(b))
    i, j = @index(Global, NTuple)
    tmp = zero(eltype(out))
    for k in 1:size(a, 2)
        @inbounds tmp += a[i, k] * b[k, j]
    end
    @inbounds out[i, j] = tmp
end
# cross-lane reverse within each block through shared memory
@kernel function _g_shrev!(out, @Const(inp))
    gid = @index(Global, Linear); lid = @index(Local, Linear)
    s = @localmem Float32 (256,)
    @inbounds s[lid] = inp[gid]
    @synchronize
    @inbounds out[gid] = s[256 - lid + 1]
end
# atomic-on-shared per-block reduction
@kernel function _g_blocksum!(out, @Const(inp))
    gid = @index(Global, Linear); gi = @index(Group, Linear)
    acc = @localmem Float32 (1,)
    @inbounds acc[1] = 0f0
    @synchronize
    Atomix.@atomic acc[1] += inp[gid]
    @synchronize
    @inbounds out[gi] = acc[1]
end
# The full KA histogram example (verbatim): @localmem + two-level shared→global
# @atomic + @synchronize + @groupsize + a 1:gs:N step-range loop + divergent ifs.
@kernel unsafe_indices=true function _g_histogram!(histogram_output, input)
    gid = @index(Group, Linear)
    lid = @index(Local, Linear)
    @uniform gs = prod(@groupsize())
    tid = (gid - 1) * gs + lid
    @uniform Nh = length(histogram_output)
    shared_histogram = @localmem eltype(input) (gs)
    for min_element in 1:gs:Nh
        @inbounds shared_histogram[lid] = 0
        @synchronize()
        max_element = min_element + gs
        if max_element > Nh
            max_element = Nh + 1
        end
        bin = tid <= length(input) ? input[tid] : 0
        if bin >= min_element && bin < max_element
            bin -= min_element - 1
            Atomix.@atomic shared_histogram[bin] += 1
        end
        @synchronize()
        if ((lid + min_element - 1) <= Nh)
            Atomix.@atomic histogram_output[lid + min_element - 1] += shared_histogram[lid]
        end
    end
end
# @private: per-thread storage (default-space alloca). Scalar + array forms;
# the array kernel takes a compile-time `::Val{M}` (not a runtime param).
@kernel function _g_privrev!(A)
    @uniform Np = prod(@groupsize())
    I = @index(Global, Linear); il = @index(Local, Linear)
    pp = @private Int (1,)
    @inbounds pp[1] = Np - il + 1
    @inbounds A[I] = pp[1]
end
@kernel function _g_privarr!(out, @Const(A), ::Val{M}) where {M}
    I = @index(Global, Linear)
    pp = @private Int (M,)
    s = 0
    @inbounds for j in 1:M; pp[j] = A[I] * j; end
    @inbounds for j in 1:M; s += pp[j]; end
    @inbounds out[I] = s
end
# 2-D @localmem tile: size(tile,d) resolves to the static dims. Copy +
# cross-lane transpose through a 16x16 shared tile.
@kernel function _g_tilecopy!(o, @Const(a))
    I, J = @index(Global, NTuple); ii, jj = @index(Local, NTuple)
    t = @localmem Float32 (16, 16)
    @inbounds t[ii, jj] = a[I, J]
    @synchronize
    @inbounds o[I, J] = t[ii, jj]
end
# KA's shared-memory tiled pattern (examples/performance.jl): `@localmem eltype(o)`
# + a `@groupsize()`-derived, BANK-PADDED (non-square) tile. Exercises (a) Core.Const
# `eltype` resolution, (b) an un-inlined `Val{BANK}`-parametric kernel body lowered
# as an outlined `func.func` (void return), and (c) a tile whose leading dim (N+1)
# differs from the access extent (so the linearisation must use the tile's own dim).
@kernel function _g_padtile!(o, @Const(a), ::Val{BANK} = Val(1)) where {BANK}
    I, J = @index(Global, NTuple); i, j = @index(Local, NTuple)
    N = @uniform @groupsize()[1]; M = @uniform @groupsize()[2]
    t = @localmem eltype(o) (N + BANK, M)
    @inbounds t[i, j] = a[I, J]
    @synchronize
    @inbounds o[I, J] = t[i, j]
end
@kernel function _g_tiletr!(o, @Const(a))
    I, J = @index(Global, NTuple); ii, jj = @index(Local, NTuple)
    t = @localmem Float32 (16, 16)
    @inbounds t[ii, jj] = a[I, J]
    @synchronize
    @inbounds o[I, J] = t[jj, ii]
end
# @simd / @unroll loops: the loopinfo hint is dropped (plain scf.for); LLVM/
# ptxas unroll. Both forms over a Val{M} bound.
@kernel function _g_simdsum!(o, @Const(a), ::Val{M}) where {M}
    Ix = @index(Global, Linear); ac = zero(eltype(o))
    @simd for k in 1:M; @inbounds ac += a[Ix] * k; end
    @inbounds o[Ix] = ac
end
@kernel function _g_unrollsum!(o, @Const(a), ::Val{M}) where {M}
    Ix = @index(Global, Linear); ac = zero(eltype(o))
    @unroll for k in 1:M
        @inbounds ac += a[Ix] * k
    end
    @inbounds o[Ix] = ac
end
# Tiled shared-memory matmul: the capstone — 2-D @localmem tiles, a k-tile loop
# with two @synchronize, and a register accumulator. The inner-product
# accumulator `out` is LOCAL to each k-tile (folded into `acc` after the inner
# loop) — one accumulator carried through BOTH loop levels trips the structurizer.
@kernel function _g_mmtiled!(C, @Const(A), @Const(B))
    I, J = @index(Global, NTuple); ii, jj = @index(Local, NTuple)
    tA = @localmem Float32 (16, 16)
    tB = @localmem Float32 (16, 16)
    acc = zero(Float32); nkt = size(A, 2) ÷ 16
    for kt in 1:nkt
        kb = (kt - 1) * 16
        @inbounds tA[ii, jj] = A[I, kb + jj]
        @inbounds tB[ii, jj] = B[kb + ii, J]
        @synchronize
        out = zero(Float32)
        @inbounds for kk in 1:16
            out += tA[ii, kk] * tB[kk, jj]
        end
        acc += out
        @synchronize
    end
    @inbounds C[I, J] = acc
end
# Closure/functor kernel arg: the kernel calls a closure `f` that captures
# device arrays (the map/reduce/broadcast pattern). The closure is flattened
# into its captured array params; the call inlines.
@kernel function _g_applyclos!(out, f)
    i = @index(Global, Linear)
    @inbounds out[i] = f(i)
end
@kernel function _g_mapclos!(n, f)
    i = @index(Global, Linear)
    if i <= n
        f(i)
    end
end
# @index(Global, Cartesian): a CartesianIndex{N}. Full-index `A[I]` + component
# `I[k]` access (the GPUArrays broadcast/copy/transpose pattern).
@kernel function _g_cartdbl!(A)
    I = @index(Global, Cartesian)
    @inbounds A[I] = A[I] * 2f0
end
@kernel function _g_carttr!(B, @Const(A))
    I = @index(Global, Cartesian)
    @inbounds B[I[2], I[1]] = A[I[1], I[2]]
end
# Numeric-union scf.if result: `flag ? Int32 : Int64` → Union{Int32,Int64},
# promoted to a common type; the i64 result is stored back into the Int32 array.
@kernel function _g_unionsel!(out, @Const(a), flag::Bool)
    i = @index(Global, Linear)
    v = flag ? a[i] : Int64(7)
    @inbounds out[i] = v % Int32
end

# Mixed-width UNSIGNED numeric union: `flag ? a[i]::UInt8 : UInt64(1)` promotes to
# UInt64, so the UInt8 branch is widened. MLIR ints are signless, so the widening
# must ZERO-extend (extui) — a sign-extend corrupts any value ≥ 128.
@kernel function _g_uwiden!(out, @Const(a), flag::Bool)
    i = @index(Global, Linear)
    r = flag ? (@inbounds a[i]) : UInt64(1)
    @inbounds out[i] = UInt64(r)
end

# UNSIGNED source widened to a SIGNED target via numeric union: `flag ? a[i]::UInt8
# : Int32(7)` promotes to Int32, and the implicit widening must read the SOURCE
# signedness (zero-extend the UInt8) — keying on the signed *target* corrupts any
# value ≥ 128 (200 → -56). This exercises the CGVal value↔type pairing.
@kernel function _g_uwiden_signed!(out, @Const(a), flag::Bool)
    i = @index(Global, Linear)
    x = flag ? (@inbounds a[i]) : Int32(7)
    @inbounds out[i] = Int32(x)
end

# Unsigned int → float must emit `uitofp`, not `sitofp` (UInt8(200) → 200.0f0).
@kernel function _g_uitofp!(out, @Const(a))
    i = @index(Global, Linear)
    @inbounds out[i] = Float32(a[i])
end

# Comparison of a union-promoted UNSIGNED operand: the widening before `ult` must
# zero-extend (UInt8(200)→UInt32 200 < 1000 == true), not sign-extend (→ false).
@kernel function _g_ucmp!(out, @Const(a), flag::Bool)
    i = @index(Global, Linear)
    x = flag ? (@inbounds a[i]) : UInt32(50)   # ::Union{UInt8,UInt32} → UInt32
    @inbounds out[i] = x < UInt32(1000)
end

# Runtime dimension extent: `size(a, d)` with a runtime `d` reads
# `getfield(a.size::Tuple, d)` with a non-const index → a select-chain over the
# per-dim `memref.dim`s.
@kernel function _g_dimsz!(out, @Const(a), d)
    i = @index(Global, Linear)
    @inbounds out[i] = size(a, d)
end

# `@ndrange()` → the launch iteration size as a compile-time-constant NTuple
# (overlaid off the CompilerMetadata's static ndrange type param, like @groupsize).
@kernel function _g_ndr!(out)
    i = @index(Global, Linear)
    @inbounds out[i] = @ndrange()[1]
end
@kernel function _g_ndr2!(out)
    i, j = @index(Global, NTuple)
    nd = @ndrange()
    @inbounds out[i, j] = nd[1] * 100 + nd[2]
end

# An infinite loop with `break` (a structurized LoopOp that doesn't promote to
# for/while) → scf.while carrying a `done` sentinel.
@kernel function _g_breakloop!(out, @Const(ns))
    i = @index(Global, Linear)
    n = @inbounds ns[i]; s = 0; k = 1
    while true
        s += k; k += 1
        k > n && break
    end
    @inbounds out[i] = s
end

# A runtime tuple index `t[d]` (non-const `d`) → select-chain over components.
@kernel function _g_tupidx!(out, @Const(a), @Const(ds))
    i = @index(Global, Linear)
    x = @inbounds a[i]; t = (x, 2x, 3x)
    @inbounds out[i] = t[@inbounds ds[i]]
end

# `@noinline` keeps this a real `:invoke` (Julia's inliner can't fold it), so the
# walker emits an OUTLINED `func.call` to a `func.func` lowered from its IR and
# MLIR `-inline` splices it back. Exercises the outlined-call worklist.
@noinline _g_poly(x::Float32) = x * x + 2f0 * x + 1f0
@kernel function _g_outline!(out, @Const(a))
    i = @index(Global, Linear)
    @inbounds out[i] = _g_poly(a[i])
end

# Explicit device throw: a negative element raises a `DomainError` (typed `Union{}`
# → `emit_exception!` signals CUDA's per-context exception flag, which the host's
# `check_exceptions()` turns into a `KernelException`). A non-negative input runs
# clean — no false exception. `@inbounds` so only the explicit throw is in play.
@kernel function _g_throw!(out, @Const(a))
    i = @index(Global, Linear)
    x = @inbounds a[i]
    if x < 0f0
        throw(DomainError(x, "negative"))
    end
    @inbounds out[i] = x + 1f0
end

# A value-returning @noinline helper whose `return` is inside a conditional branch
# (here one arm throws) is unsupported — the live `return` can't be a func.return
# inside an scf.if region. It must error CLEANLY at compile, NOT emit a duplicate
# `@__mlirkernels_exc` global (the exception-global dedup must be shared across the
# kernel + outlined-func lowering contexts) nor silently drop the return value via
# a poison func.return. Regression for both.
@noinline _g_thrhelper(x::Float32) = x < 0f0 ? throw(DomainError(x, "neg")) : x * x
@kernel function _g_outthrow!(out, @Const(a))
    i = @index(Global, Linear)
    @inbounds out[i] = _g_thrhelper(a[i])
end

# `@inbounds` honoring: inference folds `@inbounds` into `Expr(:boundscheck, false)`,
# so a NON-`@inbounds` access compiles a real bounds check (OOB → KernelException)
# while the `@inbounds` form elides it. `_g_chk!` is in-bounds; `_g_chkoob!` reads
# deliberately out of range.
@kernel function _g_chk!(c, @Const(a))
    i = @index(Global, Linear)
    c[i] = a[i]                  # non-@inbounds → checked
end
@kernel function _g_chkoob!(c, @Const(a))
    i = @index(Global, Linear)
    c[i] = a[i + 1000000]        # non-@inbounds OOB read
end

# A checked access INSIDE a loop: its OOB path is a `break` out of the loop to a
# post-loop `throw_boundserror`, threaded straight to `emit_exception!`. In-bounds
# (n == length) sums correctly; OOB (n > length) raises a KernelException.
@kernel function _g_loopchk!(out, @Const(a), n)
    i = @index(Global, Linear)
    acc = zero(eltype(out))
    for k in 1:n
        acc += a[k]              # non-@inbounds → checked, inside a loop
    end
    out[i] = acc
end

# An EXPLICIT throw inside a counted for-loop, with a carried accumulator used
# after the loop. IRStructurizer keeps the loop-exit throw (a `…; throw::Union{};
# ReturnNode()` arm) instead of dropping it to a bare break, and the loop lowering
# treats that divergent arm as a control transfer (emit_exception! + poison-yield).
# Good input sums correctly; a negative element raises a KernelException.
@kernel function _g_loopthrow!(out, @Const(a), n)
    i = @index(Global, Linear)
    acc = zero(eltype(out))
    for k in 1:n
        x = @inbounds a[k]
        if x < zero(eltype(out))
            throw(DomainError(x, "neg"))
        end
        acc += x
    end
    @inbounds out[i] = acc
end

# A VOID @noinline validation helper that only conditionally throws (returns
# nothing on the good path) called purely for effect. Inference flags it
# effect-free (the throw violates :nothrow, not :effect_free), so DCE must keep
# it via the full-removable gate — dropping it (the old effect-free-only gate)
# silently erased the check. Regression for the DCE-throwing-helper fix.
@noinline _g_check_nonneg(x::Float32) = (x < 0f0 && throw(DomainError(x, "neg")); nothing)
@kernel function _g_voidthrow!(out, @Const(a))
    i = @index(Global, Linear)
    x = @inbounds a[i]
    _g_check_nonneg(x)
    @inbounds out[i] = x + 1f0
end

# Multi-dim @index Linear: Local/Group Linear must linearise COLUMN-MAJOR over the
# block/grid (was dim-x-only → all dim>1 threads/blocks collapsed onto dim-1).
@kernel function _g_locallin!(out)
    gi = @index(Global, Linear); li = @index(Local, Linear)
    @inbounds out[gi] = li
end
@kernel function _g_grouplin!(out)
    gi = @index(Global, Linear); gid = @index(Group, Linear)
    @inbounds out[gi] = gid
end
# N-D @atomic: A[i,j] must hit the column-major linear element (was indices[1]-only).
@kernel function _g_hist2d!(H, @Const(rs), @Const(cs))
    i = @index(Global, Linear)
    @inbounds r = rs[i]; @inbounds c = cs[i]
    Atomix.@atomic H[r, c] += Int32(1)
end

# A user module whose functions share names with our Intrinsics markers; calling
# them in a kernel must invoke the USER function, not silently lower to the
# intrinsic (matched by bare name before the parentmodule guard).
module _GShadow
    @noinline group_size() = 7
    @noinline local_index() = 3
end
@kernel function _g_shadow!(out)
    i = @index(Global, Linear)
    @inbounds out[i] = _GShadow.group_size() * 10 + _GShadow.local_index()   # 73
end

# A user function whose name SHADOWS a Base math builtin (`exp`) must run as the
# user's own code (outlined func.call), not be misrouted to `math.exp` — the walker
# dispatches by callee identity (resolved binding), not by bare name.
module _GMathShadow
    @noinline exp(x) = x * 3.0f0 + 1.0f0
end
@kernel function _g_mathshadow!(out, @Const(a))
    i = @index(Global, Linear)
    @inbounds out[i] = _GMathShadow.exp(a[i])      # user exp (3x+1), NOT e^x
end

# A ComplexF32 (aggregate) flows out of an scf.if whose other branch throws — the
# throwing branch must yield a POISON aggregate. Was an `undef_value: unsupported
# type ComplexF32` compile crash (only real scalars had a poison value).
@kernel function _g_cplx_throw!(out, @Const(a))
    i = @index(Global, Linear)
    x = @inbounds a[i]
    z = abs2(x) < 1f9 ? x * (2f0 + 0f0im) : (throw(DomainError(0)); x)
    @inbounds out[i] = z
end

# Heterogeneous struct with a NESTED field: flat leaves [f32,i32,i32], so #leaves(3)
# ≠ #fields(2). Reading the nested `Tuple` field of a LOADED struct (`w.ij`) and
# reconstructing the WHOLE struct ARG (storing `w`) both need the field→leaf-offset
# map / recursive leaf gather — previously errored ("nested heterogeneous struct").
struct _WSNest; v::Float32; ij::Tuple{Int32,Int32}; end
@kernel function _g_ws_read!(out, @Const(a))
    i = @index(Global, Linear)
    w = @inbounds a[i]
    @inbounds out[i] = w.ij[1] + w.ij[2]          # read nested field of a loaded struct
end
@kernel function _g_ws_recon!(out, w::_WSNest)
    i = @index(Global, Linear)
    @inbounds out[i] = w                          # reconstruct the whole arg struct
end

# A user method EXTENDING a Base math generic on a custom type. The `:invoke`'s
# callee resolves to `Base.sin` (root Base), so a function-identity check alone would
# wave it through to `math.sin` and silently drop the user body. The dispatch must
# consult the RESOLVED METHOD (defined here, not in Base) and outline it. See
# `_invoke_method_is_stdlib`.
struct _GAngle; v::Float32; end
@noinline Base.sin(a::_GAngle) = _GAngle(a.v + 1.0f0)   # user override: x+1, NOT sin(x)
@kernel function _g_user_sin!(out, @Const(a))
    i = @index(Global, Linear)
    @inbounds out[i] = sin(_GAngle(a[i])).v
end

# A hetero struct with a SINGLE-LEAF NESTED field (`Tuple{Int64}` = 1 leaf): reading
# it hits the n==1 getfield branch, which must extract the bare `i64` leaf and re-wrap
# it as the field's `vector<1×i64>` — NOT declare the aggregate type on a bare-scalar
# struct member (an invalid `llvm.extractvalue`). See emit_getfield! hetero n==1 path.
struct _GCI1; t::Tuple{Int64}; y::Int64; end
@kernel function _g_ci1_read!(out, @Const(a))
    i = @index(Global, Linear)
    w = @inbounds a[i]
    @inbounds out[i] = w.t[1] + w.y               # read the single-leaf nested field
end

# A hetero struct whose nested aggregate field carries TRAILING padding
# (`Tuple{Float64,Int16}` = 16 bytes incl. 6 trailing pad) followed by another field:
# the flat `!llvm.struct` would place the trailing `Float32` at byte 20, but Julia
# puts it at byte 24 — a silent host/device byte mismatch. Must be REJECTED with a
# clear error, not miscompiled. See `_flat_layout_matches_julia`.
struct _GMixedPad; a::Int32; b::Tuple{Float64,Int16}; c::Float32; end
@kernel function _g_mixedpad!(out, @Const(a))
    i = @index(Global, Linear)
    @inbounds out[i] = a[i]
end

# A type-unstable scf.for carry: `acc` is Int32 but `acc + 1` transiently widens it
# to Int64, so the yielded value's width differs from the iter-arg. The :for body
# must coerce the yield to the iter-arg type (like :while) — was an scf.for verifier
# error ("iter_arg and yielded value have different type: i32 != i64").
@kernel function _g_widen_for!(out)
    i = @index(Global, Linear)
    acc = Int32(0)
    for j in 1:8
        acc = (j % 2 == 0) ? acc + Int32(1) : acc + 1
    end
    @inbounds out[i] = acc
end

@testset "GPU: KA @kernel on MLIRCUDABackend (SIMT)" begin
    if !CUDA.functional()
        @info "CUDA not functional in this env — skipping GPU backend test"
        @test true
    else
        # Every input is an MLIRArray, so KernelAbstractions.get_backend infers
        # MLIRCUDABackend from the data — no backend type is named at the call
        # sites (the path GPUArrays / AcceleratedKernels take).
        # 1-D vadd
        N = 4096
        a1 = MLIRArray(CUDA.rand(Float32, N)); b1 = MLIRArray(CUDA.rand(Float32, N))
        c1 = MLIRArray(CUDA.zeros(Float32, N))
        backend = get_backend(a1)
        @test backend isa GPUB                            # auto-dispatch → MLIRCUDABackend
        _g_vadd!(backend, 256)(c1, a1, b1; ndrange=N); CUDA.synchronize()
        @test Array(c1) == Array(a1) .+ Array(b1)
        # 2-D transpose (non-square) — catches descriptor dim-order
        bh = reshape(collect(Float32, 1:32), 8, 4)
        bt = MLIRArray(CUDA.CuArray(bh)); at = MLIRArray(CUDA.zeros(Float32, 4, 8))
        _g_transpose!(backend, (4, 4))(at, bt; ndrange=(4, 8)); CUDA.synchronize()
        @test Array(at) == permutedims(bh)
        # 2-D matmul (non-square) — scalar accumulator over a for-loop
        ah = rand(Float32, 8, 4); bbh = rand(Float32, 4, 6)
        am = MLIRArray(CUDA.CuArray(ah)); bm = MLIRArray(CUDA.CuArray(bbh)); om = MLIRArray(CUDA.zeros(Float32, 8, 6))
        _g_matmul!(backend, (4, 2))(om, am, bm; ndrange=(8, 6)); CUDA.synchronize()
        @test maximum(abs.(Array(om) .- ah * bbh)) < 1f-3

        # @localmem cross-lane reverse + atomic-on-shared block reduction
        Nl = 1024; Wl = 256; NBl = Nl ÷ Wl
        inl = MLIRArray(CUDA.CuArray(rand(Float32, Nl))); ihl = Array(inl)
        orl = MLIRArray(CUDA.zeros(Float32, Nl))
        _g_shrev!(backend, Wl)(orl, inl; ndrange=Nl); CUDA.synchronize()
        refrev = similar(ihl)
        for b in 0:(NBl-1), k in 1:Wl; refrev[b*Wl+k] = ihl[b*Wl + (Wl-k+1)]; end
        @test Array(orl) == refrev                       # cross-lane shared + barrier
        osum = MLIRArray(CUDA.zeros(Float32, NBl))
        _g_blocksum!(backend, Wl)(osum, inl; ndrange=Nl); CUDA.synchronize()
        refsum = [sum(ihl[(b*Wl+1):((b+1)*Wl)]) for b in 0:(NBl-1)]
        @test isapprox(Array(osum), refsum; rtol=1f-4)   # atomic-on-shared

        # full KA histogram
        Lh = 4096; NBINS = 256
        hin = rand(1:NBINS, Lh)
        dhin = MLIRArray(CUDA.CuArray(hin)); dhout = MLIRArray(CUDA.zeros(Int, NBINS))
        _g_histogram!(backend, (256,))(dhout, dhin; ndrange=Lh); CUDA.synchronize()
        hist_ref = zeros(Int, NBINS); for v in hin; hist_ref[v] += 1; end
        @test Array(dhout) == hist_ref                   # full KA histogram

        # @private scalar + array (with Val arg)
        Np = 64; Wp = 16
        ap = MLIRArray(CUDA.zeros(Int, Np))
        _g_privrev!(backend, Wp)(ap; ndrange=Np); CUDA.synchronize()
        @test Array(ap) == repeat(collect(Wp:-1:1), Np ÷ Wp)   # per-thread scalar
        Mp = 4; inpp = MLIRArray(CUDA.CuArray(collect(1:Np))); op = MLIRArray(CUDA.zeros(Int, Np))
        _g_privarr!(backend, Wp)(op, inpp, Val(Mp); ndrange=Np); CUDA.synchronize()
        @test Array(op) == [i * sum(1:Mp) for i in 1:Np]       # per-thread array + Val arg

        # 2-D @localmem tile copy + cross-lane transpose
        Mt = 32
        int = MLIRArray(CUDA.CuArray(reshape(collect(Float32, 1:(Mt*Mt)), Mt, Mt))); iht = Array(int)
        otc = MLIRArray(CUDA.zeros(Float32, Mt, Mt))
        _g_tilecopy!(backend, (16,16))(otc, int; ndrange=(Mt,Mt)); CUDA.synchronize()
        @test Array(otc) == iht                                # 2-D tile copy
        # KA padded-tile pattern: eltype + @groupsize dims + BANK padding (N+1≠N)
        opt = MLIRArray(CUDA.zeros(Float32, Mt, Mt))
        _g_padtile!(backend, (16,16))(opt, int; ndrange=(Mt,Mt)); CUDA.synchronize()
        @test Array(opt) == iht                                # padded shared-mem tile
        ott = MLIRArray(CUDA.zeros(Float32, Mt, Mt))
        _g_tiletr!(backend, (16,16))(ott, int; ndrange=(Mt,Mt)); CUDA.synchronize()
        reft = copy(iht)
        for bi in 0:1, bj in 0:1, ai in 1:16, aj in 1:16
            reft[bi*16+ai, bj*16+aj] = iht[bi*16+aj, bj*16+ai]
        end
        @test Array(ott) == reft                               # 2-D tile cross-lane transpose

        # @simd / @unroll reduction loops over a Val{M} bound
        Ns = 64; Ws = 16; Ms = 4
        as = MLIRArray(CUDA.CuArray(collect(1:Ns))); refs = [Array(as)[k]*sum(1:Ms) for k in 1:Ns]
        os1 = MLIRArray(CUDA.zeros(Int, Ns))
        _g_simdsum!(backend, Ws)(os1, as, Val(Ms); ndrange=Ns); CUDA.synchronize()
        @test Array(os1) == refs                               # @simd
        os2 = MLIRArray(CUDA.zeros(Int, Ns))
        _g_unrollsum!(backend, Ws)(os2, as, Val(Ms); ndrange=Ns); CUDA.synchronize()
        @test Array(os2) == refs                               # @unroll

        # code_gpu reflection: every codegen level emits its expected IR.
        ar = MLIRArray(CUDA.rand(Float32, 256)); br = MLIRArray(CUDA.rand(Float32, 256))
        cr = MLIRArray(CUDA.zeros(Float32, 256))
        kr = _g_vadd!(backend, 256)
        @test occursin("gpu.func",        _ir(code_gpu, kr, cr, ar, br; ndrange=256, level=:mlir))
        @test occursin("llvm.",           _ir(code_gpu, kr, cr, ar, br; ndrange=256, level=:lowered))
        @test occursin("ptx_kernel",      _ir(code_gpu, kr, cr, ar, br; ndrange=256, level=:llvm))
        @test occursin(".visible .entry", _ir(code_gpu, kr, cr, ar, br; ndrange=256, level=:ptx))

        # tiled matmul
        for nm in (256, 512)
            Am = MLIRArray(CUDA.rand(Float32, nm, nm)); Bm = MLIRArray(CUDA.rand(Float32, nm, nm))
            Cm = MLIRArray(CUDA.zeros(Float32, nm, nm))
            _g_mmtiled!(backend, (16, 16))(Cm, Am, Bm; ndrange=(nm, nm)); CUDA.synchronize()
            @test isapprox(Array(Cm), Array(Am) * Array(Bm); rtol=1f-2)  # tiled matmul
        end

        # Closure kernel args (the map/reduce pattern): a closure capturing 1 or
        # 2 device arrays, flattened into memref params; the call inlines.
        csrc = MLIRArray(CUDA.CuArray(collect(Float32, 1:1024)))
        cg = let s = csrc; i -> 2f0 * s[i]; end
        cout = MLIRArray(CUDA.zeros(Float32, 1024))
        _g_applyclos!(backend, 256)(cout, cg; ndrange=1024); CUDA.synchronize()
        @test Array(cout) ≈ 2f0 .* (1:1024)                    # 1-capture closure
        cdst = MLIRArray(CUDA.zeros(Float32, 1024)); csrc2 = MLIRArray(CUDA.CuArray(collect(Float32, 1:1024)))
        ch = let d = cdst, s = csrc2; i -> (@inbounds d[i] = 3f0 * s[i]); end
        _g_mapclos!(backend, 256)(1024, ch; ndrange=1024); CUDA.synchronize()
        @test Array(cdst) ≈ 3f0 .* (1:1024)                    # 2-capture closure + write

        # @index(Global, Cartesian): full-index A[I] (2-D + 1-D) + component I[k].
        cda = MLIRArray(CUDA.CuArray(rand(Float32, 16, 16))); cda0 = Array(cda)
        _g_cartdbl!(backend, (4, 4))(cda; ndrange=size(cda)); CUDA.synchronize()
        @test Array(cda) ≈ 2f0 .* cda0                         # 2-D Cartesian A[I]
        cv = MLIRArray(CUDA.CuArray(rand(Float32, 1024))); cv0 = Array(cv)
        _g_cartdbl!(backend, 256)(cv; ndrange=length(cv)); CUDA.synchronize()
        @test Array(cv) ≈ 2f0 .* cv0                           # 1-D Cartesian
        cta = MLIRArray(CUDA.CuArray(rand(Float32, 8, 12))); ctb = MLIRArray(CUDA.zeros(Float32, 12, 8))
        _g_carttr!(backend, (4, 4))(ctb, cta; ndrange=size(cta)); CUDA.synchronize()
        @test Array(ctb) == permutedims(Array(cta))            # Cartesian I[k] transpose

        # Tail-block masking: ndrange NOT a multiple of the workgroup. The grid
        # is padded (cld) and __validindex masks the out-of-range tail threads
        # (a tail thread that wrote would index out of bounds).
        for Nt in (1000, 257)
            ta = MLIRArray(CUDA.CuArray(rand(Float32, Nt))); tb = MLIRArray(CUDA.CuArray(rand(Float32, Nt)))
            tc = MLIRArray(CUDA.zeros(Float32, Nt))
            _g_vadd!(backend, 256)(tc, ta, tb; ndrange=Nt); CUDA.synchronize()
            @test Array(tc) == Array(ta) .+ Array(tb)          # 1-D masked tail
        end
        mta = MLIRArray(CUDA.CuArray(rand(Float32, 100, 70))); mta0 = Array(mta)
        _g_cartdbl!(backend, (16, 16))(mta; ndrange=size(mta)); CUDA.synchronize()
        @test Array(mta) ≈ 2f0 .* mta0                         # 2-D masked tail

        # Numeric union scf.if result (Union{Int32,Int64}) promoted to a common
        # type, then stored back into the Int32 array (value coerced at the store).
        ua = MLIRArray(CUDA.CuArray(collect(Int32, 1:64))); uo = MLIRArray(CUDA.zeros(Int32, 64))
        _g_unionsel!(backend, 16)(uo, ua, true; ndrange=64); CUDA.synchronize()
        @test Array(uo) == collect(Int32, 1:64)                # union branch: a[i]
        uo2 = MLIRArray(CUDA.zeros(Int32, 64))
        _g_unionsel!(backend, 16)(uo2, ua, false; ndrange=64); CUDA.synchronize()
        @test all(Array(uo2) .== Int32(7))                     # union branch: Int64(7)%Int32
        # unsigned mixed-width union → must zero-extend (extui), not sign-extend
        uw = MLIRArray(CUDA.CuArray(UInt8[200, 5, 130, 255])); uwo = MLIRArray(CUDA.zeros(UInt64, 4))
        _g_uwiden!(backend, 4)(uwo, uw, true; ndrange=4); CUDA.synchronize()
        @test Array(uwo) == UInt64[200, 5, 130, 255]           # zero-extended, not 0xFF…C8
        # UNSIGNED source widened to a SIGNED target (the CGVal source-signedness fix)
        uws = MLIRArray(CUDA.CuArray(UInt8[200, 5, 130, 255])); uwso = MLIRArray(CUDA.zeros(Int32, 4))
        _g_uwiden_signed!(backend, 4)(uwso, uws, true; ndrange=4); CUDA.synchronize()
        @test Array(uwso) == Int32[200, 5, 130, 255]           # zero-extended into Int32, not -56…
        # unsigned int → float must be uitofp (200 → 200.0), not sitofp (→ -56.0)
        ufa = MLIRArray(CUDA.CuArray(UInt8[200, 5, 130, 255])); ufo = MLIRArray(CUDA.zeros(Float32, 4))
        _g_uitofp!(backend, 4)(ufo, ufa; ndrange=4); CUDA.synchronize()
        @test Array(ufo) == Float32[200, 5, 130, 255]
        # unsigned compare with width promotion: zero-extend before ult, not sign-extend
        uca = MLIRArray(CUDA.CuArray(UInt8[200, 10, 255])); uco = MLIRArray(CUDA.zeros(Bool, 3))
        _g_ucmp!(backend, 3)(uco, uca, true; ndrange=3); CUDA.synchronize()
        @test Array(uco) == Bool[1, 1, 1]

        # `size(a, d)` with a RUNTIME `d` — getfield(a.size, d) at a non-const
        # index → select-chain over memref.dims.
        dm = MLIRArray(CUDA.CuArray(rand(Float32, 8, 5))); dz = MLIRArray(CUDA.zeros(Int64, 4))
        _g_dimsz!(backend, 4)(dz, dm, 1; ndrange=4); CUDA.synchronize()
        @test all(Array(dz) .== 8)
        _g_dimsz!(backend, 4)(dz, dm, 2; ndrange=4); CUDA.synchronize()
        @test all(Array(dz) .== 5)

        # @ndrange() → the launch size, folded to a compile-time constant tuple.
        nr1 = MLIRArray(CUDA.zeros(Int, 7))
        _g_ndr!(backend, 7)(nr1; ndrange=7); CUDA.synchronize()
        @test all(Array(nr1) .== 7)                            # 1-D @ndrange()[1]
        nr2 = MLIRArray(CUDA.zeros(Int, 3, 5))
        _g_ndr2!(backend, (3, 3))(nr2; ndrange=(3, 5)); CUDA.synchronize()
        @test all(Array(nr2) .== 305)                          # 2-D @ndrange() = (3,5)

        # LoopOp: `while true … break` → scf.while + done sentinel.
        bn = MLIRArray(CUDA.CuArray(Int64[3, 5, 10, 1])); bo = MLIRArray(CUDA.zeros(Int64, 4))
        _g_breakloop!(backend, 4)(bo, bn; ndrange=4); CUDA.synchronize()
        @test Array(bo) == [sum(1:n) for n in (3, 5, 10, 1)]

        # Runtime tuple index → select-chain.
        ta = MLIRArray(CUDA.CuArray(Int64[5, 5, 5])); td = MLIRArray(CUDA.CuArray(Int64[1, 2, 3]))
        to = MLIRArray(CUDA.zeros(Int64, 3))
        _g_tupidx!(backend, 4)(to, ta, td; ndrange=3); CUDA.synchronize()
        @test Array(to) == [5, 10, 15]

        # Outlined call: a `@noinline` callee → func.call to an emitted func.func,
        # spliced back by MLIR `-inline`.
        ox = rand(Float32, 64); oa = MLIRArray(CUDA.CuArray(ox)); oo = MLIRArray(CUDA.zeros(Float32, 64))
        _g_outline!(backend, 64)(oo, oa; ndrange=64); CUDA.synchronize()
        @test Array(oo) ≈ ox .^ 2 .+ 2 .* ox .+ 1

        # Device exceptions: an explicit `throw` reaches the host as a
        # `KernelException` (via CUDA's per-context exception flag), while valid
        # input never raises a false exception.
        let N = 256
            good = MLIRArray(CUDA.CuArray(rand(Float32, N) .+ 1f0))   # all > 0
            tgo = MLIRArray(CUDA.zeros(Float32, N))
            _g_throw!(backend, 64)(tgo, good; ndrange=N); CUDA.synchronize()
            @test Array(tgo) ≈ Array(good) .+ 1f0                    # no false throw
            badv = rand(Float32, N) .+ 1f0; badv[123] = -5f0          # one negative
            bad = MLIRArray(CUDA.CuArray(badv)); tbo = MLIRArray(CUDA.zeros(Float32, N))
            @test_throws CUDA.CUDACore.KernelException begin
                _g_throw!(backend, 64)(tbo, bad; ndrange=N); CUDA.synchronize()
            end
        end

        # A value-returning @noinline helper that throws → clean COMPILE error
        # (value `return` inside a conditional branch), not a duplicate-global
        # crash or a silent miscompile. Lowering to :mlir is enough to trigger it.
        let oa = MLIRArray(CUDA.zeros(Float32, 64))
            @test_throws "conditional branch" code_gpu(devnull, _g_outthrow!(backend, 64),
                MLIRArray(CUDA.zeros(Float32, 64)), oa; ndrange=64, level=:mlir)
        end

        # `@inbounds` is honored: a NON-`@inbounds` access compiles a real bounds
        # check (its OOB throw → the `@__mlirkernels_exc` exception global), while an
        # `@inbounds` kernel elides it entirely.
        let N = 256
            ca = MLIRArray(CUDA.rand(Float32, N)); cc = MLIRArray(CUDA.zeros(Float32, N))
            chk_mlir = _ir(code_gpu, _g_chk!(backend, 64), cc, ca; ndrange=N, level=:mlir)
            inb_mlir = _ir(code_gpu, _g_vadd!(backend, 256), cc, ca, ca; ndrange=N, level=:mlir)
            @test occursin("__mlirkernels_exc", chk_mlir)    # checked → bounds check
            @test !occursin("__mlirkernels_exc", inb_mlir)   # @inbounds → elided
            # checked, in-bounds: correct result, no false exception
            _g_chk!(backend, 64)(cc, ca; ndrange=N); CUDA.synchronize()
            @test Array(cc) == Array(ca)
            # checked, OOB: a real KernelException
            coob = MLIRArray(CUDA.zeros(Float32, N))
            @test_throws CUDA.CUDACore.KernelException begin
                _g_chkoob!(backend, 64)(coob, ca; ndrange=N); CUDA.synchronize()
            end
        end

        # Bounds check INSIDE a loop: the OOB `break`→post-loop-throw is threaded
        # to emit_exception!, so the kernel compiles (scf.for, break-free) and the
        # check runs. In-bounds sums correctly; OOB inside the loop raises.
        let L = 8
            la = MLIRArray(CUDA.CuArray(collect(Float32, 1:L)))
            lo = MLIRArray(CUDA.zeros(Float32, L))
            _g_loopchk!(backend, L)(lo, la, L; ndrange=L); CUDA.synchronize()
            @test all(Array(lo) .≈ sum(1:L))                  # in-bounds, looped
            loob = MLIRArray(CUDA.zeros(Float32, L))
            @test_throws CUDA.CUDACore.KernelException begin
                _g_loopchk!(backend, L)(loob, la, L + 5; ndrange=L); CUDA.synchronize()
            end
        end

        # Explicit throw INSIDE a counted for-loop (+ carried accumulator used after).
        # Good input sums; a negative element raises a KernelException.
        let L = 8
            ta = MLIRArray(CUDA.CuArray(collect(Float32, 1:L)))
            to = MLIRArray(CUDA.zeros(Float32, L))
            _g_loopthrow!(backend, L)(to, ta, L; ndrange=L); CUDA.synchronize()
            @test all(Array(to) .≈ sum(1:L))                  # good input, looped + accumulated
            badv = collect(Float32, 1:L); badv[3] = -5f0
            tbad = MLIRArray(CUDA.CuArray(badv)); tbo = MLIRArray(CUDA.zeros(Float32, L))
            @test_throws CUDA.CUDACore.KernelException begin
                _g_loopthrow!(backend, L)(tbo, tbad, L; ndrange=L); CUDA.synchronize()
            end
        end

        # A void conditionally-throwing @noinline helper called for effect: DCE
        # must KEEP it (effect-free but not nothrow) so the check fires. Good input
        # runs clean; a negative element raises.
        let N = 64
            va = MLIRArray(CUDA.CuArray(fill(2f0, N))); vo = MLIRArray(CUDA.zeros(Float32, N))
            _g_voidthrow!(backend, 64)(vo, va; ndrange=N); CUDA.synchronize()
            @test Array(vo) ≈ fill(3f0, N)                     # helper kept, no false throw
            vbad = fill(2f0, N); vbad[9] = -1f0
            vba = MLIRArray(CUDA.CuArray(vbad)); vbo = MLIRArray(CUDA.zeros(Float32, N))
            @test_throws CUDA.CUDACore.KernelException begin
                _g_voidthrow!(backend, 64)(vbo, vba; ndrange=N); CUDA.synchronize()
            end
        end

        # Multi-dim @index(Local/Group, Linear): column-major over the block/grid.
        let lo = MLIRArray(CUDA.zeros(Int, 256))    # 16×16 block, 1 block: global==local
            _g_locallin!(backend, (16, 16))(lo; ndrange=(16, 16)); CUDA.synchronize()
            @test Array(lo) == collect(1:256)        # x-only would give 16 distinct values
            go = MLIRArray(CUDA.zeros(Int, 32 * 32)) # group (16,16), ndrange (32,32) → 2×2 grid
            _g_grouplin!(backend, (16, 16))(go; ndrange=(32, 32)); CUDA.synchronize()
            @test sort(unique(Array(go))) == [1, 2, 3, 4]   # x-only would give [1,2]
        end

        # N-D @atomic A[i,j]: full column-major linear element, not indices[1].
        let nb = 4, M = 4096
            rs = rand(1:nb, M); cs = rand(1:nb, M)
            drs = MLIRArray(CUDA.CuArray(Int.(rs))); dcs = MLIRArray(CUDA.CuArray(Int.(cs)))
            Hh = MLIRArray(CUDA.zeros(Int32, nb, nb))
            _g_hist2d!(backend, 256)(Hh, drs, dcs; ndrange=M); CUDA.synchronize()
            ref = zeros(Int32, nb, nb); for k in 1:M; ref[rs[k], cs[k]] += 1; end
            @test Array(Hh) == ref
        end

        # Dynamic workgroupsize (kernel built with NO wg): the launcher occupancy-
        # tunes the block via CUDA's launch_configuration, then distributes it across
        # the dims — so it launches correctly and picks a real (prod>1) block, not
        # the degenerate one-thread-per-block. (50_000 is a unique ndrange here.)
        let n = 50_000
            va = MLIRArray(CUDA.rand(Float32, n)); vb = MLIRArray(CUDA.rand(Float32, n))
            vc = MLIRArray(CUDA.zeros(Float32, n))
            _g_vadd!(backend)(vc, va, vb; ndrange=n); CUDA.synchronize()   # dynamic wg
            @test Array(vc) ≈ Array(va) .+ Array(vb)
            tuned = [v for (k, v) in MEXT._dyn_wg_cache if k[3] == (n,)]   # key: (ctx, f, nd, ats)
            @test !isempty(tuned) && all(t -> prod(t) > 1, tuned)          # occupancy-tuned, not (1,)
        end

        # A user function sharing an intrinsic marker's name must NOT be lowered to
        # the intrinsic: the result reflects the user fns (7*10+3), not block_dim.
        let N = 64
            so = MLIRArray(CUDA.zeros(Int, N))
            _g_shadow!(backend, 32)(so; ndrange=N); CUDA.synchronize()   # block_dim 32 ≠ 7
            @test all(==(73), Array(so))
        end

        # A user function shadowing the `exp` math builtin runs as the user's code
        # (identity dispatch), not math.exp.
        let N = 32
            ma = MLIRArray(CUDA.CuArray(Float32[1, 2, 3, 4][mod1.(1:N, 4)]))
            mo = MLIRArray(CUDA.zeros(Float32, N))
            _g_mathshadow!(backend, 32)(mo, ma; ndrange=N); CUDA.synchronize()
            @test Array(mo) == Array(ma) .* 3.0f0 .+ 1.0f0
        end

        # Aggregate (ComplexF32) carried out of a throwing scf.if branch: compiles
        # (typed poison yield) and is correct when the throw doesn't fire.
        let N = 64
            ca = MLIRArray(CUDA.CuArray([ComplexF32(i, 2i) for i in 1:N]))
            co = MLIRArray(CUDA.CuArray(fill(0f0 + 0f0im, N)))
            _g_cplx_throw!(backend, 64)(co, ca; ndrange=N); CUDA.synchronize()
            @test Array(co) == [ComplexF32(i, 2i) * (2f0 + 0f0im) for i in 1:N]
        end

        # Width-changing scf.for carry (Int32 acc transiently widened to Int64):
        # compiles (yield coerced to iter-arg type) and each lane sums to 8.
        let N = 32
            wo = MLIRArray(CUDA.zeros(Int, N))
            _g_widen_for!(backend, 32)(wo; ndrange=N); CUDA.synchronize()
            @test all(==(8), Array(wo))
        end

        # Nested heterogeneous struct (flat leaves [f32,i32,i32]): read a nested
        # Tuple field of a LOADED value, and reconstruct the WHOLE struct arg.
        let N = 4
            ha = [_WSNest(Float32(k), (Int32(10k), Int32(100k))) for k in 1:N]
            wa = MLIRArray(CUDA.CuArray(ha)); wro = MLIRArray(CUDA.zeros(Int32, N))
            _g_ws_read!(backend, N)(wro, wa; ndrange=N); CUDA.synchronize()
            @test Array(wro) == Int32[10k + 100k for k in 1:N]            # nested-field read
            wv = _WSNest(5.0f0, (Int32(11), Int32(22)))
            wro2 = MLIRArray(CUDA.CuArray(fill(_WSNest(0f0, (Int32(0), Int32(0))), N)))
            _g_ws_recon!(backend, N)(wro2, wv; ndrange=N); CUDA.synchronize()
            @test all(==(wv), Array(wro2))                                # whole-arg reconstruct
        end

        # A user method extending `Base.sin` on a custom type must be OUTLINED and run
        # (identity = resolved method, not the Base generic), not lowered to math.sin.
        let N = 4
            ua = MLIRArray(CUDA.CuArray(Float32[1, 2, 3, 4]))
            umlir = _ir(code_gpu, _g_user_sin!(backend, N), MLIRArray(CUDA.zeros(Float32, N)),
                        ua; ndrange=N, level=:mlir)
            @test !occursin("math.sin", umlir)                           # user body, not math op
            uo = MLIRArray(CUDA.zeros(Float32, N))
            _g_user_sin!(backend, N)(uo, ua; ndrange=N); CUDA.synchronize()
            @test Array(uo) == Float32[2, 3, 4, 5]                        # x+1 (user sin), not sin(x)
        end

        # Single-leaf nested field read (n==1 getfield branch): compiles a valid
        # extractvalue + re-wrap and returns the right value.
        let N = 4
            ha = [_GCI1((Int64(7k),), Int64(k)) for k in 1:N]
            ca = MLIRArray(CUDA.CuArray(ha)); o = MLIRArray(CUDA.zeros(Int64, N))
            _g_ci1_read!(backend, N)(o, ca; ndrange=N); CUDA.synchronize()
            @test Array(o) == Int64[7k + k for k in 1:N]
        end

        # A nested-trailing-pad struct whose flat layout diverges from Julia's is
        # rejected with a clear error (lowering to :mlir triggers the arg-type build).
        let N = 4
            ha = [_GMixedPad(Int32(k), (Float64(k), Int16(k)), Float32(k)) for k in 1:N]
            src = MLIRArray(CUDA.CuArray(ha)); dst = MLIRArray(CUDA.CuArray(copy(ha)))
            @test_throws "diverges from Julia" code_gpu(devnull, _g_mixedpad!(backend, N),
                dst, src; ndrange=N, level=:mlir)
        end
    end
end

# The kernel cache keys on the resolved MethodInstance (CompilerCaching), not the
# function object, so REDEFINING a kernel recompiles instead of serving stale PTX.
@testset "GPU: cache invalidates on kernel redefinition" begin
    if !CUDA.functional()
        @info "CUDA not functional — skipping cache-invalidation test"
        @test true
    else
        bk = GPUB(); n = 16
        a = MLIRArray(CUDA.ones(Float32, n)); c = MLIRArray(CUDA.zeros(Float32, n))
        @eval @kernel function _g_redef!(c, @Const(a))
            i = @index(Global, Linear); @inbounds c[i] = a[i] + 10.0f0
        end
        Base.invokelatest() do
            _g_redef!(bk, n)(c, a; ndrange=n); CUDA.synchronize()
        end
        @test Array(c)[1] == 11.0f0
        # Redefine with a different body — must recompile, not reuse the +10 PTX.
        @eval @kernel function _g_redef!(c, @Const(a))
            i = @index(Global, Linear); @inbounds c[i] = a[i] + 99.0f0
        end
        Base.invokelatest() do
            _g_redef!(bk, n)(c, a; ndrange=n); CUDA.synchronize()
        end
        @test Array(c)[1] == 100.0f0
    end
end

# Concurrent launches must not corrupt the (now lock-guarded) compile + workgroup
# caches. `@spawn` interleaves the cache access even on one thread, and runs truly
# parallel under `-t>1`.
@testset "GPU: concurrent launches are cache-safe" begin
    if !CUDA.functional()
        @test true
    else
        bk = GPUB(); n = 512; ntasks = 8
        oks = fill(false, ntasks)
        b = MLIRArray(CUDA.ones(Float32, n))
        @sync for i in 1:ntasks
            Threads.@spawn begin
                a = MLIRArray(CUDA.fill(Float32(i), n)); c = MLIRArray(CUDA.zeros(Float32, n))
                _g_vadd!(bk)(c, a, b; ndrange=n); CUDA.synchronize()   # dynamic wg → caches under lock
                oks[i] = Array(c) ≈ fill(Float32(i) + 1, n)
            end
        end
        @test all(oks)
    end
end

@testset "SCI optimization (DCE/CSE/LICM) affects KA codegen" begin
    if !CUDA.functional()
        @info "CUDA not functional — skipping SCI-optimization codegen test"
        @test true
    else
        N = 256
        a = MLIRArray(CUDA.rand(Float32, N)); b = MLIRArray(CUDA.rand(Float32, N))
        c = MLIRArray(CUDA.zeros(Float32, N))
        bk = get_backend(a)
        k = _g_vadd!(bk, 256)
        # `optimize_sci!` (on by default) runs in the KA compile path; the
        # `optimize` toggle lets us compare against the un-optimized lowering.
        nlines(s) = count('\n', s)
        nops(s)   = count(r"= [a-z_]+\.[a-z_]+", s)   # MLIR op-result lines
        raw_sci = _ir(code_gpu, k, c, a, b; ndrange=N, level=:sci,  optimize=false)
        opt_sci = _ir(code_gpu, k, c, a, b; ndrange=N, level=:sci,  optimize=true)
        raw_ir  = _ir(code_gpu, k, c, a, b; ndrange=N, level=:mlir, optimize=false)
        opt_ir  = _ir(code_gpu, k, c, a, b; ndrange=N, level=:mlir, optimize=true)
        @test nlines(opt_sci) < nlines(raw_sci)   # passes transform the structured IR
        @test nops(opt_ir)   <  nops(raw_ir)      # ...and that reaches the emitted MLIR
        # The optimized kernel (the default path) still computes the right result.
        k(c, a, b; ndrange=N); CUDA.synchronize()
        @test Array(c) == Array(a) .+ Array(b)
    end
end
