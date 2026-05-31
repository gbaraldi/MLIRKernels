# Standalone Julia → StructuredIRCode frontend for the SPMD / KA / GPU paths.
#
# A minimal `AbstractInterpreter` with DEFAULT optimization params — crucially
# NOT inline-everything — so the `@noinline` marker intrinsics below survive
# inference for the walker to intercept (an inline-everything policy bulldozes
# `@noinline` and leaks the marker bodies into the IR). It carries its own
# `Intrinsics` module and an overlay `MethodTable` that frontends register
# intrinsic mappings into. Inference runs via `Base.code_ircode(f, argtypes;
# interp=…)`, and `IRStructurizer.StructuredIRCode` turns the result into an SCI.
# Depends only on Core.Compiler + IRStructurizer.
module Frontend

const CC = Core.Compiler
using IRStructurizer: StructuredIRCode
using CompilerCaching: CacheView, @setup_caching, match_method_instance,
                       typeinf!, get_source

# ----------------------------------------------------------------------------
# Frontend intrinsics — markers the MLIRKernels walker recognises by name.
# ----------------------------------------------------------------------------
#
# Each is `@noinline` with a `compilerbarrier(:type, …)` body so that, under
# default optimization, the call SURVIVES inference (no inline, no
# const-fold) with a concrete return type for the walker to replace.
module Intrinsics
    using Base: compilerbarrier

    # These markers are the functions the MLIR walker pattern-matches BY NAME
    # (`fname === :__mlirkernels_*`) to recognise a KA intrinsic. They therefore
    # carry the deliberately uncommon `__mlirkernels_` prefix so a user kernel that
    # defines its own `barrier`/`local_index`/`group_size`/… is NOT mistaken for the
    # intrinsic (the walker also gates on `isintrinsic`/parentmodule, but the mangled
    # name makes a collision essentially impossible — and lets users shadow the
    # natural names freely). cuTile.jl avoids this entirely by dispatching its
    # intrinsics on function identity (`emit_intrinsic!(::typeof(Intrinsics.f))`);
    # our walker matches names, so we mangle them instead.

    # Global linear thread index (1-based, Julia semantics). The walker binds
    # this to the SPMD lane vector (CPU) or `gpu.thread_id + block_id*block_dim`
    # (GPU SIMT).
    @noinline __mlirkernels_global_index() = compilerbarrier(:type, zero(Int32))::Int32

    # Block / workgroup index along a dimension (0-based). `dim` ∈ (0,1,2).
    @noinline __mlirkernels_block_index(dim::Int32) = compilerbarrier(:type, zero(Int32))::Int32

    # Block (workgroup) dimension along an axis.
    @noinline __mlirkernels_block_dim(dim::Int32) = compilerbarrier(:type, zero(Int32))::Int32

    # Local linear index within the workgroup (1-based). CPU = lane step+1; GPU = thread_id+1.
    @noinline __mlirkernels_local_index() = compilerbarrier(:type, zero(Int32))::Int32

    # Group (workgroup/block) linear index (1-based). CPU = bid+1 (uniform); GPU = block_id+1.
    @noinline __mlirkernels_group_index() = compilerbarrier(:type, zero(Int32))::Int32

    # Workgroup size (count). CPU = lane_width const; GPU = block_dim.
    @noinline __mlirkernels_group_size() = compilerbarrier(:type, zero(Int32))::Int32

    # Workgroup barrier (CPU: no-op; GPU: gpu.barrier). Returns nothing.
    # `Base.donotdelete` is essential: the barrier has no result and is otherwise
    # effect-free, so without it DCE deletes the call before the walker sees it
    # and `@synchronize` silently vanishes — fatal for cross-lane shared memory
    # (the reads race ahead of the writes). Same fix as `atomic_index!`.
    @noinline function __mlirkernels_barrier()
        Base.donotdelete(0)
        return compilerbarrier(:type, nothing)
    end

    # Per-thread validity for tail-block masking (KA's `if __validindex(ctx)`).
    # The walker lowers it to `∧_d (global_d < ndrange[d])` on GPU (padded last
    # block's out-of-range threads do nothing), `true` on CPU. `compilerbarrier`
    # keeps it from folding so the guarding `if` survives inference.
    @noinline __mlirkernels_valid_index() = compilerbarrier(:type, true)::Bool

    # Atomic read-modify-write at a 1-based linear index. The KA extension
    # overlays `Atomix.modify!(IndexableRef, op, x, ord)` — i.e. `KA.@atomic` /
    # `Atomix.@atomic`, KA's *portable* atomic — onto this marker, stopping the
    # default-opt inline cascade before it degrades to raw pointer arithmetic +
    # an `atomicrmw` llvmcall. The walker routes it to the `memref.atomic_rmw`
    # emitter. `op` is the reduction function (+/max/min/&/|), `idx` the 1-based
    # linear index.
    #
    # The `Base.donotdelete` is essential: the marker's result is discarded (the
    # atomic is used for its memory side effect, not its value), and
    # `compilerbarrier` is itself effect-free + nothrow, so without an effect the
    # marker is inferred effect-free and DCE deletes the whole call before the
    # walker ever sees it — the atomic silently vanishes. `donotdelete` makes the
    # method `!effect_free`, so the call is preserved for the walker to rewrite.
    @noinline function __mlirkernels_atomic_index!(arr, op, val, idx)
        Base.donotdelete(arr, val, idx)
        return compilerbarrier(:type, val)
    end

    # N-D workgroup indices (1-based), as an `NTuple{N,Int}`. The KA extension
    # overlays `__index_{Global,Local,Group}_NTuple(ctx)` onto these, reading the
    # grid dimensionality `N` from the ctx type. The walker reconstructs the per-
    # dim coordinate vectors (column-major unflatten of the flat lane/block) and
    # registers them as the tuple's components, so `i, j = @index(…, NTuple)`
    # binds `i`/`j` to the right per-lane vectors. Returning a concrete-arity
    # `NTuple{N,Int}` is what lets inference destructure the result.
    @noinline __mlirkernels_global_ntuple(::Val{N}) where {N} =
        compilerbarrier(:type, ntuple(_ -> zero(Int), Val(N)))::NTuple{N,Int}
    @noinline __mlirkernels_local_ntuple(::Val{N}) where {N} =
        compilerbarrier(:type, ntuple(_ -> zero(Int), Val(N)))::NTuple{N,Int}
    @noinline __mlirkernels_group_ntuple(::Val{N}) where {N} =
        compilerbarrier(:type, ntuple(_ -> zero(Int), Val(N)))::NTuple{N,Int}

    # Workgroup shared memory (`@localmem T dims`). The KA extension overlays
    # `SharedMemory(T, Val(dims), Val(id))` onto this marker; the walker emits a
    # workgroup-address-space `memref.alloca` of shape `dims` and routes
    # `shared[…]` accesses to it. Returns an `Array{T,N}` so indexing lowers via
    # the same memoryref path as a normal array arg. `Base.donotdelete` makes the
    # call `!effect_free` so it survives DCE and isn't CSE-merged across distinct
    # `@localmem` declarations (each must be its own buffer).
    @noinline function __mlirkernels_shared_alloc(::Type{T}, ::Val{Dims}) where {T, Dims}
        Base.donotdelete(T, Dims)
        return compilerbarrier(:type,
            Array{T, length(Dims)}(undef, Dims))::Array{T, length(Dims)}
    end

    # Per-thread private memory (`@private T dims`). KA overlays
    # `Scratchpad(ctx, T, Val(dims))` onto this. Same shape as `shared_alloc`,
    # but the walker emits a DEFAULT-address-space `memref.alloca` — per-thread
    # storage (each lane its own copy), no sharing, no barrier.
    @noinline function __mlirkernels_private_alloc(::Type{T}, ::Val{Dims}) where {T, Dims}
        Base.donotdelete(T, Dims)
        return compilerbarrier(:type,
            Array{T, length(Dims)}(undef, Dims))::Array{T, length(Dims)}
    end
end

# An intrinsic is a function defined in our Intrinsics module. (We don't
# currently need NoCallInfo because default opt params already respect
# `@noinline`; kept for future use.)
isintrinsic(@nospecialize(f)) = isa(f, Function) && parentmodule(f) === Intrinsics

# ----------------------------------------------------------------------------
# Overlay method table — frontends (KA, …) register intrinsic mappings here.
# ----------------------------------------------------------------------------

Base.Experimental.@MethodTable METHOD_TABLE

# ----------------------------------------------------------------------------
# Interpreter
# ----------------------------------------------------------------------------

# A custom (non-`nothing`) cache owner is REQUIRED for overlays to apply to
# Base/stdlib callees. With `nothing`, the interpreter reuses Julia's native
# (precompiled) CodeInstances — e.g. Base's range machinery already resolved
# `steprange_last` to the un-lowerable default, so our `@overlay` was bypassed.
# A private owner forces re-inference of reachable methods through our overlay
# method table. It also shards our `CompilerCaching` results onto their own
# `CodeInstance`s. (`@setup_caching` below derives `cache_owner` from this.)
const FRONTEND_OWNER = :MLIRKernelsFrontend

# The `V` token `@setup_caching` requires: `finish!` stacks a fresh one on each
# inferred CI's `analysis_results`, which is what wires the CI into CompilerCaching's
# (cross-session-serialisable) cache keyed by our owner. We don't memoise the SCI in
# it (the SCI is mutated downstream, so it's rebuilt fresh per call — see
# `structured`); the inference cached on the CI itself is the win. Mutable so it's
# egal-matched when deserialised from a package image (mirrors cuTile's CuTileResults).
mutable struct FrontendResults
    FrontendResults() = new()
end

struct FrontendInterpreter <: CC.AbstractInterpreter
    cache::CacheView{Symbol, FrontendResults}
    method_table::CC.CachedMethodTable{CC.OverlayMethodTable}
    inf_cache::Vector{CC.InferenceResult}
    inf_params::CC.InferenceParams
    opt_params::CC.OptimizationParams
end

function FrontendInterpreter(cache::CacheView{Symbol, FrontendResults})
    mt = CC.CachedMethodTable(CC.OverlayMethodTable(cache.world, METHOD_TABLE))
    # DEFAULT OptimizationParams — crucially NOT inline_cost_threshold=typemax,
    # so `@noinline` on our intrinsics is honoured and the marker calls survive.
    return FrontendInterpreter(cache, mt, CC.InferenceResult[],
                               CC.InferenceParams(), CC.OptimizationParams())
end
FrontendInterpreter(world::UInt=Base.get_world_counter()) =
    FrontendInterpreter(CacheView{FrontendResults}(FRONTEND_OWNER, world))

CC.InferenceParams(i::FrontendInterpreter)     = i.inf_params
CC.OptimizationParams(i::FrontendInterpreter)  = i.opt_params
CC.get_inference_cache(i::FrontendInterpreter) = i.inf_cache
CC.method_table(i::FrontendInterpreter)        = i.method_table
@static if isdefined(CC, :get_inference_world)
    CC.get_inference_world(i::FrontendInterpreter) = i.cache.world
else
    CC.get_world_counter(i::FrontendInterpreter) = i.cache.world
end

# Generates `CC.cache_owner(interp) = FRONTEND_OWNER` and a `CC.finish!` that
# stacks a fresh `FrontendResults()` onto each inferred CI's analysis results.
@setup_caching FrontendInterpreter.cache

# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

"""
    structured(f, argtypes::Type) -> (sci::StructuredIRCode, rettype)

Infer `f(argtypes...)` under the Frontend interpreter (our overlays + own
intrinsics, default opt params) and structurize the resulting IRCode into a
`StructuredIRCode`.

Caching (cuTile-style, via `CompilerCaching`): the INFERENCE is memoised on the
`CodeInstance` keyed by the resolved `MethodInstance` — a hit skips re-inference
(the expensive part), and redefining a kernel makes a new method → new MI → miss
→ re-infer (no staleness). A FRESH `StructuredIRCode` is built on every call,
because the SCI is mutated downstream (`optimize_sci!` / lowering) and must NOT be
a shared cached object.
"""
function structured(@nospecialize(f), @nospecialize(argtypes::Type))
    world = Base.get_world_counter()
    cache = CacheView{FrontendResults}(FRONTEND_OWNER, world)
    mi = match_method_instance(f, argtypes)
    mi === nothing && return _structured_uncached(f, argtypes, cache)  # no unique method
    ci = get(cache, mi, nothing)
    if ci === nothing
        typeinf!(cache, FrontendInterpreter(cache), mi)               # infer + cache on the CI
        ci = get(cache, mi, nothing)
    end
    ci === nothing && return _structured_uncached(f, argtypes, cache)
    src = get_source(ci)
    src === nothing && return _structured_uncached(f, argtypes, cache)
    ir = CC.inflate_ir(src, mi)                                       # fresh IRCode from cached inference
    return StructuredIRCode(ir), CC.widenconst(ci.rettype)
end

# Uncached fallback: run inference via the convenience entry (still under our
# interpreter, so overlays apply) and structurize directly. Used when no unique
# method matches or the CI source isn't retrievable.
function _structured_uncached(@nospecialize(f), @nospecialize(argtypes::Type), cache::CacheView)
    r = Base.code_ircode(f, Tuple(argtypes.parameters); interp=FrontendInterpreter(cache))
    isempty(r) && error("Frontend.structured: inference produced no results for $f$argtypes")
    ir, rettype = r[1]
    return StructuredIRCode(ir), CC.widenconst(rettype)
end

end # module Frontend
