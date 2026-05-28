module KernelAbstractionsExt

# Mirrors the CUDA.jl / AMDGPU.jl / oneAPI.jl / Metal.jl KA backend pattern:
#
#   1. `struct MLIRBackend <: KA.GPU` so KA's `@kernel` macro picks the
#      `gpu_*` (SIMT) function body, not the `cpu_*` (loop-splitting) one.
#
#   2. `@overlay MLIRKernels.Frontend.METHOD_TABLE` redefinitions of the KA
#      intrinsics. Inference runs under MLIRKernels's *own* Frontend
#      interpreter (src/frontend.jl) — NOT cuTile's — so the overlays map KA
#      intrinsics onto MLIRKernels's own `Frontend.Intrinsics` markers, which
#      the walker recognises by name. No cuTile dependency.
#
#   3. `(::Kernel{MLIRBackend})(args...; ndrange, workgroupsize)` builds the
#      KA `CompilerMetadata` type and calls `ka_function` → `lower_to_mlir_ka`
#      → in-process MLIR pipeline → clang → dlopen, then dispatches the grid
#      via the standard SPMD-style launch path.
#
# This extension no longer depends on cuTile in any way: the Frontend owns
# its interpreter, its Intrinsics module, and its overlay method table, so
# the intrinsic markers are defined at the package's own precompile and the
# overlays are ordinary precompile-safe method additions — no `__init__`
# cross-module eval.

using KernelAbstractions
const KA = KernelAbstractions
const NDI = KernelAbstractions.NDIteration
using MLIRKernels
const FE = MLIRKernels.Frontend

import Base.Experimental: @overlay

# ----------------------------------------------------------------------------
# 1. Backend
# ----------------------------------------------------------------------------

struct MLIRBackend <: KA.GPU end

# ----------------------------------------------------------------------------
# 2. Overlays into the Frontend method table
# ----------------------------------------------------------------------------
#
# Each maps a KA intrinsic onto a `Frontend.Intrinsics` marker (a @noinline +
# compilerbarrier function that survives inference under the Frontend's
# default-opt-params interpreter) which the walker intercepts. All are plain
# method additions to OUR table — precompile-safe.

# `__index_Global_Linear(ctx)` → global linear thread index (1-based).
@overlay FE.METHOD_TABLE KA.__index_Global_Linear(ctx) = FE.Intrinsics.global_index()

# `@index(Local, Linear)` / `@index(Group, Linear)` / `@groupsize` expand to
# one-arg `(ctx)` calls (the `:Linear` kind is a macro-stripped literal), so
# the unary overlay is the matching one. KA's 2-arg `(ctx, ::CartesianIndex)`
# defs in cpu.jl are CPU-emit-only and never reached on the Frontend path.
# NOTE: `groupsize` returns a SCALAR Int32 here (not an NTuple) — kernels use
# `@groupsize()` without `[1]` indexing.
@overlay FE.METHOD_TABLE KA.__index_Local_Linear(ctx) = FE.Intrinsics.local_index()
@overlay FE.METHOD_TABLE KA.__index_Group_Linear(ctx) = FE.Intrinsics.group_index()
@overlay FE.METHOD_TABLE KA.groupsize(ctx)            = FE.Intrinsics.group_size()

# `__validindex(ctx)` — for launches where ndrange is a multiple of the
# workgroup size, every lane is valid. Tighter (lane < ndrange) masking is a
# TODO (thread `ndrange` through and emit a per-lane mask compare).
@overlay FE.METHOD_TABLE KA.__validindex(ctx) = true

# `__synchronize()` → workgroup barrier marker. CPU SIMD has no warp barrier
# so the walker lowers `:barrier` to a no-op; on the GPU SIMT path with no
# cross-lane communication this is correct for the current scope.
@overlay FE.METHOD_TABLE KA.__synchronize() = FE.Intrinsics.barrier()

# `SharedMemory` / `Scratchpad` — not yet wired up (Phase B / B7,B8).
@overlay FE.METHOD_TABLE KA.SharedMemory(::Type{T}, ::Val, ::Val) where {T} =
    error("MLIRBackend: @localmem / SharedMemory not yet implemented")
@overlay FE.METHOD_TABLE KA.Scratchpad(ctx, ::Type, ::Val) =
    error("MLIRBackend: @private / Scratchpad not yet implemented")

# ----------------------------------------------------------------------------
# 3. KA backend protocol
# ----------------------------------------------------------------------------
#
# We only implement the methods the kernel call needs. Allocation /
# synchronisation / copyto! all fall back to KA's default `<: KA.GPU`
# behaviour on the host (we're CPU-targeting, so host == device).

KA.allocate(::MLIRBackend, ::Type{T}, dims::Tuple) where {T} =
    MLIRKernels.aligned_array(T, dims; alignment=128)

KA.zeros(b::MLIRBackend, ::Type{T}, dims::Tuple) where {T} =
    fill!(KA.allocate(b, T, dims), zero(T))
KA.ones(b::MLIRBackend, ::Type{T}, dims::Tuple) where {T} =
    fill!(KA.allocate(b, T, dims), one(T))

KA.synchronize(::MLIRBackend) = nothing
KA.functional(::MLIRBackend) = true
KA.argconvert(::MLIRBackend, x) = x
# NOTE: deliberately no `KA.get_backend(::Array) = MLIRBackend()`. Plain
# `Array` already dispatches to KA's `CPU()` backend, and overriding here
# would silently steal every `Array`-touching KA call from the default
# (and fails precompile with method-overwriting anyway). Users select the
# backend explicitly: `vadd!(MLIRBackend(), 16)(C, A, B; ndrange=N)`.

# `mkcontext` and `launch_config` follow KA's GPU defaults (no per-backend
# specialisation needed) by reusing the generic methods. We provide a
# minimal stub so KA's `partition` can build the CompilerMetadata type.

KA.mkcontext(kernel::KA.Kernel{MLIRBackend}, ndrange, iterspace) =
    KA.CompilerMetadata{KA.ndrange(kernel), NDI.NoDynamicCheck}(ndrange, iterspace)

function KA.launch_config(kernel::KA.Kernel{MLIRBackend}, ndrange, workgroupsize)
    ndrange isa Integer && (ndrange = (ndrange,))
    workgroupsize isa Integer && (workgroupsize = (workgroupsize,))
    if KA.workgroupsize(kernel) <: KA.NDIteration.DynamicSize && workgroupsize === nothing
        workgroupsize = (16,)  # default lane width for MLIRKernels SPMD lowering
    end
    iterspace, dynamic = KA.partition(kernel, ndrange, workgroupsize)
    return ndrange, workgroupsize, iterspace, dynamic
end

# ----------------------------------------------------------------------------
# 4. The kernel-callable launcher
# ----------------------------------------------------------------------------

# Resolve the effective workgroupsize. KA's StaticSize{(N,)} encodes it in
# the kernel type; DynamicSize falls back to the launch-time kwarg or our
# 16-lane default.
function _resolve_wgsize(obj::KA.Kernel{MLIRBackend}, workgroupsize)
    wg_T = KA.workgroupsize(obj)
    if wg_T <: NDI.StaticSize
        return NDI.get(wg_T)
    end
    workgroupsize === nothing && return (16,)
    workgroupsize isa Integer && return (workgroupsize,)
    return workgroupsize
end

function (obj::KA.Kernel{MLIRBackend})(args...; ndrange=nothing,
                                                  workgroupsize=nothing)
    wg = _resolve_wgsize(obj, workgroupsize)
    nd = ndrange isa Integer ? (ndrange,) : ndrange
    _, _, iterspace, _ = KA.launch_config(obj, nd, wg)

    ctx = KA.mkcontext(obj, nd, iterspace)
    ctx_T = typeof(ctx)
    arg_types = map(typeof, args)
    full_argtypes = Tuple{ctx_T, arg_types...}

    # Lane width = workgroup size. ndrange must be a multiple (the POC's
    # `__validindex == true` overlay assumes that).
    W = first(wg)
    total = prod(nd)
    total % W == 0 || error(
        "MLIRBackend: ndrange=$nd not a multiple of workgroupsize=$wg " *
        "— masked launches not yet supported")

    k = ka_function(obj.f, full_argtypes;
                    lane_width=W,
                    kernel_name=string(nameof(obj.f), "_ka"))
    k(args...; blocks=total ÷ W)
    return nothing
end

end # module KernelAbstractionsExt
