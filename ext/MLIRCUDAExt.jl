module MLIRCUDAExt

# KA → GPU SIMT path for MLIRKernels.
#
# `MLIRCUDABackend <: KA.GPU` makes KA emit the SIMT `gpu_*` kernel body; we
# infer it through the decoupled Frontend (the KA-intrinsic overlays live in
# `KernelAbstractionsExt`'s `Frontend.METHOD_TABLE`, populated whenever KA is
# loaded), lower to the `gpu` dialect via `lower_to_mlir_gpu`, run the
# gpu→nvvm pipeline, emit PTX through LLVM.jl's NVPTX backend, and launch with
# `cudacall`. Each GPU thread is one scalar lane — no `vector<W>`, hence none
# of the uniform/varying harmonization the CPU-SIMD path needs.
#
# Productionised from experiments/ka_to_gpu_dialect/07_ka_kernel_on_gpu.jl.

using KernelAbstractions
const KA = KernelAbstractions
const NDI = KernelAbstractions.NDIteration
using MLIRKernels
const MK = MLIRKernels
const FE = MLIRKernels.Frontend
using MLIR
const IR = MLIR.IR
const MLIRAPI = MLIR.API
using LLVM
using CUDA
import CUDA_Compiler_jll
import GPUArrays
import AcceleratedKernels as AK
using GPUArraysCore: GPUArraysCore, AbstractGPUArray, AbstractGPUArrayStyle, @allowscalar

struct MLIRCUDABackend <: KA.GPU end

# ----------------------------------------------------------------------------
# MLIRArray — the backend's array type.
# ----------------------------------------------------------------------------
#
# A thin wrapper around a CuArray whose `KA.get_backend` returns MLIRCUDABackend.
# This is what lets backend-agnostic KA code (GPUArrays, AcceleratedKernels) pick
# our backend automatically: those libraries dispatch on `get_backend(array)`, so
# wrapping the device array in an MLIRArray routes their `@kernel` launches here.
# We can't instead define `get_backend(::CuArray)` — that belongs to CUDA.jl's
# CUDABackend. The wrapper is host-indexable (for verification) and marshals by
# unwrapping to the inner CuArray.
#
# `<: AbstractGPUArray` (not just `AbstractArray`) makes GPUArrays' generic
# `broadcast`/`map!`/`fill!`/`mapreduce`/`sort` dispatch here: those are KA
# kernels launched via `get_backend`, so they compile through MLIRKernels. The
# scalar `getindex`/`setindex!` defer to the CuArray (CUDA's scalar-indexing
# guard still fires unless `@allowscalar`).
struct MLIRArray{T,N} <: AbstractGPUArray{T,N}
    data::CuArray{T,N}
end

unwrap(a::MLIRArray) = a.data
unwrap(@nospecialize(a)) = a

Base.size(a::MLIRArray) = size(a.data)
Base.getindex(a::MLIRArray, i::Int...) = getindex(a.data, i...)
Base.setindex!(a::MLIRArray, v, i::Int...) = (setindex!(a.data, v, i...); v)
Base.IndexStyle(::Type{<:MLIRArray}) = IndexLinear()
Base.similar(a::MLIRArray, ::Type{T}, dims::Dims) where {T} = MLIRArray(similar(a.data, T, dims))
Base.similar(::Type{MLIRArray{T}}, dims::Dims) where {T} = MLIRArray(CuArray{T}(undef, dims))
Base.Array(a::MLIRArray) = Array(a.data)
Base.pointer(a::MLIRArray) = pointer(a.data)
Base.strides(a::MLIRArray) = strides(a.data)
Base.elsize(::Type{MLIRArray{T,N}}) where {T,N} = sizeof(T)
Base.unsafe_convert(::Type{CUDA.CuPtr{T}}, a::MLIRArray{T}) where {T} =
    Base.unsafe_convert(CUDA.CuPtr{T}, a.data)

# ---- broadcasting ----------------------------------------------------------
# A broadcast `a .+ b` over MLIRArrays must (1) produce an MLIRArray and (2)
# launch GPUArrays' broadcast kernel through our backend. Both follow from a
# BroadcastStyle that is an `AbstractGPUArrayStyle` (so GPUArrays' overrides win
# over Base's scalar fallback) plus a `similar` that allocates an MLIRArray.
struct MLIRArrayStyle{N} <: AbstractGPUArrayStyle{N} end
MLIRArrayStyle(::Val{N}) where {N} = MLIRArrayStyle{N}()
MLIRArrayStyle{M}(::Val{N}) where {N,M} = MLIRArrayStyle{N}()
Base.Broadcast.BroadcastStyle(::Type{<:MLIRArray{T,N}}) where {T,N} = MLIRArrayStyle{N}()
# `dims` may be axis-ranges (normal broadcast → `axes(bc)`) or an integer `Dims`
# (e.g. GPUArrays' `_mapreduce` passing the reduced output shape). Normalise each
# to its length — `length.(dims)` alone would collapse integer dims to 1
# (`length(4)==1`), giving a wrong-shaped result for dims-reductions.
Base.similar(bc::Base.Broadcast.Broadcasted{MLIRArrayStyle{N}}, ::Type{T}, dims) where {T,N} =
    MLIRArray(CuArray{T}(undef, map(d -> d isa Integer ? Int(d) : length(d), dims)))
# GPUArrays materialises contiguous views / reshape / reinterpret via `derive`
# (a new array sharing storage). Delegate to the wrapped CuArray and rewrap, so a
# `view(::MLIRArray, …)` stays an MLIRArray (get_backend → our backend) instead of
# erroring. (AcceleratedKernels' block reductions `view` the source.)
GPUArrays.derive(::Type{T}, a::MLIRArray, osize::Dims, offset::Int) where {T} =
    MLIRArray(GPUArrays.derive(T, a.data, osize, offset))

# ---- reductions ------------------------------------------------------------
# GPUArrays' Base reductions (sum/prod/maximum/minimum/any/all/count/mapreduce)
# go through `GPUArrays.mapreducedim!`, which has no generic — each backend
# implements it. Rather than hand-roll a reduction kernel, delegate to
# AcceleratedKernels' `mapreduce` (a hard dep — backends depending on AK is
# established, e.g. AMDGPU.jl): it runs on the MLIRCUDABackend and handles both
# full and dims reductions. A `Broadcasted` input is materialised first
# (AK.mapreduce wants an `AbstractArray`).
function GPUArrays.mapreducedim!(f, op, R::MLIRArray,
                                 A::Union{AbstractArray,Base.Broadcast.Broadcasted};
                                 init=nothing)
    Ain = A isa Base.Broadcast.Broadcasted ? Base.materialize(A) : A
    # AK derives its `neutral` from the SOURCE eltype, which is wrong for a
    # type-changing map+reduce (e.g. `count`: Float32 → Bool → Int sum, or
    # `any`/`all`: Float32 → Bool via `|`/`&`). Pass `neutral` explicitly. When
    # the caller gives an explicit `init` (e.g. `findmax`/`argmax` seed their own
    # `(value,index)` identity, for which `neutral_element` isn't defined), use
    # it as the neutral too; otherwise ask GPUArrays for the result-type neutral.
    _init = init === nothing ? GPUArrays.neutral_element(op, eltype(R)) : init
    neutral = _init
    if length(R) == 1
        @allowscalar R[1] = AK.mapreduce(f, op, Ain; init=_init, neutral=neutral)
    else
        rdims = Tuple(d for d in 1:ndims(Ain) if size(R, d) == 1 && size(Ain, d) != 1)
        length(rdims) == 1 ||
            error("MLIRCUDABackend.mapreducedim!: only single-dim reductions (got dims=$rdims)")
        copyto!(R, AK.mapreduce(f, op, Ain; init=_init, neutral=neutral, dims=only(rdims)))
    end
    return R
end
Base.copyto!(d::MLIRArray, s::AbstractArray) = (copyto!(d.data, s); d)
Base.copyto!(d::AbstractArray, s::MLIRArray) = (copyto!(d, s.data); d)
Base.copyto!(d::MLIRArray, s::MLIRArray) = (copyto!(d.data, s.data); d)
# A view of an MLIRArray must copy to the host via the wrapped CuArray: the
# generic `AbstractArray` path is element-wise `getindex`, which trips CUDA's
# scalar-indexing guard. (AK's reduce/scan copy partial results with
# `Vector(@view dst[1:len])`.)
_cuview(s::SubArray{<:Any,<:Any,<:MLIRArray}) = view(parent(s).data, parentindices(s)...)
Base.copyto!(d::Array, s::SubArray{<:Any,<:Any,<:MLIRArray}) = (copyto!(d, _cuview(s)); d)
Base.Array(s::SubArray{T,N,<:MLIRArray}) where {T,N} = Array(_cuview(s))
CUDA.unsafe_free!(a::MLIRArray) = CUDA.unsafe_free!(a.data)

# ----------------------------------------------------------------------------
# Backend protocol — the backend's native array is MLIRArray; storage defers to
# CUDA.
# ----------------------------------------------------------------------------

KA.get_backend(::MLIRArray) = MLIRCUDABackend()
KA.allocate(::MLIRCUDABackend, ::Type{T}, dims::Tuple) where {T} = MLIRArray(CuArray{T}(undef, dims))
KA.zeros(::MLIRCUDABackend, ::Type{T}, dims::Tuple) where {T} = MLIRArray(CUDA.zeros(T, dims))
KA.ones(::MLIRCUDABackend, ::Type{T}, dims::Tuple) where {T} = MLIRArray(CUDA.ones(T, dims))
KA.synchronize(::MLIRCUDABackend) = CUDA.synchronize()
KA.functional(::MLIRCUDABackend) = CUDA.functional()
KA.supports_atomics(::MLIRCUDABackend) = true
# Data movement (host↔device, device↔device) defers to CUDA's copyto!.
KA.copyto!(::MLIRCUDABackend, dst, src) = (Base.copyto!(unwrap(dst), unwrap(src)); dst)

# Default workgroup for an N-D ndrange when none is given (GPUArrays' broadcast,
# a bare `kernel(backend)(...)`). Greedily fill up to `maxthreads` lanes starting
# at dim 1 and spilling into higher dims once a dim is exhausted — so a leading
# singleton (e.g. ndrange `(1, N)`) still gets a full block instead of the old
# dim-1-only `(1, 1)` (one thread per block → catastrophic occupancy). Each dim is
# capped by the ndrange extent and a conservative hardware limit (NVIDIA blocks:
# x,y ≤ 1024, z ≤ 64); 256 total stays well under the per-block product limit. The
# result keeps the SAME rank as `nd` (the launcher pads/masks the grid per dim).
function _default_wgsize(nd::Tuple; maxthreads::Int=256)
    wg = ones(Int, length(nd))
    budget = maxthreads
    for d in 1:length(nd)
        wg[d] = min(budget, nd[d], d <= 2 ? 1024 : 64)
        budget ÷= max(wg[d], 1)
        budget < 1 && break
    end
    return Tuple(wg)
end

# GPUArrays' generic `map!`/`broadcast` call `KA.launch_config(kernel, ndrange,
# workgroupsize)` and launch with `config[1]`/`config[2]`. Our launcher takes
# `ndrange`/`workgroupsize` directly and pads+masks the grid, so we just
# normalise to tuples and pick a default block size. (iterspace/dynamic — the
# 3rd/4th elements KA's own launcher uses — are unused by GPUArrays.)
@inline function KA.launch_config(::KA.Kernel{MLIRCUDABackend}, ndrange, workgroupsize)
    ndrange isa Integer && (ndrange = (ndrange,))
    workgroupsize isa Integer && (workgroupsize = (workgroupsize,))
    if workgroupsize === nothing
        workgroupsize = _default_wgsize(ndrange)
    end
    return ndrange, workgroupsize, nothing, nothing
end

# ----------------------------------------------------------------------------
# GPU compilation: SCI → gpu.module → PTX → CuFunction (cached).
# ----------------------------------------------------------------------------

# Target selection — mirror CUDA.jl. The codegen target is the host GPU's compute
# capability and the newest PTX ISA that both LLVM and the CUDA runtime support.
# CUDA derives exactly this in `compiler_config(dev)` (clamping the device cap to
# LLVM's supported set and picking the highest available PTX ISA), so we reuse that
# memoised path and pull its `PTXCompilerTarget`. Result: our device code targets
# the same `(sm_XY, +ptxNN)` CUDA.jl picks for its own kernels — instead of a
# hardcoded `sm_90/+ptx80` that is wrong on any non-Hopper GPU and stale on the ISA.
# (`GPUCompiler` is reached through CUDACore, which `using`s it — same coupling as
# `CUDA.CUDACore.create_exceptions!` for the device-exception path.)
const GPUCompiler = CUDA.CUDACore.GPUCompiler

_device_target() = CUDA.CUDACore.compiler_config(CUDA.device()).target

# PTXCompilerTarget → the MLIR `nvvm-attach-target` / LLVM TargetMachine strings.
_target_sm(t)   = "sm_$(t.cap.major)$(t.cap.minor)"
_target_feat(t) = "+ptx$(t.ptx.major)$(t.ptx.minor)"

# Inverse, for `code_gpu` reflection of a non-host arch: "sm_90"→v"9.0",
# "sm_120"→v"12.0" (the last digit is the minor); "+ptx87"→v"8.7".
_parse_cap(s)  = (m = match(r"^sm_(\d+)(\d)$", s);
                  m === nothing ? error("MLIRCUDABackend: bad sm string $s") :
                  VersionNumber(parse(Int, m[1]), parse(Int, m[2])))
_parse_ptx(s)  = (m = match(r"^\+ptx(\d+)(\d)$", s);
                  m === nothing ? error("MLIRCUDABackend: bad feature string $s") :
                  VersionNumber(parse(Int, m[1]), parse(Int, m[2])))

# Resolve the codegen target: the host device by default; `sm`/`feat` (either may
# be `nothing`) override the cap/ISA for reflection.
function _resolve_target(sm, feat)
    (sm === nothing && feat === nothing) && return _device_target()
    t = _device_target()
    cap = sm   === nothing ? t.cap : _parse_cap(sm)
    ptx = feat === nothing ? t.ptx : _parse_ptx(feat)
    return GPUCompiler.PTXCompilerTarget(; cap, ptx)
end

# The GPU lowering pipeline, parameterised by target. Only `nvvm-attach-target`
# depends on the target (chip=sm, features=ptx ISA) — the rest are fixed. Building
# it per-request (rather than a const baking sm_90/+ptx80) keeps the NVVM target
# attribute consistent with the LLVM `TargetMachine` and the `_gpu_cache` key when
# `sm`/`feat` are non-default; otherwise a non-sm_90 launch silently lowered its
# device code for sm_90. The same `feat` string ("+ptxNN") feeds both stages.
function _gpu_passes(sm::AbstractString, feat::AbstractString)
    return String[
        # Inline outlined device calls (func.call → func.func emitted for un-inlined
        # `:invoke`s) and drop the now-unused func.funcs. No-op when there are none.
        "inline", "symbol-dce",
        "nvvm-attach-target{chip=$sm features=$feat}",
        "gpu-kernel-outlining",
        "gpu.module(convert-gpu-to-nvvm)",
        "convert-scf-to-cf", "convert-cf-to-llvm", "convert-arith-to-llvm",
        "convert-vector-to-llvm",   # struct-as-vector<N×T> element: extract/insert/load
        "expand-strided-metadata", "finalize-memref-to-llvm",
        "convert-nvvm-to-llvm", "reconcile-unrealized-casts",
        "gpu-module-to-binary{format=llvm}",
    ]
end

# PTX identifiers can't contain `!` etc.; sanitise the kernel symbol.
_sym(f) = replace(string(nameof(f)), r"[^A-Za-z0-9_]" => "_")

const _gpu_cache = Dict{Any, Tuple{CUDA.CuFunction, Vector{Symbol}}}()

# Occupancy-tuned default block size (mirror CUDA.jl): per (kernel, ndrange,
# host-argtypes), the workgroup the driver's occupancy API suggested. Lets repeat
# launches skip the provisional compile + re-query. (Plain Dict, like _gpu_cache.)
const _dyn_wg_cache = Dict{Any, Tuple}()

function _extract_gpu_binary(mod)
    for op in IR.body(mod)
        IR.name(op) == "gpu.binary" || continue
        objs = IR.getattr(op, "objects")
        o0 = IR.Attribute(MLIRAPI.mlirArrayAttrGetElement(objs, 0))
        sr = MLIRAPI.mlirGPUObjectAttrGetObject(o0)
        return copy(unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(sr.data), sr.length; own=false))
    end
    error("MLIRCUDABackend: gpu pipeline produced no gpu.binary")
end

# Run an MLIR pass pipeline (list of pass strings) on `mod` in place.
# `IR.PassManager()` reads `current_context()`, so the context must be active for
# the call — `@with_context` activates/deactivates as a BALANCED pair. The old
# code activated without ever deactivating, so every compile pushed another entry
# onto the task-local context stack, which both grew unboundedly AND pinned the C
# context alive (defeating `IR.dispose` in the caller).
function _run_passes!(mod, mlir_ctx, passes)
    MK.@with_context mlir_ctx begin
        pm = IR.PassManager()
        parse(IR.OpPassManager(pm), "builtin.module(" * join(passes, ",") * ")")
        MLIRAPI.mlirPassManagerRunOnOp(pm, IR.Operation(mod)).value == 0 &&
            error("MLIRCUDABackend: GPU pass pipeline failed")
    end
    return mod
end

# Resolve libdevice externs (`__nv_fabsf`, `__nv_sqrtf`, …) that the gpu→nvvm
# pipeline emits for math ops, by linking NVIDIA's libdevice bitcode. We can't
# reuse GPUCompiler's `link_libraries!`/`compile` — those are keyed on a
# `CompilerJob` built from a Julia `MethodInstance`, which we don't have (our IR
# comes from MLIR, not Julia inference). So we link directly with LLVM.jl's
# `link!(...; only_needed=true)` — the job-free path GPUCompiler's own
# deprecation notice recommends — pulling the .bc from `CUDA_Compiler_jll`. The
# NVPTX backend runs NVVMReflect during codegen, resolving libdevice's
# `__nvvm_reflect` calls. No-op when there are no `__nv_*` references.
function _link_libdevice!(lmod)
    any(f -> LLVM.isdeclaration(f) && startswith(LLVM.name(f), "__nv_"),
        LLVM.functions(lmod)) || return
    lib = parse(LLVM.Module, read(CUDA_Compiler_jll.libdevice); lazy=true)
    LLVM.triple!(lib, LLVM.triple(lmod))
    LLVM.datalayout!(lib, LLVM.datalayout(lmod))
    LLVM.link!(lmod, lib; only_needed=true)
    return
end

# gpu.binary{format=llvm} bitcode → PTX string, with libdevice linked and LLVM's
# default -O2 run. The driver JITs PTX → SASS at module load. `stages`, when given,
# captures the LLVM IR before (`:llvm_unopt`) and after (`:llvm`) the -O2 pipeline —
# for reflection / `code_gpu`.
#
# Mirrors GPUCompiler's LLVM usage (driver.jl `JuliaContext`, mcgen.jl/optim.jl):
# the whole emission runs inside an LLVM context do-block (active for the duration,
# disposed on exit, like `JuliaContext()`), and the `TargetMachine` is built by
# GPUCompiler's own `llvm_machine(target)` so its triple/datalayout/cpu/features
# match exactly what CUDA.jl uses — we just dispose it (GPUCompiler leaves it to the
# GC). The PTX String is context-independent, so nothing is leaked per compile.
function _bitcode_to_ptx(bc, target;
                         stages::Union{Nothing,Dict{Symbol,String}}=nothing)
    LLVM.Context() do lctx
        lmod = parse(LLVM.Module, bc)
        LLVM.triple!(lmod, GPUCompiler.llvm_triple(target))
        _link_libdevice!(lmod)            # parse libdevice in the SAME context, then link
        tm = GPUCompiler.llvm_machine(target)
        tm === nothing && error("MLIRCUDABackend: NVPTX backend unavailable in this LLVM")
        try
            stages === nothing || (stages[:llvm_unopt] = string(lmod))   # linked, pre-O2
            # We emit no LLVM-level optimization of our own; run LLVM's default -O2
            # pipeline (inline/GVN/DCE) via the new pass manager — also strips the
            # lazily-linked libdevice down to the referenced functions. LLVM ≥17
            # weaves NVPTX's NVVMReflect in at PipelineStart, resolving libdevice's
            # `__nvvm_reflect`. Job-free (GPUCompiler's `optimize!` builds a custom
            # NewPMPassBuilder pipeline keyed on a CompilerJob we don't have).
            LLVM.run!("default<O2>", lmod, tm)
            stages === nothing || (stages[:llvm] = string(lmod))         # post-O2
            String(LLVM.emit(tm, lmod, LLVM.API.LLVMAssemblyFile))
        finally
            LLVM.dispose(tm)
        end
    end
end

# Env-var kernel dumping. `MLIRKERNELS_DUMP` = a comma-separated subset of
# sci,mlir,lowered,llvm_unopt,llvm,ptx (or "all") prints those levels for every
# GPU kernel — `llvm_unopt`/`llvm` are the LLVM IR before/after the -O2 pipeline
# as it compiles; `MLIRKERNELS_DUMP_FILTER=<substr>` restricts to kernels whose
# name contains <substr>. Best-effort (dumps whatever stages succeed) and goes
# to stderr, so it works even when a kernel is launched deep inside a library.
const _DUMP_ORDER = (:sci, :mlir, :lowered, :llvm_unopt, :llvm, :ptx)

function _maybe_dump_kernel(f, full_argtypes, kname; sm, feat, nd_dims, optimize=true)
    spec = get(ENV, "MLIRKERNELS_DUMP", "")
    isempty(spec) && return
    filt = get(ENV, "MLIRKERNELS_DUMP_FILTER", "")
    (isempty(filt) || occursin(filt, String(kname))) || return
    want = spec == "all" ? collect(_DUMP_ORDER) :
           Symbol[Symbol(strip(s)) for s in split(spec, ',') if !isempty(strip(s))]
    idxs = filter(!isnothing, [findfirst(==(l), _DUMP_ORDER) for l in want])
    isempty(idxs) && return
    upto = _DUMP_ORDER[maximum(idxs)]
    stages = try
        _codegen_stages(f, full_argtypes; sm, feat, upto, nd_dims, optimize)
    catch e
        printstyled(stderr, "===== [MLIRKernels dump] $kname: staging failed at :$upto =====\n";
                    color=:red, bold=true)
        showerror(stderr, e); println(stderr)
        return
    end
    for lvl in _DUMP_ORDER
        (lvl in want && haskey(stages, lvl)) || continue
        printstyled(stderr, "===== [MLIRKernels dump] $kname :$lvl =====\n"; color=:cyan, bold=true)
        println(stderr, stages[lvl])
    end
    return
end

function _compile(f, full_argtypes; sm=nothing, feat=nothing, nd_dims=Int[], optimize::Bool=true)
    target = _resolve_target(sm, feat)
    sm, feat = _target_sm(target), _target_feat(target)   # honest, device-derived key
    key = (f, full_argtypes, sm, feat, nd_dims, optimize)
    haskey(_gpu_cache, key) && return _gpu_cache[key]
    kname = _sym(f)
    _maybe_dump_kernel(f, full_argtypes, kname; sm, feat, nd_dims, optimize)

    sci, rettype = FE.structured(f, full_argtypes)
    (rettype === Nothing || rettype === Union{}) ||
        @warn "MLIRCUDABackend: kernel inferred rettype = $rettype (expected Nothing)"

    # ctx (`__ctx__`) is arg slot 2 (slot 1 is the function itself).
    mod, _pjt, mlir_ctx, kinds =
        MK.lower_to_mlir_gpu(sci, full_argtypes; kernel_name=kname, ctx_arg=2, nd_dims, optimize)

    # The MLIR context owns `mod`; once we've extracted the binary the PTX is a
    # plain String, so dispose the context (try/finally — free it even on a pass
    # failure). Without this every cache-miss leaked an MLIR context.
    ptx = try
        _run_passes!(mod, mlir_ctx, _gpu_passes(sm, feat))
        bc = _extract_gpu_binary(mod)
        _bitcode_to_ptx(bc, target)
    finally
        IR.dispose(mlir_ctx)
    end

    cumod = CuModule(ptx)
    _wire_exception_flag!(cumod)
    cufn = CuFunction(cumod, kname)
    _gpu_cache[key] = (cufn, kinds)
    return cufn, kinds
end

# If the kernel has throws, the lowering emits a module global `@__mlirkernels_exc`
# holding a pointer to CUDA's per-context `ExceptionInfo`. Point it there: a device
# throw then sets `status=1`, which CUDA's `check_exceptions()` (run in every
# `CUDA.synchronize()`) detects and raises a `KernelException` for — reusing
# CUDA.jl's mature host-side machinery. No global ⇒ kernel can't throw ⇒ no-op.
function _wire_exception_flag!(cumod::CuModule)
    g = try
        CuGlobal{UInt64}(cumod, "__mlirkernels_exc")
    catch
        return nothing                      # symbol absent → kernel has no throws
    end
    # `create_exceptions!` registers (and zeroes) the per-context ExceptionInfo and
    # returns the HOST pointer to that host-pinned MEMHOSTALLOC_DEVICEMAP buffer —
    # NOT the CuPtr from cuMemHostGetDevicePointer. The device dereferences this host
    # VA directly, which is valid because Unified Virtual Addressing makes it
    # device-accessible; it's exactly the pointer CUDA.jl threads into its own device
    # runtime, so we inherit the same UVA requirement as stock CUDA.jl (no new
    # assumption). `status` is field 0, so our device store of `i32 1` at this base
    # sets it. `check_exceptions()` (run by every `CUDA.synchronize()`) then raises.
    # (Exception infra lives in the CUDACore package, reached via `CUDA.CUDACore`.)
    ptr = CUDA.CUDACore.create_exceptions!(cumod)
    g[] = UInt64(UInt(ptr))
    return nothing
end

# ----------------------------------------------------------------------------
# Reflection — capture every codegen level for the GPU path. Mirrors the
# `_compile` pipeline but stops at, and returns the text of, each stage.
# ----------------------------------------------------------------------------

function _codegen_stages(f, full_argtypes; sm=nothing, feat=nothing,
                         upto::Symbol=:ptx, nd_dims=Int[], optimize::Bool=true)
    kname = _sym(f)
    target = _resolve_target(sm, feat)
    sm, feat = _target_sm(target), _target_feat(target)
    # `:lowered` is the full pipeline minus the final `gpu-module-to-binary` (so
    # the gpu.module is still readable LLVM/NVVM-dialect MLIR, not a binary blob).
    passes = _gpu_passes(sm, feat)
    passes_nobin = passes[1:end-1]
    order = (:sci, :mlir, :lowered, :llvm_unopt, :llvm, :ptx)
    want = findfirst(==(upto), order)
    want === nothing && error("code_gpu: unknown level :$upto (one of $order)")
    out = Dict{Symbol,String}()

    sci, _ = FE.structured(f, full_argtypes)
    # Optimize the SCI here (so the :sci level reflects the toggle); then lower
    # with optimize=false to avoid running the passes twice.
    optimize && MK.SCIOpt.optimize_sci!(sci)
    out[:sci] = sprint(show, sci)
    want == 1 && return out

    mod, _pjt, mlir_ctx, _kinds =
        MK.lower_to_mlir_gpu(sci, full_argtypes; kernel_name=kname, ctx_arg=2, nd_dims, optimize=false)
    # try/finally so the context is disposed even on the early `return out`s below.
    try
        MK.@with_context mlir_ctx begin
            out[:mlir] = sprint(show, mod)
            want == 2 && return out
            # Lower to LLVM/NVVM dialect (everything but serialise-to-binary).
            _run_passes!(mod, mlir_ctx, passes_nobin)
            out[:lowered] = sprint(show, mod)
            want == 3 && return out
            # Serialise to gpu.binary, extract bitcode → LLVM IR (pre/post-O2) + PTX.
            _run_passes!(mod, mlir_ctx, passes[end:end])
            bc = _extract_gpu_binary(mod)
            ptx = _bitcode_to_ptx(bc, target, stages=out)  # fills :llvm_unopt + :llvm
            want >= findfirst(==(:ptx), order) && (out[:ptx] = ptx)
        end
    finally
        IR.dispose(mlir_ctx)
    end
    return out
end

# ----------------------------------------------------------------------------
# Launch argument marshalling.
# ----------------------------------------------------------------------------
#
# Each `memref<…>` kernel param lowers (LLVM memref ABI) to the descriptor
# fields {allocated_ptr, aligned_ptr, offset, sizes…, strides…} passed as
# individual scalar params. A `:scalar` param passes its value directly.

function _push_memref!(flat, sig, arr::CuArray)
    p = UInt64(UInt(pointer(arr)))
    push!(flat, p);          push!(sig, Culonglong)   # allocated ptr
    push!(flat, p);          push!(sig, Culonglong)   # aligned ptr
    push!(flat, UInt64(0));  push!(sig, Culonglong)   # offset (elements)
    # The kernel addresses the array via Julia (column-major) linearisation and
    # reads `size(a,k)` as `memref.dim(a, N-k)` (Julia↔MLIR dim reversal). So the
    # LLVM descriptor's sizes/strides must be in REVERSED Julia order. (For a
    # 1-D arg `reverse` is a no-op, so vadd is unchanged.)
    for s in reverse(size(arr));    push!(flat, UInt64(s)); push!(sig, Culonglong); end
    for s in reverse(strides(arr)); push!(flat, UInt64(s)); push!(sig, Culonglong); end
    return nothing
end

# Flatten a launch arg to match the flattened param list (see
# lower_to_mlir_gpu): drop singletons (`Val`/`Type`/captured user fns — folded
# as Core.Const), unwrap arrays to their CuArray, and expand a closure/functor
# into its captured array+scalar fields (fieldname order — matching the
# signature flattening).
function _flatten_args!(out, @nospecialize(a))
    Base.issingletontype(typeof(a)) && return out
    au = unwrap(a)
    if au isa CuArray || au isa Number
        push!(out, au)
    elseif isstructtype(typeof(a))
        for fn in fieldnames(typeof(a))
            _flatten_args!(out, getfield(a, fn))
        end
    else
        error("MLIRCUDABackend: cannot marshal arg of type $(typeof(a))")
    end
    return out
end

function _marshal(args, kinds)
    # `kinds` is one symbol per flattened *param* (memref/scalar). Flatten the
    # runtime args the same way the signature was flattened, then they line up.
    flat_vals = Any[]
    for a in args; _flatten_args!(flat_vals, a); end
    length(kinds) == length(flat_vals) ||
        error("MLIRCUDABackend: $(length(kinds)) params vs $(length(flat_vals)) marshalled values")
    flat = Any[]; sig = DataType[]
    for (a, k) in zip(flat_vals, kinds)
        if k === :memref
            a isa CuArray ||
                error("MLIRCUDABackend: memref param expects a CuArray, got $(typeof(a))")
            _push_memref!(flat, sig, a)
        elseif k === :scalar
            push!(flat, a); push!(sig, typeof(a))
        else
            error("MLIRCUDABackend: unsupported param kind :$k")
        end
    end
    return flat, sig
end

# ----------------------------------------------------------------------------
# ctx type + launch geometry.
# ----------------------------------------------------------------------------

# Device array types trip scalar-indexing guards during inference; map them to
# the host `Array` of the same eltype/rank (which lowers identically).
# Does a type contain a device array anywhere in its parameter tree?
_has_device_array(@nospecialize(x)) = false
function _has_device_array(@nospecialize(T::Type))
    (T <: CuArray || T <: MLIRArray) && return true
    isconcretetype(T) && isstructtype(T) &&
        any(p -> _has_device_array(p), T.parameters)
end

_host_argtype(::Type{<:CuArray{T,N}}) where {T,N} = Array{T,N}
_host_argtype(::Type{<:MLIRArray{T,N}}) where {T,N} = Array{T,N}
function _host_argtype(@nospecialize(T::Type))
    # A wrapper/closure carrying device arrays (a closure's captures, a
    # SubArray's `.parent`, …): rebuild the type with every device-array type
    # param remapped to a host Array, recursively. Then inference indexes those
    # arrays via Array's getindex (no GPUArrays.assertscalar in the kernel IR)
    # and inlines; marshalling still unwraps the real device arrays. Element
    # type/ndims are unchanged, so the flattened memref params match. Only Type
    # params are remapped — value params (ndims, flags) are kept verbatim.
    _has_device_array(T) || return T
    try
        return T.name.wrapper{map(p -> p isa Type ? _host_argtype(p) : p,
                                  collect(T.parameters))...}
    catch
        return T
    end
end

function _resolve_wgsize(obj::KA.Kernel{MLIRCUDABackend}, workgroupsize, nd::Tuple)
    wg_T = KA.workgroupsize(obj)
    if wg_T <: NDI.StaticSize
        static = NDI.get(wg_T)
        if workgroupsize !== nothing
            wg = workgroupsize isa Integer ? (workgroupsize,) : Tuple(workgroupsize)
            wg == static || error(
                "MLIRCUDABackend: workgroupsize=$wg conflicts with the kernel's " *
                "static workgroupsize $static.")
        end
        return static
    end
    # Default block: greedily fill up to 256 lanes across the ndrange dims (see
    # `_default_wgsize`), keeping the SAME rank as the ndrange — GPUArrays' broadcast
    # launches an N-D ndrange with no workgroupsize, so a fixed `(256,)` would
    # mismatch the rank, and a leading-singleton ndrange must not collapse to 1 lane.
    if workgroupsize === nothing
        return _default_wgsize(nd)
    end
    workgroupsize isa Integer && return (workgroupsize,)
    return Tuple(workgroupsize)
end

# Build the `CompilerMetadata` *type* for inference. Static sizes give the
# grid dimensionality (used by the N-D `@index` overlays) and a clean ctx.
function _ctx_type(nd::NTuple{D,Int}, wg::NTuple{D,Int}) where {D}
    ndr = NDI.StaticSize{nd}
    wgs = NDI.StaticSize{wg}
    grp = NDI.StaticSize{map(cld, nd, wg)}
    ndobj = NDI.NDRange{D, grp, wgs, Nothing, Nothing}
    return KA.CompilerMetadata{ndr, NDI.NoDynamicCheck, Nothing, Nothing, ndobj}
end

# Shared by the launcher and `code_gpu`: resolve geometry + build the inference
# signature. Infer with HOST array types — the SCI walk only needs each arg's
# eltype/ndims (→ `memref<?×…×T>`), and inferring a kernel body's `A[i]` on a
# `CuArray` trips `GPUArrays.assertscalar`. (Marshalling still uses the real
# device arrays.)
# ndrange: explicit kwarg wins; otherwise fall back to the kernel's STATIC
# ndrange (baked in via `kernel(backend, wg, ndrange)`), as KA's testsuite does.
function _resolve_ndrange(obj::KA.Kernel{MLIRCUDABackend}, ndrange)
    if ndrange !== nothing
        return ndrange isa Integer ? (ndrange,) : Tuple(ndrange)
    end
    nd_T = KA.ndrange(obj)
    nd_T <: NDI.StaticSize ||
        error("MLIRCUDABackend: ndrange must be specified (kernel has no static ndrange)")
    return NDI.get(nd_T)
end

function _launch_setup(obj::KA.Kernel{MLIRCUDABackend}, args, ndrange, workgroupsize)
    nd = _resolve_ndrange(obj, ndrange)
    wg = _resolve_wgsize(obj, workgroupsize, nd)
    length(wg) == length(nd) || error(
        "MLIRCUDABackend: ndrange $nd ($(length(nd))-D) and workgroupsize $wg " *
        "($(length(wg))-D) must have the same number of dimensions.")
    # ndrange need not be a multiple of wg: the grid is padded (`cld`) and
    # `__validindex` masks the tail. (`unsafe_indices=true` skips the mask, so it
    # still needs an exact-multiple ndrange.)
    ctxT = _ctx_type(nd, wg)
    full_argtypes = Tuple{ctxT, map(a -> _host_argtype(typeof(a)), args)...}
    return full_argtypes, nd, wg
end

function (obj::KA.Kernel{MLIRCUDABackend})(args...; ndrange=nothing,
                                                     workgroupsize=nothing)
    nd = _resolve_ndrange(obj, ndrange)
    host_ats = map(a -> _host_argtype(typeof(a)), args)
    # "Dynamic" = no static workgroupsize and none passed → we pick the block size,
    # so it's eligible for occupancy tuning (exactly CUDA.jl's gate).
    dynamic = (KA.workgroupsize(obj) <: NDI.DynamicSize) && workgroupsize === nothing
    cachekey = (obj.f, nd, host_ats)

    wg = dynamic && haskey(_dyn_wg_cache, cachekey) ?
         _dyn_wg_cache[cachekey] : _resolve_wgsize(obj, workgroupsize, nd)
    length(wg) == length(nd) || error(
        "MLIRCUDABackend: ndrange $nd ($(length(nd))-D) and workgroupsize $wg " *
        "($(length(wg))-D) must have the same number of dimensions.")
    cufn, kinds = _compile(obj.f, Tuple{_ctx_type(nd, wg), host_ats...}; nd_dims=Int[nd...])

    # Untuned dynamic launch: mirror CUDA.jl's `threads_to_workgroupsize`. The CUDA
    # occupancy API (cuOccupancyMaxPotentialBlockSize, via `launch_configuration`)
    # picks the block size that maximises THIS kernel's occupancy given its register/
    # shared-mem use — instead of a fixed 256 — and `_default_wgsize` distributes that
    # budget across the ndrange dims (the same greedy fill CUDA uses). `wg` is baked
    # into the inference signature (via `_ctx_type`), so recompile if it changed; the
    # result is cached so later launches compile straight to the tuned block.
    if dynamic && !haskey(_dyn_wg_cache, cachekey)
        cfg = CUDA.launch_configuration(cufn; max_threads=prod(nd))
        wg_tuned = _default_wgsize(nd; maxthreads=min(prod(nd), cfg.threads))
        if wg_tuned != wg
            wg = wg_tuned
            cufn, kinds = _compile(obj.f, Tuple{_ctx_type(nd, wg), host_ats...}; nd_dims=Int[nd...])
        end
        _dyn_wg_cache[cachekey] = wg
    end

    flat, sig = _marshal(args, kinds)
    grid = map(cld, nd, wg)          # blocks per dim (padded)
    cudacall(cufn, Tuple{sig...}, flat...; threads=wg, blocks=grid)
    return nothing
end

# ----------------------------------------------------------------------------
# code_gpu — reflection entry points (see MLIRKernels.code_gpu docstring).
# ----------------------------------------------------------------------------

# Like CUDA.jl's `code_ptx`/`code_llvm`: PRINT the IR (from each level's own
# printer) to `io` (default stdout) and return nothing. The text is captured per
# stage in `_codegen_stages` (the IR objects mutate in place across the pipeline,
# so a snapshot is required). Capture via `sprint(io -> code_gpu(io, …))`.

# Low-level form: explicit (gpu_body, full_argtypes::Type). `optimize` toggles the
# SCI optimization passes (DCE/CSE/LICM) — handy for opt-vs-raw codegen diffs.
function MK.code_gpu(io::IO, @nospecialize(f), full_argtypes::Type; level::Symbol=:ptx,
                     sm=nothing, feat=nothing, nd_dims=Int[], optimize::Bool=true)
    stages = _codegen_stages(f, full_argtypes; sm, feat, upto=level, nd_dims, optimize)
    print(io, stages[level])
    return nothing
end
MK.code_gpu(@nospecialize(f), full_argtypes::Type; kwargs...) =
    MK.code_gpu(stdout, f, full_argtypes; kwargs...)

# Ergonomic form: a KA kernel + launch args (mirrors a `(obj)(args…; ndrange)`).
function MK.code_gpu(io::IO, obj::KA.Kernel{MLIRCUDABackend}, args...; level::Symbol=:ptx,
                     ndrange=nothing, workgroupsize=nothing, sm=nothing, feat=nothing,
                     optimize::Bool=true)
    full_argtypes, nd, _wg = _launch_setup(obj, args, ndrange, workgroupsize)
    return MK.code_gpu(io, obj.f, full_argtypes; level, sm, feat, nd_dims=Int[nd...], optimize)
end
MK.code_gpu(obj::KA.Kernel{MLIRCUDABackend}, args...; kwargs...) =
    MK.code_gpu(stdout, obj, args...; kwargs...)

end # module MLIRCUDAExt
