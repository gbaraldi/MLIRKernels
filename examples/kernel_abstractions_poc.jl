# Proof-of-concept: compiling a KernelAbstractions.jl kernel via cuTileCPU's
# MLIR pipeline, using the overlay-method-table pattern that every "real" KA
# backend (CUDA, AMDGPU, oneAPI, Metal) uses.
#
# ## Pattern
#
# Every GPU KA backend does the same three things:
#
#   1. Define `struct MyBackend <: KA.GPU end`. KA's `@kernel` macro emits
#      *two* function bodies — `gpu_foo(ctx, args...)` and `cpu_foo(ctx, args...)`
#      — and picks `gpu_foo` when `KA.isgpu(backend) == true`. The GPU body
#      is structurally the SIMT/SPMD shape cuTileCPU's SPMD walker already
#      handles (one logical thread per lane, gated by `__active_lane__`).
#
#   2. Declare a `Base.Experimental.@MethodTable` and use `@overlay` to
#      redefine the KA intrinsics — `__index_Global_Linear`, `__validindex`,
#      `__synchronize`, `SharedMemory`, `Scratchpad` — to expressions the
#      backend's compile pipeline knows how to lower. The overlay fires
#      during Julia type inference, so the inferred IR contains the overlay
#      bodies (not the KA calls) and the walker downstream just sees the
#      lowered ops.
#
#   3. Implement `(::Kernel{MyBackend})(args...; ndrange, workgroupsize)`
#      to drive the compile + launch.
#
# For cuTileCPU, step (2) reuses **cuTile's existing `cuTileMethodTable`**
# (`cuTile.jl/src/compiler/interpreter.jl:4`). cuTile already overlays
# `Base.sin`, `Base.cos`, etc. to its own `Intrinsics.*` — adding KA's
# `__index_Global_Linear` to the same table is just another `@overlay` line.
# `ct.emit_julia` then runs inference against `cuTileMethodTable` and sees
# the overlaid bodies, no walker changes needed for the intrinsics
# themselves.
#
# The one remaining walker change: a small clause that recognizes our
# sentinel function `_global_lane_id()` and binds it to the SPMD lane
# vector. That's the bridge between the overlay layer (which says "the
# global index is `_global_lane_id()`") and the MLIR layer (which says
# "the global index is `vector<W × i64> = bid*W + (0..W-1) + 1`").

using KernelAbstractions
const KA = KernelAbstractions
const NDI = KernelAbstractions.NDIteration
using cuTile
const ct = cuTile
using cuTileCPU

# ----------------------------------------------------------------------------
# 1. Backend
# ----------------------------------------------------------------------------

struct cuTileBackend <: KA.GPU end

# ----------------------------------------------------------------------------
# 2. Overlays against cuTile's method table
# ----------------------------------------------------------------------------
#
# `ct.emit_julia` runs inference under `cuTileInterpreter` whose method
# table view is `OverlayMethodTable(world, cuTileMethodTable)`. So
# `@overlay cuTile.cuTileMethodTable …` is the right place to redefine KA's
# intrinsics for our backend.

import Base.Experimental: @overlay

# Sentinel: the SPMD-walker recognizes a call to this as "return the lane
# vector value." For the POC we leave it as an opaque function; a
# `walk_call!` clause keyed on `:__cutilecpu_spmd_lane_id` would replace it
# with `lc.arg_vals[lc.lane_arg]` (the same lane vector lower_to_mlir_spmd
# synthesizes at the top of the scf.parallel body).
function __cutilecpu_spmd_lane_id end

# All KA-style "implicit per-launch context" intrinsics route through the
# `__ctx__` arg. The overlay erases the context dependency and rewrites
# each intrinsic to either a constant (trivial cases) or our sentinel
# (lane-indexed cases).

# `__validindex(ctx)` — for launches where ndrange % workgroupsize == 0,
# every lane is valid. Tighter version (lane < ndrange) is a TODO.
@overlay cuTile.cuTileMethodTable KA.__validindex(ctx) = true

# `__index_Global_Linear(ctx)` — the global linear thread index, 1-based.
# In our SPMD model, this is the lane index synthesized by lower_to_mlir_spmd.
@overlay cuTile.cuTileMethodTable KA.__index_Global_Linear(ctx) =
    __cutilecpu_spmd_lane_id()

# `__synchronize()` — CPU SIMD has no warp barrier. For kernels that don't
# use cross-lane communication this is a no-op. Kernels that do
# (`@localmem` + `@synchronize` reduction patterns) need a different
# strategy — either an outer scf.for over workitems or rejecting the
# kernel at compile time.
@overlay cuTile.cuTileMethodTable KA.__synchronize() = nothing

# `SharedMemory(::Type, ::Val{Dims}, ::Val{Id})` and
# `Scratchpad(ctx, ::Type, ::Val{Dims})` — these allocate per-workgroup or
# per-lane memory. For a POC we error if the kernel uses them; production
# would lower to `memref.alloca` (stack-allocated, per-OMP-thread).
@overlay cuTile.cuTileMethodTable KA.SharedMemory(::Type{T}, ::Val, ::Val) where {T} =
    error("cuTileBackend: @localmem / SharedMemory not yet implemented")
@overlay cuTile.cuTileMethodTable KA.Scratchpad(ctx, ::Type, ::Val) =
    error("cuTileBackend: @private / Scratchpad not yet implemented")

# ----------------------------------------------------------------------------
# 3. (Sketch) Backend launch surface
# ----------------------------------------------------------------------------
#
# A real implementation would define:
#
#   KA.get_backend, KA.allocate, KA.zeros, KA.copyto!, KA.synchronize,
#   KA.functional, KA.mkcontext, KA.launch_config, KA.argconvert,
#   (::Kernel{cuTileBackend})(args...; ndrange, workgroupsize)
#
# The kernel-callable method builds a CompilerMetadata type, calls
# something analogous to spmd_function on (obj.f, (ctx_T, arg_types...)),
# and launches with `blocks = cld(prod(ndrange), prod(workgroupsize))`.
#
# We don't implement that here — the POC stops at "verify the overlays
# fire during inference."

# ----------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------

@kernel function vadd!(C, A, B)
    i = @index(Global, Linear)
    @inbounds C[i] = A[i] + B[i]
end

println("=" ^ 60)
println("Step 1: KA picks the gpu_* variant for `cuTileBackend`")
println("=" ^ 60)
k = vadd!(cuTileBackend())
println("Kernel type: ", typeof(k))
println("Underlying fn: ", k.f)
@assert k.f === (@__MODULE__).var"gpu_vadd!" "expected the gpu_* body"
println("✓ Backend picks gpu_vadd! (the SIMT body)\n")

println("=" ^ 60)
println("Step 2: build a CompilerMetadata type matching ndrange=1024, wg=16")
println("=" ^ 60)
const N      = 1024
const WGSIZE = 16
ndrange_T = NDI.StaticSize{(N,)}
wg_T      = NDI.StaticSize{(WGSIZE,)}
groups_T  = NDI.StaticSize{(N ÷ WGSIZE,)}
ndr_obj_T = NDI.NDRange{1, groups_T, wg_T, Nothing, Nothing}
NoCheck   = NDI.NoDynamicCheck
ctx_T     = KA.CompilerMetadata{ndrange_T, NoCheck, Nothing, Nothing, ndr_obj_T}
A_T       = Vector{Float32}
println("ctx_T = $(ctx_T)\n")

println("=" ^ 60)
println("Step 3: run cuTileCPU's pipeline on gpu_vadd!")
println("        (with cuTile's interpreter consulting cuTileMethodTable +")
println("         our overlays for __validindex / __index_Global_Linear)")
println("=" ^ 60)
try
    mlir = cuTileCPU.code_mlir(k.f, (ctx_T, A_T, A_T, A_T))
    println("✓ SUCCESS — overlays fired AND walker handled the lowered body\n")
    println(mlir)
catch e
    msg = sprint(showerror, e)
    if occursin("__cutilecpu_spmd_lane_id", msg)
        println("✓ Overlays fired — KA intrinsics replaced by our sentinel")
        println("✗ Walker stopped at our sentinel (expected without the clause)")
        println("    $(first(split(msg, '\n')))\n")
        println("→ Next step: add to src/lower.jl walk_call! a clause:")
        println("      elseif fname === :__cutilecpu_spmd_lane_id")
        println("          return lc.arg_vals[lc.lane_arg]    # SPMD lane vector")
        println("      end")
        println("  Then implement a (::Kernel{cuTileBackend})(args...; ndrange,")
        println("  workgroupsize) method that builds the ctx type and dispatches")
        println("  through spmd_function-style compile+launch.\n")
        println("  Once these are in place, every KA kernel that uses only")
        println("  @index(Global, Linear) compiles via the same path — no")
        println("  per-kernel walker changes.")
    elseif occursin("KA.__validindex", msg) || occursin("KA.__index_Global_Linear", msg) ||
           occursin("__validindex", msg) || occursin("__index_Global_Linear", msg)
        println("✗ Overlays DID NOT fire — KA intrinsic still in inferred IR")
        println("    $(first(split(msg, '\n')))\n")
        println("→ Check that cuTileMethodTable was actually consulted.")
        println("  `ct.emit_julia` should run inference under cuTileInterpreter")
        println("  with `OverlayMethodTable(world, cuTileMethodTable)`.")
    else
        println("Unexpected error:")
        println("    $(first(split(msg, '\n')))\n")
        println("Full message:")
        println(msg)
    end
end
