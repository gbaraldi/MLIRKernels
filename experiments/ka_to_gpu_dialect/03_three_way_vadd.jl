# Three-way vadd comparison on the H100:
#
#   (a) MLIR `gpu.module` (our 02 pipeline)   — handwritten gpu-dialect IR,
#                                                lowered + handed to LLVM's
#                                                NVPTX backend via LLVM.jl,
#                                                launched with cudacall
#
#   (b) CUDA.jl SIMT (`@cuda` kernel)         — the canonical Julia-on-GPU
#                                                style: `threadIdx().x`,
#                                                `blockIdx().x` etc.,
#                                                compiled via GPUCompiler
#
#   (c) cuTile tile-based kernel              — `ct.bid(1)`, `ct.load`,
#                                                `ct.store`; cuTile's whole-
#                                                tile abstraction lowered
#                                                via bytecode + tileiras
#                                                (closed-source) to CUBIN
#
# Outputs:
#   - PTX side-by-side for (a) and (b). cuTile doesn't expose PTX
#     directly (its tileiras backend produces CUBIN); we still time it
#     for runtime comparison.
#   - Runtime in μs and effective DRAM bandwidth (GB/s) for all three.

using cuTileCPU
using MLIR
const IR = MLIR.IR
using LLVM
using CUDA
using cuTile
const ct = cuTile

const HERE = @__DIR__

# ---------------------------------------------------------------------------
# (a) MLIR pipeline reproduced from 02_extract_and_launch.jl
# ---------------------------------------------------------------------------

const GPU_PASSES = String[
    "nvvm-attach-target{chip=sm_90 features=+ptx80}",
    "gpu-kernel-outlining",
    "gpu.module(convert-gpu-to-nvvm)",
    "convert-scf-to-cf",
    "convert-cf-to-llvm",
    "convert-arith-to-llvm",
    "expand-strided-metadata",
    "finalize-memref-to-llvm",
    "convert-nvvm-to-llvm",
    "reconcile-unrealized-casts",
    "gpu-module-to-binary{format=llvm}",
]
_pipeline_str(passes) = "builtin.module(" * join(passes, ",") * ")"

function _stringref_to_bytes(sr::MLIR.API.MlirStringRef)
    return copy(unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(sr.data), sr.length; own=false))
end
function extract_object_bytes(mod::IR.Module)
    for op in IR.body(mod)
        IR.name(op) == "gpu.binary" || continue
        objects_attr = IR.getattr(op, "objects")
        first_obj = IR.Attribute(MLIR.API.mlirArrayAttrGetElement(objects_attr, 0))
        return _stringref_to_bytes(MLIR.API.mlirGPUObjectAttrGetObject(first_obj))
    end
    error("no gpu.binary")
end

function build_mlir_kernel()
    ctx = cuTileCPU.fresh_context()
    IR.activate(ctx)
    mlir_text = read(joinpath(HERE, "01_handwritten_vadd_gpu.mlir"), String)
    mod = parse(IR.Module, mlir_text)
    pm = IR.PassManager()
    parse(IR.OpPassManager(pm), _pipeline_str(GPU_PASSES))
    status = MLIR.API.mlirPassManagerRunOnOp(pm, IR.Operation(mod))
    status.value == 0 && error("pipeline failed")

    bc = extract_object_bytes(mod)
    llvm_ctx = LLVM.Context()
    llvm_mod = LLVM.context!(llvm_ctx) do
        parse(LLVM.Module, bc)
    end
    triple = "nvptx64-nvidia-cuda"
    LLVM.triple!(llvm_mod, triple)
    target = LLVM.Target(; triple)
    tm = LLVM.TargetMachine(target, triple, "sm_90", "+ptx80")
    LLVM.asm_verbosity!(tm, true)
    ptx = String(LLVM.emit(tm, llvm_mod, LLVM.API.LLVMAssemblyFile))
    cumod = CuModule(ptx)
    kernel = CuFunction(cumod, "vadd")
    return ptx, kernel
end

println("=" ^ 60)
println("(a) MLIR gpu.module → PTX → driver")
println("=" ^ 60)
mlir_ptx, mlir_kernel = build_mlir_kernel()
println("PTX size: $(length(mlir_ptx)) bytes")

# ---------------------------------------------------------------------------
# (b) CUDA.jl SIMT kernel
# ---------------------------------------------------------------------------

function vadd_simt!(C, A, B, n)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds C[i] = A[i] + B[i]
    end
    return
end

println("\n" * "=" ^ 60)
println("(b) CUDA.jl SIMT @cuda kernel")
println("=" ^ 60)
A_d = CUDA.zeros(Float32, 1024)
B_d = CUDA.zeros(Float32, 1024)
C_d = CUDA.zeros(Float32, 1024)
simt_ptx = sprint() do io
    CUDA.code_ptx(io, vadd_simt!,
                  Tuple{CuDeviceVector{Float32,1}, CuDeviceVector{Float32,1},
                        CuDeviceVector{Float32,1}, Int32}; kernel=true)
end
println("PTX size: $(length(simt_ptx)) bytes")

# ---------------------------------------------------------------------------
# (c) cuTile tile-based kernel
# ---------------------------------------------------------------------------

function vadd_cutile(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                    c::ct.TileArray{Float32,1}, tile_size::Int)
    pid = ct.bid(1)
    ta = ct.load(a; index=pid, shape=(tile_size,))
    tb = ct.load(b; index=pid, shape=(tile_size,))
    ct.store(c; index=pid, tile=ta + tb)
    return
end

println("\n" * "=" ^ 60)
println("(c) cuTile tile-based kernel")
println("=" ^ 60)
println("(cuTile's tileiras backend produces CUBIN, not PTX — skipping PTX dump)")

# ---------------------------------------------------------------------------
# PTX side-by-side: count instructions for the inner kernel body
# ---------------------------------------------------------------------------

# Extract the body of the first `.entry` kernel: everything between the
# `{` that follows the param list and the matching `\n}` at column 0.
# CUDA.jl mangles the kernel name and may emit `.maxntid`/alignment attrs
# between `)` and `{`, so allow non-`{` chars there.
function entry_body(ptx)
    m = match(r"\.entry\s+\S+\s*\([^)]*\)[^{]*\{(.*?)\n\}"s, ptx)
    return m === nothing ? nothing : m.captures[1]
end

function ptx_summary(ptx, name)
    body = something(entry_body(ptx), ptx)
    # Count instructions (lines that look like `op\b`) excluding directives,
    # labels, .reg/.param declarations, and blank lines.
    insns = 0
    loads = 0
    stores = 0
    fmas = 0
    branches = 0
    for line in split(body, '\n')
        s = strip(line)
        # Skip blank lines, comments, directives (.reg/.param/.loc/...),
        # and labels ($L__BB...:). NOTE: do NOT skip `@%p ...` lines — a
        # leading `@%pN` is a *predicate guard* on a real instruction
        # (typically `@%p1 bra ...`), not a directive.
        (isempty(s) || startswith(s, "//") || startswith(s, ".") ||
         endswith(s, ":")) && continue
        insns += 1
        if occursin(r"^ld\.", s);                            loads += 1
        elseif occursin(r"^st\.", s);                        stores += 1
        elseif occursin("fma.", s) || occursin("add.f32", s); fmas += 1
        elseif occursin(r"\bbra\b", s);                      branches += 1
        end
    end
    return (; total=insns, loads, stores, fmas, branches)
end

mlir_summary = ptx_summary(mlir_ptx, "MLIR")
simt_summary = ptx_summary(simt_ptx, "SIMT")
println("\n" * "=" ^ 60)
println("PTX instruction count comparison (kernel body)")
println("=" ^ 60)
println(rpad("metric",         16), rpad("MLIR",  10), "CUDA.jl SIMT")
for k in propertynames(mlir_summary)
    println(rpad(string(k),    16),
            rpad(getproperty(mlir_summary, k), 10),
            getproperty(simt_summary, k))
end

# ---------------------------------------------------------------------------
# Runtime comparison
# ---------------------------------------------------------------------------

println("\n" * "=" ^ 60)
println("Runtime comparison (N = 16M, Float32, sm_90)")
println("=" ^ 60)

const N = 16 * 1024 * 1024
const BLOCK = 256
const GRID  = cld(N, BLOCK)

A_host = rand(Float32, N)
B_host = rand(Float32, N)
A = CuArray(A_host)
B = CuArray(B_host)
C = CUDA.zeros(Float32, N)
gb = 3 * N * sizeof(Float32) / 1e9

function bench(launch_fn; warmup=3, samples=20)
    for _ in 1:warmup; launch_fn(); end
    CUDA.synchronize()
    best = typemax(Float64)
    for _ in 1:samples
        t = CUDA.@elapsed launch_fn()
        best = min(best, t)
    end
    return best
end

# (a) MLIR — memref descriptor packing
function memref_desc(arr::CuArray{T,1}) where {T}
    p = UInt64(UInt(pointer(arr)))
    (p, p, UInt64(0), UInt64(length(arr)), UInt64(1))
end
ad = memref_desc(A); bd = memref_desc(B); cd_ = memref_desc(C)
SIG_MLIR = Tuple{ntuple(_->Culonglong, 16)...}
args_mlir = (ad[1],ad[2],ad[3],ad[4],ad[5],
             bd[1],bd[2],bd[3],bd[4],bd[5],
             cd_[1],cd_[2],cd_[3],cd_[4],cd_[5],
             UInt64(N))
t_mlir = bench() do
    cudacall(mlir_kernel, SIG_MLIR, args_mlir...;
             threads=BLOCK, blocks=GRID)
end

# (b) CUDA.jl SIMT
t_simt = bench() do
    @cuda threads=BLOCK blocks=GRID vadd_simt!(C, A, B, Int32(N))
end

# (c) cuTile — requires Blackwell (sm_100+); Tile IR isn't supported on
# Hopper. Wrap so the experiment still reports (a) vs (b) on sm_90.
const TILE = 16
t_cutile = nothing
cutile_err = nothing
try
    A_tile = ct.TileArray(A)
    B_tile = ct.TileArray(B)
    C_tile = ct.TileArray(C)
    ct_kernel = ct.cufunction(vadd_cutile,
        Tuple{ct.TileArray{Float32,1}, ct.TileArray{Float32,1},
              ct.TileArray{Float32,1}, ct.Constant{Int,TILE}})
    n_blocks = cld(N, TILE)
    global t_cutile = bench() do
        ct_kernel(A_tile, B_tile, C_tile, ct.Constant(TILE); blocks=n_blocks)
    end
catch e
    global cutile_err = e
end

println()
println(rpad("pipeline",          26), rpad("μs",       10), "GB/s")
println(rpad("(a) MLIR gpu→PTX",  26), rpad(round(t_mlir*1e6, digits=2),  10), round(gb/t_mlir,  digits=1))
println(rpad("(b) CUDA.jl SIMT",  26), rpad(round(t_simt*1e6, digits=2),  10), round(gb/t_simt,  digits=1))
if t_cutile !== nothing
    println(rpad("(c) cuTile tile=16",26), rpad(round(t_cutile*1e6, digits=2),10), round(gb/t_cutile,digits=1))
else
    println(rpad("(c) cuTile tile=16",26), "unavailable: ", sprint(showerror, cutile_err) |> x -> first(split(x, '\n')))
end

# Print the inner body of each PTX so you can diff by eye.
println("\n" * "=" ^ 60)
println("MLIR PTX body")
println("=" ^ 60)
println(something(entry_body(mlir_ptx), "(no match)"))

println("\n" * "=" ^ 60)
println("CUDA.jl SIMT PTX body")
println("=" ^ 60)
println(something(entry_body(simt_ptx), "(no match)"))

# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------
#
# PTX shape (kernel body, sm_90):
#   - MLIR is ~3 instructions longer. The delta is entirely in addressing:
#       * MLIR does index math in 64-bit (`mul.wide.u32` + `add.s64` +
#         `shl.b64`) because MLIR `index` lowers to i64 and memref offset
#         arithmetic is 64-bit.
#       * MLIR emits 3 × `cvta.to.global` (generic→global addrspace cast)
#         because the memref base pointers arrive as generic pointers; the
#         flat memref descriptor doesn't carry an addrspace.
#       * CUDA.jl folds the index into one `mad.lo.s32` (32-bit, because
#         the kernel's `n::Int32` keeps the math 32-bit) and CuDeviceArray
#         pointers are already global-qualified.
#   - Both: 6 param loads, 1 global load pair, 1 add.f32, 1 store, 1
#     predicated branch. Structurally the same kernel.
#
# Runtime (N = 16M Float32, memory-bound):
#   - MLIR and CUDA.jl SIMT both land ~2850 GB/s — ~95% of the H100 NVL's
#     ~3 TB/s HBM3. The codegen differences are invisible: vadd is
#     bandwidth-bound, so the kernel that issues loads fast enough to
#     saturate DRAM wins, and both do.
#
# cuTile:
#   - Tile IR (cuTile's tileiras → CUBIN backend) is NOT supported on
#     Hopper (sm_90); it requires Blackwell (sm_100+). On this H100 the
#     cuTile path can't compile at all. The cuTileCPU package is unaffected
#     (it never uses tileiras — it re-lowers cuTile's StructuredIRCode via
#     our own MLIR pipeline), but a *native-GPU* cuTile comparison needs
#     Blackwell hardware.
#
# Takeaway for a unified MLIR codegen framework:
#   - The MLIR gpu-dialect → LLVM-NVPTX path produces production-quality
#     PTX that saturates DRAM bandwidth identically to GPUCompiler's SIMT
#     output. The remaining gap (64-bit index math, generic→global casts)
#     is addressable: pass index width as a pass option / use
#     `#gpu.address_space<global>` on the memref args so the casts fold
#     away. Neither matters for memory-bound kernels; both would matter
#     for compute-bound ones where instruction count drives occupancy.
