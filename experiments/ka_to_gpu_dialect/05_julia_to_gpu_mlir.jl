# Step 3 (design-doc step 1): Julia source → gpu.module MLIR, no
# hand-written MLIR. Uses MLIRKernels's NEW `lower_to_mlir_gpu` SIMT walker.
#
# This closes the frontend half of the GPU path. Experiments 01-04 ran
# hand-written gpu-dialect MLIR through the pipeline; here the gpu.module
# is *generated* from a plain Julia function by the same SCI walker that
# powers the CPU/SPMD path, in a new ScalarSIMT emission mode.
#
# The kernel is a plain Julia SIMT function: trailing `gid` is the global
# thread index (1-based), `n` is the length, `if gid <= n` is the bounds
# guard. The walker:
#   - binds `gid` to `gpu.thread_id.x + gpu.block_id.x*gpu.block_dim.x + 1`
#   - lowers `c[gid] = a[gid] + b[gid]` to scalar memref.load/addf/store
#   - keeps the `gid <= n` guard as scf.if
#   - emits memref args in #gpu.address_space<global> (exp 04's tuning,
#     baked into the walker)

using cuTile          # the inference/structurizer pipeline
using MLIRKernels

# Plain Julia SIMT kernel. `gid` is the synthesized global thread index;
# the host never passes it (the launcher drops the trailing arg, like the
# SPMD path). `Vector{Float32}` args + Int32 length/index.
function vadd_simt(c::Vector{Float32}, a::Vector{Float32}, b::Vector{Float32},
                   n::Int32, gid::Int32)
    if gid <= n
        @inbounds c[gid] = a[gid] + b[gid]
    end
    return
end

argtypes = Tuple{Vector{Float32}, Vector{Float32}, Vector{Float32}, Int32, Int32}

println("=" ^ 60)
println("Julia source → SCI → gpu.module MLIR")
println("=" ^ 60)

# Run cuTile's inference + structurizer to get the SCI (+ analyses).
sci, rettype, _, _ = MLIRKernels._structured_with_analyses(vadd_simt, argtypes)
@assert rettype === Nothing

mod, param_julia_types, mlir_ctx, param_kinds =
    MLIRKernels.lower_to_mlir_gpu(sci, argtypes; kernel_name="vadd")

MLIRKernels.MLIR.IR.activate(mlir_ctx)
mlir_text = sprint(show, mod)
println(mlir_text)

println("\n=== Sanity checks ===")
checks = [
    ("gpu.module envelope",          occursin("gpu.module", mlir_text)),
    ("gpu.func kernel",              occursin("gpu.func", mlir_text)),
    ("gpu.thread_id",                occursin("gpu.thread_id", mlir_text)),
    ("gpu.block_id",                 occursin("gpu.block_id", mlir_text)),
    ("global address space",         occursin("#gpu.address_space<global>", mlir_text) ||
                                     occursin("memory_space = #gpu.address_space<global>", mlir_text)),
    ("scalar memref.load (no vector)", occursin("memref.load", mlir_text) &&
                                       !occursin("vector.transfer_read", mlir_text)),
    ("scf.if bounds guard kept",     occursin("scf.if", mlir_text)),
    ("gpu.return",                   occursin("gpu.return", mlir_text)),
]
function report(checks)
    allok = true
    for (name, ok) in checks
        println(ok ? "  ✓ " : "  ✗ ", name)
        allok &= ok
    end
    return allok
end
allok = report(checks)
println(allok ? "\n✓ All checks passed — Julia → gpu.module MLIR works" :
                "\n✗ some checks failed")

# ---------------------------------------------------------------------------
# Lower the *generated* module through the GPU pipeline to PTX — confirms
# the walker output is valid input to the same path experiments 01-04 used.
# ---------------------------------------------------------------------------

using LLVM
const IR = MLIRKernels.MLIR.IR
const MLIRAPI = MLIRKernels.MLIR.API

const GPU_PASSES = String[
    "nvvm-attach-target{chip=sm_90 features=+ptx80}",
    "gpu-kernel-outlining",
    "gpu.module(convert-gpu-to-nvvm)",
    "convert-scf-to-cf", "convert-cf-to-llvm", "convert-arith-to-llvm",
    "expand-strided-metadata", "finalize-memref-to-llvm",
    "convert-nvvm-to-llvm", "reconcile-unrealized-casts",
    "gpu-module-to-binary{format=llvm}",
]

println("\n" * "=" ^ 60)
println("Lowering the GENERATED module → PTX")
println("=" ^ 60)
pm = IR.PassManager()
parse(IR.OpPassManager(pm),
      "builtin.module(" * join(GPU_PASSES, ",") * ")")
st = MLIRAPI.mlirPassManagerRunOnOp(pm, IR.Operation(mod))
if st.value == 0
    println("✗ pipeline failed on the generated module")
else
    # Pull the LLVM bitcode out of the gpu.binary, emit PTX via LLVM.jl.
    bc = let out=nothing
        for op in IR.body(mod)
            IR.name(op) == "gpu.binary" || continue
            objs = IR.getattr(op, "objects")
            o0 = IR.Attribute(MLIRAPI.mlirArrayAttrGetElement(objs, 0))
            sr = MLIRAPI.mlirGPUObjectAttrGetObject(o0)
            out = copy(unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(sr.data), sr.length; own=false))
        end
        out
    end
    lctx = LLVM.Context()
    lmod = LLVM.context!(lctx) do; parse(LLVM.Module, bc); end
    triple = "nvptx64-nvidia-cuda"
    LLVM.triple!(lmod, triple)
    tm = LLVM.TargetMachine(LLVM.Target(; triple), triple, "sm_90", "+ptx80")
    LLVM.asm_verbosity!(tm, true)
    ptx = String(LLVM.emit(tm, lmod, LLVM.API.LLVMAssemblyFile))
    m = match(r"\.entry\s+\S+\s*\([^)]*\)[^{]*\{(.*?)\n\}"s, ptx)
    println("✓ generated module lowered to PTX ($(length(ptx)) bytes)")
    println("\n=== generated-kernel PTX body ===")
    println(m === nothing ? "(no match)" : m.captures[1])
end
