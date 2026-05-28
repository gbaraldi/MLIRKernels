# Lower the address-space-tuned vadd (04_addrspace_tuned.mlir) to PTX and
# compare its body against the untuned 01 kernel. Tests whether the two
# addressing knobs flagged in experiment 03 actually close the gap to
# CUDA.jl SIMT's PTX.

using cuTileCPU
using MLIR
const IR = MLIR.IR
using LLVM

const HERE = @__DIR__

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

function _bytes(sr::MLIR.API.MlirStringRef)
    copy(unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(sr.data), sr.length; own=false))
end
function extract_bc(mod::IR.Module)
    for op in IR.body(mod)
        IR.name(op) == "gpu.binary" || continue
        objs = IR.getattr(op, "objects")
        o0 = IR.Attribute(MLIR.API.mlirArrayAttrGetElement(objs, 0))
        return _bytes(MLIR.API.mlirGPUObjectAttrGetObject(o0))
    end
    error("no gpu.binary")
end

function mlir_file_to_ptx(path)
    ctx = cuTileCPU.fresh_context()
    IR.activate(ctx)
    mod = parse(IR.Module, read(path, String))
    pm = IR.PassManager()
    parse(IR.OpPassManager(pm), _pipeline_str(GPU_PASSES))
    MLIR.API.mlirPassManagerRunOnOp(pm, IR.Operation(mod)).value == 0 &&
        error("pipeline failed on $path")
    bc = extract_bc(mod)
    lctx = LLVM.Context()
    lmod = LLVM.context!(lctx) do; parse(LLVM.Module, bc); end
    triple = "nvptx64-nvidia-cuda"
    LLVM.triple!(lmod, triple)
    tm = LLVM.TargetMachine(LLVM.Target(; triple), triple, "sm_90", "+ptx80")
    LLVM.asm_verbosity!(tm, true)
    return String(LLVM.emit(tm, lmod, LLVM.API.LLVMAssemblyFile))
end

function entry_body(ptx)
    m = match(r"\.entry\s+\S+\s*\([^)]*\)[^{]*\{(.*?)\n\}"s, ptx)
    m === nothing ? nothing : m.captures[1]
end
function count_insns(ptx)
    body = something(entry_body(ptx), ptx)
    n = cvta = 0
    for line in split(body, '\n')
        s = strip(line)
        (isempty(s) || startswith(s, "//") || startswith(s, ".") ||
         endswith(s, ":")) && continue
        n += 1
        occursin("cvta.", s) && (cvta += 1)
    end
    return (; total=n, cvta)
end

ptx_untuned = mlir_file_to_ptx(joinpath(HERE, "01_handwritten_vadd_gpu.mlir"))
ptx_tuned   = mlir_file_to_ptx(joinpath(HERE, "04_addrspace_tuned.mlir"))

u = count_insns(ptx_untuned)
t = count_insns(ptx_tuned)

println("=" ^ 60)
println("Addressing-knob effect on PTX (sm_90 vadd)")
println("=" ^ 60)
println(rpad("variant", 34), rpad("insns", 8), "cvta.to.global")
println(rpad("01 untuned (generic ptr, i64)", 34), rpad(u.total, 8), u.cvta)
println(rpad("04 tuned (global addrspace, i32)", 34), rpad(t.total, 8), t.cvta)
println()
println("=== 04 tuned PTX body ===")
println(something(entry_body(ptx_tuned), "(no match)"))

# ---------------------------------------------------------------------------
# Findings (measured, H100 / sm_90)
# ---------------------------------------------------------------------------
#
#   variant                            insns   cvta.to.global
#   01 untuned (generic ptr, i64)      24      3
#   04 tuned   (global addrspace, i32) 19      0
#   CUDA.jl SIMT (experiment 03)       21      0
#
# Both addressing knobs landed:
#   - `#gpu.address_space<global>` on the memref args  → 3 cvta casts gone.
#   - i32 index math                                   → `mad.lo.s32` +
#     `mul.wide.s32` (32-bit, like CUDA.jl) instead of i64
#     `mul.wide.u32`+`add.s64`+`shl.b64`.
#
# Result: the tuned MLIR-emitted PTX (19 insns) is actually TIGHTER than
# GPUCompiler's SIMT output (21 insns) for this kernel, and structurally
# identical (mad.lo.s32 index, mul.wide.s32 offset, predicated branch,
# global ld/st). MLIR's gpu-dialect → LLVM-NVPTX path is not leaving
# anything on the table here once the frontend annotates address spaces
# and index width — which is exactly what a real walker would do from the
# kernel's type information (a `CuDeviceArray{T}` arg → global memref;
# an `Int32` loop bound → i32 index).
#
# This is the concrete spec for the eventual Julia→gpu.module walker:
#   * array args  → memref<?xT, #gpu.address_space<global>>
#   * index/lane  → i32 where the source type is 32-bit, else i64
#   * the lane id sentinel → gpu.thread_id + gpu.block_id*gpu.block_dim
