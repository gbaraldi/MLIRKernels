# Capstone: a plain Julia function compiled to a GPU kernel by
# `lower_to_mlir_gpu` and LAUNCHED on the H100, verified correct, timed
# against CUDA.jl's native SIMT @cuda.
#
# This is experiments 02 (launch) + 05 (Julia→gpu.module walker) combined:
# no hand-written MLIR anywhere. The kernel is the Julia function below;
# everything downstream is generated.

using cuTile
using MLIRKernels
using MLIR
const IR = MLIR.IR
const MLIRAPI = MLIR.API
using LLVM
using CUDA

# ---------------------------------------------------------------------------
# The kernel — plain Julia SIMT. Trailing `gid` is the global thread index
# (synthesized in MLIR; the host never passes it). `n` is the length.
# ---------------------------------------------------------------------------

function vadd_simt(c::Vector{Float32}, a::Vector{Float32}, b::Vector{Float32},
                   n::Int32, gid::Int32)
    if gid <= n
        @inbounds c[gid] = a[gid] + b[gid]
    end
    return
end

const ARGT = Tuple{Vector{Float32}, Vector{Float32}, Vector{Float32}, Int32, Int32}

# ---------------------------------------------------------------------------
# Compile: Julia → SCI → gpu.module → PTX → CuFunction
# ---------------------------------------------------------------------------

const GPU_PASSES = String[
    "nvvm-attach-target{chip=sm_90 features=+ptx80}",
    "gpu-kernel-outlining",
    "gpu.module(convert-gpu-to-nvvm)",
    "convert-scf-to-cf", "convert-cf-to-llvm", "convert-arith-to-llvm",
    "expand-strided-metadata", "finalize-memref-to-llvm",
    "convert-nvvm-to-llvm", "reconcile-unrealized-casts",
    "gpu-module-to-binary{format=llvm}",
]

function compile_julia_to_cufunction(f, argt; kernel_name="vadd")
    sci, rettype, _, _ = MLIRKernels._structured_with_analyses(f, argt)
    @assert rettype === Nothing
    mod, _, mlir_ctx, _ = MLIRKernels.lower_to_mlir_gpu(sci, argt; kernel_name)

    IR.activate(mlir_ctx)
    pm = IR.PassManager()
    parse(IR.OpPassManager(pm), "builtin.module(" * join(GPU_PASSES, ",") * ")")
    MLIRAPI.mlirPassManagerRunOnOp(pm, IR.Operation(mod)).value == 0 &&
        error("GPU pipeline failed")

    bc = let out=nothing
        for op in IR.body(mod)
            IR.name(op) == "gpu.binary" || continue
            objs = IR.getattr(op, "objects")
            o0 = IR.Attribute(MLIRAPI.mlirArrayAttrGetElement(objs, 0))
            sr = MLIRAPI.mlirGPUObjectAttrGetObject(o0)
            out = copy(unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(sr.data), sr.length; own=false))
        end
        out === nothing && error("no gpu.binary"); out
    end

    lctx = LLVM.Context()
    lmod = LLVM.context!(lctx) do; parse(LLVM.Module, bc); end
    triple = "nvptx64-nvidia-cuda"
    LLVM.triple!(lmod, triple)
    tm = LLVM.TargetMachine(LLVM.Target(; triple), triple, "sm_90", "+ptx80")
    LLVM.asm_verbosity!(tm, true)
    ptx = String(LLVM.emit(tm, lmod, LLVM.API.LLVMAssemblyFile))
    cumod = CuModule(ptx)
    return CuFunction(cumod, kernel_name)
end

println("=" ^ 60)
println("Compiling Julia `vadd_simt` → gpu.module → PTX → CuFunction")
println("=" ^ 60)
kernel = compile_julia_to_cufunction(vadd_simt, ARGT)
println("✓ CuFunction handle: ", kernel.handle)

# ---------------------------------------------------------------------------
# Launch + verify
# ---------------------------------------------------------------------------
#
# memref<?xf32, #gpu.address_space<global>> still flattens to the 5-field
# descriptor (allocated, aligned, offset, size, stride). 3 memrefs + the
# `n::Int32` uniform = 3*5 + 1 = 16 args. The trailing `gid` is NOT a
# host arg (synthesized in the kernel).

const N = 16 * 1024 * 1024
A_host = rand(Float32, N); B_host = rand(Float32, N)
A = CuArray(A_host); B = CuArray(B_host); C = CUDA.zeros(Float32, N)

function desc(arr::CuArray)
    p = UInt64(UInt(pointer(arr)))
    (p, p, UInt64(0), UInt64(length(arr)), UInt64(1))
end
ad, bd, cd_ = desc(A), desc(B), desc(C)

# Arg layout matches the gpu.func param order: c, a, b, n  (kernel sig is
# (c, a, b, n, gid); lower_to_mlir_gpu drops gid as the synthesized lane).
# Each memref = 5 u64; n = i32. cudacall wants exact C types.
const SIG = Tuple{Culonglong,Culonglong,Culonglong,Culonglong,Culonglong,  # c
                  Culonglong,Culonglong,Culonglong,Culonglong,Culonglong,  # a
                  Culonglong,Culonglong,Culonglong,Culonglong,Culonglong,  # b
                  Cint}                                                     # n
args = (cd_..., ad..., bd..., Cint(N))

const BLOCK = 256
const GRID  = cld(N, BLOCK)

println("\n=== Launching generated kernel: grid=$GRID block=$BLOCK N=$N ===")
cudacall(kernel, SIG, args...; threads=BLOCK, blocks=GRID)
CUDA.synchronize()

C_host = Array(C)
err = maximum(abs.(C_host .- (A_host .+ B_host)))
println("max abs diff = $err")
@assert err == 0 "generated kernel produced wrong results"
println("✓ Julia-source-generated GPU kernel runs CORRECTLY on the H100")

# ---------------------------------------------------------------------------
# Timing vs CUDA.jl native SIMT
# ---------------------------------------------------------------------------

function vadd_cuda!(C, A, B, n)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds C[i] = A[i] + B[i]
    end
    return
end

function bench(fn; warmup=3, samples=20)
    for _ in 1:warmup; fn(); end
    CUDA.synchronize()
    best = typemax(Float64)
    for _ in 1:samples
        best = min(best, CUDA.@elapsed fn())
    end
    return best
end

gb = 3 * N * sizeof(Float32) / 1e9
t_gen  = bench(() -> cudacall(kernel, SIG, args...; threads=BLOCK, blocks=GRID))
t_cuda = bench(() -> @cuda(threads=BLOCK, blocks=GRID, vadd_cuda!(C, A, B, Int32(N))))

println("\n" * "=" ^ 60)
println("Runtime (N = $N, H100 sm_90)")
println("=" ^ 60)
println(rpad("pipeline", 34), rpad("μs", 10), "GB/s")
println(rpad("Julia→MLIR-gpu→PTX (generated)", 34),
        rpad(round(t_gen*1e6, digits=2), 10), round(gb/t_gen, digits=1))
println(rpad("CUDA.jl @cuda SIMT (GPUCompiler)", 34),
        rpad(round(t_cuda*1e6, digits=2), 10), round(gb/t_cuda, digits=1))
println("\nratio (generated / CUDA.jl) = $(round(t_gen/t_cuda, digits=3))x")
