"""
    cuTileCPU

CPU backend for cuTile.jl. Takes a cuTile kernel + argtypes, runs cuTile's
existing inference + structurization pipeline, then lowers the resulting
StructuredIRCode to MLIR (high-level dialects: `scf`, `arith`, `memref`,
`vector`, `math`, `func`), runs the standard CPU lowering pipeline via
`mlir-opt` and `mlir-translate` from `MLIR_jll`, JIT-compiles the result via
`clang` into a shared object, and exposes a launch entry point.

# Quick start

```julia
using cuTile, cuTileCPU
const ct = cuTile

function vadd(a, b, c, tile_size::Int)
    pid = ct.bid(1)
    ta = ct.load(a; index=pid, shape=(tile_size,))
    tb = ct.load(b; index=pid, shape=(tile_size,))
    ct.store(c; index=pid, tile=ta + tb)
    return
end

# Aligned host buffers (the kernel's TileArray ArraySpec demands alignment
# > 16 bytes, which plain `Vector{Float32}` doesn't guarantee).
n = 1024
a = cuTileCPU.aligned_array(Float32, n)
b = cuTileCPU.aligned_array(Float32, n)
c = cuTileCPU.aligned_array(Float32, n)
copyto!(a, 1:n); copyto!(b, 1:n); fill!(c, 0)

k = cuTileCPU.cpu_function(vadd, (a, b, c, ct.Constant(16)))
k(a, b, c, ct.Constant(16); blocks=(n ÷ 16,))
@assert c == a .+ b
```

# Reflection

```julia
println(cuTileCPU.code_mlir(vadd, (a, b, c, ct.Constant(16))))
println(cuTileCPU.code_llvm(vadd, (a, b, c, ct.Constant(16))))
```
"""
module cuTileCPU

using cuTile
const ct = cuTile
using cuTile: BFloat16

using Reactant
using Reactant.MLIR.IR
const IR = Reactant.MLIR.IR
const Dialects = Reactant.MLIR.Dialects

using IRStructurizer: Block, BlockArgument, YieldOp, ContinueOp, BreakOp,
                      ConditionOp, IfOp, ForOp, WhileOp, LoopOp,
                      StructuredIRCode, Undef
import IRStructurizer
using Core: SSAValue, Argument, ReturnNode
using Core.Compiler: widenconst

using Libdl
import LLVM_full_jll
using LLVMOpenMP_jll: libomp_path

include("allocator.jl")
include("lower.jl")
include("compile.jl")
include("launch.jl")
include("reflect.jl")

export aligned_array, cpu_function, parallel_for, @parallel_for,
       spmd_function,
       code_mlir, code_mlir_lowered, code_llvm

end # module
