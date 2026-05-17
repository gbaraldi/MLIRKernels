# cuTileCPU.jl

A CPU backend for [cuTile.jl](https://github.com/JuliaGPU/cuTile.jl). Takes a
cuTile kernel + arg types, runs cuTile's existing inference and structurization
pipeline, then lowers the resulting `StructuredIRCode` to high-level MLIR
(`scf`/`arith`/`memref`/`vector`/`math`/`func` dialects), runs the standard
CPU lowering pipeline via `mlir-opt` and `mlir-translate`, JIT-compiles the
result via `clang`, and dispatches the grid over OpenMP threads.

The same cuTile kernel that the CUDA backend compiles also runs here — no
kernel-side changes needed.

## Status

**270 tests passing.** Every flagship kernel from cuTile.jl's perf-table works
on CPU.

| Kernel | cuTileCPU |
|---|---|
| Vector add (F32/F16/BF16) | ✓ |
| Matrix transpose | ✓ |
| Layer norm (forward + backward) | ✓ |
| Matrix multiply (F32; mixed-precision BF16→F32 in MLIR) | ✓ |
| Batched matmul | ✓ |
| FFT (1-stage DFT + 3-stage Cooley-Tukey) | ✓ |
| Mixture of Experts | ✓ |
| Attention (simple, non-flash) | ✓ |
| Flash attention (online softmax) | ✓ |
| Softmax | ✓ |

Plus full atomics (add/max/min/and/or/xor/xchg/cas), Philox RNG (uniform +
normal Float32, per-block stream divergence), gather/scatter, 20+ math
intrinsics, and alignment + stride-divisibility info propagation from
cuTile's dataflow analyses.

## Quick start

```julia
using cuTile, cuTileCPU
const ct = cuTile

# Plain cuTile kernel — identical to what the CUDA backend compiles.
function vadd(a, b, c, tile_size::Int)
    pid = ct.bid(1)
    ta = ct.load(a; index=pid, shape=(tile_size,))
    tb = ct.load(b; index=pid, shape=(tile_size,))
    ct.store(c; index=pid, tile=ta + tb)
    return
end

# The kernel's TileArray ArraySpec demands 128-byte alignment by default;
# Julia's Vector{T} only guarantees 16. Use aligned_array.
n = 1024
a = cuTileCPU.aligned_array(Float32, n; alignment=128)
b = cuTileCPU.aligned_array(Float32, n; alignment=128)
c = cuTileCPU.aligned_array(Float32, n; alignment=128)
copyto!(a, 1:n); copyto!(b, 101:100+n); fill!(c, 0f0)

# Explicit parallel-for is the natural launch surface.
cuTileCPU.@parallel_for blocks = n ÷ 16  vadd(a, b, c, ct.Constant(16))
@assert c ≈ a .+ b
```

## Architecture

```
@parallel_for blocks=N kernel(args)                  ← host surface
  → parallel_for(f, args; blocks)
    → cpu_function (cached on (f, argtypes, n_grid_dims))
      ├─ ct.emit_julia + emit_structured + run_passes!     ← cuTile pipeline
      │   ├─ canonicalize, CSE, LICM, DCE, FMA fusion, …
      │   └─ divby_info, bounds_info  (kept, not discarded)
      ├─ lower_to_mlir (the walker)
      │   ├─ TileArray  → memref<?…xT, strided<[1, ?, …]>>
      │   ├─ ArraySpec.alignment  → memref.assume_alignment
      │   ├─ divby_info  → llvm.intr.assume on strides
      │   ├─ bid(k)  → block argument of scf.parallel
      │   ├─ kernel body  → arith / vector / math / memref ops
      │   └─ ~50 intrinsic clauses (mma, gather/scatter, atomics, RNG, …)
      ├─ mlir-opt | mlir-translate | clang -O2 -fopenmp
      └─ dlopen + ccall  (synchronous; libomp under the hood)
```

The pipeline goes via **high-level MLIR dialects only** (`scf`, `arith`,
`memref`, `vector`, `math`, `func`) — never `llvm` dialect directly. MLIR's
upstream conversion passes take care of getting us to LLVM IR. This keeps the
CPU path aligned with other potential targets (CUDA-via-NVVM, XLA, etc.) that
would share the high-level MLIR but lower differently.

## Module layout

```
cuTileCPU/
├── Project.toml
├── README.md                      ← you are here
├── src/
│   ├── cuTileCPU.jl               ← module entry
│   ├── allocator.jl               ← aligned_array (posix_memalign + unsafe_wrap)
│   ├── lower.jl                   ← StructuredIRCode → MLIR walker (~50 clauses)
│   ├── compile.jl                 ← mlir-opt | mlir-translate | clang pipeline
│   ├── launch.jl                  ← cpu_function, CPUKernel, @parallel_for
│   └── reflect.jl                 ← code_mlir, code_mlir_lowered, code_llvm
├── test/
│   ├── runtests.jl                ← 270 tests across every flagship kernel
│   └── dump_sci.jl                ← diagnostic: dumps cuTile SCI for each kernel
└── bench/
    ├── bench_vadd.jl              ← vadd memory bandwidth bench (cache-flushed)
    ├── bench_matmul.jl            ← matmul GFLOPS vs OpenBLAS
    └── bench_bmm.jl               ← batched matmul GFLOPS
```

## Public API

```julia
# Allocator
aligned_array(T, dims...; alignment=64)               → Array{T,N}
aligned_array(T, dims::NTuple; alignment=64)          → Array{T,N}

# Compilation (cached)
cpu_function(f, argtypes::Type; n_grid_dims=1)        → CPUKernel
cpu_function(f, args::Tuple; n_grid_dims=1)           → CPUKernel  (derives argtypes)

# Launch
(k::CPUKernel)(args...; blocks)                       → nothing
parallel_for(f, args; blocks)                         → nothing
@parallel_for blocks=N f(args...)                     → nothing  (macro form)

# Reflection
code_mlir(f, argtypes)                                → String   (MLIR before lowering)
code_mlir_lowered(f, argtypes)                        → String   (MLIR in LLVM dialect)
code_llvm(f, argtypes)                                → String   (textual LLVM IR)
```

## Performance

Measured on a 64-thread machine with cache-flushed `time_min` (`@belapsed`-style
back-to-back samples produce inflated cache-resident numbers — see
`bench/bench_vadd.jl`).

### vadd (memory-bandwidth-bound)

| n | per-array | touched | cuTileCPU | Julia broadcast | `Threads.@threads` |
|---|---|---|---|---|---|
| 64 K | 256 KB | 0.8 MB | 11 GB/s | **38 GB/s** | 2.4 GB/s |
| 1 M | 4 MB | 13 MB | **185 GB/s** | 35 GB/s | 34 GB/s |
| 16 M | 64 MB | 201 MB | **720 GB/s** | 46 GB/s | 383 GB/s |
| 64 M | 256 MB | 805 MB | **440 GB/s** | 33 GB/s | 240 GB/s |
| 256 M | 1 GB | 3.2 GB | **237 GB/s** | 32 GB/s | 210 GB/s |

At DRAM scale (256 M = 1 GB/array): **~7× faster than serial Julia broadcast,
~1.1× faster than `Threads.@threads`**. The MLIR vector path with
ArraySpec-driven alignment + strided layout produces tighter code than
hand-rolled threaded Julia.

At small N (< 64 K): cuTileCPU loses to serial broadcast because OpenMP
fork/join overhead (~70 μs floor) dominates the work. Fixable by compiling a
serial variant; not done yet.

### matmul F32 (compute-bound)

64-thread, BM=BN=BK=64 tiles.

| Shape | cuTileCPU | OpenBLAS | Naive triple-loop |
|---|---|---|---|
| 128³ | 13 GFLOPS | 40 | 4.6 |
| 512³ | 112 GFLOPS | 712 | 23 |
| 1024³ | 399 GFLOPS | 1418 | 39 |
| 2048³ | **772 GFLOPS** | 2186 | 24 |

cuTileCPU is **~35% of OpenBLAS** with no hand-tuning, and **4–30× faster than
hand-rolled threaded Julia**. The gap to BLAS is mostly tile-size and register
blocking — BLAS uses ~192×64 inner tiles with software prefetching; we use
fixed 64×64 with no inner blocking. Users can pass `ct.Constant(128)` or
`(256)` to widen the tile.

## What we pull from cuTile

The walker uses cuTile's analyses, not just its IR:

- **`run_passes!`** runs cuTile's full pipeline before we walk:
  canonicalize → constprop → FMA fusion → CSE → alias → token ordering →
  RNG decomposition → LICM → divisibility/bounds analyses → no-wrap → DCE.
  We get the optimized SCI.
- **`divby_info`** (from cuTile's divisibility analysis) is consumed at func
  entry to emit per-stride `llvm.intr.assume(stride % N == 0)`. LLVM's
  vectorizer picks these up to prove aligned access on non-leading dims.
- **`ArraySpec.alignment`** (from `TileArray`'s spec) drives
  `memref.assume_alignment` on the base pointer.
- **`ArraySpec.contiguous`** drives the `strided<[1, ?, ?]>` layout in the
  memref type so LLVM has a compile-time proof of unit stride.

## How a kernel becomes MLIR

For the vadd kernel above:

```mlir
module {
  func.func @vadd(%arg0: memref<?xf32, strided<[1]>>,
                  %arg1: memref<?xf32, strided<[1]>>,
                  %arg2: memref<?xf32, strided<[1]>>,
                  %arg3: i32,                          // KernelState.seed
                  %arg4: index)                        // grid dim
                  attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.assume_alignment %arg0, 128 : memref<?xf32, strided<[1]>>
    %1 = memref.assume_alignment %arg1, 128 : memref<?xf32, strided<[1]>>
    %2 = memref.assume_alignment %arg2, 128 : memref<?xf32, strided<[1]>>
    scf.parallel (%bid) = (%c0) to (%arg4) step (%c1) {
      %off = arith.muli %bid, %c16 : index
      %va = vector.transfer_read %0[%off], %cst {in_bounds = [true]}
              : memref<?xf32, strided<[1]>>, vector<16xf32>
      %vb = vector.transfer_read %1[%off], %cst {in_bounds = [true]}
              : memref<?xf32, strided<[1]>>, vector<16xf32>
      %sum = arith.addf %va, %vb : vector<16xf32>
      vector.transfer_write %sum, %2[%off] {in_bounds = [true]}
              : vector<16xf32>, memref<?xf32, strided<[1]>>
      scf.reduce
    }
    return
  }
}
```

## Why external `mlir-opt` + `clang`?

`libReactantExtra.so` (the C library Reactant's MLIR.jl bindings link
against) doesn't expose `mlirExecutionEngine*` or register the conversion
passes we need (`convert-vector-to-scf`, `convert-scf-to-openmp`,
`finalize-memref-to-llvm`, …) with the textual pass-pipeline parser. So we
drive `mlir-opt` and `mlir-translate` from `MLIR_jll`/`LLVM_full_jll` as
external processes, then `clang` to build a `.so`, then `dlopen` it.

When those bits become available in `libReactantExtra`, the host-side
pipeline can collapse to in-process `IR.run!(pm, mod) + IR.ExecutionEngine`.

## Dependencies

- **cuTile** (the front end and analyses)
- **Reactant.MLIR.IR** (MLIR builder bindings — vendored via the Reactant package)
- **IRStructurizer** (the structurized IR types cuTile produces)
- **LLVM_full_jll** (mlir-opt, mlir-translate, clang)
- **LLVMOpenMP_jll** (libomp, linked into JIT'd .so for the parallel grid)

Reactant's MLIR.jl bindings include four dialects we added during this work:
`SCF.jl`, `Vector.jl`, `Math.jl`. (Reactant ships Arith, MemRef, Func, LLVM,
plus its own StableHLO/CUDATile/etc. but didn't have SCF/Vector/Math by
default — we regenerated those via Reactant's `make-bindings.jl` Bazel
toolchain.)

## Known gaps

1. **Small-N launch overhead** — ~70 μs OpenMP fork/join floor. Compiling a
   serial variant alongside the parallel one would let small grids skip
   libomp entirely. Not done yet.
2. **bounds_info per-consumer refinement** — captured from cuTile's analysis,
   but only stride-divisibility is consumed at func entry. Tighter facts at
   memory-op sites would help vectorization further.
3. **Matmul tile-size tuning** — fixed BM=BN=BK=64 in our test kernels. Users
   can pass `ct.Constant(128)`/`(256)` to widen; haven't characterized the
   sweet spot.
4. **In-process MLIR JIT** — needs `libReactantExtra` to expose more API.
   Today we shell out to `mlir-opt`/`mlir-translate`/`clang`.
