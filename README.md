# cuTileCPU.jl

A CPU backend for [cuTile.jl](https://github.com/JuliaGPU/cuTile.jl). Takes a
cuTile kernel + arg types, runs cuTile's existing inference and structurization
pipeline, then lowers the resulting `StructuredIRCode` to high-level MLIR
(`scf`/`arith`/`memref`/`vector`/`math`/`func` dialects), runs the standard
CPU lowering pipeline **in-process via [MLIR.jl](https://github.com/JuliaLLVM/MLIR.jl)**,
JIT-compiles the LLVM IR via `clang`, and dispatches the grid over OpenMP
threads.

The same cuTile kernel that the CUDA backend compiles also runs here — no
kernel-side changes needed.

## Status

**283 tests passing on Julia 1.13 + MLIR_jll v20.** Every flagship kernel from
cuTile.jl's perf-table works on CPU.

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
| SPMD-on-SIMD (ISPC-style scalar Julia) | ✓ |

Plus full atomics (add/max/min/and/or/xor/xchg/cas), Philox RNG (uniform +
normal Float32, per-block stream divergence), gather/scatter, 20+ math
intrinsics, and alignment + stride-divisibility info propagation from cuTile's
dataflow analyses.

### Julia version compatibility

| Julia | libLLVM | MLIR_jll | Tests passing |
|---|---|---|---|
| **1.13.0-rc1+** | 20.1.8 | **20.1.8** | **283 / 283** |
| 1.12.6 | 18.1.7 | 18.1.7 | ~218 / 283 — MLIR 18 lacks `lower-vector-multi-reduction` (registered through libMLIR-C only from MLIR 19+) and `math.tanh` translation interface; the affected kernels (reductions / math intrinsics / FFT) error |

Julia 1.13 is the recommended target. The 1.12 fallback works for everything
except the kernels that need MLIR 19+ passes.

#### Julia 1.13 setup

`MLIR.jl` on Julia 1.13 needs the fix in
[JuliaLLVM/MLIR.jl#88](https://github.com/JuliaLLVM/MLIR.jl/pull/88) (Julia
1.13's stricter `@ccall` macro rejects MLIR.jl's `@ccall (Ref[]).fn(...)`
form). Until the PR lands, dev the fork:

```julia
] dev https://github.com/gbaraldi/MLIR.jl#fix-julia-1.13-ccall
```

The `Project.toml` here pins `MLIR_jll = "18, 19, 20"` and omits v21
deliberately — MLIR_jll v21 has a registry compat constraint
(`libLLVM_jll = "21.1.2-21"`) that the Pkg resolver silently ignores for
stdlib JLLs, picks v21 on Julia 1.13, and then `MLIR_jll.is_available() ==
false` at runtime.

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

### SPMD mode (ISPC-style)

For kernels you'd rather write as plain scalar Julia (no `ct.load`/`ct.store`,
no whole-tile arithmetic), `spmd_function` lifts a function whose trailing
arg is a scalar lane index to `lane_width`-wide vector MLIR:

```julia
function vadd_spmd(a::Vector{Float32}, b::Vector{Float32},
                   c::Vector{Float32}, i::Int)
    @inbounds c[i] = a[i] + b[i]
    return
end

k = cuTileCPU.spmd_function(vadd_spmd,
    (Vector{Float32}, Vector{Float32}, Vector{Float32}, Int);
    lane_width=16, alignment=128)
k(a, b, c, 0; blocks = cld(n, 16))
```

The walker rebinds `i` to a synthesized `vector<W × i64>` of lane indices and
lifts every scalar op to its vector form. Contiguous `a[i]` is detected as a
fast path and lowers to `vector.transfer_read/write`; indirect `a[idx[i]]`
falls back to `vector.gather/scatter`.

## Architecture

```
@parallel_for blocks=N kernel(args)                  ← host surface
  → parallel_for(f, args; blocks)
    → cpu_function (cached on (f, argtypes, n_grid_dims))
      ├─ ct.emit_julia + emit_structured + run_passes!     ← cuTile pipeline
      │   ├─ canonicalize, CSE, LICM, DCE, FMA fusion, …
      │   └─ divby_info, bounds_info  (kept, not discarded)
      ├─ lower_to_mlir (the walker, ~3000 lines)
      │   ├─ TileArray  → memref<?…xT, strided<[1, ?, …]>>
      │   ├─ ArraySpec.alignment  → memref.assume_alignment
      │   ├─ divby_info  → llvm.intr.assume on strides
      │   ├─ bid(k)  → block argument of scf.parallel
      │   ├─ kernel body  → arith / vector / math / memref ops
      │   └─ ~50 intrinsic clauses (mma, gather/scatter, atomics, RNG, …)
      ├─ compile_module_to_so(mod, mlir_ctx; passes=DEFAULT_PASSES)
      │   ├─ MLIR.IR.PassManager + textual pipeline       ← in-process opt
      │   ├─ mlirTranslateModuleToLLVMIR + LLVMPrintModuleToString
      │   └─ clang -O2 -shared -fPIC (only external step) ← LLVM-IR → .so
      └─ dlopen + ccall  (synchronous; libomp under the hood)
```

The pipeline goes via **high-level MLIR dialects only** (`scf`, `arith`,
`memref`, `vector`, `math`, `func`) — never `llvm` dialect directly. MLIR's
upstream conversion passes take care of getting us to LLVM IR. This keeps the
CPU path aligned with other potential targets (CUDA-via-NVVM, XLA, etc.) that
would share the high-level MLIR but lower differently.

### MLIR pass pipeline (in-process)

Run via `MLIR.IR.PassManager` against the same context the walker emitted into
— no textual round-trip:

```
convert-math-to-llvm                ← rsqrt, exp, sin, cos, sqrt, log, … → llvm.intr.*
convert-math-to-libm                ← tanh (no LLVM intrinsic on MLIR 20+) → tanhf
func.func(lower-vector-multi-reduction)  ← MLIR 19+ only; nested under func.func
convert-vector-to-scf
convert-vector-to-llvm              ← lowers vector.extract from libm scalarization
convert-scf-to-openmp
convert-openmp-to-llvm
convert-scf-to-cf
lower-affine
expand-strided-metadata
finalize-memref-to-llvm
convert-arith-to-llvm
convert-func-to-llvm
convert-cf-to-llvm
convert-ub-to-llvm
reconcile-unrealized-casts
```

Pass ordering is load-bearing in two places:

- `convert-math-to-llvm` MUST precede `convert-math-to-libm` — otherwise libm
  emits a `rsqrtf` call to a function that doesn't exist in standard libm
  and dlopen fails at launch.
- `convert-vector-to-llvm` MUST follow `convert-math-to-libm` — libm
  scalarizes vector math by emitting `vector.extract` per lane + scalar call,
  and those extracts need lowering.

Several MLIR-version-conditional emissions live in the walker
(`MLIR.MLIR_VERSION[]`-keyed): `vector.multi_reduction.reduction_dims`
(I64ArrayAttr on 18, DenseI64ArrayAttr on 19+), `memref.atomic_rmw kind` enum
codes (reordered between 18 and 20, `xori` removed in 20),
`memref.assume_alignment` result form, `vector.step` (native on 19+, falls
back to `arith.constant dense<[0..N-1]>` on 18).

## Module layout

```
cuTileCPU/
├── Project.toml
├── README.md                      ← you are here
├── src/
│   ├── cuTileCPU.jl               ← module entry; fresh_context() + @with_* macros
│   ├── allocator.jl               ← aligned_array (posix_memalign + unsafe_wrap)
│   ├── lower.jl                   ← StructuredIRCode → MLIR walker (~50 clauses)
│   ├── compile.jl                 ← in-process pass pipeline + clang
│   ├── launch.jl                  ← cpu_function, spmd_function, CPUKernel, @parallel_for
│   └── reflect.jl                 ← code_mlir, code_mlir_lowered, code_llvm
├── test/
│   ├── runtests.jl                ← 283 tests across every flagship kernel
│   └── dump_sci.jl                ← diagnostic: dumps cuTile SCI for each kernel
└── bench/
    ├── bench_vadd.jl              ← vadd memory bandwidth bench (cache-flushed)
    ├── bench_matmul.jl            ← matmul GFLOPS vs OpenBLAS
    ├── bench_matmul_reg.jl        ← rank-1 register-tile matmul (~65% of OpenBLAS)
    ├── bench_bmm.jl               ← batched matmul GFLOPS
    ├── bench_spmd.jl              ← SPMD vs tile vadd at DRAM scale
    └── perf_research/             ← linalg-path matmul experiments + Triton-CPU comparison
```

## Public API

```julia
# Allocator
aligned_array(T, dims...; alignment=64)               → Array{T,N}
aligned_array(T, dims::NTuple; alignment=64)          → Array{T,N}

# Compilation (cached)
cpu_function(f, argtypes::Type; n_grid_dims=1)        → CPUKernel
cpu_function(f, args::Tuple; n_grid_dims=1)           → CPUKernel  (derives argtypes)
spmd_function(f, argtypes::Type; lane_width=16, alignment=16)  → CPUKernel
spmd_function(f, args::Tuple; lane_width=16, alignment=16)     → CPUKernel

# Launch
(k::CPUKernel)(args...; blocks)                       → nothing
parallel_for(f, args; blocks)                         → nothing
@parallel_for blocks=N f(args...)                     → nothing  (macro form)

# Reflection
code_mlir(f, argtypes; spmd=false, lane_width=16, alignment=16)     → String
code_mlir_lowered(f, argtypes; spmd=false, lane_width=16, …)        → String
code_llvm(f, argtypes; spmd=false, lane_width=16, …)                → String
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

64-thread, BM=BN=BK=64 tiles (default contract-lowering on MLIR 19+).

| Shape | cuTileCPU (`vector.contract`) | cuTileCPU (`matmul_reg`, rank-1 outer-product) | OpenBLAS | Naive |
|---|---|---|---|---|
| 1024³ | 399 GFLOPS | **~920 GFLOPS (~65%)** | 1418 | 39 |

The rank-1 register-tile path (`matmul_reg_kernel` in tests) is the
recommended pattern for CPU — same BLAS-microkernel shape (rank-1 outer
product with phi-carried accumulator). The default `vector.contract` path is
~27% of OpenBLAS at 1024³. See `bench/bench_matmul_reg.jl`.

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
    memref.assume_alignment %arg0, 128 : memref<?xf32, strided<[1]>>
    memref.assume_alignment %arg1, 128 : memref<?xf32, strided<[1]>>
    memref.assume_alignment %arg2, 128 : memref<?xf32, strided<[1]>>
    scf.parallel (%bid) = (%c0) to (%arg4) step (%c1) {
      %off = arith.muli %bid, %c16 : index
      %va = vector.transfer_read %arg0[%off], %cst {in_bounds = [true]}
              : memref<?xf32, strided<[1]>>, vector<16xf32>
      %vb = vector.transfer_read %arg1[%off], %cst {in_bounds = [true]}
              : memref<?xf32, strided<[1]>>, vector<16xf32>
      %sum = arith.addf %va, %vb : vector<16xf32>
      vector.transfer_write %sum, %arg2[%off] {in_bounds = [true]}
              : vector<16xf32>, memref<?xf32, strided<[1]>>
      scf.reduce
    }
    return
  }
}
```

After the pipeline → LLVM IR → clang `-O2`, the inner loop on an AVX-512 box
is three ZMM instructions: `vmovups (a) → zmm; vaddps (b), zmm → zmm; vmovups
zmm → (c)`.

## Dependencies

- **cuTile** (the front end and analyses)
- **MLIR.jl** (MLIR builder bindings + in-process PassManager — pending the
  Julia 1.13 `@ccall` fix in JuliaLLVM/MLIR.jl#88)
- **IRStructurizer** (the structurized IR types cuTile produces)
- **MLIR_jll** (libMLIR-C — the C ABI for the MLIR builder and PassManager)
- **LLVM_full_jll** (clang only; mlir-opt and mlir-translate no longer needed
  since we moved the pipeline in-process)
- **LLVMOpenMP_jll** (libomp, linked into JIT'd .so for the parallel grid)

## Known gaps

1. **Small-N launch overhead** — ~70 μs OpenMP fork/join floor. Compiling a
   serial variant alongside the parallel one would let small grids skip
   libomp entirely. `spmd_function(...; serial=true)` already does this for
   SPMD kernels; the tile path doesn't yet expose it.
2. **bounds_info per-consumer refinement** — captured from cuTile's analysis,
   but only stride-divisibility is consumed at func entry. Tighter facts at
   memory-op sites would help vectorization further.
3. **Matmul tile-size tuning** — default `vector.contract` lowering on
   MLIR 19+ doesn't expose the `outerproduct` option (it was a CLI-flag
   feature, not exposed in the textual pipeline). The handwritten
   `matmul_reg_kernel` rank-1 outer-product pattern remains the recommended
   path for compute-bound code.
4. **Julia 1.12 / MLIR 18 path** — has 14 errored / 1 failed tests because
   MLIR 18 doesn't register `lower-vector-multi-reduction` through libMLIR-C
   and lacks the `math.tanh` translation interface. Self-heals on Julia 1.13.
