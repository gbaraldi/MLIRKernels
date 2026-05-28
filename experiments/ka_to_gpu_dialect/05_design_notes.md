# A unified MLIR-based codegen framework (working title: `MLIRGPUCompiler.jl`)

Synthesis of experiments 01–04 into a concrete architecture. The goal: one
Julia → MLIR → {NVPTX, ROCDL, CPU} codegen path, shaped like GPUCompiler.jl,
reusing CUDA.jl / AMDGPU.jl for driver management and LLVM.jl for the final
target backend.

## What the experiments established

| # | Result | Implication |
|---|---|---|
| 01 | Hand-written `gpu.module` → PTX via MLIR.jl in-process pipeline works on MLIR_jll v20 | The MLIR→NVPTX pipeline needs no external tools; runs through `IR.PassManager` + LLVM.jl |
| 02 | Extracted LLVM bitcode (`format=llvm`) → LLVM.jl NVPTX TargetMachine → PTX → `CUDA.CuModule` → launch on H100 | We reuse the *same* NVPTX backend CUDA.jl uses; no duplicate codegen. PTX is the driver boundary, not a second backend. |
| 03 | vadd: MLIR vs CUDA.jl SIMT both ~2850 GB/s (95% of H100 HBM3); cuTile blocked on Hopper (Tile IR needs Blackwell) | Memory-bound kernels: MLIR path is bandwidth-equivalent to GPUCompiler. cuTile native-GPU comparison needs sm_100+. |
| 04 | `#gpu.address_space<global>` + i32 index → MLIR PTX *tighter* than GPUCompiler (19 vs 21 insns) | The frontend just needs to annotate addrspace + index width from arg types; no backend work needed. |

**Bottom line:** the GPU-dialect → LLVM-NVPTX → driver path is proven and
competitive. The missing piece is the **Julia → `gpu.module` walker** — the
frontend that turns inferred Julia IR into the gpu-dialect MLIR we've so far
hand-written.

## Architecture (mirrors GPUCompiler.jl)

```
                         ┌──────────────────────────────────────────┐
   frontends             │  KA backend │ cuTile flavor │ plain Julia │
                         └──────┬───────────────┬──────────────┬─────┘
                                │ overlays (MethodTable) per frontend
                                ▼
        ┌───────────────────────────────────────────────────────────┐
   shared inference     │  Julia inference (custom AbstractInterpreter │
   + structurization    │  + MethodTable)  →  IRStructurizer (SCI)     │
        └──────────────────────────────┬────────────────────────────┘
                                        │  StructuredIRCode + analyses
                                        ▼
        ┌───────────────────────────────────────────────────────────┐
   generic walker       │  SCI → high-level MLIR (scf/arith/memref/    │
   (target-parametric)  │  vector/math/func/gpu) — emission mode keyed │
                        │  on the target                               │
        └──────────────────────────────┬────────────────────────────┘
                                        │  MLIR module (high-level dialects)
                ┌───────────────────────┼───────────────────────┐
                ▼                       ▼                       ▼
        NVVMTarget              ROCDLTarget               CPUTarget
        gpu→nvvm pipeline       gpu→rocdl pipeline        scf→openmp+vector
        LLVM NVPTX backend      LLVM AMDGPU backend       (+ Polygeist affine)
        (via LLVM.jl)           (via LLVM.jl)             clang -O2 + libomp
                │                       │                       │
                ▼                       ▼                       ▼
        CUDA.jl driver          AMDGPU.jl driver          dlopen + ccall
        (CuModule/CuFunction)   (ROCModule/...)           (this is MLIRKernels)
```

### `AbstractTarget` hierarchy

```julia
abstract type AbstractTarget end

struct NVVMTarget  <: AbstractTarget; cap::VersionNumber; ptx::VersionNumber; end
struct ROCDLTarget <: AbstractTarget; arch::String; end
struct CPUTarget   <: AbstractTarget; cpu::String; features::String; use_polygeist::Bool; end
```

Each target answers three questions (the GPUCompiler `CompilerTarget` shape):

1. **`mlir_pipeline(::Target)`** → the MLIR pass list. NVVM:
   `nvvm-attach-target{chip,features}` → `gpu-kernel-outlining` →
   `gpu.module(convert-gpu-to-nvvm)` → `convert-{scf,cf,arith,memref,nvvm}-to-llvm`
   → `gpu-module-to-binary{format=llvm}`. CPU: the existing MLIRKernels
   `DEFAULT_PASSES` (+ optionally Polygeist's `affine`/`polygeist` passes
   for tiling/parallelism that upstream MLIR doesn't have).
2. **`emit_target_asm(::Target, llvm_mod)`** → PTX/HSACO/host `.so`. GPU
   targets hand the LLVM module to LLVM.jl's matching `TargetMachine`
   (experiment 02); CPU shells out to clang.
3. **`load_and_launch(::Target, asm, args; grid, block)`** → the driver.
   NVVM: `CUDA.CuModule` + `cudacall` (experiment 02/03). ROCDL:
   AMDGPU.jl's module loader. CPU: `dlopen` + `ccall` (MLIRKernels today).

### The generic walker — what's target-parametric

Today MLIRKernels's `lower_to_mlir` always emits `func.func` + `scf.parallel`
over the grid + vectorized (`vector.transfer_read`) tile bodies — the CPU
SPMD-on-SIMD model. The GPU target needs a different emission for the same
SCI:

| concept | CPU emission (today) | GPU (SIMT) emission |
|---|---|---|
| kernel | `func.func @k` | `gpu.module { gpu.func @k kernel }` |
| grid surface | `scf.parallel (%bid) = ...` | implicit; `gpu.launch_func` from host |
| block id | `scf.parallel` block arg | `gpu.block_id x` |
| lane / global idx | synthesized `vector<W×iX>` lane vector | `gpu.thread_id + gpu.block_id*gpu.block_dim` (scalar) |
| element load | `vector.transfer_read` (W-wide) | `memref.load` (scalar, 1 thread = 1 elem) |
| array arg | `memref<?xT, strided<[1]>>` | `memref<?xT, #gpu.address_space<global>>` (exp 04) |
| index width | i64 | i32 where source is 32-bit (exp 04) |
| bounds check | elided (`@inbounds` assumed) | `scf.if %gid < n` kept (exp 01/04) |

So the walker gains an **emission strategy** parameter — the arith/math/
control-flow clauses are shared verbatim; only the kernel-envelope, the
grid/lane mapping, and the load/store width differ. Concretely:
`LowerCtx` already has the `spmd::Bool` flag; this generalizes to
`emission::EmissionStrategy ∈ {TileSPMD, ScalarSIMT}` and the target picks it.

### What gets pulled out of cuTile

The inference + structurization + optimization layer is currently cuTile-
internal (`ct.emit_julia`, `ct.emit_structured`, `ct.run_passes!`, the
`cuTileMethodTable`, the divby/bounds analyses). For a shared framework these
move down into a target-agnostic package (IRStructurizer is the natural home,
or a new `StructuredCompiler.jl`):

- **`emit_structured`** + the SCI types — already in IRStructurizer; cuTile
  re-exports. Keep there.
- **`run_passes!`** and the divby/bounds dataflow analyses — currently in
  cuTile. These are generic SSA optimizations (canonicalize, CSE, LICM, FMA
  fusion, divisibility, bounds). Pull into IRStructurizer so both cuTile and
  the GPU/CPU MLIR path benefit. Experiments 03/04 show the GPU path wants
  the same alignment/divisibility facts (to prove `ld.global.nc` / vectorized
  loads).
- **`cuTileMethodTable`** + the overlay mechanism — the *frontend* layer.
  Each frontend (KA, cuTile-flavor, plain Julia) gets its own MethodTable;
  the shared piece is the `AbstractInterpreter` plumbing that runs inference
  against a supplied table. Pull the interpreter scaffolding out; keep the
  per-frontend overlays in their respective packages.

### Where Polygeist fits (CPU target)

Experiments 01–04 are GPU. For the CPU target, the gap vs hand-tuned code is
loop optimization (tiling, fusion, parallelism extraction) that upstream
MLIR's `scf`/`affine` passes do weakly. Polygeist's passes
(`--affine-cfg`, `--polygeist-mem2reg`, `--raise-scf-to-affine`, the affine
tiling/parallelization suite) are the missing CPU-side optimizer. They slot
into `CPUTarget.mlir_pipeline` between the walker output and
`convert-vector-to-llvm`:

```
SCI → walker (scf/memref/vector)
    → [Polygeist] raise-scf-to-affine → affine tiling/fusion/parallel
    → lower-affine → convert-scf-to-openmp → ... → LLVM → clang
```

Polygeist isn't in MLIR_jll; it'd be a separate JLL or a vendored build.
This is the CPU analogue of what `convert-gpu-to-nvvm` does for GPU: the
target-specific optimization that the generic high-level dialects don't
carry. Defer until the CPU path's loop-nest perf (matmul, stencils) is the
bottleneck — vadd-class memory-bound kernels don't need it.

## Driver reuse — the CUDA.jl-as-runtime pattern

cuTile already does this: it produces CUBIN, then uses `CUDA.CuModule` /
`CuFunction` / the launch machinery for everything driver-related (context,
stream, module load, kernel launch, memory). We do the same (experiment 02):

- **Don't** reimplement `cuModuleLoadData` / `cuLaunchKernel` / context
  management. `CUDA.CuModule(ptx)` + `cudacall(fn, sig, args...)` is the
  whole surface.
- **Do** own the compile pipeline (Julia → MLIR → PTX). That's the novel
  part; the driver is solved.
- Argument marshalling: the memref descriptor (allocated, aligned, offset,
  size, stride) is the one piece of glue — experiment 02 packs it as flat
  u64s. A `KernelAdaptor`-style `Adapt.adapt` rule (like CUDA.jl's
  `cudaconvert`) maps `CuArray` → descriptor. For `#gpu.address_space<global>`
  memrefs the aligned pointer is already a device global pointer, so no
  cvta needed at the ABI boundary either.

## Sequenced implementation plan

1. **Generic walker emission strategy** — add `ScalarSIMT` alongside the
   existing tile/SPMD emission in MLIRKernels's `lower.jl`. Smallest change
   that produces a `gpu.module` from an SCI. Reuses every arith/math/cf
   clause.
2. **`NVVMTarget` + pipeline + LLVM.jl PTX emit + CUDA.jl launch** — lift
   experiments 02/04 driver code into a reusable `compile(::NVVMTarget, ...)`.
3. **KA frontend on the GPU target** — the `MLIRBackend` overlay infra
   (already built, `ext/KernelAbstractionsExt.jl`) but with the target set
   to `NVVMTarget` instead of `CPUTarget`. The `__index_Global_Linear`
   overlay → sentinel → `gpu.thread_id + block_id*block_dim` (vs the CPU
   path's lane vector). This closes the "KA kernel → MLIR gpu → PTX →
   compare to CUDA.jl SIMT" loop end-to-end from Julia source.
4. **Pull inference/structurizer/analyses out of cuTile** into the shared
   layer. Mechanical but touches cuTile; do it once the GPU + CPU targets
   both consume it.
5. **Reduction kernel** (`@synchronize`/`@localmem`) — exercises `gpu.barrier`
   + `memref.alloca` in `#gpu.address_space<workgroup>`. The first kernel
   that needs real cross-thread semantics, not just per-thread SIMT.
6. **Polygeist for the CPU target** — when loop-nest perf is the bottleneck.
7. **`ROCDLTarget`** — same shape as NVVM, `convert-gpu-to-rocdl` +
   AMDGPU.jl driver. Validates the target abstraction generalizes.

## Open questions

- **cuTile native-GPU comparison** needs Blackwell (sm_100+) — Tile IR is
  unsupported on Hopper. Until then we can only compare cuTile *CPU*
  (MLIRKernels) against the GPU MLIR path, not cuTile-GPU vs MLIR-GPU.
- **Shared memory addressing** — `#gpu.address_space<workgroup>` lowering
  through `convert-gpu-to-nvvm` needs validating (experiment 06/reduction).
- **Where the package lives** — `MLIRGPUCompiler.jl` as a new package with
  MLIRKernels becoming its CPU-target reference, or grow MLIRKernels into the
  multi-target thing and rename. The walker + target abstraction is the
  reusable core either way.
