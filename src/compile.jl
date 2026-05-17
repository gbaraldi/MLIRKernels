# MLIR module → shared object, via the external `mlir-opt` + `mlir-translate`
# from `MLIR_jll` and the system `clang`.
#
# Why external tools? `libReactantExtra.so` (the C library Reactant's MLIR
# bindings link against) doesn't register the conversion passes we need
# (convert-vector-to-scf, convert-scf-to-openmp, finalize-memref-to-llvm,
# expand-strided-metadata, convert-openmp-to-llvm, convert-vector-to-llvm,
# convert-func-to-llvm, …) with the textual pass-pipeline parser, and the
# `mlirExecutionEngine*` C API isn't exposed at all. So we drive `mlir-opt`
# and `mlir-translate` from `MLIR_jll` (which include all upstream passes),
# then `clang` to build a `.so`, then `dlopen` it.
#
# When (or if) `libReactantExtra.so` exposes the missing pieces this can
# collapse to in-process IR.run!(pm, mod) + ExecutionEngine.

# libomp from LLVMOpenMP_jll. `libomp_path` is the JLL's lazy accessor; we
# resolve to a directory + `-lomp` link line for clang. Override via
# `ENV["CUTILECPU_LIBOMP_DIR"]` if you need a custom build.
_libomp_dir() = get(ENV, "CUTILECPU_LIBOMP_DIR", dirname(libomp_path))

# mlir-opt, mlir-translate, clang from LLVM_full_jll. The JLL only exposes
# `artifact_dir`; the tools live in `tools/`. Override individually via env
# vars (CUTILECPU_MLIR_OPT, CUTILECPU_MLIR_TRANSLATE, CUTILECPU_CLANG) if
# you want a different toolchain.
_mlir_opt() = get(ENV, "CUTILECPU_MLIR_OPT",
                  joinpath(LLVM_full_jll.artifact_dir, "tools", "mlir-opt"))
_mlir_translate() = get(ENV, "CUTILECPU_MLIR_TRANSLATE",
                        joinpath(LLVM_full_jll.artifact_dir, "tools", "mlir-translate"))
_clang() = get(ENV, "CUTILECPU_CLANG",
               joinpath(LLVM_full_jll.artifact_dir, "tools", "clang"))

# Default lowering pipeline: cuTile-style `scf.parallel` + vector dialect →
# OpenMP → LLVM dialect → LLVM IR. Use the SERIAL_PASSES pipeline below to
# drop OpenMP entirely (no libomp dependency, kernel runs single-threaded).
const DEFAULT_PASSES = String[
    # Lower `vector.multi_reduction` into individual `vector.reduction`
    # (and friends) so the later vector-to-llvm pass can handle it.
    "--lower-vector-multi-reduction",
    "--convert-vector-to-scf",
    "--convert-scf-to-openmp",
    # `scf-to-openmp` wraps the kernel body in `memref.alloca_scope`, which
    # requires its region to have at most one basic block. `scf-to-cf` would
    # multiply blocks. We sidestep this by running `--convert-openmp-to-llvm`
    # first to rewrite the alloca_scope into LLVM regions that tolerate
    # branches, and only then convert scf.if/for to cf.
    "--convert-openmp-to-llvm",
    "--convert-scf-to-cf",
    "--lower-affine",
    # `vector-contract-lowering=outerproduct`: lower `vector.contract` to
    # `vector.outerproduct`, which then becomes a tight loop of FMAs along
    # the reduction axis. The default `=dot` strategy fully unrolls a 64×64
    # matmul into ~500k scalar ops; the `=matmul` strategy bottlenecks too;
    # outerproduct stays compact (~25k LLVM IR lines for our test sizes).
    "--convert-vector-to-llvm=vector-contract-lowering=outerproduct",
    "--expand-strided-metadata",
    "--finalize-memref-to-llvm",
    # `math.exp` (and friends) on f32/vector<...xf32> → llvm.intr.exp etc.
    # Must come before `--convert-arith-to-llvm` so any arith ops the math
    # lowering generates are still in the arith dialect for that pass.
    "--convert-math-to-llvm",
    "--convert-arith-to-llvm",
    "--convert-func-to-llvm",
    "--convert-cf-to-llvm",
    # `vector.contract` lowering can introduce `ub.poison` for undef padding;
    # the LLVM-IR translator needs this lowered first.
    "--convert-ub-to-llvm",
    "--reconcile-unrealized-casts",
]

# Single-threaded lowering pipeline: same shape as DEFAULT_PASSES, but
# `convert-scf-to-cf` runs directly on `scf.parallel` (it gets degraded to a
# serial `scf.for` automatically) and we never touch the `omp` dialect. The
# resulting `.so` has no libomp dependency and runs the entire grid on the
# calling thread.
#
# Useful for:
# - Debugging (no thread interleave — deterministic execution order)
# - Small grids where OpenMP's ~70 μs fork/join floor dominates the work
# - Deployments without LLVMOpenMP_jll
const SERIAL_PASSES = String[
    "--lower-vector-multi-reduction",
    "--convert-vector-to-scf",
    # NO `--convert-scf-to-openmp`. scf.parallel runs serially via scf-to-cf.
    "--convert-scf-to-cf",
    "--lower-affine",
    "--convert-vector-to-llvm=vector-contract-lowering=outerproduct",
    "--expand-strided-metadata",
    "--finalize-memref-to-llvm",
    "--convert-math-to-llvm",
    "--convert-arith-to-llvm",
    "--convert-func-to-llvm",
    "--convert-cf-to-llvm",
    "--convert-ub-to-llvm",
    "--reconcile-unrealized-casts",
]

"""
    lower_mlir_text(mlir_text::String) -> String

Run the standard CPU lowering pipeline on a textual MLIR module. Returns
LLVM-dialect MLIR (still textual). Throws if `mlir-opt` fails.
"""
function lower_mlir_text(mlir_text::String; passes=DEFAULT_PASSES)
    workdir = mktempdir(; prefix="cuTileCPU_lower_")
    in_path  = joinpath(workdir, "kernel.mlir")
    write(in_path, mlir_text)
    exe = _mlir_opt()
    return read(`$exe $in_path $passes`, String)
end

"""
    translate_to_llvmir(lowered_mlir::String) -> String

`mlir-translate --mlir-to-llvmir` on LLVM-dialect MLIR.
"""
function translate_to_llvmir(lowered_mlir::String)
    exe = _mlir_translate()
    out_path = tempname() * ".ll"
    open(pipeline(`$exe --mlir-to-llvmir`; stdout=out_path), "r+") do io
        write(io, lowered_mlir)
        close(io.in)
    end
    return read(out_path, String)
end

"""
    compile_to_so(mlir_text::String; kernel_name, passes=DEFAULT_PASSES) -> so_path

End-to-end: MLIR → lowered MLIR → LLVM IR → `.so`. The .so is rpath-linked
against libomp (from LLVMOpenMP_jll, or `CUTILECPU_LIBOMP_DIR` override) when
the pipeline emits OpenMP runtime calls — `passes === SERIAL_PASSES` (or any
pipeline without `--convert-scf-to-openmp` / `--convert-openmp-to-llvm`)
skips the libomp link.

The libomp-need detection is *static*: if any of the input pass flags
mentions `openmp`, we link libomp. Otherwise we don't.
"""
function compile_to_so(mlir_text::String; kernel_name::String,
                       opt_level::Int=2, passes=DEFAULT_PASSES,
                       clang::String=_clang())
    workdir = mktempdir(; prefix="cuTileCPU_$(kernel_name)_")
    mlir_path = joinpath(workdir, "$(kernel_name).mlir")
    llvm_path = joinpath(workdir, "$(kernel_name).ll")
    so_path   = joinpath(workdir, "$(kernel_name).so")
    write(mlir_path, mlir_text)

    opt_exe = _mlir_opt()
    lowered = read(`$opt_exe $mlir_path $passes`, String)

    tr_exe = _mlir_translate()
    open(pipeline(`$tr_exe --mlir-to-llvmir`; stdout=llvm_path), "r+") do io
        write(io, lowered)
        close(io.in)
    end

    needs_libomp = any(p -> occursin("openmp", p), passes)
    if needs_libomp
        libomp_dir = _libomp_dir()
        run(`$clang -O$opt_level -shared -fPIC $llvm_path
             -L$libomp_dir -Wl,-rpath,$libomp_dir -lomp
             -o $so_path`)
    else
        run(`$clang -O$opt_level -shared -fPIC $llvm_path -o $so_path`)
    end
    return so_path
end
