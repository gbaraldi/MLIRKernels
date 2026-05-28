# End-to-end KernelAbstractions.jl kernel running through MLIRKernels's
# MLIR pipeline. Loads the package extension and launches a vadd kernel
# through the same path as the SPMD/cuTile backends.
#
# Pattern (mirrors CUDA.jl / AMDGPU.jl / oneAPI.jl / Metal.jl):
#
#   - `MLIRBackend <: KA.GPU` (defined in `ext/KernelAbstractionsExt.jl`)
#     picks the SIMT `gpu_*` body that `@kernel` emits.
#
#   - `@overlay cuTile.cuTileMethodTable` redefines KA intrinsics so cuTile's
#     interpreter sees overlay bodies (no KA calls in the inferred IR).
#     `__index_Global_Linear` is overlaid to a sentinel function the walker
#     binds to the SPMD lane vector.
#
#   - `(::Kernel{MLIRBackend})(args...; ndrange, workgroupsize)` calls
#     `ka_function` → `lower_to_mlir_ka` → in-process MLIR pipeline → clang
#     → dlopen, then dispatches via the standard SPMD launch path.

using KernelAbstractions
const KA = KernelAbstractions
using cuTile  # triggers the extension on MLIRKernels
using MLIRKernels

# Reach into the package extension. Julia's standard pattern after 1.9 is
# `Base.get_extension(parent, :ExtName)`; this fires only after both the
# parent and the weakdep have been loaded.
const KAExt = Base.get_extension(MLIRKernels, :KernelAbstractionsExt)
const MLIRBackend = KAExt.MLIRBackend

@kernel function vadd!(C, A, B)
    i = @index(Global, Linear)
    @inbounds C[i] = A[i] + B[i]
end

const N = 1024
A = MLIRKernels.aligned_array(Float32, N; alignment=128); copyto!(A, 1:N)
B = MLIRKernels.aligned_array(Float32, N; alignment=128); copyto!(B, (N+1):(2N))
C = MLIRKernels.aligned_array(Float32, N; alignment=128); fill!(C, 0f0)

kernel = vadd!(MLIRBackend(), 16)
kernel(C, A, B; ndrange=N)

@assert C ≈ A .+ B
println("✓ KernelAbstractions vadd compiled + ran via MLIRKernels")
println("  first 4: ", C[1:4])
println("  last 4:  ", C[(N-3):N])
