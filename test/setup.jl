# Shared setup for the ParallelTestRunner test files. Evaluated (via
# runtests.jl `init_code`) into each test file's sandbox module before the file
# runs. The KA path is cuTile-free, so this is intentionally minimal —
# CUDA/LLVM/KernelAbstractions/Atomix are conditionally loaded inside the
# guarded GPU/KA test files themselves.
using MLIRKernels
