using MLIRKernels
using ParallelTestRunner

# Discovery scans pwd() by default; pin it to this directory.
cd(@__DIR__)

# Shared setup (see setup.jl) is re-evaluated in each test file's sandbox module.
const init_code = quote
    include($(joinpath(@__DIR__, "setup.jl")))
end

# Auto-discover test files; drop the non-test helpers from the walked tree.
testsuite = find_tests(@__DIR__)
delete!(testsuite, "setup")
delete!(testsuite, "dump_sci")

runtests(MLIRKernels, ARGS; testsuite, init_code)
