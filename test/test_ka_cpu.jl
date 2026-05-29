# test_ka_cpu.jl — extracted from the former monolithic runtests.jl.
# Shared usings + kernel defs come from setup.jl (via init_code).

# KernelAbstractions CPU backend (MLIRBackend <: KA.GPU). Guarded on KA
# being loadable: the package's own test env (--project=.) has KA only as
# a weakdep, so this skips there and runs whenever KA is present (e.g. an
# env that adds it). Validates the fully cuTile-decoupled KA path: KA
# @kernel → Frontend.structured (own interpreter/intrinsics) → MLIR → clang.
@testset "KA: vadd via MLIRBackend (CPU, decoupled)" begin
    ka_loaded = try
        @eval using KernelAbstractions
        true
    catch
        false
    end
    if !ka_loaded
        @info "KernelAbstractions not in this env — skipping KA backend test"
        @test true  # placeholder so the testset is non-empty
    else
        KA = KernelAbstractions
        KAExt = Base.get_extension(MLIRKernels, :KernelAbstractionsExt)
        Backend = KAExt.MLIRBackend
        @eval begin
            @kernel function _ka_vadd!(C, A, B)
                i = @index(Global, Linear)
                @inbounds C[i] = A[i] + B[i]
            end
        end
        N = 4096; W = 16
        A = MLIRKernels.aligned_array(Float32, N; alignment=128); copyto!(A, rand(Float32, N))
        B = MLIRKernels.aligned_array(Float32, N; alignment=128); copyto!(B, rand(Float32, N))
        C = MLIRKernels.aligned_array(Float32, N; alignment=128); fill!(C, 0f0)
        (@eval _ka_vadd!)(Backend(), W)(C, A, B; ndrange=N)
        @test C ≈ A .+ B
        # The @noinline global_index marker must survive inference under
        # the Frontend interpreter (default opt params) — i.e. appear as a
        # call in the SCI, not be inlined/folded away.
        gpu_body = @eval gpu__ka_vadd!
        ctxT = let
            ndr = KA.NDIteration.StaticSize{(N,)}
            wg  = KA.NDIteration.StaticSize{(W,)}
            grp = KA.NDIteration.StaticSize{(N ÷ W,)}
            ndro = KA.NDIteration.NDRange{1, grp, wg, Nothing, Nothing}
            KA.CompilerMetadata{ndr, KA.NDIteration.NoDynamicCheck, Nothing, Nothing, ndro}
        end
        sci, rt = MLIRKernels.Frontend.structured(gpu_body,
            Tuple{ctxT, Vector{Float32}, Vector{Float32}, Vector{Float32}})
        @test rt === Nothing
        @test occursin("global_index", sprint(show, sci))
    end
end

# KA.@atomic — KernelAbstractions' *portable* atomic (= Atomix.@atomic,
# which CUDA/AMDGPU/oneAPI all override for device arrays). The KA extension
# overlays `Atomix.modify!(IndexableRef, op, x, ord)` onto the Frontend
# `atomic_index!` marker, which the walker lowers to `memref.atomic_rmw`.
# Covers: varying-index float add (per-lane scatter), uniform-slot float add
# and integer max/min (lane-reduction → single atomic per block), atomicity
# under cross-block contention, and the float-min/max MLIR-version gate.
@testset "KA: @atomic via MLIRBackend (Atomix portable path)" begin
    # The portable atomic is Atomix's `@atomic` — KA itself just re-exports
    # it (`import Atomix: @atomic`, cf. KernelAbstractions/examples/
    # histogram.jl), and our KA extension overlays `Atomix.modify!` onto the
    # Frontend `atomic_index!` marker. We import it straight from Atomix so
    # bare `@atomic` in a kernel is unambiguously the portable atomic, NOT
    # Base's scalar `@atomic`.
    ka_loaded = try
        @eval using KernelAbstractions
        @eval using Atomix: @atomic
        true
    catch
        false
    end
    if !ka_loaded
        @info "KernelAbstractions not in this env — skipping KA.@atomic test"
        @test true
    else
        KA = KernelAbstractions
        Backend = Base.get_extension(MLIRKernels, :KernelAbstractionsExt).MLIRBackend
        W = 16
        @eval begin
            @kernel function _ka_hist!(bins, @Const(idx))
                i = @index(Global, Linear)
                @inbounds @atomic bins[idx[i]] += 1f0
            end
            @kernel function _ka_amax!(out, @Const(x))
                i = @index(Global, Linear)
                @inbounds @atomic out[1] max x[i]
            end
            @kernel function _ka_amin!(out, @Const(x))
                i = @index(Global, Linear)
                @inbounds @atomic out[1] min x[i]
            end
        end

        # (a) Histogram: varying per-lane index → per-lane atomic scatter.
        N = 4096; NB = 8
        idx  = Int32[(j % NB) + 1 for j in 0:N-1]
        bins = MLIRKernels.aligned_array(Float32, NB; alignment=128); fill!(bins, 0f0)
        (@eval _ka_hist!)(Backend(), W)(bins, idx; ndrange=N)
        @test all(==(Float32(N ÷ NB)), bins)

        # (b) Atomicity: every lane → one slot; no lost updates across blocks.
        M = 65536
        ones_idx = ones(Int32, M)
        acc = MLIRKernels.aligned_array(Float32, 1; alignment=128); fill!(acc, 0f0)
        (@eval _ka_hist!)(Backend(), W)(acc, ones_idx; ndrange=M)
        @test acc[1] == Float32(M)

        # (c) Integer max/min into a uniform slot → vector.reduction + one
        #     atomic per block (maxs/mins lower on every supported MLIR).
        Ni = 256
        xi = Int32.(collect(1:Ni)); xi[100] = Int32(9999)
        omax = MLIRKernels.aligned_array(Int32, 1; alignment=128); omax[1] = typemin(Int32)
        omin = MLIRKernels.aligned_array(Int32, 1; alignment=128); omin[1] = typemax(Int32)
        (@eval _ka_amax!)(Backend(), W)(omax, xi; ndrange=Ni)
        (@eval _ka_amin!)(Backend(), W)(omin, xi; ndrange=Ni)
        @test omax[1] == 9999
        @test omin[1] == 1

        # (d) The Atomix overlay → atomic_index! marker must survive inference
        #     (DCE would otherwise delete the unused-result call).
        gpu_body = @eval gpu__ka_hist!
        ctxT = let
            ndr = KA.NDIteration.StaticSize{(N,)}; wg = KA.NDIteration.StaticSize{(W,)}
            grp = KA.NDIteration.StaticSize{(N ÷ W,)}
            ndro = KA.NDIteration.NDRange{1, grp, wg, Nothing, Nothing}
            KA.CompilerMetadata{ndr, KA.NDIteration.NoDynamicCheck, Nothing, Nothing, ndro}
        end
        sci, rt = MLIRKernels.Frontend.structured(gpu_body,
            Tuple{ctxT, Vector{Float32}, Vector{Int32}})
        @test occursin("atomic_index!", sprint(show, sci))

        # (e) Float min/max atomics need MLIR ≥ 21 (LLVM 20's memref→llvm
        #     doesn't lower `maxnumf`/`minnumf`). On older MLIR the walker
        #     raises a clear, actionable error rather than emitting IR that
        #     dies at LLVM translation; on MLIR ≥ 21 it just works.
        xf = Float32.(collect(1:Ni)); xf[100] = 9999f0
        of = MLIRKernels.aligned_array(Float32, 1; alignment=128); of[1] = -Inf32
        if MLIRKernels.MLIR.MLIR_VERSION[] < v"21"
            @test_throws Exception (@eval _ka_amax!)(Backend(), W)(of, xf; ndrange=Ni)
        else
            (@eval _ka_amax!)(Backend(), W)(of, xf; ndrange=Ni)
            @test of[1] == 9999f0
        end

        # (f) Counter idiom `@atomic out[1] += c` with a UNIFORM scalar
        #     value. Each of the W lanes runs the statement, so the slot must
        #     gain W*c per block (== ndrange total). A naive single atomic per
        #     block would undercount by exactly W — guard against that
        #     regression (the value is broadcast to W lanes then reduced).
        @eval begin
            @kernel function _ka_ctr!(out)
                i = @index(Global, Linear)
                @inbounds @atomic out[1] += 1f0
            end
        end
        ctr = MLIRKernels.aligned_array(Float32, 1; alignment=128); ctr[1] = 0f0
        (@eval _ka_ctr!)(Backend(), W)(ctr; ndrange=256)
        @test ctr[1] == 256f0   # NOT 256/W

        # (g) The backend lowers a 1-D grid/workgroup; multi-dimensional
        #     ndrange or workgroupsize would silently corrupt Local/Group
        #     indices, so the launcher must reject them. And a launch-time
        #     workgroupsize that conflicts with a static one must error
        #     rather than be silently ignored.
        @test_throws Exception (@eval _ka_ctr!)(Backend(), W)(ctr; ndrange=(8, 4))
        @test_throws Exception (@eval _ka_ctr!)(Backend(), W)(ctr; ndrange=256, workgroupsize=2W)
    end
end

# Multi-dimensional support: N-D `@index(Global, NTuple)` + N-D array
# indexing `A[i,j]`. The workgroup is flattened to a 1-D lane vector, per-dim
# coords reconstructed by column-major unflatten, and `A[i,j]` linearised
# (column-major) to a gather/scatter over a flattened (`reinterpret_cast`)
# rank-1 view. 2-D transpose (KA's `naive_transpose`) is the end-to-end gate.
@testset "KA: multi-dim @index(Global, NTuple) + A[i,j]" begin
    ka_loaded = try; @eval using KernelAbstractions; true; catch; false; end
    if !ka_loaded
        @info "KernelAbstractions not in this env — skipping multi-dim test"
        @test true
    else
        KA = KernelAbstractions
        Backend = Base.get_extension(MLIRKernels, :KernelAbstractionsExt).MLIRBackend
        @eval begin
            @kernel function _ka_transpose!(a, @Const(b))
                i, j = @index(Global, NTuple)
                @inbounds a[i, j] = b[j, i]
            end
        end
        M = 8
        b = MLIRKernels.aligned_array(Float32, (M, M); alignment=128)
        copyto!(b, reshape(collect(Float32, 1:(M * M)), M, M))
        a = MLIRKernels.aligned_array(Float32, (M, M); alignment=128)
        fill!(a, 0f0)
        (@eval _ka_transpose!)(Backend(), (4, 4))(a, b; ndrange=(M, M))
        @test a == permutedims(b)            # full N-D index + linearised A[i,j]
    end
end

