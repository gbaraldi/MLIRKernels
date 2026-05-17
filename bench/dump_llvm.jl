# Dump LLVM IR for the two matmul variants to compare what clang actually
# generated. Writes contract.ll and reg.ll into bench/llvm_dump/.

using cuTile
const ct = cuTile
using cuTileCPU

# Same kernels as bench_matmul_reg.jl
function matmul_reg_kernel(A::ct.TileArray{T,2}, B::ct.TileArray{T,2},
                           C::ct.TileArray{T,2},
                           RM::Int, RN::Int, K::Int) where {T}
    bid_m = ct.bid(1)
    bid_n = ct.bid(2)
    acc = zeros(T, (RM, RN))
    for k in 1:K
        a_col = ct.load(A; index=(bid_m, k), shape=(RM, 1))
        b_row = ct.load(B; index=(k, bid_n), shape=(1, RN))
        acc = muladd(a_col, b_row, acc)
    end
    ct.store(C; index=(bid_m, bid_n), tile=acc)
    return
end

function matmul_tile_kernel(A::ct.TileArray{T,2}, B::ct.TileArray{T,2},
                            C::ct.TileArray{T,2},
                            BM::Int, BN::Int, BK::Int) where {T}
    bid_m = ct.bid(1)
    bid_n = ct.bid(2)
    acc = zeros(T, (BM, BN))
    for k in 1:cld(size(A, 2), BK)
        a = ct.load(A; index=(bid_m, k),     shape=(BM, BK))
        b = ct.load(B; index=(k,     bid_n), shape=(BK, BN))
        acc = muladd(a, b, acc)
    end
    ct.store(C; index=(bid_m, bid_n), tile=acc)
    return
end

M = N = K = 1024
A = cuTileCPU.aligned_array(Float32, M, K; alignment=128)
B = cuTileCPU.aligned_array(Float32, K, N; alignment=128)
C = cuTileCPU.aligned_array(Float32, M, N; alignment=128)

mkpath("llvm_dump")

# Contract path (tile 16×16×16 — the best contract result)
println("Dumping contract LLVM (tile 16×16×16, 1024³)...")
ll_contract = cuTileCPU.code_llvm(matmul_tile_kernel,
    (A, B, C, ct.Constant(16), ct.Constant(16), ct.Constant(16));
    n_grid_dims=2)
write("llvm_dump/contract_16.ll", ll_contract)
println("  $(count('\n', ll_contract)) lines → llvm_dump/contract_16.ll")

# Contract path tile 64×64×64
println("Dumping contract LLVM (tile 64×64×64, 1024³)...")
ll_contract_64 = cuTileCPU.code_llvm(matmul_tile_kernel,
    (A, B, C, ct.Constant(64), ct.Constant(64), ct.Constant(64));
    n_grid_dims=2)
write("llvm_dump/contract_64.ll", ll_contract_64)
println("  $(count('\n', ll_contract_64)) lines → llvm_dump/contract_64.ll")

# Register tile path
println("Dumping reg-tile LLVM (16×16, K=1024)...")
ll_reg = cuTileCPU.code_llvm(matmul_reg_kernel,
    (A, B, C, ct.Constant(16), ct.Constant(16), ct.Constant(1024));
    n_grid_dims=2)
write("llvm_dump/reg_16.ll", ll_reg)
println("  $(count('\n', ll_reg)) lines → llvm_dump/reg_16.ll")
