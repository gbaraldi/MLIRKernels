# Dump SCIs for kernels exercising IfOp, ForOp, and reductions so we can see
# exactly what shape the walker has to handle. Not a test — just inspection.

using cuTile
const ct = cuTile

function if_branch(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                   flag::Int32)
    bid = ct.bid(1)
    tile = ct.load(a; index=bid, shape=(16,))
    if flag == Int32(0)
        return nothing
    end
    ct.store(b; index=bid, tile = tile * 2.0f0)
    return
end

function counted_loop(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                      n::Int32)
    bid = ct.bid(1)
    acc = zeros(Float32, (16,))
    for j in Int32(1):n
        acc = acc + ct.load(a; index=bid, shape=(16,))
    end
    ct.store(b; index=bid, tile=acc)
    return
end

function row_sum(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,1})
    bid = ct.bid(1)
    tile = ct.load(a; index=(bid, 1), shape=(1, 128))
    s = sum(tile; dims=2)
    ct.store(b; index=bid, tile=s)
    return
end

function softmax_row(A::ct.TileArray{Float32,2}, Y::ct.TileArray{Float32,2},
                     BLOCK_N::Int)
    bid = ct.bid(1)
    row = ct.load(A; index=(bid, 1), shape=(1, BLOCK_N))
    m   = maximum(row; dims=2)
    shifted = row .- m
    e   = exp.(shifted)
    s   = sum(e; dims=2)
    y   = e ./ s
    ct.store(Y; index=(bid, 1), tile=y)
    return
end

function vadd_gather(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                     c::ct.TileArray{Float32,1}, tile::Int)
    bid = ct.bid(1)
    offsets = ct.arange(tile)
    base = ct.Tile((bid - Int32(1)) * Int32(tile))
    indices = ct.broadcast_to(base, (tile,)) .+ offsets

    a_tile = ct.gather(a, indices)
    b_tile = ct.gather(b, indices)
    sum_tile = a_tile + b_tile
    ct.scatter(c, indices, sum_tile)
    return
end

function layernorm_row(X::ct.TileArray{Float32,2}, Y::ct.TileArray{Float32,2},
                       eps::Float32, BLOCK_N::Int)
    bid = ct.bid(1)
    x   = ct.load(X; index=(bid, 1), shape=(1, BLOCK_N))
    n   = Float32(BLOCK_N)
    μ   = sum(x; dims=2) ./ n
    Δ   = x .- μ
    σ²  = sum(Δ .* Δ; dims=2) ./ n
    inv = ct.rsqrt.(σ² .+ eps)
    y   = Δ .* inv
    ct.store(Y; index=(bid, 1), tile=y)
    return
end

function layernorm_bwd_row(X::ct.TileArray{Float32,2},
                           dY::ct.TileArray{Float32,2},
                           dX::ct.TileArray{Float32,2},
                           eps::Float32, BLOCK_N::Int)
    bid = ct.bid(1)
    x  = ct.load(X;  index=(bid, 1), shape=(1, BLOCK_N))
    dy = ct.load(dY; index=(bid, 1), shape=(1, BLOCK_N))

    n  = Float32(BLOCK_N)
    μ  = sum(x; dims=2) ./ n
    Δ  = x .- μ
    σ² = sum(Δ .* Δ; dims=2) ./ n
    inv_std = ct.rsqrt.(σ² .+ eps)
    x_hat = Δ .* inv_std

    sum1 = sum(dy; dims=2)
    sum2 = sum(dy .* x_hat; dims=2)
    dx = (dy .- sum1 ./ n .- x_hat .* (sum2 ./ n)) .* inv_std

    ct.store(dX; index=(bid, 1), tile=dx)
    return
end

function attention_kernel(Q::ct.TileArray{Float32,2}, K::ct.TileArray{Float32,2},
                          V::ct.TileArray{Float32,2}, O::ct.TileArray{Float32,2},
                          BM::Int, BN::Int, D::Int)
    bid = ct.bid(1)
    q = ct.load(Q; index=(bid, 1), shape=(BM, D))
    k = ct.load(K; index=(1, 1),   shape=(BN, D))
    v = ct.load(V; index=(1, 1),   shape=(BN, D))
    s_acc = zeros(Float32, (BM, BN))
    kT = permutedims(k, (2, 1))
    s = muladd(q, kT, s_acc)
    m = maximum(s; dims=2)
    sh = s .- m
    e = exp.(sh)
    ssum = sum(e; dims=2)
    p = e ./ ssum
    o_acc = zeros(Float32, (BM, D))
    o = muladd(p, v, o_acc)
    ct.store(O; index=(bid, 1), tile=o)
    return
end

function flash_attn_kernel(Q::ct.TileArray{Float32,2}, K::ct.TileArray{Float32,2},
                           V::ct.TileArray{Float32,2}, O::ct.TileArray{Float32,2},
                           BM::Int, BN::Int, D::Int, N_KV::Int)
    bid = ct.bid(1)
    q = ct.load(Q; index=(bid, 1), shape=(BM, D))
    m = ct.fill(-Inf32, (BM, 1))
    l = zeros(Float32, (BM, 1))
    o = zeros(Float32, (BM, D))
    for kbi in 1:cld(N_KV, BN)
        k = ct.load(K; index=(kbi, 1), shape=(BN, D))
        v = ct.load(V; index=(kbi, 1), shape=(BN, D))
        kT = permutedims(k, (2, 1))
        s = muladd(q, kT, zeros(Float32, (BM, BN)))
        m_new_chunk = maximum(s; dims=2)
        m_new = max.(m, m_new_chunk)
        α = exp.(m .- m_new)
        p = exp.(s .- m_new)
        l = α .* l .+ sum(p; dims=2)
        o = α .* o .+ muladd(p, v, zeros(Float32, (BM, D)))
        m = m_new
    end
    o = o ./ l
    ct.store(O; index=(bid, 1), tile=o)
    return
end

const Spec1 = ct.ArraySpec{1}(128, true, (0,), (32,))
const Spec2 = ct.ArraySpec{2}(128, true, (0, 0), (32, 32))
const Spec3 = ct.ArraySpec{3}(128, true, (0, 0, 0), (32, 32, 32))
const TA1 = ct.TileArray{Float32, 1, Spec1}
const TA2 = ct.TileArray{Float32, 2, Spec2}
const TA3 = ct.TileArray{Float32, 3, Spec3}

# 1-stage complex DFT kernel via batched matrix DFT. Inputs / outputs are packed
# (real, imag) along a leading dim of size 2.
function dft_kernel(X_packed::ct.TileArray{Float32,3},   # (2, N, BS)
                    Y_packed::ct.TileArray{Float32,3},   # (2, N, BS)
                    W_packed::ct.TileArray{Float32,3},   # (2, N, N)
                    N::Int, BS::Int)
    bid = ct.bid(1)
    X_ri = ct.load(X_packed; index=(1, 1, 1), shape=(2, N, BS))
    X_r = reshape(ct.extract(X_ri, (1, 1, 1), (1, N, BS)), (N, BS))
    X_i = reshape(ct.extract(X_ri, (2, 1, 1), (1, N, BS)), (N, BS))

    W_ri = ct.load(W_packed; index=(1, 1, 1), shape=(2, N, N))
    W_r = reshape(ct.extract(W_ri, (1, 1, 1), (1, N, N)), (N, N))
    W_i = reshape(ct.extract(W_ri, (2, 1, 1), (1, N, N)), (N, N))

    # Y_r = W_r * X_r - W_i * X_i
    # Y_i = W_r * X_i + W_i * X_r
    Y_r = W_r * X_r - W_i * X_i
    Y_i = W_r * X_i + W_i * X_r

    Y_r_packed = reshape(Y_r, (1, N, BS))
    Y_i_packed = reshape(Y_i, (1, N, BS))
    Y_ri = ct.cat((Y_r_packed, Y_i_packed), 1)
    ct.store(Y_packed; index=(1, 1, 1), tile=Y_ri)
    return
end

println("===== if_branch =====")
for (sci, rt) in ct.code_structured(if_branch, Tuple{TA1, TA1, Int32}; optimize=true)
    show(stdout, MIME"text/plain"(), sci); println()
end

println("\n===== counted_loop =====")
for (sci, rt) in ct.code_structured(counted_loop, Tuple{TA1, TA1, Int32}; optimize=true)
    show(stdout, MIME"text/plain"(), sci); println()
end

println("\n===== row_sum =====")
for (sci, rt) in ct.code_structured(row_sum, Tuple{TA2, TA1}; optimize=true)
    show(stdout, MIME"text/plain"(), sci); println()
end

println("\n===== softmax_row =====")
for (sci, rt) in ct.code_structured(softmax_row, Tuple{TA2, TA2, ct.Constant{Int,128}}; optimize=true)
    show(stdout, MIME"text/plain"(), sci); println()
end

println("\n===== layernorm_row =====")
for (sci, rt) in ct.code_structured(layernorm_row, Tuple{TA2, TA2, Float32, ct.Constant{Int,128}}; optimize=true)
    show(stdout, MIME"text/plain"(), sci); println()
end

println("\n===== layernorm_bwd_row =====")
for (sci, rt) in ct.code_structured(layernorm_bwd_row, Tuple{TA2, TA2, TA2, Float32, ct.Constant{Int,128}}; optimize=true)
    show(stdout, MIME"text/plain"(), sci); println()
end

println("\n===== vadd_gather =====")
for (sci, rt) in ct.code_structured(vadd_gather, Tuple{TA1, TA1, TA1, ct.Constant{Int,16}}; optimize=true)
    show(stdout, MIME"text/plain"(), sci); println()
end

println("\n===== attention_kernel =====")
for (sci, rt) in ct.code_structured(attention_kernel,
    Tuple{TA2, TA2, TA2, TA2,
          ct.Constant{Int,64}, ct.Constant{Int,64}, ct.Constant{Int,64}};
    optimize=true)
    show(stdout, MIME"text/plain"(), sci); println()
end

println("\n===== flash_attn_kernel =====")
for (sci, rt) in ct.code_structured(flash_attn_kernel,
    Tuple{TA2, TA2, TA2, TA2,
          ct.Constant{Int,32}, ct.Constant{Int,32}, ct.Constant{Int,64},
          ct.Constant{Int,128}};
    optimize=true)
    show(stdout, MIME"text/plain"(), sci); println()
end

println("\n===== dft_kernel =====")
for (sci, rt) in ct.code_structured(dft_kernel,
    Tuple{TA3, TA3, TA3,
          ct.Constant{Int,16}, ct.Constant{Int,2}};
    optimize=true)
    show(stdout, MIME"text/plain"(), sci); println()
end

# MoE routing kernel — Mixture of Experts (option 3: per-block expert dispatch).
# One block per token: read pre-assigned expert id, atomically claim a slot in
# the expert's region of Y, load the token + expert weights, matmul, store.
const TAI1 = ct.TileArray{Int32, 1, Spec1}
function moe_routing_kernel(
        X::ct.TileArray{Float32,2},
        Y::ct.TileArray{Float32,2},
        expert_ids::ct.TileArray{Int32,1},
        counters::ct.TileArray{Int32,1},
        slot_tokens::ct.TileArray{Int32,1},
        Wexp::ct.TileArray{Float32,3},
        D::Int, D_out::Int, MAX_PER_EXPERT::Int)
    bid = ct.bid(1)
    expert = expert_ids[bid]
    slot = ct.atomic_add(counters, expert, Int32(1);
                         memory_order=ct.MemoryOrder.AcqRel)
    slot_in_y = (expert - Int32(1)) * Int32(MAX_PER_EXPERT) + slot + Int32(1)
    slot_tokens[slot_in_y] = bid
    x_tile = ct.load(X; index=(Int32(1), bid), shape=(D, 1))
    w_tile = ct.load(Wexp; index=(Int32(1), Int32(1), expert),
                     shape=(D_out, D, 1))
    w_2d = reshape(w_tile, (D_out, D))
    acc = zeros(Float32, (D_out, 1))
    y_tile = muladd(w_2d, x_tile, acc)
    ct.store(Y; index=(Int32(1), slot_in_y), tile=y_tile)
    return
end

println("\n===== moe_routing_kernel =====")
for (sci, rt) in ct.code_structured(moe_routing_kernel,
    Tuple{TA2, TA2, TAI1, TAI1, TAI1, TA3,
          ct.Constant{Int,32}, ct.Constant{Int,32}, ct.Constant{Int,8}};
    optimize=true)
    show(stdout, MIME"text/plain"(), sci); println()
end
