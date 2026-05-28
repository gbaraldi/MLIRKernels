// vadd, but with the two addressing knobs that experiment 03 flagged as
// the source of MLIR's 3-instruction overhead vs CUDA.jl SIMT:
//
//   1. memref args carry `#gpu.address_space<global>` — so the base
//      pointers are already global-qualified and the `cvta.to.global`
//      generic→global casts fold away.
//
//   2. the index is computed in i32 (`gpu.thread_id` etc. are index-typed,
//      so we index_cast to i32 and do the gid math in 32-bit) — matching
//      CUDA.jl's `mad.lo.s32` instead of MLIR's default i64 `mul.wide +
//      add.s64 + shl.b64`.
//
// Compare the emitted PTX against 03's MLIR body and against CUDA.jl SIMT.

module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @vadd(%a: memref<?xf32, #gpu.address_space<global>>,
                   %b: memref<?xf32, #gpu.address_space<global>>,
                   %c: memref<?xf32, #gpu.address_space<global>>,
                   %n: i32) kernel {
      %tid  = gpu.thread_id x
      %bid  = gpu.block_id x
      %bdim = gpu.block_dim x
      // Do the index math in i32 to match CUDA.jl's 32-bit `mad.lo.s32`.
      %tid_i  = arith.index_cast %tid  : index to i32
      %bid_i  = arith.index_cast %bid  : index to i32
      %bdim_i = arith.index_cast %bdim : index to i32
      %off  = arith.muli %bid_i, %bdim_i : i32
      %gid  = arith.addi %off, %tid_i : i32
      %ib   = arith.cmpi ult, %gid, %n : i32
      scf.if %ib {
        // memref.load wants an index, so widen the i32 gid once here.
        %gidx = arith.index_cast %gid : i32 to index
        %av = memref.load %a[%gidx] : memref<?xf32, #gpu.address_space<global>>
        %bv = memref.load %b[%gidx] : memref<?xf32, #gpu.address_space<global>>
        %sum = arith.addf %av, %bv : f32
        memref.store %sum, %c[%gidx] : memref<?xf32, #gpu.address_space<global>>
      }
      gpu.return
    }
  }
}
