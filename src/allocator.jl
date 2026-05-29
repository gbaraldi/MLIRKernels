# Aligned heap allocations as plain `Array{T,N}`.
#
# Julia's `Array{T}` is only 16-byte aligned (the GC heap doesn't promise
# more). An `alignment` hint of 64 or 128 bytes lets the vectorizer emit
# aligned vector load/stores; when we lower with
# `memref.assume_alignment %ptr, N` and the host hands us a less-aligned
# buffer, the program is UB.
#
# We allocate with `posix_memalign`, wrap the pointer in a real `Array` via
# `unsafe_wrap(Array, ..., own=true)`, and let Julia's GC call `free()` when
# the array becomes unreachable. The result is a normal `Array{T,N}` —
# indexable, broadcastable, etc. — that passes every type test for
# `AbstractArray`/`DenseArray`. Drop-in.

"""
    aligned_array(T, dims...; alignment=64) -> Array{T,N}

Allocate an `Array{T,N}` whose data pointer is aligned to `alignment` bytes.
Backed by `posix_memalign`; freed via libc `free` when the array becomes
unreachable.

`alignment` must be a power of two ≥ `sizeof(Ptr{Cvoid})` (per POSIX). Typical
values: 32, 64, 128 — match the `alignment` you pass to
`spmd_function`/`ka_function`.

# Examples
```julia
a = aligned_array(Float32, 1024; alignment=128)
@assert UInt(pointer(a)) % 128 == 0
```
"""
function aligned_array(::Type{T}, dims::Int...; alignment::Int=64) where T
    return aligned_array(T, dims; alignment)
end

function aligned_array(::Type{T}, dims::NTuple{N,Int};
                       alignment::Int=64) where {T,N}
    _check_alignment(alignment, T)
    nbytes = prod(dims) * sizeof(T)
    ptr_ref = Ref{Ptr{Cvoid}}()
    ret = ccall(:posix_memalign, Cint,
                (Ptr{Ptr{Cvoid}}, Csize_t, Csize_t),
                ptr_ref, alignment, nbytes)
    ret == 0 ||
        error("posix_memalign failed (T=$T, dims=$dims, align=$alignment): " *
              "errno=$ret")
    # `own=true` ⇒ Julia frees via libc `free()` when GC collects the array.
    # That matches posix_memalign's free contract.
    return unsafe_wrap(Array, Ptr{T}(ptr_ref[]), dims; own=true)
end

function _check_alignment(alignment::Int, ::Type{T}) where T
    alignment > 0 || error("alignment must be positive, got $alignment")
    ispow2(alignment) || error("alignment must be a power of two, got $alignment")
    # POSIX requires alignment to be a multiple of sizeof(void*).
    if alignment % sizeof(Ptr{Cvoid}) != 0
        error("alignment ($alignment) must be a multiple of sizeof(Ptr{Cvoid}) " *
              "(=$(sizeof(Ptr{Cvoid})))")
    end
    return nothing
end

"""
    pointer_aligned(a, alignment::Int) -> Bool

Check that `pointer(a)` is aligned to `alignment` bytes. Used by the launch
adaptor to verify host buffers match the kernel's `ArraySpec.alignment`
before the ccall.
"""
pointer_aligned(a::AbstractArray, alignment::Int) =
    UInt(pointer(a)) % alignment == 0
