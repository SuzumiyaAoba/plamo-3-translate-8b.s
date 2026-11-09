// Benchmark-only layouts. Logical tensors and BF16 bits remain unchanged.
// BEGIN_LAYOUT_READERS
template<uint R, uint K>
inline ulong layout_index(ulong i, uint columns) {
    if (!R) return i;
    ulong row = i / columns;
    uint col = uint(i % columns);
    uint rr = R ? R : 1, kk = K ? K : 1;
    return (row / rr) * (ulong(rr) * columns) + ulong(col / kk) * (rr * kk) + (row % rr) * kk + col % kk;
}
template<bool Packed, uint R, uint K>
inline ushort layout_weight_bits(device const ushort *w, ulong i, constant Linear &p) {
    return weight_bits<Packed>(w, layout_index<R, K>(i, p.columns), p);
}
template<uint R, uint K>
inline ulong layout_matrix_index(uint row, uint col, uint columns) {
    if (!R) return ulong(row) * columns + col;
    uint rr = R ? R : 1, kk = K ? K : 1;
    return (ulong(row / rr) * rr) * columns + ulong(col / kk) * (rr * kk) + (row % rr) * kk + col % kk;
}
template<bool Packed, uint R, uint K>
inline float4 layout_matrix_weight4(device const ushort *w, uint row, uint col, ulong cursor, constant Linear &p) {
    return weight4<Packed>(w, R ? cursor : ulong(row) * p.columns + col, p);
}
template<bool Packed, uint R, uint K>
inline ushort layout_matrix_bits(device const ushort *w, uint row, uint col, constant Linear &p) {
    return weight_bits<Packed>(w, layout_matrix_index<R, K>(row, col, p.columns), p);
}
// END_LAYOUT_READERS

// Each lane copies four BF16 values. Output writes are contiguous; the
// inverse permutation selects original rows without a second model copy.
kernel void layout_reorder(device const ushort *source [[buffer(0)]], device ushort *out [[buffer(1)]],
                           constant uint4 &p [[buffer(2)]], uint vector [[thread_position_in_grid]]) {
    ulong i = ulong(vector) * 4;
    if (i >= ulong(p.x) * p.y) return;
    uint R = p.z, K = p.w;
    ulong group = i / (ulong(R) * p.y), local = i % (ulong(R) * p.y);
    uint col_group = uint(local / (R * K)), within = uint(local % (R * K));
    ulong row = group * R + within / K;
    uint col = col_group * K + within % K;
    *((device ushort4 *)(out + i)) = *((device const ushort4 *)(source + row * p.y + col));
}

// A warp reads adjacent physical weights across several rows. Save partials
// in original lane order, then use the original SIMD reduction for bit parity.
template<bool Packed, uint R, uint K>
inline void layout_matvec(device const ushort *w, device const float *x, device float *y,
                          constant Linear &p, uint3 group, uint simd, uint lane,
                          threadgroup float *partials) {
    uint row = R == 4 ? lane % 4 : lane / 4;
    uint original_lane = R == 4 ? simd * 8 + lane / 4 : simd * 4 + lane % 4;
    ulong base = ulong(group.x) * R * p.columns;
    float4 sum0 = 0.0f, sum1 = 0.0f;
    for (uint k = original_lane * 8; k < p.columns; k += 256) {
        ulong i = base + ulong(k / K) * (R * K) + row * K + k % K;
        float4 a = weight4<Packed>(w, i, p), b = weight4<Packed>(w, i + 4, p);
        device const float *input = x + ulong(group.y) * p.input_stride + k;
        sum0 = fma(a, *((device const float4 *)input), sum0);
        sum1 = fma(b, *((device const float4 *)(input + 4)), sum1);
    }
    float4 sum = sum0 + sum1;
    partials[row * 32 + original_lane] = (sum.x + sum.y) + (sum.z + sum.w);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float value = simd_sum(partials[simd * 32 + lane]);
    if (!lane) y[ulong(group.y) * p.output_stride + group.x * R + simd] = value;
}

#define LAYOUT_VECTOR(NAME, PACKED, R, K) \
kernel void NAME(device const ushort *w [[buffer(0)]], device const float *x [[buffer(1)]], \
                 device float *y [[buffer(2)]], constant Linear &p [[buffer(3)]], \
                 uint3 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]], \
                 uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float partials[R * 32]; \
    layout_matvec<PACKED, R, K>(w, x, y, p, group, simd, lane, partials); }
LAYOUT_VECTOR(layout_vec_4x8, false, 4, 8)
LAYOUT_VECTOR(layout_vec_4x8_packed, true, 4, 8)
LAYOUT_VECTOR(layout_vec_8x32, false, 8, 32)
LAYOUT_VECTOR(layout_vec_8x32_packed, true, 8, 32)

// Also test keeping the original one-row-per-SIMD-group schedule. Adjacent
// row groups can share cache lines without the shared-memory transpose.
template<bool Packed, uint R, uint K>
inline void layout_simple_matvec(device const ushort *w, device const float *x, device float *y,
                                 constant Linear &p, uint3 group, uint simd, uint lane) {
    uint row = group.x * 4 + simd;
    if (row >= p.rows) return;
    float4 sum0 = 0.0f, sum1 = 0.0f;
    for (uint k = lane * 8; k < p.columns; k += 256) {
        ulong i = layout_matrix_index<R, K>(row, k, p.columns);
        device const float *input = x + ulong(group.y) * p.input_stride + k;
        sum0 = fma(weight4<Packed>(w, i, p), *((device const float4 *)input), sum0);
        sum1 = fma(weight4<Packed>(w, i + 4, p), *((device const float4 *)(input + 4)), sum1);
    }
    float4 sum = sum0 + sum1;
    float value = simd_sum((sum.x + sum.y) + (sum.z + sum.w));
    if (!lane) y[ulong(group.y) * p.output_stride + row] = value;
}
#define LAYOUT_SIMPLE_VECTOR(NAME, PACKED, R, K) \
kernel void NAME(device const ushort *w [[buffer(0)]], device const float *x [[buffer(1)]], \
                 device float *y [[buffer(2)]], constant Linear &p [[buffer(3)]], \
                 uint3 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]], \
                 uint lane [[thread_index_in_simdgroup]]) { \
    layout_simple_matvec<PACKED, R, K>(w, x, y, p, group, simd, lane); }
LAYOUT_SIMPLE_VECTOR(layout_simple_vec_4x8, false, 4, 8)
LAYOUT_SIMPLE_VECTOR(layout_simple_vec_4x8_packed, true, 4, 8)
LAYOUT_SIMPLE_VECTOR(layout_simple_vec_8x32, false, 8, 32)
LAYOUT_SIMPLE_VECTOR(layout_simple_vec_8x32_packed, true, 8, 32)
LAYOUT_SIMPLE_VECTOR(layout_simple_vec_32x32, false, 32, 32)
LAYOUT_SIMPLE_VECTOR(layout_simple_vec_32x32_packed, true, 32, 32)

#define LAYOUT_MATRIX(NAME, PACKED, N, R, K) \
kernel void NAME(device const ushort *w [[buffer(0)]], device const float *x [[buffer(1)]], \
                 device float *y [[buffer(2)]], constant Linear &p [[buffer(3)]], \
                 uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint simd [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float weights[32 * 32], inputs[N * 32]; \
    matmul_tile<PACKED, N, R, K>(w, x, y, p, group, tid, simd, weights, inputs, weights); }
#define LAYOUT_MATRICES(SUFFIX, PACKED, R, K) \
LAYOUT_MATRIX(layout_mm8_##SUFFIX, PACKED, 8, R, K) \
LAYOUT_MATRIX(layout_mm16_##SUFFIX, PACKED, 16, R, K) \
LAYOUT_MATRIX(layout_mm24_##SUFFIX, PACKED, 24, R, K) \
LAYOUT_MATRIX(layout_mm32_##SUFFIX, PACKED, 32, R, K)
LAYOUT_MATRICES(4x8, false, 4, 8)
LAYOUT_MATRICES(4x8_packed, true, 4, 8)
LAYOUT_MATRICES(8x32, false, 8, 32)
LAYOUT_MATRICES(8x32_packed, true, 8, 32)
LAYOUT_MATRICES(32x32, false, 32, 32)
LAYOUT_MATRICES(32x32_packed, true, 32, 32)

#define LAYOUT_UNPACK(NAME, PACKED, R, K) \
kernel void NAME(device const ushort *w [[buffer(0)]], device ushort *out [[buffer(1)]], \
                 constant Linear &p [[buffer(2)]], uint i [[thread_position_in_grid]]) { \
    if (ulong(i) < ulong(p.rows) * p.columns) out[i] = layout_weight_bits<PACKED, R, K>(w, i, p); }
LAYOUT_UNPACK(layout_unpack_row, false, 0, 0)
LAYOUT_UNPACK(layout_unpack_row_packed, true, 0, 0)
LAYOUT_UNPACK(layout_unpack_4x8, false, 4, 8)
LAYOUT_UNPACK(layout_unpack_4x8_packed, true, 4, 8)
LAYOUT_UNPACK(layout_unpack_8x32, false, 8, 32)
LAYOUT_UNPACK(layout_unpack_8x32_packed, true, 8, 32)
LAYOUT_UNPACK(layout_unpack_32x32, false, 32, 32)
LAYOUT_UNPACK(layout_unpack_32x32_packed, true, 32, 32)
