#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
#pragma clang fp contract(off)

struct Linear { uint rows, columns, tokens, input_stride, output_stride, aligned_weights, weight_offset; };
inline float bf16(ushort x) { return as_type<float>(uint(x) << 16); }

// A cache representation, not quantization: all sixteen original BF16 bits
// are recovered before the same FP32 arithmetic as the ordinary kernels.
#define PACK_BLOCK 512u
struct PackBlock { uint offset, exponent; };
inline uint exponent_delta(device const uchar *data, uint i, uint bits) {
    if (!bits) return 0;
    uint bit = i * bits, shift = bit & 31;
    device const uint *words = (device const uint *)(data + PACK_BLOCK);
    uint value = words[bit >> 5] >> shift;
    if (shift + bits > 32) value |= words[(bit >> 5) + 1] << (32 - shift);
    return value & ((1u << bits) - 1);
}
template<bool Packed>
inline ushort weight_bits(device const ushort *w, ulong i, constant Linear &p) {
    if (!Packed) return w[i];
    i += p.weight_offset;
    PackBlock h = ((device const PackBlock *)w)[i / PACK_BLOCK];
    device const uchar *data = (device const uchar *)w + h.offset;
    uint j = uint(i) & (PACK_BLOCK - 1), low = data[j];
    uint exponent = (h.exponent & 255) + exponent_delta(data, j, h.exponent >> 8);
    return ushort((low & 127) | ((low & 128) << 8) | (exponent << 7));
}
template<bool Packed>
inline float4 weight4(device const ushort *w, ulong i, constant Linear &p) {
    if (!Packed) return as_type<float4>(uint4(*((device const ushort4 *)(w + i))) << 16);
    i += p.weight_offset;
    PackBlock h = ((device const PackBlock *)w)[i / PACK_BLOCK];
    device const uchar *data = (device const uchar *)w + h.offset;
    uint j = uint(i) & (PACK_BLOCK - 1), bits = h.exponent >> 8;
    uint4 low = uint4(*((device const uchar4 *)(data + j))), delta;
    if (!bits) delta = 0;
    else {
        // Four adjacent exponent deltas occupy at most 32 bits. Read their
        // complete bit field once, including a possible word boundary.
        uint bit = j * bits, shift = bit & 31;
        device const uint *words = (device const uint *)(data + PACK_BLOCK);
        uint v = words[bit >> 5] >> shift;
        if (shift + 4 * bits > 32) v |= words[(bit >> 5) + 1] << (32 - shift);
        delta = (uint4(v) >> uint4(0, bits, 2 * bits, 3 * bits)) & ((1u << bits) - 1);
    }
    uint4 exponent = delta + (h.exponent & 255);
    return as_type<float4>(((low & 127) | ((low & 128) << 8) | (exponent << 7)) << 16);
}

kernel void plamo_pack_analyze(device const ushort *w [[buffer(0)]], device PackBlock *headers [[buffer(1)]],
                                constant uint &count [[buffer(2)]], uint block [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                uint simd [[simdgroup_index_in_threadgroup]]) {
    uint minimum = 255, maximum = 0;
    for (uint j = tid; j < PACK_BLOCK; j += 128) {
        ulong i = ulong(block) * PACK_BLOCK + j;
        if (i < count) { uint e = (w[i] >> 7) & 255; minimum = min(minimum, e); maximum = max(maximum, e); }
    }
    threadgroup uint minima[4], maxima[4];
    minimum = simd_min(minimum); maximum = simd_max(maximum);
    if (!lane) { minima[simd] = minimum; maxima[simd] = maximum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!tid) {
        minimum = min(min(minima[0], minima[1]), min(minima[2], minima[3]));
        maximum = max(max(maxima[0], maxima[1]), max(maxima[2], maxima[3]));
        uint span = maximum - minimum, bits = span ? 32 - clz(span) : 0;
        headers[block] = {0, minimum | (bits << 8)};
    }
}

kernel void plamo_pack_encode(device const ushort *w [[buffer(0)]], device const PackBlock *headers [[buffer(1)]],
                               device uchar *output [[buffer(2)]], constant uint &count [[buffer(3)]],
                               uint block [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    PackBlock h = headers[block];
    if (!tid) ((device PackBlock *)output)[block] = h;
    device uchar *data = output + h.offset;
    ulong first = ulong(block) * PACK_BLOCK;
    for (uint j = tid; j < PACK_BLOCK; j += 128) {
        ushort v = first + j < count ? w[first + j] : 0;
        data[j] = uchar((v & 127) | ((v >> 8) & 128));
    }
    uint bits = h.exponent >> 8, base = h.exponent & 255;
    for (uint word = tid; word < bits * (PACK_BLOCK / 32); word += 128) {
        uint begin = word * 32, end = begin + 32, value = 0;
        for (uint j = begin / bits; j < min(PACK_BLOCK, (end + bits - 1) / bits); ++j) {
            uint e = first + j < count ? ((w[first + j] >> 7) & 255) - base : 0;
            int shift = int(j * bits) - int(begin);
            value |= shift < 0 ? e >> uint(-shift) : e << uint(shift);
        }
        ((device uint *)(data + PACK_BLOCK))[word] = value;
    }
}

// Make each retained page accessible to the GPU before API readiness. Reading
// one byte per OS page avoids a dummy model inference or a full tensor scan.
kernel void plamo_cache_touch(device const uchar *weights [[buffer(0)]], device uint *sink [[buffer(1)]],
                              constant uint2 &p [[buffer(2)]], uint page [[thread_position_in_grid]]) {
    ulong offset = ulong(page) * p.y;
    if (offset < p.x) sink[page] = weights[offset];
}

// One SIMD group reduces a row. Adjacent lanes read adjacent BF16 values.
template<bool Packed>
inline void matvec(device const ushort *w, device const float *x, device float *y,
                    constant Linear &p, uint3 group, uint simd, uint lane) {
    uint row = group.x * 4 + simd;
    if (row >= p.rows) return;
    float4 sum0 = 0.0f, sum1 = 0.0f;
    for (uint k = lane * 8; k < p.columns; k += 256) {
        if (!(p.columns % 8) && p.aligned_weights) {
            float4 weights = weight4<Packed>(w, ulong(row) * p.columns + k, p);
            float4 input = *((device const float4 *)(x + ulong(group.y) * p.input_stride + k));
            sum0 = fma(weights, input, sum0);
            weights = weight4<Packed>(w, ulong(row) * p.columns + k + 4, p);
            input = *((device const float4 *)(x + ulong(group.y) * p.input_stride + k + 4));
            sum1 = fma(weights, input, sum1);
        } else {
            for (uint j = 0; j < 4; ++j) {
                if (k + j < p.columns) sum0[j] = fma(bf16(weight_bits<Packed>(w, ulong(row) * p.columns + k + j, p)),
                    x[ulong(group.y) * p.input_stride + k + j], sum0[j]);
                if (k + j + 4 < p.columns) sum1[j] = fma(bf16(weight_bits<Packed>(w, ulong(row) * p.columns + k + j + 4, p)),
                    x[ulong(group.y) * p.input_stride + k + j + 4], sum1[j]);
            }
        }
    }
    float4 sum = sum0 + sum1;
    float value = simd_sum((sum.x + sum.y) + (sum.z + sum.w));
    if (!lane) y[ulong(group.y) * p.output_stride + row] = value;
}

// Narrow prompts benefit from independent SIMD reductions without matrix
// tile barriers. Eight tokens reuse each BF16 load; operands stay FP32.
template<bool Packed>
inline void matmul_vector(device const ushort *w, device const float *x, device float *y,
                           constant Linear &p, uint3 group, uint simd, uint lane) {
    uint row = group.x * 4 + simd, first = group.y * 8;
    if (row >= p.rows) return;
    float4 sums[8];
    #pragma unroll
    for (uint t = 0; t < 8; ++t) sums[t] = 0.0f;
    for (uint k = lane * 4; k < p.columns; k += 128) {
        float4 weights = 0.0f;
        if (!(p.columns % 4) && p.aligned_weights) {
            weights = weight4<Packed>(w, ulong(row) * p.columns + k, p);
        } else {
            for (uint j = 0; j < 4; ++j) if (k + j < p.columns) weights[j] = bf16(weight_bits<Packed>(w, ulong(row) * p.columns + k + j, p));
        }
        #pragma unroll
        for (uint t = 0; t < 8; ++t) if (first + t < p.tokens) {
            float4 input = 0.0f;
            if (!(p.columns % 4)) input = *((device const float4 *)(x + ulong(first + t) * p.input_stride + k));
            else for (uint j = 0; j < 4; ++j) if (k + j < p.columns) input[j] = x[ulong(first + t) * p.input_stride + k + j];
            sums[t] = fma(weights, input, sums[t]);
        }
    }
    #pragma unroll
    for (uint t = 0; t < 8; ++t) if (first + t < p.tokens) {
        float value = simd_sum((sums[t].x + sums[t].y) + (sums[t].z + sums[t].w));
        if (!lane) y[ulong(first + t) * p.output_stride + row] = value;
    }
}

// 32 output rows x 8/16/24/32 prompt tokens. BF16 expands exactly to FP32; both
// operands and accumulation remain FP32 in every temporary tile.
template<bool Packed, uint N>
inline void matmul_tile(device const ushort *w, device const float *x, device float *y,
                        constant Linear &p, uint3 group, uint tid, uint simd,
                        threadgroup float *weights, threadgroup float *inputs, threadgroup float *outputs) {
    simdgroup_float8x8 total0(0.0f), total1(0.0f), total2(0.0f), total3(0.0f);
    for (uint base = 0; base < p.columns; base += 32) {
        for (uint i = tid; i < 32 * 8; i += 128) {
            uint row = group.x * 32 + i / 8, col = base + (i % 8) * 4;
            float4 value = 0.0f;
            if (row < p.rows && col + 3 < p.columns && !(p.columns % 4) && p.aligned_weights)
                value = weight4<Packed>(w, ulong(row) * p.columns + col, p);
            else for (uint j = 0; j < 4; ++j)
                if (row < p.rows && col + j < p.columns) value[j] = bf16(weight_bits<Packed>(w, ulong(row) * p.columns + col + j, p));
            for (uint j = 0; j < 4; ++j) weights[i * 4 + j] = value[j];
        }
        for (uint i = tid; i < N * 8; i += 128) {
            uint token = group.y * N + i / 8, col = base + (i % 8) * 4;
            float4 value = 0.0f;
            if (token < p.tokens && col + 3 < p.columns && !(p.input_stride % 4))
                value = *((device const float4 *)(x + ulong(token) * p.input_stride + col));
            else for (uint j = 0; j < 4; ++j)
                if (token < p.tokens && col + j < p.columns) value[j] = x[ulong(token) * p.input_stride + col + j];
            for (uint j = 0; j < 4; ++j) inputs[i * 4 + j] = value[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 32; k += 8) {
            simdgroup_float8x8 a, b0, b1, b2, b3;
            simdgroup_load(a, weights + simd * 8 * 32 + k, 32);
            simdgroup_load(b0, inputs + k, 32, ulong2(0), true);
            if (N > 8) simdgroup_load(b1, inputs + 8 * 32 + k, 32, ulong2(0), true);
            if (N > 16) simdgroup_load(b2, inputs + 16 * 32 + k, 32, ulong2(0), true);
            if (N > 24) simdgroup_load(b3, inputs + 24 * 32 + k, 32, ulong2(0), true);
            simdgroup_multiply_accumulate(total0, a, b0, total0);
            if (N > 8) simdgroup_multiply_accumulate(total1, a, b1, total1);
            if (N > 16) simdgroup_multiply_accumulate(total2, a, b2, total2);
            if (N > 24) simdgroup_multiply_accumulate(total3, a, b3, total3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(total0, outputs + simd * 8 * N, N);
    if (N > 8) simdgroup_store(total1, outputs + simd * 8 * N + 8, N);
    if (N > 16) simdgroup_store(total2, outputs + simd * 8 * N + 16, N);
    if (N > 24) simdgroup_store(total3, outputs + simd * 8 * N + 24, N);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 32 * N; i += 128) {
        uint row = group.x * 32 + i / N, token = group.y * N + i % N;
        if (row < p.rows && token < p.tokens) y[ulong(token) * p.output_stride + row] = outputs[i];
    }
}

#define VECTOR_KERNEL(NAME, BODY, PACKED) \
kernel void NAME(device const ushort *w [[buffer(0)]], device const float *x [[buffer(1)]], \
                 device float *y [[buffer(2)]], constant Linear &p [[buffer(3)]], \
                 uint3 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]], \
                 uint lane [[thread_index_in_simdgroup]]) { BODY<PACKED>(w, x, y, p, group, simd, lane); }
VECTOR_KERNEL(plamo_matvec, matvec, false)
VECTOR_KERNEL(plamo_matvec_packed, matvec, true)
VECTOR_KERNEL(plamo_matmul_vector, matmul_vector, false)
VECTOR_KERNEL(plamo_matmul_vector_packed, matmul_vector, true)

#define MATRIX_KERNEL(NAME, PACKED, N) \
kernel void NAME(device const ushort *w [[buffer(0)]], device const float *x [[buffer(1)]], \
                 device float *y [[buffer(2)]], constant Linear &p [[buffer(3)]], \
                 uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint simd [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float weights[32 * 32], inputs[N * 32]; \
    matmul_tile<PACKED, N>(w, x, y, p, group, tid, simd, weights, inputs, weights); }
MATRIX_KERNEL(plamo_matmul, false, 32)
MATRIX_KERNEL(plamo_matmul8, false, 8)
MATRIX_KERNEL(plamo_matmul16, false, 16)
MATRIX_KERNEL(plamo_matmul_short, false, 24)
MATRIX_KERNEL(plamo_matmul_packed, true, 32)
MATRIX_KERNEL(plamo_matmul8_packed, true, 8)
MATRIX_KERNEL(plamo_matmul16_packed, true, 16)
MATRIX_KERNEL(plamo_matmul_short_packed, true, 24)

// RMSNorm stays entirely FP32; exponentials use the CPU polynomial.
// Pairwise FP32 reduction of nonnegative squares. Each lane's partial and the
// four SIMD-group sums stay FP32; the shallower sum avoids the long serial
// rounding chain as well as thousands of idle threads across decode layers.
inline void rms_inverse(device const float *x, uint hidden, uint tid,
                            threadgroup float *partials, threadgroup float &inverse) {
    float sum = 0.0f;
    for (uint i = tid; i < hidden; i += 128) sum += x[i] * x[i];
    sum = simd_sum(sum);
    if (!(tid & 31)) partials[tid >> 5] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!tid) inverse = 1.0f / sqrt(((partials[0] + partials[1]) + (partials[2] + partials[3])) / float(hidden) + 0.000001f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

struct MLP { uint hidden, intermediate, tokens; };
kernel void plamo_mlp_norm(device const float *x [[buffer(0)]], device const ushort *w [[buffer(1)]],
                            device float *y [[buffer(2)]], constant MLP &p [[buffer(3)]],
                            uint token [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float partials[4], inverse;
    ulong base = ulong(token) * p.hidden;
    rms_inverse(x + base, p.hidden, tid, partials, inverse);
    for (uint i = tid; i < p.hidden; i += 128) y[base + i] = (x[base + i] * inverse) * (bf16(w[i]) + 1.0f);
}

inline float plamo_exp(float x) {
    if (x < -87.0f) return 0.0f;
    if (x > 88.0f) return INFINITY;
    float n = rint(x * 1.4426950408889634f);
    float r = fma(-n, 0.6931471805599453f, x);
    float y = fma(0.0001984126984126984f, r, 0.001388888888888889f);
    y = fma(y, r, 0.008333333333333333f);
    y = fma(y, r, 0.041666666666666664f);
    y = fma(y, r, 0.16666666666666666f);
    y = fma(y, r, 0.5f);
    y = fma(y, r, 1.0f);
    y = fma(y, r, 1.0f);
    return y * as_type<float>(uint(int(n) + 127) << 23);
}

kernel void plamo_mlp_swiglu(device float *x [[buffer(0)]], constant MLP &p [[buffer(1)]],
                             uint2 pos [[thread_position_in_grid]]) {
    if (pos.x >= p.intermediate || pos.y >= p.tokens) return;
    ulong i = ulong(pos.y) * 2 * p.intermediate + pos.x;
    x[i] = (x[i] / (1.0f + plamo_exp(-x[i]))) * x[i + p.intermediate];
}

kernel void plamo_mlp_residual(device float *state [[buffer(0)]], device const ushort *w [[buffer(1)]],
                                device const float *branch [[buffer(2)]], constant MLP &p [[buffer(3)]],
                                uint token [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float partials[4], inverse;
    ulong base = ulong(token) * p.hidden;
    rms_inverse(branch + base, p.hidden, tid, partials, inverse);
    for (uint i = tid; i < p.hidden; i += 128)
        state[base + i] += (branch[base + i] * inverse) * (bf16(w[i]) + 0.08944271909999159f);
}

struct Decode { uint qh, kh, dim, count; };
inline float2 rotate_pair(float2 x, float2 cs) {
    return float2(x.x * cs.x - x.y * cs.y, x.y * cs.x + x.x * cs.y);
}

// Shift a full sliding cache, or rebuild a stale rotated cache. Each thread
// owns one coordinate pair, so the overlapping forward copy has no races.
kernel void plamo_decode_keys(device float *k [[buffer(0)]], device float *v [[buffer(1)]],
                              device float *rotated [[buffer(2)]], device const float *table [[buffer(3)]],
                              constant Decode &p [[buffer(4)]], constant uint &shift [[buffer(5)]],
                              uint pair [[thread_position_in_grid]]) {
    uint half_dim = p.dim / 2, head = pair / half_dim, col = pair % half_dim;
    if (head >= p.kh) return;
    uint width = p.kh * p.dim, component = head * p.dim + col;
    for (uint row = 0; row + 1 < p.count; ++row) {
        ulong source = ulong(row + shift) * width + component, dest = ulong(row) * width + component;
        float2 key = float2(k[source], k[source + half_dim]);
        if (shift) {
            k[dest] = key.x; k[dest + half_dim] = key.y;
            v[dest] = v[source]; v[dest + half_dim] = v[source + half_dim];
        }
        ulong t = ulong(row) * p.dim + col;
        float2 r = rotate_pair(key, float2(table[t], table[t + half_dim]));
        rotated[dest] = r.x; rotated[dest + half_dim] = r.y;
    }
}

// Normalize Q/K with the CPU's sequential FP32 sum, rotate Q and the appended
// K, and append V. CPU and GPU refer to the same original KV allocations.
kernel void plamo_decode_qkv(device float *qkv [[buffer(0)]], device const ushort *qn [[buffer(1)]],
                             device const ushort *kn [[buffer(2)]], device float *k [[buffer(3)]],
                             device float *v [[buffer(4)]], device float *rotated [[buffer(5)]],
                             device const float *table [[buffer(6)]], constant Decode &p [[buffer(7)]],
                             uint head [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float inverse;
    ulong base = ulong(head) * p.dim;
    if (!tid) {
        float sum = 0.0f;
        for (uint i = 0; i < p.dim; ++i) sum += qkv[base + i] * qkv[base + i];
        inverse = 1.0f / sqrt(sum / float(p.dim) + 0.000001f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint half_dim = p.dim / 2;
    device const ushort *w = head < p.qh ? qn : kn;
    for (uint col = tid; col < half_dim; col += 128) {
        float2 raw = float2((qkv[base + col] * inverse) * (bf16(w[col]) + 1.0f),
                            (qkv[base + col + half_dim] * inverse) * (bf16(w[col + half_dim]) + 1.0f));
        ulong t = ulong(p.count - 1) * p.dim + col;
        float2 r = rotate_pair(raw, float2(table[t], table[t + half_dim]));
        if (head < p.qh) { qkv[base + col] = r.x; qkv[base + col + half_dim] = r.y; }
        else {
            ulong dest = (ulong(p.count - 1) * p.kh + head - p.qh) * p.dim + col;
            k[dest] = raw.x; k[dest + half_dim] = raw.y;
            rotated[dest] = r.x; rotated[dest + half_dim] = r.y;
            ulong vi = ulong(p.qh + p.kh + head - p.qh) * p.dim + col;
            v[dest] = qkv[vi]; v[dest + half_dim] = qkv[vi + half_dim];
        }
    }
}

// Parallel heads and score rows, with the original per-dot FMA order,
// sequential softmax denominator and weighted-value accumulation preserved.
kernel void plamo_decode_attention(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
                                   device const float *v [[buffer(2)]], device float *out [[buffer(3)]],
                                   device float *scores [[buffer(4)]], constant Decode &p [[buffer(5)]],
                                   uint head [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float maximum, denominator;
    uint kh = head / (p.qh / p.kh), width = p.kh * p.dim;
    device float *s = scores + ulong(head) * p.count;
    float scale = 1.0f / sqrt(float(p.dim));
    for (uint row = tid; row < p.count; row += 128) {
        float sum = 0.0f;
        for (uint col = 0; col < p.dim; ++col)
            sum = fma(q[ulong(head) * p.dim + col], k[(ulong(row) * p.kh + kh) * p.dim + col], sum);
        s[row] = sum * scale;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (!tid) {
        float m = s[0];
        for (uint row = 1; row < p.count; ++row) m = max(m, s[row]);
        maximum = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = tid; row < p.count; row += 128) s[row] = plamo_exp(s[row] - maximum);
    threadgroup_barrier(mem_flags::mem_device);
    if (!tid) {
        float sum = 0.0f;
        for (uint row = 0; row < p.count; ++row) sum += s[row];
        denominator = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = tid; row < p.count; row += 128) s[row] /= denominator;
    threadgroup_barrier(mem_flags::mem_device);
    for (uint col = tid; col < p.dim; col += 128) {
        float sum = 0.0f;
        for (uint row = 0; row < p.count; ++row)
            sum = fma(s[row], v[ulong(row) * width + kh * p.dim + col], sum);
        out[ulong(head) * p.dim + col] = sum;
    }
}


#define DECODE_RESIDUAL(NAME, OFFSET) \
kernel void NAME(device float *state [[buffer(0)]], device const ushort *w [[buffer(1)]], \
                 device const float *branch [[buffer(2)]], constant MLP &p [[buffer(3)]], \
                 uint token [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup float partials[4], inverse; \
    ulong base = ulong(token) * p.hidden; \
    rms_inverse(branch + base, p.hidden, tid, partials, inverse); \
    for (uint i = tid; i < p.hidden; i += 128) state[base + i] += (branch[base + i] * inverse) * (bf16(w[i]) + OFFSET); }
DECODE_RESIDUAL(plamo_decode_residual, 0.2f)

struct Prefill { uint qh, kh, dim, prefix, tokens, capacity, query_offset; };

// Normalize all prompt Q/K heads in parallel. For a sliding batch that would
// overwrite a prefix still needed by earlier queries, retain normalized K/V
// in QKV until attention has finished. Otherwise use canonical KV directly.
kernel void plamo_prefill_qkv(device float *qkv [[buffer(0)]], device const ushort *qn [[buffer(1)]],
                               device const ushort *kn [[buffer(2)]], device float *k [[buffer(3)]],
                               device float *v [[buffer(4)]], device float *rotated [[buffer(5)]],
                               device const float *table [[buffer(6)]], constant Prefill &p [[buffer(7)]],
                               uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint head = group.x, token = group.y, half_dim = p.dim / 2;
    ulong width = ulong(p.qh + 2 * p.kh) * p.dim, base = ulong(token) * width + ulong(head) * p.dim;
    threadgroup float inverse;
    if (!tid) {
        float sum = 0.0f;
        for (uint col = 0; col < p.dim; ++col) sum += qkv[base + col] * qkv[base + col];
        inverse = 1.0f / sqrt(sum / float(p.dim) + 0.000001f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const ushort *w = head < p.qh ? qn : kn;
    for (uint col = tid; col < half_dim; col += 128) {
        float2 raw = float2((qkv[base + col] * inverse) * (bf16(w[col]) + 1.0f),
                           (qkv[base + col + half_dim] * inverse) * (bf16(w[col + half_dim]) + 1.0f));
        uint position = min(p.prefix + token, p.capacity - 1);
        ulong t = ulong(position) * p.dim + col;
        float2 r = rotate_pair(raw, float2(table[t], table[t + half_dim]));
        if (head < p.qh) {
            qkv[base + col] = r.x; qkv[base + col + half_dim] = r.y;
        } else if (p.prefix + p.tokens <= p.capacity) {
            ulong dest = (ulong(p.prefix + token) * p.kh + head - p.qh) * p.dim + col;
            k[dest] = raw.x; k[dest + half_dim] = raw.y;
            rotated[dest] = r.x; rotated[dest + half_dim] = r.y;
            v[dest] = qkv[base + ulong(p.kh) * p.dim + col];
            v[dest + half_dim] = qkv[base + ulong(p.kh) * p.dim + col + half_dim];
        } else {
            qkv[base + col] = raw.x; qkv[base + col + half_dim] = raw.y;
        }
    }
}

// Each head/query owns one score row. Query tiles reuse bounded score scratch
// without storing a tokens x tokens attention matrix for the whole prompt.
// Keep sequential FP32 dot products, softmax sum and value accumulation.
kernel void plamo_prefill_attention(device const float *qkv [[buffer(0)]], device const float *k [[buffer(1)]],
                                     device const float *v [[buffer(2)]], device const float *rotated [[buffer(3)]],
                                     device const float *table [[buffer(4)]], device float *out [[buffer(5)]],
                                     device float *scores [[buffer(6)]], constant Prefill &p [[buffer(7)]],
                                     uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint head = group.x, token = p.query_offset + group.y, kh = head / (p.qh / p.kh);
    uint count = min(p.prefix + token + 1, p.capacity), first = p.prefix + token + 1 - count;
    uint half_dim = p.dim / 2, kv_width = p.kh * p.dim, width = (p.qh + 2 * p.kh) * p.dim;
    bool sliding = p.prefix + p.tokens > p.capacity;
    device const float *q = qkv + ulong(token) * width + ulong(head) * p.dim;
    device float *s = scores + (ulong(group.y) * p.qh + head) * min(p.prefix + p.tokens, p.capacity);
    threadgroup float maximum, denominator;
    float scale = 1.0f / sqrt(float(p.dim));
    for (uint row = tid; row < count; row += 128) {
        float sum = 0.0f;
        if (!sliding) {
            for (uint col = 0; col < p.dim; ++col)
                sum = fma(q[col], rotated[(ulong(row) * p.kh + kh) * p.dim + col], sum);
        } else {
            uint source = first + row;
            device const float *key = source < p.prefix ? k + ulong(source) * kv_width + kh * p.dim :
                qkv + ulong(source - p.prefix) * width + (p.qh + kh) * p.dim;
            // Sliding RoPE uses local positions after each eviction, matching
            // CPU rebasing even across several window lengths in one batch.
            for (uint col = 0; col < p.dim; ++col) {
                uint pair = col % half_dim;
                ulong t = ulong(row) * p.dim + pair;
                float2 r = rotate_pair(float2(key[pair], key[pair + half_dim]), float2(table[t], table[t + half_dim]));
                sum = fma(q[col], col < half_dim ? r.x : r.y, sum);
            }
        }
        s[row] = sum * scale;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (!tid) {
        float value = s[0];
        for (uint row = 1; row < count; ++row) value = max(value, s[row]);
        maximum = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = tid; row < count; row += 128) s[row] = plamo_exp(s[row] - maximum);
    threadgroup_barrier(mem_flags::mem_device);
    if (!tid) {
        float sum = 0.0f;
        for (uint row = 0; row < count; ++row) sum += s[row];
        denominator = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = tid; row < count; row += 128) s[row] /= denominator;
    threadgroup_barrier(mem_flags::mem_device);
    for (uint col = tid; col < p.dim; col += 128) {
        float sum = 0.0f;
        for (uint row = 0; row < count; ++row) {
            uint source = first + row;
            float value = !sliding || source < p.prefix ? v[ulong(sliding ? source : row) * kv_width + kh * p.dim + col] :
                qkv[ulong(source - p.prefix) * width + (p.qh + p.kh + kh) * p.dim + col];
            sum = fma(s[row], value, sum);
        }
        out[(ulong(token) * p.qh + head) * p.dim + col] = sum;
    }
}

// Commit only the final sliding window after every query has consumed the old
// prefix. A thread owns a coordinate pair and copies rows forward in order.
kernel void plamo_prefill_store(device const float *qkv [[buffer(0)]], device float *k [[buffer(1)]],
                                 device float *v [[buffer(2)]], device float *rotated [[buffer(3)]],
                                 device const float *table [[buffer(4)]], constant Prefill &p [[buffer(5)]],
                                 uint pair [[thread_position_in_grid]]) {
    uint half_dim = p.dim / 2, head = pair / half_dim, col = pair % half_dim;
    if (head >= p.kh) return;
    uint count = min(p.prefix + p.tokens, p.capacity), first = p.prefix + p.tokens - count;
    uint width = (p.qh + 2 * p.kh) * p.dim;
    for (uint row = 0; row < count; ++row) {
        uint source = first + row;
        ulong dest = (ulong(row) * p.kh + head) * p.dim + col;
        device const float *key = source < p.prefix ? k + (ulong(source) * p.kh + head) * p.dim :
            qkv + ulong(source - p.prefix) * width + (p.qh + head) * p.dim;
        device const float *value = source < p.prefix ? v + (ulong(source) * p.kh + head) * p.dim :
            qkv + ulong(source - p.prefix) * width + (p.qh + p.kh + head) * p.dim;
        float2 raw = float2(key[col], key[col + half_dim]), val = float2(value[col], value[col + half_dim]);
        ulong t = ulong(row) * p.dim + col;
        float2 r = rotate_pair(raw, float2(table[t], table[t + half_dim]));
        k[dest] = raw.x; k[dest + half_dim] = raw.y;
        v[dest] = val.x; v[dest + half_dim] = val.y;
        rotated[dest] = r.x; rotated[dest + half_dim] = r.y;
    }
}
