#include <metal_stdlib>
using namespace metal;

// Helper for strided indexing - compute linear index from position and strides
inline uint compute_strided_index(uint3 pos, constant uint* shape, constant int* strides, uint ndim) {
    uint idx = 0;
    for (uint i = 0; i < ndim; i++) {
        uint coord = i == 0 ? pos.x : (i == 1 ? pos.y : pos.z);
        if (i < ndim) {
            coord = coord % shape[i];
            idx += coord * strides[i];
        }
    }
    return idx;
}

// ===== WHERE OPERATIONS =====
#define DEFINE_WHERE(type) \
kernel void where_##type(device type* out [[buffer(0)]], \
                       device const uchar* cond [[buffer(1)]], \
                       device const type* if_true [[buffer(2)]], \
                       device const type* if_false [[buffer(3)]], \
                       constant uint* shape [[buffer(4)]], \
                       constant int* cond_strides [[buffer(5)]], \
                       constant int* true_strides [[buffer(6)]], \
                       constant int* false_strides [[buffer(7)]], \
                       constant uint& ndim [[buffer(8)]], \
                       constant uint& cond_offset [[buffer(9)]], \
                       constant uint& true_offset [[buffer(10)]], \
                       constant uint& false_offset [[buffer(11)]], \
                       uint gid [[thread_position_in_grid]]) { \
    uint total_size = ndim == 0 ? 1 : shape[0] * (ndim > 1 ? shape[1] : 1) * (ndim > 2 ? shape[2] : 1); \
    if (gid >= total_size) return; \
    \
    uint3 pos; \
    uint temp = gid; \
    if (ndim > 2) { \
        pos.z = temp % shape[2]; \
        temp /= shape[2]; \
    } else { \
        pos.z = 0; \
    } \
    if (ndim > 1) { \
        pos.y = temp % shape[1]; \
        temp /= shape[1]; \
    } else { \
        pos.y = 0; \
    } \
    pos.x = temp; \
    \
    uint cond_idx = cond_offset + compute_strided_index(pos, shape, cond_strides, ndim); \
    uint true_idx = true_offset + compute_strided_index(pos, shape, true_strides, ndim); \
    uint false_idx = false_offset + compute_strided_index(pos, shape, false_strides, ndim); \
    \
    out[gid] = cond[cond_idx] ? if_true[true_idx] : if_false[false_idx]; \
}

// Alternative WHERE for higher dimensions (>3)
#define DEFINE_WHERE_ND(type) \
kernel void where_##type##_nd(device type* out [[buffer(0)]], \
                            device const uchar* condition [[buffer(1)]], \
                            device const type* true_val [[buffer(2)]], \
                            device const type* false_val [[buffer(3)]], \
                            constant uint* shape [[buffer(4)]], \
                            constant int* true_strides [[buffer(5)]], \
                            constant int* false_strides [[buffer(6)]], \
                            constant uint& ndim [[buffer(7)]], \
                            constant uint& size [[buffer(8)]], \
                            constant uint& true_offset [[buffer(9)]], \
                            constant uint& false_offset [[buffer(10)]], \
                            uint gid [[thread_position_in_grid]]) { \
    if (gid >= size) return; \
    \
    uint coords[8]; \
    uint temp = gid; \
    \
    for (int i = ndim - 1; i >= 0; i--) { \
        coords[i] = temp % shape[i]; \
        temp /= shape[i]; \
    } \
    \
    uint true_idx = true_offset; \
    uint false_idx = false_offset; \
    for (uint i = 0; i < ndim; i++) { \
        true_idx += coords[i] * true_strides[i]; \
        false_idx += coords[i] * false_strides[i]; \
    } \
    \
    out[gid] = condition[gid] ? true_val[true_idx] : false_val[false_idx]; \
}

// ===== CAST OPERATIONS =====
#define DEFINE_CAST(from_type, to_type) \
kernel void cast_##from_type##_to_##to_type(device to_type* out [[buffer(0)]], \
                                           device const from_type* in [[buffer(1)]], \
                                           constant uint& size [[buffer(2)]], \
                                           uint gid [[thread_position_in_grid]]) { \
    if (gid >= size) return; \
    out[gid] = to_type(in[gid]); \
}

// Special case for bool (uchar) casts
#define DEFINE_CAST_TO_BOOL(from_type) \
kernel void cast_##from_type##_to_uchar(device uchar* out [[buffer(0)]], \
                                       device const from_type* in [[buffer(1)]], \
                                       constant uint& size [[buffer(2)]], \
                                       uint gid [[thread_position_in_grid]]) { \
    if (gid >= size) return; \
    out[gid] = uchar(in[gid] != 0 ? 1 : 0); \
}

// ===== FILL OPERATIONS =====
#define DEFINE_FILL(type) \
kernel void fill_##type(device type* out [[buffer(0)]], \
                       constant type& value [[buffer(1)]], \
                       constant uint& size [[buffer(2)]], \
                       uint gid [[thread_position_in_grid]]) { \
    if (gid >= size) return; \
    out[gid] = value; \
}

// ===== GATHER OPERATIONS =====
#define DEFINE_GATHER(type) \
kernel void gather_##type(device type* out [[buffer(0)]], \
                         device const type* data [[buffer(1)]], \
                         device const int* indices [[buffer(2)]], \
                         constant uint& axis_size [[buffer(3)]], \
                         constant uint& inner_size [[buffer(4)]], \
                         constant uint& indices_size [[buffer(5)]], \
                         constant uint& outer_size [[buffer(6)]], \
                         uint gid [[thread_position_in_grid]]) { \
    if (gid >= outer_size * indices_size * inner_size) return; \
    \
    uint temp = gid; \
    uint inner_pos = temp % inner_size; \
    temp /= inner_size; \
    uint idx_pos = temp % indices_size; \
    uint outer_pos = temp / indices_size; \
    \
    int idx = indices[idx_pos]; \
    if (idx < 0) idx += axis_size; \
    \
    uint data_idx = outer_pos * axis_size * inner_size + uint(idx) * inner_size + inner_pos; \
    out[gid] = data[data_idx]; \
}

// ===== SCATTER OPERATIONS =====
#define DEFINE_SCATTER(type) \
kernel void scatter_##type(device type* out [[buffer(0)]], \
                          device const int* indices [[buffer(1)]], \
                          device const type* updates [[buffer(2)]], \
                          constant uint& axis_size [[buffer(3)]], \
                          constant uint& inner_size [[buffer(4)]], \
                          constant uint& indices_size [[buffer(5)]], \
                          uint gid [[thread_position_in_grid]]) { \
    if (gid >= indices_size * inner_size) return; \
    \
    uint idx_pos = gid / inner_size; \
    uint inner_pos = gid % inner_size; \
    \
    int idx = indices[idx_pos]; \
    if (idx < 0) idx += axis_size; \
    \
    uint out_idx = uint(idx) * inner_size + inner_pos; \
    out[out_idx] = updates[gid]; \
}

#define DEFINE_SCATTER_ADD(type) \
kernel void scatter_add_##type(device type* out [[buffer(0)]], \
                              device const int* indices [[buffer(1)]], \
                              device const type* updates [[buffer(2)]], \
                              constant uint& axis_size [[buffer(3)]], \
                              constant uint& inner_size [[buffer(4)]], \
                              constant uint& indices_size [[buffer(5)]], \
                              uint gid [[thread_position_in_grid]]) { \
    if (gid >= indices_size * inner_size) return; \
    \
    uint idx_pos = gid / inner_size; \
    uint inner_pos = gid % inner_size; \
    \
    int idx = indices[idx_pos]; \
    if (idx < 0) idx += axis_size; \
    \
    uint out_idx = uint(idx) * inner_size + inner_pos; \
    out[out_idx] += updates[gid]; \
}

// Instantiate WHERE operations
DEFINE_WHERE(float)
DEFINE_WHERE(half)
DEFINE_WHERE(int)
DEFINE_WHERE(long)
DEFINE_WHERE(char)
DEFINE_WHERE(uchar)
DEFINE_WHERE(short)
DEFINE_WHERE(ushort)

DEFINE_WHERE_ND(float)
DEFINE_WHERE_ND(half)
DEFINE_WHERE_ND(int)
DEFINE_WHERE_ND(long)
DEFINE_WHERE_ND(char)
DEFINE_WHERE_ND(uchar)
DEFINE_WHERE_ND(short)
DEFINE_WHERE_ND(ushort)

// Instantiate all CAST combinations
// From float
DEFINE_CAST(float, float)
DEFINE_CAST(float, half)
DEFINE_CAST(float, int)
DEFINE_CAST(float, long)
DEFINE_CAST(float, char)
DEFINE_CAST_TO_BOOL(float)
DEFINE_CAST(float, short)
DEFINE_CAST(float, ushort)

// From half
DEFINE_CAST(half, float)
DEFINE_CAST(half, half)
DEFINE_CAST(half, int)
DEFINE_CAST(half, long)
DEFINE_CAST(half, char)
DEFINE_CAST(half, uchar)
DEFINE_CAST(half, short)
DEFINE_CAST(half, ushort)

// From int
DEFINE_CAST(int, float)
DEFINE_CAST(int, half)
DEFINE_CAST(int, int)
DEFINE_CAST(int, long)
DEFINE_CAST(int, char)
DEFINE_CAST_TO_BOOL(int)
DEFINE_CAST(int, short)
DEFINE_CAST(int, ushort)

// From long
DEFINE_CAST(long, float)
DEFINE_CAST(long, half)
DEFINE_CAST(long, int)
DEFINE_CAST(long, long)
DEFINE_CAST(long, char)
DEFINE_CAST(long, uchar)
DEFINE_CAST(long, short)
DEFINE_CAST(long, ushort)

// From char
DEFINE_CAST(char, float)
DEFINE_CAST(char, half)
DEFINE_CAST(char, int)
DEFINE_CAST(char, long)
DEFINE_CAST(char, char)
DEFINE_CAST(char, uchar)
DEFINE_CAST(char, short)
DEFINE_CAST(char, ushort)

// From uchar
DEFINE_CAST(uchar, float)
DEFINE_CAST(uchar, half)
DEFINE_CAST(uchar, int)
DEFINE_CAST(uchar, long)
DEFINE_CAST(uchar, char)
DEFINE_CAST(uchar, uchar)
DEFINE_CAST(uchar, short)
DEFINE_CAST(uchar, ushort)

// From short
DEFINE_CAST(short, float)
DEFINE_CAST(short, half)
DEFINE_CAST(short, int)
DEFINE_CAST(short, long)
DEFINE_CAST(short, char)
DEFINE_CAST(short, uchar)
DEFINE_CAST(short, short)
DEFINE_CAST(short, ushort)

// From ushort
DEFINE_CAST(ushort, float)
DEFINE_CAST(ushort, half)
DEFINE_CAST(ushort, int)
DEFINE_CAST(ushort, long)
DEFINE_CAST(ushort, char)
DEFINE_CAST(ushort, uchar)
DEFINE_CAST(ushort, short)
DEFINE_CAST(ushort, ushort)

// Instantiate FILL operations
DEFINE_FILL(float)
DEFINE_FILL(half)
DEFINE_FILL(int)
DEFINE_FILL(long)
DEFINE_FILL(char)
DEFINE_FILL(uchar)
DEFINE_FILL(short)
DEFINE_FILL(ushort)

// Instantiate GATHER operations
DEFINE_GATHER(float)
DEFINE_GATHER(int)
DEFINE_GATHER(uchar)

// Instantiate SCATTER operations
DEFINE_SCATTER(float)
DEFINE_SCATTER(int)
DEFINE_SCATTER(uchar)

DEFINE_SCATTER_ADD(float)
DEFINE_SCATTER_ADD(int)
DEFINE_SCATTER_ADD(uchar)

// ===== ASSIGN STRIDED OPERATIONS =====
#define DEFINE_ASSIGN_STRIDED(type) \
kernel void assign_strided_##type(device type* dst [[buffer(0)]], \
                                 device const type* src [[buffer(1)]], \
                                 constant uint* shape [[buffer(2)]], \
                                 constant int* src_strides [[buffer(3)]], \
                                 constant uint& ndim [[buffer(4)]], \
                                 constant uint& src_offset [[buffer(5)]], \
                                 uint gid [[thread_position_in_grid]]) { \
    uint total_size = 1; \
    for (uint i = 0; i < ndim; i++) { \
        total_size *= shape[i]; \
    } \
    if (gid >= total_size) return; \
    \
    uint temp = gid; \
    uint src_idx = src_offset; \
    \
    for (int i = int(ndim) - 1; i >= 0; i--) { \
        uint coord = temp % shape[i]; \
        temp /= shape[i]; \
        src_idx += coord * src_strides[i]; \
    } \
    \
    dst[gid] = src[src_idx]; \
}

// Instantiate assign_strided for all types
DEFINE_ASSIGN_STRIDED(float)
DEFINE_ASSIGN_STRIDED(half)
DEFINE_ASSIGN_STRIDED(int)
DEFINE_ASSIGN_STRIDED(long)
DEFINE_ASSIGN_STRIDED(char)
DEFINE_ASSIGN_STRIDED(uchar)
DEFINE_ASSIGN_STRIDED(short)
DEFINE_ASSIGN_STRIDED(ushort)

// ===== GATHER STRIDED OPERATIONS =====
#define DEFINE_GATHER_STRIDED(type) \
kernel void gather_strided_##type(device type* out [[buffer(0)]], \
                                 device const type* data [[buffer(1)]], \
                                 device const int* indices [[buffer(2)]], \
                                 constant uint* out_shape [[buffer(3)]], \
                                 constant uint* data_shape [[buffer(4)]], \
                                 constant int* data_strides [[buffer(5)]], \
                                 constant int* indices_strides [[buffer(6)]], \
                                 constant uint& ndim [[buffer(7)]], \
                                 constant uint& axis [[buffer(8)]], \
                                 constant uint& data_offset [[buffer(9)]], \
                                 constant uint& indices_offset [[buffer(10)]], \
                                 uint gid [[thread_position_in_grid]]) { \
    uint total_size = 1; \
    for (uint i = 0; i < ndim; i++) { \
        total_size *= out_shape[i]; \
    } \
    if (gid >= total_size) return; \
    \
    uint temp = gid; \
    uint data_idx = data_offset; \
    uint indices_idx = indices_offset; \
    \
    for (int i = int(ndim) - 1; i >= 0; i--) { \
        uint coord = temp % out_shape[i]; \
        temp /= out_shape[i]; \
        \
        if (uint(i) == axis) { \
            indices_idx += coord * indices_strides[i]; \
        } else { \
            data_idx += coord * data_strides[i]; \
        } \
    } \
    \
    int idx = indices[indices_idx]; \
    \
    int axis_size = int(data_shape[axis]); \
    if (idx < 0) idx += axis_size; \
    \
    idx = clamp(idx, 0, axis_size - 1); \
    \
    data_idx += uint(idx) * data_strides[axis]; \
    \
    out[gid] = data[data_idx]; \
}

// Instantiate gather_strided for needed types
DEFINE_GATHER_STRIDED(float)
DEFINE_GATHER_STRIDED(int)
DEFINE_GATHER_STRIDED(uchar)

// Simplified Threefry random number generator
kernel void threefry_int32(device int* out [[buffer(0)]],
                          device const int* key [[buffer(1)]],
                          device const int* counter [[buffer(2)]],
                          constant uint& size [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    
    uint k0 = uint(key[gid % 2]);
    uint k1 = uint(key[(gid + 1) % 2]);
    uint c0 = uint(counter[gid % 2]);
    uint c1 = uint(counter[(gid + 1) % 2]);
    
    uint x0 = c0 + k0;
    uint x1 = c1 + k1;
    
    x0 += x1;
    x1 = (x1 << 13) | (x1 >> 19);
    x1 ^= x0;
    
    x0 += x1;
    x1 = (x1 << 17) | (x1 >> 15);
    x1 ^= x0;
    
    out[gid] = int(x0 ^ x1);
}