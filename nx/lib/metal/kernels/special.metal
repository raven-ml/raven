#include <metal_stdlib>
using namespace metal;

// Where (conditional select) operation
kernel void where_float(device float* out [[buffer(0)]],
                       device const uchar* cond [[buffer(1)]],
                       device const float* if_true [[buffer(2)]],
                       device const float* if_false [[buffer(3)]],
                       constant uint& size [[buffer(4)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = cond[gid] ? if_true[gid] : if_false[gid];
}


kernel void where_int(device int* out [[buffer(0)]],
                     device const uchar* cond [[buffer(1)]],
                     device const int* if_true [[buffer(2)]],
                     device const int* if_false [[buffer(3)]],
                     constant uint& size [[buffer(4)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = cond[gid] ? if_true[gid] : if_false[gid];
}

kernel void where_long(device long* out [[buffer(0)]],
                      device const uchar* cond [[buffer(1)]],
                      device const long* if_true [[buffer(2)]],
                      device const long* if_false [[buffer(3)]],
                      constant uint& size [[buffer(4)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = cond[gid] ? if_true[gid] : if_false[gid];
}

kernel void where_uchar(device uchar* out [[buffer(0)]],
                       device const uchar* cond [[buffer(1)]],
                       device const uchar* if_true [[buffer(2)]],
                       device const uchar* if_false [[buffer(3)]],
                       constant uint& size [[buffer(4)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = cond[gid] ? if_true[gid] : if_false[gid];
}

// Type casting kernels

kernel void cast_float_to_int(device int* out [[buffer(0)]],
                             device const float* in [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = int(in[gid]);
}

kernel void cast_float_to_long(device long* out [[buffer(0)]],
                              device const float* in [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = long(in[gid]);
}



kernel void cast_int_to_float(device float* out [[buffer(0)]],
                             device const int* in [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = float(in[gid]);
}


kernel void cast_int_to_long(device long* out [[buffer(0)]],
                            device const int* in [[buffer(1)]],
                            constant uint& size [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = long(in[gid]);
}

kernel void cast_long_to_int(device int* out [[buffer(0)]],
                            device const long* in [[buffer(1)]],
                            constant uint& size [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = int(in[gid]);
}

kernel void cast_long_to_float(device float* out [[buffer(0)]],
                              device const long* in [[buffer(1)]],
                              constant uint& size [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = float(in[gid]);
}

kernel void cast_uchar_to_int(device int* out [[buffer(0)]],
                             device const uchar* in [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = int(in[gid]);
}

kernel void cast_int_to_uchar(device uchar* out [[buffer(0)]],
                             device const int* in [[buffer(1)]],
                             constant uint& size [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = uchar(in[gid] != 0 ? 1 : 0);
}

// Fill kernel - assigns a constant value
kernel void fill_float(device float* out [[buffer(0)]],
                      constant float& value [[buffer(1)]],
                      constant uint& size [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = value;
}


kernel void fill_int(device int* out [[buffer(0)]],
                    constant int& value [[buffer(1)]],
                    constant uint& size [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = value;
}

kernel void fill_long(device long* out [[buffer(0)]],
                     constant long& value [[buffer(1)]],
                     constant uint& size [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = value;
}

kernel void fill_uchar(device uchar* out [[buffer(0)]],
                      constant uchar& value [[buffer(1)]],
                      constant uint& size [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    out[gid] = value;
}

// Gather operation
kernel void gather_float(device float* out [[buffer(0)]],
                        device const float* data [[buffer(1)]],
                        device const int* indices [[buffer(2)]],
                        constant uint& axis_size [[buffer(3)]],
                        constant uint& inner_size [[buffer(4)]],
                        constant uint& indices_size [[buffer(5)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= indices_size * inner_size) return;
    
    uint idx_pos = gid / inner_size;
    uint inner_pos = gid % inner_size;
    
    int idx = indices[idx_pos];
    if (idx < 0) idx += axis_size;
    
    uint data_idx = uint(idx) * inner_size + inner_pos;
    out[gid] = data[data_idx];
}

kernel void gather_int(device int* out [[buffer(0)]],
                      device const int* data [[buffer(1)]],
                      device const int* indices [[buffer(2)]],
                      constant uint& axis_size [[buffer(3)]],
                      constant uint& inner_size [[buffer(4)]],
                      constant uint& indices_size [[buffer(5)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= indices_size * inner_size) return;
    
    uint idx_pos = gid / inner_size;
    uint inner_pos = gid % inner_size;
    
    int idx = indices[idx_pos];
    if (idx < 0) idx += axis_size;
    
    uint data_idx = uint(idx) * inner_size + inner_pos;
    out[gid] = data[data_idx];
}

// Scatter operation
kernel void scatter_float(device float* out [[buffer(0)]],
                         device const int* indices [[buffer(1)]],
                         device const float* updates [[buffer(2)]],
                         constant uint& axis_size [[buffer(3)]],
                         constant uint& inner_size [[buffer(4)]],
                         constant uint& indices_size [[buffer(5)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= indices_size * inner_size) return;
    
    uint idx_pos = gid / inner_size;
    uint inner_pos = gid % inner_size;
    
    int idx = indices[idx_pos];
    if (idx < 0) idx += axis_size;
    
    uint out_idx = uint(idx) * inner_size + inner_pos;
    out[out_idx] = updates[gid];
}

kernel void scatter_int(device int* out [[buffer(0)]],
                       device const int* indices [[buffer(1)]],
                       device const int* updates [[buffer(2)]],
                       constant uint& axis_size [[buffer(3)]],
                       constant uint& inner_size [[buffer(4)]],
                       constant uint& indices_size [[buffer(5)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= indices_size * inner_size) return;
    
    uint idx_pos = gid / inner_size;
    uint inner_pos = gid % inner_size;
    
    int idx = indices[idx_pos];
    if (idx < 0) idx += axis_size;
    
    uint out_idx = uint(idx) * inner_size + inner_pos;
    out[out_idx] = updates[gid];
}

kernel void scatter_uchar(device uchar* out [[buffer(0)]],
                         device const int* indices [[buffer(1)]],
                         device const uchar* updates [[buffer(2)]],
                         constant uint& axis_size [[buffer(3)]],
                         constant uint& inner_size [[buffer(4)]],
                         constant uint& indices_size [[buffer(5)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= indices_size * inner_size) return;
    
    uint idx_pos = gid / inner_size;
    uint inner_pos = gid % inner_size;
    
    int idx = indices[idx_pos];
    if (idx < 0) idx += axis_size;
    
    uint out_idx = uint(idx) * inner_size + inner_pos;
    out[out_idx] = updates[gid];
}

// Threefry random number generator (simplified version)
kernel void threefry_int32(device int* out [[buffer(0)]],
                          device const int* key [[buffer(1)]],
                          device const int* counter [[buffer(2)]],
                          constant uint& size [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= size) return;
    
    // Simplified Threefry implementation
    // In practice, you'd want a full implementation
    uint k0 = uint(key[gid % 2]);
    uint k1 = uint(key[(gid + 1) % 2]);
    uint c0 = uint(counter[gid % 2]);
    uint c1 = uint(counter[(gid + 1) % 2]);
    
    // Mix operations (simplified)
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
