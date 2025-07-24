#include <metal_stdlib>
using namespace metal;

// Complex number type
struct complex_float {
    float real;
    float imag;
    
    complex_float() : real(0.0f), imag(0.0f) {}
    complex_float(float r, float i) : real(r), imag(i) {}
};

// Compute twiddle factor W_N^k = e^(-2πik/N) for forward FFT
// For inverse FFT, use e^(2πik/N)
inline complex_float twiddle_factor(int k, int N, bool inverse) {
    float angle = (inverse ? 2.0f : -2.0f) * M_PI_F * float(k) / float(N);
    return complex_float(cos(angle), sin(angle));
}

// Bit reversal function
inline uint bit_reverse(uint x, uint bits) {
    uint result = 0;
    for (uint i = 0; i < bits; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// 1D FFT kernel using Cooley-Tukey algorithm
kernel void fft_1d_complex64(device float2* data [[buffer(0)]],
                            constant uint& size [[buffer(1)]],
                            constant uint& stride [[buffer(2)]],
                            constant uint& offset [[buffer(3)]],
                            constant bool& inverse [[buffer(4)]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint tg_size [[threads_per_threadgroup]],
                            uint gid [[threadgroup_position_in_grid]]) {
    // Each threadgroup processes one FFT
    uint fft_offset = offset + gid * size * stride;
    
    // Use float2 array in threadgroup memory instead of complex_float
    threadgroup float2 shared_data[1024]; // Max FFT size
    
    // Load data with bit reversal
    uint num_bits = 0;
    uint temp = size - 1;
    while (temp > 0) {
        num_bits++;
        temp >>= 1;
    }
    
    for (uint i = tid; i < size; i += tg_size) {
        uint j = bit_reverse(i, num_bits);
        uint idx = fft_offset + j * stride;
        shared_data[i] = data[idx];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // FFT computation
    for (uint stage_size = 2; stage_size <= size; stage_size *= 2) {
        uint half_stage = stage_size / 2;
        
        // Each thread handles specific butterfly pairs
        for (uint butterfly_group = tid; butterfly_group < size / 2; butterfly_group += tg_size) {
            uint stage_idx = butterfly_group / half_stage;
            uint butterfly_idx = butterfly_group % half_stage;
            
            uint idx1 = stage_idx * stage_size + butterfly_idx;
            uint idx2 = idx1 + half_stage;
            
            if (idx2 < size) {
                complex_float twiddle = twiddle_factor(butterfly_idx * size / stage_size, size, inverse);
                
                float2 a = shared_data[idx1];
                float2 b = shared_data[idx2];
                
                // Compute b * twiddle
                float2 b_twiddle;
                b_twiddle.x = b.x * twiddle.real - b.y * twiddle.imag;
                b_twiddle.y = b.x * twiddle.imag + b.y * twiddle.real;
                
                // Compute butterfly
                shared_data[idx1] = float2(a.x + b_twiddle.x, a.y + b_twiddle.y);
                shared_data[idx2] = float2(a.x - b_twiddle.x, a.y - b_twiddle.y);
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write back results
    for (uint i = tid; i < size; i += tg_size) {
        uint idx = fft_offset + i * stride;
        data[idx] = shared_data[i];
    }
}

// Multi-dimensional FFT kernel
kernel void fft_multi_complex64(device float2* output [[buffer(0)]],
                               device const float2* input [[buffer(1)]],
                               constant uint* shape [[buffer(2)]],
                               constant int* in_strides [[buffer(3)]],
                               constant int* out_strides [[buffer(4)]],
                               constant uint& ndim [[buffer(5)]],
                               constant int& in_offset [[buffer(6)]],
                               constant int& out_offset [[buffer(7)]],
                               constant uint* axes [[buffer(8)]],
                               constant uint& num_axes [[buffer(9)]],
                               constant bool& inverse [[buffer(10)]],
                               uint3 gid [[thread_position_in_grid]]) {
    // This kernel performs FFT along specified axes
    // For now, we'll implement a simpler version that copies data
    // The actual multi-dimensional FFT will be done by calling 1D FFT multiple times
    
    uint total_size = 1;
    for (uint i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    
    uint idx = gid.x;
    if (idx >= total_size) return;
    
    // Compute multi-dimensional indices
    uint coords[16]; // Max 16 dimensions
    uint temp = idx;
    for (int i = int(ndim) - 1; i >= 0; i--) {
        coords[i] = temp % shape[i];
        temp /= shape[i];
    }
    
    // Compute input and output indices
    int in_idx = in_offset;
    int out_idx = out_offset;
    for (uint i = 0; i < ndim; i++) {
        in_idx += int(coords[i]) * in_strides[i];
        out_idx += int(coords[i]) * out_strides[i];
    }
    
    // Copy data (actual FFT computation will be done by multiple 1D FFT calls)
    output[out_idx] = input[in_idx];
}