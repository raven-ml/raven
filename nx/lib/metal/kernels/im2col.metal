#include <metal_stdlib>
using namespace metal;

// Unfold (im2col) kernel - extracts sliding windows
kernel void unfold_float(device float* out [[buffer(0)]],
                        device const float* in [[buffer(1)]],
                        constant uint* in_shape [[buffer(2)]],         // [batch, channels, *spatial_dims]
                        constant uint* kernel_size [[buffer(3)]],      // [kernel_dims...]
                        constant uint* stride [[buffer(4)]],           // [stride_dims...]
                        constant uint* dilation [[buffer(5)]],         // [dilation_dims...]
                        constant uint* padding [[buffer(6)]],          // [pad_before, pad_after for each dim]
                        constant uint* out_spatial [[buffer(7)]],      // output spatial dimensions
                        constant uint& n_spatial [[buffer(8)]],        // number of spatial dimensions
                        constant uint& channels [[buffer(9)]],
                        constant uint& kernel_elements [[buffer(10)]],
                        constant uint& num_blocks [[buffer(11)]],
                        uint3 gid [[thread_position_in_grid]]) {
    uint b = gid.z;                  // batch index
    uint block_idx = gid.y;          // block index (flattened spatial output position)
    uint col_row = gid.x;            // column row (channel * kernel_elements + kernel_idx)
    
    uint batch_size = in_shape[0];
    if (b >= batch_size || block_idx >= num_blocks || col_row >= channels * kernel_elements) return;
    
    // Decompose col_row into channel and kernel indices
    uint c = col_row / kernel_elements;
    uint kernel_idx = col_row % kernel_elements;
    
    // Convert block index to output spatial coordinates
    uint temp = block_idx;
    uint block_coords[16]; // Max spatial dims
    for (int i = n_spatial - 1; i >= 0; i--) {
        block_coords[i] = temp % out_spatial[i];
        temp /= out_spatial[i];
    }
    
    // Convert kernel index to kernel coordinates
    temp = kernel_idx;
    uint kernel_coords[16];
    for (int i = n_spatial - 1; i >= 0; i--) {
        kernel_coords[i] = temp % kernel_size[i];
        temp /= kernel_size[i];
    }
    
    // Calculate padded input coordinates
    uint padded_coords[16];
    bool in_bounds = true;
    for (uint i = 0; i < n_spatial; i++) {
        int coord = block_coords[i] * stride[i] + kernel_coords[i] * dilation[i];
        if (coord < int(padding[2*i]) || coord >= int(padding[2*i] + in_shape[2+i])) {
            in_bounds = false;
            break;
        }
        padded_coords[i] = coord - padding[2*i];
    }
    
    float value = 0.0;
    if (in_bounds) {
        // Calculate input offset
        uint in_offset = b * in_shape[1];  // batch offset
        in_offset = (in_offset + c);       // channel offset
        
        // Add spatial offsets
        for (uint i = 0; i < n_spatial; i++) {
            in_offset = in_offset * in_shape[2+i] + padded_coords[i];
        }
        
        value = in[in_offset];
    }
    
    // Write to output at [b, col_row, block_idx]
    uint out_offset = b * (channels * kernel_elements * num_blocks) + col_row * num_blocks + block_idx;
    out[out_offset] = value;
}

// Fold (col2im) kernel - combines sliding windows back
kernel void fold_float(device float* out [[buffer(0)]],
                      device const float* in [[buffer(1)]],
                      constant uint* output_size [[buffer(2)]],      // spatial output dimensions
                      constant uint* kernel_size [[buffer(3)]],      // [kernel_dims...]
                      constant uint* stride [[buffer(4)]],           // [stride_dims...]  
                      constant uint* dilation [[buffer(5)]],         // [dilation_dims...]
                      constant uint* padding [[buffer(6)]],          // [pad_before, pad_after for each dim]
                      constant uint* out_spatial [[buffer(7)]],      // output block spatial dimensions
                      constant uint& n_spatial [[buffer(8)]],        // number of spatial dimensions
                      constant uint& batch_size [[buffer(9)]],
                      constant uint& channels [[buffer(10)]],
                      constant uint& kernel_elements [[buffer(11)]],
                      constant uint& num_blocks [[buffer(12)]],
                      uint3 gid [[thread_position_in_grid]]) {
    uint b = gid.z;
    uint c = gid.y;
    uint spatial_idx = gid.x;
    
    if (b >= batch_size || c >= channels) return;
    
    // Calculate padded output size
    uint padded_size[16];
    uint padded_numel = 1;
    for (uint i = 0; i < n_spatial; i++) {
        padded_size[i] = output_size[i] + padding[2*i] + padding[2*i+1];
        padded_numel *= padded_size[i];
    }
    
    if (spatial_idx >= padded_numel) return;
    
    // Convert spatial index to coordinates
    uint temp = spatial_idx;
    uint out_coords[16];
    for (int i = n_spatial - 1; i >= 0; i--) {
        out_coords[i] = temp % padded_size[i];
        temp /= padded_size[i];
    }
    
    // Accumulate contributions from all overlapping blocks
    float sum = 0.0;
    
    // Iterate over all possible blocks that could contribute to this output position
    for (uint block_idx = 0; block_idx < num_blocks; block_idx++) {
        // Convert block index to block coordinates
        temp = block_idx;
        uint block_coords[16];
        for (int i = n_spatial - 1; i >= 0; i--) {
            block_coords[i] = temp % out_spatial[i];
            temp /= out_spatial[i];
        }
        
        // Check if this block contributes to our output position
        bool contributes = true;
        uint kernel_idx = 0;
        uint kernel_multiplier = 1;
        
        for (int i = n_spatial - 1; i >= 0; i--) {
            int block_start = block_coords[i] * stride[i];
            int rel_pos = int(out_coords[i]) - block_start;
            
            // Check if position is within this block's receptive field
            if (rel_pos < 0 || rel_pos % dilation[i] != 0) {
                contributes = false;
                break;
            }
            
            uint kernel_coord = rel_pos / dilation[i];
            if (kernel_coord >= kernel_size[i]) {
                contributes = false;
                break;
            }
            
            kernel_idx += kernel_coord * kernel_multiplier;
            kernel_multiplier *= kernel_size[i];
        }
        
        if (contributes) {
            // Read from input at [b, c * kernel_elements + kernel_idx, block_idx]
            uint input_row = c * kernel_elements + kernel_idx;
            uint in_offset = b * (channels * kernel_elements * num_blocks) + 
                           input_row * num_blocks + block_idx;
            sum += in[in_offset];
        }
    }
    
    // Write accumulated sum to output
    uint out_offset = b * channels;
    for (uint i = 0; i < n_spatial; i++) {
        out_offset = out_offset * padded_size[i] + out_coords[i];
    }
    out[out_offset] = sum;
}

// Int versions of unfold/fold
kernel void unfold_int(device int* out [[buffer(0)]],
                      device const int* in [[buffer(1)]],
                      constant uint* in_shape [[buffer(2)]],
                      constant uint* kernel_size [[buffer(3)]],
                      constant uint* stride [[buffer(4)]],
                      constant uint* dilation [[buffer(5)]],
                      constant uint* padding [[buffer(6)]],
                      constant uint* out_spatial [[buffer(7)]],
                      constant uint& n_spatial [[buffer(8)]],
                      constant uint& channels [[buffer(9)]],
                      constant uint& kernel_elements [[buffer(10)]],
                      constant uint& num_blocks [[buffer(11)]],
                      uint3 gid [[thread_position_in_grid]]) {
    uint b = gid.z;
    uint block_idx = gid.y;
    uint col_row = gid.x;
    
    uint batch_size = in_shape[0];
    if (b >= batch_size || block_idx >= num_blocks || col_row >= channels * kernel_elements) return;
    
    uint c = col_row / kernel_elements;
    uint kernel_idx = col_row % kernel_elements;
    
    uint temp = block_idx;
    uint block_coords[16];
    for (int i = n_spatial - 1; i >= 0; i--) {
        block_coords[i] = temp % out_spatial[i];
        temp /= out_spatial[i];
    }
    
    temp = kernel_idx;
    uint kernel_coords[16];
    for (int i = n_spatial - 1; i >= 0; i--) {
        kernel_coords[i] = temp % kernel_size[i];
        temp /= kernel_size[i];
    }
    
    uint padded_coords[16];
    bool in_bounds = true;
    for (uint i = 0; i < n_spatial; i++) {
        int coord = block_coords[i] * stride[i] + kernel_coords[i] * dilation[i];
        if (coord < int(padding[2*i]) || coord >= int(padding[2*i] + in_shape[2+i])) {
            in_bounds = false;
            break;
        }
        padded_coords[i] = coord - padding[2*i];
    }
    
    int value = 0;
    if (in_bounds) {
        uint in_offset = b * in_shape[1];
        in_offset = (in_offset + c);
        
        for (uint i = 0; i < n_spatial; i++) {
            in_offset = in_offset * in_shape[2+i] + padded_coords[i];
        }
        
        value = in[in_offset];
    }
    
    uint out_offset = b * (channels * kernel_elements * num_blocks) + col_row * num_blocks + block_idx;
    out[out_offset] = value;
}

kernel void fold_int(device int* out [[buffer(0)]],
                    device const int* in [[buffer(1)]],
                    constant uint* output_size [[buffer(2)]],
                    constant uint* kernel_size [[buffer(3)]],
                    constant uint* stride [[buffer(4)]],
                    constant uint* dilation [[buffer(5)]],
                    constant uint* padding [[buffer(6)]],
                    constant uint* out_spatial [[buffer(7)]],
                    constant uint& n_spatial [[buffer(8)]],
                    constant uint& batch_size [[buffer(9)]],
                    constant uint& channels [[buffer(10)]],
                    constant uint& kernel_elements [[buffer(11)]],
                    constant uint& num_blocks [[buffer(12)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint b = gid.z;
    uint c = gid.y;
    uint spatial_idx = gid.x;
    
    if (b >= batch_size || c >= channels) return;
    
    uint padded_size[16];
    uint padded_numel = 1;
    for (uint i = 0; i < n_spatial; i++) {
        padded_size[i] = output_size[i] + padding[2*i] + padding[2*i+1];
        padded_numel *= padded_size[i];
    }
    
    if (spatial_idx >= padded_numel) return;
    
    uint temp = spatial_idx;
    uint out_coords[16];
    for (int i = n_spatial - 1; i >= 0; i--) {
        out_coords[i] = temp % padded_size[i];
        temp /= padded_size[i];
    }
    
    int sum = 0;
    
    for (uint block_idx = 0; block_idx < num_blocks; block_idx++) {
        temp = block_idx;
        uint block_coords[16];
        for (int i = n_spatial - 1; i >= 0; i--) {
            block_coords[i] = temp % out_spatial[i];
            temp /= out_spatial[i];
        }
        
        bool contributes = true;
        uint kernel_idx = 0;
        uint kernel_multiplier = 1;
        
        for (int i = n_spatial - 1; i >= 0; i--) {
            int block_start = block_coords[i] * stride[i];
            int rel_pos = int(out_coords[i]) - block_start;
            
            if (rel_pos < 0 || rel_pos % dilation[i] != 0) {
                contributes = false;
                break;
            }
            
            uint kernel_coord = rel_pos / dilation[i];
            if (kernel_coord >= kernel_size[i]) {
                contributes = false;
                break;
            }
            
            kernel_idx += kernel_coord * kernel_multiplier;
            kernel_multiplier *= kernel_size[i];
        }
        
        if (contributes) {
            uint input_row = c * kernel_elements + kernel_idx;
            uint in_offset = b * (channels * kernel_elements * num_blocks) + 
                           input_row * num_blocks + block_idx;
            sum += in[in_offset];
        }
    }
    
    uint out_offset = b * channels;
    for (uint i = 0; i < n_spatial; i++) {
        out_offset = out_offset * padded_size[i] + out_coords[i];
    }
    out[out_offset] = sum;
}

// Double versions of unfold/fold
// NOTE: Commented out because Metal doesn't support double on most hardware
/*
kernel void unfold_double(device double* out [[buffer(0)]],
                         device const double* in [[buffer(1)]],
                         constant uint* in_shape [[buffer(2)]],
                         constant uint* kernel_size [[buffer(3)]],
                         constant uint* stride [[buffer(4)]],
                         constant uint* dilation [[buffer(5)]],
                         constant uint* padding [[buffer(6)]],
                         constant uint* out_spatial [[buffer(7)]],
                         constant uint& n_spatial [[buffer(8)]],
                         constant uint& channels [[buffer(9)]],
                         constant uint& kernel_elements [[buffer(10)]],
                         constant uint& num_blocks [[buffer(11)]],
                         uint3 gid [[thread_position_in_grid]]) {
    uint b = gid.z;
    uint block_idx = gid.y;
    uint col_row = gid.x;
    
    uint batch_size = in_shape[0];
    if (b >= batch_size || block_idx >= num_blocks || col_row >= channels * kernel_elements) return;
    
    uint c = col_row / kernel_elements;
    uint kernel_idx = col_row % kernel_elements;
    
    uint temp = block_idx;
    uint block_coords[16];
    for (int i = n_spatial - 1; i >= 0; i--) {
        block_coords[i] = temp % out_spatial[i];
        temp /= out_spatial[i];
    }
    
    temp = kernel_idx;
    uint kernel_coords[16];
    for (int i = n_spatial - 1; i >= 0; i--) {
        kernel_coords[i] = temp % kernel_size[i];
        temp /= kernel_size[i];
    }
    
    uint padded_coords[16];
    bool in_bounds = true;
    for (uint i = 0; i < n_spatial; i++) {
        int coord = block_coords[i] * stride[i] + kernel_coords[i] * dilation[i];
        if (coord < int(padding[2*i]) || coord >= int(padding[2*i] + in_shape[2+i])) {
            in_bounds = false;
            break;
        }
        padded_coords[i] = coord - padding[2*i];
    }
    
    double value = 0.0;
    if (in_bounds) {
        uint in_offset = b * in_shape[1];
        in_offset = (in_offset + c);
        
        for (uint i = 0; i < n_spatial; i++) {
            in_offset = in_offset * in_shape[2+i] + padded_coords[i];
        }
        
        value = in[in_offset];
    }
    
    uint out_offset = b * (channels * kernel_elements * num_blocks) + col_row * num_blocks + block_idx;
    out[out_offset] = value;
}

kernel void fold_double(device double* out [[buffer(0)]],
                       device const double* in [[buffer(1)]],
                       constant uint* output_size [[buffer(2)]],
                       constant uint* kernel_size [[buffer(3)]],
                       constant uint* stride [[buffer(4)]],
                       constant uint* dilation [[buffer(5)]],
                       constant uint* padding [[buffer(6)]],
                       constant uint* out_spatial [[buffer(7)]],
                       constant uint& n_spatial [[buffer(8)]],
                       constant uint& batch_size [[buffer(9)]],
                       constant uint& channels [[buffer(10)]],
                       constant uint& kernel_elements [[buffer(11)]],
                       constant uint& num_blocks [[buffer(12)]],
                       uint3 gid [[thread_position_in_grid]]) {
    uint b = gid.z;
    uint c = gid.y;
    uint spatial_idx = gid.x;
    
    if (b >= batch_size || c >= channels) return;
    
    uint padded_size[16];
    uint padded_numel = 1;
    for (uint i = 0; i < n_spatial; i++) {
        padded_size[i] = output_size[i] + padding[2*i] + padding[2*i+1];
        padded_numel *= padded_size[i];
    }
    
    if (spatial_idx >= padded_numel) return;
    
    uint temp = spatial_idx;
    uint out_coords[16];
    for (int i = n_spatial - 1; i >= 0; i--) {
        out_coords[i] = temp % padded_size[i];
        temp /= padded_size[i];
    }
    
    double sum = 0.0;
    
    for (uint block_idx = 0; block_idx < num_blocks; block_idx++) {
        temp = block_idx;
        uint block_coords[16];
        for (int i = n_spatial - 1; i >= 0; i--) {
            block_coords[i] = temp % out_spatial[i];
            temp /= out_spatial[i];
        }
        
        bool contributes = true;
        uint kernel_idx = 0;
        uint kernel_multiplier = 1;
        
        for (int i = n_spatial - 1; i >= 0; i--) {
            int block_start = block_coords[i] * stride[i];
            int rel_pos = int(out_coords[i]) - block_start;
            
            if (rel_pos < 0 || rel_pos % dilation[i] != 0) {
                contributes = false;
                break;
            }
            
            uint kernel_coord = rel_pos / dilation[i];
            if (kernel_coord >= kernel_size[i]) {
                contributes = false;
                break;
            }
            
            kernel_idx += kernel_coord * kernel_multiplier;
            kernel_multiplier *= kernel_size[i];
        }
        
        if (contributes) {
            uint input_row = c * kernel_elements + kernel_idx;
            uint in_offset = b * (channels * kernel_elements * num_blocks) + 
                           input_row * num_blocks + block_idx;
            sum += in[in_offset];
        }
    }
    
    uint out_offset = b * channels;
    for (uint i = 0; i < n_spatial; i++) {
        out_offset = out_offset * padded_size[i] + out_coords[i];
    }
    out[out_offset] = sum;
}

kernel void unfold_long(device long* out [[buffer(0)]],
                       device const long* in [[buffer(1)]],
                       constant uint* in_shape [[buffer(2)]],
                       constant uint* kernel_size [[buffer(3)]],
                       constant uint* stride [[buffer(4)]],
                       constant uint* dilation [[buffer(5)]],
                       constant uint* padding [[buffer(6)]],
                       constant uint* out_spatial [[buffer(7)]],
                       constant uint& n_spatial [[buffer(8)]],
                       constant uint& channels [[buffer(9)]],
                       constant uint& kernel_elements [[buffer(10)]],
                       constant uint& num_blocks [[buffer(11)]],
                       uint3 gid [[thread_position_in_grid]]) {
    uint b = gid.z;
    uint block_idx = gid.y;
    uint col_row = gid.x;
    
    uint batch_size = in_shape[0];
    if (b >= batch_size || block_idx >= num_blocks || col_row >= channels * kernel_elements) return;
    
    uint c = col_row / kernel_elements;
    uint kernel_idx = col_row % kernel_elements;
    
    uint temp = block_idx;
    uint block_coords[16];
    for (int i = n_spatial - 1; i >= 0; i--) {
        block_coords[i] = temp % out_spatial[i];
        temp /= out_spatial[i];
    }
    
    temp = kernel_idx;
    uint kernel_coords[16];
    for (int i = n_spatial - 1; i >= 0; i--) {
        kernel_coords[i] = temp % kernel_size[i];
        temp /= kernel_size[i];
    }
    
    uint padded_coords[16];
    bool in_bounds = true;
    for (uint i = 0; i < n_spatial; i++) {
        int coord = block_coords[i] * stride[i] + kernel_coords[i] * dilation[i];
        if (coord < int(padding[2*i]) || coord >= int(padding[2*i] + in_shape[2+i])) {
            in_bounds = false;
            break;
        }
        padded_coords[i] = coord - padding[2*i];
    }
    
    long value = 0;
    if (in_bounds) {
        uint in_offset = b * in_shape[1];
        in_offset = (in_offset + c);
        
        for (uint i = 0; i < n_spatial; i++) {
            in_offset = in_offset * in_shape[2+i] + padded_coords[i];
        }
        
        value = in[in_offset];
    }
    
    uint out_offset = b * (channels * kernel_elements * num_blocks) + col_row * num_blocks + block_idx;
    out[out_offset] = value;
}

kernel void fold_long(device long* out [[buffer(0)]],
                     device const long* in [[buffer(1)]],
                     constant uint* output_size [[buffer(2)]],
                     constant uint* kernel_size [[buffer(3)]],
                     constant uint* stride [[buffer(4)]],
                     constant uint* dilation [[buffer(5)]],
                     constant uint* padding [[buffer(6)]],
                     constant uint* out_spatial [[buffer(7)]],
                     constant uint& n_spatial [[buffer(8)]],
                     constant uint& batch_size [[buffer(9)]],
                     constant uint& channels [[buffer(10)]],
                     constant uint& kernel_elements [[buffer(11)]],
                     constant uint& num_blocks [[buffer(12)]],
                     uint3 gid [[thread_position_in_grid]]) {
    uint b = gid.z;
    uint c = gid.y;
    uint spatial_idx = gid.x;
    
    if (b >= batch_size || c >= channels) return;
    
    uint padded_size[16];
    uint padded_numel = 1;
    for (uint i = 0; i < n_spatial; i++) {
        padded_size[i] = output_size[i] + padding[2*i] + padding[2*i+1];
        padded_numel *= padded_size[i];
    }
    
    if (spatial_idx >= padded_numel) return;
    
    uint temp = spatial_idx;
    uint out_coords[16];
    for (int i = n_spatial - 1; i >= 0; i--) {
        out_coords[i] = temp % padded_size[i];
        temp /= padded_size[i];
    }
    
    long sum = 0;
    
    for (uint block_idx = 0; block_idx < num_blocks; block_idx++) {
        temp = block_idx;
        uint block_coords[16];
        for (int i = n_spatial - 1; i >= 0; i--) {
            block_coords[i] = temp % out_spatial[i];
            temp /= out_spatial[i];
        }
        
        bool contributes = true;
        uint kernel_idx = 0;
        uint kernel_multiplier = 1;
        
        for (int i = n_spatial - 1; i >= 0; i--) {
            int block_start = block_coords[i] * stride[i];
            int rel_pos = int(out_coords[i]) - block_start;
            
            if (rel_pos < 0 || rel_pos % dilation[i] != 0) {
                contributes = false;
                break;
            }
            
            uint kernel_coord = rel_pos / dilation[i];
            if (kernel_coord >= kernel_size[i]) {
                contributes = false;
                break;
            }
            
            kernel_idx += kernel_coord * kernel_multiplier;
            kernel_multiplier *= kernel_size[i];
        }
        
        if (contributes) {
            uint input_row = c * kernel_elements + kernel_idx;
            uint in_offset = b * (channels * kernel_elements * num_blocks) + 
                           input_row * num_blocks + block_idx;
            sum += in[in_offset];
        }
    }
    
    uint out_offset = b * channels;
    for (uint i = 0; i < n_spatial; i++) {
        out_offset = out_offset * padded_size[i] + out_coords[i];
    }
    out[out_offset] = sum;
}*/
