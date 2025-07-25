#include <metal_stdlib>
using namespace metal;

// Helper functions for complex operations
inline float2 complex_mul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline float2 complex_conj(float2 a) {
    return float2(a.x, -a.y);
}

inline float complex_abs(float2 a) {
    return sqrt(a.x * a.x + a.y * a.y);
}

inline float2 complex_div(float2 a, float b) {
    return float2(a.x / b, a.y / b);
}

// ===== CHOLESKY DECOMPOSITION =====
// Implements Cholesky-Banachiewicz algorithm (row-by-row version)

kernel void cholesky_float(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint4& params [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    const uint batch_idx = gid;
    const uint batch_size = params.x;
    const uint n = params.y;
    const bool upper = params.z > 0;
    
    if (batch_idx >= batch_size || n > 64) return;
    
    const uint batch_offset = batch_idx * n * n;
    threadgroup float cache[64 * 64];
    
    // Copy input to cache
    for (uint idx = tid; idx < n * n; idx += tg_size) {
        cache[idx] = in[batch_offset + idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Cholesky-Banachiewicz algorithm
    if (upper) {
        // Upper triangular: A = U^T * U
        for (uint j = 0; j < n; j++) {
            // Compute diagonal element U[j,j]
            if (tid == 0) {
                float sum = 0;
                for (uint k = 0; k < j; k++) {
                    float val = cache[k * n + j];
                    sum += val * val;
                }
                float diag = cache[j * n + j] - sum;
                if (diag <= 0) {
                    cache[j * n + j] = NAN;
                } else {
                    cache[j * n + j] = sqrt(diag);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Compute off-diagonal elements U[i,j] for i < j
            for (uint i = tid; i < j; i += tg_size) {
                float sum = 0;
                for (uint k = 0; k < i; k++) {
                    sum += cache[k * n + i] * cache[k * n + j];
                }
                cache[i * n + j] = (cache[i * n + j] - sum) / cache[i * n + i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        // Zero out lower triangle
        for (uint idx = tid; idx < n * n; idx += tg_size) {
            uint i = idx / n;
            uint j = idx % n;
            if (i > j) cache[idx] = 0;
        }
    } else {
        // Lower triangular: A = L * L^T
        for (uint i = 0; i < n; i++) {
            // Compute diagonal element L[i,i]
            if (tid == 0) {
                float sum = 0;
                for (uint k = 0; k < i; k++) {
                    float val = cache[i * n + k];
                    sum += val * val;
                }
                float diag = cache[i * n + i] - sum;
                if (diag <= 0) {
                    cache[i * n + i] = NAN;
                } else {
                    cache[i * n + i] = sqrt(diag);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Compute off-diagonal elements L[j,i] for j > i
            for (uint j = i + 1 + tid; j < n; j += tg_size) {
                float sum = 0;
                for (uint k = 0; k < i; k++) {
                    sum += cache[j * n + k] * cache[i * n + k];
                }
                cache[j * n + i] = (cache[j * n + i] - sum) / cache[i * n + i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        // Zero out upper triangle
        for (uint idx = tid; idx < n * n; idx += tg_size) {
            uint i = idx / n;
            uint j = idx % n;
            if (i < j) cache[idx] = 0;
        }
    }
    
    // Copy result back to global memory
    for (uint idx = tid; idx < n * n; idx += tg_size) {
        out[batch_offset + idx] = cache[idx];
    }
}

// Complex Cholesky
kernel void cholesky_float2(
    device const float2* in [[buffer(0)]],
    device float2* out [[buffer(1)]],
    constant uint4& params [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    const uint batch_idx = gid;
    const uint batch_size = params.x;
    const uint n = params.y;
    const bool upper = params.z > 0;
    
    if (batch_idx >= batch_size || n > 64) return;
    
    const uint batch_offset = batch_idx * n * n;
    threadgroup float2 cache[64 * 64];
    
    // Copy input to cache
    for (uint idx = tid; idx < n * n; idx += tg_size) {
        cache[idx] = in[batch_offset + idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (upper) {
        // Upper triangular: A = U^H * U
        for (uint j = 0; j < n; j++) {
            // Compute diagonal element (always real for Hermitian matrices)
            if (tid == 0) {
                float sum = 0;
                for (uint k = 0; k < j; k++) {
                    float2 val = cache[k * n + j];
                    sum += val.x * val.x + val.y * val.y;
                }
                float diag = cache[j * n + j].x - sum;
                if (diag <= 0) {
                    cache[j * n + j] = float2(NAN, 0);
                } else {
                    cache[j * n + j] = float2(sqrt(diag), 0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Compute off-diagonal elements
            for (uint i = tid; i < j; i += tg_size) {
                float2 sum = float2(0, 0);
                for (uint k = 0; k < i; k++) {
                    float2 u_ki = cache[k * n + i];
                    float2 u_kj = cache[k * n + j];
                    // conj(u_ki) * u_kj
                    sum.x += u_ki.x * u_kj.x + u_ki.y * u_kj.y;
                    sum.y += u_ki.x * u_kj.y - u_ki.y * u_kj.x;
                }
                float2 result = cache[i * n + j] - sum;
                float diag = cache[i * n + i].x;
                cache[i * n + j] = complex_div(result, diag);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        // Zero out lower triangle
        for (uint idx = tid; idx < n * n; idx += tg_size) {
            uint i = idx / n;
            uint j = idx % n;
            if (i > j) cache[idx] = float2(0, 0);
        }
    } else {
        // Lower triangular: A = L * L^H
        for (uint i = 0; i < n; i++) {
            // Compute diagonal element
            if (tid == 0) {
                float sum = 0;
                for (uint k = 0; k < i; k++) {
                    float2 val = cache[i * n + k];
                    sum += val.x * val.x + val.y * val.y;
                }
                float diag = cache[i * n + i].x - sum;
                if (diag <= 0) {
                    cache[i * n + i] = float2(NAN, 0);
                } else {
                    cache[i * n + i] = float2(sqrt(diag), 0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Compute off-diagonal elements
            for (uint j = i + 1 + tid; j < n; j += tg_size) {
                float2 sum = float2(0, 0);
                for (uint k = 0; k < i; k++) {
                    float2 l_jk = cache[j * n + k];
                    float2 l_ik = cache[i * n + k];
                    // l_jk * conj(l_ik)
                    sum.x += l_jk.x * l_ik.x + l_jk.y * l_ik.y;
                    sum.y += l_jk.y * l_ik.x - l_jk.x * l_ik.y;
                }
                float2 result = cache[j * n + i] - sum;
                float diag = cache[i * n + i].x;
                cache[j * n + i] = complex_div(result, diag);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        // Zero out upper triangle
        for (uint idx = tid; idx < n * n; idx += tg_size) {
            uint i = idx / n;
            uint j = idx % n;
            if (i < j) cache[idx] = float2(0, 0);
        }
    }
    
    // Copy result back
    for (uint idx = tid; idx < n * n; idx += tg_size) {
        out[batch_offset + idx] = cache[idx];
    }
}

// ===== QR DECOMPOSITION =====
// Implements Householder reflections algorithm

kernel void qr_float(
    device const float* in [[buffer(0)]],
    device float* q [[buffer(1)]],
    device float* r [[buffer(2)]],
    constant uint4& params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    const uint batch_idx = gid;
    const uint batch_size = params.x;
    const uint m = params.y;
    const uint n = params.z;
    const bool reduced = params.w > 0;
    
    if (batch_idx >= batch_size || m > 128 || n > 128) return;
    
    const uint batch_offset_in = batch_idx * m * n;
    const uint k = min(m, n);
    
    const uint q_cols = reduced ? k : m;
    const uint r_rows = reduced ? k : m;
    const uint batch_offset_q = batch_idx * m * q_cols;
    const uint batch_offset_r = batch_idx * r_rows * n;
    
    threadgroup float a_cache[128 * 128];
    threadgroup float v[128];
    threadgroup float tau;
    
    // Copy input to R and working cache
    for (uint idx = tid; idx < m * n; idx += tg_size) {
        uint i = idx / n;
        uint j = idx % n;
        if (i < r_rows) {
            r[batch_offset_r + i * n + j] = in[batch_offset_in + i * n + j];
        }
        a_cache[idx] = in[batch_offset_in + idx];
    }
    
    // Initialize Q to identity
    for (uint idx = tid; idx < m * q_cols; idx += tg_size) {
        uint i = idx / q_cols;
        uint j = idx % q_cols;
        q[batch_offset_q + idx] = (i == j) ? 1.0f : 0.0f;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Householder QR decomposition
    for (uint col = 0; col < k; col++) {
        // Compute Householder vector
        if (tid == 0) {
            float norm_sq = 0;
            for (uint i = col; i < m; i++) {
                float val = a_cache[i * n + col];
                norm_sq += val * val;
            }
            float norm = sqrt(norm_sq);
            
            float x0 = a_cache[col * n + col];
            float sign = (x0 >= 0) ? 1.0f : -1.0f;
            float u1 = x0 + sign * norm;
            
            v[col] = u1;
            for (uint i = col + 1; i < m; i++) {
                v[i] = a_cache[i * n + col];
            }
            
            float v_norm_sq = u1 * u1;
            for (uint i = col + 1; i < m; i++) {
                v_norm_sq += v[i] * v[i];
            }
            tau = 2.0f / v_norm_sq;
            
            if (col < r_rows) {
                r[batch_offset_r + col * n + col] = -sign * norm;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Apply Householder reflection to remaining columns of A
        for (uint j = col + 1 + tid; j < n; j += tg_size) {
            float dot = 0;
            for (uint i = col; i < m; i++) {
                dot += v[i] * a_cache[i * n + j];
            }
            
            for (uint i = col; i < m; i++) {
                a_cache[i * n + j] -= tau * v[i] * dot;
            }
            
            if (col < r_rows) {
                r[batch_offset_r + col * n + j] = a_cache[col * n + j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Apply Householder reflection to Q
        for (uint j = tid; j < q_cols; j += tg_size) {
            float dot = 0;
            for (uint i = col; i < m; i++) {
                dot += v[i] * q[batch_offset_q + i * q_cols + j];
            }
            
            for (uint i = col; i < m; i++) {
                q[batch_offset_q + i * q_cols + j] -= tau * v[i] * dot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Zero out below diagonal in R
        if (tid == 0 && col < r_rows) {
            for (uint i = col + 1; i < r_rows; i++) {
                r[batch_offset_r + i * n + col] = 0;
            }
        }
    }
}

// Complex QR
kernel void qr_float2(
    device const float2* in [[buffer(0)]],
    device float2* q [[buffer(1)]],
    device float2* r [[buffer(2)]],
    constant uint4& params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    const uint batch_idx = gid;
    const uint batch_size = params.x;
    const uint m = params.y;
    const uint n = params.z;
    const bool reduced = params.w > 0;
    
    if (batch_idx >= batch_size || m > 128 || n > 128) return;
    
    const uint batch_offset_in = batch_idx * m * n;
    const uint k = min(m, n);
    
    const uint q_cols = reduced ? k : m;
    const uint r_rows = reduced ? k : m;
    const uint batch_offset_q = batch_idx * m * q_cols;
    const uint batch_offset_r = batch_idx * r_rows * n;
    
    threadgroup float2 a_cache[128 * 128];
    threadgroup float2 v[128];
    threadgroup float tau_real;
    
    // Copy input to R and cache
    for (uint idx = tid; idx < m * n; idx += tg_size) {
        uint i = idx / n;
        uint j = idx % n;
        if (i < r_rows) {
            r[batch_offset_r + i * n + j] = in[batch_offset_in + i * n + j];
        }
        a_cache[idx] = in[batch_offset_in + idx];
    }
    
    // Initialize Q to identity
    for (uint idx = tid; idx < m * q_cols; idx += tg_size) {
        uint i = idx / q_cols;
        uint j = idx % q_cols;
        q[batch_offset_q + idx] = (i == j) ? float2(1, 0) : float2(0, 0);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Complex Householder QR
    for (uint col = 0; col < k; col++) {
        if (tid == 0) {
            float norm_sq = 0;
            for (uint i = col; i < m; i++) {
                float2 val = a_cache[i * n + col];
                norm_sq += val.x * val.x + val.y * val.y;
            }
            
            if (norm_sq > 1e-10f) {
                float norm = sqrt(norm_sq);
                float2 x0 = a_cache[col * n + col];
                
                // Phase of x0
                float phase = atan2(x0.y, x0.x);
                float2 phase_factor = float2(cos(phase), sin(phase));
                
                // v[0] = x0 + norm * e^(i*phase)
                v[col] = x0 + float2(norm * phase_factor.x, norm * phase_factor.y);
                
                for (uint i = col + 1; i < m; i++) {
                    v[i] = a_cache[i * n + col];
                }
                
                float v_norm_sq = 0;
                for (uint i = col; i < m; i++) {
                    v_norm_sq += v[i].x * v[i].x + v[i].y * v[i].y;
                }
                tau_real = 2.0f / v_norm_sq;
                
                if (col < r_rows) {
                    r[batch_offset_r + col * n + col] = float2(-norm * phase_factor.x, -norm * phase_factor.y);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Apply to remaining columns
        for (uint j = col + 1 + tid; j < n; j += tg_size) {
            float2 dot = float2(0, 0);
            for (uint i = col; i < m; i++) {
                float2 vi = v[i];
                float2 aij = a_cache[i * n + j];
                // conj(v[i]) * a[i,j]
                dot.x += vi.x * aij.x + vi.y * aij.y;
                dot.y += vi.x * aij.y - vi.y * aij.x;
            }
            dot.x *= tau_real;
            dot.y *= tau_real;
            
            for (uint i = col; i < m; i++) {
                float2 vi = v[i];
                // a[i,j] -= vi * dot
                a_cache[i * n + j].x -= vi.x * dot.x - vi.y * dot.y;
                a_cache[i * n + j].y -= vi.x * dot.y + vi.y * dot.x;
            }
            
            if (col < r_rows) {
                r[batch_offset_r + col * n + j] = a_cache[col * n + j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Apply to Q
        for (uint j = tid; j < q_cols; j += tg_size) {
            float2 dot = float2(0, 0);
            for (uint i = col; i < m; i++) {
                float2 vi = v[i];
                float2 qij = q[batch_offset_q + i * q_cols + j];
                // conj(vi) * q[i,j]
                dot.x += vi.x * qij.x + vi.y * qij.y;
                dot.y += vi.x * qij.y - vi.y * qij.x;
            }
            dot.x *= tau_real;
            dot.y *= tau_real;
            
            for (uint i = col; i < m; i++) {
                float2 vi = v[i];
                // q[i,j] -= vi * dot
                q[batch_offset_q + i * q_cols + j].x -= vi.x * dot.x - vi.y * dot.y;
                q[batch_offset_q + i * q_cols + j].y -= vi.x * dot.y + vi.y * dot.x;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Zero below diagonal
        if (tid == 0 && col < r_rows) {
            for (uint i = col + 1; i < m; i++) {
                a_cache[i * n + col] = float2(0, 0);
            }
            for (uint i = col + 1; i < r_rows; i++) {
                r[batch_offset_r + i * n + col] = float2(0, 0);
            }
        }
    }
}

// ===== TRIANGULAR SOLVE =====
// Solves AX = B where A is triangular

kernel void triangular_solve_float(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* x [[buffer(2)]],
    constant uint4& params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    const uint batch_idx = gid;
    const uint batch_size = params.x;
    const uint n = params.y;
    const uint k = params.z;
    const bool upper = params.w > 0;
    
    if (batch_idx >= batch_size || n > 64 || k > 64) return;
    
    const uint batch_offset_a = batch_idx * n * n;
    const uint batch_offset_b = batch_idx * n * k;
    
    threadgroup float a_cache[64 * 64];
    threadgroup float x_cache[64 * 64];
    
    // Copy A and B to cache
    for (uint idx = tid; idx < n * n; idx += tg_size) {
        a_cache[idx] = a[batch_offset_a + idx];
    }
    for (uint idx = tid; idx < n * k; idx += tg_size) {
        x_cache[idx] = b[batch_offset_b + idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Solve for each column of X
    for (uint col = 0; col < k; col++) {
        if (upper) {
            // Back substitution
            for (int i = n - 1; i >= 0; i--) {
                if (tid == 0) {
                    float sum = x_cache[i * k + col];
                    for (uint j = i + 1; j < n; j++) {
                        sum -= a_cache[i * n + j] * x_cache[j * k + col];
                    }
                    x_cache[i * k + col] = sum / a_cache[i * n + i];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        } else {
            // Forward substitution
            for (uint i = 0; i < n; i++) {
                if (tid == 0) {
                    float sum = x_cache[i * k + col];
                    for (uint j = 0; j < i; j++) {
                        sum -= a_cache[i * n + j] * x_cache[j * k + col];
                    }
                    x_cache[i * k + col] = sum / a_cache[i * n + i];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
    
    // Copy result back
    for (uint idx = tid; idx < n * k; idx += tg_size) {
        x[batch_offset_b + idx] = x_cache[idx];
    }
}

// Complex triangular solve
kernel void triangular_solve_float2(
    device const float2* a [[buffer(0)]],
    device const float2* b [[buffer(1)]],
    device float2* x [[buffer(2)]],
    constant uint4& params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    const uint batch_idx = gid;
    const uint batch_size = params.x;
    const uint n = params.y;
    const uint k = params.z;
    const bool upper = params.w > 0;
    
    if (batch_idx >= batch_size || n > 64 || k > 64) return;
    
    const uint batch_offset_a = batch_idx * n * n;
    const uint batch_offset_b = batch_idx * n * k;
    
    threadgroup float2 a_cache[64 * 64];
    threadgroup float2 x_cache[64 * 64];
    
    // Copy A and B to cache
    for (uint idx = tid; idx < n * n; idx += tg_size) {
        a_cache[idx] = a[batch_offset_a + idx];
    }
    for (uint idx = tid; idx < n * k; idx += tg_size) {
        x_cache[idx] = b[batch_offset_b + idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Solve for each column
    for (uint col = 0; col < k; col++) {
        if (upper) {
            // Back substitution
            for (int i = n - 1; i >= 0; i--) {
                if (tid == 0) {
                    float2 sum = x_cache[i * k + col];
                    for (uint j = i + 1; j < n; j++) {
                        float2 aij = a_cache[i * n + j];
                        float2 xj = x_cache[j * k + col];
                        sum.x -= aij.x * xj.x - aij.y * xj.y;
                        sum.y -= aij.x * xj.y + aij.y * xj.x;
                    }
                    // Divide by diagonal element
                    float2 aii = a_cache[i * n + i];
                    float det = aii.x * aii.x + aii.y * aii.y;
                    x_cache[i * k + col].x = (sum.x * aii.x + sum.y * aii.y) / det;
                    x_cache[i * k + col].y = (sum.y * aii.x - sum.x * aii.y) / det;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        } else {
            // Forward substitution
            for (uint i = 0; i < n; i++) {
                if (tid == 0) {
                    float2 sum = x_cache[i * k + col];
                    for (uint j = 0; j < i; j++) {
                        float2 aij = a_cache[i * n + j];
                        float2 xj = x_cache[j * k + col];
                        sum.x -= aij.x * xj.x - aij.y * xj.y;
                        sum.y -= aij.x * xj.y + aij.y * xj.x;
                    }
                    // Divide by diagonal element
                    float2 aii = a_cache[i * n + i];
                    float det = aii.x * aii.x + aii.y * aii.y;
                    x_cache[i * k + col].x = (sum.x * aii.x + sum.y * aii.y) / det;
                    x_cache[i * k + col].y = (sum.y * aii.x - sum.x * aii.y) / det;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
    
    // Copy result back
    for (uint idx = tid; idx < n * k; idx += tg_size) {
        x[batch_offset_b + idx] = x_cache[idx];
    }
}

// ===== SYMMETRIC/HERMITIAN EIGENVALUE DECOMPOSITION =====
// Implements tridiagonalization + QR algorithm

kernel void eig_symmetric_float(
    device const float* in [[buffer(0)]],
    device float* eigenvalues [[buffer(1)]],
    device float* eigenvectors [[buffer(2)]],
    constant uint4& params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    const uint batch_idx = gid;
    const uint batch_size = params.x;
    const uint n = params.y;
    const bool compute_vectors = params.z > 0;
    
    if (batch_idx >= batch_size || n > 64) return;
    
    const uint batch_offset = batch_idx * n * n;
    const uint batch_offset_eval = batch_idx * n;
    
    threadgroup float a[64 * 64];
    threadgroup float q[64 * 64];
    threadgroup float diag[64];
    threadgroup float off_diag[64];
    
    // Copy input
    for (uint idx = tid; idx < n * n; idx += tg_size) {
        a[idx] = in[batch_offset + idx];
        if (compute_vectors) {
            uint i = idx / n;
            uint j = idx % n;
            q[idx] = (i == j) ? 1.0f : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Tridiagonalization using Householder reflections
    for (uint k = 0; k < n - 2; k++) {
        if (tid == 0) {
            // Compute Householder vector for column k
            float norm_sq = 0;
            for (uint i = k + 1; i < n; i++) {
                float val = a[i * n + k];
                norm_sq += val * val;
            }
            
            if (norm_sq > 1e-10f) {
                float norm = sqrt(norm_sq);
                float x0 = a[(k + 1) * n + k];
                float sign = (x0 >= 0) ? 1.0f : -1.0f;
                
                // Apply Householder transformation
                float u1 = x0 + sign * norm;
                float tau = 2.0f / (u1 * u1 + norm_sq);
                
                // Update matrix
                for (uint i = k + 1; i < n; i++) {
                    for (uint j = k + 1; j < n; j++) {
                        float vi = (i == k + 1) ? u1 : a[i * n + k];
                        float vj = (j == k + 1) ? u1 : a[j * n + k];
                        a[i * n + j] -= tau * vi * vj;
                        a[j * n + i] = a[i * n + j]; // Maintain symmetry
                    }
                }
                
                // Update eigenvectors if needed
                if (compute_vectors) {
                    for (uint i = 0; i < n; i++) {
                        float dot = 0;
                        for (uint j = k + 1; j < n; j++) {
                            float vj = (j == k + 1) ? u1 : a[j * n + k];
                            dot += vj * q[i * n + j];
                        }
                        dot *= tau;
                        for (uint j = k + 1; j < n; j++) {
                            float vj = (j == k + 1) ? u1 : a[j * n + k];
                            q[i * n + j] -= vj * dot;
                        }
                    }
                }
                
                // Store result
                a[(k + 1) * n + k] = -sign * norm;
                a[k * n + (k + 1)] = -sign * norm;
                for (uint i = k + 2; i < n; i++) {
                    a[i * n + k] = 0;
                    a[k * n + i] = 0;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Extract diagonal and off-diagonal
    if (tid < n) {
        diag[tid] = a[tid * n + tid];
        if (tid < n - 1) {
            off_diag[tid] = a[tid * n + (tid + 1)];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // QR algorithm on tridiagonal matrix
    const uint max_iterations = 30 * n;
    for (uint iter = 0; iter < max_iterations; iter++) {
        // Check convergence
        bool converged = true;
        if (tid == 0) {
            for (uint i = 0; i < n - 1; i++) {
                if (abs(off_diag[i]) > 1e-10f * (abs(diag[i]) + abs(diag[i + 1]))) {
                    converged = false;
                    break;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (converged) break;
        
        // Wilkinson shift
        float shift = 0;
        if (tid == 0) {
            float d = (diag[n - 2] - diag[n - 1]) * 0.5f;
            float sign_d = (d >= 0) ? 1.0f : -1.0f;
            shift = diag[n - 1] - off_diag[n - 2] * off_diag[n - 2] / 
                   (d + sign_d * sqrt(d * d + off_diag[n - 2] * off_diag[n - 2]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // QR step with shift
        if (tid == 0) {
            float c = 1, s = 0;
            float p = diag[0] - shift;
            
            for (uint k = 0; k < n - 1; k++) {
                float q_val = off_diag[k];
                float r = sqrt(p * p + q_val * q_val);
                
                if (r > 1e-10f) {
                    c = p / r;
                    s = q_val / r;
                    
                    diag[k] = r + shift;
                    if (k > 0) {
                        off_diag[k - 1] = s * off_diag[k - 1];
                    }
                    
                    p = c * (diag[k + 1] - shift) - s * off_diag[k];
                    diag[k + 1] = s * s * (diag[k + 1] - shift) + c * c * diag[k + 1] + 
                                  2 * s * c * off_diag[k];
                    
                    if (k < n - 2) {
                        off_diag[k + 1] = c * off_diag[k + 1];
                    }
                    off_diag[k] = s * p;
                    
                    // Update eigenvectors
                    if (compute_vectors) {
                        for (uint i = 0; i < n; i++) {
                            float qik = q[i * n + k];
                            float qik1 = q[i * n + (k + 1)];
                            q[i * n + k] = c * qik - s * qik1;
                            q[i * n + (k + 1)] = s * qik + c * qik1;
                        }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Sort eigenvalues in descending order
    if (tid == 0) {
        for (uint i = 0; i < n - 1; i++) {
            uint max_idx = i;
            float max_val = diag[i];
            
            for (uint j = i + 1; j < n; j++) {
                if (diag[j] > max_val) {
                    max_val = diag[j];
                    max_idx = j;
                }
            }
            
            if (max_idx != i) {
                // Swap eigenvalues
                float temp = diag[i];
                diag[i] = diag[max_idx];
                diag[max_idx] = temp;
                
                // Swap eigenvectors
                if (compute_vectors) {
                    for (uint k = 0; k < n; k++) {
                        float temp = q[k * n + i];
                        q[k * n + i] = q[k * n + max_idx];
                        q[k * n + max_idx] = temp;
                    }
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Write results
    if (tid < n) {
        eigenvalues[batch_offset_eval + tid] = diag[tid];
    }
    
    if (compute_vectors) {
        for (uint idx = tid; idx < n * n; idx += tg_size) {
            eigenvectors[batch_offset + idx] = q[idx];
        }
    }
}

// Hermitian eigenvalue for complex (eigenvalues are always real)
// Implemented in linalg_complex_eig.metal

// ===== SVD =====
// Implements Golub-Kahan bidiagonalization + QR iteration

kernel void svd_float(
    device const float* in [[buffer(0)]],
    device float* u [[buffer(1)]],
    device float* s [[buffer(2)]],
    device float* vh [[buffer(3)]],
    constant uint4& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    const uint batch_idx = gid;
    const uint batch_size = params.x;
    const uint m = params.y;
    const uint n = params.z;
    const bool full_matrices = params.w > 0;
    
    if (batch_idx >= batch_size || m > 64 || n > 64) return;
    
    const uint batch_offset_in = batch_idx * m * n;
    const uint k = min(m, n);
    
    threadgroup float a[64 * 64];
    threadgroup float u_work[64 * 64];
    threadgroup float v_work[64 * 64];
    threadgroup float diag[64];
    threadgroup float super_diag[64];
    
    // Initialize
    for (uint idx = tid; idx < m * n; idx += tg_size) {
        a[idx] = in[batch_offset_in + idx];
    }
    for (uint idx = tid; idx < m * m; idx += tg_size) {
        uint i = idx / m;
        uint j = idx % m;
        u_work[idx] = (i == j) ? 1.0f : 0.0f;
    }
    for (uint idx = tid; idx < n * n; idx += tg_size) {
        uint i = idx / n;
        uint j = idx % n;
        v_work[idx] = (i == j) ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Bidiagonalization using Householder reflections
    for (uint step = 0; step < k; step++) {
        // Left Householder (column)
        if (tid == 0 && step < m - 1) {
            float norm_sq = 0;
            for (uint i = step; i < m; i++) {
                float val = a[i * n + step];
                norm_sq += val * val;
            }
            
            if (norm_sq > 1e-10f) {
                float norm = sqrt(norm_sq);
                float x0 = a[step * n + step];
                float sign = (x0 >= 0) ? 1.0f : -1.0f;
                
                // Apply Householder transformation
                float u1 = x0 + sign * norm;
                float tau = 2.0f / (u1 * u1 + norm_sq - x0 * x0);
                
                // Update A
                for (uint j = step; j < n; j++) {
                    float dot = 0;
                    dot += u1 * a[step * n + j];
                    for (uint i = step + 1; i < m; i++) {
                        dot += a[i * n + step] * a[i * n + j];
                    }
                    dot *= tau;
                    
                    a[step * n + j] -= u1 * dot;
                    for (uint i = step + 1; i < m; i++) {
                        a[i * n + j] -= a[i * n + step] * dot;
                    }
                }
                
                // Update U
                for (uint j = 0; j < m; j++) {
                    float dot = 0;
                    dot += u1 * u_work[step * m + j];
                    for (uint i = step + 1; i < m; i++) {
                        dot += a[i * n + step] * u_work[i * m + j];
                    }
                    dot *= tau;
                    
                    u_work[step * m + j] -= u1 * dot;
                    for (uint i = step + 1; i < m; i++) {
                        u_work[i * m + j] -= a[i * n + step] * dot;
                    }
                }
                
                a[step * n + step] = -sign * norm;
                for (uint i = step + 1; i < m; i++) {
                    a[i * n + step] = 0;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Right Householder (row)
        if (tid == 0 && step < n - 2 && step < m) {
            float norm_sq = 0;
            for (uint j = step + 1; j < n; j++) {
                float val = a[step * n + j];
                norm_sq += val * val;
            }
            
            if (norm_sq > 1e-10f) {
                float norm = sqrt(norm_sq);
                float x0 = a[step * n + (step + 1)];
                float sign = (x0 >= 0) ? 1.0f : -1.0f;
                
                // Apply Householder transformation
                float u1 = x0 + sign * norm;
                float tau = 2.0f / (u1 * u1 + norm_sq - x0 * x0);
                
                // Update A
                for (uint i = step; i < m; i++) {
                    float dot = 0;
                    dot += a[i * n + (step + 1)] * u1;
                    for (uint j = step + 2; j < n; j++) {
                        dot += a[i * n + j] * a[step * n + j];
                    }
                    dot *= tau;
                    
                    a[i * n + (step + 1)] -= dot * u1;
                    for (uint j = step + 2; j < n; j++) {
                        a[i * n + j] -= dot * a[step * n + j];
                    }
                }
                
                // Update V
                for (uint i = 0; i < n; i++) {
                    float dot = 0;
                    dot += v_work[i * n + (step + 1)] * u1;
                    for (uint j = step + 2; j < n; j++) {
                        dot += v_work[i * n + j] * a[step * n + j];
                    }
                    dot *= tau;
                    
                    v_work[i * n + (step + 1)] -= dot * u1;
                    for (uint j = step + 2; j < n; j++) {
                        v_work[i * n + j] -= dot * a[step * n + j];
                    }
                }
                
                a[step * n + (step + 1)] = -sign * norm;
                for (uint j = step + 2; j < n; j++) {
                    a[step * n + j] = 0;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Extract bidiagonal elements
    if (tid < k) {
        diag[tid] = a[tid * n + tid];
        if (tid < k - 1) {
            super_diag[tid] = a[tid * n + (tid + 1)];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // QR iteration on bidiagonal matrix
    for (uint iter = 0; iter < 30 * k; iter++) {
        // Check convergence
        bool converged = true;
        if (tid == 0) {
            for (uint i = 0; i < k - 1; i++) {
                if (abs(super_diag[i]) > 1e-10f * (abs(diag[i]) + abs(diag[i + 1]))) {
                    converged = false;
                    break;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (converged) break;
        
        // Find subproblem
        uint start = 0, end = k - 1;
        if (tid == 0) {
            for (int i = k - 2; i >= 0; i--) {
                if (abs(super_diag[i]) <= 1e-10f * (abs(diag[i]) + abs(diag[i + 1]))) {
                    start = i + 1;
                    break;
                }
            }
            
            // Handle 2x2 subproblems directly
            if (end - start == 1) {
                // 2x2 SVD
                float f = diag[start];
                float g = super_diag[start];
                float h = diag[end];
                
                float ft = f * f;
                float gt = g * g;
                float ht = h * h;
                
                float trace = ft + ht;
                float det = ft * ht - gt * gt;
                float disc = sqrt(max(0.0f, trace * trace - 4 * det));
                
                float s1 = sqrt((trace + disc) / 2);
                float s2 = sqrt((trace - disc) / 2);
                
                diag[start] = s1;
                diag[end] = s2;
                super_diag[start] = 0;
                
                // Compute rotations
                float theta = 0.5f * atan2(2 * g * h, ft - ht);
                float c = cos(theta);
                float s = sin(theta);
                
                // Update U and V
                for (uint i = 0; i < m; i++) {
                    float ui_start = u_work[i * m + start];
                    float ui_end = u_work[i * m + end];
                    u_work[i * m + start] = c * ui_start - s * ui_end;
                    u_work[i * m + end] = s * ui_start + c * ui_end;
                }
                
                for (uint i = 0; i < n; i++) {
                    float vi_start = v_work[i * n + start];
                    float vi_end = v_work[i * n + end];
                    v_work[i * n + start] = c * vi_start - s * vi_end;
                    v_work[i * n + end] = s * vi_start + c * vi_end;
                }
                
                continue;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Wilkinson shift for larger subproblems
        if (tid == 0 && end - start > 1) {
            // Compute shift from trailing 2x2 submatrix
            float d1 = diag[end - 1];
            float d2 = diag[end];
            float e = super_diag[end - 1];
            
            float a = d1 * d1 + ((end > start + 1) ? super_diag[end - 2] * super_diag[end - 2] : 0.0f);
            float b = d1 * e;
            float c = e * e + d2 * d2;
            
            float disc = sqrt(max(0.0f, (a - c) * (a - c) + 4 * b * b));
            float lambda = (a + c - disc) / 2;
            
            // QR step with shift
            float f = diag[start] * diag[start] - lambda;
            float g = diag[start] * super_diag[start];
            
            for (uint i = start; i < end; i++) {
                // Compute Givens rotation
                float r = sqrt(f * f + g * g);
                float c = f / r;
                float s = g / r;
                
                // Apply to bidiagonal matrix
                if (i > start) {
                    super_diag[i - 1] = r;
                }
                
                f = c * diag[i] + s * super_diag[i];
                super_diag[i] = -s * diag[i] + c * super_diag[i];
                g = s * diag[i + 1];
                diag[i + 1] = c * diag[i + 1];
                
                // Update V
                for (uint j = 0; j < n; j++) {
                    float vji = v_work[j * n + i];
                    float vji1 = v_work[j * n + (i + 1)];
                    v_work[j * n + i] = c * vji + s * vji1;
                    v_work[j * n + (i + 1)] = -s * vji + c * vji1;
                }
                
                // Second Givens rotation
                r = sqrt(f * f + g * g);
                c = f / r;
                s = g / r;
                
                diag[i] = r;
                f = c * super_diag[i] + s * diag[i + 1];
                diag[i + 1] = -s * super_diag[i] + c * diag[i + 1];
                
                if (i < end - 1) {
                    g = s * super_diag[i + 1];
                    super_diag[i + 1] = c * super_diag[i + 1];
                }
                super_diag[i] = f;
                
                // Update U
                for (uint j = 0; j < m; j++) {
                    float uji = u_work[j * m + i];
                    float uji1 = u_work[j * m + (i + 1)];
                    u_work[j * m + i] = c * uji + s * uji1;
                    u_work[j * m + (i + 1)] = -s * uji + c * uji1;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Make all singular values positive
    if (tid < k) {
        if (diag[tid] < 0) {
            diag[tid] = -diag[tid];
            for (uint i = 0; i < m; i++) {
                u_work[i * m + tid] = -u_work[i * m + tid];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Sort singular values in descending order
    if (tid == 0) {
        for (uint i = 0; i < k - 1; i++) {
            uint max_idx = i;
            float max_val = diag[i];
            
            for (uint j = i + 1; j < k; j++) {
                if (diag[j] > max_val) {
                    max_val = diag[j];
                    max_idx = j;
                }
            }
            
            if (max_idx != i) {
                // Swap singular values
                float temp = diag[i];
                diag[i] = diag[max_idx];
                diag[max_idx] = temp;
                
                // Swap U columns
                for (uint row = 0; row < m; row++) {
                    float temp = u_work[row * m + i];
                    u_work[row * m + i] = u_work[row * m + max_idx];
                    u_work[row * m + max_idx] = temp;
                }
                
                // Swap V columns
                for (uint row = 0; row < n; row++) {
                    float temp = v_work[row * n + i];
                    v_work[row * n + i] = v_work[row * n + max_idx];
                    v_work[row * n + max_idx] = temp;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Write results
    const uint u_cols = full_matrices ? m : k;
    const uint vh_rows = full_matrices ? n : k;
    const uint batch_offset_u = batch_idx * m * u_cols;
    const uint batch_offset_s = batch_idx * k;
    const uint batch_offset_vh = batch_idx * vh_rows * n;
    
    // Write U
    for (uint idx = tid; idx < m * u_cols; idx += tg_size) {
        uint i = idx / u_cols;
        uint j = idx % u_cols;
        u[batch_offset_u + idx] = u_work[i * m + j];
    }
    
    // Write singular values
    if (tid < k) {
        s[batch_offset_s + tid] = diag[tid];
    }
    
    // Write V^T
    for (uint idx = tid; idx < vh_rows * n; idx += tg_size) {
        uint i = idx / n;
        uint j = idx % n;
        vh[batch_offset_vh + idx] = v_work[j * n + i];
    }
}

// Complex SVD - implemented in linalg_complex_svd.metal

// ===== GENERAL EIGENVALUE DECOMPOSITION =====
// Implements Hessenberg reduction + Francis QR algorithm

kernel void eig_float(
    device const float* in [[buffer(0)]],
    device float2* eigenvalues [[buffer(1)]],
    device float2* eigenvectors [[buffer(2)]],
    constant uint4& params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    const uint batch_idx = gid;
    const uint batch_size = params.x;
    const uint n = params.y;
    const bool compute_vectors = params.z > 0;
    
    if (batch_idx >= batch_size || n > 64) return;
    
    const uint batch_offset = batch_idx * n * n;
    const uint batch_offset_eval = batch_idx * n;
    
    threadgroup float h[64 * 64];
    threadgroup float q[64 * 64];
    threadgroup float real_eval[64];
    threadgroup float imag_eval[64];
    
    // Copy input to H
    for (uint idx = tid; idx < n * n; idx += tg_size) {
        h[idx] = in[batch_offset + idx];
        if (compute_vectors) {
            uint i = idx / n;
            uint j = idx % n;
            q[idx] = (i == j) ? 1.0f : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Hessenberg reduction
    for (uint k = 0; k < n - 2; k++) {
        if (tid == 0) {
            // Find Householder vector
            float norm_sq = 0;
            for (uint i = k + 1; i < n; i++) {
                float val = h[i * n + k];
                norm_sq += val * val;
            }
            
            if (norm_sq > 1e-10f) {
                float norm = sqrt(norm_sq);
                float x0 = h[(k + 1) * n + k];
                float sign = (x0 >= 0) ? 1.0f : -1.0f;
                
                float u1 = x0 + sign * norm;
                float tau = 2.0f / (u1 * u1 + norm_sq - x0 * x0);
                
                // Apply from left: H = (I - tau*v*v^T) * H
                for (uint j = k; j < n; j++) {
                    float dot = 0;
                    dot += u1 * h[(k + 1) * n + j];
                    for (uint i = k + 2; i < n; i++) {
                        dot += h[i * n + k] * h[i * n + j];
                    }
                    dot *= tau;
                    
                    h[(k + 1) * n + j] -= u1 * dot;
                    for (uint i = k + 2; i < n; i++) {
                        h[i * n + j] -= h[i * n + k] * dot;
                    }
                }
                
                // Apply from right: H = H * (I - tau*v*v^T)
                for (uint i = 0; i < n; i++) {
                    float dot = 0;
                    dot += h[i * n + (k + 1)] * u1;
                    for (uint j = k + 2; j < n; j++) {
                        dot += h[i * n + j] * h[j * n + k];
                    }
                    dot *= tau;
                    
                    h[i * n + (k + 1)] -= dot * u1;
                    for (uint j = k + 2; j < n; j++) {
                        h[i * n + j] -= dot * h[j * n + k];
                    }
                }
                
                // Update Q if computing eigenvectors
                if (compute_vectors) {
                    for (uint i = 0; i < n; i++) {
                        float dot = 0;
                        dot += q[i * n + (k + 1)] * u1;
                        for (uint j = k + 2; j < n; j++) {
                            dot += q[i * n + j] * h[j * n + k];
                        }
                        dot *= tau;
                        
                        q[i * n + (k + 1)] -= dot * u1;
                        for (uint j = k + 2; j < n; j++) {
                            q[i * n + j] -= dot * h[j * n + k];
                        }
                    }
                }
                
                // Set subdiagonal element
                h[(k + 1) * n + k] = -sign * norm;
                for (uint i = k + 2; i < n; i++) {
                    h[i * n + k] = 0;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Francis QR algorithm on Hessenberg matrix
    const uint max_iterations = 30 * n;
    uint n_active = n;
    
    for (uint iter = 0; iter < max_iterations && n_active > 1; iter++) {
        // Check for convergence at bottom
        if (tid == 0) {
            for (int i = n_active - 1; i > 0; i--) {
                if (abs(h[i * n + (i - 1)]) <= 1e-10f * (abs(h[(i - 1) * n + (i - 1)]) + abs(h[i * n + i]))) {
                    h[i * n + (i - 1)] = 0;
                    
                    // Check if we have a converged eigenvalue
                    if (i == n_active - 1) {
                        real_eval[i] = h[i * n + i];
                        imag_eval[i] = 0;
                        n_active--;
                    } else if (i == n_active - 2) {
                        // 2x2 block at bottom
                        float a = h[(n_active - 2) * n + (n_active - 2)];
                        float b = h[(n_active - 2) * n + (n_active - 1)];
                        float c = h[(n_active - 1) * n + (n_active - 2)];
                        float d = h[(n_active - 1) * n + (n_active - 1)];
                        
                        float trace = a + d;
                        float det = a * d - b * c;
                        float disc = trace * trace - 4 * det;
                        
                        if (disc >= 0) {
                            float sqrt_disc = sqrt(disc);
                            real_eval[n_active - 2] = (trace + sqrt_disc) / 2;
                            real_eval[n_active - 1] = (trace - sqrt_disc) / 2;
                            imag_eval[n_active - 2] = 0;
                            imag_eval[n_active - 1] = 0;
                        } else {
                            real_eval[n_active - 2] = trace / 2;
                            real_eval[n_active - 1] = trace / 2;
                            imag_eval[n_active - 2] = sqrt(-disc) / 2;
                            imag_eval[n_active - 1] = -sqrt(-disc) / 2;
                        }
                        n_active -= 2;
                    }
                    break;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (n_active <= 1) break;
        
        // Wilkinson shift
        float shift = 0;
        if (tid == 0) {
            float a = h[(n_active - 2) * n + (n_active - 2)];
            float b = h[(n_active - 2) * n + (n_active - 1)];
            float c = h[(n_active - 1) * n + (n_active - 2)];
            float d = h[(n_active - 1) * n + (n_active - 1)];
            
            float trace = a + d;
            float det = a * d - b * c;
            float disc = trace * trace - 4 * det;
            
            float eval1 = (trace + sqrt(max(0.0f, disc))) / 2;
            float eval2 = (trace - sqrt(max(0.0f, disc))) / 2;
            
            shift = (abs(d - eval1) < abs(d - eval2)) ? eval1 : eval2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // QR step with shift
        if (tid == 0) {
            // First column of (H - shift*I)
            float x = h[0] - shift;
            float y = h[1 * n + 0];
            float z = (n_active > 2) ? h[2 * n + 0] : 0;
            
            for (uint k = 0; k < n_active - 1; k++) {
                // Determine rotation to eliminate y
                float r = sqrt(x * x + y * y);
                if (r > 1e-10f) {
                    float c = x / r;
                    float s = y / r;
                    
                    // Apply rotation from left
                    for (uint j = k; j < n; j++) {
                        float h_kj = h[k * n + j];
                        float h_k1j = h[(k + 1) * n + j];
                        h[k * n + j] = c * h_kj + s * h_k1j;
                        h[(k + 1) * n + j] = -s * h_kj + c * h_k1j;
                    }
                    
                    // Apply rotation from right
                    for (uint i = 0; i < min(k + 3, n_active); i++) {
                        float h_ik = h[i * n + k];
                        float h_ik1 = h[i * n + (k + 1)];
                        h[i * n + k] = c * h_ik + s * h_ik1;
                        h[i * n + (k + 1)] = -s * h_ik + c * h_ik1;
                    }
                    
                    // Update Q if needed
                    if (compute_vectors) {
                        for (uint i = 0; i < n; i++) {
                            float q_ik = q[i * n + k];
                            float q_ik1 = q[i * n + (k + 1)];
                            q[i * n + k] = c * q_ik + s * q_ik1;
                            q[i * n + (k + 1)] = -s * q_ik + c * q_ik1;
                        }
                    }
                }
                
                // Update x, y, z for next iteration
                if (k < n_active - 2) {
                    x = h[(k + 1) * n + k];
                    y = h[(k + 2) * n + k];
                    z = (k < n_active - 3) ? h[(k + 3) * n + k] : 0;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Handle remaining 1x1 block
    if (tid == 0 && n_active == 1) {
        real_eval[0] = h[0];
        imag_eval[0] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Write eigenvalues
    if (tid < n) {
        eigenvalues[batch_offset_eval + tid] = float2(real_eval[tid], imag_eval[tid]);
    }
    
    // Compute eigenvectors if requested
    if (compute_vectors) {
        // For real eigenvalues, use inverse iteration
        // For complex eigenvalues, eigenvectors will also be complex
        // This is a simplified version - full implementation would handle all cases
        for (uint idx = tid; idx < n * n; idx += tg_size) {
            eigenvectors[batch_offset + idx] = float2(q[idx], 0);
        }
    }
}

// Complex eigenvalue decomposition - implemented in linalg_complex_eig.metal

// Double precision stubs
// kernel void cholesky_double(...) removed - Metal doesn't support double precision on all devices
// kernel void qr_double(...) removed - Metal doesn't support double precision on all devices
// kernel void triangular_solve_double(...) removed - Metal doesn't support double precision on all devices
// kernel void eig_symmetric_double(...) removed - Metal doesn't support double precision on all devices
// kernel void svd_double(...) removed - Metal doesn't support double precision on all devices
// kernel void eig_double(...) removed - Metal doesn't support double precision on all devices