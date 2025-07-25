#include <metal_stdlib>
using namespace metal;

// Complex SVD implementation
// This is a separate file to keep the implementation organized

// Helper for 2x2 complex SVD
void svd_2x2_complex(float2 a11, float2 a12, float2 a21, float2 a22,
                     thread float2& u11, thread float2& u12, thread float2& u21, thread float2& u22,
                     thread float& s1, thread float& s2,
                     thread float2& v11, thread float2& v12, thread float2& v21, thread float2& v22) {
    // Compute A^H * A
    float2 aha11, aha12, aha22;
    
    // aha11 = conj(a11)*a11 + conj(a21)*a21
    aha11.x = a11.x * a11.x + a11.y * a11.y + a21.x * a21.x + a21.y * a21.y;
    aha11.y = 0;
    
    // aha12 = conj(a11)*a12 + conj(a21)*a22
    aha12.x = a11.x * a12.x + a11.y * a12.y + a21.x * a22.x + a21.y * a22.y;
    aha12.y = a11.y * a12.x - a11.x * a12.y + a21.y * a22.x - a21.x * a22.y;
    
    // aha22 = conj(a12)*a12 + conj(a22)*a22
    aha22.x = a12.x * a12.x + a12.y * a12.y + a22.x * a22.x + a22.y * a22.y;
    aha22.y = 0;
    
    // Eigenvalues of A^H * A (real and positive)
    float trace = aha11.x + aha22.x;
    float det = aha11.x * aha22.x - (aha12.x * aha12.x + aha12.y * aha12.y);
    float disc = sqrt(max(0.0f, trace * trace - 4 * det));
    
    float eval1 = (trace + disc) / 2;
    float eval2 = (trace - disc) / 2;
    
    s1 = sqrt(max(0.0f, eval1));
    s2 = sqrt(max(0.0f, eval2));
    
    // Compute V from eigenvectors of A^H * A
    float norm12 = sqrt(aha12.x * aha12.x + aha12.y * aha12.y);
    if (norm12 > 1e-10f) {
        // First eigenvector
        v11 = aha12;
        v21.x = eval1 - aha11.x;
        v21.y = -aha11.y;
        float norm = sqrt(v11.x * v11.x + v11.y * v11.y + v21.x * v21.x + v21.y * v21.y);
        v11.x /= norm; v11.y /= norm;
        v21.x /= norm; v21.y /= norm;
        
        // Second eigenvector
        v12 = aha12;
        v22.x = eval2 - aha11.x;
        v22.y = -aha11.y;
        norm = sqrt(v12.x * v12.x + v12.y * v12.y + v22.x * v22.x + v22.y * v22.y);
        v12.x /= norm; v12.y /= norm;
        v22.x /= norm; v22.y /= norm;
    } else {
        v11 = float2(1, 0); v12 = float2(0, 0);
        v21 = float2(0, 0); v22 = float2(1, 0);
    }
    
    // Compute U = A * V * S^(-1)
    if (s1 > 1e-10f) {
        // u1 = (a * v1) / s1
        float2 av1_1 = float2(a11.x * v11.x - a11.y * v11.y + a12.x * v21.x - a12.y * v21.y,
                             a11.x * v11.y + a11.y * v11.x + a12.x * v21.y + a12.y * v21.x);
        float2 av1_2 = float2(a21.x * v11.x - a21.y * v11.y + a22.x * v21.x - a22.y * v21.y,
                             a21.x * v11.y + a21.y * v11.x + a22.x * v21.y + a22.y * v21.x);
        u11.x = av1_1.x / s1; u11.y = av1_1.y / s1;
        u21.x = av1_2.x / s1; u21.y = av1_2.y / s1;
    } else {
        u11 = float2(1, 0); u21 = float2(0, 0);
    }
    
    if (s2 > 1e-10f) {
        // u2 = (a * v2) / s2
        float2 av2_1 = float2(a11.x * v12.x - a11.y * v12.y + a12.x * v22.x - a12.y * v22.y,
                             a11.x * v12.y + a11.y * v12.x + a12.x * v22.y + a12.y * v22.x);
        float2 av2_2 = float2(a21.x * v12.x - a21.y * v12.y + a22.x * v22.x - a22.y * v22.y,
                             a21.x * v12.y + a21.y * v12.x + a22.x * v22.y + a22.y * v22.x);
        u12.x = av2_1.x / s2; u12.y = av2_1.y / s2;
        u22.x = av2_2.x / s2; u22.y = av2_2.y / s2;
    } else {
        u12 = float2(0, 0); u22 = float2(1, 0);
    }
}

kernel void svd_float2(
    device const float2* in [[buffer(0)]],
    device float2* u [[buffer(1)]],
    device float* s [[buffer(2)]],
    device float2* vh [[buffer(3)]],
    constant uint4& params [[buffer(4)]], // batch_size, m, n, full_matrices
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
    
    threadgroup float2 a[64 * 64];
    threadgroup float2 u_work[64 * 64];
    threadgroup float2 v_work[64 * 64];
    threadgroup float diag[64];
    threadgroup float super_diag[64];
    threadgroup float2 householder_v[64];
    threadgroup float tau_real;
    
    // Initialize
    for (uint idx = tid; idx < m * n; idx += tg_size) {
        a[idx] = in[batch_offset_in + idx];
    }
    
    // Initialize U and V to identity
    for (uint idx = tid; idx < m * m; idx += tg_size) {
        uint i = idx / m;
        uint j = idx % m;
        u_work[idx] = (i == j) ? float2(1, 0) : float2(0, 0);
    }
    for (uint idx = tid; idx < n * n; idx += tg_size) {
        uint i = idx / n;
        uint j = idx % n;
        v_work[idx] = (i == j) ? float2(1, 0) : float2(0, 0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 1: Bidiagonalization using complex Householder reflections
    for (uint step = 0; step < k; step++) {
        // Left Householder (column)
        if (tid == 0 && step < m - 1) {
            float norm_sq = 0;
            for (uint i = step; i < m; i++) {
                float2 val = a[i * n + step];
                norm_sq += val.x * val.x + val.y * val.y;
            }
            
            if (norm_sq > 1e-10f) {
                float norm = sqrt(norm_sq);
                float2 x0 = a[step * n + step];
                
                // Compute phase
                float phase = atan2(x0.y, x0.x);
                float2 phase_factor = float2(cos(phase), sin(phase));
                
                // v[0] = x0 + norm * e^(i*phase)
                householder_v[step] = x0 + float2(norm * phase_factor.x, norm * phase_factor.y);
                
                // Copy rest of column
                for (uint i = step + 1; i < m; i++) {
                    householder_v[i] = a[i * n + step];
                }
                
                // Compute tau
                float v_norm_sq = 0;
                for (uint i = step; i < m; i++) {
                    v_norm_sq += householder_v[i].x * householder_v[i].x + 
                                householder_v[i].y * householder_v[i].y;
                }
                tau_real = 2.0f / v_norm_sq;
                
                // Apply to remaining columns
                for (uint j = step; j < n; j++) {
                    float2 dot = float2(0, 0);
                    for (uint i = step; i < m; i++) {
                        // conj(v[i]) * a[i,j]
                        float2 vi = householder_v[i];
                        float2 aij = a[i * n + j];
                        dot.x += vi.x * aij.x + vi.y * aij.y;
                        dot.y += vi.x * aij.y - vi.y * aij.x;
                    }
                    dot.x *= tau_real;
                    dot.y *= tau_real;
                    
                    for (uint i = step; i < m; i++) {
                        float2 vi = householder_v[i];
                        // a[i,j] -= vi * dot
                        a[i * n + j].x -= vi.x * dot.x - vi.y * dot.y;
                        a[i * n + j].y -= vi.x * dot.y + vi.y * dot.x;
                    }
                }
                
                // Update U
                for (uint j = 0; j < m; j++) {
                    float2 dot = float2(0, 0);
                    for (uint i = step; i < m; i++) {
                        float2 vi = householder_v[i];
                        float2 uij = u_work[i * m + j];
                        // conj(vi) * u[i,j]
                        dot.x += vi.x * uij.x + vi.y * uij.y;
                        dot.y += vi.x * uij.y - vi.y * uij.x;
                    }
                    dot.x *= tau_real;
                    dot.y *= tau_real;
                    
                    for (uint i = step; i < m; i++) {
                        float2 vi = householder_v[i];
                        // u[i,j] -= vi * dot
                        u_work[i * m + j].x -= vi.x * dot.x - vi.y * dot.y;
                        u_work[i * m + j].y -= vi.x * dot.y + vi.y * dot.x;
                    }
                }
                
                // Set final value
                a[step * n + step] = float2(-norm * phase_factor.x, -norm * phase_factor.y);
                for (uint i = step + 1; i < m; i++) {
                    a[i * n + step] = float2(0, 0);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Right Householder (row)
        if (tid == 0 && step < n - 2 && step < m) {
            float norm_sq = 0;
            for (uint j = step + 1; j < n; j++) {
                float2 val = a[step * n + j];
                norm_sq += val.x * val.x + val.y * val.y;
            }
            
            if (norm_sq > 1e-10f) {
                float norm = sqrt(norm_sq);
                float2 x0 = a[step * n + (step + 1)];
                
                // Compute phase
                float phase = atan2(x0.y, x0.x);
                float2 phase_factor = float2(cos(phase), sin(phase));
                
                // Store Householder vector
                householder_v[step + 1] = x0 + float2(norm * phase_factor.x, norm * phase_factor.y);
                for (uint j = step + 2; j < n; j++) {
                    householder_v[j] = a[step * n + j];
                }
                
                // Compute tau
                float v_norm_sq = 0;
                for (uint j = step + 1; j < n; j++) {
                    v_norm_sq += householder_v[j].x * householder_v[j].x + 
                                householder_v[j].y * householder_v[j].y;
                }
                tau_real = 2.0f / v_norm_sq;
                
                // Apply to remaining rows
                for (uint i = step; i < m; i++) {
                    float2 dot = float2(0, 0);
                    for (uint j = step + 1; j < n; j++) {
                        // a[i,j] * conj(v[j])
                        float2 aij = a[i * n + j];
                        float2 vj = householder_v[j];
                        dot.x += aij.x * vj.x + aij.y * vj.y;
                        dot.y += aij.y * vj.x - aij.x * vj.y;
                    }
                    dot.x *= tau_real;
                    dot.y *= tau_real;
                    
                    for (uint j = step + 1; j < n; j++) {
                        float2 vj = householder_v[j];
                        // a[i,j] -= dot * vj
                        a[i * n + j].x -= dot.x * vj.x - dot.y * vj.y;
                        a[i * n + j].y -= dot.x * vj.y + dot.y * vj.x;
                    }
                }
                
                // Update V
                for (uint i = 0; i < n; i++) {
                    float2 dot = float2(0, 0);
                    for (uint j = step + 1; j < n; j++) {
                        float2 vij = v_work[i * n + j];
                        float2 vj = householder_v[j];
                        // v[i,j] * conj(vj)
                        dot.x += vij.x * vj.x + vij.y * vj.y;
                        dot.y += vij.y * vj.x - vij.x * vj.y;
                    }
                    dot.x *= tau_real;
                    dot.y *= tau_real;
                    
                    for (uint j = step + 1; j < n; j++) {
                        float2 vj = householder_v[j];
                        // v[i,j] -= dot * vj
                        v_work[i * n + j].x -= dot.x * vj.x - dot.y * vj.y;
                        v_work[i * n + j].y -= dot.x * vj.y + dot.y * vj.x;
                    }
                }
                
                // Set final value
                a[step * n + (step + 1)] = float2(-norm * phase_factor.x, -norm * phase_factor.y);
                for (uint j = step + 2; j < n; j++) {
                    a[step * n + j] = float2(0, 0);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Extract bidiagonal elements (real parts only, as imaginary parts should be ~0)
    if (tid < k) {
        float2 diag_complex = a[tid * n + tid];
        diag[tid] = sqrt(diag_complex.x * diag_complex.x + diag_complex.y * diag_complex.y);
        
        if (tid < k - 1) {
            float2 super_complex = a[tid * n + (tid + 1)];
            super_diag[tid] = sqrt(super_complex.x * super_complex.x + super_complex.y * super_complex.y);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 2: QR iteration on real bidiagonal matrix
    // (Same as real SVD since bidiagonal is now real)
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
            // Find largest diagonal block
            for (int i = k - 2; i >= 0; i--) {
                if (abs(super_diag[i]) <= 1e-10f * (abs(diag[i]) + abs(diag[i + 1]))) {
                    start = i + 1;
                    break;
                }
            }
            
            // Handle 2x2 subproblems
            if (end - start == 1) {
                // For complex, we need to handle the phase correctly
                float2 u11, u12, u21, u22, v11, v12, v21, v22;
                float s1, s2;
                
                // Create effective 2x2 complex matrix from bidiagonal
                float2 b11 = float2(diag[start], 0);
                float2 b12 = float2(super_diag[start], 0);
                float2 b21 = float2(0, 0);
                float2 b22 = float2(diag[end], 0);
                
                svd_2x2_complex(b11, b12, b21, b22,
                              u11, u12, u21, u22, s1, s2,
                              v11, v12, v21, v22);
                
                diag[start] = s1;
                diag[end] = s2;
                super_diag[start] = 0;
                
                // Update U and V with complex rotations
                for (uint i = 0; i < m; i++) {
                    float2 ui_start = u_work[i * m + start];
                    float2 ui_end = u_work[i * m + end];
                    
                    float2 new_start, new_end;
                    // u[i,:] = u[i,:] * [u11 u12; u21 u22]
                    new_start.x = ui_start.x * u11.x - ui_start.y * u11.y + ui_end.x * u12.x - ui_end.y * u12.y;
                    new_start.y = ui_start.x * u11.y + ui_start.y * u11.x + ui_end.x * u12.y + ui_end.y * u12.x;
                    new_end.x = ui_start.x * u21.x - ui_start.y * u21.y + ui_end.x * u22.x - ui_end.y * u22.y;
                    new_end.y = ui_start.x * u21.y + ui_start.y * u21.x + ui_end.x * u22.y + ui_end.y * u22.x;
                    
                    u_work[i * m + start] = new_start;
                    u_work[i * m + end] = new_end;
                }
                
                for (uint i = 0; i < n; i++) {
                    float2 vi_start = v_work[i * n + start];
                    float2 vi_end = v_work[i * n + end];
                    
                    float2 new_start, new_end;
                    // v[i,:] = v[i,:] * [v11 v12; v21 v22]
                    new_start.x = vi_start.x * v11.x - vi_start.y * v11.y + vi_end.x * v12.x - vi_end.y * v12.y;
                    new_start.y = vi_start.x * v11.y + vi_start.y * v11.x + vi_end.x * v12.y + vi_end.y * v12.x;
                    new_end.x = vi_start.x * v21.x - vi_start.y * v21.y + vi_end.x * v22.x - vi_end.y * v22.y;
                    new_end.y = vi_start.x * v21.y + vi_start.y * v21.x + vi_end.x * v22.y + vi_end.y * v22.x;
                    
                    v_work[i * n + start] = new_start;
                    v_work[i * n + end] = new_end;
                }
                
                continue;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Wilkinson shift for larger subproblems (same as real case)
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
                
                // Update V (real rotations on complex matrices)
                for (uint j = 0; j < n; j++) {
                    float2 vji = v_work[j * n + i];
                    float2 vji1 = v_work[j * n + (i + 1)];
                    v_work[j * n + i] = float2(c * vji.x + s * vji1.x, c * vji.y + s * vji1.y);
                    v_work[j * n + (i + 1)] = float2(-s * vji.x + c * vji1.x, -s * vji.y + c * vji1.y);
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
                    float2 uji = u_work[j * m + i];
                    float2 uji1 = u_work[j * m + (i + 1)];
                    u_work[j * m + i] = float2(c * uji.x + s * uji1.x, c * uji.y + s * uji1.y);
                    u_work[j * m + (i + 1)] = float2(-s * uji.x + c * uji1.x, -s * uji.y + c * uji1.y);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Make all singular values positive
    if (tid < k) {
        if (diag[tid] < 0) {
            diag[tid] = -diag[tid];
            // Flip corresponding column of U
            for (uint i = 0; i < m; i++) {
                u_work[i * m + tid].x = -u_work[i * m + tid].x;
                u_work[i * m + tid].y = -u_work[i * m + tid].y;
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
                    float2 temp_c = u_work[row * m + i];
                    u_work[row * m + i] = u_work[row * m + max_idx];
                    u_work[row * m + max_idx] = temp_c;
                }
                
                // Swap V columns
                for (uint row = 0; row < n; row++) {
                    float2 temp_c = v_work[row * n + i];
                    v_work[row * n + i] = v_work[row * n + max_idx];
                    v_work[row * n + max_idx] = temp_c;
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
    
    // Write V^H (conjugate transpose of V)
    for (uint idx = tid; idx < vh_rows * n; idx += tg_size) {
        uint i = idx / n;
        uint j = idx % n;
        float2 v_val = v_work[j * n + i];
        // Conjugate
        vh[batch_offset_vh + idx] = float2(v_val.x, -v_val.y);
    }
}