#include <metal_stdlib>
using namespace metal;

// Complex eigenvalue decomposition implementation
kernel void eig_float2(
    device const float2* in [[buffer(0)]],
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
    
    threadgroup float2 h[64 * 64];
    threadgroup float2 q[64 * 64];
    threadgroup float2 eigenvals[64];
    
    // Copy input to H
    for (uint idx = tid; idx < n * n; idx += tg_size) {
        h[idx] = in[batch_offset + idx];
        if (compute_vectors) {
            uint i = idx / n;
            uint j = idx % n;
            q[idx] = (i == j) ? float2(1, 0) : float2(0, 0);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Complex Hessenberg reduction
    for (uint k = 0; k < n - 2; k++) {
        if (tid == 0) {
            // Compute Householder vector
            float norm_sq = 0;
            for (uint i = k + 1; i < n; i++) {
                float2 val = h[i * n + k];
                norm_sq += val.x * val.x + val.y * val.y;
            }
            
            if (norm_sq > 1e-10f) {
                float norm = sqrt(norm_sq);
                float2 x0 = h[(k + 1) * n + k];
                
                // Phase of x0
                float phase = atan2(x0.y, x0.x);
                float2 phase_factor = float2(cos(phase), sin(phase));
                
                // v[0] = x0 + norm * e^(i*phase)
                float2 u1 = x0 + float2(norm * phase_factor.x, norm * phase_factor.y);
                
                // Compute tau
                float v_norm_sq = u1.x * u1.x + u1.y * u1.y + norm_sq - (x0.x * x0.x + x0.y * x0.y);
                float tau_real = 2.0f / v_norm_sq;
                
                // Apply from left: H = (I - tau*v*v^H) * H
                for (uint j = k; j < n; j++) {
                    float2 dot = float2(0, 0);
                    
                    // conj(u1) * h[(k+1),j]
                    float2 h_k1j = h[(k + 1) * n + j];
                    dot.x += u1.x * h_k1j.x + u1.y * h_k1j.y;
                    dot.y += u1.x * h_k1j.y - u1.y * h_k1j.x;
                    
                    for (uint i = k + 2; i < n; i++) {
                        float2 vi = h[i * n + k];
                        float2 hij = h[i * n + j];
                        // conj(vi) * hij
                        dot.x += vi.x * hij.x + vi.y * hij.y;
                        dot.y += vi.x * hij.y - vi.y * hij.x;
                    }
                    
                    dot.x *= tau_real;
                    dot.y *= tau_real;
                    
                    // h[(k+1),j] -= u1 * dot
                    h[(k + 1) * n + j].x -= u1.x * dot.x - u1.y * dot.y;
                    h[(k + 1) * n + j].y -= u1.x * dot.y + u1.y * dot.x;
                    
                    for (uint i = k + 2; i < n; i++) {
                        float2 vi = h[i * n + k];
                        // h[i,j] -= vi * dot
                        h[i * n + j].x -= vi.x * dot.x - vi.y * dot.y;
                        h[i * n + j].y -= vi.x * dot.y + vi.y * dot.x;
                    }
                }
                
                // Apply from right: H = H * (I - tau*v*v^H)
                for (uint i = 0; i < n; i++) {
                    float2 dot = float2(0, 0);
                    
                    // h[i,(k+1)] * conj(u1)
                    float2 h_ik1 = h[i * n + (k + 1)];
                    dot.x += h_ik1.x * u1.x + h_ik1.y * u1.y;
                    dot.y += h_ik1.y * u1.x - h_ik1.x * u1.y;
                    
                    for (uint j = k + 2; j < n; j++) {
                        float2 hij = h[i * n + j];
                        float2 vj = h[j * n + k];
                        // hij * conj(vj)
                        dot.x += hij.x * vj.x + hij.y * vj.y;
                        dot.y += hij.y * vj.x - hij.x * vj.y;
                    }
                    
                    dot.x *= tau_real;
                    dot.y *= tau_real;
                    
                    // h[i,(k+1)] -= dot * u1
                    h[i * n + (k + 1)].x -= dot.x * u1.x - dot.y * u1.y;
                    h[i * n + (k + 1)].y -= dot.x * u1.y + dot.y * u1.x;
                    
                    for (uint j = k + 2; j < n; j++) {
                        float2 vj = h[j * n + k];
                        // h[i,j] -= dot * vj
                        h[i * n + j].x -= dot.x * vj.x - dot.y * vj.y;
                        h[i * n + j].y -= dot.x * vj.y + dot.y * vj.x;
                    }
                }
                
                // Update Q if computing eigenvectors
                if (compute_vectors) {
                    for (uint i = 0; i < n; i++) {
                        float2 dot = float2(0, 0);
                        
                        float2 q_ik1 = q[i * n + (k + 1)];
                        dot.x += q_ik1.x * u1.x + q_ik1.y * u1.y;
                        dot.y += q_ik1.y * u1.x - q_ik1.x * u1.y;
                        
                        for (uint j = k + 2; j < n; j++) {
                            float2 qij = q[i * n + j];
                            float2 vj = h[j * n + k];
                            dot.x += qij.x * vj.x + qij.y * vj.y;
                            dot.y += qij.y * vj.x - qij.x * vj.y;
                        }
                        
                        dot.x *= tau_real;
                        dot.y *= tau_real;
                        
                        q[i * n + (k + 1)].x -= dot.x * u1.x - dot.y * u1.y;
                        q[i * n + (k + 1)].y -= dot.x * u1.y + dot.y * u1.x;
                        
                        for (uint j = k + 2; j < n; j++) {
                            float2 vj = h[j * n + k];
                            q[i * n + j].x -= dot.x * vj.x - dot.y * vj.y;
                            q[i * n + j].y -= dot.x * vj.y + dot.y * vj.x;
                        }
                    }
                }
                
                // Set subdiagonal element
                h[(k + 1) * n + k] = float2(-norm * phase_factor.x, -norm * phase_factor.y);
                for (uint i = k + 2; i < n; i++) {
                    h[i * n + k] = float2(0, 0);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Complex Francis QR algorithm on Hessenberg matrix
    const uint max_iterations = 30 * n;
    uint n_active = n;
    
    for (uint iter = 0; iter < max_iterations && n_active > 1; iter++) {
        // Check for convergence at bottom
        if (tid == 0) {
            for (int i = n_active - 1; i > 0; i--) {
                float2 subdiag = h[i * n + (i - 1)];
                float subdiag_norm = sqrt(subdiag.x * subdiag.x + subdiag.y * subdiag.y);
                
                float2 diag1 = h[(i - 1) * n + (i - 1)];
                float2 diag2 = h[i * n + i];
                float diag_norm = sqrt(diag1.x * diag1.x + diag1.y * diag1.y) + 
                                 sqrt(diag2.x * diag2.x + diag2.y * diag2.y);
                
                if (subdiag_norm <= 1e-10f * diag_norm) {
                    h[i * n + (i - 1)] = float2(0, 0);
                    
                    if (i == n_active - 1) {
                        // Converged eigenvalue
                        eigenvals[i] = h[i * n + i];
                        n_active--;
                    }
                    break;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (n_active <= 1) break;
        
        // Wilkinson shift (complex version)
        float2 shift = float2(0, 0);
        if (tid == 0) {
            // Compute shift from trailing 2x2 submatrix
            float2 a = h[(n_active - 2) * n + (n_active - 2)];
            float2 b = h[(n_active - 2) * n + (n_active - 1)];
            float2 c = h[(n_active - 1) * n + (n_active - 2)];
            float2 d = h[(n_active - 1) * n + (n_active - 1)];
            
            // Trace and determinant
            float2 trace = float2(a.x + d.x, a.y + d.y);
            float2 det;
            det.x = a.x * d.x - a.y * d.y - (b.x * c.x - b.y * c.y);
            det.y = a.x * d.y + a.y * d.x - (b.x * c.y + b.y * c.x);
            
            // Discriminant = trace^2 - 4*det
            float2 disc;
            disc.x = trace.x * trace.x - trace.y * trace.y - 4 * det.x;
            disc.y = 2 * trace.x * trace.y - 4 * det.y;
            
            // sqrt(disc)
            float disc_abs = sqrt(disc.x * disc.x + disc.y * disc.y);
            float disc_phase = atan2(disc.y, disc.x) / 2;
            float2 sqrt_disc = float2(sqrt(disc_abs) * cos(disc_phase), 
                                    sqrt(disc_abs) * sin(disc_phase));
            
            // Eigenvalues = (trace +- sqrt_disc) / 2
            float2 eval1 = float2((trace.x + sqrt_disc.x) / 2, (trace.y + sqrt_disc.y) / 2);
            float2 eval2 = float2((trace.x - sqrt_disc.x) / 2, (trace.y - sqrt_disc.y) / 2);
            
            // Choose closer eigenvalue
            float2 diff1 = float2(d.x - eval1.x, d.y - eval1.y);
            float2 diff2 = float2(d.x - eval2.x, d.y - eval2.y);
            float dist1 = diff1.x * diff1.x + diff1.y * diff1.y;
            float dist2 = diff2.x * diff2.x + diff2.y * diff2.y;
            
            shift = (dist1 < dist2) ? eval1 : eval2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // QR step with shift
        if (tid == 0) {
            // First column of (H - shift*I)
            float2 x = float2(h[0].x - shift.x, h[0].y - shift.y);
            float2 y = h[1 * n + 0];
            
            for (uint k = 0; k < n_active - 1; k++) {
                // Givens rotation to eliminate y
                float norm_x = sqrt(x.x * x.x + x.y * x.y);
                float norm_y = sqrt(y.x * y.x + y.y * y.y);
                float r = sqrt(norm_x * norm_x + norm_y * norm_y);
                
                if (r > 1e-10f) {
                    // Complex Givens rotation
                    float2 c = float2(norm_x / r, 0);
                    float2 s;
                    if (norm_x > 1e-10f) {
                        // s = conj(x)/|x| * y/r
                        float2 x_normalized = float2(x.x / norm_x, -x.y / norm_x);
                        s.x = (x_normalized.x * y.x - x_normalized.y * y.y) / r;
                        s.y = (x_normalized.x * y.y + x_normalized.y * y.x) / r;
                    } else {
                        s = float2(1, 0);
                    }
                    
                    // Apply rotation from left
                    for (uint j = k; j < n; j++) {
                        float2 h_kj = h[k * n + j];
                        float2 h_k1j = h[(k + 1) * n + j];
                        
                        // [c s; -conj(s) c] * [h_kj; h_k1j]
                        float2 new_hkj, new_hk1j;
                        new_hkj.x = c.x * h_kj.x + s.x * h_k1j.x - s.y * h_k1j.y;
                        new_hkj.y = c.x * h_kj.y + s.x * h_k1j.y + s.y * h_k1j.x;
                        new_hk1j.x = -s.x * h_kj.x - s.y * h_kj.y + c.x * h_k1j.x;
                        new_hk1j.y = s.x * h_kj.y - s.y * h_kj.x + c.x * h_k1j.y;
                        
                        h[k * n + j] = new_hkj;
                        h[(k + 1) * n + j] = new_hk1j;
                    }
                    
                    // Apply rotation from right
                    for (uint i = 0; i < min(k + 3, n_active); i++) {
                        float2 h_ik = h[i * n + k];
                        float2 h_ik1 = h[i * n + (k + 1)];
                        
                        // [h_ik h_ik1] * [c -s; conj(s) c]
                        float2 new_hik, new_hik1;
                        new_hik.x = h_ik.x * c.x - h_ik1.x * s.x - h_ik1.y * s.y;
                        new_hik.y = h_ik.y * c.x - h_ik1.y * s.x + h_ik1.x * s.y;
                        new_hik1.x = h_ik.x * s.x - h_ik.y * s.y + h_ik1.x * c.x;
                        new_hik1.y = h_ik.x * s.y + h_ik.y * s.x + h_ik1.y * c.x;
                        
                        h[i * n + k] = new_hik;
                        h[i * n + (k + 1)] = new_hik1;
                    }
                    
                    // Update Q if needed
                    if (compute_vectors) {
                        for (uint i = 0; i < n; i++) {
                            float2 q_ik = q[i * n + k];
                            float2 q_ik1 = q[i * n + (k + 1)];
                            
                            float2 new_qik, new_qik1;
                            new_qik.x = q_ik.x * c.x - q_ik1.x * s.x - q_ik1.y * s.y;
                            new_qik.y = q_ik.y * c.x - q_ik1.y * s.x + q_ik1.x * s.y;
                            new_qik1.x = q_ik.x * s.x - q_ik.y * s.y + q_ik1.x * c.x;
                            new_qik1.y = q_ik.x * s.y + q_ik.y * s.x + q_ik1.y * c.x;
                            
                            q[i * n + k] = new_qik;
                            q[i * n + (k + 1)] = new_qik1;
                        }
                    }
                }
                
                // Update x, y for next iteration
                if (k < n_active - 2) {
                    x = h[(k + 1) * n + k];
                    y = h[(k + 2) * n + k];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Handle remaining 1x1 block
    if (tid == 0 && n_active == 1) {
        eigenvals[0] = h[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Write eigenvalues
    if (tid < n) {
        eigenvalues[batch_offset_eval + tid] = eigenvals[tid];
    }
    
    // Write eigenvectors
    if (compute_vectors) {
        for (uint idx = tid; idx < n * n; idx += tg_size) {
            eigenvectors[batch_offset + idx] = q[idx];
        }
    }
}

// Hermitian eigenvalue kernel
kernel void eig_symmetric_float2(
    device const float2* in [[buffer(0)]],
    device float* eigenvalues [[buffer(1)]],
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
    
    threadgroup float2 a[64 * 64];
    threadgroup float2 q[64 * 64];
    threadgroup float diag[64];
    threadgroup float off_diag[64];
    
    // Copy input
    for (uint idx = tid; idx < n * n; idx += tg_size) {
        a[idx] = in[batch_offset + idx];
        if (compute_vectors) {
            uint i = idx / n;
            uint j = idx % n;
            q[idx] = (i == j) ? float2(1, 0) : float2(0, 0);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Complex Householder tridiagonalization
    for (uint k = 0; k < n - 2; k++) {
        if (tid == 0) {
            float norm_sq = 0;
            for (uint i = k + 1; i < n; i++) {
                float2 val = a[i * n + k];
                norm_sq += val.x * val.x + val.y * val.y;
            }
            
            if (norm_sq > 1e-10f) {
                float norm = sqrt(norm_sq);
                float2 x0 = a[(k + 1) * n + k];
                
                // Phase
                float phase = atan2(x0.y, x0.x);
                float2 phase_factor = float2(cos(phase), sin(phase));
                
                // v[0] = x0 + norm * e^(i*phase)
                float2 u1 = x0 + float2(norm * phase_factor.x, norm * phase_factor.y);
                
                // Compute tau
                float v_norm_sq = u1.x * u1.x + u1.y * u1.y + norm_sq - (x0.x * x0.x + x0.y * x0.y);
                float tau_real = 2.0f / v_norm_sq;
                
                // Apply Householder: A = (I - tau*v*v^H) * A * (I - tau*v*v^H)
                // Since A is Hermitian, we only need to update the lower triangle
                
                // First compute w = tau * A * v
                for (uint i = k + 1; i < n; i++) {
                    for (uint j = k + 1; j <= i; j++) {
                        float2 aij = (i == j) ? a[i * n + j] : a[i * n + j];
                        float2 aji = (i == j) ? aij : float2(a[j * n + i].x, -a[j * n + i].y);
                        
                        // Compute updates
                        float2 vi = (i == k + 1) ? u1 : a[i * n + k];
                        float2 vj = (j == k + 1) ? u1 : a[j * n + k];
                        
                        // A[i,j] -= tau * (vi * conj(vj) * trace + ...)
                        float2 update;
                        update.x = tau_real * (vi.x * vj.x + vi.y * vj.y);
                        update.y = tau_real * (vi.y * vj.x - vi.x * vj.y);
                        
                        a[i * n + j].x -= update.x;
                        a[i * n + j].y -= update.y;
                        
                        if (i != j) {
                            a[j * n + i] = float2(a[i * n + j].x, -a[i * n + j].y);
                        }
                    }
                }
                
                // Update Q
                if (compute_vectors) {
                    for (uint i = 0; i < n; i++) {
                        float2 dot = float2(0, 0);
                        
                        float2 qi_k1 = q[i * n + (k + 1)];
                        dot.x += qi_k1.x * u1.x + qi_k1.y * u1.y;
                        dot.y += qi_k1.y * u1.x - qi_k1.x * u1.y;
                        
                        for (uint j = k + 2; j < n; j++) {
                            float2 qij = q[i * n + j];
                            float2 vj = a[j * n + k];
                            dot.x += qij.x * vj.x + qij.y * vj.y;
                            dot.y += qij.y * vj.x - qij.x * vj.y;
                        }
                        
                        dot.x *= tau_real;
                        dot.y *= tau_real;
                        
                        q[i * n + (k + 1)].x -= dot.x * u1.x - dot.y * u1.y;
                        q[i * n + (k + 1)].y -= dot.x * u1.y + dot.y * u1.x;
                        
                        for (uint j = k + 2; j < n; j++) {
                            float2 vj = a[j * n + k];
                            q[i * n + j].x -= dot.x * vj.x - dot.y * vj.y;
                            q[i * n + j].y -= dot.x * vj.y + dot.y * vj.x;
                        }
                    }
                }
                
                // Set final values
                a[(k + 1) * n + k] = float2(-norm, 0);
                a[k * n + (k + 1)] = float2(-norm, 0);
                for (uint i = k + 2; i < n; i++) {
                    a[i * n + k] = float2(0, 0);
                    a[k * n + i] = float2(0, 0);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Extract real tridiagonal matrix (diagonal elements are real for Hermitian matrices)
    if (tid < n) {
        diag[tid] = a[tid * n + tid].x;
        if (tid < n - 1) {
            float2 off_diag_complex = a[tid * n + (tid + 1)];
            off_diag[tid] = sqrt(off_diag_complex.x * off_diag_complex.x + 
                                off_diag_complex.y * off_diag_complex.y);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // QR algorithm on real tridiagonal matrix (same as real symmetric case)
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
                    
                    // Update eigenvectors (complex rotation on real rotation matrix)
                    if (compute_vectors) {
                        for (uint i = 0; i < n; i++) {
                            float2 qik = q[i * n + k];
                            float2 qik1 = q[i * n + (k + 1)];
                            q[i * n + k] = float2(c * qik.x - s * qik1.x, c * qik.y - s * qik1.y);
                            q[i * n + (k + 1)] = float2(s * qik.x + c * qik1.x, s * qik.y + c * qik1.y);
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
                        float2 temp_c = q[k * n + i];
                        q[k * n + i] = q[k * n + max_idx];
                        q[k * n + max_idx] = temp_c;
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