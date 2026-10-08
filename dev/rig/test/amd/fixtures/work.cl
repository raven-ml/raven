/* The kernels the AMD suite's laws run, over workgroups of 64 work-items.
   None reads its dispatch packet or the implicit arguments. */

/* After about delay * 8,000 cycles, copies the n words at src to dst. */
kernel void copy(global uint *dst, global const uint *src, uint n,
                 uint delay) {
  for (uint i = 0; i < delay; i++) __builtin_amdgcn_s_sleep(127);
  for (uint i = __builtin_amdgcn_workitem_id_x(); i < n; i += 64)
    dst[i] = src[i];
}

/* out[i] = i, through 256 bytes of local data share. */
kernel void shared(global uint *out) {
  local uint tile[64];
  uint l = __builtin_amdgcn_workitem_id_x();
  tile[63 - l] = l;
  __builtin_amdgcn_s_barrier();
  out[l] = tile[63 - l];
}

/* out[i] = in[i] + 1 for i < n, over workgroups of 64. */
kernel void inc(global uint *out, global const uint *in, uint n) {
  uint i = __builtin_amdgcn_workgroup_id_x() * 64 + __builtin_amdgcn_workitem_id_x();
  if (i < n) out[i] = in[i] + 1;
}
