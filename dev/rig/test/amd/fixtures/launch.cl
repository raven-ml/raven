/* The kernels the AMD suites launch. They learn their grid from the implicit
   arguments; only packet reads its dispatch packet. */

#define TX __builtin_amdgcn_workgroup_size_x()
#define TY __builtin_amdgcn_workgroup_size_y()
#define TZ __builtin_amdgcn_workgroup_size_z()

/* The conformance laws' ids (Rig_gpu_support.Conformance.launch_binary):
   out[k] = a + b * k + f for each work-item k of the grid, k = g * T + t, the
   group's index g and the item's index t in its group counted x fastest. */
kernel void ids(global uint *out, ulong a, uint b, float f) {
  const constant uint *blocks =
      (const constant uint *)__builtin_amdgcn_implicitarg_ptr();
  uint gx = blocks[0];
  uint gy = blocks[1];
  uint t = __builtin_amdgcn_workitem_id_x() +
           TX * (__builtin_amdgcn_workitem_id_y() +
                 TY * __builtin_amdgcn_workitem_id_z());
  uint g = __builtin_amdgcn_workgroup_id_x() +
           gx * (__builtin_amdgcn_workgroup_id_y() +
                 gy * __builtin_amdgcn_workgroup_id_z());
  uint k = g * TX * TY * TZ + t;
  out[k] = (uint)a + b * k + (uint)f;
}

/* The conformance laws' twice: dst[k] = 2 * src[k] + c over a grid of one
   dimension. */
kernel void twice(global uint *dst, global const uint *src, uint c) {
  uint k = __builtin_amdgcn_workgroup_id_x() * TX + __builtin_amdgcn_workitem_id_x();
  dst[k] = 2 * src[k] + c;
}

/* out[i] = i for the work-items i of a group of n, through n words of the
   dynamic LDS at tile, after the 64 words of LDS the kernel takes itself. */
kernel void lds(global uint *out, local uint *tile, uint n) {
  local uint own[64];
  uint l = __builtin_amdgcn_workitem_id_x();
  own[l % 64] = l;
  tile[n - 1 - l] = l;
  __builtin_amdgcn_s_barrier();
  out[l] = tile[n - 1 - l] + own[l % 64] - l;
}

/* out[i] = i * k for the 64 work-items i of a group, through 1024 words of
   scratch. */
kernel void scratch(global uint *out, uint k) {
  volatile uint p[1024];
  uint l = __builtin_amdgcn_workitem_id_x();
  for (uint i = 0; i < 1024; i++) p[(i * 7 + l) % 1024] = i * k;
  out[l] = p[(l * 7 + l) % 1024];
}

/* out[0] = the first word of its dispatch packet. */
kernel void packet(global uint *out) {
  out[0] = *(const constant uint *)__builtin_amdgcn_dispatch_ptr();
}

/* out[0] = the first word of its dispatch packet plus k, through 1024 words
   of scratch. */
kernel void packet_scratch(global uint *out, uint k) {
  volatile uint p[1024];
  for (uint i = 0; i < 1024; i++) p[i] = i * k;
  out[0] = *(const constant uint *)__builtin_amdgcn_dispatch_ptr() + p[k % 1024];
}
