/* out[i] = a[i] + b[i] for each work-item i of a grid of workgroups of 64.
   It reads no dispatch packet and no implicit argument: its three arguments
   are the addresses of the arrays. */

#define LANE \
  (__builtin_amdgcn_workgroup_id_x() * 64 + __builtin_amdgcn_workitem_id_x())

kernel void add(global const float *a, global const float *b,
                global float *out) {
  uint i = LANE;
  out[i] = a[i] + b[i];
}
