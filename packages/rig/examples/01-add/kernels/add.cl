/* out[i] = a[i] + b[i] for each work-item i of the grid below n. Its group
   size comes from the implicit arguments the dispatch writes. */

kernel void add(global const float *a, global const float *b,
                global float *out, ulong n) {
  ulong i = (ulong)__builtin_amdgcn_workgroup_id_x() *
                __builtin_amdgcn_workgroup_size_x() +
            __builtin_amdgcn_workitem_id_x();
  if (i < n) out[i] = a[i] + b[i];
}
