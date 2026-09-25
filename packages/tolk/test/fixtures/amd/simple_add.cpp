extern "C" __attribute__((global)) void simple_add(int* out, const int* a, const int* b, int n) {
  int i = __builtin_amdgcn_workgroup_id_x() * 64 + __builtin_amdgcn_workitem_id_x();
  if (i < n) out[i] = a[i] + b[i];
}
