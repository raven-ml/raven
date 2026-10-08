/* kernels.cl's kernels, but for double_index, which computes 3 * i: a code
   object of the same size whose kernel of that name runs other code. */

#define LANE (__builtin_amdgcn_workgroup_id_x() * 64 + __builtin_amdgcn_workitem_id_x())

kernel void empty(void) {}

/* out[i] = 3 * i */
kernel void double_index(global uint *out) {
  uint i = LANE;
  out[i] = 3 * i;
}

/* Sleeps for about n * 8,000 cycles, then sets *flag to 1. */
kernel void spin(global volatile uint *flag, uint n) {
  for (uint i = 0; i < n; i++) __builtin_amdgcn_s_sleep(127);
  *flag = 1;
}

/* Stores to address 0, which no GPU maps: a page fault. */
kernel void wild(void) { *(volatile global uint *)0 = 1; }
