/* s_ttracedata and s_ttracedata_imm, eight of each a wave: their user data
   packets, 0x06 and 0x46, among the waves' starts and ends. */

#define GID \
  ((int)__builtin_amdgcn_workgroup_id_x() * 64 + (int)__builtin_amdgcn_workitem_id_x())
#define REP8(x) x x x x x x x x

__kernel void k_ttrace(__global int *p) {
  __asm__ volatile(REP8("s_mov_b32 m0, 0x1234\ns_ttracedata\n"));
  __asm__ volatile(REP8("s_ttracedata_imm 0x55\n"));
  p[GID] = 1;
}
