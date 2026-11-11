/* Kernels that read many launch arguments, for the AMD suites and bench. Each
   runs over workgroups of 32 work-items, one wave of gfx1201, and reads its
   arguments through the address of its argument segment. None reads its
   dispatch packet. */

/* The words after lap's header, which make its arguments 4 KiB. */
#define LAP_WORDS 1020

typedef struct {
  uint w[LAP_WORDS];
} lap_words;

typedef struct {
  uint w[62];
} scalar_words;

/* What a group stores for a word that differs, or-ed with its index, and
   for a grid that differs. */
#define LAP_WORD 0x80000000u
#define LAP_GRID 0xC0000000u

static inline uint lane(void) { return __builtin_amdgcn_workitem_id_x(); }

/* The least i below n whose word w[i] is not (seq << 12) | i, or n. Each
   work-item loads every 32nd word, all loads before any compare. */
static inline uint least_wrong(const constant uint *w, uint seq, uint n) {
  uint bad = n;
#pragma unroll
  for (uint k = 0; k < n; k += 32) {
    uint i = k + lane();
    uint x = w[i < n ? i : n - 1];
    bad = ((i < n) & (x != (seq << 12 | i)) & (bad == n)) ? i : bad;
  }
  uint least = n;
  for (uint k = 0; k < 32; k++) {
    uint b = __builtin_amdgcn_readlane(bad, k);
    least = b < least ? b : least;
  }
  return least;
}

/* Checks its 4 KiB of arguments, as vector loads at each work-item's own
   index, as a kernel stages a block of arguments: group g stores out[g],
   which is seq if each word p.w[i] is (seq << 12) | i and its grid, read from
   the implicit arguments, is groups groups of 32; LAP_GRID if the grid
   differs; else LAP_WORD | i for the least i whose word differs. */
kernel void lap(global uint *out, uint seq, uint groups, lap_words p) {
  const constant uint *w =
      (const constant uint *)__builtin_amdgcn_kernarg_segment_ptr() + 4;
  const constant uint *blocks =
      (const constant uint *)__builtin_amdgcn_implicitarg_ptr();
  uint least = least_wrong(w, seq, LAP_WORDS);
  if (lane() != 0) return;
  out[__builtin_amdgcn_workgroup_id_x()] =
      blocks[0] != groups || __builtin_amdgcn_workgroup_size_x() != 32
          ? LAP_GRID
      : least < LAP_WORDS ? LAP_WORD | least
                          : seq;
}

/* Group g stores out[g] = the sum of p's words, which its waves read by
   scalar loads, as a kernel reads its ordinary arguments: 256 bytes with
   out. */
kernel void scalars(global uint *out, scalar_words p) {
  const constant uint *w =
      (const constant uint *)__builtin_amdgcn_kernarg_segment_ptr() + 2;
  uint s = 0;
#pragma unroll
  for (uint i = 0; i < 62; i++) s += w[i];
  if (lane() == 0) out[__builtin_amdgcn_workgroup_id_x()] = s;
}
