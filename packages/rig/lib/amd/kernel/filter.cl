/* SPDX-License-Identifier: ISC
 * Copyright (c) 2026 The Raven authors.
 *
 * One work-item stores at [w] the counter filter word of its own VMID
 * (Rig_amd_abi.Counter.filter): FORCE_EN, bit 0, and VMID_EN's bit for the
 * VMID, from bit 1. */

/* hwreg(HW_REG_WAVE_HW_ID2, 24, 4): the wave's VMID, bits 27 to 24. */
#define HW_REG_WAVE_HW_ID2 24
#define VMID (HW_REG_WAVE_HW_ID2 | (24 << 6) | ((4 - 1) << 11))

kernel void filter(global uint *w) {
  *w = 1u | (1u << (__builtin_amdgcn_s_getreg(VMID) + 1));
}
