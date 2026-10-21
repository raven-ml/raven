/* SPDX-License-Identifier: MIT */

/* Excerpt of https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/drivers/gpu/drm/nouveau/include/nvfw/fw.h?h=v6.18. */

struct nvfw_bin_hdr {
	u32 bin_magic;
	u32 bin_ver;
	u32 bin_size;
	u32 header_offset;
	u32 data_offset;
	u32 data_size;
};
