/* SPDX-License-Identifier: MIT */

struct nvfw_hs_header_v2 {
	u32 sig_prod_offset;
	u32 sig_prod_size;
	u32 patch_loc;
	u32 patch_sig;
	u32 meta_data_offset;
	u32 meta_data_size;
	u32 num_sig;
	u32 header_offset;
	u32 header_size;
};

struct nvfw_hs_load_header_v2 {
	u32 os_code_offset;
	u32 os_code_size;
	u32 os_data_offset;
	u32 os_data_size;
	u32 num_apps;
	struct {
		u32 offset;
		u32 size;
	} app[];
};
