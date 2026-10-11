/*
 * Copyright 2012 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

struct common_firmware_header {
	uint32_t size_bytes; /* size of the entire header+image(s) in bytes */
	uint32_t header_size_bytes; /* size of just the header in bytes */
	uint16_t header_version_major; /* header version */
	uint16_t header_version_minor; /* header version */
	uint16_t ip_version_major; /* IP version */
	uint16_t ip_version_minor; /* IP version */
	uint32_t ucode_version;
	uint32_t ucode_size_bytes; /* size of ucode in bytes */
	uint32_t ucode_array_offset_bytes; /* payload offset from the start of the header */
	uint32_t crc32;  /* crc32 checksum of the payload */
};
struct smc_firmware_header_v1_0 {
	struct common_firmware_header header;
	uint32_t ucode_start_addr;
};
struct smc_soft_pptable_entry {
        uint32_t id;
        uint32_t ppt_offset_bytes;
        uint32_t ppt_size_bytes;
};
struct smc_firmware_header_v2_1 {
        struct smc_firmware_header_v1_0 v1_0;
        uint32_t pptable_count;
        uint32_t pptable_entry_offset;
};
struct psp_fw_bin_desc {
	uint32_t fw_type;
	uint32_t fw_version;
	uint32_t offset_bytes;
	uint32_t size_bytes;
};
enum psp_fw_type {
	PSP_FW_TYPE_UNKOWN,
	PSP_FW_TYPE_PSP_SOS,
	PSP_FW_TYPE_PSP_SYS_DRV,
	PSP_FW_TYPE_PSP_KDB,
	PSP_FW_TYPE_PSP_TOC,
	PSP_FW_TYPE_PSP_SPL,
	PSP_FW_TYPE_PSP_RL,
	PSP_FW_TYPE_PSP_SOC_DRV,
	PSP_FW_TYPE_PSP_INTF_DRV,
	PSP_FW_TYPE_PSP_DBG_DRV,
	PSP_FW_TYPE_PSP_RAS_DRV,
	PSP_FW_TYPE_PSP_IPKEYMGR_DRV,
	PSP_FW_TYPE_PSP_SPDM_DRV,
	PSP_FW_TYPE_MAX_INDEX,
};
struct psp_firmware_header_v2_0 {
	struct common_firmware_header header;
	uint32_t psp_fw_bin_count;
	struct psp_fw_bin_desc psp_fw_bin[];
};
struct psp_firmware_header_v2_1 {
	struct common_firmware_header header;
	uint32_t psp_fw_bin_count;
	uint32_t psp_aux_fw_bin_index;
	struct psp_fw_bin_desc psp_fw_bin[];
};
struct gfx_firmware_header_v1_0 {
	struct common_firmware_header header;
	uint32_t ucode_feature_version;
	uint32_t jt_offset; /* jt location */
	uint32_t jt_size;  /* size of jt */
};
struct gfx_firmware_header_v2_0 {
	struct common_firmware_header header;
	uint32_t ucode_feature_version;
	uint32_t ucode_size_bytes;
	uint32_t ucode_offset_bytes;
	uint32_t data_size_bytes;
	uint32_t data_offset_bytes;
	uint32_t ucode_start_addr_lo;
	uint32_t ucode_start_addr_hi;
};
struct rlc_firmware_header_v2_0 {
	struct common_firmware_header header;
	uint32_t ucode_feature_version;
	uint32_t jt_offset; /* jt location */
	uint32_t jt_size;  /* size of jt */
	uint32_t save_and_restore_offset;
	uint32_t clear_state_descriptor_offset;
	uint32_t avail_scratch_ram_locations;
	uint32_t reg_restore_list_size;
	uint32_t reg_list_format_start;
	uint32_t reg_list_format_separate_start;
	uint32_t starting_offsets_start;
	uint32_t reg_list_format_size_bytes; /* size of reg list format array in bytes */
	uint32_t reg_list_format_array_offset_bytes; /* payload offset from the start of the header */
	uint32_t reg_list_size_bytes; /* size of reg list array in bytes */
	uint32_t reg_list_array_offset_bytes; /* payload offset from the start of the header */
	uint32_t reg_list_format_separate_size_bytes; /* size of reg list format array in bytes */
	uint32_t reg_list_format_separate_array_offset_bytes; /* payload offset from the start of the header */
	uint32_t reg_list_separate_size_bytes; /* size of reg list array in bytes */
	uint32_t reg_list_separate_array_offset_bytes; /* payload offset from the start of the header */
};
struct rlc_firmware_header_v2_1 {
	struct rlc_firmware_header_v2_0 v2_0;
	uint32_t reg_list_format_direct_reg_list_length; /* length of direct reg list format array */
	uint32_t save_restore_list_cntl_ucode_ver;
	uint32_t save_restore_list_cntl_feature_ver;
	uint32_t save_restore_list_cntl_size_bytes;
	uint32_t save_restore_list_cntl_offset_bytes;
	uint32_t save_restore_list_gpm_ucode_ver;
	uint32_t save_restore_list_gpm_feature_ver;
	uint32_t save_restore_list_gpm_size_bytes;
	uint32_t save_restore_list_gpm_offset_bytes;
	uint32_t save_restore_list_srm_ucode_ver;
	uint32_t save_restore_list_srm_feature_ver;
	uint32_t save_restore_list_srm_size_bytes;
	uint32_t save_restore_list_srm_offset_bytes;
};
struct rlc_firmware_header_v2_2 {
	struct rlc_firmware_header_v2_1 v2_1;
	uint32_t rlc_iram_ucode_size_bytes;
	uint32_t rlc_iram_ucode_offset_bytes;
	uint32_t rlc_dram_ucode_size_bytes;
	uint32_t rlc_dram_ucode_offset_bytes;
};
struct rlc_firmware_header_v2_3 {
    struct rlc_firmware_header_v2_2 v2_2;
    uint32_t rlcp_ucode_version;
    uint32_t rlcp_ucode_feature_version;
    uint32_t rlcp_ucode_size_bytes;
    uint32_t rlcp_ucode_offset_bytes;
    uint32_t rlcv_ucode_version;
    uint32_t rlcv_ucode_feature_version;
    uint32_t rlcv_ucode_size_bytes;
    uint32_t rlcv_ucode_offset_bytes;
};
struct sdma_firmware_header_v2_0 {
	struct common_firmware_header header;
	uint32_t ucode_feature_version;
	uint32_t ctx_ucode_size_bytes; /* context thread ucode size */
	uint32_t ctx_jt_offset; /* context thread jt location */
	uint32_t ctx_jt_size; /* context thread size of jt */
	uint32_t ctl_ucode_offset;
	uint32_t ctl_ucode_size_bytes; /* control thread ucode size */
	uint32_t ctl_jt_offset; /* control thread jt location */
	uint32_t ctl_jt_size; /* control thread size of jt */
};
struct sdma_firmware_header_v3_0 {
    struct common_firmware_header header;
    uint32_t ucode_feature_version;
    uint32_t ucode_offset_bytes;
    uint32_t ucode_size_bytes;
};
struct imu_firmware_header_v1_0 {
    struct common_firmware_header header;
    uint32_t imu_iram_ucode_size_bytes;
    uint32_t imu_iram_ucode_offset_bytes;
    uint32_t imu_dram_ucode_size_bytes;
    uint32_t imu_dram_ucode_offset_bytes;
};
