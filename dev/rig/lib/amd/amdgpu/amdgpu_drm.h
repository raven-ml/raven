/* amdgpu_drm.h -- Public header for the amdgpu driver -*- linux-c -*-
 *
 * Copyright 2000 Precision Insight, Inc., Cedar Park, Texas.
 * Copyright 2000 VA Linux Systems, Inc., Fremont, California.
 * Copyright 2002 Tungsten Graphics, Inc., Cedar Park, Texas.
 * Copyright 2014 Advanced Micro Devices, Inc.
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
 * Authors:
 *    Kevin E. Martin <martin@valinux.com>
 *    Gareth Hughes <gareth@valinux.com>
 *    Keith Whitwell <keith@tungstengraphics.com>
 */

/* An excerpt of Linux's include/uapi/drm/amdgpu_drm.h (commit
   33970e1351f5e511029602454979f3de7e22260f): the render node's requests the
   amdgpu path makes, its INFO query of the device and its context requests,
   with the request numbers drm.h defines. */

#ifndef RIG_AMD_AMDGPU_DRM_H
#define RIG_AMD_AMDGPU_DRM_H

#include <linux/ioctl.h>
#include <linux/types.h>

#define DRM_IOCTL_BASE 'd'
#define DRM_IOW(nr, type) _IOW(DRM_IOCTL_BASE, nr, type)
#define DRM_IOWR(nr, type) _IOWR(DRM_IOCTL_BASE, nr, type)
#define DRM_COMMAND_BASE 0x40

#define DRM_AMDGPU_CTX			0x02
#define DRM_AMDGPU_INFO			0x05
#define DRM_IOCTL_AMDGPU_CTX		DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_CTX, union drm_amdgpu_ctx)
#define DRM_IOCTL_AMDGPU_INFO		DRM_IOW(DRM_COMMAND_BASE + DRM_AMDGPU_INFO, struct drm_amdgpu_info)

#define AMDGPU_CTX_OP_ALLOC_CTX	1
#define AMDGPU_CTX_OP_FREE_CTX	2
#define AMDGPU_CTX_OP_QUERY_STATE	3
#define AMDGPU_CTX_OP_QUERY_STATE2	4
#define AMDGPU_CTX_OP_GET_STABLE_PSTATE	5
#define AMDGPU_CTX_OP_SET_STABLE_PSTATE	6
#define AMDGPU_CTX_STABLE_PSTATE_FLAGS_MASK  0xf
#define AMDGPU_CTX_STABLE_PSTATE_NONE  0
#define AMDGPU_CTX_STABLE_PSTATE_STANDARD  1

struct drm_amdgpu_ctx_in {
	/** AMDGPU_CTX_OP_* */
	__u32	op;
	/** Flags */
	__u32	flags;
	__u32	ctx_id;
	/** AMDGPU_CTX_PRIORITY_* */
	__s32	priority;
};

union drm_amdgpu_ctx_out {
		struct {
			__u32	ctx_id;
			__u32	_pad;
		} alloc;

		struct {
			/** For future use, no flags defined so far */
			__u64	flags;
			/** Number of resets caused by this context so far. */
			__u32	hangs;
			/** Reset status since the last call of the ioctl. */
			__u32	reset_status;
		} state;

		struct {
			__u32	flags;
			__u32	_pad;
		} pstate;
};

union drm_amdgpu_ctx {
	struct drm_amdgpu_ctx_in in;
	union drm_amdgpu_ctx_out out;
};

#define AMDGPU_INFO_DEV_INFO			0x16

struct drm_amdgpu_query_fw {
	/** AMDGPU_INFO_FW_* */
	__u32 fw_type;
	/**
	 * Index of the IP if there are more IPs of
	 * the same type.
	 */
	__u32 ip_instance;
	/**
	 * Index of the engine. Whether this is used depends
	 * on the firmware type. (e.g. MEC, SDMA)
	 */
	__u32 index;
	__u32 _pad;
};

struct drm_amdgpu_info {
	/* Where the return value will be stored */
	__u64 return_pointer;
	/* The size of the return value. Just like "size" in "snprintf",
	 * it limits how many bytes the kernel can write. */
	__u32 return_size;
	/* The query request id. */
	__u32 query;

	union {
		struct {
			__u32 id;
			__u32 _pad;
		} mode_crtc;

		struct {
			/** AMDGPU_HW_IP_* */
			__u32 type;
			/**
			 * Index of the IP if there are more IPs of the same
			 * type. Ignored by AMDGPU_INFO_HW_IP_COUNT.
			 */
			__u32 ip_instance;
		} query_hw_ip;

		struct {
			__u32 dword_offset;
			/** number of registers to read */
			__u32 count;
			__u32 instance;
			/** For future use, no flags defined so far */
			__u32 flags;
		} read_mmr_reg;

		struct {
			uint32_t aperture;
			uint32_t _pad;
		} virtual_range;

		struct drm_amdgpu_query_fw query_fw;

		struct {
			__u32 type;
			__u32 offset;
		} vbios_info;

		struct {
			__u32 type;
		} sensor_info;

		struct {
			__u32 type;
		} video_cap;
	};
};

struct drm_amdgpu_info_device {
	/** PCI Device ID */
	__u32 device_id;
	/** Internal chip revision: A0, A1, etc.) */
	__u32 chip_rev;
	__u32 external_rev;
	/** Revision id in PCI Config space */
	__u32 pci_rev;
	__u32 family;
	__u32 num_shader_engines;
	__u32 num_shader_arrays_per_engine;
	/* in KHz */
	__u32 gpu_counter_freq;
	__u64 max_engine_clock;
	__u64 max_memory_clock;
	/* cu information */
	__u32 cu_active_number;
	/* NOTE: cu_ao_mask is INVALID, DON'T use it */
	__u32 cu_ao_mask;
	__u32 cu_bitmap[4][4];
	/** Render backend pipe mask. One render backend is CB+DB. */
	__u32 enabled_rb_pipes_mask;
	__u32 num_rb_pipes;
	__u32 num_hw_gfx_contexts;
	/* PCIe version (the smaller of the GPU and the CPU/motherboard) */
	__u32 pcie_gen;
	__u64 ids_flags;
	/** Starting virtual address for UMDs. */
	__u64 virtual_address_offset;
	/** The maximum virtual address */
	__u64 virtual_address_max;
	/** Required alignment of virtual addresses. */
	__u32 virtual_address_alignment;
	/** Page table entry - fragment size */
	__u32 pte_fragment_size;
	__u32 gart_page_size;
	/** constant engine ram size*/
	__u32 ce_ram_size;
	/** video memory type info*/
	__u32 vram_type;
	/** video memory bit width*/
	__u32 vram_bit_width;
	/* vce harvesting instance */
	__u32 vce_harvest_config;
	/* gfx double offchip LDS buffers */
	__u32 gc_double_offchip_lds_buf;
	/* NGG Primitive Buffer */
	__u64 prim_buf_gpu_addr;
	/* NGG Position Buffer */
	__u64 pos_buf_gpu_addr;
	/* NGG Control Sideband */
	__u64 cntl_sb_buf_gpu_addr;
	/* NGG Parameter Cache */
	__u64 param_buf_gpu_addr;
	__u32 prim_buf_size;
	__u32 pos_buf_size;
	__u32 cntl_sb_buf_size;
	__u32 param_buf_size;
	/* wavefront size*/
	__u32 wave_front_size;
	/* shader visible vgprs*/
	__u32 num_shader_visible_vgprs;
	/* CU per shader array*/
	__u32 num_cu_per_sh;
	/* number of tcc blocks*/
	__u32 num_tcc_blocks;
	/* gs vgt table depth*/
	__u32 gs_vgt_table_depth;
	/* gs primitive buffer depth*/
	__u32 gs_prim_buffer_depth;
	/* max gs wavefront per vgt*/
	__u32 max_gs_waves_per_vgt;
	/* PCIe number of lanes (the smaller of the GPU and the CPU/motherboard) */
	__u32 pcie_num_lanes;
	/* always on cu bitmap */
	__u32 cu_ao_bitmap[4][4];
	/** Starting high virtual address for UMDs. */
	__u64 high_va_offset;
	/** The maximum high virtual address */
	__u64 high_va_max;
	/* gfx10 pa_sc_tile_steering_override */
	__u32 pa_sc_tile_steering_override;
	/* disabled TCCs */
	__u64 tcc_disabled_mask;
	__u64 min_engine_clock;
	__u64 min_memory_clock;
	/* The following fields are only set on gfx11+, older chips set 0. */
	__u32 tcp_cache_size;       /* AKA GL0, VMEM cache */
	__u32 num_sqc_per_wgp;
	__u32 sqc_data_cache_size;  /* AKA SMEM cache */
	__u32 sqc_inst_cache_size;
	__u32 gl1c_cache_size;
	__u32 gl2c_cache_size;
	__u64 mall_size;            /* AKA infinity cache */
	/* high 32 bits of the rb pipes mask */
	__u32 enabled_rb_pipes_mask_hi;
	/* shadow area size for gfx11 */
	__u32 shadow_size;
	/* shadow area base virtual alignment for gfx11 */
	__u32 shadow_alignment;
	/* context save area size for gfx11 */
	__u32 csa_size;
	/* context save area base virtual alignment for gfx11 */
	__u32 csa_alignment;
	/* Userq IP mask (1 << AMDGPU_HW_IP_*) */
	__u32 userq_ip_mask;
	__u32 pad;
};

#endif
