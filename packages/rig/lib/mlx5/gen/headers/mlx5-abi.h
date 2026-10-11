/* SPDX-License-Identifier: ((GPL-2.0 WITH Linux-syscall-note) OR Linux-OpenIB) */
/*
 * Copyright (c) 2013-2015, Mellanox Technologies. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/* Excerpt of https://raw.githubusercontent.com/torvalds/linux/v6.12/include/uapi/rdma/mlx5-abi.h. */

enum mlx5_lib_caps {
	MLX5_LIB_CAP_4K_UAR	= (__u64)1 << 0,
	MLX5_LIB_CAP_DYN_UAR	= (__u64)1 << 1,
};

struct mlx5_ib_alloc_ucontext_req_v2 {
	__u32	total_num_bfregs;
	__u32	num_low_latency_bfregs;
	__u32	flags;
	__u32	comp_mask;
	__u8	max_cqe_version;
	__u8	reserved0;
	__u16	reserved1;
	__u32	reserved2;
	__aligned_u64 lib_caps;
};

struct mlx5_ib_alloc_ucontext_resp {
	__u32	qp_tab_size;
	__u32	bf_reg_size;
	__u32	tot_bfregs;
	__u32	cache_line_size;
	__u16	max_sq_desc_sz;
	__u16	max_rq_desc_sz;
	__u32	max_send_wqebb;
	__u32	max_recv_wr;
	__u32	max_srq_recv_wr;
	__u16	num_ports;
	__u16	flow_action_flags;
	__u32	comp_mask;
	__u32	response_length;
	__u8	cqe_version;
	__u8	cmds_supp_uhw;
	__u8	eth_min_inline;
	__u8	clock_info_versions;
	__aligned_u64 hca_core_clock_offset;
	__u32	log_uar_size;
	__u32	num_uars_per_page;
	__u32	num_dyn_bfregs;
	__u32	dump_fill_mkey;
};

struct mlx5_ib_create_cq {
	__aligned_u64 buf_addr;
	__aligned_u64 db_addr;
	__u32	cqe_size;
	__u8    cqe_comp_en;
	__u8    cqe_comp_res_format;
	__u16	flags;
	__u16	uar_page_index;
	__u16	reserved0;
	__u32	reserved1;
};

struct mlx5_ib_create_cq_resp {
	__u32	cqn;
	__u32	reserved;
};

struct mlx5_ib_create_qp_dci_streams {
	__u8 log_num_concurent;
	__u8 log_num_errored;
};

struct mlx5_ib_create_qp {
	__aligned_u64 buf_addr;
	__aligned_u64 db_addr;
	__u32	sq_wqe_count;
	__u32	rq_wqe_count;
	__u32	rq_wqe_shift;
	__u32	flags;
	__u32	uidx;
	__u32	bfreg_index;
	union {
		__aligned_u64 sq_buf_addr;
		__aligned_u64 access_key;
	};
	__u32  ece_options;
	struct  mlx5_ib_create_qp_dci_streams dci_streams;
	__u16 reserved;
};

struct mlx5_ib_create_qp_resp {
	__u32	bfreg_index;
	__u32   ece_options;
	__u32	comp_mask;
	__u32	tirn;
	__u32	tisn;
	__u32	rqn;
	__u32	sqn;
	__u32   reserved1;
	__u64	tir_icm_addr;
};

enum mlx5_ib_mmap_cmd {
	MLX5_IB_MMAP_REGULAR_PAGE               = 0,
	MLX5_IB_MMAP_GET_CONTIGUOUS_PAGES       = 1,
	MLX5_IB_MMAP_WC_PAGE                    = 2,
	MLX5_IB_MMAP_NC_PAGE                    = 3,
	/* 5 is chosen in order to be compatible with old versions of libmlx5 */
	MLX5_IB_MMAP_CORE_CLOCK                 = 5,
	MLX5_IB_MMAP_ALLOC_WC                   = 6,
	MLX5_IB_MMAP_CLOCK_INFO                 = 7,
	MLX5_IB_MMAP_DEVICE_MEM                 = 8,
};
