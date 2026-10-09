/*
 * Copyright (c) 2018, Mellanox Technologies inc.  All rights reserved.
 * Copyright (c) 2020, Intel Corporation. All rights reserved.
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

/* Excerpt of https://raw.githubusercontent.com/torvalds/linux/v6.12/include/uapi/rdma/ib_user_ioctl_cmds.h. */

#define UVERBS_ID_NS_SHIFT 12

enum uverbs_default_objects {
	UVERBS_OBJECT_DEVICE, /* No instances of DEVICE are allowed */
	UVERBS_OBJECT_PD,
	UVERBS_OBJECT_COMP_CHANNEL,
	UVERBS_OBJECT_CQ,
	UVERBS_OBJECT_QP,
	UVERBS_OBJECT_SRQ,
	UVERBS_OBJECT_AH,
	UVERBS_OBJECT_MR,
	UVERBS_OBJECT_MW,
	UVERBS_OBJECT_FLOW,
	UVERBS_OBJECT_XRCD,
	UVERBS_OBJECT_RWQ_IND_TBL,
	UVERBS_OBJECT_WQ,
	UVERBS_OBJECT_FLOW_ACTION,
	UVERBS_OBJECT_DM,
	UVERBS_OBJECT_COUNTERS,
	UVERBS_OBJECT_ASYNC_EVENT,
};

enum {
	UVERBS_ID_DRIVER_NS = 1UL << UVERBS_ID_NS_SHIFT,
	UVERBS_ATTR_UHW_IN = UVERBS_ID_DRIVER_NS,
	UVERBS_ATTR_UHW_OUT,
	UVERBS_ID_DRIVER_NS_WITH_UHW,
};

enum uverbs_methods_device {
	UVERBS_METHOD_INVOKE_WRITE,
	UVERBS_METHOD_INFO_HANDLES,
	UVERBS_METHOD_QUERY_PORT,
	UVERBS_METHOD_GET_CONTEXT,
	UVERBS_METHOD_QUERY_CONTEXT,
	UVERBS_METHOD_QUERY_GID_TABLE,
	UVERBS_METHOD_QUERY_GID_ENTRY,
};

enum uverbs_attrs_invoke_write_cmd_attr_ids {
	UVERBS_ATTR_CORE_IN,
	UVERBS_ATTR_CORE_OUT,
	UVERBS_ATTR_WRITE_CMD,
};

enum uverbs_methods_mr {
	UVERBS_METHOD_DM_MR_REG,
	UVERBS_METHOD_MR_DESTROY,
	UVERBS_METHOD_ADVISE_MR,
	UVERBS_METHOD_QUERY_MR,
	UVERBS_METHOD_REG_DMABUF_MR,
};

enum uverbs_attrs_reg_dmabuf_mr_cmd_attr_ids {
	UVERBS_ATTR_REG_DMABUF_MR_HANDLE,
	UVERBS_ATTR_REG_DMABUF_MR_PD_HANDLE,
	UVERBS_ATTR_REG_DMABUF_MR_OFFSET,
	UVERBS_ATTR_REG_DMABUF_MR_LENGTH,
	UVERBS_ATTR_REG_DMABUF_MR_IOVA,
	UVERBS_ATTR_REG_DMABUF_MR_FD,
	UVERBS_ATTR_REG_DMABUF_MR_ACCESS_FLAGS,
	UVERBS_ATTR_REG_DMABUF_MR_RESP_LKEY,
	UVERBS_ATTR_REG_DMABUF_MR_RESP_RKEY,
};
