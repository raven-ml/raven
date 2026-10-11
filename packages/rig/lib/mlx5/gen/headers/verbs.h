/*
 * Copyright (c) 2004, 2005 Topspin Communications.  All rights reserved.
 * Copyright (c) 2004, 2011-2012 Intel Corporation.  All rights reserved.
 * Copyright (c) 2005, 2006, 2007 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2005 PathScale, Inc.  All rights reserved.
 * Copyright (c) 2020 Intel Corporation.  All rights reserved.
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

/* Excerpt of https://raw.githubusercontent.com/linux-rdma/rdma-core/v56.0/libibverbs/verbs.h. */

enum ibv_mtu {
	IBV_MTU_256  = 1,
	IBV_MTU_512  = 2,
	IBV_MTU_1024 = 3,
	IBV_MTU_2048 = 4,
	IBV_MTU_4096 = 5
};

enum ibv_port_state {
	IBV_PORT_NOP		= 0,
	IBV_PORT_DOWN		= 1,
	IBV_PORT_INIT		= 2,
	IBV_PORT_ARMED		= 3,
	IBV_PORT_ACTIVE		= 4,
	IBV_PORT_ACTIVE_DEFER	= 5
};

enum {
	IBV_LINK_LAYER_UNSPECIFIED,
	IBV_LINK_LAYER_INFINIBAND,
	IBV_LINK_LAYER_ETHERNET,
};

enum ibv_event_type {
	IBV_EVENT_CQ_ERR,
	IBV_EVENT_QP_FATAL,
	IBV_EVENT_QP_REQ_ERR,
	IBV_EVENT_QP_ACCESS_ERR,
	IBV_EVENT_COMM_EST,
	IBV_EVENT_SQ_DRAINED,
	IBV_EVENT_PATH_MIG,
	IBV_EVENT_PATH_MIG_ERR,
	IBV_EVENT_DEVICE_FATAL,
	IBV_EVENT_PORT_ACTIVE,
	IBV_EVENT_PORT_ERR,
	IBV_EVENT_LID_CHANGE,
	IBV_EVENT_PKEY_CHANGE,
	IBV_EVENT_SM_CHANGE,
	IBV_EVENT_SRQ_ERR,
	IBV_EVENT_SRQ_LIMIT_REACHED,
	IBV_EVENT_QP_LAST_WQE_REACHED,
	IBV_EVENT_CLIENT_REREGISTER,
	IBV_EVENT_GID_CHANGE,
	IBV_EVENT_WQ_FATAL,
};

enum ibv_qp_attr_mask {
	IBV_QP_STATE			= 1 << 	0,
	IBV_QP_CUR_STATE		= 1 << 	1,
	IBV_QP_EN_SQD_ASYNC_NOTIFY	= 1 << 	2,
	IBV_QP_ACCESS_FLAGS		= 1 << 	3,
	IBV_QP_PKEY_INDEX		= 1 << 	4,
	IBV_QP_PORT			= 1 << 	5,
	IBV_QP_QKEY			= 1 << 	6,
	IBV_QP_AV			= 1 << 	7,
	IBV_QP_PATH_MTU			= 1 << 	8,
	IBV_QP_TIMEOUT			= 1 << 	9,
	IBV_QP_RETRY_CNT		= 1 << 10,
	IBV_QP_RNR_RETRY		= 1 << 11,
	IBV_QP_RQ_PSN			= 1 << 12,
	IBV_QP_MAX_QP_RD_ATOMIC		= 1 << 13,
	IBV_QP_ALT_PATH			= 1 << 14,
	IBV_QP_MIN_RNR_TIMER		= 1 << 15,
	IBV_QP_SQ_PSN			= 1 << 16,
	IBV_QP_MAX_DEST_RD_ATOMIC	= 1 << 17,
	IBV_QP_PATH_MIG_STATE		= 1 << 18,
	IBV_QP_CAP			= 1 << 19,
	IBV_QP_DEST_QPN			= 1 << 20,
	/* These bits were supported on older kernels, but never exposed from
	   libibverbs:
	_IBV_QP_SMAC   			= 1 << 21,
	_IBV_QP_ALT_SMAC		= 1 << 22,
	_IBV_QP_VID    			= 1 << 23,
	_IBV_QP_ALT_VID 		= 1 << 24,
	*/
	IBV_QP_RATE_LIMIT		= 1 << 25,
};

enum ibv_qp_state {
	IBV_QPS_RESET,
	IBV_QPS_INIT,
	IBV_QPS_RTR,
	IBV_QPS_RTS,
	IBV_QPS_SQD,
	IBV_QPS_SQE,
	IBV_QPS_ERR,
	IBV_QPS_UNKNOWN
};
