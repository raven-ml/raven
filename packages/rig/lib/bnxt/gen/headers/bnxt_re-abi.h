/*
 * Broadcom NetXtreme-E User Space RoCE driver
 *
 * Copyright (c) 2015-2017, Broadcom. All rights reserved.  The term
 * Broadcom refers to Broadcom Limited and/or its subsidiaries.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Description: ABI data structure definition
 */

/* Excerpt of https://raw.githubusercontent.com/linux-rdma/rdma-core/v56.0/providers/bnxt_re/bnxt_re-abi.h. */

enum bnxt_re_db_que_type {
	BNXT_RE_QUE_TYPE_SQ		= 0x00,
	BNXT_RE_QUE_TYPE_RQ		= 0x01,
	BNXT_RE_QUE_TYPE_SRQ		= 0x02,
	BNXT_RE_QUE_TYPE_SRQ_ARM	= 0x03,
	BNXT_RE_QUE_TYPE_CQ		= 0x04,
	BNXT_RE_QUE_TYPE_CQ_ARMSE	= 0x05,
	BNXT_RE_QUE_TYPE_CQ_ARMALL	= 0x06,
	BNXT_RE_QUE_TYPE_CQ_ARMENA	= 0x07,
	BNXT_RE_QUE_TYPE_SRQ_ARMENA	= 0x08,
	BNXT_RE_QUE_TYPE_CQ_CUT_ACK	= 0x09,
	BNXT_RE_PUSH_TYPE_START		= 0x0C,
	BNXT_RE_PUSH_TYPE_END		= 0x0D,
	BNXT_RE_QUE_TYPE_NULL		= 0x0F
};

enum bnxt_re_db_mask {
	BNXT_RE_DB_INDX_MASK		= 0xFFFFFFUL,
	BNXT_RE_DB_PILO_MASK		= 0x0FFUL,
	BNXT_RE_DB_PILO_SHIFT		= 0x18,
	BNXT_RE_DB_QID_MASK		= 0xFFFFFUL,
	BNXT_RE_DB_PIHI_MASK            = 0xF00UL,
	BNXT_RE_DB_PIHI_SHIFT           = 0x0C, /* Because mask is 0xF00 */
	BNXT_RE_DB_TYP_MASK             = 0x0FUL,
	BNXT_RE_DB_TYP_SHIFT            = 0x1C,
	BNXT_RE_DB_VALID_SHIFT          = 0x1A,
	BNXT_RE_DB_EPOCH_SHIFT          = 0x18,
	BNXT_RE_DB_TOGGLE_SHIFT		= 0x19,
};
