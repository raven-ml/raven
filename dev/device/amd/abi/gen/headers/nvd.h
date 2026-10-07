/*
 * Copyright 2019 Advanced Micro Devices, Inc.
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

#define	PACKET_TYPE3	3
#define	PACKET3_DISPATCH_DIRECT				0x15
#define	PACKET3_PRED_EXEC				0x23
#define	PACKET3_WRITE_DATA				0x37
#define		WRITE_DATA_DST_SEL(x)                   ((x) << 8)
#define		WR_ONE_ADDR                             (1 << 16)
#define		WR_CONFIRM                              (1 << 20)
#define	PACKET3_WAIT_REG_MEM				0x3C
#define		WAIT_REG_MEM_FUNCTION(x)                ((x) << 0)
#define		WAIT_REG_MEM_MEM_SPACE(x)               ((x) << 4)
#define 	PACKET3_WAIT_REG_MEM__FUNCTION__EQUAL_TO_THE_REFERENCE_VALUE 3
#define 	PACKET3_WAIT_REG_MEM__FUNCTION__GREATER_THAN_OR_EQUAL_REFERENCE_VALUE 5
#define	PACKET3_INDIRECT_BUFFER				0x3F
#define		INDIRECT_BUFFER_VALID                   (1 << 23)
#define	PACKET3_COPY_DATA				0x40
#define 	PACKET3_COPY_DATA__SRC_SEL(x) ((((unsigned)(x)) & 0xF) << 0)
#define 	PACKET3_COPY_DATA__DST_SEL(x) ((((unsigned)(x)) & 0xF) << 8)
#define 	PACKET3_COPY_DATA__COUNT_SEL(x) ((((unsigned)(x)) & 0x1) << 16)
#define 	PACKET3_COPY_DATA__WR_CONFIRM(x) ((((unsigned)(x)) & 0x1) << 20)
#define	PACKET3_EVENT_WRITE				0x46
#define		EVENT_TYPE(x)                           ((x) << 0)
#define		EVENT_INDEX(x)                          ((x) << 8)
#define	PACKET3_RELEASE_MEM				0x49
#define		PACKET3_RELEASE_MEM_EVENT_TYPE(x)	((x) << 0)
#define		PACKET3_RELEASE_MEM_EVENT_INDEX(x)	((x) << 8)
#define		PACKET3_RELEASE_MEM_GCR_GLM_WB		(1 << 12)
#define		PACKET3_RELEASE_MEM_GCR_GLM_INV		(1 << 13)
#define		PACKET3_RELEASE_MEM_GCR_GLV_INV		(1 << 14)
#define		PACKET3_RELEASE_MEM_GCR_GL1_INV		(1 << 15)
#define		PACKET3_RELEASE_MEM_GCR_GL2_INV		(1 << 20)
#define		PACKET3_RELEASE_MEM_GCR_GL2_WB		(1 << 21)
#define		PACKET3_RELEASE_MEM_GCR_SEQ		(1 << 22)
#define		PACKET3_RELEASE_MEM_DATA_SEL(x)		((x) << 29)
#define		PACKET3_RELEASE_MEM_INT_SEL(x)		((x) << 24)
#define	PACKET3_ACQUIRE_MEM				0x58
#define 	PACKET3_ACQUIRE_MEM_GCR_CNTL_GLI_INV(x) ((x) << 0)
#define 	PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_WB(x) ((x) << 4)
#define 	PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_INV(x) ((x) << 5)
#define 	PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_WB(x) ((x) << 6)
#define 	PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_INV(x) ((x) << 7)
#define 	PACKET3_ACQUIRE_MEM_GCR_CNTL_GLV_INV(x) ((x) << 8)
#define 	PACKET3_ACQUIRE_MEM_GCR_CNTL_GL1_INV(x) ((x) << 9)
#define 	PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_INV(x) ((x) << 14)
#define 	PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_WB(x) ((x) << 15)
#define	PACKET3_SET_SH_REG				0x76
#define		PACKET3_SET_SH_REG_START			0x00002c00
#define		PACKET3_SET_SH_REG_END				0x00003000
#define	PACKET3_SET_UCONFIG_REG				0x79
#define		PACKET3_SET_UCONFIG_REG_START			0x0000c000
#define	PACKET3_WAIT_REG_MEM64				0x93
