/*
 * Copyright 2016 Advanced Micro Devices, Inc.
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
 * Author: Huang Rui
 *
 */

#define PSP_FENCE_BUFFER_SIZE	0x1000
#define PSP_CMD_BUFFER_SIZE	0x1000
#define PSP_1_MEG		0x100000
#define PSP_TMR_ALIGNMENT	0x100000
enum psp_bootloader_cmd {
	PSP_BL__LOAD_SYSDRV		= 0x10000,
	PSP_BL__LOAD_SOSDRV		= 0x20000,
	PSP_BL__LOAD_KEY_DATABASE	= 0x80000,
	PSP_BL__LOAD_SOCDRV             = 0xB0000,
	PSP_BL__LOAD_DBGDRV             = 0xC0000,
	PSP_BL__LOAD_HADDRV		= PSP_BL__LOAD_DBGDRV,
	PSP_BL__LOAD_INTFDRV		= 0xD0000,
	PSP_BL__LOAD_RASDRV		= 0xE0000,
	PSP_BL__LOAD_IPKEYMGRDRV	= 0xF0000,
	PSP_BL__DRAM_LONG_TRAIN		= 0x100000,
	PSP_BL__DRAM_SHORT_TRAIN	= 0x200000,
	PSP_BL__LOAD_TOS_SPL_TABLE	= 0x10000000,
	PSP_BL__LOAD_SPDMDRV		= 0x20000000,
};
enum psp_ring_type {
	PSP_RING_TYPE__INVALID = 0,
	/*
	 * These values map to the way the PSP kernel identifies the
	 * rings.
	 */
	PSP_RING_TYPE__UM = 1, /* User mode ring (formerly called RBI) */
	PSP_RING_TYPE__KM = 2  /* Kernel mode ring (formerly called GPCOM) */
};
