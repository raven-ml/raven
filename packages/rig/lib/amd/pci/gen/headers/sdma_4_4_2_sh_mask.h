/*
 * Copyright 2021 Advanced Micro Devices, Inc.
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

#define SDMA_CNTL__TRAP_ENABLE_MASK                                                                           0x00000001L
#define SDMA_CNTL__UTC_L1_ENABLE_MASK                                                                         0x00000002L
#define SDMA_CNTL__SEM_WAIT_INT_ENABLE_MASK                                                                   0x00000004L
#define SDMA_CNTL__DATA_SWAP_ENABLE_MASK                                                                      0x00000008L
#define SDMA_CNTL__FENCE_SWAP_ENABLE_MASK                                                                     0x00000010L
#define SDMA_CNTL__MIDCMD_PREEMPT_ENABLE_MASK                                                                 0x00000020L
#define SDMA_CNTL__MIDCMD_EXPIRE_ENABLE_MASK                                                                  0x00000040L
#define SDMA_CNTL__REG_WRITE_PROTECT_INT_ENABLE_MASK                                                          0x00000080L
#define SDMA_CNTL__INVALID_DOORBELL_INT_ENABLE_MASK                                                           0x00000100L
#define SDMA_CNTL__VM_HOLE_INT_ENABLE_MASK                                                                    0x00000200L
#define SDMA_CNTL__DRAM_ECC_INT_ENABLE_MASK                                                                   0x00000400L
#define SDMA_CNTL__PAGE_RETRY_TIMEOUT_INT_ENABLE_MASK                                                         0x00000800L
#define SDMA_CNTL__PAGE_NULL_INT_ENABLE_MASK                                                                  0x00001000L
#define SDMA_CNTL__PAGE_FAULT_INT_ENABLE_MASK                                                                 0x00002000L
#define SDMA_CNTL__NACK_GEN_ERR_INT_ENABLE_MASK                                                               0x00004000L
#define SDMA_CNTL__MIDCMD_WORLDSWITCH_ENABLE_MASK                                                             0x00020000L
#define SDMA_CNTL__AUTO_CTXSW_ENABLE_MASK                                                                     0x00040000L
#define SDMA_CNTL__DRM_RESTORE_ENABLE_MASK                                                                    0x00080000L
#define SDMA_CNTL__CTXEMPTY_INT_ENABLE_MASK                                                                   0x10000000L
#define SDMA_CNTL__FROZEN_INT_ENABLE_MASK                                                                     0x20000000L
#define SDMA_CNTL__IB_PREEMPT_INT_ENABLE_MASK                                                                 0x40000000L
#define SDMA_CNTL__RB_PREEMPT_INT_ENABLE_MASK                                                                 0x80000000L
#define SDMA_GFX_RB_CNTL__RB_ENABLE_MASK                                                                      0x00000001L
#define SDMA_GFX_RB_CNTL__RB_SIZE_MASK                                                                        0x0000003EL
#define SDMA_GFX_RB_CNTL__RB_SWAP_ENABLE_MASK                                                                 0x00000200L
#define SDMA_GFX_RB_CNTL__RPTR_WRITEBACK_ENABLE_MASK                                                          0x00001000L
#define SDMA_GFX_RB_CNTL__RPTR_WRITEBACK_SWAP_ENABLE_MASK                                                     0x00002000L
#define SDMA_GFX_RB_CNTL__RPTR_WRITEBACK_TIMER_MASK                                                           0x001F0000L
#define SDMA_GFX_RB_CNTL__RB_PRIV_MASK                                                                        0x00800000L
#define SDMA_GFX_RB_CNTL__RB_VMID_MASK                                                                        0x0F000000L
#define SDMA_GFX_RB_BASE__ADDR_MASK                                                                           0xFFFFFFFFL
#define SDMA_GFX_RB_BASE_HI__ADDR_MASK                                                                        0x00FFFFFFL
#define SDMA_GFX_RB_RPTR__OFFSET_MASK                                                                         0xFFFFFFFFL
#define SDMA_GFX_RB_RPTR_HI__OFFSET_MASK                                                                      0xFFFFFFFFL
#define SDMA_GFX_RB_WPTR__OFFSET_MASK                                                                         0xFFFFFFFFL
#define SDMA_GFX_RB_WPTR_HI__OFFSET_MASK                                                                      0xFFFFFFFFL
#define SDMA_GFX_RB_RPTR_ADDR_HI__ADDR_MASK                                                                   0xFFFFFFFFL
#define SDMA_GFX_RB_RPTR_ADDR_LO__RPTR_WB_IDLE_MASK                                                           0x00000001L
#define SDMA_GFX_RB_RPTR_ADDR_LO__ADDR_MASK                                                                   0xFFFFFFFCL
#define SDMA_GFX_IB_CNTL__IB_ENABLE_MASK                                                                      0x00000001L
#define SDMA_GFX_IB_CNTL__IB_SWAP_ENABLE_MASK                                                                 0x00000010L
#define SDMA_GFX_IB_CNTL__SWITCH_INSIDE_IB_MASK                                                               0x00000100L
#define SDMA_GFX_IB_CNTL__CMD_VMID_MASK                                                                       0x000F0000L
#define SDMA_GFX_IB_CNTL__IB_PRIV_MASK                                                                        0x80000000L
#define SDMA_GFX_DOORBELL__ENABLE_MASK                                                                        0x10000000L
#define SDMA_GFX_DOORBELL__CAPTURED_MASK                                                                      0x40000000L
#define SDMA_GFX_DOORBELL_OFFSET__OFFSET_MASK                                                                 0x0FFFFFFCL
#define SDMA_GFX_RB_WPTR_POLL_ADDR_HI__ADDR_MASK                                                              0xFFFFFFFFL
#define SDMA_GFX_RB_WPTR_POLL_ADDR_LO__ADDR_MASK                                                              0xFFFFFFFCL
#define SDMA_GFX_MINOR_PTR_UPDATE__ENABLE_MASK                                                                0x00000001L
