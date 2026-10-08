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

#define IH_RB_CNTL__RB_ENABLE_MASK                                                                            0x00000001L
#define IH_RB_CNTL__RB_SIZE_MASK                                                                              0x0000003EL
#define IH_RB_CNTL__WPTR_WRITEBACK_ENABLE_MASK                                                                0x00000100L
#define IH_RB_CNTL__RB_FULL_DRAIN_ENABLE_MASK                                                                 0x00000200L
#define IH_RB_CNTL__FULL_DRAIN_CLEAR_MASK                                                                     0x00000400L
#define IH_RB_CNTL__PAGE_RB_CLEAR_MASK                                                                        0x00000800L
#define IH_RB_CNTL__RB_USED_INT_THRESHOLD_MASK                                                                0x0000F000L
#define IH_RB_CNTL__WPTR_OVERFLOW_ENABLE_MASK                                                                 0x00010000L
#define IH_RB_CNTL__ENABLE_INTR_MASK                                                                          0x00020000L
#define IH_RB_CNTL__MC_SWAP_MASK                                                                              0x000C0000L
#define IH_RB_CNTL__MC_SNOOP_MASK                                                                             0x00100000L
#define IH_RB_CNTL__RPTR_REARM_MASK                                                                           0x00200000L
#define IH_RB_CNTL__MC_RO_MASK                                                                                0x00400000L
#define IH_RB_CNTL__MC_VMID_MASK                                                                              0x0F000000L
#define IH_RB_CNTL__MC_SPACE_MASK                                                                             0x70000000L
#define IH_RB_CNTL__WPTR_OVERFLOW_CLEAR_MASK                                                                  0x80000000L
#define IH_RB_BASE__ADDR_MASK                                                                                 0xFFFFFFFFL
#define IH_RB_BASE_HI__ADDR_MASK                                                                              0x000000FFL
#define IH_RB_RPTR__OFFSET_MASK                                                                               0x0003FFFCL
#define IH_RB_WPTR__RB_OVERFLOW_MASK                                                                          0x00000001L
#define IH_RB_WPTR__OFFSET_MASK                                                                               0x0003FFFCL
#define IH_RB_WPTR__RB_LEFT_NONE_MASK                                                                         0x00040000L
#define IH_RB_WPTR__RB_MAY_OVERFLOW_MASK                                                                      0x00080000L
#define IH_RB_WPTR_ADDR_HI__ADDR_MASK                                                                         0x0000FFFFL
#define IH_RB_WPTR_ADDR_LO__ADDR_MASK                                                                         0xFFFFFFFCL
#define IH_DOORBELL_RPTR__OFFSET_MASK                                                                         0x03FFFFFFL
#define IH_DOORBELL_RPTR__ENABLE_MASK                                                                         0x10000000L
#define IH_RB_CNTL_RING1__RB_ENABLE_MASK                                                                      0x00000001L
#define IH_RB_CNTL_RING1__RB_SIZE_MASK                                                                        0x0000003EL
#define IH_RB_CNTL_RING1__RB_FULL_DRAIN_ENABLE_MASK                                                           0x00000200L
#define IH_RB_CNTL_RING1__FULL_DRAIN_CLEAR_MASK                                                               0x00000400L
#define IH_RB_CNTL_RING1__PAGE_RB_CLEAR_MASK                                                                  0x00000800L
#define IH_RB_CNTL_RING1__RB_USED_INT_THRESHOLD_MASK                                                          0x0000F000L
#define IH_RB_CNTL_RING1__WPTR_OVERFLOW_ENABLE_MASK                                                           0x00010000L
#define IH_RB_CNTL_RING1__MC_SWAP_MASK                                                                        0x000C0000L
#define IH_RB_CNTL_RING1__MC_SNOOP_MASK                                                                       0x00100000L
#define IH_RB_CNTL_RING1__MC_RO_MASK                                                                          0x00400000L
#define IH_RB_CNTL_RING1__MC_VMID_MASK                                                                        0x0F000000L
#define IH_RB_CNTL_RING1__MC_SPACE_MASK                                                                       0x70000000L
#define IH_RB_CNTL_RING1__WPTR_OVERFLOW_CLEAR_MASK                                                            0x80000000L
#define IH_RB_BASE_RING1__ADDR_MASK                                                                           0xFFFFFFFFL
#define IH_RB_BASE_HI_RING1__ADDR_MASK                                                                        0x000000FFL
#define IH_RB_RPTR_RING1__OFFSET_MASK                                                                         0x0003FFFCL
#define IH_RB_WPTR_RING1__RB_OVERFLOW_MASK                                                                    0x00000001L
#define IH_RB_WPTR_RING1__OFFSET_MASK                                                                         0x0003FFFCL
#define IH_RB_WPTR_RING1__RB_LEFT_NONE_MASK                                                                   0x00040000L
#define IH_RB_WPTR_RING1__RB_MAY_OVERFLOW_MASK                                                                0x00080000L
#define IH_DOORBELL_RPTR_RING1__OFFSET_MASK                                                                   0x03FFFFFFL
#define IH_DOORBELL_RPTR_RING1__ENABLE_MASK                                                                   0x10000000L
#define IH_INT_FLOOD_CNTL__HIGHWATER_MASK                                                                     0x00000007L
#define IH_INT_FLOOD_CNTL__FLOOD_CNTL_ENABLE_MASK                                                             0x00000008L
#define IH_INT_FLOOD_CNTL__CLEAR_INT_FLOOD_STATUS_MASK                                                        0x00000010L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT1_IS_STORM_CLIENT_MASK                                               0x00000002L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT2_IS_STORM_CLIENT_MASK                                               0x00000004L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT3_IS_STORM_CLIENT_MASK                                               0x00000008L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT4_IS_STORM_CLIENT_MASK                                               0x00000010L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT5_IS_STORM_CLIENT_MASK                                               0x00000020L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT6_IS_STORM_CLIENT_MASK                                               0x00000040L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT7_IS_STORM_CLIENT_MASK                                               0x00000080L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT8_IS_STORM_CLIENT_MASK                                               0x00000100L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT9_IS_STORM_CLIENT_MASK                                               0x00000200L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT10_IS_STORM_CLIENT_MASK                                              0x00000400L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT11_IS_STORM_CLIENT_MASK                                              0x00000800L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT12_IS_STORM_CLIENT_MASK                                              0x00001000L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT13_IS_STORM_CLIENT_MASK                                              0x00002000L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT14_IS_STORM_CLIENT_MASK                                              0x00004000L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT15_IS_STORM_CLIENT_MASK                                              0x00008000L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT16_IS_STORM_CLIENT_MASK                                              0x00010000L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT17_IS_STORM_CLIENT_MASK                                              0x00020000L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT18_IS_STORM_CLIENT_MASK                                              0x00040000L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT19_IS_STORM_CLIENT_MASK                                              0x00080000L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT20_IS_STORM_CLIENT_MASK                                              0x00100000L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT21_IS_STORM_CLIENT_MASK                                              0x00200000L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT22_IS_STORM_CLIENT_MASK                                              0x00400000L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT23_IS_STORM_CLIENT_MASK                                              0x00800000L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT24_IS_STORM_CLIENT_MASK                                              0x01000000L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT25_IS_STORM_CLIENT_MASK                                              0x02000000L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT26_IS_STORM_CLIENT_MASK                                              0x04000000L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT27_IS_STORM_CLIENT_MASK                                              0x08000000L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT28_IS_STORM_CLIENT_MASK                                              0x10000000L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT29_IS_STORM_CLIENT_MASK                                              0x20000000L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT30_IS_STORM_CLIENT_MASK                                              0x40000000L
#define IH_STORM_CLIENT_LIST_CNTL__CLIENT31_IS_STORM_CLIENT_MASK                                              0x80000000L
#define IH_MSI_STORM_CTRL__DELAY_MASK                                                                         0x00000FFFL
