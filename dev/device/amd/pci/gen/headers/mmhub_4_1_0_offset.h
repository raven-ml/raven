/*
 * Copyright 2023 Advanced Micro Devices, Inc.
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

#define regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB                                                     0x04c8
#define regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB_BASE_IDX                                            0
#define regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB                                                     0x04c9
#define regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB_BASE_IDX                                            0
#define regMMVM_L2_CNTL                                                                                 0x04e4
#define regMMVM_L2_CNTL_BASE_IDX                                                                        0
#define regMMVM_L2_CNTL2                                                                                0x04e5
#define regMMVM_L2_CNTL2_BASE_IDX                                                                       0
#define regMMVM_L2_CNTL3                                                                                0x04e6
#define regMMVM_L2_CNTL3_BASE_IDX                                                                       0
#define regMMVM_L2_PROTECTION_FAULT_CNTL                                                                0x04ec
#define regMMVM_L2_PROTECTION_FAULT_CNTL_BASE_IDX                                                       0
#define regMMVM_L2_PROTECTION_FAULT_CNTL2                                                               0x04ed
#define regMMVM_L2_PROTECTION_FAULT_CNTL2_BASE_IDX                                                      0
#define regMMVM_L2_PROTECTION_FAULT_STATUS_LO32                                                         0x04f0
#define regMMVM_L2_PROTECTION_FAULT_STATUS_LO32_BASE_IDX                                                0
#define regMMVM_L2_PROTECTION_FAULT_ADDR_LO32                                                           0x04f2
#define regMMVM_L2_PROTECTION_FAULT_ADDR_LO32_BASE_IDX                                                  0
#define regMMVM_L2_PROTECTION_FAULT_ADDR_HI32                                                           0x04f3
#define regMMVM_L2_PROTECTION_FAULT_ADDR_HI32_BASE_IDX                                                  0
#define regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32                                                   0x04f4
#define regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32_BASE_IDX                                          0
#define regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32                                                   0x04f5
#define regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32_BASE_IDX                                          0
#define regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32                                             0x04f7
#define regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32_BASE_IDX                                    0
#define regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32                                             0x04f8
#define regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32_BASE_IDX                                    0
#define regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32                                            0x04f9
#define regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32_BASE_IDX                                   0
#define regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32                                            0x04fa
#define regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32_BASE_IDX                                   0
#define regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32                                                0x04fb
#define regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32_BASE_IDX                                       0
#define regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32                                                0x04fc
#define regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32_BASE_IDX                                       0
#define regMMVM_L2_CNTL4                                                                                0x04fd
#define regMMVM_L2_CNTL4_BASE_IDX                                                                       0
#define regMMVM_L2_BANK_SELECT_RESERVED_CID2                                                            0x0500
#define regMMVM_L2_BANK_SELECT_RESERVED_CID2_BASE_IDX                                                   0
#define regMMVM_L2_CNTL5                                                                                0x0503
#define regMMVM_L2_CNTL5_BASE_IDX                                                                       0
#define regMMMC_VM_FB_LOCATION_BASE                                                                     0x0554
#define regMMMC_VM_FB_LOCATION_BASE_BASE_IDX                                                            0
#define regMMMC_VM_FB_LOCATION_TOP                                                                      0x0555
#define regMMMC_VM_FB_LOCATION_TOP_BASE_IDX                                                             0
#define regMMMC_VM_AGP_TOP                                                                              0x0556
#define regMMMC_VM_AGP_TOP_BASE_IDX                                                                     0
#define regMMMC_VM_AGP_BOT                                                                              0x0557
#define regMMMC_VM_AGP_BOT_BASE_IDX                                                                     0
#define regMMMC_VM_AGP_BASE                                                                             0x0558
#define regMMMC_VM_AGP_BASE_BASE_IDX                                                                    0
#define regMMMC_VM_SYSTEM_APERTURE_LOW_ADDR                                                             0x0559
#define regMMMC_VM_SYSTEM_APERTURE_LOW_ADDR_BASE_IDX                                                    0
#define regMMMC_VM_SYSTEM_APERTURE_HIGH_ADDR                                                            0x055a
#define regMMMC_VM_SYSTEM_APERTURE_HIGH_ADDR_BASE_IDX                                                   0
#define regMMMC_VM_MX_L1_TLB_CNTL                                                                       0x055b
#define regMMMC_VM_MX_L1_TLB_CNTL_BASE_IDX                                                              0
#define regMMVM_CONTEXT0_CNTL                                                                           0x0564
#define regMMVM_CONTEXT0_CNTL_BASE_IDX                                                                  0
#define regMMVM_INVALIDATE_ENG17_SEM                                                                    0x0586
#define regMMVM_INVALIDATE_ENG17_SEM_BASE_IDX                                                           0
#define regMMVM_INVALIDATE_ENG17_REQ                                                                    0x0598
#define regMMVM_INVALIDATE_ENG17_REQ_BASE_IDX                                                           0
#define regMMVM_INVALIDATE_ENG17_ACK                                                                    0x05aa
#define regMMVM_INVALIDATE_ENG17_ACK_BASE_IDX                                                           0
#define regMMVM_INVALIDATE_ENG0_ADDR_RANGE_LO32                                                         0x05ab
#define regMMVM_INVALIDATE_ENG0_ADDR_RANGE_LO32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG0_ADDR_RANGE_HI32                                                         0x05ac
#define regMMVM_INVALIDATE_ENG0_ADDR_RANGE_HI32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG1_ADDR_RANGE_LO32                                                         0x05ad
#define regMMVM_INVALIDATE_ENG1_ADDR_RANGE_LO32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG1_ADDR_RANGE_HI32                                                         0x05ae
#define regMMVM_INVALIDATE_ENG1_ADDR_RANGE_HI32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG2_ADDR_RANGE_LO32                                                         0x05af
#define regMMVM_INVALIDATE_ENG2_ADDR_RANGE_LO32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG2_ADDR_RANGE_HI32                                                         0x05b0
#define regMMVM_INVALIDATE_ENG2_ADDR_RANGE_HI32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG3_ADDR_RANGE_LO32                                                         0x05b1
#define regMMVM_INVALIDATE_ENG3_ADDR_RANGE_LO32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG3_ADDR_RANGE_HI32                                                         0x05b2
#define regMMVM_INVALIDATE_ENG3_ADDR_RANGE_HI32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG4_ADDR_RANGE_LO32                                                         0x05b3
#define regMMVM_INVALIDATE_ENG4_ADDR_RANGE_LO32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG4_ADDR_RANGE_HI32                                                         0x05b4
#define regMMVM_INVALIDATE_ENG4_ADDR_RANGE_HI32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG5_ADDR_RANGE_LO32                                                         0x05b5
#define regMMVM_INVALIDATE_ENG5_ADDR_RANGE_LO32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG5_ADDR_RANGE_HI32                                                         0x05b6
#define regMMVM_INVALIDATE_ENG5_ADDR_RANGE_HI32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG6_ADDR_RANGE_LO32                                                         0x05b7
#define regMMVM_INVALIDATE_ENG6_ADDR_RANGE_LO32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG6_ADDR_RANGE_HI32                                                         0x05b8
#define regMMVM_INVALIDATE_ENG6_ADDR_RANGE_HI32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG7_ADDR_RANGE_LO32                                                         0x05b9
#define regMMVM_INVALIDATE_ENG7_ADDR_RANGE_LO32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG7_ADDR_RANGE_HI32                                                         0x05ba
#define regMMVM_INVALIDATE_ENG7_ADDR_RANGE_HI32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG8_ADDR_RANGE_LO32                                                         0x05bb
#define regMMVM_INVALIDATE_ENG8_ADDR_RANGE_LO32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG8_ADDR_RANGE_HI32                                                         0x05bc
#define regMMVM_INVALIDATE_ENG8_ADDR_RANGE_HI32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG9_ADDR_RANGE_LO32                                                         0x05bd
#define regMMVM_INVALIDATE_ENG9_ADDR_RANGE_LO32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG9_ADDR_RANGE_HI32                                                         0x05be
#define regMMVM_INVALIDATE_ENG9_ADDR_RANGE_HI32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG10_ADDR_RANGE_LO32                                                        0x05bf
#define regMMVM_INVALIDATE_ENG10_ADDR_RANGE_LO32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG10_ADDR_RANGE_HI32                                                        0x05c0
#define regMMVM_INVALIDATE_ENG10_ADDR_RANGE_HI32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG11_ADDR_RANGE_LO32                                                        0x05c1
#define regMMVM_INVALIDATE_ENG11_ADDR_RANGE_LO32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG11_ADDR_RANGE_HI32                                                        0x05c2
#define regMMVM_INVALIDATE_ENG11_ADDR_RANGE_HI32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG12_ADDR_RANGE_LO32                                                        0x05c3
#define regMMVM_INVALIDATE_ENG12_ADDR_RANGE_LO32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG12_ADDR_RANGE_HI32                                                        0x05c4
#define regMMVM_INVALIDATE_ENG12_ADDR_RANGE_HI32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG13_ADDR_RANGE_LO32                                                        0x05c5
#define regMMVM_INVALIDATE_ENG13_ADDR_RANGE_LO32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG13_ADDR_RANGE_HI32                                                        0x05c6
#define regMMVM_INVALIDATE_ENG13_ADDR_RANGE_HI32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG14_ADDR_RANGE_LO32                                                        0x05c7
#define regMMVM_INVALIDATE_ENG14_ADDR_RANGE_LO32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG14_ADDR_RANGE_HI32                                                        0x05c8
#define regMMVM_INVALIDATE_ENG14_ADDR_RANGE_HI32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG15_ADDR_RANGE_LO32                                                        0x05c9
#define regMMVM_INVALIDATE_ENG15_ADDR_RANGE_LO32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG15_ADDR_RANGE_HI32                                                        0x05ca
#define regMMVM_INVALIDATE_ENG15_ADDR_RANGE_HI32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG16_ADDR_RANGE_LO32                                                        0x05cb
#define regMMVM_INVALIDATE_ENG16_ADDR_RANGE_LO32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG16_ADDR_RANGE_HI32                                                        0x05cc
#define regMMVM_INVALIDATE_ENG16_ADDR_RANGE_HI32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG17_ADDR_RANGE_LO32                                                        0x05cd
#define regMMVM_INVALIDATE_ENG17_ADDR_RANGE_LO32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG17_ADDR_RANGE_HI32                                                        0x05ce
#define regMMVM_INVALIDATE_ENG17_ADDR_RANGE_HI32_BASE_IDX                                               0
#define regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32                                                      0x05cf
#define regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32_BASE_IDX                                             0
#define regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32                                                      0x05d0
#define regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32_BASE_IDX                                             0
#define regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32                                                     0x05ef
#define regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32_BASE_IDX                                            0
#define regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32                                                     0x05f0
#define regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32_BASE_IDX                                            0
#define regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32                                                       0x060f
#define regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32_BASE_IDX                                              0
#define regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32                                                       0x0610
#define regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32_BASE_IDX                                              0
