/*
 * Copyright 2022 Advanced Micro Devices, Inc.
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

#define regMM_ATC_L2_MISC_CG                                                                            0x06cd
#define regMM_ATC_L2_MISC_CG_BASE_IDX                                                                   1
#define regMMVM_L2_CNTL                                                                                 0x0700
#define regMMVM_L2_CNTL_BASE_IDX                                                                        1
#define regMMVM_L2_CNTL2                                                                                0x0701
#define regMMVM_L2_CNTL2_BASE_IDX                                                                       1
#define regMMVM_L2_CNTL3                                                                                0x0702
#define regMMVM_L2_CNTL3_BASE_IDX                                                                       1
#define regMMVM_L2_PROTECTION_FAULT_CNTL                                                                0x0708
#define regMMVM_L2_PROTECTION_FAULT_CNTL_BASE_IDX                                                       1
#define regMMVM_L2_PROTECTION_FAULT_CNTL2                                                               0x0709
#define regMMVM_L2_PROTECTION_FAULT_CNTL2_BASE_IDX                                                      1
#define regMMVM_L2_PROTECTION_FAULT_STATUS                                                              0x070c
#define regMMVM_L2_PROTECTION_FAULT_STATUS_BASE_IDX                                                     1
#define regMMVM_L2_PROTECTION_FAULT_ADDR_LO32                                                           0x070d
#define regMMVM_L2_PROTECTION_FAULT_ADDR_LO32_BASE_IDX                                                  1
#define regMMVM_L2_PROTECTION_FAULT_ADDR_HI32                                                           0x070e
#define regMMVM_L2_PROTECTION_FAULT_ADDR_HI32_BASE_IDX                                                  1
#define regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32                                                   0x070f
#define regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32_BASE_IDX                                          1
#define regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32                                                   0x0710
#define regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32_BASE_IDX                                          1
#define regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32                                             0x0712
#define regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32_BASE_IDX                                    1
#define regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32                                             0x0713
#define regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32_BASE_IDX                                    1
#define regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32                                            0x0714
#define regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32_BASE_IDX                                   1
#define regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32                                            0x0715
#define regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32_BASE_IDX                                   1
#define regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32                                                0x0716
#define regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32_BASE_IDX                                       1
#define regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32                                                0x0717
#define regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32_BASE_IDX                                       1
#define regMMVM_L2_CNTL4                                                                                0x0718
#define regMMVM_L2_CNTL4_BASE_IDX                                                                       1
#define regMMVM_L2_BANK_SELECT_RESERVED_CID2                                                            0x071b
#define regMMVM_L2_BANK_SELECT_RESERVED_CID2_BASE_IDX                                                   1
#define regMMVM_L2_CNTL5                                                                                0x071e
#define regMMVM_L2_CNTL5_BASE_IDX                                                                       1
#define regMMVM_CONTEXT0_CNTL                                                                           0x0740
#define regMMVM_CONTEXT0_CNTL_BASE_IDX                                                                  1
#define regMMVM_INVALIDATE_ENG17_SEM                                                                    0x0762
#define regMMVM_INVALIDATE_ENG17_SEM_BASE_IDX                                                           1
#define regMMVM_INVALIDATE_ENG17_REQ                                                                    0x0774
#define regMMVM_INVALIDATE_ENG17_REQ_BASE_IDX                                                           1
#define regMMVM_INVALIDATE_ENG17_ACK                                                                    0x0786
#define regMMVM_INVALIDATE_ENG17_ACK_BASE_IDX                                                           1
#define regMMVM_INVALIDATE_ENG0_ADDR_RANGE_LO32                                                         0x0787
#define regMMVM_INVALIDATE_ENG0_ADDR_RANGE_LO32_BASE_IDX                                                1
#define regMMVM_INVALIDATE_ENG0_ADDR_RANGE_HI32                                                         0x0788
#define regMMVM_INVALIDATE_ENG0_ADDR_RANGE_HI32_BASE_IDX                                                1
#define regMMVM_INVALIDATE_ENG1_ADDR_RANGE_LO32                                                         0x0789
#define regMMVM_INVALIDATE_ENG1_ADDR_RANGE_LO32_BASE_IDX                                                1
#define regMMVM_INVALIDATE_ENG1_ADDR_RANGE_HI32                                                         0x078a
#define regMMVM_INVALIDATE_ENG1_ADDR_RANGE_HI32_BASE_IDX                                                1
#define regMMVM_INVALIDATE_ENG2_ADDR_RANGE_LO32                                                         0x078b
#define regMMVM_INVALIDATE_ENG2_ADDR_RANGE_LO32_BASE_IDX                                                1
#define regMMVM_INVALIDATE_ENG2_ADDR_RANGE_HI32                                                         0x078c
#define regMMVM_INVALIDATE_ENG2_ADDR_RANGE_HI32_BASE_IDX                                                1
#define regMMVM_INVALIDATE_ENG3_ADDR_RANGE_LO32                                                         0x078d
#define regMMVM_INVALIDATE_ENG3_ADDR_RANGE_LO32_BASE_IDX                                                1
#define regMMVM_INVALIDATE_ENG3_ADDR_RANGE_HI32                                                         0x078e
#define regMMVM_INVALIDATE_ENG3_ADDR_RANGE_HI32_BASE_IDX                                                1
#define regMMVM_INVALIDATE_ENG4_ADDR_RANGE_LO32                                                         0x078f
#define regMMVM_INVALIDATE_ENG4_ADDR_RANGE_LO32_BASE_IDX                                                1
#define regMMVM_INVALIDATE_ENG4_ADDR_RANGE_HI32                                                         0x0790
#define regMMVM_INVALIDATE_ENG4_ADDR_RANGE_HI32_BASE_IDX                                                1
#define regMMVM_INVALIDATE_ENG5_ADDR_RANGE_LO32                                                         0x0791
#define regMMVM_INVALIDATE_ENG5_ADDR_RANGE_LO32_BASE_IDX                                                1
#define regMMVM_INVALIDATE_ENG5_ADDR_RANGE_HI32                                                         0x0792
#define regMMVM_INVALIDATE_ENG5_ADDR_RANGE_HI32_BASE_IDX                                                1
#define regMMVM_INVALIDATE_ENG6_ADDR_RANGE_LO32                                                         0x0793
#define regMMVM_INVALIDATE_ENG6_ADDR_RANGE_LO32_BASE_IDX                                                1
#define regMMVM_INVALIDATE_ENG6_ADDR_RANGE_HI32                                                         0x0794
#define regMMVM_INVALIDATE_ENG6_ADDR_RANGE_HI32_BASE_IDX                                                1
#define regMMVM_INVALIDATE_ENG7_ADDR_RANGE_LO32                                                         0x0795
#define regMMVM_INVALIDATE_ENG7_ADDR_RANGE_LO32_BASE_IDX                                                1
#define regMMVM_INVALIDATE_ENG7_ADDR_RANGE_HI32                                                         0x0796
#define regMMVM_INVALIDATE_ENG7_ADDR_RANGE_HI32_BASE_IDX                                                1
#define regMMVM_INVALIDATE_ENG8_ADDR_RANGE_LO32                                                         0x0797
#define regMMVM_INVALIDATE_ENG8_ADDR_RANGE_LO32_BASE_IDX                                                1
#define regMMVM_INVALIDATE_ENG8_ADDR_RANGE_HI32                                                         0x0798
#define regMMVM_INVALIDATE_ENG8_ADDR_RANGE_HI32_BASE_IDX                                                1
#define regMMVM_INVALIDATE_ENG9_ADDR_RANGE_LO32                                                         0x0799
#define regMMVM_INVALIDATE_ENG9_ADDR_RANGE_LO32_BASE_IDX                                                1
#define regMMVM_INVALIDATE_ENG9_ADDR_RANGE_HI32                                                         0x079a
#define regMMVM_INVALIDATE_ENG9_ADDR_RANGE_HI32_BASE_IDX                                                1
#define regMMVM_INVALIDATE_ENG10_ADDR_RANGE_LO32                                                        0x079b
#define regMMVM_INVALIDATE_ENG10_ADDR_RANGE_LO32_BASE_IDX                                               1
#define regMMVM_INVALIDATE_ENG10_ADDR_RANGE_HI32                                                        0x079c
#define regMMVM_INVALIDATE_ENG10_ADDR_RANGE_HI32_BASE_IDX                                               1
#define regMMVM_INVALIDATE_ENG11_ADDR_RANGE_LO32                                                        0x079d
#define regMMVM_INVALIDATE_ENG11_ADDR_RANGE_LO32_BASE_IDX                                               1
#define regMMVM_INVALIDATE_ENG11_ADDR_RANGE_HI32                                                        0x079e
#define regMMVM_INVALIDATE_ENG11_ADDR_RANGE_HI32_BASE_IDX                                               1
#define regMMVM_INVALIDATE_ENG12_ADDR_RANGE_LO32                                                        0x079f
#define regMMVM_INVALIDATE_ENG12_ADDR_RANGE_LO32_BASE_IDX                                               1
#define regMMVM_INVALIDATE_ENG12_ADDR_RANGE_HI32                                                        0x07a0
#define regMMVM_INVALIDATE_ENG12_ADDR_RANGE_HI32_BASE_IDX                                               1
#define regMMVM_INVALIDATE_ENG13_ADDR_RANGE_LO32                                                        0x07a1
#define regMMVM_INVALIDATE_ENG13_ADDR_RANGE_LO32_BASE_IDX                                               1
#define regMMVM_INVALIDATE_ENG13_ADDR_RANGE_HI32                                                        0x07a2
#define regMMVM_INVALIDATE_ENG13_ADDR_RANGE_HI32_BASE_IDX                                               1
#define regMMVM_INVALIDATE_ENG14_ADDR_RANGE_LO32                                                        0x07a3
#define regMMVM_INVALIDATE_ENG14_ADDR_RANGE_LO32_BASE_IDX                                               1
#define regMMVM_INVALIDATE_ENG14_ADDR_RANGE_HI32                                                        0x07a4
#define regMMVM_INVALIDATE_ENG14_ADDR_RANGE_HI32_BASE_IDX                                               1
#define regMMVM_INVALIDATE_ENG15_ADDR_RANGE_LO32                                                        0x07a5
#define regMMVM_INVALIDATE_ENG15_ADDR_RANGE_LO32_BASE_IDX                                               1
#define regMMVM_INVALIDATE_ENG15_ADDR_RANGE_HI32                                                        0x07a6
#define regMMVM_INVALIDATE_ENG15_ADDR_RANGE_HI32_BASE_IDX                                               1
#define regMMVM_INVALIDATE_ENG16_ADDR_RANGE_LO32                                                        0x07a7
#define regMMVM_INVALIDATE_ENG16_ADDR_RANGE_LO32_BASE_IDX                                               1
#define regMMVM_INVALIDATE_ENG16_ADDR_RANGE_HI32                                                        0x07a8
#define regMMVM_INVALIDATE_ENG16_ADDR_RANGE_HI32_BASE_IDX                                               1
#define regMMVM_INVALIDATE_ENG17_ADDR_RANGE_LO32                                                        0x07a9
#define regMMVM_INVALIDATE_ENG17_ADDR_RANGE_LO32_BASE_IDX                                               1
#define regMMVM_INVALIDATE_ENG17_ADDR_RANGE_HI32                                                        0x07aa
#define regMMVM_INVALIDATE_ENG17_ADDR_RANGE_HI32_BASE_IDX                                               1
#define regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32                                                      0x07ab
#define regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32_BASE_IDX                                             1
#define regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32                                                      0x07ac
#define regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32_BASE_IDX                                             1
#define regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32                                                     0x07cb
#define regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32_BASE_IDX                                            1
#define regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32                                                     0x07cc
#define regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32_BASE_IDX                                            1
#define regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32                                                       0x07eb
#define regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32_BASE_IDX                                              1
#define regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32                                                       0x07ec
#define regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32_BASE_IDX                                              1
#define regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB                                                     0x08d8
#define regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB_BASE_IDX                                            1
#define regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB                                                     0x08d9
#define regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB_BASE_IDX                                            1
#define regMMMC_VM_FB_LOCATION_BASE                                                                     0x08ec
#define regMMMC_VM_FB_LOCATION_BASE_BASE_IDX                                                            1
#define regMMMC_VM_FB_LOCATION_TOP                                                                      0x08ed
#define regMMMC_VM_FB_LOCATION_TOP_BASE_IDX                                                             1
#define regMMMC_VM_AGP_TOP                                                                              0x08ee
#define regMMMC_VM_AGP_TOP_BASE_IDX                                                                     1
#define regMMMC_VM_AGP_BOT                                                                              0x08ef
#define regMMMC_VM_AGP_BOT_BASE_IDX                                                                     1
#define regMMMC_VM_AGP_BASE                                                                             0x08f0
#define regMMMC_VM_AGP_BASE_BASE_IDX                                                                    1
#define regMMMC_VM_SYSTEM_APERTURE_LOW_ADDR                                                             0x08f1
#define regMMMC_VM_SYSTEM_APERTURE_LOW_ADDR_BASE_IDX                                                    1
#define regMMMC_VM_SYSTEM_APERTURE_HIGH_ADDR                                                            0x08f2
#define regMMMC_VM_SYSTEM_APERTURE_HIGH_ADDR_BASE_IDX                                                   1
#define regMMMC_VM_MX_L1_TLB_CNTL                                                                       0x08f3
#define regMMMC_VM_MX_L1_TLB_CNTL_BASE_IDX                                                              1
