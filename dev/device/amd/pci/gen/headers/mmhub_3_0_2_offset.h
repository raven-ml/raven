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

#define regMMVM_L2_CNTL                                                                                 0x0680
#define regMMVM_L2_CNTL_BASE_IDX                                                                        0
#define regMMVM_L2_CNTL2                                                                                0x0681
#define regMMVM_L2_CNTL2_BASE_IDX                                                                       0
#define regMMVM_L2_CNTL3                                                                                0x0682
#define regMMVM_L2_CNTL3_BASE_IDX                                                                       0
#define regMMVM_L2_PROTECTION_FAULT_CNTL                                                                0x0688
#define regMMVM_L2_PROTECTION_FAULT_CNTL_BASE_IDX                                                       0
#define regMMVM_L2_PROTECTION_FAULT_CNTL2                                                               0x0689
#define regMMVM_L2_PROTECTION_FAULT_CNTL2_BASE_IDX                                                      0
#define regMMVM_L2_PROTECTION_FAULT_STATUS                                                              0x068c
#define regMMVM_L2_PROTECTION_FAULT_STATUS_BASE_IDX                                                     0
#define regMMVM_L2_PROTECTION_FAULT_ADDR_LO32                                                           0x068d
#define regMMVM_L2_PROTECTION_FAULT_ADDR_LO32_BASE_IDX                                                  0
#define regMMVM_L2_PROTECTION_FAULT_ADDR_HI32                                                           0x068e
#define regMMVM_L2_PROTECTION_FAULT_ADDR_HI32_BASE_IDX                                                  0
#define regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32                                                   0x068f
#define regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32_BASE_IDX                                          0
#define regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32                                                   0x0690
#define regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32_BASE_IDX                                          0
#define regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32                                             0x0692
#define regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32_BASE_IDX                                    0
#define regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32                                             0x0693
#define regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32_BASE_IDX                                    0
#define regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32                                            0x0694
#define regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32_BASE_IDX                                   0
#define regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32                                            0x0695
#define regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32_BASE_IDX                                   0
#define regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32                                                0x0696
#define regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32_BASE_IDX                                       0
#define regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32                                                0x0697
#define regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32_BASE_IDX                                       0
#define regMMVM_L2_CNTL4                                                                                0x0698
#define regMMVM_L2_CNTL4_BASE_IDX                                                                       0
#define regMMVM_L2_BANK_SELECT_RESERVED_CID2                                                            0x069b
#define regMMVM_L2_BANK_SELECT_RESERVED_CID2_BASE_IDX                                                   0
#define regMMVM_L2_CNTL5                                                                                0x069e
#define regMMVM_L2_CNTL5_BASE_IDX                                                                       0
#define regMMVM_CONTEXT0_CNTL                                                                           0x06c0
#define regMMVM_CONTEXT0_CNTL_BASE_IDX                                                                  0
#define regMMVM_INVALIDATE_ENG17_SEM                                                                    0x06e2
#define regMMVM_INVALIDATE_ENG17_SEM_BASE_IDX                                                           0
#define regMMVM_INVALIDATE_ENG17_REQ                                                                    0x06f4
#define regMMVM_INVALIDATE_ENG17_REQ_BASE_IDX                                                           0
#define regMMVM_INVALIDATE_ENG17_ACK                                                                    0x0706
#define regMMVM_INVALIDATE_ENG17_ACK_BASE_IDX                                                           0
#define regMMVM_INVALIDATE_ENG0_ADDR_RANGE_LO32                                                         0x0707
#define regMMVM_INVALIDATE_ENG0_ADDR_RANGE_LO32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG0_ADDR_RANGE_HI32                                                         0x0708
#define regMMVM_INVALIDATE_ENG0_ADDR_RANGE_HI32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG1_ADDR_RANGE_LO32                                                         0x0709
#define regMMVM_INVALIDATE_ENG1_ADDR_RANGE_LO32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG1_ADDR_RANGE_HI32                                                         0x070a
#define regMMVM_INVALIDATE_ENG1_ADDR_RANGE_HI32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG2_ADDR_RANGE_LO32                                                         0x070b
#define regMMVM_INVALIDATE_ENG2_ADDR_RANGE_LO32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG2_ADDR_RANGE_HI32                                                         0x070c
#define regMMVM_INVALIDATE_ENG2_ADDR_RANGE_HI32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG3_ADDR_RANGE_LO32                                                         0x070d
#define regMMVM_INVALIDATE_ENG3_ADDR_RANGE_LO32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG3_ADDR_RANGE_HI32                                                         0x070e
#define regMMVM_INVALIDATE_ENG3_ADDR_RANGE_HI32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG4_ADDR_RANGE_LO32                                                         0x070f
#define regMMVM_INVALIDATE_ENG4_ADDR_RANGE_LO32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG4_ADDR_RANGE_HI32                                                         0x0710
#define regMMVM_INVALIDATE_ENG4_ADDR_RANGE_HI32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG5_ADDR_RANGE_LO32                                                         0x0711
#define regMMVM_INVALIDATE_ENG5_ADDR_RANGE_LO32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG5_ADDR_RANGE_HI32                                                         0x0712
#define regMMVM_INVALIDATE_ENG5_ADDR_RANGE_HI32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG6_ADDR_RANGE_LO32                                                         0x0713
#define regMMVM_INVALIDATE_ENG6_ADDR_RANGE_LO32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG6_ADDR_RANGE_HI32                                                         0x0714
#define regMMVM_INVALIDATE_ENG6_ADDR_RANGE_HI32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG7_ADDR_RANGE_LO32                                                         0x0715
#define regMMVM_INVALIDATE_ENG7_ADDR_RANGE_LO32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG7_ADDR_RANGE_HI32                                                         0x0716
#define regMMVM_INVALIDATE_ENG7_ADDR_RANGE_HI32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG8_ADDR_RANGE_LO32                                                         0x0717
#define regMMVM_INVALIDATE_ENG8_ADDR_RANGE_LO32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG8_ADDR_RANGE_HI32                                                         0x0718
#define regMMVM_INVALIDATE_ENG8_ADDR_RANGE_HI32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG9_ADDR_RANGE_LO32                                                         0x0719
#define regMMVM_INVALIDATE_ENG9_ADDR_RANGE_LO32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG9_ADDR_RANGE_HI32                                                         0x071a
#define regMMVM_INVALIDATE_ENG9_ADDR_RANGE_HI32_BASE_IDX                                                0
#define regMMVM_INVALIDATE_ENG10_ADDR_RANGE_LO32                                                        0x071b
#define regMMVM_INVALIDATE_ENG10_ADDR_RANGE_LO32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG10_ADDR_RANGE_HI32                                                        0x071c
#define regMMVM_INVALIDATE_ENG10_ADDR_RANGE_HI32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG11_ADDR_RANGE_LO32                                                        0x071d
#define regMMVM_INVALIDATE_ENG11_ADDR_RANGE_LO32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG11_ADDR_RANGE_HI32                                                        0x071e
#define regMMVM_INVALIDATE_ENG11_ADDR_RANGE_HI32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG12_ADDR_RANGE_LO32                                                        0x071f
#define regMMVM_INVALIDATE_ENG12_ADDR_RANGE_LO32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG12_ADDR_RANGE_HI32                                                        0x0720
#define regMMVM_INVALIDATE_ENG12_ADDR_RANGE_HI32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG13_ADDR_RANGE_LO32                                                        0x0721
#define regMMVM_INVALIDATE_ENG13_ADDR_RANGE_LO32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG13_ADDR_RANGE_HI32                                                        0x0722
#define regMMVM_INVALIDATE_ENG13_ADDR_RANGE_HI32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG14_ADDR_RANGE_LO32                                                        0x0723
#define regMMVM_INVALIDATE_ENG14_ADDR_RANGE_LO32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG14_ADDR_RANGE_HI32                                                        0x0724
#define regMMVM_INVALIDATE_ENG14_ADDR_RANGE_HI32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG15_ADDR_RANGE_LO32                                                        0x0725
#define regMMVM_INVALIDATE_ENG15_ADDR_RANGE_LO32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG15_ADDR_RANGE_HI32                                                        0x0726
#define regMMVM_INVALIDATE_ENG15_ADDR_RANGE_HI32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG16_ADDR_RANGE_LO32                                                        0x0727
#define regMMVM_INVALIDATE_ENG16_ADDR_RANGE_LO32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG16_ADDR_RANGE_HI32                                                        0x0728
#define regMMVM_INVALIDATE_ENG16_ADDR_RANGE_HI32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG17_ADDR_RANGE_LO32                                                        0x0729
#define regMMVM_INVALIDATE_ENG17_ADDR_RANGE_LO32_BASE_IDX                                               0
#define regMMVM_INVALIDATE_ENG17_ADDR_RANGE_HI32                                                        0x072a
#define regMMVM_INVALIDATE_ENG17_ADDR_RANGE_HI32_BASE_IDX                                               0
#define regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32                                                      0x072b
#define regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32_BASE_IDX                                             0
#define regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32                                                      0x072c
#define regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32_BASE_IDX                                             0
#define regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32                                                     0x074b
#define regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32_BASE_IDX                                            0
#define regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32                                                     0x074c
#define regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32_BASE_IDX                                            0
#define regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32                                                       0x076b
#define regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32_BASE_IDX                                              0
#define regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32                                                       0x076c
#define regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32_BASE_IDX                                              0
#define regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB                                                     0x0858
#define regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB_BASE_IDX                                            0
#define regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB                                                     0x0859
#define regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB_BASE_IDX                                            0
#define regMMMC_VM_FB_LOCATION_BASE                                                                     0x086c
#define regMMMC_VM_FB_LOCATION_BASE_BASE_IDX                                                            0
#define regMMMC_VM_FB_LOCATION_TOP                                                                      0x086d
#define regMMMC_VM_FB_LOCATION_TOP_BASE_IDX                                                             0
#define regMMMC_VM_AGP_TOP                                                                              0x086e
#define regMMMC_VM_AGP_TOP_BASE_IDX                                                                     0
#define regMMMC_VM_AGP_BOT                                                                              0x086f
#define regMMMC_VM_AGP_BOT_BASE_IDX                                                                     0
#define regMMMC_VM_AGP_BASE                                                                             0x0870
#define regMMMC_VM_AGP_BASE_BASE_IDX                                                                    0
#define regMMMC_VM_SYSTEM_APERTURE_LOW_ADDR                                                             0x0871
#define regMMMC_VM_SYSTEM_APERTURE_LOW_ADDR_BASE_IDX                                                    0
#define regMMMC_VM_SYSTEM_APERTURE_HIGH_ADDR                                                            0x0872
#define regMMMC_VM_SYSTEM_APERTURE_HIGH_ADDR_BASE_IDX                                                   0
#define regMMMC_VM_MX_L1_TLB_CNTL                                                                       0x0873
#define regMMMC_VM_MX_L1_TLB_CNTL_BASE_IDX                                                              0
