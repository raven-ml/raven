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

#define regVM_L2_CNTL                                                                                   0x0b60
#define regVM_L2_CNTL_BASE_IDX                                                                          0
#define regVM_L2_CNTL2                                                                                  0x0b61
#define regVM_L2_CNTL2_BASE_IDX                                                                         0
#define regVM_L2_CNTL3                                                                                  0x0b62
#define regVM_L2_CNTL3_BASE_IDX                                                                         0
#define regVM_L2_PROTECTION_FAULT_CNTL                                                                  0x0b67
#define regVM_L2_PROTECTION_FAULT_CNTL_BASE_IDX                                                         0
#define regVM_L2_PROTECTION_FAULT_CNTL2                                                                 0x0b68
#define regVM_L2_PROTECTION_FAULT_CNTL2_BASE_IDX                                                        0
#define regVM_L2_PROTECTION_FAULT_STATUS                                                                0x0b6b
#define regVM_L2_PROTECTION_FAULT_STATUS_BASE_IDX                                                       0
#define regVM_L2_PROTECTION_FAULT_ADDR_LO32                                                             0x0b6c
#define regVM_L2_PROTECTION_FAULT_ADDR_LO32_BASE_IDX                                                    0
#define regVM_L2_PROTECTION_FAULT_ADDR_HI32                                                             0x0b6d
#define regVM_L2_PROTECTION_FAULT_ADDR_HI32_BASE_IDX                                                    0
#define regVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32                                                     0x0b6e
#define regVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32_BASE_IDX                                            0
#define regVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32                                                     0x0b6f
#define regVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32_BASE_IDX                                            0
#define regVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32                                               0x0b71
#define regVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32_BASE_IDX                                      0
#define regVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32                                               0x0b72
#define regVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32_BASE_IDX                                      0
#define regVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32                                              0x0b73
#define regVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32_BASE_IDX                                     0
#define regVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32                                              0x0b74
#define regVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32_BASE_IDX                                     0
#define regVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32                                                  0x0b75
#define regVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32_BASE_IDX                                         0
#define regVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32                                                  0x0b76
#define regVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32_BASE_IDX                                         0
#define regVM_L2_CNTL4                                                                                  0x0b77
#define regVM_L2_CNTL4_BASE_IDX                                                                         0
#define regVM_L2_CNTL5                                                                                  0x0b78
#define regVM_L2_CNTL5_BASE_IDX                                                                         0
#define regVM_L2_BANK_SELECT_RESERVED_CID2                                                              0x0b7b
#define regVM_L2_BANK_SELECT_RESERVED_CID2_BASE_IDX                                                     0
#define regVM_CONTEXT0_CNTL                                                                             0x0ba0
#define regVM_CONTEXT0_CNTL_BASE_IDX                                                                    0
#define regVM_INVALIDATE_ENG17_SEM                                                                      0x0bc2
#define regVM_INVALIDATE_ENG17_SEM_BASE_IDX                                                             0
#define regVM_INVALIDATE_ENG17_REQ                                                                      0x0bd4
#define regVM_INVALIDATE_ENG17_REQ_BASE_IDX                                                             0
#define regVM_INVALIDATE_ENG17_ACK                                                                      0x0be6
#define regVM_INVALIDATE_ENG17_ACK_BASE_IDX                                                             0
#define regVM_INVALIDATE_ENG0_ADDR_RANGE_LO32                                                           0x0be7
#define regVM_INVALIDATE_ENG0_ADDR_RANGE_LO32_BASE_IDX                                                  0
#define regVM_INVALIDATE_ENG0_ADDR_RANGE_HI32                                                           0x0be8
#define regVM_INVALIDATE_ENG0_ADDR_RANGE_HI32_BASE_IDX                                                  0
#define regVM_INVALIDATE_ENG1_ADDR_RANGE_LO32                                                           0x0be9
#define regVM_INVALIDATE_ENG1_ADDR_RANGE_LO32_BASE_IDX                                                  0
#define regVM_INVALIDATE_ENG1_ADDR_RANGE_HI32                                                           0x0bea
#define regVM_INVALIDATE_ENG1_ADDR_RANGE_HI32_BASE_IDX                                                  0
#define regVM_INVALIDATE_ENG2_ADDR_RANGE_LO32                                                           0x0beb
#define regVM_INVALIDATE_ENG2_ADDR_RANGE_LO32_BASE_IDX                                                  0
#define regVM_INVALIDATE_ENG2_ADDR_RANGE_HI32                                                           0x0bec
#define regVM_INVALIDATE_ENG2_ADDR_RANGE_HI32_BASE_IDX                                                  0
#define regVM_INVALIDATE_ENG3_ADDR_RANGE_LO32                                                           0x0bed
#define regVM_INVALIDATE_ENG3_ADDR_RANGE_LO32_BASE_IDX                                                  0
#define regVM_INVALIDATE_ENG3_ADDR_RANGE_HI32                                                           0x0bee
#define regVM_INVALIDATE_ENG3_ADDR_RANGE_HI32_BASE_IDX                                                  0
#define regVM_INVALIDATE_ENG4_ADDR_RANGE_LO32                                                           0x0bef
#define regVM_INVALIDATE_ENG4_ADDR_RANGE_LO32_BASE_IDX                                                  0
#define regVM_INVALIDATE_ENG4_ADDR_RANGE_HI32                                                           0x0bf0
#define regVM_INVALIDATE_ENG4_ADDR_RANGE_HI32_BASE_IDX                                                  0
#define regVM_INVALIDATE_ENG5_ADDR_RANGE_LO32                                                           0x0bf1
#define regVM_INVALIDATE_ENG5_ADDR_RANGE_LO32_BASE_IDX                                                  0
#define regVM_INVALIDATE_ENG5_ADDR_RANGE_HI32                                                           0x0bf2
#define regVM_INVALIDATE_ENG5_ADDR_RANGE_HI32_BASE_IDX                                                  0
#define regVM_INVALIDATE_ENG6_ADDR_RANGE_LO32                                                           0x0bf3
#define regVM_INVALIDATE_ENG6_ADDR_RANGE_LO32_BASE_IDX                                                  0
#define regVM_INVALIDATE_ENG6_ADDR_RANGE_HI32                                                           0x0bf4
#define regVM_INVALIDATE_ENG6_ADDR_RANGE_HI32_BASE_IDX                                                  0
#define regVM_INVALIDATE_ENG7_ADDR_RANGE_LO32                                                           0x0bf5
#define regVM_INVALIDATE_ENG7_ADDR_RANGE_LO32_BASE_IDX                                                  0
#define regVM_INVALIDATE_ENG7_ADDR_RANGE_HI32                                                           0x0bf6
#define regVM_INVALIDATE_ENG7_ADDR_RANGE_HI32_BASE_IDX                                                  0
#define regVM_INVALIDATE_ENG8_ADDR_RANGE_LO32                                                           0x0bf7
#define regVM_INVALIDATE_ENG8_ADDR_RANGE_LO32_BASE_IDX                                                  0
#define regVM_INVALIDATE_ENG8_ADDR_RANGE_HI32                                                           0x0bf8
#define regVM_INVALIDATE_ENG8_ADDR_RANGE_HI32_BASE_IDX                                                  0
#define regVM_INVALIDATE_ENG9_ADDR_RANGE_LO32                                                           0x0bf9
#define regVM_INVALIDATE_ENG9_ADDR_RANGE_LO32_BASE_IDX                                                  0
#define regVM_INVALIDATE_ENG9_ADDR_RANGE_HI32                                                           0x0bfa
#define regVM_INVALIDATE_ENG9_ADDR_RANGE_HI32_BASE_IDX                                                  0
#define regVM_INVALIDATE_ENG10_ADDR_RANGE_LO32                                                          0x0bfb
#define regVM_INVALIDATE_ENG10_ADDR_RANGE_LO32_BASE_IDX                                                 0
#define regVM_INVALIDATE_ENG10_ADDR_RANGE_HI32                                                          0x0bfc
#define regVM_INVALIDATE_ENG10_ADDR_RANGE_HI32_BASE_IDX                                                 0
#define regVM_INVALIDATE_ENG11_ADDR_RANGE_LO32                                                          0x0bfd
#define regVM_INVALIDATE_ENG11_ADDR_RANGE_LO32_BASE_IDX                                                 0
#define regVM_INVALIDATE_ENG11_ADDR_RANGE_HI32                                                          0x0bfe
#define regVM_INVALIDATE_ENG11_ADDR_RANGE_HI32_BASE_IDX                                                 0
#define regVM_INVALIDATE_ENG12_ADDR_RANGE_LO32                                                          0x0bff
#define regVM_INVALIDATE_ENG12_ADDR_RANGE_LO32_BASE_IDX                                                 0
#define regVM_INVALIDATE_ENG12_ADDR_RANGE_HI32                                                          0x0c00
#define regVM_INVALIDATE_ENG12_ADDR_RANGE_HI32_BASE_IDX                                                 0
#define regVM_INVALIDATE_ENG13_ADDR_RANGE_LO32                                                          0x0c01
#define regVM_INVALIDATE_ENG13_ADDR_RANGE_LO32_BASE_IDX                                                 0
#define regVM_INVALIDATE_ENG13_ADDR_RANGE_HI32                                                          0x0c02
#define regVM_INVALIDATE_ENG13_ADDR_RANGE_HI32_BASE_IDX                                                 0
#define regVM_INVALIDATE_ENG14_ADDR_RANGE_LO32                                                          0x0c03
#define regVM_INVALIDATE_ENG14_ADDR_RANGE_LO32_BASE_IDX                                                 0
#define regVM_INVALIDATE_ENG14_ADDR_RANGE_HI32                                                          0x0c04
#define regVM_INVALIDATE_ENG14_ADDR_RANGE_HI32_BASE_IDX                                                 0
#define regVM_INVALIDATE_ENG15_ADDR_RANGE_LO32                                                          0x0c05
#define regVM_INVALIDATE_ENG15_ADDR_RANGE_LO32_BASE_IDX                                                 0
#define regVM_INVALIDATE_ENG15_ADDR_RANGE_HI32                                                          0x0c06
#define regVM_INVALIDATE_ENG15_ADDR_RANGE_HI32_BASE_IDX                                                 0
#define regVM_INVALIDATE_ENG16_ADDR_RANGE_LO32                                                          0x0c07
#define regVM_INVALIDATE_ENG16_ADDR_RANGE_LO32_BASE_IDX                                                 0
#define regVM_INVALIDATE_ENG16_ADDR_RANGE_HI32                                                          0x0c08
#define regVM_INVALIDATE_ENG16_ADDR_RANGE_HI32_BASE_IDX                                                 0
#define regVM_INVALIDATE_ENG17_ADDR_RANGE_LO32                                                          0x0c09
#define regVM_INVALIDATE_ENG17_ADDR_RANGE_LO32_BASE_IDX                                                 0
#define regVM_INVALIDATE_ENG17_ADDR_RANGE_HI32                                                          0x0c0a
#define regVM_INVALIDATE_ENG17_ADDR_RANGE_HI32_BASE_IDX                                                 0
#define regVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32                                                        0x0c0b
#define regVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32_BASE_IDX                                               0
#define regVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32                                                        0x0c0c
#define regVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32_BASE_IDX                                               0
#define regVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32                                                       0x0c2b
#define regVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32_BASE_IDX                                              0
#define regVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32                                                       0x0c2c
#define regVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32_BASE_IDX                                              0
#define regVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32                                                         0x0c4b
#define regVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32_BASE_IDX                                                0
#define regVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32                                                         0x0c4c
#define regVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32_BASE_IDX                                                0
#define regMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB                                                       0x0c88
#define regMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB_BASE_IDX                                              0
#define regMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB                                                       0x0c89
#define regMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB_BASE_IDX                                              0
#define regMC_VM_XGMI_LFB_CNTL                                                                          0x0c97
#define regMC_VM_XGMI_LFB_CNTL_BASE_IDX                                                                 0
#define regMC_VM_XGMI_LFB_SIZE                                                                          0x0c98
#define regMC_VM_XGMI_LFB_SIZE_BASE_IDX                                                                 0
#define regMC_VM_FB_LOCATION_BASE                                                                       0x0c9c
#define regMC_VM_FB_LOCATION_BASE_BASE_IDX                                                              0
#define regMC_VM_FB_LOCATION_TOP                                                                        0x0c9d
#define regMC_VM_FB_LOCATION_TOP_BASE_IDX                                                               0
#define regMC_VM_AGP_TOP                                                                                0x0c9e
#define regMC_VM_AGP_TOP_BASE_IDX                                                                       0
#define regMC_VM_AGP_BOT                                                                                0x0c9f
#define regMC_VM_AGP_BOT_BASE_IDX                                                                       0
#define regMC_VM_AGP_BASE                                                                               0x0ca0
#define regMC_VM_AGP_BASE_BASE_IDX                                                                      0
#define regMC_VM_SYSTEM_APERTURE_LOW_ADDR                                                               0x0ca1
#define regMC_VM_SYSTEM_APERTURE_LOW_ADDR_BASE_IDX                                                      0
#define regMC_VM_SYSTEM_APERTURE_HIGH_ADDR                                                              0x0ca2
#define regMC_VM_SYSTEM_APERTURE_HIGH_ADDR_BASE_IDX                                                     0
#define regMC_VM_MX_L1_TLB_CNTL                                                                         0x0ca3
#define regMC_VM_MX_L1_TLB_CNTL_BASE_IDX                                                                0
