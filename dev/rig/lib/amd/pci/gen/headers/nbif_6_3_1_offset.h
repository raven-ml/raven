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

#define regBIF_BX_PF0_RSMU_INDEX                                                                        0x0000
#define regBIF_BX_PF0_RSMU_INDEX_BASE_IDX                                                               1
#define regBIF_BX_PF0_RSMU_DATA                                                                         0x0001
#define regBIF_BX_PF0_RSMU_DATA_BASE_IDX                                                                1
#define regBIF_BX0_PCIE_INDEX2                                                                          0x000e
#define regBIF_BX0_PCIE_INDEX2_BASE_IDX                                                                 0
#define regBIF_BX0_PCIE_DATA2                                                                           0x000f
#define regBIF_BX0_PCIE_DATA2_BASE_IDX                                                                  0
#define regBIF_BX0_PCIE_INDEX2_HI                                                                       0x0011
#define regBIF_BX0_PCIE_INDEX2_HI_BASE_IDX                                                              0
#define regBIF_BX0_INTERRUPT_CNTL                                                                       0x00f1
#define regBIF_BX0_INTERRUPT_CNTL_BASE_IDX                                                              2
#define regBIF_BX0_INTERRUPT_CNTL2                                                                      0x00f2
#define regBIF_BX0_INTERRUPT_CNTL2_BASE_IDX                                                             2
#define regBIF_BX0_BIF_DOORBELL_INT_CNTL                                                                0x00fe
#define regBIF_BX0_BIF_DOORBELL_INT_CNTL_BASE_IDX                                                       2
#define regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL                                                             0x012d
#define regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL_BASE_IDX                                                    2
#define regBIF_BX0_REMAP_HDP_REG_FLUSH_CNTL                                                             0x012e
#define regBIF_BX0_REMAP_HDP_REG_FLUSH_CNTL_BASE_IDX                                                    2
#define regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN                                                           0x00c0
#define regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN_BASE_IDX                                                  2
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_0_CTRL                                                           0x01cb
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_0_CTRL_BASE_IDX                                                  2
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_1_CTRL                                                           0x01cc
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_1_CTRL_BASE_IDX                                                  2
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_2_CTRL                                                           0x01cd
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_2_CTRL_BASE_IDX                                                  2
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_3_CTRL                                                           0x01ce
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_3_CTRL_BASE_IDX                                                  2
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_4_CTRL                                                           0x01cf
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_4_CTRL_BASE_IDX                                                  2
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_5_CTRL                                                           0x01d0
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_5_CTRL_BASE_IDX                                                  2
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_6_CTRL                                                           0x01d1
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_6_CTRL_BASE_IDX                                                  2
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_7_CTRL                                                           0x01d2
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_7_CTRL_BASE_IDX                                                  2
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_8_CTRL                                                           0x01d3
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_8_CTRL_BASE_IDX                                                  2
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_9_CTRL                                                           0x01d4
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_9_CTRL_BASE_IDX                                                  2
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_10_CTRL                                                          0x01d5
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_10_CTRL_BASE_IDX                                                 2
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_11_CTRL                                                          0x01d6
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_11_CTRL_BASE_IDX                                                 2
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_12_CTRL                                                          0x01d7
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_12_CTRL_BASE_IDX                                                 2
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_13_CTRL                                                          0x01d8
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_13_CTRL_BASE_IDX                                                 2
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_14_CTRL                                                          0x01d9
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_14_CTRL_BASE_IDX                                                 2
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_15_CTRL                                                          0x01da
#define regGDC_S2A0_S2A_DOORBELL_ENTRY_15_CTRL_BASE_IDX                                                 2
#define regRCC_DEV0_EPF2_STRAP2                                                                         0xd102
#define regRCC_DEV0_EPF2_STRAP2_BASE_IDX                                                                5
#define regBIF_BX_DEV0_EPF0_VF0_HDP_MEM_COHERENCY_FLUSH_CNTL                                            0x00f7
#define regBIF_BX_DEV0_EPF0_VF0_HDP_MEM_COHERENCY_FLUSH_CNTL_BASE_IDX                                   2
