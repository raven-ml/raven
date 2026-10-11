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

#define regBIF_BX0_PCIE_INDEX2                                                                          0x000e
#define regBIF_BX0_PCIE_INDEX2_BASE_IDX                                                                 0
#define regBIF_BX0_PCIE_DATA2                                                                           0x000f
#define regBIF_BX0_PCIE_DATA2_BASE_IDX                                                                  0
#define regBIF_BX0_PCIE_INDEX2_HI                                                                       0x0011
#define regBIF_BX0_PCIE_INDEX2_HI_BASE_IDX                                                              0
#define regBIF_BX_PF0_RSMU_INDEX                                                                        0x0000
#define regBIF_BX_PF0_RSMU_INDEX_BASE_IDX                                                               1
#define regBIF_BX_PF0_RSMU_DATA                                                                         0x0001
#define regBIF_BX_PF0_RSMU_DATA_BASE_IDX                                                                1
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
#define regRCC_DEV0_EPF2_STRAP2                                                                         0xd102
#define regRCC_DEV0_EPF2_STRAP2_BASE_IDX 8
#define regDOORBELL0_CTRL_ENTRY_0                                                                       0xcd00
#define regDOORBELL0_CTRL_ENTRY_0_BASE_IDX 8
#define regDOORBELL0_CTRL_ENTRY_1                                                                       0xcd01
#define regDOORBELL0_CTRL_ENTRY_1_BASE_IDX 8
#define regDOORBELL0_CTRL_ENTRY_2                                                                       0xcd02
#define regDOORBELL0_CTRL_ENTRY_2_BASE_IDX 8
#define regDOORBELL0_CTRL_ENTRY_3                                                                       0xcd03
#define regDOORBELL0_CTRL_ENTRY_3_BASE_IDX 8
#define regDOORBELL0_CTRL_ENTRY_4                                                                       0xcd04
#define regDOORBELL0_CTRL_ENTRY_4_BASE_IDX 8
#define regDOORBELL0_CTRL_ENTRY_5                                                                       0xcd05
#define regDOORBELL0_CTRL_ENTRY_5_BASE_IDX 8
#define regDOORBELL0_CTRL_ENTRY_6                                                                       0xcd06
#define regDOORBELL0_CTRL_ENTRY_6_BASE_IDX 8
#define regDOORBELL0_CTRL_ENTRY_7                                                                       0xcd07
#define regDOORBELL0_CTRL_ENTRY_7_BASE_IDX 8
#define regDOORBELL0_CTRL_ENTRY_8                                                                       0xcd08
#define regDOORBELL0_CTRL_ENTRY_8_BASE_IDX 8
#define regDOORBELL0_CTRL_ENTRY_9                                                                       0xcd09
#define regDOORBELL0_CTRL_ENTRY_9_BASE_IDX 8
#define regDOORBELL0_CTRL_ENTRY_10                                                                      0xcd0a
#define regDOORBELL0_CTRL_ENTRY_10_BASE_IDX 8
#define regDOORBELL0_CTRL_ENTRY_11                                                                      0xcd0b
#define regDOORBELL0_CTRL_ENTRY_11_BASE_IDX 8
#define regDOORBELL0_CTRL_ENTRY_12                                                                      0xcd0c
#define regDOORBELL0_CTRL_ENTRY_12_BASE_IDX 8
#define regDOORBELL0_CTRL_ENTRY_13                                                                      0xcd0d
#define regDOORBELL0_CTRL_ENTRY_13_BASE_IDX 8
#define regDOORBELL0_CTRL_ENTRY_14                                                                      0xcd0e
#define regDOORBELL0_CTRL_ENTRY_14_BASE_IDX 8
#define regDOORBELL0_CTRL_ENTRY_15                                                                      0xcd0f
#define regDOORBELL0_CTRL_ENTRY_15_BASE_IDX 8
#define regDOORBELL0_CTRL_ENTRY_16                                                                      0xcd10
#define regDOORBELL0_CTRL_ENTRY_16_BASE_IDX 8
#define regDOORBELL0_CTRL_ENTRY_17                                                                      0xcd11
#define regDOORBELL0_CTRL_ENTRY_17_BASE_IDX 8
#define regDOORBELL0_CTRL_ENTRY_18                                                                      0xcd12
#define regDOORBELL0_CTRL_ENTRY_18_BASE_IDX 8
#define regDOORBELL0_CTRL_ENTRY_19                                                                      0xcd13
#define regDOORBELL0_CTRL_ENTRY_19_BASE_IDX 8
#define regDOORBELL0_CTRL_ENTRY_20                                                                      0xcd14
#define regDOORBELL0_CTRL_ENTRY_20_BASE_IDX 8
#define regBIFC_DOORBELL_ACCESS_EN_PF                                                                   0xcf6e
#define regBIFC_DOORBELL_ACCESS_EN_PF_BASE_IDX 8
#define regBIFC_GFX_INT_MONITOR_MASK                                                                    0xe8ad
#define regBIFC_GFX_INT_MONITOR_MASK_BASE_IDX 8
#define regS2A_DOORBELL_ENTRY_0_CTRL 0x7a80
#define regS2A_DOORBELL_ENTRY_0_CTRL_BASE_IDX 5
#define regS2A_DOORBELL_ENTRY_1_CTRL 0x7a81
#define regS2A_DOORBELL_ENTRY_1_CTRL_BASE_IDX 5
#define regS2A_DOORBELL_ENTRY_2_CTRL 0x7a82
#define regS2A_DOORBELL_ENTRY_2_CTRL_BASE_IDX 5
#define regS2A_DOORBELL_ENTRY_3_CTRL 0x7a83
#define regS2A_DOORBELL_ENTRY_3_CTRL_BASE_IDX 5
#define regS2A_DOORBELL_ENTRY_4_CTRL 0x7a84
#define regS2A_DOORBELL_ENTRY_4_CTRL_BASE_IDX 5
#define regS2A_DOORBELL_ENTRY_5_CTRL 0x7a85
#define regS2A_DOORBELL_ENTRY_5_CTRL_BASE_IDX 5
#define regS2A_DOORBELL_ENTRY_6_CTRL 0x7a86
#define regS2A_DOORBELL_ENTRY_6_CTRL_BASE_IDX 5
#define regS2A_DOORBELL_ENTRY_7_CTRL 0x7a87
#define regS2A_DOORBELL_ENTRY_7_CTRL_BASE_IDX 5
#define regS2A_DOORBELL_ENTRY_8_CTRL 0x7a88
#define regS2A_DOORBELL_ENTRY_8_CTRL_BASE_IDX 5
#define regS2A_DOORBELL_ENTRY_9_CTRL 0x7a89
#define regS2A_DOORBELL_ENTRY_9_CTRL_BASE_IDX 5
#define regS2A_DOORBELL_ENTRY_10_CTRL 0x7a8a
#define regS2A_DOORBELL_ENTRY_10_CTRL_BASE_IDX 5
#define regS2A_DOORBELL_ENTRY_11_CTRL 0x7a8b
#define regS2A_DOORBELL_ENTRY_11_CTRL_BASE_IDX 5
#define regS2A_DOORBELL_ENTRY_12_CTRL 0x7a8c
#define regS2A_DOORBELL_ENTRY_12_CTRL_BASE_IDX 5
#define regS2A_DOORBELL_ENTRY_13_CTRL 0x7a8d
#define regS2A_DOORBELL_ENTRY_13_CTRL_BASE_IDX 5
#define regS2A_DOORBELL_ENTRY_14_CTRL 0x7a8e
#define regS2A_DOORBELL_ENTRY_14_CTRL_BASE_IDX 5
#define regS2A_DOORBELL_ENTRY_15_CTRL 0x7a8f
#define regS2A_DOORBELL_ENTRY_15_CTRL_BASE_IDX 5
#define regXCC_DOORBELL_FENCE 0x740c
#define regXCC_DOORBELL_FENCE_BASE_IDX 5
#define regBIF_BX_DEV0_EPF0_VF0_HDP_MEM_COHERENCY_FLUSH_CNTL                                            0x00f7
#define regBIF_BX_DEV0_EPF0_VF0_HDP_MEM_COHERENCY_FLUSH_CNTL_BASE_IDX                                   2
