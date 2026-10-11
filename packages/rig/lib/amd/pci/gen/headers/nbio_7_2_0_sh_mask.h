/*
 * Copyright (C) 2020  Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
 * AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#define BIF_BX_PF0_RSMU_INDEX__RSMU_INDEX_MASK                                                                0xFFFFFFFFL
#define BIF_BX_PF0_RSMU_DATA__RSMU_DATA_MASK                                                                  0xFFFFFFFFL
#define BIF_BX0_PCIE_INDEX2__PCIE_INDEX2_MASK                                                                 0xFFFFFFFFL
#define BIF_BX0_PCIE_DATA2__PCIE_DATA2_MASK                                                                   0xFFFFFFFFL
#define BIF_BX0_INTERRUPT_CNTL__IH_DUMMY_RD_OVERRIDE_MASK                                                     0x00000001L
#define BIF_BX0_INTERRUPT_CNTL__IH_DUMMY_RD_EN_MASK                                                           0x00000002L
#define BIF_BX0_INTERRUPT_CNTL__IH_REQ_NONSNOOP_EN_MASK                                                       0x00000008L
#define BIF_BX0_INTERRUPT_CNTL__IH_INTR_DLY_CNTR_MASK                                                         0x000000F0L
#define BIF_BX0_INTERRUPT_CNTL__BIF_RB_REQ_NONSNOOP_EN_MASK                                                   0x00008000L
#define BIF_BX0_INTERRUPT_CNTL__DUMMYRD_BYPASS_IN_MSI_EN_MASK                                                 0x00010000L
#define BIF_BX0_INTERRUPT_CNTL__ALWAYS_SEND_INTPKT_AFTER_DUMMYRD_DIS_MASK                                     0x00020000L
#define BIF_BX0_INTERRUPT_CNTL__BIF_RB_REQ_RELAX_ORDER_EN_MASK                                                0x00040000L
#define BIF_BX0_INTERRUPT_CNTL2__IH_DUMMY_RD_ADDR_MASK                                                        0xFFFFFFFFL
#define BIF_BX0_BIF_DOORBELL_INT_CNTL__DOORBELL_INTERRUPT_STATUS_MASK                                         0x00000001L
#define BIF_BX0_BIF_DOORBELL_INT_CNTL__RAS_CNTLR_INTERRUPT_STATUS_MASK                                        0x00000002L
#define BIF_BX0_BIF_DOORBELL_INT_CNTL__RAS_ATHUB_ERR_EVENT_INTERRUPT_STATUS_MASK                              0x00000004L
#define BIF_BX0_BIF_DOORBELL_INT_CNTL__DOORBELL_INTERRUPT_CLEAR_MASK                                          0x00010000L
#define BIF_BX0_BIF_DOORBELL_INT_CNTL__RAS_CNTLR_INTERRUPT_CLEAR_MASK                                         0x00020000L
#define BIF_BX0_BIF_DOORBELL_INT_CNTL__RAS_ATHUB_ERR_EVENT_INTERRUPT_CLEAR_MASK                               0x00040000L
#define BIF_BX0_BIF_DOORBELL_INT_CNTL__DOORBELL_INTERRUPT_DISABLE_MASK                                        0x01000000L
#define BIF_BX0_BIF_DOORBELL_INT_CNTL__RAS_CNTLR_INTERRUPT_DISABLE_MASK                                       0x02000000L
#define BIF_BX0_BIF_DOORBELL_INT_CNTL__RAS_ATHUB_ERR_EVENT_INTERRUPT_DISABLE_MASK                             0x04000000L
#define BIF_BX0_BIF_DOORBELL_INT_CNTL__SET_DB_INTR_STATUS_WHEN_RB_ENABLE_MASK                                 0x10000000L
#define BIF_BX0_BIF_DOORBELL_INT_CNTL__SET_IOH_RAS_INTR_STATUS_WHEN_RB_ENABLE_MASK                            0x20000000L
#define BIF_BX0_BIF_DOORBELL_INT_CNTL__SET_ATH_RAS_INTR_STATUS_WHEN_RB_ENABLE_MASK                            0x40000000L
#define BIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL__ADDRESS_MASK                                                        0x0007FFFCL
#define BIF_BX0_REMAP_HDP_REG_FLUSH_CNTL__ADDRESS_MASK                                                        0x0007FFFCL
#define RCC_DEV0_EPF2_STRAP2__STRAP_NO_SOFT_RESET_DEV0_F2_MASK                                                0x00000080L
#define RCC_DEV0_EPF2_STRAP2__STRAP_RESIZE_BAR_EN_DEV0_F2_MASK                                                0x00000100L
#define RCC_DEV0_EPF2_STRAP2__STRAP_MAX_PASID_WIDTH_DEV0_F2_MASK                                              0x00003E00L
#define RCC_DEV0_EPF2_STRAP2__STRAP_MSI_PERVECTOR_MASK_CAP_DEV0_F2_MASK                                       0x00004000L
#define RCC_DEV0_EPF2_STRAP2__STRAP_AER_EN_DEV0_F2_MASK                                                       0x00010000L
#define RCC_DEV0_EPF2_STRAP2__STRAP_ACS_EN_DEV0_F2_MASK                                                       0x00020000L
#define RCC_DEV0_EPF2_STRAP2__STRAP_CPL_ABORT_ERR_EN_DEV0_F2_MASK                                             0x00100000L
#define RCC_DEV0_EPF2_STRAP2__STRAP_DPA_EN_DEV0_F2_MASK                                                       0x00200000L
#define RCC_DEV0_EPF2_STRAP2__STRAP_VC_EN_DEV0_F2_MASK                                                        0x00800000L
#define RCC_DEV0_EPF2_STRAP2__STRAP_MSI_MULTI_CAP_DEV0_F2_MASK                                                0x07000000L
#define RCC_DEV0_EPF2_STRAP2__STRAP_PASID_EN_DEV0_F2_MASK                                                     0x10000000L
#define RCC_DEV0_EPF2_STRAP2__STRAP_PASID_EXE_PERMISSION_SUPPORTED_DEV0_F2_MASK                               0x20000000L
#define RCC_DEV0_EPF2_STRAP2__STRAP_PASID_GLOBAL_INVALIDATE_SUPPORTED_DEV0_F2_MASK                            0x40000000L
#define RCC_DEV0_EPF2_STRAP2__STRAP_PASID_PRIV_MODE_SUPPORTED_DEV0_F2_MASK                                    0x80000000L
