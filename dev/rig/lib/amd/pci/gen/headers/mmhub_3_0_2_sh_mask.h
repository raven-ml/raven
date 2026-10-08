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

#define MMVM_L2_CNTL__ENABLE_L2_CACHE_MASK                                                                    0x00000001L
#define MMVM_L2_CNTL__ENABLE_L2_FRAGMENT_PROCESSING_MASK                                                      0x00000002L
#define MMVM_L2_CNTL__L2_CACHE_PTE_ENDIAN_SWAP_MODE_MASK                                                      0x0000000CL
#define MMVM_L2_CNTL__L2_CACHE_PDE_ENDIAN_SWAP_MODE_MASK                                                      0x00000030L
#define MMVM_L2_CNTL__L2_PDE0_CACHE_TAG_GENERATION_MODE_MASK                                                  0x00000100L
#define MMVM_L2_CNTL__ENABLE_L2_PTE_CACHE_LRU_UPDATE_BY_WRITE_MASK                                            0x00000200L
#define MMVM_L2_CNTL__ENABLE_L2_PDE0_CACHE_LRU_UPDATE_BY_WRITE_MASK                                           0x00000400L
#define MMVM_L2_CNTL__ENABLE_DEFAULT_PAGE_OUT_TO_SYSTEM_MEMORY_MASK                                           0x00000800L
#define MMVM_L2_CNTL__L2_PDE0_CACHE_SPLIT_MODE_MASK                                                           0x00007000L
#define MMVM_L2_CNTL__EFFECTIVE_L2_QUEUE_SIZE_MASK                                                            0x00038000L
#define MMVM_L2_CNTL__PDE_FAULT_CLASSIFICATION_MASK                                                           0x00040000L
#define MMVM_L2_CNTL__CONTEXT1_IDENTITY_ACCESS_MODE_MASK                                                      0x00180000L
#define MMVM_L2_CNTL__IDENTITY_MODE_FRAGMENT_SIZE_MASK                                                        0x03E00000L
#define MMVM_L2_CNTL__L2_PTE_CACHE_ADDR_MODE_MASK                                                             0x0C000000L
#define MMVM_L2_CNTL2__INVALIDATE_ALL_L1_TLBS_MASK                                                            0x00000001L
#define MMVM_L2_CNTL2__INVALIDATE_L2_CACHE_MASK                                                               0x00000002L
#define MMVM_L2_CNTL2__DISABLE_INVALIDATE_PER_DOMAIN_MASK                                                     0x00200000L
#define MMVM_L2_CNTL2__DISABLE_BIGK_CACHE_OPTIMIZATION_MASK                                                   0x00400000L
#define MMVM_L2_CNTL2__L2_PTE_CACHE_VMID_MODE_MASK                                                            0x03800000L
#define MMVM_L2_CNTL2__INVALIDATE_CACHE_MODE_MASK                                                             0x0C000000L
#define MMVM_L2_CNTL2__PDE_CACHE_EFFECTIVE_SIZE_MASK                                                          0x70000000L
#define MMVM_L2_CNTL3__BANK_SELECT_MASK                                                                       0x0000003FL
#define MMVM_L2_CNTL3__L2_CACHE_UPDATE_MODE_MASK                                                              0x000000C0L
#define MMVM_L2_CNTL3__L2_CACHE_UPDATE_WILDCARD_REFERENCE_VALUE_MASK                                          0x00001F00L
#define MMVM_L2_CNTL3__L2_CACHE_BIGK_FRAGMENT_SIZE_MASK                                                       0x000F8000L
#define MMVM_L2_CNTL3__L2_CACHE_BIGK_ASSOCIATIVITY_MASK                                                       0x00100000L
#define MMVM_L2_CNTL3__L2_CACHE_4K_EFFECTIVE_SIZE_MASK                                                        0x00E00000L
#define MMVM_L2_CNTL3__L2_CACHE_BIGK_EFFECTIVE_SIZE_MASK                                                      0x0F000000L
#define MMVM_L2_CNTL3__L2_CACHE_4K_FORCE_MISS_MASK                                                            0x10000000L
#define MMVM_L2_CNTL3__L2_CACHE_BIGK_FORCE_MISS_MASK                                                          0x20000000L
#define MMVM_L2_CNTL3__PDE_CACHE_FORCE_MISS_MASK                                                              0x40000000L
#define MMVM_L2_CNTL3__L2_CACHE_4K_ASSOCIATIVITY_MASK                                                         0x80000000L
#define MMVM_L2_PROTECTION_FAULT_CNTL__CLEAR_PROTECTION_FAULT_STATUS_ADDR_MASK                                0x00000001L
#define MMVM_L2_PROTECTION_FAULT_CNTL__ALLOW_SUBSEQUENT_PROTECTION_FAULT_STATUS_ADDR_UPDATES_MASK             0x00000002L
#define MMVM_L2_PROTECTION_FAULT_CNTL__RANGE_PROTECTION_FAULT_ENABLE_DEFAULT_MASK                             0x00000004L
#define MMVM_L2_PROTECTION_FAULT_CNTL__PDE0_PROTECTION_FAULT_ENABLE_DEFAULT_MASK                              0x00000008L
#define MMVM_L2_PROTECTION_FAULT_CNTL__PDE1_PROTECTION_FAULT_ENABLE_DEFAULT_MASK                              0x00000010L
#define MMVM_L2_PROTECTION_FAULT_CNTL__PDE2_PROTECTION_FAULT_ENABLE_DEFAULT_MASK                              0x00000020L
#define MMVM_L2_PROTECTION_FAULT_CNTL__TRANSLATE_FURTHER_PROTECTION_FAULT_ENABLE_DEFAULT_MASK                 0x00000040L
#define MMVM_L2_PROTECTION_FAULT_CNTL__NACK_PROTECTION_FAULT_ENABLE_DEFAULT_MASK                              0x00000080L
#define MMVM_L2_PROTECTION_FAULT_CNTL__DUMMY_PAGE_PROTECTION_FAULT_ENABLE_DEFAULT_MASK                        0x00000100L
#define MMVM_L2_PROTECTION_FAULT_CNTL__VALID_PROTECTION_FAULT_ENABLE_DEFAULT_MASK                             0x00000200L
#define MMVM_L2_PROTECTION_FAULT_CNTL__READ_PROTECTION_FAULT_ENABLE_DEFAULT_MASK                              0x00000400L
#define MMVM_L2_PROTECTION_FAULT_CNTL__WRITE_PROTECTION_FAULT_ENABLE_DEFAULT_MASK                             0x00000800L
#define MMVM_L2_PROTECTION_FAULT_CNTL__EXECUTE_PROTECTION_FAULT_ENABLE_DEFAULT_MASK                           0x00001000L
#define MMVM_L2_PROTECTION_FAULT_CNTL__CLIENT_ID_NO_RETRY_FAULT_INTERRUPT_MASK                                0x1FFFE000L
#define MMVM_L2_PROTECTION_FAULT_CNTL__OTHER_CLIENT_ID_NO_RETRY_FAULT_INTERRUPT_MASK                          0x20000000L
#define MMVM_L2_PROTECTION_FAULT_CNTL__CRASH_ON_NO_RETRY_FAULT_MASK                                           0x40000000L
#define MMVM_L2_PROTECTION_FAULT_CNTL__CRASH_ON_RETRY_FAULT_MASK                                              0x80000000L
#define MMVM_L2_PROTECTION_FAULT_CNTL2__CLIENT_ID_PRT_FAULT_INTERRUPT_MASK                                    0x0000FFFFL
#define MMVM_L2_PROTECTION_FAULT_CNTL2__OTHER_CLIENT_ID_PRT_FAULT_INTERRUPT_MASK                              0x00010000L
#define MMVM_L2_PROTECTION_FAULT_CNTL2__ACTIVE_PAGE_MIGRATION_PTE_MASK                                        0x00020000L
#define MMVM_L2_PROTECTION_FAULT_CNTL2__ACTIVE_PAGE_MIGRATION_PTE_READ_RETRY_MASK                             0x00040000L
#define MMVM_L2_PROTECTION_FAULT_CNTL2__ENABLE_RETRY_FAULT_INTERRUPT_MASK                                     0x00080000L
#define MMVM_L2_PROTECTION_FAULT_STATUS__MORE_FAULTS_MASK                                                     0x00000001L
#define MMVM_L2_PROTECTION_FAULT_STATUS__WALKER_ERROR_MASK                                                    0x0000000EL
#define MMVM_L2_PROTECTION_FAULT_STATUS__PERMISSION_FAULTS_MASK                                               0x000000F0L
#define MMVM_L2_PROTECTION_FAULT_STATUS__MAPPING_ERROR_MASK                                                   0x00000100L
#define MMVM_L2_PROTECTION_FAULT_STATUS__CID_MASK                                                             0x0003FE00L
#define MMVM_L2_PROTECTION_FAULT_STATUS__RW_MASK                                                              0x00040000L
#define MMVM_L2_PROTECTION_FAULT_STATUS__ATOMIC_MASK                                                          0x00080000L
#define MMVM_L2_PROTECTION_FAULT_STATUS__VMID_MASK                                                            0x00F00000L
#define MMVM_L2_PROTECTION_FAULT_STATUS__VF_MASK                                                              0x01000000L
#define MMVM_L2_PROTECTION_FAULT_STATUS__VFID_MASK                                                            0x1E000000L
#define MMVM_L2_PROTECTION_FAULT_STATUS__PRT_MASK                                                             0x20000000L
#define MMVM_L2_PROTECTION_FAULT_ADDR_LO32__LOGICAL_PAGE_ADDR_LO32_MASK                                       0xFFFFFFFFL
#define MMVM_L2_PROTECTION_FAULT_ADDR_HI32__LOGICAL_PAGE_ADDR_HI4_MASK                                        0x0000000FL
#define MMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32__PHYSICAL_PAGE_ADDR_LO32_MASK                              0xFFFFFFFFL
#define MMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32__PHYSICAL_PAGE_ADDR_HI4_MASK                               0x0000000FL
#define MMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32__LOGICAL_PAGE_NUMBER_LO32_MASK                       0xFFFFFFFFL
#define MMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32__LOGICAL_PAGE_NUMBER_HI4_MASK                        0x0000000FL
#define MMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32__LOGICAL_PAGE_NUMBER_LO32_MASK                      0xFFFFFFFFL
#define MMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32__LOGICAL_PAGE_NUMBER_HI4_MASK                       0x0000000FL
#define MMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32__PHYSICAL_PAGE_OFFSET_LO32_MASK                         0xFFFFFFFFL
#define MMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32__PHYSICAL_PAGE_OFFSET_HI4_MASK                          0x0000000FL
#define MMVM_L2_CNTL4__L2_CACHE_4K_PARTITION_COUNT_MASK                                                       0x0000003FL
#define MMVM_L2_CNTL4__VMC_TAP_PDE_REQUEST_PHYSICAL_MASK                                                      0x00000040L
#define MMVM_L2_CNTL4__VMC_TAP_PTE_REQUEST_PHYSICAL_MASK                                                      0x00000080L
#define MMVM_L2_CNTL4__MM_NONRT_IFIFO_ACTIVE_TRANSACTION_LIMIT_MASK                                           0x0003FF00L
#define MMVM_L2_CNTL4__MM_SOFTRT_IFIFO_ACTIVE_TRANSACTION_LIMIT_MASK                                          0x0FFC0000L
#define MMVM_L2_CNTL4__BPM_CGCGLS_OVERRIDE_MASK                                                               0x10000000L
#define MMVM_L2_CNTL4__GC_CH_FGCG_OFF_MASK                                                                    0x20000000L
#define MMVM_L2_CNTL4__VFIFO_HEAD_OF_QUEUE_MASK                                                               0x40000000L
#define MMVM_L2_CNTL4__VFIFO_VISIBLE_BANK_SILOS_MASK                                                          0x80000000L
#define MMVM_L2_BANK_SELECT_RESERVED_CID2__RESERVED_READ_CLIENT_ID_MASK                                       0x000001FFL
#define MMVM_L2_BANK_SELECT_RESERVED_CID2__RESERVED_WRITE_CLIENT_ID_MASK                                      0x0007FC00L
#define MMVM_L2_BANK_SELECT_RESERVED_CID2__ENABLE_MASK                                                        0x00100000L
#define MMVM_L2_BANK_SELECT_RESERVED_CID2__RESERVED_CACHE_INVALIDATION_MODE_MASK                              0x01000000L
#define MMVM_L2_BANK_SELECT_RESERVED_CID2__RESERVED_CACHE_PRIVATE_INVALIDATION_MASK                           0x02000000L
#define MMVM_L2_BANK_SELECT_RESERVED_CID2__RESERVED_CACHE_FRAGMENT_SIZE_MASK                                  0x7C000000L
#define MMVM_L2_CNTL5__L2_CACHE_SMALLK_FRAGMENT_SIZE_MASK                                                     0x0000001FL
#define MMVM_L2_CNTL5__WALKER_PRIORITY_CLIENT_ID_MASK                                                         0x00003FE0L
#define MMVM_L2_CNTL5__WALKER_FETCH_PDE_NOALLOC_ENABLE_MASK                                                   0x00004000L
#define MMVM_L2_CNTL5__WALKER_FETCH_PDE_MTYPE_ENABLE_MASK                                                     0x00008000L
#define MMVM_L2_CNTL5__MM_CLIENT_RET_FGCG_OFF_MASK                                                            0x00010000L
#define MMVM_L2_CNTL5__UTCL2_ATC_REQ_FGCG_OFF_MASK                                                            0x00020000L
#define MMVM_CONTEXT0_CNTL__ENABLE_CONTEXT_MASK                                                               0x00000001L
#define MMVM_CONTEXT0_CNTL__PAGE_TABLE_DEPTH_MASK                                                             0x00000006L
#define MMVM_CONTEXT0_CNTL__PAGE_TABLE_BLOCK_SIZE_MASK                                                        0x00000078L
#define MMVM_CONTEXT0_CNTL__RETRY_PERMISSION_OR_INVALID_PAGE_FAULT_MASK                                       0x00000080L
#define MMVM_CONTEXT0_CNTL__RETRY_OTHER_FAULT_MASK                                                            0x00000100L
#define MMVM_CONTEXT0_CNTL__RANGE_PROTECTION_FAULT_ENABLE_INTERRUPT_MASK                                      0x00000200L
#define MMVM_CONTEXT0_CNTL__RANGE_PROTECTION_FAULT_ENABLE_DEFAULT_MASK                                        0x00000400L
#define MMVM_CONTEXT0_CNTL__DUMMY_PAGE_PROTECTION_FAULT_ENABLE_INTERRUPT_MASK                                 0x00000800L
#define MMVM_CONTEXT0_CNTL__DUMMY_PAGE_PROTECTION_FAULT_ENABLE_DEFAULT_MASK                                   0x00001000L
#define MMVM_CONTEXT0_CNTL__PDE0_PROTECTION_FAULT_ENABLE_INTERRUPT_MASK                                       0x00002000L
#define MMVM_CONTEXT0_CNTL__PDE0_PROTECTION_FAULT_ENABLE_DEFAULT_MASK                                         0x00004000L
#define MMVM_CONTEXT0_CNTL__VALID_PROTECTION_FAULT_ENABLE_INTERRUPT_MASK                                      0x00008000L
#define MMVM_CONTEXT0_CNTL__VALID_PROTECTION_FAULT_ENABLE_DEFAULT_MASK                                        0x00010000L
#define MMVM_CONTEXT0_CNTL__READ_PROTECTION_FAULT_ENABLE_INTERRUPT_MASK                                       0x00020000L
#define MMVM_CONTEXT0_CNTL__READ_PROTECTION_FAULT_ENABLE_DEFAULT_MASK                                         0x00040000L
#define MMVM_CONTEXT0_CNTL__WRITE_PROTECTION_FAULT_ENABLE_INTERRUPT_MASK                                      0x00080000L
#define MMVM_CONTEXT0_CNTL__WRITE_PROTECTION_FAULT_ENABLE_DEFAULT_MASK                                        0x00100000L
#define MMVM_CONTEXT0_CNTL__EXECUTE_PROTECTION_FAULT_ENABLE_INTERRUPT_MASK                                    0x00200000L
#define MMVM_CONTEXT0_CNTL__EXECUTE_PROTECTION_FAULT_ENABLE_DEFAULT_MASK                                      0x00400000L
#define MMVM_CONTEXT0_CNTL__SECURE_PROTECTION_FAULT_ENABLE_INTERRUPT_MASK                                     0x00800000L
#define MMVM_CONTEXT0_CNTL__SECURE_PROTECTION_FAULT_ENABLE_DEFAULT_MASK                                       0x01000000L
#define MMVM_INVALIDATE_ENG17_SEM__SEMAPHORE_MASK                                                             0x00000001L
#define MMVM_INVALIDATE_ENG17_REQ__PER_VMID_INVALIDATE_REQ_MASK                                               0x0000FFFFL
#define MMVM_INVALIDATE_ENG17_REQ__FLUSH_TYPE_MASK                                                            0x00070000L
#define MMVM_INVALIDATE_ENG17_REQ__INVALIDATE_L2_PTES_MASK                                                    0x00080000L
#define MMVM_INVALIDATE_ENG17_REQ__INVALIDATE_L2_PDE0_MASK                                                    0x00100000L
#define MMVM_INVALIDATE_ENG17_REQ__INVALIDATE_L2_PDE1_MASK                                                    0x00200000L
#define MMVM_INVALIDATE_ENG17_REQ__INVALIDATE_L2_PDE2_MASK                                                    0x00400000L
#define MMVM_INVALIDATE_ENG17_REQ__INVALIDATE_L1_PTES_MASK                                                    0x00800000L
#define MMVM_INVALIDATE_ENG17_REQ__CLEAR_PROTECTION_FAULT_STATUS_ADDR_MASK                                    0x01000000L
#define MMVM_INVALIDATE_ENG17_REQ__LOG_REQUEST_MASK                                                           0x02000000L
#define MMVM_INVALIDATE_ENG17_REQ__INVALIDATE_4K_PAGES_ONLY_MASK                                              0x04000000L
#define MMVM_INVALIDATE_ENG17_ACK__PER_VMID_INVALIDATE_ACK_MASK                                               0x0000FFFFL
#define MMVM_INVALIDATE_ENG17_ACK__SEMAPHORE_MASK                                                             0x00010000L
#define MMVM_INVALIDATE_ENG0_ADDR_RANGE_LO32__S_BIT_MASK                                                      0x00000001L
#define MMVM_INVALIDATE_ENG0_ADDR_RANGE_LO32__LOGI_PAGE_ADDR_RANGE_LO31_MASK                                  0xFFFFFFFEL
#define MMVM_INVALIDATE_ENG0_ADDR_RANGE_HI32__LOGI_PAGE_ADDR_RANGE_HI5_MASK                                   0x0000001FL
#define MMVM_INVALIDATE_ENG1_ADDR_RANGE_LO32__S_BIT_MASK                                                      0x00000001L
#define MMVM_INVALIDATE_ENG1_ADDR_RANGE_LO32__LOGI_PAGE_ADDR_RANGE_LO31_MASK                                  0xFFFFFFFEL
#define MMVM_INVALIDATE_ENG1_ADDR_RANGE_HI32__LOGI_PAGE_ADDR_RANGE_HI5_MASK                                   0x0000001FL
#define MMVM_INVALIDATE_ENG2_ADDR_RANGE_LO32__S_BIT_MASK                                                      0x00000001L
#define MMVM_INVALIDATE_ENG2_ADDR_RANGE_LO32__LOGI_PAGE_ADDR_RANGE_LO31_MASK                                  0xFFFFFFFEL
#define MMVM_INVALIDATE_ENG2_ADDR_RANGE_HI32__LOGI_PAGE_ADDR_RANGE_HI5_MASK                                   0x0000001FL
#define MMVM_INVALIDATE_ENG3_ADDR_RANGE_LO32__S_BIT_MASK                                                      0x00000001L
#define MMVM_INVALIDATE_ENG3_ADDR_RANGE_LO32__LOGI_PAGE_ADDR_RANGE_LO31_MASK                                  0xFFFFFFFEL
#define MMVM_INVALIDATE_ENG3_ADDR_RANGE_HI32__LOGI_PAGE_ADDR_RANGE_HI5_MASK                                   0x0000001FL
#define MMVM_INVALIDATE_ENG4_ADDR_RANGE_LO32__S_BIT_MASK                                                      0x00000001L
#define MMVM_INVALIDATE_ENG4_ADDR_RANGE_LO32__LOGI_PAGE_ADDR_RANGE_LO31_MASK                                  0xFFFFFFFEL
#define MMVM_INVALIDATE_ENG4_ADDR_RANGE_HI32__LOGI_PAGE_ADDR_RANGE_HI5_MASK                                   0x0000001FL
#define MMVM_INVALIDATE_ENG5_ADDR_RANGE_LO32__S_BIT_MASK                                                      0x00000001L
#define MMVM_INVALIDATE_ENG5_ADDR_RANGE_LO32__LOGI_PAGE_ADDR_RANGE_LO31_MASK                                  0xFFFFFFFEL
#define MMVM_INVALIDATE_ENG5_ADDR_RANGE_HI32__LOGI_PAGE_ADDR_RANGE_HI5_MASK                                   0x0000001FL
#define MMVM_INVALIDATE_ENG6_ADDR_RANGE_LO32__S_BIT_MASK                                                      0x00000001L
#define MMVM_INVALIDATE_ENG6_ADDR_RANGE_LO32__LOGI_PAGE_ADDR_RANGE_LO31_MASK                                  0xFFFFFFFEL
#define MMVM_INVALIDATE_ENG6_ADDR_RANGE_HI32__LOGI_PAGE_ADDR_RANGE_HI5_MASK                                   0x0000001FL
#define MMVM_INVALIDATE_ENG7_ADDR_RANGE_LO32__S_BIT_MASK                                                      0x00000001L
#define MMVM_INVALIDATE_ENG7_ADDR_RANGE_LO32__LOGI_PAGE_ADDR_RANGE_LO31_MASK                                  0xFFFFFFFEL
#define MMVM_INVALIDATE_ENG7_ADDR_RANGE_HI32__LOGI_PAGE_ADDR_RANGE_HI5_MASK                                   0x0000001FL
#define MMVM_INVALIDATE_ENG8_ADDR_RANGE_LO32__S_BIT_MASK                                                      0x00000001L
#define MMVM_INVALIDATE_ENG8_ADDR_RANGE_LO32__LOGI_PAGE_ADDR_RANGE_LO31_MASK                                  0xFFFFFFFEL
#define MMVM_INVALIDATE_ENG8_ADDR_RANGE_HI32__LOGI_PAGE_ADDR_RANGE_HI5_MASK                                   0x0000001FL
#define MMVM_INVALIDATE_ENG9_ADDR_RANGE_LO32__S_BIT_MASK                                                      0x00000001L
#define MMVM_INVALIDATE_ENG9_ADDR_RANGE_LO32__LOGI_PAGE_ADDR_RANGE_LO31_MASK                                  0xFFFFFFFEL
#define MMVM_INVALIDATE_ENG9_ADDR_RANGE_HI32__LOGI_PAGE_ADDR_RANGE_HI5_MASK                                   0x0000001FL
#define MMVM_INVALIDATE_ENG10_ADDR_RANGE_LO32__S_BIT_MASK                                                     0x00000001L
#define MMVM_INVALIDATE_ENG10_ADDR_RANGE_LO32__LOGI_PAGE_ADDR_RANGE_LO31_MASK                                 0xFFFFFFFEL
#define MMVM_INVALIDATE_ENG10_ADDR_RANGE_HI32__LOGI_PAGE_ADDR_RANGE_HI5_MASK                                  0x0000001FL
#define MMVM_INVALIDATE_ENG11_ADDR_RANGE_LO32__S_BIT_MASK                                                     0x00000001L
#define MMVM_INVALIDATE_ENG11_ADDR_RANGE_LO32__LOGI_PAGE_ADDR_RANGE_LO31_MASK                                 0xFFFFFFFEL
#define MMVM_INVALIDATE_ENG11_ADDR_RANGE_HI32__LOGI_PAGE_ADDR_RANGE_HI5_MASK                                  0x0000001FL
#define MMVM_INVALIDATE_ENG12_ADDR_RANGE_LO32__S_BIT_MASK                                                     0x00000001L
#define MMVM_INVALIDATE_ENG12_ADDR_RANGE_LO32__LOGI_PAGE_ADDR_RANGE_LO31_MASK                                 0xFFFFFFFEL
#define MMVM_INVALIDATE_ENG12_ADDR_RANGE_HI32__LOGI_PAGE_ADDR_RANGE_HI5_MASK                                  0x0000001FL
#define MMVM_INVALIDATE_ENG13_ADDR_RANGE_LO32__S_BIT_MASK                                                     0x00000001L
#define MMVM_INVALIDATE_ENG13_ADDR_RANGE_LO32__LOGI_PAGE_ADDR_RANGE_LO31_MASK                                 0xFFFFFFFEL
#define MMVM_INVALIDATE_ENG13_ADDR_RANGE_HI32__LOGI_PAGE_ADDR_RANGE_HI5_MASK                                  0x0000001FL
#define MMVM_INVALIDATE_ENG14_ADDR_RANGE_LO32__S_BIT_MASK                                                     0x00000001L
#define MMVM_INVALIDATE_ENG14_ADDR_RANGE_LO32__LOGI_PAGE_ADDR_RANGE_LO31_MASK                                 0xFFFFFFFEL
#define MMVM_INVALIDATE_ENG14_ADDR_RANGE_HI32__LOGI_PAGE_ADDR_RANGE_HI5_MASK                                  0x0000001FL
#define MMVM_INVALIDATE_ENG15_ADDR_RANGE_LO32__S_BIT_MASK                                                     0x00000001L
#define MMVM_INVALIDATE_ENG15_ADDR_RANGE_LO32__LOGI_PAGE_ADDR_RANGE_LO31_MASK                                 0xFFFFFFFEL
#define MMVM_INVALIDATE_ENG15_ADDR_RANGE_HI32__LOGI_PAGE_ADDR_RANGE_HI5_MASK                                  0x0000001FL
#define MMVM_INVALIDATE_ENG16_ADDR_RANGE_LO32__S_BIT_MASK                                                     0x00000001L
#define MMVM_INVALIDATE_ENG16_ADDR_RANGE_LO32__LOGI_PAGE_ADDR_RANGE_LO31_MASK                                 0xFFFFFFFEL
#define MMVM_INVALIDATE_ENG16_ADDR_RANGE_HI32__LOGI_PAGE_ADDR_RANGE_HI5_MASK                                  0x0000001FL
#define MMVM_INVALIDATE_ENG17_ADDR_RANGE_LO32__S_BIT_MASK                                                     0x00000001L
#define MMVM_INVALIDATE_ENG17_ADDR_RANGE_LO32__LOGI_PAGE_ADDR_RANGE_LO31_MASK                                 0xFFFFFFFEL
#define MMVM_INVALIDATE_ENG17_ADDR_RANGE_HI32__LOGI_PAGE_ADDR_RANGE_HI5_MASK                                  0x0000001FL
#define MMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32__PAGE_DIRECTORY_ENTRY_LO32_MASK                               0xFFFFFFFFL
#define MMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32__PAGE_DIRECTORY_ENTRY_HI32_MASK                               0xFFFFFFFFL
#define MMVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32__LOGICAL_PAGE_NUMBER_LO32_MASK                               0xFFFFFFFFL
#define MMVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32__LOGICAL_PAGE_NUMBER_HI4_MASK                                0x0000000FL
#define MMVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32__LOGICAL_PAGE_NUMBER_LO32_MASK                                 0xFFFFFFFFL
#define MMVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32__LOGICAL_PAGE_NUMBER_HI4_MASK                                  0x0000000FL
#define MMMC_VM_FB_OFFSET__FB_OFFSET_MASK                                                                     0x00FFFFFFL
#define MMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB__PHYSICAL_PAGE_NUMBER_LSB_MASK                               0xFFFFFFFFL
#define MMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB__PHYSICAL_PAGE_NUMBER_MSB_MASK                               0x0000000FL
#define MMMC_VM_FB_LOCATION_BASE__FB_BASE_MASK                                                                0x00FFFFFFL
#define MMMC_VM_FB_LOCATION_TOP__FB_TOP_MASK                                                                  0x00FFFFFFL
#define MMMC_VM_AGP_TOP__AGP_TOP_MASK                                                                         0x00FFFFFFL
#define MMMC_VM_AGP_BOT__AGP_BOT_MASK                                                                         0x00FFFFFFL
#define MMMC_VM_AGP_BASE__AGP_BASE_MASK                                                                       0x00FFFFFFL
#define MMMC_VM_SYSTEM_APERTURE_LOW_ADDR__LOGICAL_ADDR_MASK                                                   0x3FFFFFFFL
#define MMMC_VM_SYSTEM_APERTURE_HIGH_ADDR__LOGICAL_ADDR_MASK                                                  0x3FFFFFFFL
#define MMMC_VM_MX_L1_TLB_CNTL__ENABLE_L1_TLB_MASK                                                            0x00000001L
#define MMMC_VM_MX_L1_TLB_CNTL__SYSTEM_ACCESS_MODE_MASK                                                       0x00000018L
#define MMMC_VM_MX_L1_TLB_CNTL__SYSTEM_APERTURE_UNMAPPED_ACCESS_MASK                                          0x00000020L
#define MMMC_VM_MX_L1_TLB_CNTL__ENABLE_ADVANCED_DRIVER_MODEL_MASK                                             0x00000040L
#define MMMC_VM_MX_L1_TLB_CNTL__ECO_BITS_MASK                                                                 0x00000780L
#define MMMC_VM_MX_L1_TLB_CNTL__MTYPE_MASK                                                                    0x00003800L
