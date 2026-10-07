/*******************************************************************************
    Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/

#define NVC9C0_QMDV03_00_QMD_GROUP_ID                              MW(133:128)
#define NVC9C0_QMDV03_00_SM_GLOBAL_CACHING_ENABLE                  MW(134:134)
#define NVC9C0_QMDV03_00_INVALIDATE_TEXTURE_HEADER_CACHE           MW(186:186)
#define NVC9C0_QMDV03_00_INVALIDATE_TEXTURE_SAMPLER_CACHE          MW(187:187)
#define NVC9C0_QMDV03_00_INVALIDATE_TEXTURE_DATA_CACHE             MW(188:188)
#define NVC9C0_QMDV03_00_INVALIDATE_SHADER_DATA_CACHE              MW(189:189)
#define NVC9C0_QMDV03_00_PROGRAM_PREFETCH_ADDR_LOWER_SHIFTED       MW(287:256)
#define NVC9C0_QMDV03_00_CWD_MEMBAR_TYPE                           MW(369:368)
#define NVC9C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR              0x00000001
#define NVC9C0_QMDV03_00_API_VISIBLE_CALL_LIMIT                    MW(378:378)
#define NVC9C0_QMDV03_00_API_VISIBLE_CALL_LIMIT_NO_CHECK           0x00000001
#define NVC9C0_QMDV03_00_SAMPLER_INDEX                             MW(382:382)
#define NVC9C0_QMDV03_00_SAMPLER_INDEX_VIA_HEADER_INDEX            0x00000001
#define NVC9C0_QMDV03_00_CTA_RASTER_WIDTH                          MW(415:384)
#define NVC9C0_QMDV03_00_CTA_RASTER_HEIGHT                         MW(431:416)
#define NVC9C0_QMDV03_00_CTA_RASTER_DEPTH                          MW(463:448)
#define NVC9C0_QMDV03_00_DEPENDENT_QMD0_POINTER                    MW(511:480)
#define NVC9C0_QMDV03_00_DEPENDENT_QMD0_ENABLE                     MW(512:512)
#define NVC9C0_QMDV03_00_DEPENDENT_QMD0_ACTION                     MW(515:513)
#define NVC9C0_QMDV03_00_DEPENDENT_QMD0_ACTION_QMD_SCHEDULE        0x00000001
#define NVC9C0_QMDV03_00_DEPENDENT_QMD0_PREFETCH                   MW(516:516)
#define NVC9C0_QMDV03_00_SHARED_MEMORY_SIZE                        MW(561:544)
#define NVC9C0_QMDV03_00_MIN_SM_CONFIG_SHARED_MEM_SIZE             MW(567:562)
#define NVC9C0_QMDV03_00_MAX_SM_CONFIG_SHARED_MEM_SIZE             MW(574:569)
#define NVC9C0_QMDV03_00_QMD_MAJOR_VERSION                         MW(583:580)
#define NVC9C0_QMDV03_00_CTA_THREAD_DIMENSION0                     MW(607:592)
#define NVC9C0_QMDV03_00_CTA_THREAD_DIMENSION1                     MW(623:608)
#define NVC9C0_QMDV03_00_CTA_THREAD_DIMENSION2                     MW(639:624)
#define NVC9C0_QMDV03_00_CONSTANT_BUFFER_VALID(i)                  MW((640+(i)*1):(640+(i)*1))
#define NVC9C0_QMDV03_00_REGISTER_COUNT_V                          MW(656:648)
#define NVC9C0_QMDV03_00_TARGET_SM_CONFIG_SHARED_MEM_SIZE          MW(662:657)
#define NVC9C0_QMDV03_00_BARRIER_COUNT                             MW(767:763)
#define NVC9C0_QMDV03_00_RELEASE0_ADDRESS_LOWER                    MW(799:768)
#define NVC9C0_QMDV03_00_RELEASE0_ADDRESS_UPPER                    MW(807:800)
#define NVC9C0_QMDV03_00_RELEASE0_MEMBAR_TYPE                      MW(819:819)
#define NVC9C0_QMDV03_00_RELEASE0_MEMBAR_TYPE_FE_NONE              0x00000000
#define NVC9C0_QMDV03_00_RELEASE0_ENABLE                           MW(823:823)
#define NVC9C0_QMDV03_00_RELEASE0_PAYLOAD64B                       MW(829:829)
#define NVC9C0_QMDV03_00_RELEASE0_STRUCTURE_SIZE                   MW(831:830)
#define NVC9C0_QMDV03_00_RELEASE0_STRUCTURE_SIZE_SEMAPHORE_FOUR_WORDS 0x00000000
#define NVC9C0_QMDV03_00_RELEASE0_STRUCTURE_SIZE_SEMAPHORE_TWO_WORDS 0x00000002
#define NVC9C0_QMDV03_00_RELEASE0_PAYLOAD_LOWER                    MW(863:832)
#define NVC9C0_QMDV03_00_RELEASE0_PAYLOAD_UPPER                    MW(895:864)
#define NVC9C0_QMDV03_00_RELEASE1_ADDRESS_LOWER                    MW(927:896)
#define NVC9C0_QMDV03_00_RELEASE1_ADDRESS_UPPER                    MW(935:928)
#define NVC9C0_QMDV03_00_RELEASE1_MEMBAR_TYPE                      MW(947:947)
#define NVC9C0_QMDV03_00_RELEASE1_ENABLE                           MW(951:951)
#define NVC9C0_QMDV03_00_RELEASE1_PAYLOAD64B                       MW(957:957)
#define NVC9C0_QMDV03_00_RELEASE1_STRUCTURE_SIZE                   MW(959:958)
#define NVC9C0_QMDV03_00_RELEASE1_PAYLOAD_LOWER                    MW(991:960)
#define NVC9C0_QMDV03_00_RELEASE1_PAYLOAD_UPPER                    MW(1023:992)
#define NVC9C0_QMDV03_00_CONSTANT_BUFFER_ADDR_LOWER(i)             MW((1055+(i)*64):(1024+(i)*64))
#define NVC9C0_QMDV03_00_CONSTANT_BUFFER_ADDR_UPPER(i)             MW((1072+(i)*64):(1056+(i)*64))
#define NVC9C0_QMDV03_00_CONSTANT_BUFFER_INVALIDATE(i)             MW((1074+(i)*64):(1074+(i)*64))
#define NVC9C0_QMDV03_00_CONSTANT_BUFFER_SIZE_SHIFTED4(i)          MW((1087+(i)*64):(1075+(i)*64))
#define NVC9C0_QMDV03_00_PROGRAM_ADDRESS_LOWER                     MW(1567:1536)
#define NVC9C0_QMDV03_00_PROGRAM_ADDRESS_UPPER                     MW(1584:1568)
#define NVC9C0_QMDV03_00_SHADER_LOCAL_MEMORY_HIGH_SIZE             MW(1623:1600)
#define NVC9C0_QMDV03_00_PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED       MW(1640:1632)
#define NVC9C0_QMDV03_00_PROGRAM_PREFETCH_SIZE                     MW(1649:1641)
#define NVC9C0_QMDV03_00_SASS_VERSION                              MW(1663:1656)
#define NVC9C0_QMDV03_00_DEBUG_ID_LOWER                            MW(2047:2016)
