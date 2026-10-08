/*******************************************************************************
    Copyright (c) 2003-2016 NVIDIA Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/

#define NV_MMU_VER3_PDE_IS_PTE                                           0:0 /* RWXVF */

#define NV_MMU_VER3_PDE_VALID                                            0:0 /* RWXVF */

#define NV_MMU_VER3_PDE_APERTURE                                         2:1 /* RWXVF */

#define NV_MMU_VER3_PDE_APERTURE_INVALID                          0x00000000 /* RW--V */

#define NV_MMU_VER3_PDE_APERTURE_VIDEO_MEMORY                     0x00000001 /* RW--V */

#define NV_MMU_VER3_PDE_APERTURE_SYSTEM_COHERENT_MEMORY           0x00000002 /* RW--V */

#define NV_MMU_VER3_PDE_APERTURE_SYSTEM_NON_COHERENT_MEMORY       0x00000003 /* RW--V */

#define NV_MMU_VER3_PDE_PCF                                                                        5:3 /* RWXVF */

#define NV_MMU_VER3_PDE_PCF_VALID_CACHED_ATS_NOT_ALLOWED                                    0x00000002 /* RW--V */

#define NV_MMU_VER3_PDE_ADDRESS                                             51:12 /* RWXVF */

#define NV_MMU_VER3_DUAL_PDE_IS_PTE                                           0:0 /* RWXVF */

#define NV_MMU_VER3_DUAL_PDE_VALID                                            0:0 /* RWXVF */

#define NV_MMU_VER3_DUAL_PDE_APERTURE_BIG                                     2:1 /* RWXVF */

#define NV_MMU_VER3_DUAL_PDE_PCF_BIG                                                                        5:3 /* RWXVF */

#define NV_MMU_VER3_DUAL_PDE_ADDRESS_BIG                                     51:8 /* RWXVF */

#define NV_MMU_VER3_DUAL_PDE_APERTURE_SMALL                                 66:65 /* RWXVF */

#define NV_MMU_VER3_DUAL_PDE_APERTURE_SMALL_VIDEO_MEMORY               0x00000001 /* RW--V */

#define NV_MMU_VER3_DUAL_PDE_PCF_SMALL                                                                      69:67 /* RWXVF */

#define NV_MMU_VER3_DUAL_PDE_PCF_SMALL_VALID_CACHED_ATS_NOT_ALLOWED                                    0x00000002 /* RW--V */

#define NV_MMU_VER3_DUAL_PDE_ADDRESS_SMALL                                 115:76 /* RWXVF */

#define NV_MMU_VER3_PTE_VALID                                            0:0 /* RWXVF */

#define NV_MMU_VER3_PTE_APERTURE                                         2:1 /* RWXVF */

#define NV_MMU_VER3_PTE_APERTURE_VIDEO_MEMORY                     0x00000000 /* RW--V */

#define NV_MMU_VER3_PTE_APERTURE_PEER_MEMORY                      0x00000001 /* RW--V */

#define NV_MMU_VER3_PTE_APERTURE_SYSTEM_COHERENT_MEMORY           0x00000002 /* RW--V */

#define NV_MMU_VER3_PTE_APERTURE_SYSTEM_NON_COHERENT_MEMORY       0x00000003 /* RW--V */

#define NV_MMU_VER3_PTE_PCF                                                                        7:3 /* RWXVF */

#define NV_MMU_VER3_PTE_PCF_REGULAR_RW_ATOMIC_CACHED_ACE                                    0x00000000 /* RW--V */

#define NV_MMU_VER3_PTE_PCF_REGULAR_RW_ATOMIC_UNCACHED_ACE                                  0x00000001 /* RW--V */

#define NV_MMU_VER3_PTE_KIND                                           11:8 /* RWXVF */

#define NV_MMU_VER3_PTE_ADDRESS                                         51:12 /* RWXVF */

#define NV_MMU_VER3_PTE_ADDRESS_SYS                                     51:12 /* RWXVF */

#define NV_MMU_VER3_PTE_ADDRESS_PEER                                    51:12 /* RWXVF */

#define NV_MMU_VER3_PTE_ADDRESS_VID                                     39:12 /* RWXVF */

#define NV_MMU_VER3_PTE_PEER_ID                63:(64-3) /* RWXVF */
