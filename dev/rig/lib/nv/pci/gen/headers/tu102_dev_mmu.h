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

#define NV_MMU_PTE_KIND_GENERIC_MEMORY                                                  0x06 /* R---V */

#define NV_MMU_VER2_PDE_IS_PTE                                           0:0 /* RWXVF */

#define NV_MMU_VER2_PDE_IS_PDE                                           0:0 /* RWXVF */

#define NV_MMU_VER2_PDE_VALID                                            0:0 /* RWXVF */

#define NV_MMU_VER2_PDE_APERTURE                                         2:1 /* RWXVF */

#define NV_MMU_VER2_PDE_APERTURE_INVALID                          0x00000000 /* RW--V */

#define NV_MMU_VER2_PDE_APERTURE_VIDEO_MEMORY                     0x00000001 /* RW--V */

#define NV_MMU_VER2_PDE_APERTURE_SYSTEM_COHERENT_MEMORY           0x00000002 /* RW--V */

#define NV_MMU_VER2_PDE_APERTURE_SYSTEM_NON_COHERENT_MEMORY       0x00000003 /* RW--V */

#define NV_MMU_VER2_PDE_VOL                                              3:3 /* RWXVF */

#define NV_MMU_VER2_PDE_NO_ATS                                           5:5 /* RWXVF */

#define NV_MMU_VER2_PDE_ADDRESS_SYS                                     53:8 /* RWXVF */

#define NV_MMU_VER2_PDE_ADDRESS_VID             (35-3):8 /* RWXVF */

#define NV_MMU_VER2_PDE_ADDRESS_VID_PEER       35:(36-3) /* RWXVF */

#define NV_MMU_VER2_DUAL_PDE_IS_PTE                                           0:0 /* RWXVF */

#define NV_MMU_VER2_DUAL_PDE_IS_PDE                                           0:0 /* RWXVF */

#define NV_MMU_VER2_DUAL_PDE_VALID                                            0:0 /* RWXVF */

#define NV_MMU_VER2_DUAL_PDE_APERTURE_BIG                                     2:1 /* RWXVF */

#define NV_MMU_VER2_DUAL_PDE_VOL_BIG                                          3:3 /* RWXVF */

#define NV_MMU_VER2_DUAL_PDE_NO_ATS                                      5:5 /* RWXVF */

#define NV_MMU_VER2_DUAL_PDE_ADDRESS_BIG_SYS                                 53:(8-4) /* RWXVF */

#define NV_MMU_VER2_DUAL_PDE_ADDRESS_BIG_VID         (35-3):(8-4) /* RWXVF */

#define NV_MMU_VER2_DUAL_PDE_ADDRESS_BIG_VID_PEER   35:(36-3) /* RWXVF */

#define NV_MMU_VER2_DUAL_PDE_APERTURE_SMALL                                 66:65 /* RWXVF */

#define NV_MMU_VER2_DUAL_PDE_APERTURE_SMALL_VIDEO_MEMORY               0x00000001 /* RW--V */

#define NV_MMU_VER2_DUAL_PDE_VOL_SMALL                                      67:67 /* RWXVF */

#define NV_MMU_VER2_DUAL_PDE_ADDRESS_SMALL_SYS                             117:72 /* RWXVF */

#define NV_MMU_VER2_DUAL_PDE_ADDRESS_SMALL_VID      (99-3):72 /* RWXVF */

#define NV_MMU_VER2_DUAL_PDE_ADDRESS_SMALL_VID_PEER 99:(100-3) /* RWXVF */

#define NV_MMU_VER2_PTE_VALID                                            0:0 /* RWXVF */

#define NV_MMU_VER2_PTE_APERTURE                                         2:1 /* RWXVF */

#define NV_MMU_VER2_PTE_APERTURE_VIDEO_MEMORY                     0x00000000 /* RW--V */

#define NV_MMU_VER2_PTE_APERTURE_PEER_MEMORY                      0x00000001 /* RW--V */

#define NV_MMU_VER2_PTE_APERTURE_SYSTEM_COHERENT_MEMORY           0x00000002 /* RW--V */

#define NV_MMU_VER2_PTE_APERTURE_SYSTEM_NON_COHERENT_MEMORY       0x00000003 /* RW--V */

#define NV_MMU_VER2_PTE_VOL                                              3:3 /* RWXVF */

#define NV_MMU_VER2_PTE_ENCRYPTED                                        4:4 /* RWXVF */

#define NV_MMU_VER2_PTE_PRIVILEGE                                        5:5 /* RWXVF */

#define NV_MMU_VER2_PTE_READ_ONLY                                        6:6 /* RWXVF */

#define NV_MMU_VER2_PTE_ATOMIC_DISABLE                                   7:7 /* RWXVF */

#define NV_MMU_VER2_PTE_ADDRESS_SYS                                     53:8 /* RWXVF */

#define NV_MMU_VER2_PTE_ADDRESS_VID             (35-3):8 /* RWXVF */

#define NV_MMU_VER2_PTE_ADDRESS_VID_PEER       35:(36-3) /* RWXVF */

#define NV_MMU_VER2_PTE_COMPTAGLINE   (20+35):36 /* RWXVF */

#define NV_MMU_VER2_PTE_KIND                                           63:56 /* RWXVF */
