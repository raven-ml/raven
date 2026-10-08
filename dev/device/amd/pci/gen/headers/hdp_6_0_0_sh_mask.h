/*
 * Copyright 2021 Advanced Micro Devices, Inc.
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

#define HDP_MEM_POWER_CTRL__ATOMIC_MEM_POWER_CTRL_EN_MASK                                                     0x00000001L
#define HDP_MEM_POWER_CTRL__ATOMIC_MEM_POWER_LS_EN_MASK                                                       0x00000002L
#define HDP_MEM_POWER_CTRL__ATOMIC_MEM_POWER_DS_EN_MASK                                                       0x00000004L
#define HDP_MEM_POWER_CTRL__ATOMIC_MEM_POWER_SD_EN_MASK                                                       0x00000008L
#define HDP_MEM_POWER_CTRL__ATOMIC_MEM_IDLE_HYSTERESIS_MASK                                                   0x00000070L
#define HDP_MEM_POWER_CTRL__ATOMIC_MEM_POWER_UP_RECOVER_DELAY_MASK                                            0x00003F00L
#define HDP_MEM_POWER_CTRL__ATOMIC_MEM_POWER_DOWN_ENTER_DELAY_MASK                                            0x0000C000L
#define HDP_MEM_POWER_CTRL__RC_MEM_POWER_CTRL_EN_MASK                                                         0x00010000L
#define HDP_MEM_POWER_CTRL__RC_MEM_POWER_LS_EN_MASK                                                           0x00020000L
#define HDP_MEM_POWER_CTRL__RC_MEM_POWER_DS_EN_MASK                                                           0x00040000L
#define HDP_MEM_POWER_CTRL__RC_MEM_POWER_SD_EN_MASK                                                           0x00080000L
#define HDP_MEM_POWER_CTRL__RC_MEM_IDLE_HYSTERESIS_MASK                                                       0x00700000L
#define HDP_MEM_POWER_CTRL__RC_MEM_POWER_UP_RECOVER_DELAY_MASK                                                0x3F000000L
#define HDP_MEM_POWER_CTRL__RC_MEM_POWER_DOWN_ENTER_DELAY_MASK                                                0xC0000000L
