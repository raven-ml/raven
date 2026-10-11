/*******************************************************************************
    Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

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

#define NVC7B5_SET_SEMAPHORE_A                                                  (0x00000240)
#define NVC7B5_SET_SEMAPHORE_B                                                  (0x00000244)
#define NVC7B5_SET_SEMAPHORE_PAYLOAD                                            (0x00000248)
#define NVC7B5_SET_SEMAPHORE_PAYLOAD_UPPER                                      (0x0000024C)
#define NVC7B5_LAUNCH_DMA                                                       (0x00000300)
#define NVC7B5_LAUNCH_DMA_DATA_TRANSFER_TYPE                                    1:0
#define NVC7B5_LAUNCH_DMA_DATA_TRANSFER_TYPE_NON_PIPELINED                      (0x00000002)
#define NVC7B5_LAUNCH_DMA_FLUSH_ENABLE                                          2:2
#define NVC7B5_LAUNCH_DMA_FLUSH_ENABLE_TRUE                                     (0x00000001)
#define NVC7B5_LAUNCH_DMA_FLUSH_TYPE                                            25:25
#define NVC7B5_LAUNCH_DMA_FLUSH_TYPE_SYS                                        (0x00000000)
#define NVC7B5_LAUNCH_DMA_SEMAPHORE_TYPE                                        4:3
#define NVC7B5_LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_ONE_WORD_SEMAPHORE             (0x00000001)
#define NVC7B5_LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_FOUR_WORD_SEMAPHORE            (0x00000002)
#define NVC7B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT                                     7:7
#define NVC7B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT_PITCH                               (0x00000001)
#define NVC7B5_LAUNCH_DMA_DST_MEMORY_LAYOUT                                     8:8
#define NVC7B5_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH                               (0x00000001)
#define NVC7B5_LAUNCH_DMA_SEMAPHORE_PAYLOAD_SIZE                                27:27
#define NVC7B5_LAUNCH_DMA_SEMAPHORE_PAYLOAD_SIZE_TWO_WORD                       (0x00000001)
#define NVC7B5_OFFSET_IN_UPPER                                                  (0x00000400)
#define NVC7B5_OFFSET_IN_LOWER                                                  (0x00000404)
#define NVC7B5_OFFSET_OUT_UPPER                                                 (0x00000408)
#define NVC7B5_OFFSET_OUT_LOWER                                                 (0x0000040C)
#define NVC7B5_LINE_LENGTH_IN                                                   (0x00000418)
