/*
 * SPDX-FileCopyrightText: Copyright (c) 2005-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/* Excerpt of https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/615.71.09/src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080perf.h. */

#define NV_SUBPROC_NAME_MAX_LENGTH                    100

#define NV2080_CTRL_PERF_BOOST_FLAGS_CMD                1:0

#define NV2080_CTRL_PERF_BOOST_FLAGS_CMD_BOOST_TO_MAX (0x00000002)

#define NV2080_CTRL_PERF_BOOST_FLAGS_CUDA               4:4

#define NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_YES         (0x00000001)

#define NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_PRIORITY          6:6

#define NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_PRIORITY_HIGH     (0x00000001)

#define NV2080_CTRL_PERF_BOOST_DURATION_INFINITE      0xffffffff // If set this way, the boost will last until cleared.

#define NV2080_CTRL_CMD_PERF_BOOST                    (0x2080200a) /* finn: Evaluated from "(FINN_NV20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | NV2080_CTRL_PERF_BOOST_PARAMS_MESSAGE_ID" */

typedef struct NV2080_CTRL_PERF_BOOST_PARAMS {
    NvU32 flags;
    NvU32 duration;
} NV2080_CTRL_PERF_BOOST_PARAMS;

typedef struct NV2080_CTRL_PERF_GPUMON_ENGINE_UTIL_SAMPLE {
    /*!
     * Percentage during the sample that the engine remains busy. This
     * is in units of pct*100.
     */
    NvU32 util;
    /*!
     * Scaling factor to convert utilization from full GPU to per vGPU.
     */
    NvU32 vgpuScale;
    /*!
     * Process ID of the process that was active on the engine when the
     * sample was taken. If no process is active then NV2080_GPUMON_PID_INVALID
     * will be returned.
     */
    NvU32 procId;
    /*!
     * Process ID of the process in the vGPU VM that was active on the engine when
     * the sample was taken. If no process is active then NV2080_GPUMON_PID_INVALID
     * will be returned.
     */
    NvU32 subProcessID;
     /*!
     * Process name of the process in the vGPU VM that was active on the engine when
     * the sample was taken. If no process is active then NULL will be returned.
     */
    char  subProcessName[NV_SUBPROC_NAME_MAX_LENGTH];
    /*!
     * PID struct pointer of the process that was active on the engine when the
     * the sample was taken. If no process is active then NULL pointer
     * will be returned
     */
    NV_DECLARE_ALIGNED(NvU64 pOsPidInfo, 8);
} NV2080_CTRL_PERF_GPUMON_ENGINE_UTIL_SAMPLE;

typedef struct NV2080_CTRL_PERF_GPUMON_PERFMON_UTIL_SAMPLE {
    /*!
     * Base GPU monitoring sample.
     */
    NV_DECLARE_ALIGNED(NV2080_CTRL_GPUMON_SAMPLE base, 8);
    /*!
     * FB bandwidth utilization sample.
     */
    NV_DECLARE_ALIGNED(NV2080_CTRL_PERF_GPUMON_ENGINE_UTIL_SAMPLE fb, 8);
    /*!
     * GR utilization sample.
     */
    NV_DECLARE_ALIGNED(NV2080_CTRL_PERF_GPUMON_ENGINE_UTIL_SAMPLE gr, 8);
    /*!
     * NV ENCODER utilization sample.
     */
    NV_DECLARE_ALIGNED(NV2080_CTRL_PERF_GPUMON_ENGINE_UTIL_SAMPLE nvenc, 8);
    /*!
     * NV DECODER utilization sample.
     */
    NV_DECLARE_ALIGNED(NV2080_CTRL_PERF_GPUMON_ENGINE_UTIL_SAMPLE nvdec, 8);
    /*!
     * NV JPEG utilization sample.
     */
    NV_DECLARE_ALIGNED(NV2080_CTRL_PERF_GPUMON_ENGINE_UTIL_SAMPLE nvjpg, 8);
    /*!
     * NV OFA utilization sample.
     */
    NV_DECLARE_ALIGNED(NV2080_CTRL_PERF_GPUMON_ENGINE_UTIL_SAMPLE nvofa, 8);
} NV2080_CTRL_PERF_GPUMON_PERFMON_UTIL_SAMPLE;

#define NV2080_CTRL_PERF_GPUMON_SAMPLE_COUNT_PERFMON_UTIL       72

#define NV2080_CTRL_CMD_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_V2 (0x20802096) /* finn: Evaluated from "(FINN_NV20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | NV2080_CTRL_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_V2_PARAMS_MESSAGE_ID" */

typedef struct NV2080_CTRL_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_V2_PARAMS {
    /*!
    * Type of the sample, see NV2080_CTRL_GPUMON_SAMPLE_TYPE_* for reference.
    */
    NvU8  type;
    /*!
    * Size of the buffer, this should be
    * bufSize ==  NV2080_CTRL_*_GPUMON_SAMPLE_COUNT_*
    *    sizeof(derived type of NV2080_CTRL_GPUMON_SAMPLE).
    */
    NvU32 bufSize;
    /*!
    * Number of samples in ring buffer.
    */
    NvU32 count;
    /*!
    * tracks the offset of the tail in the circular queue array pSamples.
    */
    NvU32 tracker;
    /*!
    * A circular queue with size == bufSize.
    *
    * @note This circular queue wraps around after 10 seconds of sampling,
    * and it is clients' responsibility to query within this time frame in
    * order to avoid losing samples.
    * @note With one exception, this queue contains last 10 seconds of samples
    * with tracker poiniting to oldest entry and entry before tracker as the
    * newest entry. Exception is when queue is not full (i.e. tracker is
    * pointing to a zeroed out entry), in that case valid entries are between 0
    * and tracker.
    * @note Clients can store tracker from previous query in order to provide
    * samples since last read.
    */
    NV_DECLARE_ALIGNED(NV2080_CTRL_PERF_GPUMON_PERFMON_UTIL_SAMPLE samples[NV2080_CTRL_PERF_GPUMON_SAMPLE_COUNT_PERFMON_UTIL], 8);
} NV2080_CTRL_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_V2_PARAMS;
