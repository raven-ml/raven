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

#define PPSMC_MSG_GetSmuVersion                  0x2
#define PPSMC_MSG_EnableAllSmuFeatures           0x6
#define PPSMC_MSG_GetRunningSmuFeaturesLow       0xC
#define PPSMC_MSG_GetRunningSmuFeaturesHigh      0xD
#define PPSMC_MSG_SetDriverDramAddrHigh          0xE
#define PPSMC_MSG_SetDriverDramAddrLow           0xF
#define PPSMC_MSG_SetSoftMinByFreq               0x19
#define PPSMC_MSG_SetSoftMaxByFreq               0x1A
#define PPSMC_MSG_GetDpmFreqByIndex              0x1F
#define PPSMC_MSG_Mode1Reset                     0x2F
