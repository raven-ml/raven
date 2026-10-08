/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *
 **********************************************************************************************************************/

/* Excerpt of https://raw.githubusercontent.com/GPUOpen-Drivers/pal/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx9/chip/gfx9_plus_merged_f32_mec_pm4_packets.h. */

typedef union PM4_MEC_TYPE_3_HEADER
{
    struct
    {
        uint32_t reserved1 :  8;
        uint32_t opcode    :  8;
        uint32_t count     : 14;
        uint32_t type      :  2;
    };
    uint32_t u32All;
} PM4_MEC_TYPE_3_HEADER;
typedef struct PM4_MEC_WAIT_REG_MEM64
{
    union
    {
        PM4_MEC_TYPE_3_HEADER header;
        uint32_t u32All;
    } ordinal1;

    union
    {
        union
        {
            struct
            {
                MEC_WAIT_REG_MEM64_function_enum     function      :  3;
                uint32_t                             reserved1     :  1;
                MEC_WAIT_REG_MEM64_mem_space_enum    mem_space     :  2;
                MEC_WAIT_REG_MEM64_operation_enum    operation     :  2;
                uint32_t                             reserved2     : 14;
                uint32_t                             mes_intr_pipe :  2;
                uint32_t                             mes_action    :  1;
                MEC_WAIT_REG_MEM64_cache_policy_enum cache_policy  :  2;
                uint32_t                             reserved3     :  5;
            };
        } bitfields;
        uint32_t u32All;
    } ordinal2;

    union
    {
        union
        {
            struct
            {
                uint32_t reserved1        :  3;
                uint32_t mem_poll_addr_lo : 29;
            };
        } bitfieldsA;
        union
        {
            struct
            {
                uint32_t reg_poll_addr : 18;
                uint32_t reserved2     : 14;
            };
        } bitfieldsB;
        union
        {
            struct
            {
                uint32_t reg_write_addr1 : 18;
                uint32_t reserved3       : 14;
            };
        } bitfieldsC;
        uint32_t u32All;
    } ordinal3;

    union
    {
        uint32_t mem_poll_addr_hi;
        union
        {
            struct
            {
                uint32_t reg_write_addr2 : 18;
                uint32_t reserved1       : 14;
            };
        } bitfieldsB;
        uint32_t u32All;
    } ordinal4;

    union
    {
        uint32_t reference;
        uint32_t u32All;
    } ordinal5;

    union
    {
        uint32_t reference_hi;
        uint32_t u32All;
    } ordinal6;

    union
    {
        uint32_t mask;
        uint32_t u32All;
    } ordinal7;

    union
    {
        uint32_t mask_hi;
        uint32_t u32All;
    } ordinal8;

    union
    {
        union
        {
            struct
            {
                uint32_t poll_interval             : 16;
                uint32_t reserved1                 : 15;
                uint32_t optimize_ace_offload_mode :  1;
            };
        } bitfields;
        uint32_t u32All;
    } ordinal9;
} PM4_MEC_WAIT_REG_MEM64;
