# /// script
# requires-python = ">=3.10"
# ///
"""Generates defs.ml, the tables of rig_amd_abi: the GC registers of each GC
version and the bases of their segments, the constants of PM4, SDMA and AQL
packets and of thread traces, the layouts of AQL's dispatch packet, of the
scratch buffer descriptor and of the kernel descriptor, and LLVM's AMDGPU
processors.

Run from the worktree root:

  uv run dev/rig/lib/amd/abi/gen/gen.py
  uv run dev/rig/lib/amd/abi/gen/gen.py --check
  uv run dev/rig/lib/amd/abi/gen/gen.py --excerpt [--check]

The inputs are excerpts of AMD's, LLVM's and PAL's headers, in headers/: each
is a header's licence notice, the URL it is cut from, and the lines this script
reads, verbatim and in the header's order (a #define, an enumerator, a struct's
definition, a table of a document). SOURCES names the header each comes from,
at its version. defs.ml carries the notices of all of them.
--excerpt makes them from the upstream headers, each pinned in pins.json by URL
and SHA-256 and checked against its pin; downloads are kept in --cache, and
--pin records the digests of headers not yet pinned. Generating reads the
excerpts alone, offline. --check generates into memory and fails if a
committed file differs.

Text is read and written as latin-1, one character per byte, so every byte
of a header round-trips into its excerpt as upstream wrote it. The headers are
ASCII C; AMDGPUUsage.rst holds UTF-8 outside the lines kept, which latin-1
reads without failing.

Where two headers define a value the library takes once (PM4's in soc15d.h and
nvd.h, a thread trace value in each SOC enumeration, an SDMA field in each
version's header), the script fails unless they agree. Every table is a
literal, which the compiler lays out as static data: nothing is built when a
program starts.
"""

import argparse
import hashlib
import json
import pathlib
import re
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
HEADERS = HERE / "headers"
PINS = HERE / "pins.json"
OUT = HERE.parent / "defs.ml"

# Sources, each at the commit or tag in its URL

KERNEL = ("https://raw.githubusercontent.com/ROCm/ROCK-Kernel-Driver/33970e1351f5e511029602454979f3de7e22260f/"
          "drivers/gpu/drm/amd/")
ROCM = "https://raw.githubusercontent.com/ROCm/rocm-systems/cccc350dc620e61ae2554978b62ab3532dc10bd9/projects/"
LLVM = "https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-20.1.0/llvm/"
PAL = "https://raw.githubusercontent.com/GPUOpen-Drivers/pal/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/"

# The GC versions whose register headers exist.
GC_VERSIONS = [(9, 4, 3), (11, 0, 0), (11, 0, 3), (11, 5, 0), (12, 0, 0)]


def gc_header(ver, kind):
    return f"gc_{'_'.join(map(str, ver))}_{kind}.h"


# Each excerpt and the header it is cut from.
SOURCES = {
    **{gc_header(v, k): KERNEL + "include/asic_reg/gc/" + gc_header(v, k)
       for v in GC_VERSIONS for k in ("offset", "sh_mask")},  # the Linux kernel's GC registers
    "vega20_ip_offset.h": KERNEL + "include/vega20_ip_offset.h",  # GC segment bases, GFX9
    "sienna_cichlid_ip_offset.h": KERNEL + "include/sienna_cichlid_ip_offset.h",  # GFX10 on
    "soc15d.h": KERNEL + "amdgpu/soc15d.h",  # PM4, GFX9
    "nvd.h": KERNEL + "amdgpu/nvd.h",  # PM4, GFX10 on
    "kfd_pm4_headers_ai.h": KERNEL + "amdkfd/kfd_pm4_headers_ai.h",  # RELEASE_MEM's enumerations
    "vega10_sdma_pkt_open.h": KERNEL + "amdgpu/vega10_sdma_pkt_open.h",  # SDMA 4
    "navi10_sdma_pkt_open.h": KERNEL + "amdgpu/navi10_sdma_pkt_open.h",  # SDMA 5
    "sdma_v6_0_0_pkt_open.h": KERNEL + "amdgpu/sdma_v6_0_0_pkt_open.h",  # SDMA 6
    "vega10_enum.h": ROCM + "aqlprofile/linux/vega10_enum.h",  # SOC enumerations, GFX9
    "soc21_enum.h": ROCM + "aqlprofile/linux/soc21_enum.h",  # GFX11
    "soc24_enum.h": ROCM + "aqlprofile/linux/soc24_enum.h",  # GFX12
    "hsa.h": ROCM + "rocr-runtime/runtime/hsa-runtime/inc/hsa.h",  # AQL
    "registers.h": ROCM + "rocr-runtime/runtime/hsa-runtime/core/inc/registers.h",  # buffer descriptors
    "AMDHSAKernelDescriptor.h": LLVM + "include/llvm/Support/AMDHSAKernelDescriptor.h",
    "ELF.h": LLVM + "include/llvm/BinaryFormat/ELF.h",
    "AMDGPUUsage.rst": LLVM + "docs/AMDGPUUsage.rst",  # processors
    # performance counters
    "counter_defs.yaml": ROCM + "rocprofiler-compute/src/rocprof_compute_soc/profile_configs/counter_defs.yaml",
    "gfx9_plus_merged_f32_mec_pm4_packets.h": PAL + "gfx9/chip/gfx9_plus_merged_f32_mec_pm4_packets.h",
    "gfx12_merged_f32_mec_pm4_packets.h": PAL + "gfx12/chip/gfx12_merged_f32_mec_pm4_packets.h",
}

# counter_defs.yaml carries no notice: its project's licence, the MIT licence
# of ROCM + "rocprofiler-compute/LICENSE.md", covers it.
ROCPROF_LICENCE = """\
MIT License

Copyright (C) Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# What the runtime reads

# The GC registers kept of each version: those of compute queues, dispatches,
# performance counters, thread traces, and of bringing the GPU up (its queues,
# its memory hub, its firmware engines).
VM = r"reg(GC|MM)"
GC_REGISTERS = [
    r"regGRBM_(CNTL|GFX_CNTL|GFX_INDEX|SOFT_RESET)",
    r"regSCRATCH_REG[0-35-7]",
    r"regRLC_(SPARE_INT|CNTL|SRM_CNTL|SPM_MC_CNTL|CP_SCHEDULERS|RLCS_BOOTLOAD_STATUS|SAFE_MODE|CGCG_CGLS_CTRL|CGTT_MGCG_OVERRIDE)",
    r"regCP_(HQD_ACTIVE|HQD_DEQUEUE_REQUEST|HQD_EOP_CONTROL|HQD_IB_CONTROL|HQD_PERSISTENT_STATE|HQD_PQ_CONTROL|"
    r"HQD_PQ_DOORBELL_CONTROL|HQD_PQ_WPTR_HI|MQD_BASE_ADDR|MQD_CONTROL|STAT|MEC_CNTL|MEC_RS64_CNTL|ME_CNTL|"
    r"MEC_DOORBELL_RANGE_(LOWER|UPPER)|PQ_STATUS|RB_WPTR_POLL_CNTL|INT_CNTL|ME1_PIPE0_INT_CNTL|(PFP|ME|MEC_RS64)_PRGRM_CNTR_START(_HI)?)",
    r"regSH_MEM_(CONFIG|BASES)", r"regSPI_COMPUTE_QUEUE_RESET", r"regTCP_(CNTL|UTCL1_CNTL2)",
    r"regGB_ADDR_CONFIG", r"regCOMPUTE_TMPRING_SIZE",
    r"regSDMA[01]_RLC_CGCG_CTRL", r"regSDMA0_(F32_CNTL|MCU_CNTL|WATCHDOG_CNTL|UTCL1_CNTL|UTCL1_PAGE|CNTL)",
    r"regSDMA0_QUEUE0_(RB_CNTL|RB_BASE(_HI)?|RB_RPTR(_HI)?|RB_WPTR(_HI)?|RB_RPTR_ADDR_(LO|HI)|"
    r"RB_WPTR_POLL_ADDR_(LO|HI)|DOORBELL|DOORBELL_OFFSET|MINOR_PTR_UPDATE|IB_CNTL|PREEMPT)",
    r"regCOMPUTE_(DISPATCH_INITIATOR|START_X|PGM_LO|DISPATCH_SCRATCH_BASE_LO|PGM_RSRC1|RESOURCE_LIMITS|RESTART_X|"
    r"PGM_RSRC3|USER_DATA_0|PERFCOUNT_ENABLE|THREAD_TRACE_ENABLE)",
    r"regCP_PERFMON_CNTL(_1)?", r"regSQ_PERFCOUNTER_(CTRL2?|MASK)", r"reg(GRBM|GL2C|TCC|SQ)_PERFCOUNTER\d+_(SELECT|LO|HI)",
    r"regSQ_THREAD_TRACE_\w+", r"regSPI_CONFIG_CNTL", r"regSPI_SQG_EVENT_CTL",
    VM + r"VM_CONTEXT0_(CNTL|PAGE_TABLE_(START|END|BASE)_ADDR_(LO32|HI32))",
    VM + r"VM_INVALIDATE_ENG17_(REQ|ACK|SEM)",
    VM + r"VM_INVALIDATE_ENG\d+_ADDR_RANGE_(LO32|HI32)",
    VM + r"VM_L2_(CNTL[2-5]?|PROTECTION_FAULT_(CNTL2?|STATUS(_LO32)?|DEFAULT_ADDR_(LO32|HI32)|ADDR_(LO32|HI32))|"
    r"CONTEXT1_IDENTITY_APERTURE_(LOW|HIGH)_ADDR_(LO32|HI32)|CONTEXT_IDENTITY_PHYSICAL_OFFSET_(LO32|HI32)|"
    r"BANK_SELECT_RESERVED_CID2)",
    VM + r"MC_VM_(AGP_(BASE|BOT|TOP)|SYSTEM_APERTURE_(LOW|HIGH)_ADDR|SYSTEM_APERTURE_DEFAULT_ADDR_(LSB|MSB)|"
    r"MX_L1_TLB_CNTL|FB_LOCATION_(BASE|TOP)|XGMI_LFB_(CNTL|SIZE))",
    r"regMM_ATC_L2_MISC_CG",
]
GC_SEGMENTS = [f"GC_BASE__INST0_SEG{i}" for i in range(6)]
GC_BASES = {9: "vega20_ip_offset.h", 10: "sienna_cichlid_ip_offset.h"}

# The registers a dispatch sets, by the name of their address in the
# dispatch's record.
DISPATCH_REGISTERS = [("pgm_lo", "regCOMPUTE_PGM_LO"), ("pgm_rsrc1", "regCOMPUTE_PGM_RSRC1"),
                      ("pgm_rsrc3", "regCOMPUTE_PGM_RSRC3"), ("tmpring_size", "regCOMPUTE_TMPRING_SIZE"),
                      ("scratch_base_lo", "regCOMPUTE_DISPATCH_SCRATCH_BASE_LO"),
                      ("restart_x", "regCOMPUTE_RESTART_X"), ("user_data_0", "regCOMPUTE_USER_DATA_0"),
                      ("resource_limits", "regCOMPUTE_RESOURCE_LIMITS"), ("start_x", "regCOMPUTE_START_X")]

# PM4: the same in soc15d.h (GFX9) and nvd.h (GFX10 on).
PM4_CONSTANTS = [
    "PACKET_TYPE3", "PACKET3_SET_SH_REG", "PACKET3_SET_SH_REG_START", "PACKET3_SET_SH_REG_END", "PACKET3_SET_UCONFIG_REG",
    "PACKET3_SET_UCONFIG_REG_START", "PACKET3_PRED_EXEC", "PACKET3_WAIT_REG_MEM", "PACKET3_ACQUIRE_MEM",
    "PACKET3_RELEASE_MEM", "PACKET3_DISPATCH_DIRECT", "PACKET3_EVENT_WRITE", "PACKET3_INDIRECT_BUFFER", "PACKET3_COPY_DATA",
    "PACKET3_WRITE_DATA", "INDIRECT_BUFFER_VALID", "WR_ONE_ADDR", "WR_CONFIRM",
    "PACKET3_WAIT_REG_MEM__FUNCTION__EQUAL_TO_THE_REFERENCE_VALUE",
    "PACKET3_WAIT_REG_MEM__FUNCTION__GREATER_THAN_OR_EQUAL_REFERENCE_VALUE",
]
# The release's enumerations, in kfd_pm4_headers_ai.h.
PM4_ENUMS = ["event_index__mec_release_mem__end_of_pipe", "data_sel__mec_release_mem__send_32_bit_low",
             "data_sel__mec_release_mem__send_64_bit_data", "int_sel__mec_release_mem__none",
             "int_sel__mec_release_mem__send_interrupt_after_write_confirm"]
# soc15d.h alone: the destinations of WRITE_DATA, and the source and destination
# of COPY_DATA.
PM4_SOC15_ONLY = ["PACKET3_WRITE_DATA__DST_SEL__MEM_MAPPED_REGISTER", "PACKET3_WRITE_DATA__DST_SEL__MEMORY",
                  "PACKET3_COPY_DATA__SRC_SEL__PERFCOUNTERS", "PACKET3_COPY_DATA__SRC_SEL__GPU_CLOCK_COUNT",
                  "PACKET3_COPY_DATA__DST_SEL__TC_L2", "PACKET3_COPY_DATA__COUNT_SEL__64_BITS_OF_DATA",
                  "PACKET3_COPY_DATA__WR_CONFIRM__WAIT_FOR_CONFIRMATION"]
# Fields, as the shift of their first bit, from the argument macros of both
# headers: (GFX9's name, GFX10's).
PM4_SHIFTS = [("WAIT_REG_MEM_MEM_SPACE",) * 2, ("WAIT_REG_MEM_FUNCTION",) * 2, ("WRITE_DATA_DST_SEL",) * 2,
              ("PACKET3_COPY_DATA__SRC_SEL",) * 2, ("PACKET3_COPY_DATA__DST_SEL",) * 2,
              ("PACKET3_COPY_DATA__COUNT_SEL",) * 2, ("PACKET3_COPY_DATA__WR_CONFIRM",) * 2, ("EVENT_TYPE",) * 2,
              ("EVENT_INDEX",) * 2, ("DATA_SEL", "PACKET3_RELEASE_MEM_DATA_SEL"),
              ("INT_SEL", "PACKET3_RELEASE_MEM_INT_SEL"), ("EVENT_TYPE", "PACKET3_RELEASE_MEM_EVENT_TYPE"),
              ("EVENT_INDEX", "PACKET3_RELEASE_MEM_EVENT_INDEX")]
PM4_NV_SHIFTS = [f"PACKET3_ACQUIRE_MEM_GCR_CNTL_{f}" for f in
                 ("GLI_INV", "GLM_INV", "GLM_WB", "GLK_INV", "GLK_WB", "GLV_INV", "GL1_INV", "GL2_INV", "GL2_WB")]
PM4_NV_CONSTANTS = ["PACKET3_WAIT_REG_MEM64"] + [
    f"PACKET3_RELEASE_MEM_GCR_{f}" for f in ("GLM_WB", "GLM_INV", "GL2_WB", "SEQ")]
PM4_SOC15_SHIFTS = [f"PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_{f}" for f in
                    ("SH_ICACHE_ACTION_ENA", "SH_KCACHE_ACTION_ENA", "TC_ACTION_ENA", "TCL1_ACTION_ENA", "TC_WB_ACTION_ENA")]
PM4_SOC15_CONSTANTS = ["EOP_TC_WB_ACTION_EN", "EOP_TC_NC_ACTION_EN"]

# WAIT_REG_MEM64, which the kernel's headers name but do not lay out: PAL's
# layout of it, (field, byte offset and bytes, or bit offset and bits), which
# the encoder writes in this order.
WAIT_REG_MEM64 = "PM4_MEC_WAIT_REG_MEM64"
WAIT_REG_MEM64_LAYOUT = [
    ("ordinal2__bitfields__function", ("bits", 32, 3)), ("ordinal2__bitfields__mem_space", ("bits", 36, 2)),
    ("ordinal3__bitfieldsA__mem_poll_addr_lo", ("bits", 67, 29)), ("ordinal4__mem_poll_addr_hi", (12, 4)),
    ("ordinal5__reference", (16, 4)), ("ordinal6__reference_hi", (20, 4)), ("ordinal7__mask", (24, 4)),
    ("ordinal8__mask_hi", (28, 4)), ("ordinal9__bitfields__poll_interval", ("bits", 256, 16))]
PAL_HEADERS = ["gfx9_plus_merged_f32_mec_pm4_packets.h", "gfx12_merged_f32_mec_pm4_packets.h"]

# The events and thread trace values of the SOC enumerations, the same in each
# that defines them.
SOCS = ["vega10_enum.h", "soc21_enum.h", "soc24_enum.h"]
SOC_EVENTS = ["CACHE_FLUSH_AND_INV_TS_EVENT", "CS_PARTIAL_FLUSH", "THREAD_TRACE_MARKER", "THREAD_TRACE_FINISH"]
SOC_TRACE = ["SQ_TT_RT_FREQ_4096_CLK", "SQ_TT_WTYPE_INCLUDE_PS_BIT", "SQ_TT_WTYPE_INCLUDE_GS_BIT",
             "SQ_TT_WTYPE_INCLUDE_HS_BIT", "SQ_TT_WTYPE_INCLUDE_CS_BIT", "SQ_TT_TOKEN_MASK_SQDEC_BIT",
             "SQ_TT_TOKEN_MASK_SHDEC_BIT", "SQ_TT_TOKEN_MASK_GFXUDEC_BIT", "SQ_TT_TOKEN_MASK_COMP_BIT",
             "SQ_TT_TOKEN_MASK_CONTEXT_BIT", "SQ_TT_TOKEN_MASK_CONFIG_BIT", "SQ_TT_TOKEN_EXCLUDE_VMEMEXEC_SHIFT", "SQ_TT_TOKEN_EXCLUDE_ALUEXEC_SHIFT",
             "SQ_TT_TOKEN_EXCLUDE_VALUINST_SHIFT", "SQ_TT_TOKEN_EXCLUDE_IMMEDIATE_SHIFT", "SQ_TT_TOKEN_EXCLUDE_INST_SHIFT"]

# GFX9's thread trace tokens: each SQ_THREAD_TRACE_TOKEN_* type of
# vega10_enum.h, and the SQ_THREAD_TRACE_WORD_* registers of gc_9_4_3's
# headers that lay out its words, in order. The words of a type follow its
# name, the _1_OF_2 and _2_OF_2 of a token of two; WAVE_ALLOC and WAVE_END
# take WORD_WAVE, EVENT_CS and EVENT_GFX1 WORD_EVENT, REG_CSPRIV WORD_REG_CS.
GFX9_TOKENS = {
    "MISC": ["MISC"], "TIMESTAMP": ["TIMESTAMP_1_OF_2", "TIMESTAMP_2_OF_2"], "REG": ["REG_1_OF_2", "REG_2_OF_2"],
    "WAVE_START": ["WAVE_START"], "WAVE_ALLOC": ["WAVE"], "REG_CSPRIV": ["REG_CS_1_OF_2", "REG_CS_2_OF_2"],
    "WAVE_END": ["WAVE"], "EVENT": ["EVENT"], "EVENT_CS": ["EVENT"], "EVENT_GFX1": ["EVENT"], "INST": ["INST"],
    "INST_PC": ["INST_PC_1_OF_2", "INST_PC_2_OF_2"], "INST_USERDATA": ["INST_USERDATA_1_OF_2", "INST_USERDATA_2_OF_2"],
    "ISSUE": ["ISSUE"], "PERF": ["PERF_1_OF_2", "PERF_2_OF_2"], "REG_CS": ["REG_CS_1_OF_2", "REG_CS_2_OF_2"],
}
# The fields the decoder reads: (word, field).
GFX9_TOKEN_FIELDS = [("CMN", "TIME_DELTA"), ("MISC", "TIME_DELTA"), ("WAVE_START", "CU_ID"), ("WAVE_START", "WAVE_ID"),
                     ("WAVE_START", "SIMD_ID"), ("WAVE", "CU_ID"), ("WAVE", "WAVE_ID"), ("WAVE", "SIMD_ID"),
                     ("TIMESTAMP_1_OF_2", "TIME_LO"), ("TIMESTAMP_2_OF_2", "TIME_HI")]

# SDMA packets: the same in each version's header but for the fence's memory
# type, from version 5.
SDMA_PKT = {(4, 0, 0): "vega10_sdma_pkt_open.h", (5, 0, 0): "navi10_sdma_pkt_open.h",
            (6, 0, 0): "sdma_v6_0_0_pkt_open.h"}
SDMA_OPS = ["SDMA_OP_COPY", "SDMA_OP_FENCE", "SDMA_OP_TRAP", "SDMA_OP_POLL_REGMEM", "SDMA_OP_TIMESTAMP",
            "SDMA_SUBOP_COPY_LINEAR", "SDMA_SUBOP_TIMESTAMP_GET_GLOBAL"]
SDMA_FIELDS = ["SDMA_PKT_COPY_LINEAR_HEADER_sub_op", "SDMA_PKT_POLL_REGMEM_HEADER_func",
               "SDMA_PKT_POLL_REGMEM_HEADER_mem_poll", "SDMA_PKT_POLL_REGMEM_DW5_interval",
               "SDMA_PKT_POLL_REGMEM_DW5_retry_count", "SDMA_PKT_FENCE_HEADER_mtype",
               "SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_sub_op"]
SDMA_OPTIONAL = ["SDMA_PKT_FENCE_HEADER_mtype"]  # GFX9 engines take no memory type

# AQL, from hsa.h.
HSA_CONSTANTS = ["HSA_PACKET_HEADER_TYPE", "HSA_PACKET_HEADER_BARRIER", "HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE",
                 "HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE", "HSA_FENCE_SCOPE_SYSTEM", "HSA_PACKET_TYPE_VENDOR_SPECIFIC",
                 "HSA_PACKET_TYPE_KERNEL_DISPATCH", "HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS"]
DISPATCH = "hsa_kernel_dispatch_packet_s"
DISPATCH_FIELDS = ["header", "setup", "workgroup_size_x", "workgroup_size_y", "workgroup_size_z", "grid_size_x",
                   "grid_size_y", "grid_size_z", "private_segment_size", "group_segment_size", "kernel_object",
                   "kernarg_address"]

# The scratch buffer descriptor, by GC major: the unions of its words 1 and 3
# in registers.h, and the values its fields take.
SQ_BUF_RSRC = {9: ("SQ_BUF_RSRC_WORD1", "SQ_BUF_RSRC_WORD3"),
               11: ("SQ_BUF_RSRC_WORD1_GFX11", "SQ_BUF_RSRC_WORD3_GFX11"),
               12: ("SQ_BUF_RSRC_WORD1_GFX11", "SQ_BUF_RSRC_WORD3_GFX12")}
SQ_WORD1 = ["BASE_ADDRESS_HI", "SWIZZLE_ENABLE"]
SQ_WORD3 = ["DST_SEL_X", "DST_SEL_Y", "DST_SEL_Z", "DST_SEL_W", "ADD_TID_ENABLE", "TYPE"]
SQ_WORD3_OPTIONAL = ["NUM_FORMAT", "DATA_FORMAT", "ELEMENT_SIZE", "INDEX_STRIDE", "FORMAT", "OOB_SELECT"]
SQ_CONSTANTS = ["SQ_SEL_X", "SQ_SEL_Y", "SQ_SEL_Z", "SQ_SEL_W", "SQ_RSRC_BUF", "BUF_FORMAT_32_UINT",
                "BUF_NUM_FORMAT_UINT", "BUF_DATA_FORMAT_32"]

# The kernel descriptor's fields, by their offsets in AMDHSAKernelDescriptor.h,
# and its code properties.
KD = "kernel_descriptor_t"
KD_FIELDS = ["group_segment_fixed_size", "private_segment_fixed_size", "kernarg_size", "kernel_code_entry_byte_offset",
             "compute_pgm_rsrc3", "compute_pgm_rsrc1", "compute_pgm_rsrc2", "kernel_code_properties"]
KD_PROPERTIES = ["ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER", "ENABLE_SGPR_DISPATCH_PTR", "ENABLE_WAVEFRONT_SIZE32"]
# ELF.h's values of the AMDGPU header: its machine, its ABI version, and the
# fields of its flags; and the type of a section of notes and of the note of a
# code object's metadata.
ELF_CONSTANTS = ["EM_AMDGPU", "ELFABIVERSION_AMDGPU_HSA_V6", "EF_AMDGPU_MACH",
                 "EF_AMDGPU_GENERIC_VERSION", "EF_AMDGPU_GENERIC_VERSION_OFFSET",
                 "SHT_NOTE", "NT_AMDGPU_METADATA"]
RST_TABLES = ["amdgpu-ef-amdgpu-mach-table", "amdgpu-generic-processor-table"]

# The performance counters of the blocks a profile counts, for the processors
# the library knows, as rocprofiler's table defines them: each counter's name,
# block and event, which differ between processors of one GC major.
COUNTER_BLOCKS = ["GRBM", "GL2C", "TCC", "SQ"]
COUNTER_PROCESSORS = ["gfx942", "gfx950", "gfx11", "gfx12"]

# The configuration the headers are read under: a little-endian processor, a
# 64-bit HSA model.
DEFINED = {"LITTLEENDIAN_CPU", "HSA_LARGE_MODEL", "HSA_LITTLE_ENDIAN"}

# Reading C headers

DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(\w+)(\([\w, ]*\))?[ \t]*(.*?)[ \t]*(?:/\*.*?\*/|//.*)?[ \t]*$", re.M)
ENUMERATOR = re.compile(r"^[ \t]*(\w+)[ \t]*=[ \t]*([^,/\n]+?)[ \t]*,?[ \t]*(?:/\*.*|//.*)?$", re.M)


def defines(text):
    """{name: (parameters or None, body)} of [text]'s #define lines."""
    return {m.group(1): (m.group(2), m.group(3)) for m in DEFINE.finditer(text)}


def enumerators(text):
    return {m.group(1): m.group(2) for m in ENUMERATOR.finditer(text)}


def evaluate(expr, names, arg=None):
    """The integer of the C constant expression [expr]: numbers, names of
    [names], casts to integer types, and integer operators. [arg] is the value
    of a macro's parameter."""
    e = re.sub(r"\((unsigned|unsigned int|uint32_t|uint64_t|int)\)", "", expr)
    e = re.sub(r"\b(0[xX][0-9a-fA-F]+|\d+)[uUlL]*\b", r"\1", e)

    def name(m):
        n = m.group(0)
        if n in ("x", "_x") and arg is not None:
            return str(arg)
        if n.startswith(("0x", "0X")) or n.isdigit():
            return n
        if n not in names:
            sys.exit(f"{expr}: {n} is not defined")
        return str(evaluate(names[n], names))
    e = re.sub(r"\b[A-Za-z_]\w*\b", name, e)
    if not re.fullmatch(r"[0-9a-fA-FxX ()|&<>~+\-*]*", e):
        sys.exit(f"{expr}: not a constant expression")
    return eval(e, {"__builtins__": {}})


def constants(text, wanted):
    """{name: value} of the macros and enumerators [wanted] of [text]."""
    ds, es = defines(text), enumerators(text)
    names = {**es, **{n: b for n, (p, b) in ds.items() if p is None}}
    out = {}
    for n in wanted:
        if n not in names:
            sys.exit(f"undefined constant {n}")
        out[n] = evaluate(names[n], names) & 0xffff_ffff_ffff_ffff
    return out


def shifts(text, wanted):
    """The first bit of each argument macro [name(x)], [name(1)]'s."""
    ds = defines(text)
    names = {n: b for n, (p, b) in ds.items() if p is None}
    out = {}
    for n in wanted:
        if n not in ds or ds[n][0] is None:
            sys.exit(f"undefined argument macro {n}")
        out[n] = (evaluate(ds[n][1], names, arg=1) & 0xffffffff).bit_length() - 1
    return out


def preprocess(text):
    """[text] without its comments, its #if branches taken under [DEFINED]."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    out = []
    taking = []  # per open #if: (this branch is taken, a branch was taken)
    for line in text.splitlines():
        m = re.match(r"\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)", line)
        if m is None:
            if all(t for t, _ in taking):
                out.append(line)
            continue
        d, cond = m.group(1), m.group(2)
        holds = any(n in DEFINED for n in re.findall(r"\w+", cond))
        if d in ("if", "ifdef"):
            taking.append((holds, holds))
        elif d == "ifndef":
            taking.append((not holds, not holds))
        elif d == "elif":
            _, done = taking[-1]
            taking[-1] = (not done and holds, done or holds)
        elif d == "else":
            _, done = taking[-1]
            taking[-1] = (not done, True)
        else:
            taking.pop()
    return "\n".join(out)


# The bytes and alignment of the scalar types of the headers' structs.
SCALARS = {"uint8_t": 1, "uint16_t": 2, "uint32_t": 4, "uint64_t": 8, "int8_t": 1, "int16_t": 2, "int32_t": 4,
           "int64_t": 8, "unsigned int": 4, "signed int": 4, "int": 4, "float": 4, "void*": 8}


def layout(text, name):
    """(sizeof, {path: (byte offset, bytes) or ("bits", bit offset, bits)}) of
    the struct or union [name] defined in [text], nested fields joined with
    "__", for x86_64 Linux. An array's bytes are its element's."""
    src = preprocess(text)
    tokens = re.findall(r"[A-Za-z_]\w*\s*\*|[A-Za-z_]\w*|\d+|[{};:\[\],]", src)
    defs = {}  # every struct and union: name -> token range

    def definition(i):
        """The struct or union whose keyword is at [i]: (kind, tag, members, end)."""
        kind = tokens[i]
        j = i + 1
        tag = None
        if tokens[j] != "{":
            tag = tokens[j]
            j += 1
        depth, k = 0, j
        while True:
            if tokens[k] == "{":
                depth += 1
            elif tokens[k] == "}":
                depth -= 1
                if depth == 0:
                    return kind, tag, (j + 1, k), k + 1
            k += 1

    def opens(i):
        """Whether the struct or union keyword at [i] starts a definition."""
        j = i + 1 if tokens[i + 1] == "{" else i + 2
        return j < len(tokens) and tokens[j] == "{"

    for i, t in enumerate(tokens):
        if t in ("struct", "union") and i + 2 < len(tokens) and opens(i):
            kind, tag, body, end = definition(i)
            defs.setdefault(tag, (kind, body))
            if i > 0 and tokens[i - 1] == "typedef":
                defs.setdefault(tokens[end], (kind, body))

    def walk(kind, body, base, prefix, fields):
        """Lays out the members of [body] from bit [base]; the size in bits."""
        i, end = body
        pos = 0  # bits from [base]
        size = 0
        align = 1
        while i < end:
            if tokens[i] in ("struct", "union"):
                k, _, inner, after = definition(i)
                names = []
                while tokens[after] != ";":
                    if tokens[after] not in (",",):
                        names.append(tokens[after])
                    after += 1
                ialign = 4  # every nested record of these headers holds 32-bit words
                start = 0 if kind == "union" else -(-pos // (ialign * 8)) * ialign * 8
                for n in names or [None]:
                    p = prefix if n is None else (prefix + "__" if prefix else "") + n
                    isize = walk(k, inner, base + start, p, fields)
                    if n is not None:
                        fields[p] = ((base + start) // 8, isize // 8)
                if kind == "union":
                    size = max(size, isize)
                else:
                    pos = start + isize
                align = max(align, ialign)
                i = after + 1
                continue
            j = i
            while tokens[j] not in (";",):
                j += 1
            decl = tokens[i:j]
            i = j + 1
            if not decl:
                continue
            if ":" in decl:
                c = decl.index(":")
                ftype, fname, width = " ".join(decl[:c - 1]), decl[c - 1], int(decl[c + 1])
                start = 0 if kind == "union" else pos
                p = (prefix + "__" if prefix else "") + fname
                fields[p] = ("bits", base + start, width)
                if kind == "union":
                    size = max(size, width)
                else:
                    pos += width
                align = max(align, 4)
                continue
            count = 1
            if "[" in decl:
                b = decl.index("[")
                count = int(decl[b + 1])
                decl = decl[:b]
            ftype, fname = " ".join(decl[:-1]).replace(" *", "*").replace("const ", ""), decl[-1]
            if ftype.endswith("*"):
                ftype = "void*"
            if ftype in SCALARS:
                fbytes = SCALARS[ftype]
                falign = fbytes
            elif ftype in defs:
                k, inner = defs[ftype]
                sub = {}
                fbytes = walk(k, inner, 0, "", sub) // 8
                falign = 8 if fbytes % 8 == 0 else 4
            elif ftype.endswith("_enum"):
                fbytes, falign = 4, 4
            else:
                sys.exit(f"{name}: unknown type {ftype}")
            start = 0 if kind == "union" else -(-pos // (falign * 8)) * falign * 8
            p = (prefix + "__" if prefix else "") + fname
            fields[p] = ((base + start) // 8, fbytes)
            if kind == "union":
                size = max(size, fbytes * count * 8)
            else:
                pos = start + fbytes * count * 8
            align = max(align, falign)
        total = size if kind == "union" else pos
        return -(-total // (align * 8)) * align * 8

    for tag, (kind, body) in defs.items():
        if tag == name:
            fields = {}
            return walk(kind, body, 0, "", fields) // 8, fields
    sys.exit(f"no struct {name}")

# Registers


def split_name(name):
    pos = next((i for i, c in enumerate(name) if c.isupper()), len(name))
    return name[:pos], name[pos:]


def normalize(reg):
    """A VM register of the GC's hub takes its prefix, as regGCVM_CONTEXT0_CNTL."""
    s = split_name(reg)
    return s[0] + "GC" + s[1] if s[1].startswith(("VM_", "MC_VM_")) else reg


REG_DEFINE = re.compile(r"^#define\s+((?:mm|reg)\w+?)(_BASE_IDX)?\s+(0x[\da-fA-F]+|\d+)[uUlL]*\s*$", re.M)
MASK_DEFINE = re.compile(r"^#define\s+(\w+?)__(\w+)_MASK\s+(0x[\da-fA-F]+|\d+)[uUlL]*\s*$", re.M)


def kept(reg):
    return any(re.fullmatch(p, normalize(reg)) for p in GC_REGISTERS)


def gc_registers(offsets, masks):
    """The registers of a GC version: {name: (offset, segment, [(field, lo, hi)])}."""
    offs, segs = {}, {}
    for m in REG_DEFINE.finditer(offsets):
        (segs if m.group(2) else offs)[m.group(1)] = int(m.group(3), 0)
    fields = {}
    for m in MASK_DEFINE.finditer(masks):
        mask = int(m.group(3), 0)
        fields.setdefault(m.group(1), []).append((m.group(2).lower(), (mask & -mask).bit_length() - 1,
                                                   mask.bit_length() - 1))
    return {normalize(r): (off, segs[r], fields.get(split_name(r)[1], []))
            for r, off in offs.items() if r in segs and kept(r)}

# Processors


def rst_table(lines, name):
    """The rows of the reStructuredText simple table [name], each a list of its
    columns' lines."""
    start = lines.index(f"     :name: {name}") + 2
    rule = lines[start]
    cols = [(m.start(), m.end()) for m in re.finditer(r"=+", rule)]
    body = lines[start + 1:]
    head = next(i for i, l in enumerate(body) if l.strip() and set(l.strip()) <= {"=", " "})
    rows = []
    for l in body[head + 1:]:
        if l.strip() and set(l.strip()) <= {"=", " "}:
            return rows
        cells = [l[a:b if i + 1 < len(cols) else None].strip() for i, (a, b) in enumerate(cols)]
        if cells[0] or not rows:
            rows.append([[] for _ in cols])
        for i, c in enumerate(cells):
            if c:
                rows[-1][i].append(c)
    sys.exit(f"table {name} does not end")


def rst_table_lines(lines, name):
    """The lines of the table [name]: from its directive to its last rule."""
    at = lines.index(f"     :name: {name}")
    start = max(i for i in range(at) if lines[i].lstrip().startswith(".. table::"))
    rules = [i for i in range(at + 2, len(lines)) if lines[i].strip() and set(lines[i].strip()) <= {"=", " "}]
    return lines[start:rules[2] + 1]


def processors(elf_h, usage):
    """LLVM's AMDGCN processors by their [EF_AMDGPU_MACH] value, and the
    processors each generic one lists, from ELF.h and AMDGPUUsage.rst, which
    must agree."""
    enums = {n: evaluate(v, {}) for n, v in enumerators(elf_h).items() if re.fullmatch(r"0x[0-9a-fA-F]+|\d+", v)}
    missing = [c for c in ELF_CONSTANTS if c not in enums]
    if missing:
        sys.exit(f"ELF.h lacks {missing}")
    lines = usage.splitlines()
    code = re.compile(r"``([^`]+)``")
    machs = []
    for name, value, desc in rst_table(lines, RST_TABLES[0]):
        m = code.fullmatch(name[0])
        if not (m and m.group(1).startswith("EF_AMDGPU_MACH_AMDGCN_")):
            continue
        if enums.get(m.group(1)) != int(value[0], 16):
            sys.exit(f"{m.group(1)}: ELF.h and AMDGPUUsage.rst differ")
        machs.append((int(value[0], 16), code.match(desc[0]).group(1)))
    names = {n for _, n in machs}
    generic = []
    for row in rst_table(lines, RST_TABLES[1]):
        name = code.fullmatch(row[0][0]).group(1)
        members = [code.search(l).group(1) for l in row[2] if l.startswith("- ")]
        if name not in names or not set(members) <= names:
            sys.exit(f"{name}: a generic processor or member without an EF_AMDGPU_MACH value")
        generic.append((name, members))
    return {c: enums[c] for c in ELF_CONSTANTS}, machs, generic

# Excerpts

KD_PROPERTY = re.compile(r"^\s*KERNEL_CODE_PROPERTY\((\w+),\s*(\d+),\s*(\d+)\),.*$", re.M)


def wanted():
    """For each excerpt: the macros, enumerators and structs it keeps."""
    w = {name: {"defines": set(), "optional": set(), "enums": set(), "structs": set()} for name in SOURCES}
    w["soc15d.h"]["defines"] |= {*PM4_CONSTANTS, *PM4_SOC15_ONLY, *PM4_SOC15_CONSTANTS, *PM4_SOC15_SHIFTS,
                                 *(g for g, _ in PM4_SHIFTS)}
    w["nvd.h"]["defines"] |= {*PM4_CONSTANTS, *PM4_NV_CONSTANTS, *PM4_NV_SHIFTS, *(n for _, n in PM4_SHIFTS)}
    w["kfd_pm4_headers_ai.h"]["enums"] |= set(PM4_ENUMS)
    for h in SDMA_PKT.values():
        fields = {f"{n}_{k}" for n in SDMA_FIELDS for k in ("mask", "shift")}
        optional = {f"{n}_{k}" for n in SDMA_OPTIONAL for k in ("mask", "shift")}
        w[h]["defines"] |= {*SDMA_OPS, *(fields - optional)}
        w[h]["optional"] |= optional
    for h in GC_BASES.values():
        w[h]["optional"] |= set(GC_SEGMENTS)
    for h in SOCS:
        w[h]["optional"] |= {*SOC_EVENTS, *SOC_TRACE}
    w["vega10_enum.h"]["enums"] |= {f"SQ_THREAD_TRACE_TOKEN_{t}" for t in GFX9_TOKENS}
    w["hsa.h"]["enums"] |= set(HSA_CONSTANTS)
    w["hsa.h"]["structs"] |= {DISPATCH, "hsa_signal_s"}
    w["registers.h"]["enums"] |= set(SQ_CONSTANTS)
    w["registers.h"]["structs"] |= {u for pair in SQ_BUF_RSRC.values() for u in pair}
    w["AMDHSAKernelDescriptor.h"]["enums"] |= {f.upper() + "_OFFSET" for f in KD_FIELDS}
    w["AMDHSAKernelDescriptor.h"]["structs"] |= {KD}
    w["ELF.h"]["enums"] |= set(ELF_CONSTANTS)
    for h in PAL_HEADERS:
        w[h]["structs"] |= {WAIT_REG_MEM64, "PM4_MEC_TYPE_3_HEADER"}
    return w


def counter_defs(text):
    """The (name, block, event, processors) of counter_defs.yaml's counters
    of COUNTER_BLOCKS: a list of counters, each "- name:", whose definitions
    each list "- architectures:", then "block:" and "event:", indented one
    way or another."""
    out, name, procs, block, lines = [], None, [], None, {}
    for i, l in enumerate(text.splitlines()):
        m = re.match(r"  - name: (\S+)$", l)
        if m:
            name, procs, block = m.group(1), [], None
            continue
        if re.match(r"\s*- architectures:$", l):
            procs, block, first = [], None, i
            continue
        m = re.match(r"\s*- (gfx\w+)$", l)
        if m:
            procs.append(m.group(1))
            continue
        m = re.match(r"\s*block: (\w+)$", l)
        if m:
            block = m.group(1)
            continue
        m = re.match(r"\s*event: (\d+)$", l)
        if m and block in COUNTER_BLOCKS:
            out.append((name, block, int(m.group(1)), procs))
            lines[(name, block, m.group(1))] = (first, i)
    return out, lines


def licence(name, text):
    """The header's leading comments, its licence notice, then the source the
    excerpt is cut from; for a document, a comment that names its source and
    licence."""
    if name.endswith(".yaml"):
        return (f"# Excerpt of {SOURCES[name]}.\n"
                "# rocprofiler-compute's LICENSE.md covers it:\n#\n"
                + "".join(f"# {l}".rstrip() + "\n" for l in ROCPROF_LICENCE.splitlines()))
    if name.endswith(".rst"):
        return (f".. Excerpt of {SOURCES[name]}.\n"
                ".. LLVM is under the Apache License v2.0 with LLVM Exceptions (SPDX:\n"
                ".. Apache-2.0 WITH LLVM-exception).\n")
    m = re.match(r"\s*((?:/\*.*?\*/\s*|//[^\n]*\n)+)", text, re.S)
    if m is None:
        sys.exit(f"{name}: no licence notice")
    return m.group(1).rstrip() + f"\n\n/* Excerpt of {SOURCES[name]}. */\n"


def blocks(text, names):
    """The lines of the definitions of the structs and unions [names]: from the
    line of their keyword to the line of their closing brace."""
    lines = text.splitlines()
    out, found = set(), set()
    for i, l in enumerate(lines):
        m = re.match(r"\s*(?:typedef\s+)?(?:struct|union)\s+(\w+)\s*(\{|$)", l)
        if m is None or m.group(1) not in names or m.group(1) in found:
            continue
        depth, j = 0, i
        while True:
            depth += lines[j].count("{") - lines[j].count("}")
            if "{" in "".join(lines[i:j + 1]) and depth == 0:
                break
            j += 1
        out |= set(range(i, j + 1))
        found.add(m.group(1))
    if found != names:
        sys.exit(f"no definition of {sorted(names - found)}")
    return out


def excerpt(name, text):
    """[name]'s excerpt of [text]: its licence notice, its URL and the lines
    this script reads, in order."""
    if name.endswith(".rst"):
        lines = text.splitlines()
        return licence(name, text) + "\n" + "\n\n".join("\n".join(rst_table_lines(lines, t)) for t in RST_TABLES) + "\n"
    if name.endswith(".yaml"):
        lines = text.splitlines()
        keep = set()
        _, spans = counter_defs(text)
        for (n, _, _), (first, last) in spans.items():
            keep.add(next(i for i in range(first, -1, -1) if lines[i] == f"  - name: {n}"))
            keep.add(next(i for i in range(first, -1, -1) if lines[i].strip() == "definitions:"))
            keep |= set(range(first, last + 1))
        return licence(name, text) + "\n".join(lines[i] for i in sorted(keep)) + "\n"
    w = wanted()[name]
    lines = text.splitlines()
    keep = set(blocks(text, w["structs"])) if w["structs"] else set()
    ds = {m.group(1) for m in DEFINE.finditer(text)}
    for i, l in enumerate(lines):
        m = DEFINE.match(l)
        if m and m.group(1) in w["defines"] | w["optional"]:
            keep.add(i)
        e = ENUMERATOR.match(l)
        if e and e.group(1) in w["enums"] | w["optional"]:
            keep.add(i)
        r = REG_DEFINE.match(l)
        if name.startswith("gc_") and r and kept(r.group(1)):
            keep.add(i)
        k = MASK_DEFINE.match(l)
        if name.startswith("gc_") and name.endswith("sh_mask.h") and k and kept("reg" + k.group(1)):
            keep.add(i)
        if name == "ELF.h" and e and e.group(1).startswith("EF_AMDGPU_MACH_AMDGCN_"):
            keep.add(i)
        p = KD_PROPERTY.match(l)
        if name == "AMDHSAKernelDescriptor.h" and p and p.group(1) in KD_PROPERTIES:
            keep.add(i)
    # The macros a kept one refers to.
    names = {DEFINE.match(lines[i]).group(1) for i in keep if DEFINE.match(lines[i])}
    while True:
        refs = {n for i in keep if DEFINE.match(lines[i])
                for n in re.findall(r"\b[A-Za-z_]\w*\b", DEFINE.match(lines[i]).group(3)) if n in ds} - names
        if not refs:
            break
        for i, l in enumerate(lines):
            m = DEFINE.match(l)
            if m and m.group(1) in refs:
                keep.add(i)
        names |= refs
    found = {DEFINE.match(lines[i]).group(1) for i in keep if DEFINE.match(lines[i])}
    found |= {ENUMERATOR.match(lines[i]).group(1) for i in keep if ENUMERATOR.match(lines[i])}
    missing = (w["defines"] | w["enums"]) - found
    if missing:
        sys.exit(f"{name}: no {sorted(missing)}")
    return licence(name, text) + "\n" + "\n".join(lines[i] for i in sorted(keep)) + "\n"


def fetch(url, cache, pins, pin):
    path = cache / (hashlib.sha256(url.encode()).hexdigest()[:16] + "-" + url.rsplit("/", 1)[-1])
    if not path.exists():
        print(f"fetching {url}", file=sys.stderr)
        req = urllib.request.Request(url, headers={"User-Agent": "raven-gen"})
        with urllib.request.urlopen(req, timeout=120) as r:
            path.write_bytes(r.read())
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if url in pins and pins[url] != digest:
        sys.exit(f"{url}: SHA-256 {digest}, expected {pins[url]}")
    if url not in pins:
        if not pin:
            sys.exit(f"{url} is not pinned; run with --pin to record {digest}")
        pins[url] = digest
    return data.decode("latin-1")

# Emission


def ml_name(c):
    return c.lower()


def ml_int(v):
    return f"0x{v:x}" if v > 9 else str(v)


def ml_version(v):
    return "(%d, %d, %d)" % v


def ml_field(v):
    return f"({v[1]}, {v[2]})" if v[0] == "bits" else f"({v[0]}, {v[1]})"


def same(what, values):
    if len(set(values)) != 1:
        sys.exit(f"{what} differs: {values}")
    return values[0]


MIT = """\
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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE."""


def notice(text):
    """The lines of [text]'s licence notice, a run of // comments, without
    their markers and rules."""
    m = re.match(r"\s*((?://[^\n]*\n)+)", text)
    lines = [l[2:].removeprefix(" ").rstrip() for l in m.group(1).splitlines()]
    return "\n".join(l for l in lines if not l.startswith("//")).strip("\n")


def header(h):
    """The licence header of defs.ml: raven's, then the notices of the sources
    of its values, from the excerpts [h]."""
    kind = {n: "llvm" if "LLVM Exceptions" in t else "ncsa" if "NCSA" in t else "mit" for n, t in h.items()}
    ncsa = sorted(n for n in h if kind[n] == "ncsa")
    llvm = sorted(n for n in h if kind[n] == "llvm")
    owners = sorted({m.group(0).rstrip(" */") for n, t in h.items() if kind[n] == "mit"
                     for m in re.finditer(r"Copyright [^\n]*?Advanced Micro Devices, Inc\.[^\n]*", t)})
    ncsa_notices = {notice(h[n]) for n in ncsa}
    if len(ncsa_notices) != 1:
        sys.exit(f"the notices of {ncsa} differ")
    indent = lambda s: "\n".join(f"   {l}".rstrip() for l in s.splitlines())
    return (
        "(*---------------------------------------------------------------------------\n"
        "  Copyright (c) 2026 The Raven authors. All rights reserved.\n"
        "  SPDX-License-Identifier: ISC\n"
        "  ---------------------------------------------------------------------------*)\n\n"
        "(* Generated by gen/gen.py from the excerpts in gen/headers; do not edit.\n"
        "   The command that regenerates this file is in gen/gen.py.\n\n"
        "   The values are AMD's and LLVM's, copied from their headers and documents.\n\n"
        f"   AMD's {' and '.join(ncsa)} are under the University of Illinois/NCSA\n"
        "   Open Source License:\n\n"
        + indent(ncsa_notices.pop()) + "\n\n"
        "   AMD's other sources are under the MIT licence:\n\n"
        + indent("\n".join(owners)) + "\n\n" + indent(MIT) + "\n\n"
        f"   LLVM's {', '.join(llvm[:-1])} and {llvm[-1]} are under\n"
        "   the Apache License v2.0 with LLVM Exceptions (SPDX-License-Identifier:\n"
        "   Apache-2.0 WITH LLVM-exception), whose text is in rig's LICENSE-llvm. *)\n"
    )


def generate(h):
    """defs.ml, from the excerpts [h], by name."""
    out = [header(h)]

    # Registers: each a literal record, each version's list of them and its
    # lookup by name, a match the compiler turns into a search of the names.
    # Versions are constant constructors and lookups are functions, so that
    # nothing is built when a program starts.
    tags = {ver: "Gc_" + "_".join(map(str, ver)) for ver in GC_VERSIONS}
    gc_regs = {}
    out += ["(* Registers *)", "",
            "type register = {", "  name : string;", "  offset : int;", "  segment : int;",
            "  fields : (string * (int * int)) list;", "}", "",
            "(* The GC versions with registers. *)",
            "type gc = No_gc | " + " | ".join(tags.values()), ""]
    for ver in GC_VERSIONS:
        v = "_".join(map(str, ver))
        regs = gc_registers(h[gc_header(ver, "offset")], h[gc_header(ver, "sh_mask")])
        gc_regs[ver] = regs
        out.append(f"(* GC {'.'.join(map(str, ver))}, its fields as (name, (lowest bit, highest bit)) *)")
        out.append("")
        for n, (off, seg, fields) in regs.items():
            fs = "; ".join(f"({json.dumps(f)}, ({lo}, {hi}))" for f, lo, hi in fields)
            out.append(f"let gc_{v}_{n} = {{ name = {json.dumps(n)}; offset = {ml_int(off)}; segment = {seg}; "
                       f"fields = [ {fs} ] }}")
        out += ["", f"let gc_{v}_registers = ["]
        out += [f"  gc_{v}_{n};" for n in regs]
        out += ["]", "", f"let gc_{v}_find = function"]
        out += [f"  | {json.dumps(n)} -> Some gc_{v}_{n}" for n in regs]
        out += ["  | _ -> None", ""]
    out += ["(* The GC version whose registers a GPU of GC [(major, minor, stepping)]",
            "   takes: the latest of its major at or before it. *)",
            "let gc ((major, minor, stepping) : int * int * int) =", "  match major with"]
    for major in sorted({v[0] for v in GC_VERSIONS}):
        vs = sorted((v for v in GC_VERSIONS if v[0] == major), reverse=True)
        conds = [f"if minor > {mi} || (minor = {mi} && stepping >= {st}) then {tags[(major, mi, st)]}"
                 for _, mi, st in vs]
        out.append(f"  | {major} -> " + " else ".join(conds) + " else No_gc")
    out += ["  | _ -> No_gc", "", "(* The registers of a GC version, in its headers' order. *)",
            "let registers = function", "  | No_gc -> []"]
    out += [f"  | {t} -> gc_{'_'.join(map(str, v))}_registers" for v, t in tags.items()]
    out += ["", "(* The register of a GC version named [name], if any. *)", "let find gc name =",
            "  match gc with", "  | No_gc -> None"]
    out += [f"  | {t} -> gc_{'_'.join(map(str, v))}_find name" for v, t in tags.items()]
    out.append("")
    # Segment bases, as a match on the GC major and the segment.
    gens = []
    for major, name in GC_BASES.items():
        present = defines(h[name])
        bv = constants(h[name], [n for n in GC_SEGMENTS if n in present])
        bases = [bv.get(n, 0) for n in GC_SEGMENTS]
        while bases and bases[-1] == 0:  # the headers write 0 for no base
            bases.pop()
        if 0 in bases:
            sys.exit(f"{name}: a GC segment without a base before one with")
        gens.append((major, name, bases))
    out += ["(* The base of a GC register segment in PM4's register space, by the GC",
            "   major version from which it holds, or [-1] for a segment with none. *)",
            "let gc_base major segment ="]
    for k, (major, name, bases) in enumerate(sorted(gens, reverse=True)):
        kw = "if" if k == 0 else "else if"
        out.append(f"  {kw} major >= {major} then (* {name} *)")
        out.append("    (match segment with")
        out += [f"    | {i} -> {ml_int(b)}" for i, b in enumerate(bases)]
        out.append("    | _ -> -1)")
    out += ["  else -1", ""]

    # What a dispatch writes, resolved per GC version: no dispatch looks a
    # register up by name or computes its address.
    out += ["(* The registers a dispatch sets on a GC version: each one's address in",
            "   PM4's register space; RESOURCE_LIMITS.WAVES_PER_SH as (lowest bit,",
            "   highest bit); and DISPATCH_INITIATOR's word for waves of 32 and of 64",
            "   lanes, with COMPUTE_SHADER_EN and FORCE_START_AT_000 set, and CS_W32_EN",
            "   for 32 lanes where the GC has it; and COMPUTE_TMPRING_SIZE, whose",
            "   fields a dispatch's scratch sets. *)",
            "type dispatch = {"]
    out += [f"  {f} : int;" for f, _ in DISPATCH_REGISTERS]
    out += ["  waves_per_sh : int * int;", "  initiator_wave32 : int;", "  initiator_wave64 : int;",
            "  tmpring : register;", "}", ""]

    def base(major, segment):
        b = [bs for m, _, bs in sorted(gens) if m <= major]
        return b[-1][segment] if b and segment < len(b[-1]) else None

    def word(fields, values):
        w = 0
        for f, v in values:
            lo, hi = next((lo, hi) for n, lo, hi in fields if n == f)
            w |= (v & ((1 << (hi - lo + 1)) - 1)) << lo
        return w
    dispatches = {}
    for ver, regs in gc_regs.items():
        names = [r for _, r in DISPATCH_REGISTERS] + ["regCOMPUTE_DISPATCH_INITIATOR"]
        if any(r not in regs or base(ver[0], regs[r][1]) is None for r in names):
            continue
        v = "_".join(map(str, ver))
        fs = [f"{f} = {ml_int(base(ver[0], regs[r][1]) + regs[r][0])}" for f, r in DISPATCH_REGISTERS]
        limits = dict((n, (lo, hi)) for n, lo, hi in regs["regCOMPUTE_RESOURCE_LIMITS"][2])["waves_per_sh"]
        init = regs["regCOMPUTE_DISPATCH_INITIATOR"][2]
        on = [("force_start_at_000", 1), ("compute_shader_en", 1)]
        w32 = on + ([("cs_w32_en", 1)] if any(n == "cs_w32_en" for n, _, _ in init) else [])
        fs += [f"waves_per_sh = ({limits[0]}, {limits[1]})", f"initiator_wave32 = {ml_int(word(init, w32))}",
               f"initiator_wave64 = {ml_int(word(init, on))}", f"tmpring = gc_{v}_regCOMPUTE_TMPRING_SIZE"]
        out.append(f"let gc_{v}_dispatch = {{ " + "; ".join(fs) + " }")
        dispatches[ver] = v
    out += ["", "(* The registers a dispatch sets on a GC version, if it has them all. *)",
            "let dispatch = function", "  | No_gc -> None"]
    out += [f"  | {t} -> " + (f"Some gc_{dispatches[ver]}_dispatch" if ver in dispatches else "None")
            for ver, t in tags.items()]
    out.append("")

    # PM4
    kfd = constants(h["kfd_pm4_headers_ai.h"], PM4_ENUMS)
    sv = constants(h["soc15d.h"], PM4_CONSTANTS + PM4_SOC15_ONLY + PM4_SOC15_CONSTANTS)
    nvv = constants(h["nvd.h"], PM4_CONSTANTS + PM4_NV_CONSTANTS)
    out += ["(* PM4, the same in soc15d.h (GFX9) and nvd.h (GFX10 on), and the release's",
            "   enumerations in kfd_pm4_headers_ai.h *)", ""]
    out += [f"let {ml_name(n)} = {ml_int(same(n, [sv[n], nvv[n]]))}" for n in PM4_CONSTANTS]
    out += [f"let {ml_name(n)} = {ml_int(kfd[n])}" for n in PM4_ENUMS]
    ss = shifts(h["soc15d.h"], sorted({g for g, _ in PM4_SHIFTS} | set(PM4_SOC15_SHIFTS)))
    ns = shifts(h["nvd.h"], sorted({n for _, n in PM4_SHIFTS} | set(PM4_NV_SHIFTS)))
    seen = set()
    for g, n in PM4_SHIFTS:
        v = same(g, [ss[g], ns[n]])
        if g not in seen:
            out.append(f"let {ml_name(g)} = {v}")
            seen.add(g)
    out += ["", "(* nvd.h alone, GFX10 on *)", ""]
    out += [f"let {ml_name(n)} = {ns[n]}" for n in PM4_NV_SHIFTS]
    out += [f"let {ml_name(n)} = {ml_int(nvv[n])}" for n in PM4_NV_CONSTANTS]
    out += ["", "(* soc15d.h alone, GFX9 *)", ""]
    out += [f"let {ml_name(n)} = {ss[n]}" for n in PM4_SOC15_SHIFTS]
    out += [f"let {ml_name(n)} = {ml_int(sv[n])}" for n in PM4_SOC15_CONSTANTS + PM4_SOC15_ONLY]
    out.append("")

    # WAIT_REG_MEM64: the 32-bit wait's control word, then 64-bit address,
    # reference and mask, then the poll interval, in every generation PAL lays
    # it out for.
    for name in PAL_HEADERS:
        size, fields = layout(h[name], WAIT_REG_MEM64)
        if size != 36 or any(fields.get(f) != at for f, at in WAIT_REG_MEM64_LAYOUT):
            sys.exit(f"{name}: {WAIT_REG_MEM64} is not laid out as the encoder writes it")
    if ss["WAIT_REG_MEM_FUNCTION"] != 0 or ss["WAIT_REG_MEM_MEM_SPACE"] != 4:
        sys.exit(f"{WAIT_REG_MEM64}'s control word differs from WAIT_REG_MEM's")

    # Events and thread traces
    def soc_enum(n):
        found = [evaluate(es[n], es) for es in (enumerators(h[s]) for s in SOCS) if n in es]
        if not found:
            sys.exit(f"no SOC enumeration defines {n}")
        return same(n, found)
    out += ["(* Events and thread trace values, the same in each SOC enumeration that",
            "   defines them *)", ""]
    out += [f"let {ml_name(n)} = {ml_int(soc_enum(n))}" for n in SOC_EVENTS + SOC_TRACE]
    out.append("")

    # Performance counters
    counters, _ = counter_defs(h["counter_defs.yaml"])
    out += ["(* The performance counters of each processor, by name: (name, block,",
            "   event). *)", "", "let counters = function"]
    for proc in COUNTER_PROCESSORS:
        cs = sorted({(n, b, e) for n, b, e, ps in counters if proc in ps})
        out.append(f"  | {json.dumps(proc)} ->")
        out.append("      [|")
        out += [f"        ({json.dumps(n)}, {json.dumps(b)}, {e});" for n, b, e in cs]
        out.append("      |]")
    out.append("  | _ -> [||]")
    out.append("")

    # GFX9's thread trace tokens
    masks = {}
    for m in MASK_DEFINE.finditer(h[gc_header((9, 4, 3), "sh_mask")]):
        if m.group(1).startswith("SQ_THREAD_TRACE_WORD_"):
            masks.setdefault(m.group(1)[len("SQ_THREAD_TRACE_WORD_"):], {})[m.group(2)] = int(m.group(3), 0)

    def halfwords(word):
        if word not in masks:
            sys.exit(f"gc_9_4_3_sh_mask.h has no SQ_THREAD_TRACE_WORD_{word}")
        return 1 if max(masks[word].values()).bit_length() <= 16 else 2
    vega10 = constants(h["vega10_enum.h"], [f"SQ_THREAD_TRACE_TOKEN_{t}" for t in GFX9_TOKENS])
    tokens = [(name, vega10[f"SQ_THREAD_TRACE_TOKEN_{name}"], sum(halfwords(w) for w in words))
              for name, words in GFX9_TOKENS.items()]
    if sorted(v for _, v, _ in tokens) != list(range(16)):
        sys.exit("GFX9's thread trace token types are not 0 to 15")
    out += ["(* GFX9's thread trace tokens, from vega10_enum.h and gc_9_4_3_sh_mask.h:",
            "   each type, and its 16-bit words. *)", "let sq_thread_trace_tokens = ["]
    out += [f"  ({v}, {n}); (* {name} *)" for name, v, n in sorted(tokens, key=lambda t: t[1])]
    out.append("]")
    out += [f"let sq_thread_trace_token_{name.lower()} = {v}" for name, v, _ in tokens
            if name in ("MISC", "TIMESTAMP", "WAVE_START", "WAVE_END")]
    for word, f in GFX9_TOKEN_FIELDS:
        mask = masks[word][f]
        out.append(f"let sq_thread_trace_word_{word.lower()}__{f.lower()} = "
                   f"({(mask & -mask).bit_length() - 1}, {mask.bit_length() - 1})")
    out.append("")

    # SDMA
    out += ["(* SDMA: (mask, shift) for a field *)", ""]
    want = SDMA_OPS + [f"{n}_{k}" for n in SDMA_FIELDS for k in ("mask", "shift")]
    svals = {}
    for ver, name in SDMA_PKT.items():
        ds = {n for n in defines(h[name])}
        svals[ver] = constants(h[name], [n for n in want if n in ds])
    for n in SDMA_OPS:
        out.append(f"let {ml_name(n)} = {ml_int(same(n, [v[n] for v in svals.values()]))}")
    # A literal, so that no build initialises it, -opaque ones included.
    trap = same("SDMA_OP_TRAP", [v["SDMA_OP_TRAP"] for v in svals.values()])
    out += ["", "(* The trap packet: SDMA_OP_TRAP, then its interrupt context, 0. *)",
            f"let sdma_trap = [ Packet.Dword {ml_int(trap)}; Packet.Dword 0 ]", ""]
    for n in SDMA_FIELDS:
        have = [(v[n + "_mask"], v[n + "_shift"]) for v in svals.values() if n + "_mask" in v]
        if len(have) != len(svals) and n not in SDMA_OPTIONAL:
            sys.exit(f"an SDMA header has no {n}")
        mask, sh = same(n, have)
        out.append(f"let {ml_name(n)} = ({ml_int(mask)}, {sh})")
    first = min(ver for ver, v in svals.items() if "SDMA_PKT_FENCE_HEADER_mtype_mask" in v)
    out += [f"let sdma_fence_mtype_from = {ml_version(first)}", ""]

    # AQL
    out += ["(* AQL: hsa.h's constants, and its kernel dispatch packet as (byte offset,", "   bytes) *)", ""]
    hv = constants(h["hsa.h"], HSA_CONSTANTS)
    out += [f"let {ml_name(n)} = {ml_int(hv[n])}" for n in HSA_CONSTANTS]
    out.append("")
    size, fields = layout(h["hsa.h"], DISPATCH)
    out += ["module Dispatch = struct", f"  let sizeof = {size}"]
    out += [f"  let {f} = {ml_field(fields[f])}" for f in DISPATCH_FIELDS]
    out += ["end", ""]

    # Scratch buffer descriptors
    rv = constants(h["registers.h"], SQ_CONSTANTS)
    out += ["(* The scratch buffer descriptor's words 1 and 3, by GC major: each field", "   as (bit offset, bits). *)", ""]
    out += ["type sq_buf_rsrc = {"]
    out += [f"  {ml_name(f) if f != 'TYPE' else 'type_'} : int * int;" for f in SQ_WORD1 + SQ_WORD3]
    out += [f"  {ml_name(f)} : (int * int) option;" for f in SQ_WORD3_OPTIONAL]
    out += ["}", "", "let sq_buf_rsrc = ["]
    for major, (w1, w3) in SQ_BUF_RSRC.items():
        def bits(union):
            _, fields = layout(h["registers.h"], union)
            return {k.split("__")[-1]: v for k, v in fields.items() if v[0] == "bits"}
        b1, b3 = bits(w1), bits(w3)
        fs = [f"{ml_name(f)} = ({b1[f][1]}, {b1[f][2]})" for f in SQ_WORD1]
        fs += [f"{ml_name(f) if f != 'TYPE' else 'type_'} = ({b3[f][1]}, {b3[f][2]})" for f in SQ_WORD3]
        fs += [f"{ml_name(f)} = " + (f"Some ({b3[f][1]}, {b3[f][2]})" if f in b3 else "None")
               for f in SQ_WORD3_OPTIONAL]
        out.append(f"  ({major}, {{ " + "; ".join(fs) + " });")
    out += ["]", ""]
    out += [f"let {ml_name(n)} = {ml_int(rv[n])}" for n in SQ_CONSTANTS]
    out.append("")

    # Code objects
    kd = h["AMDHSAKernelDescriptor.h"]
    size, fields = layout(kd, KD)
    offsets = constants(kd, [f.upper() + "_OFFSET" for f in KD_FIELDS])
    out += ["(* The kernel descriptor: each field is (byte offset, bytes). *)", "module Kernel_descriptor = struct",
            f"  let sizeof = {size}"]
    for f in KD_FIELDS:
        if fields[f][0] != offsets[f.upper() + "_OFFSET"]:
            sys.exit(f"{KD}.{f} is not at its {f.upper()}_OFFSET")
        out.append(f"  let {f} = {ml_field(fields[f])}")
    out += ["end", "", "(* Its code properties. *)"]
    props = {m.group(1): ((1 << int(m.group(3))) - 1) << int(m.group(2)) for m in KD_PROPERTY.finditer(kd)}
    out += [f"let amd_kernel_code_properties_{ml_name(p)} = {ml_int(props[p])}" for p in KD_PROPERTIES]
    elf, machs, generic = processors(h["ELF.h"], h["AMDGPUUsage.rst"])
    out += ["", "(* The ELF header of a code object, and its notes' types. *)"]
    out += [f"let {ml_name(c)} = {ml_int(v)}" for c, v in elf.items()]
    out += ["", "(* LLVM's AMDGCN processors, by their EF_AMDGPU_MACH value. *)", "let processors = ["]
    out += [f"  ({ml_int(v)}, {json.dumps(n)});" for v, n in machs]
    out += ["]", "", "(* The generic processors, and the processors that run their code objects. *)", "let generic = ["]
    out += [f"  ({json.dumps(n)}, [ " + "; ".join(json.dumps(m) for m in ms) + " ]);" for n, ms in generic]
    out.append("]")
    return "\n".join(out) + "\n"

# Command line


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--excerpt", action="store_true", help="make the excerpts from the pinned headers")
    p.add_argument("--check", action="store_true", help="fail if a committed file differs")
    p.add_argument("--pin", action="store_true", help="record the digests of headers not yet pinned")
    p.add_argument("--cache", type=pathlib.Path, default=pathlib.Path.home() / ".cache/raven/amd-gen")
    a = p.parse_args()
    if a.excerpt:
        pins = json.loads(PINS.read_text()) if PINS.exists() else {}
        a.cache.mkdir(parents=True, exist_ok=True)
        files = {HEADERS / n: excerpt(n, fetch(url, a.cache, pins, a.pin)) for n, url in SOURCES.items()}
        if a.pin:
            PINS.write_text(json.dumps({u: d for u, d in pins.items() if u in SOURCES.values()}, indent=1,
                                       sort_keys=True) + "\n")
    else:
        h = {n: (HEADERS / n).read_text(encoding="latin-1") for n in SOURCES}
        files = {OUT: generate(h)}
    for path, text in files.items():
        if a.check:
            if not path.exists() or path.read_text(encoding="latin-1") != text:
                sys.exit(f"{path.name} differs from what gen.py makes")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="latin-1")
    if a.check:
        print("up to date")


if __name__ == "__main__":
    main()
