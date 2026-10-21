# /// script
# requires-python = ">=3.10"
# ///
"""Generates the tables of AMD's GPUs from AMD's, LLVM's and PAL's sources:

- lib/amd/abi/defs.ml, the tables of rig_amd_abi: the GC registers of each
  GC version and the bases of their segments, the constants of PM4, SDMA
  and AQL packets and of thread traces, the layouts of AQL's dispatch
  packet, of the scratch buffer descriptor and of the kernel descriptor,
  and LLVM's AMDGPU processors;
- lib/amd/pci/defs.ml, the tables of the boot over PCI: the layouts of the
  discovery table and its blocks' names and hardware IDs, the registers of
  each block but GC (whose are rig_amd_abi's) at each version with headers,
  what the boot reads before the discovery table and of a virtual function,
  the page-table entries, the doorbells and queue descriptors, the security
  processor's commands and ring, the power manager's messages, the
  interrupt sources, the layouts of firmware images' headers and the types
  the security processor loads them as, and the pinned firmware images;
- headers/kernel/{kfd_ioctl,amdgpu_drm,drm}.h, the excerpts of Linux's
  uapi headers that rig_amd_amdgpu_stubs.c compiles against, never the
  build machine's headers, which may predate a definition it uses: the
  requests of KFD (/dev/kfd) and of amdgpu's render node the library
  makes, the structures of their parameters and the flags they take.

Run from the worktree root:

  uv run packages/rig/lib/amd/gen/gen.py
  uv run packages/rig/lib/amd/gen/gen.py --check
  uv run packages/rig/lib/amd/gen/gen.py --excerpt [--check]

The inputs are excerpts, in headers/: headers/ORIGIN/NAME is cut from the
file NAME of ORIGINS[ORIGIN], at the commit or tag in its URL, and SOURCES
names that file. Each is the file's licence notice, the URL it is cut from,
and the lines this script reads, verbatim and in the file's order (a
#define, an enumerator, an enumeration, a struct's definition, a table of a
document). The uapi excerpts hold whole definitions instead, with the
definitions they depend on, as a 64-bit Linux build compiles them.
headers/firmware.tsv lists the pinned firmware images of linux-firmware at
FIRMWARE_COMMIT: each image's path, BLAKE2b-256 digest and URL, exactly the
images of IMAGES; the page of rig firmware, in packages/rig/bin/help.ml,
names FIRMWARE_COMMIT too.

--excerpt makes the excerpts and firmware.tsv from the upstream files, each
pinned in pins.json by URL and SHA-256 and checked against its pin; it
fails on a header whose notice is not MIT's, NCSA's or LLVM's. Downloads
are kept in --cache, and --pin records the digests of files not yet
pinned. Generating reads the excerpts and firmware.tsv alone, offline.
--check generates into memory and fails if a committed file differs. Each
defs.ml carries the notices of the excerpts it reads.

Text is read and written as latin-1, one character per byte, so every byte
of a header round-trips into its excerpt as upstream wrote it. The headers
are ASCII C; AMDGPUUsage.rst holds UTF-8 outside the lines kept, which
latin-1 reads without failing.

Where two headers define a value a library takes once (PM4's in soc15d.h
and nvd.h, a thread trace value in each SOC enumeration, an SDMA field in
each version's header), the script fails unless they agree. Every table is
a literal, which the compiler lays out as static data: nothing is built
when a program starts.
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
HEADERS = HERE / "headers"
PINS = HERE / "pins.json"
LIB = HERE.parent
ABI_OUT = LIB / "abi/defs.ml"
PCI_OUT = LIB / "pci/defs.ml"

# Sources

ORIGINS = {
    "kernel": "https://raw.githubusercontent.com/ROCm/ROCK-Kernel-Driver/33970e1351f5e511029602454979f3de7e22260f/",
    "rocm": "https://raw.githubusercontent.com/ROCm/rocm-systems/cccc350dc620e61ae2554978b62ab3532dc10bd9/projects/",
    "llvm": "https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-20.1.0/llvm/",
    "pal": "https://raw.githubusercontent.com/GPUOpen-Drivers/pal/c5e800072a32f68b6ccc4422936d96167c6e0728/"
           "src/core/hw/gfxip/",
}
AMD = "drivers/gpu/drm/amd/"


def sources(origin, paths):
    """{excerpt: URL} of the files [paths] of [origin], each excerpt named by
    its origin and the file's name."""
    return {f"{origin}/{p.rsplit('/', 1)[-1]}": ORIGINS[origin] + p for p in paths}


def reg_header(block, ver, kind):
    """The excerpt of [block]'s registers at [ver]: MP 11.0's is unversioned."""
    stem = "mp_11_0" if (block, ver) == ("mp", (11, 0, 0)) else f"{block}_{'_'.join(map(str, ver))}"
    return f"kernel/{stem}_{kind}.h"


def reg_sources(block, versions):
    d = {"osssys": "oss"}.get(block, block)
    return {reg_header(block, v, k): ORIGINS["kernel"] + f"{AMD}include/asic_reg/{d}/" + reg_header(block, v, k)[7:]
            for v in versions for k in ("offset", "sh_mask")}


def base(name):
    """The file's name of the excerpt [name]."""
    return name.split("/", 1)[1]


# rig_amd_abi

# The GC versions whose register headers exist.
GC_VERSIONS = [(9, 4, 3), (11, 0, 0), (11, 0, 3), (11, 5, 0), (12, 0, 0)]

ABI = {
    **reg_sources("gc", GC_VERSIONS),  # the Linux kernel's GC registers
    **sources("kernel", [AMD + "include/vega20_ip_offset.h",  # GC segment bases, GFX9
                         AMD + "include/sienna_cichlid_ip_offset.h",  # GFX10 on
                         AMD + "amdgpu/soc15d.h",  # PM4, GFX9
                         AMD + "amdgpu/nvd.h",  # PM4, GFX10 on
                         AMD + "amdkfd/kfd_pm4_headers_ai.h",  # RELEASE_MEM's enumerations
                         AMD + "amdgpu/vega10_sdma_pkt_open.h",  # SDMA 4
                         AMD + "amdgpu/navi10_sdma_pkt_open.h",  # SDMA 5
                         AMD + "amdgpu/sdma_v6_0_0_pkt_open.h"]),  # SDMA 6
    **sources("rocm", ["aqlprofile/linux/vega10_enum.h",  # SOC enumerations, GFX9
                       "aqlprofile/linux/soc21_enum.h",  # GFX11
                       "aqlprofile/linux/soc24_enum.h",  # GFX12
                       "rocr-runtime/runtime/hsa-runtime/inc/hsa.h",  # AQL
                       "rocr-runtime/runtime/hsa-runtime/core/inc/registers.h",  # buffer descriptors
                       # performance counters
                       "rocprofiler-compute/src/rocprof_compute_soc/profile_configs/counter_defs.yaml"]),
    **sources("llvm", ["include/llvm/Support/AMDHSAKernelDescriptor.h", "include/llvm/BinaryFormat/ELF.h",
                       "docs/AMDGPUUsage.rst"]),  # processors
    **sources("pal", ["gfx9/chip/gfx9_plus_merged_f32_mec_pm4_packets.h",
                      "gfx12/chip/gfx12_merged_f32_mec_pm4_packets.h"]),
}

# counter_defs.yaml carries no notice: its project's licence, the MIT licence
# of rocprofiler-compute/LICENSE.md, covers it.
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
    r"regSDMA0_QUEUE[01]_(RB_CNTL|RB_BASE(_HI)?|RB_RPTR(_HI)?|RB_WPTR(_HI)?|RB_RPTR_ADDR_(LO|HI)|"
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
GC_BASES = {9: "kernel/vega20_ip_offset.h", 10: "kernel/sienna_cichlid_ip_offset.h"}

# The registers a dispatch sets, by the name of their address in the
# dispatch's record.
DISPATCH_REGISTERS = [("pgm_lo", "regCOMPUTE_PGM_LO"), ("pgm_rsrc1", "regCOMPUTE_PGM_RSRC1"),
                      ("pgm_rsrc3", "regCOMPUTE_PGM_RSRC3"), ("tmpring_size", "regCOMPUTE_TMPRING_SIZE"),
                      ("scratch_base_lo", "regCOMPUTE_DISPATCH_SCRATCH_BASE_LO"),
                      ("restart_x", "regCOMPUTE_RESTART_X"), ("user_data_0", "regCOMPUTE_USER_DATA_0"),
                      ("resource_limits", "regCOMPUTE_RESOURCE_LIMITS"), ("start_x", "regCOMPUTE_START_X")]

# PM4: the same in soc15d.h (GFX9) and nvd.h (GFX10 on).
SOC15D, NVD, KFD_PM4 = "kernel/soc15d.h", "kernel/nvd.h", "kernel/kfd_pm4_headers_ai.h"
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
PAL_HEADERS = ["pal/gfx9_plus_merged_f32_mec_pm4_packets.h", "pal/gfx12_merged_f32_mec_pm4_packets.h"]

# The events and thread trace values of the SOC enumerations, the same in each
# that defines them.
SOCS = ["rocm/vega10_enum.h", "rocm/soc21_enum.h", "rocm/soc24_enum.h"]
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
SDMA_PKT = {(4, 0, 0): "kernel/vega10_sdma_pkt_open.h", (5, 0, 0): "kernel/navi10_sdma_pkt_open.h",
            (6, 0, 0): "kernel/sdma_v6_0_0_pkt_open.h"}
SDMA_OPS = ["SDMA_OP_COPY", "SDMA_OP_FENCE", "SDMA_OP_TRAP", "SDMA_OP_POLL_REGMEM", "SDMA_OP_TIMESTAMP",
            "SDMA_OP_CONST_FILL", "SDMA_SUBOP_COPY_LINEAR", "SDMA_SUBOP_TIMESTAMP_GET_GLOBAL"]
SDMA_FIELDS = ["SDMA_PKT_COPY_LINEAR_HEADER_sub_op", "SDMA_PKT_POLL_REGMEM_HEADER_func",
               "SDMA_PKT_POLL_REGMEM_HEADER_mem_poll", "SDMA_PKT_POLL_REGMEM_DW5_interval",
               "SDMA_PKT_POLL_REGMEM_DW5_retry_count", "SDMA_PKT_FENCE_HEADER_mtype",
               "SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_sub_op", "SDMA_PKT_CONSTANT_FILL_HEADER_fillsize"]
SDMA_OPTIONAL = ["SDMA_PKT_FENCE_HEADER_mtype"]  # GFX9 engines take no memory type

# AQL, from hsa.h.
HSA = "rocm/hsa.h"
HSA_CONSTANTS = ["HSA_PACKET_HEADER_TYPE", "HSA_PACKET_HEADER_BARRIER", "HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE",
                 "HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE", "HSA_FENCE_SCOPE_SYSTEM", "HSA_PACKET_TYPE_VENDOR_SPECIFIC",
                 "HSA_PACKET_TYPE_KERNEL_DISPATCH", "HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS"]
DISPATCH = "hsa_kernel_dispatch_packet_s"
DISPATCH_FIELDS = ["header", "setup", "workgroup_size_x", "workgroup_size_y", "workgroup_size_z", "grid_size_x",
                   "grid_size_y", "grid_size_z", "private_segment_size", "group_segment_size", "kernel_object",
                   "kernarg_address"]

# The scratch buffer descriptor, by GC major: the unions of its words 1 and 3
# in registers.h, and the values its fields take.
REGISTERS_H = "rocm/registers.h"
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
KD_H, ELF_H, USAGE = "llvm/AMDHSAKernelDescriptor.h", "llvm/ELF.h", "llvm/AMDGPUUsage.rst"
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
COUNTERS = "rocm/counter_defs.yaml"
COUNTER_BLOCKS = ["GRBM", "GL2C", "TCC", "SQ"]
COUNTER_PROCESSORS = ["gfx942", "gfx950", "gfx11", "gfx12"]

# rig_amd_pci

PCI = {
    **sources("kernel", [AMD + "include/discovery.h",  # the discovery table
                         AMD + "amdgpu/amdgpu_discovery.h",  # where it lies
                         AMD + "include/soc15_hw_ip.h",  # the blocks' hardware IDs
                         AMD + "amdgpu/amdgpu_ucode.h",  # firmware images' headers
                         AMD + "amdgpu/psp_gfx_if.h",  # the types the PSP loads images as
                         AMD + "amdgpu/amdgpu_vm.h",  # page-table entries
                         AMD + "amdgpu/amdgpu_psp.h",  # the security processor's bootloader and ring
                         AMD + "amdgpu/amdgpu_doorbell.h",  # doorbell indices
                         AMD + "amdgpu/amdgpu_discovery.c",  # the registers read before discovery
                         AMD + "amdgpu/amdgpu_virt.h",  # virtual functions
                         AMD + "amdgpu/mxgpu_nv.h",  # their mailbox
                         AMD + "include/v9_structs.h",  # GFX9's queue descriptors
                         AMD + "include/v11_structs.h",  # GFX11's
                         AMD + "include/v12_structs.h",  # GFX12's
                         AMD + "include/vega10_enum.h",  # memory types, GFX9
                         AMD + "include/soc21_enum.h",  # GFX11
                         AMD + "include/soc24_enum.h",  # GFX12
                         AMD + "include/soc15_ih_clientid.h"]),  # interrupt clients
    **sources("kernel", [AMD + f"include/ivsrcid/{d}/irqsrcs_{b}.h"  # interrupt sources
                         for d, b in (("gfx", "gfx_9_0"), ("gfx", "gfx_11_0_0"), ("gfx", "gfx_12_0_0"),
                                      ("sdma0", "sdma0_4_0"), ("sdma0", "sdma0_5_0"))]),
}

# The power manager's messages and clocks, by the MP1 versions amdgpu drives
# with each message header (amdgpu_smu.c's smu_set_funcs): the header of the
# messages, and the header of the clocks.
SMU_TABLES = [
    ([(13, 0, 0), (13, 0, 10)], "smu_v13_0_0_ppsmc.h", "smu13_driver_if_v13_0_0.h", "smu13_driver_if_v13_0_0.h"),
    ([(13, 0, 7)], "smu_v13_0_7_ppsmc.h", "smu13_driver_if_v13_0_7.h", "smu13_driver_if_v13_0_7.h"),
    ([(13, 0, 6), (13, 0, 14)], "smu_v13_0_6_ppsmc.h", "smu13_driver_if_v13_0_6.h", "smu_v13_0_6_pmfw.h"),
    ([(13, 0, 12)], "smu_v13_0_12_ppsmc.h", "smu13_driver_if_v13_0_6.h", "smu_v13_0_12_pmfw.h"),
    ([(14, 0, 2), (14, 0, 3)], "smu_v14_0_2_ppsmc.h", "smu14_driver_if_v14_0.h", "smu14_driver_if_v14_0.h"),
]
SMU_TABLES = [(vs, *(f"kernel/{h}" for h in hs)) for vs, *hs in SMU_TABLES]
for _, *_hs in SMU_TABLES:
    PCI.update(sources("kernel", [AMD + "pm/swsmu/inc/pmfw_if/" + base(h) for h in _hs]))
SMU_MESSAGES = ["PPSMC_MSG_" + m for m in (
    "SetDriverDramAddrHigh", "SetDriverDramAddrLow", "EnableAllSmuFeatures", "GetSmuVersion", "GfxDriverReset",
    "Mode1Reset", "GetDpmFreqByIndex", "SetSoftMinByFreq", "SetSoftMaxByFreq", "QueryValidMcaCount",
    "McaBankDumpDW", "QueryValidMcaCeCount", "McaBankCeDumpDW", "GetRunningSmuFeaturesLow",
    "GetRunningSmuFeaturesHigh", "GetEnabledSmuFeaturesLow", "GetEnabledSmuFeaturesHigh")]
SMU_CLOCKS = ["PPCLK_UCLK", "PPCLK_FCLK", "PPCLK_SOCCLK", "PPCLK_GFXCLK"]
# The feature bit of each clock's DPM, a define (FEATURE_DPM_X_BIT) or an
# enumerator (FEATURE_DPM_X), named FEATURE_DPM_X.
SMU_FEATURES = ["FEATURE_DPM_UCLK", "FEATURE_DPM_FCLK", "FEATURE_DPM_SOCCLK", "FEATURE_DPM_GFXCLK"]
SMU_FEATURE = re.compile(r"\b(FEATURE_DPM_(?:UCLK|FCLK|SOCCLK|GFXCLK))(?:_BIT)?\b\s*=?\s*(0x[0-9a-fA-F]+|\d+)")

# Interrupts: the clients' enumerations, and the source headers of the blocks
# whose interrupts the library reads.
IH_CLIENTID = "kernel/soc15_ih_clientid.h"
IH_ENUMS = ["soc15_ih_clientid", "soc21_ih_clientid"]
IH_CLIENTS = ["SOC15_IH_CLIENTID_GRBM_CP", "SOC15_IH_CLIENTID_UTCL2"] + [
    f"SOC15_IH_CLIENTID_SE{i}SH" for i in range(4)] + [f"SOC15_IH_CLIENTID_SDMA{i}" for i in range(8)] + [
    "SOC21_IH_CLIENTID_GRBM_CP", "SOC21_IH_CLIENTID_GFX"]
IH_SOURCES = ["kernel/irqsrcs_gfx_9_0.h", "kernel/irqsrcs_gfx_11_0_0.h", "kernel/irqsrcs_gfx_12_0_0.h",
              "kernel/irqsrcs_sdma0_4_0.h", "kernel/irqsrcs_sdma0_5_0.h"]
SRCID = re.compile(r"^\s*#\s*define\s+(\w+?)__SRCID__(\w+)\s+(0x[0-9a-fA-F]+|\d+)\b", re.M)

# Page-table entries: their bits, the first bit of their fields, the levels
# of the tables, and the uncached memory type of each generation.
PTE_BITS = ["AMDGPU_PTE_VALID", "AMDGPU_PTE_SYSTEM", "AMDGPU_PTE_SNOOPED", "AMDGPU_PTE_EXECUTABLE",
            "AMDGPU_PTE_READABLE", "AMDGPU_PTE_WRITEABLE", "AMDGPU_PTE_TF", "AMDGPU_PDE_PTE", "AMDGPU_PDE_PTE_GFX12",
            "AMDGPU_PTE_IS_PTE"]
PTE_SHIFTS = ["AMDGPU_PTE_FRAG", "AMDGPU_PDE_BFS", "AMDGPU_PTE_MTYPE_VG10_SHIFT", "AMDGPU_PTE_MTYPE_NV10_SHIFT",
              "AMDGPU_PTE_MTYPE_GFX12_SHIFT"]
VM_LEVELS = ["AMDGPU_VM_PDB2", "AMDGPU_VM_PDB1", "AMDGPU_VM_PDB0", "AMDGPU_VM_PTB"]
MTYPES = {"kernel/vega10_enum.h": "soc15", "kernel/soc21_enum.h": "soc21", "kernel/soc24_enum.h": "soc24"}

# The register headers of each block but GC, by the stem of their names, at
# each version with headers the GPUs the library boots program; and GC's,
# rig_amd_abi's, for the few registers below.
REG_FILES = {
    "mmhub": [(1, 8, 0), (3, 0, 0), (3, 0, 1), (3, 0, 2), (4, 1, 0)],
    "nbio": [(4, 3, 0), (7, 2, 0), (7, 7, 0), (7, 9, 0)],
    "nbif": [(6, 3, 1)],
    "mp": [(11, 0, 0), (13, 0, 0), (14, 0, 2)],
    "hdp": [(4, 4, 2), (6, 0, 0), (7, 0, 0)],
    "osssys": [(4, 4, 2), (6, 0, 0), (7, 0, 0)],
    "sdma": [(4, 4, 2)],
    "gc": GC_VERSIONS,
}
for _b, _vs in REG_FILES.items():
    PCI.update(reg_sources(_b, _vs))

# The registers the library reads and writes, by block: names that match.
REG_INVENTORY = {
    "mmhub": [
        r"regMMVM_CONTEXT0_(CNTL|PAGE_TABLE_(START|END|BASE)_ADDR_(LO32|HI32))",
        r"regMMVM_INVALIDATE_ENG17_(REQ|ACK|SEM)",
        r"regMMVM_INVALIDATE_ENG\d+_ADDR_RANGE_(LO32|HI32)",
        r"regMMVM_L2_(CNTL[2-5]?|PROTECTION_FAULT_(CNTL2?|STATUS(_LO32)?|DEFAULT_ADDR_(LO32|HI32)|ADDR_(LO32|HI32))|"
        r"CONTEXT1_IDENTITY_APERTURE_(LOW|HIGH)_ADDR_(LO32|HI32)|CONTEXT_IDENTITY_PHYSICAL_OFFSET_(LO32|HI32)|"
        r"BANK_SELECT_RESERVED_CID2)",
        r"regMMMC_VM_(AGP_(BASE|BOT|TOP)|SYSTEM_APERTURE_(LOW|HIGH)_ADDR|SYSTEM_APERTURE_DEFAULT_ADDR_(LSB|MSB)|"
        r"MX_L1_TLB_CNTL|FB_LOCATION_(BASE|TOP)|FB_OFFSET|XGMI_LFB_(CNTL|SIZE))",
        r"regMM_ATC_L2_MISC_CG",
    ],
    "nbio": [
        r"regBIF_BX_PF0_RSMU_(INDEX|DATA)",
        r"regBIF_BX0_(PCIE_INDEX2(_HI)?|PCIE_DATA2|REMAP_HDP_(MEM|REG)_FLUSH_CNTL|BIF_DOORBELL_INT_CNTL|INTERRUPT_CNTL2?)",
        r"regBIF_BX_DEV0_EPF0_VF0_HDP_MEM_COHERENCY_FLUSH_CNTL",
        r"regBIFC_(GFX_INT_MONITOR_MASK|DOORBELL_ACCESS_EN_PF)", r"regXCC_DOORBELL_FENCE",
        r"regDOORBELL0_CTRL_ENTRY_\d+", r"reg(GDC_S2A0_S2A|S2A)_DOORBELL_ENTRY_\d+_CTRL",
        r"regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN", r"regRCC_DEV0_EPF2_STRAP2",
    ],
    "mp": [r"reg(MP0|MPASP)_SMN_C2PMSG_\d+", r"mmMP1_SMN_C2PMSG_\d+"],
    "hdp": [r"regHDP_MEM_POWER_CTRL"],
    "osssys": [
        r"regIH_(RB_BASE|RB_BASE_HI|RB_CNTL|RB_RPTR|RB_WPTR|DOORBELL_RPTR)(_RING1)?",
        r"regIH_(RB_WPTR_ADDR_(LO|HI)|STORM_CLIENT_LIST_CNTL|INT_FLOOD_CNTL|MSI_STORM_CTRL)",
    ],
    "sdma": [r"regSDMA_GFX_(RB_CNTL|RB_BASE(_HI)?|RB_RPTR(_HI)?|RB_WPTR(_HI)?|RB_RPTR_ADDR_(LO|HI)|"
             r"RB_WPTR_POLL_ADDR_(LO|HI)|DOORBELL|DOORBELL_OFFSET|MINOR_PTR_UPDATE|IB_CNTL)", r"regSDMA_CNTL"],
}
REG_INVENTORY["nbif"] = REG_INVENTORY["nbio"]
# GC's registers are rig_amd_abi's, but for those that say which compute
# units or work-group processors a shader array has fused off, and the memory
# controller's offset GC 9 keeps (MMHUB's from GC 10).
REG_INVENTORY["gc"] = [r"reg(CC_GC|GC_USER)_SHADER_ARRAY_CONFIG", r"regMC_VM_FB_OFFSET"]
# The hub whose name a block's VM registers take, as regMMVM_L2_CNTL for
# MMHUB's regVM_L2_CNTL.
HUBS = {"mmhub": "MM"}

# Compute queues: the doorbells of the KIQ and the first MEC and SDMA rings,
# the memory queue descriptors (MQD) of each GC generation, and the shader
# memory modes of each SOC.
DOORBELLS = ["AMDGPU_NAVI10_DOORBELL_KIQ", "AMDGPU_NAVI10_DOORBELL_MEC_RING0", "AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE0"]
MQDS = {"kernel/v9_structs.h": ("v9_mqd", 9), "kernel/v11_structs.h": ("v11_compute_mqd", 11),
        "kernel/v12_structs.h": ("v12_compute_mqd", 12)}
MQD_FIELDS = ["header", "cp_mqd_base_addr_lo", "cp_mqd_base_addr_hi", "cp_hqd_pipe_priority", "cp_hqd_queue_priority",
              "cp_hqd_quantum", "cp_hqd_persistent_state", "cp_hqd_pq_base_lo", "cp_hqd_pq_base_hi",
              "cp_hqd_pq_rptr_report_addr_lo", "cp_hqd_pq_rptr_report_addr_hi", "cp_hqd_pq_wptr_poll_addr_lo",
              "cp_hqd_pq_wptr_poll_addr_hi", "cp_hqd_pq_doorbell_control", "cp_hqd_pq_control", "cp_hqd_ib_control",
              "cp_hqd_hq_status0", "cp_mqd_control", "cp_hqd_vmid", "cp_hqd_aql_control", "cp_hqd_eop_base_addr_lo",
              "cp_hqd_eop_base_addr_hi", "cp_hqd_eop_control"]
MQD_OPTIONAL = ["compute_tg_chunk_size", "compute_current_logic_xcc_id", "cp_mqd_stride_size"]
SH_MEM = ["SH_MEM_ADDRESS_MODE_64", "SH_MEM_ALIGNMENT_MODE_UNALIGNED"]

# Before its discovery table is read: the registers that hold the GPU's
# memory size and its firmware's readiness, and the memory index window a
# small BAR reads the table through; a virtual function's identifier and
# mailbox.
EARLY = ["mmRCC_CONFIG_MEMSIZE", "mmMP0_SMN_C2PMSG_33", "mmMM_INDEX", "mmMM_INDEX_HI", "mmMM_DATA"]
VF = ["mmRCC_IOV_FUNC_IDENTIFIER"]
MAILBOX = ["NV_MAILBOX_POLL_ACK_TIMEDOUT", "NV_MAILBOX_POLL_MSG_TIMEDOUT", "mmMAILBOX_CONTROL",
           "NV_MAIBOX_CONTROL_TRN_OFFSET_BYTE", "mmMAILBOX_MSGBUF_TRN_DW0", "mmMAILBOX_MSGBUF_RCV_DW0"]
MAILBOX_ENUMS = ["IDH_REQ_GPU_INIT_ACCESS", "IDH_REQ_GPU_FINI_ACCESS", "IDH_READY_TO_ACCESS_GPU"]

FIRMWARE_COMMIT = "0a6871b19abf5d6e024b5d208b101ae53e7fa0de"
FIRMWARE_URL = f"https://gitlab.com/kernel-firmware/linux-firmware/-/raw/{FIRMWARE_COMMIT}/"
FIRMWARE = "firmware.tsv"

# The images of the GC versions the library boots (9.4.3, 9.5.0, 11.0.0,
# 11.0.2, 12.0.0, 12.0.1: their MEC, RLC and, from 11, IMU; from 12 the
# PFP's and ME's too, whose RS64 engines start from them), and every PSP, SMU
# and SDMA image of the discrete
# GPUs of those generations: which of them a GPU names is its discovery
# table's fact.
GC_IMAGES = [f"gc_{v}_{e}.bin" for v in ("9_4_3", "9_5_0") for e in ("mec", "rlc")] + [
    f"gc_{v}_{e}.bin" for v in ("11_0_0", "11_0_2") for e in ("imu", "mec", "rlc")] + [
    f"gc_{v}_{e}.bin" for v in ("12_0_0", "12_0_1") for e in ("imu", "me", "mec", "pfp", "rlc")]
OTHER_IMAGES = [f"psp_{v}_sos.bin" for v in ("13_0_0", "13_0_6", "13_0_7", "13_0_10", "13_0_12", "13_0_14",
                                               "13_0_15", "14_0_2", "14_0_3")] + [
    f"smu_{v}.bin" for v in ("13_0_0", "13_0_6", "13_0_7", "13_0_10", "13_0_14", "14_0_2", "14_0_3")] + [
    f"sdma_{v}.bin" for v in ("4_4_2", "4_4_4", "4_4_5", "6_0_0", "6_0_1", "6_0_2", "6_0_3", "7_0_0", "7_0_1")]
IMAGES = sorted(GC_IMAGES + OTHER_IMAGES)

DISCOVERY, UCODE, PSP_GFX, PSP_H = "kernel/discovery.h", "kernel/amdgpu_ucode.h", "kernel/psp_gfx_if.h", "kernel/amdgpu_psp.h"
DISCOVERY_CONSTANTS = ["BINARY_SIGNATURE", "DISCOVERY_TABLE_SIGNATURE", "GC_TABLE_ID", "HARVEST_TABLE_SIGNATURE"]
DISCOVERY_TABLES = ["IP_DISCOVERY", "GC", "HARVEST_INFO"]
DISCOVERY_PLACE = ["DISCOVERY_TMR_OFFSET", "DISCOVERY_TMR_SIZE"]
# Each struct and the fields kept of it; [] keeps only its size.
DISCOVERY_STRUCTS = {
    "binary_header": ["binary_signature", "binary_checksum", "binary_size", "table_list"],
    "table_info": ["offset", "checksum"],
    "ip_discovery_header": ["signature", "version", "size", "num_dies", "die_info", "base_addr_64_bit"],
    "die_info": ["die_offset"],
    "die_header": ["die_id", "num_ips"],
    "ip_v4": ["hw_id", "instance_number", "num_base_address", "major", "minor", "revision"],
    "gpu_info_header": ["table_id", "version_major", "size"],
    "gc_info_v1_0": ["gc_num_se", "gc_num_wgp0_per_sa", "gc_num_wgp1_per_sa", "gc_max_waves_per_simd",
                     "gc_max_scratch_slots_per_cu", "gc_lds_size", "gc_num_sa_per_se"],
    "gc_info_v2_0": ["gc_num_se", "gc_num_cu_per_sh", "gc_num_sh_per_se", "gc_max_waves_per_simd",
                     "gc_max_scratch_slots_per_cu", "gc_lds_size"],
    "harvest_info_header": ["signature"],
    "harvest_info": ["hw_id", "number_instance"],
    "harvest_table": ["list"],
}
# Firmware headers: each struct and the fields kept of it.
UCODE_STRUCTS = {
    "common_firmware_header": ["header_version_major", "header_version_minor", "ucode_version", "ucode_size_bytes",
                               "ucode_array_offset_bytes"],
    "psp_fw_bin_desc": ["fw_type", "offset_bytes", "size_bytes"],
    "psp_firmware_header_v2_0": ["psp_fw_bin_count", "psp_fw_bin"],
    "psp_firmware_header_v2_1": ["psp_fw_bin_count", "psp_aux_fw_bin_index", "psp_fw_bin"],
    "smc_soft_pptable_entry": ["id", "ppt_offset_bytes", "ppt_size_bytes"],
    "smc_firmware_header_v2_1": ["pptable_count", "pptable_entry_offset"],
    "sdma_firmware_header_v2_0": ["ctx_ucode_size_bytes", "ctl_ucode_offset", "ctl_ucode_size_bytes"],
    "sdma_firmware_header_v3_0": ["ucode_size_bytes"],
    "gfx_firmware_header_v1_0": ["jt_offset", "jt_size"],
    "gfx_firmware_header_v2_0": ["ucode_size_bytes", "data_size_bytes", "data_offset_bytes", "ucode_start_addr_lo",
                                 "ucode_start_addr_hi"],
    "imu_firmware_header_v1_0": ["imu_iram_ucode_size_bytes", "imu_dram_ucode_size_bytes"],
    "rlc_firmware_header_v2_1": [f"save_restore_list_{l}_{f}_bytes" for l in ("cntl", "gpm", "srm")
                                 for f in ("size", "offset")],
    "rlc_firmware_header_v2_2": [f"rlc_{m}_ucode_{f}_bytes" for m in ("iram", "dram") for f in ("size", "offset")],
    "rlc_firmware_header_v2_3": [f"rlc{m}_ucode_{f}_bytes" for m in ("p", "v") for f in ("size", "offset")],
}
# The structs the kept ones nest.
UCODE_NESTED = ["smc_firmware_header_v1_0", "rlc_firmware_header_v2_0"]
# The components of the PSP's own image, and the types the PSP loads the
# others' pieces as.
PSP_FW_TYPES = ["PSP_FW_TYPE_PSP_SOS", "PSP_FW_TYPE_PSP_SYS_DRV", "PSP_FW_TYPE_PSP_KDB", "PSP_FW_TYPE_PSP_TOC",
                "PSP_FW_TYPE_PSP_SPL", "PSP_FW_TYPE_PSP_RL", "PSP_FW_TYPE_PSP_SOC_DRV", "PSP_FW_TYPE_PSP_INTF_DRV",
                "PSP_FW_TYPE_PSP_DBG_DRV", "PSP_FW_TYPE_PSP_RAS_DRV", "PSP_FW_TYPE_PSP_IPKEYMGR_DRV",
                "PSP_FW_TYPE_PSP_SPDM_DRV"]
GFX_FW_TYPES = ["GFX_FW_TYPE_" + n for n in (
    "CP_ME", "CP_PFP", "CP_MEC", "CP_MEC_ME1", "RLC_V", "RLC_G", "SDMA0", "SDMA1", "SDMA2", "SDMA3", "SMU",
    "RLC_RESTORE_LIST_GPM_MEM", "RLC_RESTORE_LIST_SRM_MEM", "RLC_RESTORE_LIST_SRM_CNTL", "RLC_P", "RLC_IRAM",
    "RLC_DRAM_BOOT", "P2S_TABLE", "REG_LIST", "IMU_I", "IMU_D", "SDMA_UCODE_TH0", "SDMA_UCODE_TH1", "RS64_PFP",
    "RS64_ME", "RS64_MEC", "RS64_PFP_P0_STACK", "RS64_ME_P0_STACK", "RS64_MEC_P0_STACK")]
ENUMS = {UCODE: ["psp_fw_type"],
         PSP_GFX: ["psp_gfx_fw_type", "psp_gfx_crtl_cmd_id", "psp_gfx_cmd_id"],
         PSP_H: ["psp_bootloader_cmd", "psp_ring_type"]}

# The security processor: its commands, ring frames and bootloader steps.
PSP_COMMANDS = ["GFX_CTRL_CMD_ID_DESTROY_RINGS", "GFX_CMD_ID_SETUP_TMR", "GFX_CMD_ID_LOAD_IP_FW", "GFX_CMD_ID_LOAD_TOC",
                "GFX_CMD_ID_AUTOLOAD_RLC", "GFX_CMD_ID_SRIOV_SPATIAL_PART"]
PSP_BOOT = ["PSP_BL__LOAD_KEY_DATABASE", "PSP_BL__LOAD_TOS_SPL_TABLE", "PSP_BL__LOAD_SYSDRV", "PSP_BL__LOAD_SOCDRV",
            "PSP_BL__LOAD_INTFDRV", "PSP_BL__LOAD_DBGDRV", "PSP_BL__LOAD_RASDRV", "PSP_BL__LOAD_IPKEYMGRDRV",
            "PSP_BL__LOAD_SPDMDRV", "PSP_BL__LOAD_SOSDRV",
            "PSP_RING_TYPE__KM"]
PSP_SIZES = ["PSP_FENCE_BUFFER_SIZE", "PSP_CMD_BUFFER_SIZE", "PSP_1_MEG", "PSP_TMR_ALIGNMENT"]
# Each command's struct, the fields kept, and the member the layout stops at
# (a union this library does not read).
PSP_STRUCTS = {
    "psp_gfx_cmd_setup_tmr": (["buf_phy_addr_lo", "buf_phy_addr_hi", "buf_size", "virt_phy_addr",
                               "system_phy_addr_lo", "system_phy_addr_hi"], None),
    "psp_gfx_cmd_load_ip_fw": (["fw_phy_addr_lo", "fw_phy_addr_hi", "fw_size", "fw_type"], None),
    "psp_gfx_cmd_load_toc": (["toc_phy_addr_lo", "toc_phy_addr_hi", "toc_size"], None),
    "psp_gfx_cmd_sriov_spatial_part": (["mode"], None),
    "psp_gfx_resp": (["status", "tmr_size"], "uresp"),
    "psp_gfx_rb_frame": (["cmd_buf_addr_lo", "cmd_buf_addr_hi", "fence_addr_lo", "fence_addr_hi", "fence_value"], None),
}

# The blocks a boot programs, by their hardware IDs' names.
HWIDS = ["GC", "SDMA0", "SDMA1", "SDMA2", "SDMA3", "MP0", "MP1", "MMHUB", "OSSSYS", "NBIF", "HDP"]

# rig_amd_amdgpu

# The uapi excerpts. A name is read from the first header that defines it.
UAPI = sources("kernel", ["include/uapi/linux/kfd_ioctl.h", "include/uapi/drm/amdgpu_drm.h",
                          "include/uapi/drm/drm.h"])

# The definitions the stubs use.
UAPI_NAMES = [
    # KFD's memory
    "KFD_IOC_ALLOC_MEM_FLAGS_VRAM", "KFD_IOC_ALLOC_MEM_FLAGS_GTT", "KFD_IOC_ALLOC_MEM_FLAGS_USERPTR",
    "KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP", "KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE", "KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE",
    "KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC", "KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE", "KFD_IOC_ALLOC_MEM_FLAGS_COHERENT",
    "KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED",
    # KFD's events
    "KFD_IOC_EVENT_SIGNAL", "KFD_IOC_EVENT_MEMORY", "KFD_IOC_EVENT_HW_EXCEPTION",
    # KFD's queues
    "KFD_IOC_QUEUE_TYPE_COMPUTE", "KFD_IOC_QUEUE_TYPE_COMPUTE_AQL", "KFD_IOC_QUEUE_TYPE_SDMA",
    "KFD_MAX_QUEUE_PERCENTAGE",
    # the render node
    "AMDGPU_INFO_DEV_INFO", "AMDGPU_CTX_OP_ALLOC_CTX", "AMDGPU_CTX_OP_FREE_CTX",
    "AMDGPU_CTX_OP_SET_STABLE_PSTATE", "AMDGPU_CTX_STABLE_PSTATE_STANDARD",
    # requests
    "AMDKFD_IOC_GET_VERSION", "AMDKFD_IOC_CREATE_QUEUE", "AMDKFD_IOC_DESTROY_QUEUE", "AMDKFD_IOC_CREATE_EVENT",
    "AMDKFD_IOC_DESTROY_EVENT", "AMDKFD_IOC_RESET_EVENT", "AMDKFD_IOC_WAIT_EVENTS", "AMDKFD_IOC_ACQUIRE_VM",
    "AMDKFD_IOC_ALLOC_MEMORY_OF_GPU", "AMDKFD_IOC_FREE_MEMORY_OF_GPU", "AMDKFD_IOC_MAP_MEMORY_TO_GPU",
    "AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU", "AMDKFD_IOC_RUNTIME_ENABLE", "AMDKFD_IOC_GET_CLOCK_COUNTERS",
    "AMDKFD_IOC_EXPORT_DMABUF",
    "DRM_IOCTL_AMDGPU_INFO", "DRM_IOCTL_AMDGPU_CTX",
    # structures
    "kfd_ioctl_get_version_args", "kfd_ioctl_create_queue_args", "kfd_ioctl_destroy_queue_args",
    "kfd_ioctl_create_event_args", "kfd_ioctl_destroy_event_args", "kfd_ioctl_reset_event_args",
    "kfd_ioctl_wait_events_args", "kfd_event_data", "kfd_ioctl_acquire_vm_args",
    "kfd_ioctl_alloc_memory_of_gpu_args", "kfd_ioctl_free_memory_of_gpu_args",
    "kfd_ioctl_map_memory_to_gpu_args", "kfd_ioctl_unmap_memory_from_gpu_args", "kfd_ioctl_runtime_enable_args",
    "kfd_ioctl_get_clock_counters_args", "kfd_ioctl_export_dmabuf_args",
    "drm_amdgpu_info", "drm_amdgpu_info_device", "drm_amdgpu_ctx",
]


def all_sources():
    """{excerpt: URL} of every library's excerpts: one excerpt of each file."""
    out = {}
    for n, url in [*ABI.items(), *PCI.items(), *UAPI.items()]:
        if out.setdefault(n, url) != url:
            sys.exit(f"{n} is cut from both {out[n]} and {url}")
    if len(set(out.values())) != len(out):
        sys.exit("a file has two excerpts")
    return out


SOURCES = all_sources()

# Reading C headers

# The configuration the headers are read under: a little-endian processor, a
# 64-bit HSA model.
DEFINED = {"LITTLEENDIAN_CPU", "HSA_LARGE_MODEL", "HSA_LITTLE_ENDIAN"}

DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(\w+)(\([\w, ]*\))?[ \t]*(.*?)[ \t]*(?:/\*.*?\*/|//.*)?[ \t]*$", re.M)
ENUMERATOR = re.compile(r"^[ \t]*(\w+)[ \t]*=[ \t]*([^,/\n]+?)[ \t]*,?[ \t]*(?:/\*.*|//.*)?$", re.M)


def defines(text):
    """{name: body} of [text]'s #define lines without parameters."""
    return {m.group(1): m.group(3) for m in DEFINE.finditer(text) if m.group(2) is None}


def arg_defines(text):
    """{name: (parameter, body)} of [text]'s #define lines of one parameter."""
    return {m.group(1): (m.group(2).strip("() "), m.group(3)) for m in DEFINE.finditer(text)
            if m.group(2) is not None and "," not in m.group(2)}


def enumerators(text):
    """{name: value} of [text]'s enumerators with an explicit value."""
    return {m.group(1): m.group(2) for m in ENUMERATOR.finditer(text)}


def evaluate(expr, names):
    """The integer of the C constant expression [expr]: numbers, names of
    [names], casts to integer types, and integer operators."""
    e = re.sub(r"\((unsigned|unsigned int|uint32_t|uint64_t|int)\)", "", expr)
    e = re.sub(r"\b(0[xX][0-9a-fA-F]+|\d+)[uUlL]*\b", r"\1", e)

    def name(m):
        n = m.group(0)
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
    names = {**enumerators(text), **defines(text)}
    out = {}
    for n in wanted:
        if n not in names:
            sys.exit(f"undefined constant {n}")
        out[n] = evaluate(names[n], names) & 0xffff_ffff_ffff_ffff
    return out


def shift(text, name):
    """The first bit [name(1)] sets, for an argument macro [name]."""
    ds = arg_defines(text)
    if name not in ds:
        sys.exit(f"undefined argument macro {name}")
    param, body = ds[name]
    body = re.sub(rf"\b{param}\b", "1", body)
    return (evaluate(body, defines(text)) & 0xffff_ffff_ffff_ffff).bit_length() - 1


def enum_values(text, wanted):
    """{name: value} of the enumerators [wanted] of [text], counting from the
    last explicit value as C does."""
    src = preprocess(text)
    out = {}
    for body in re.findall(r"enum\s*\w*\s*\{(.*?)\}", src, re.S):
        nxt = 0
        for item in body.split(","):
            item = item.strip()
            if not item:
                continue
            name, _, value = item.partition("=")
            name = name.strip()
            nxt = evaluate(value.strip(), {n: str(v) for n, v in out.items()}) if value else nxt
            out[name] = nxt
            nxt += 1
    missing = set(wanted) - set(out)
    if missing:
        sys.exit(f"undefined enumerators {sorted(missing)}")
    return {n: out[n] for n in wanted}


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


def packed_layout(text, name, upto=None):
    """(sizeof, {field: (byte offset, bytes) or ("bits", bit offset, bits)}) of
    the struct [name] of [text], laid out under #pragma pack(1): every member
    at the bit where the one before ends, a bit field's run ending on a byte,
    a union's members at its start. Nested members are named by their own
    names; an array's bytes are its element's; a flexible array takes none.
    The layout stops at the member [upto]."""
    src = preprocess(text)
    src = re.sub(r"DECLARE_FLEX_ARRAY\((\w+),\s*(\w+)\)", r"\1 \2[]", src)
    src = src.replace("__packed", "")
    # Array lengths named by an enumerator or a macro.
    sizes = {**{m.group(1): m.group(2) for m in re.finditer(r"\b(\w+)\s*=\s*(\d+)\b", src)},
             **{n: b for n, b in defines(text).items() if b.isdigit()}}
    tokens = re.findall(r"[A-Za-z_]\w*|\d+|[{};:\[\],]", src)
    defs = {}

    def definition(i):
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
                    return tokens[i], tag, (j + 1, k), k + 1
            k += 1

    for i, t in enumerate(tokens):
        if t in ("struct", "union") and i + 1 < len(tokens):
            j = i + 1 if tokens[i + 1] == "{" else i + 2
            if j < len(tokens) and tokens[j] == "{":
                kind, tag, body, end = definition(i)
                if tag:
                    defs.setdefault(tag, (kind, body))
                if i > 0 and tokens[i - 1] == "typedef":
                    defs.setdefault(tokens[end], (kind, body))

    def walk(kind, body, base, fields):
        """Lays out [body] from bit [base]; its size in bits."""
        i, end = body
        pos, size = 0, 0
        while i < end:
            if tokens[i] in ("struct", "union") and tokens[i + 1] == "{":
                k, _, inner, after = definition(i)
                start = 0 if kind == "union" else -(-pos // 8) * 8
                isize = walk(k, inner, base + start, fields)
                size = max(size, isize) if kind == "union" else size
                pos = pos if kind == "union" else start + isize
                # Past the member's name, if it has one, and its semicolon.
                while tokens[after] != ";":
                    after += 1
                i = after + 1
                continue
            j = i
            while tokens[j] != ";":
                j += 1
            decl = tokens[i:j]
            i = j + 1
            if not decl:
                continue
            if ":" in decl:
                c = decl.index(":")
                width = int(decl[c + 1])
                start = 0 if kind == "union" else pos
                fields[decl[c - 1]] = ("bits", base + start, width)
                size = max(size, width) if kind == "union" else size
                pos = pos if kind == "union" else pos + width
                continue
            count = 1
            if "[" in decl:
                b = decl.index("[")
                count = 0 if decl[b + 1] == "]" else int(sizes.get(decl[b + 1], decl[b + 1]))
                decl = decl[:b]
            ftype, fname = " ".join(t for t in decl[:-1] if t != "struct"), decl[-1]
            if fname == upto:
                return pos
            if ftype in SCALARS:
                fbytes = SCALARS[ftype]
            elif ftype.startswith("enum "):
                fbytes = 4
            elif ftype in defs:
                k, inner = defs[ftype]
                fbytes = walk(k, inner, 0, {}) // 8
            else:
                sys.exit(f"{name}: unknown type {ftype}")
            start = 0 if kind == "union" else -(-pos // 8) * 8
            fields[fname] = ((base + start) // 8, fbytes)
            bits = fbytes * count * 8
            size = max(size, bits) if kind == "union" else size
            pos = pos if kind == "union" else start + bits
        total = size if kind == "union" else pos
        return -(-total // 8) * 8

    if name not in defs:
        sys.exit(f"no struct {name}")
    kind, body = defs[name]
    fields = {}
    return walk(kind, body, 0, fields) // 8, fields


# Registers

REG_DEFINE = re.compile(r"^\s*#\s*define\s+((?:mm|reg)\w+)\s+(0x[\da-fA-F]+|\d+)[uUlL]*\b", re.M)
MASK_DEFINE = re.compile(r"^\s*#\s*define\s+(\w+?)__(\w+)_MASK\s+(0x[\da-fA-F]+|\d+)[uUlL]*\b", re.M)


def split_name(name):
    """(prefix, name) of a register's macro: "reg" or "mm", and the rest."""
    pos = next((i for i, c in enumerate(name) if c.isupper()), len(name))
    return name[:pos], name[pos:]


def hubbed(reg, hub):
    """[reg], a VM register of [hub]'s headers named with its hub, if any:
    GC's regVM_L2_CNTL is regGCVM_L2_CNTL."""
    p, rest = split_name(reg)
    return p + hub + rest if hub and rest.startswith(("VM_", "MC_VM_")) else reg


def kept(reg, patterns):
    return any(re.fullmatch(p, reg) for p in patterns)


def registers(offsets, masks, patterns, hub=None):
    """{name: (offset, segment, [(field, lowest bit, highest bit)])} of the
    registers of a block's offset and mask headers whose names, with their
    [hub], match [patterns]."""
    defs = {m.group(1): int(m.group(2), 0) for m in REG_DEFINE.finditer(offsets)}
    fields = {}
    for m in MASK_DEFINE.finditer(masks):
        mask = int(m.group(3), 0)
        fields.setdefault(m.group(1), []).append(
            (m.group(2).lower(), (mask & -mask).bit_length() - 1, mask.bit_length() - 1))
    out = {}
    for reg, off in defs.items():
        name = hubbed(reg, hub)
        if reg.endswith("_BASE_IDX") or f"{reg}_BASE_IDX" not in defs or not kept(name, patterns):
            continue
        out[name] = (off, defs[f"{reg}_BASE_IDX"], fields.get(split_name(reg)[1], []))
    return out


def register_lines(lines, patterns, hub=None):
    """The lines of a register header that define the registers [patterns]
    keep, their segments and their fields' masks."""
    keep = set()
    for i, l in enumerate(lines):
        r = REG_DEFINE.match(l)
        if r and kept(hubbed(r.group(1).removesuffix("_BASE_IDX"), hub), patterns):
            keep.add(i)
        k = MASK_DEFINE.match(l)
        if k and kept(hubbed("reg" + k.group(1), hub), patterns):
            keep.add(i)
    return keep


def hw_ids(text):
    """[(name, id)] of soc15_hw_ip.h's hardware IDs, in the header's order,
    aliases resolved."""
    names = defines(text)
    return [(n[:-len("_HWID")], evaluate(b, names)) for n, b in names.items() if n.endswith("_HWID")]

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

# Definitions, of the uapi headers

COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
MACRO = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(\w+)(\([\w\s,]*\))?((?:[^\n]*\\\n)*[^\n]*)$", re.M)
TYPEDEF = re.compile(r"^[ \t]*typedef\b", re.M)
TAGGED = re.compile(r"^[ \t]*(struct|union|enum)[ \t]+(\w+)\s*\{", re.M)
IDENT = re.compile(r"\b[A-Za-z_]\w*\b")
DIRECTIVE = re.compile(r"^[ \t]*#[ \t]*(if|ifdef|ifndef|elif|else|endif|define|undef)\b(.*)$")

# The macros a 64-bit Linux build of a program defines that the headers'
# conditionals test; every other name a conditional tests is undefined.
PLATFORM = {"__linux__", "__LP64__"}


def blank(text):
    """[text] with its comments blanked out, newlines kept, so offsets hold."""
    return COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), text)


def matching(text, i, pair="{}"):
    """The index of the bracket that closes the one at [i]."""
    depth = 0
    for j in range(i, len(text)):
        depth += {pair[0]: 1, pair[1]: -1}.get(text[j], 0)
        if depth == 0:
            return j
    sys.exit("unbalanced brackets")


class Item:
    """A definition of a header: a define, a struct, union or enum, or a
    typedef of another type, with its span in the header's text."""

    def __init__(self, kind, names, start, end, **k):
        self.kind, self.names, self.start, self.end = kind, names, start, end
        self.__dict__.update(k)


def condition(directive, expr, defined):
    """Whether the conditional [directive] [expr] holds, given the macros
    [defined] so far, by name, with their bodies (None for a function-like
    one). A name the expression uses that no macro defines is 0."""
    if directive in ("ifdef", "ifndef"):
        return (expr.split()[0] in defined) == (directive == "ifdef")
    e = re.sub(r"\bdefined\s*(?:\(\s*(\w+)\s*\)|(\w+))",
               lambda m: "1" if (m.group(1) or m.group(2)) in defined else "0", expr)
    for _ in range(16):
        e = IDENT.sub(lambda m: f"({defined[m.group()]})" if defined.get(m.group()) else "0", e)
    e = re.sub(r"\b(0[xX][0-9a-fA-F]+|\d+)[uUlL]+\b", r"\1", e)
    e = e.replace("&&", " and ").replace("||", " or ")
    e = re.sub(r"!(?!=)", " not ", e)
    try:
        return bool(eval(e, {"__builtins__": {}}))
    except Exception:
        sys.exit(f"cannot evaluate the conditional {expr}")


def active(b):
    """For each line of the blanked text [b], whether a 64-bit Linux build
    compiles it."""
    defined = dict.fromkeys(PLATFORM, "1")
    stack, out = [], []
    for line in b.split("\n"):
        live = all(taken for taken, _ in stack)
        out.append(live)
        m = DIRECTIVE.match(line)
        if not m:
            continue
        d, rest = m.group(1), m.group(2).strip()
        if d in ("if", "ifdef", "ifndef"):
            c = live and condition(d, rest, defined)
            stack.append([c, c])
        elif d == "elif":
            outer = all(taken for taken, _ in stack[:-1])
            c = outer and not stack[-1][1] and condition("if", rest, defined)
            stack[-1] = [c, stack[-1][1] or c]
        elif d == "else":
            outer = all(taken for taken, _ in stack[:-1])
            stack[-1] = [outer and not stack[-1][1], True]
        elif d == "endif":
            stack.pop()
        elif d == "define" and live:
            m = re.match(r"(\w+)(\()?\s*(.*)", rest)
            defined[m.group(1)] = None if m.group(2) else (m.group(3) or "1")
        elif d == "undef" and live:
            defined.pop(rest.split()[0], None)
    return out


def items(text):
    """The definitions a 64-bit Linux build compiles of the header [text]."""
    b = blank(text)
    lines = active(b)
    out = []
    for m in MACRO.finditer(b):
        body = m.group(3).replace("\\\n", " ").strip()
        out.append(Item("define", [m.group(1)], m.start(), m.end(), body=body))
    for m in TYPEDEF.finditer(b):
        end = b.index(";", m.end())
        brace = b.find("{", m.end())
        if 0 <= brace < end:
            head = b[m.end():brace]
            close = matching(b, brace)
            end = b.index(";", close)
            names = [n.strip() for n in b[close + 1:end].split(",")]
            kind = re.search(r"\b(struct|union|enum)\b", head).group(1)
            tag = re.sub(r"\b(volatile|struct|union|enum)\b", "", head).strip()
            it = Item(kind, names + ([tag] if tag else []), m.start(), end + 1, body=b[brace + 1:close])
            out.append(it)
            if kind == "enum":
                out += enum_items(it)
        else:
            words = b[m.end():end].split()
            out.append(Item("alias", [words[-1]], m.start(), end + 1, target=words[-2]))
    for m in TAGGED.finditer(b):
        close = matching(b, m.end() - 1)
        end = b.index(";", close)
        if b[close + 1:end].strip() == "":
            it = Item(m.group(1), [m.group(2)], m.start(), end + 1, body=b[m.end():close])
            out.append(it)
            if it.kind == "enum":
                out += enum_items(it)
    return [it for it in out if lines[b.count("\n", 0, it.start)]]


def enum_items(enum):
    """The constants of [enum], each an item over the enum's span."""
    entries = [e.split("=", 1) for e in (e.strip() for e in enum.body.split(",")) if e]
    return [Item("constant", [e[0].strip()], enum.start, enum.end, enum=enum) for e in entries]


class Model:
    """The definitions of the headers, by name, from the first header that
    defines each."""

    def __init__(self, texts):
        self.where = {}
        for h, text in texts.items():
            for it in items(text):
                it.header = h
                for n in it.names:
                    if n in self.where and self.where[n].header == h:
                        old = self.where[n]
                        if text[old.start:old.end] != text[it.start:it.end] and it.kind == "define":
                            sys.exit(f"{h}: {n} is defined twice")
                    self.where.setdefault(n, it)

    def deps(self, name):
        """The names [name]'s definition refers to."""
        it = self.where[name]
        if it.kind == "constant":
            return set(it.enum.names)
        if it.kind == "alias":
            return {it.target}
        return set(IDENT.findall(it.body))

    def closure(self, names):
        """The definitions [names] need, by header: every name their
        definitions refer to that some header defines."""
        seen, todo = set(), list(names)
        while todo:
            n = todo.pop()
            if n in seen:
                continue
            seen.add(n)
            if n not in self.where:
                if n not in names:
                    continue
                sys.exit(f"no definition of {n}")
            todo += [d for d in self.deps(n) if d in self.where and d != n]
        keep = {}
        for n in seen:
            if n in self.where:
                it = self.where[n]
                keep.setdefault(it.header, {})[it.start] = it
        return keep

# Excerpts

KD_PROPERTY = re.compile(r"^\s*KERNEL_CODE_PROPERTY\((\w+),\s*(\d+),\s*(\d+)\),.*$", re.M)


def abi_wanted():
    """For each excerpt rig_amd_abi reads: the macros, enumerators and structs
    it keeps."""
    w = {name: {"defines": set(), "optional": set(), "enums": set(), "structs": set()} for name in ABI}
    w[SOC15D]["defines"] |= {*PM4_CONSTANTS, *PM4_SOC15_ONLY, *PM4_SOC15_CONSTANTS, *PM4_SOC15_SHIFTS,
                             *(g for g, _ in PM4_SHIFTS)}
    w[NVD]["defines"] |= {*PM4_CONSTANTS, *PM4_NV_CONSTANTS, *PM4_NV_SHIFTS, *(n for _, n in PM4_SHIFTS)}
    w[KFD_PM4]["enums"] |= set(PM4_ENUMS)
    for h in SDMA_PKT.values():
        fields = {f"{n}_{k}" for n in SDMA_FIELDS for k in ("mask", "shift")}
        optional = {f"{n}_{k}" for n in SDMA_OPTIONAL for k in ("mask", "shift")}
        w[h]["defines"] |= {*SDMA_OPS, *(fields - optional)}
        w[h]["optional"] |= optional
    for h in GC_BASES.values():
        w[h]["optional"] |= set(GC_SEGMENTS)
    for h in SOCS:
        w[h]["optional"] |= {*SOC_EVENTS, *SOC_TRACE}
    w["rocm/vega10_enum.h"]["enums"] |= {f"SQ_THREAD_TRACE_TOKEN_{t}" for t in GFX9_TOKENS}
    w[HSA]["enums"] |= set(HSA_CONSTANTS)
    w[HSA]["structs"] |= {DISPATCH, "hsa_signal_s"}
    w[REGISTERS_H]["enums"] |= set(SQ_CONSTANTS)
    w[REGISTERS_H]["structs"] |= {u for pair in SQ_BUF_RSRC.values() for u in pair}
    w[KD_H]["enums"] |= {f.upper() + "_OFFSET" for f in KD_FIELDS}
    w[KD_H]["structs"] |= {KD}
    w[ELF_H]["enums"] |= set(ELF_CONSTANTS)
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
    """The file's leading comments, its licence notice, then the source the
    excerpt is cut from; for a document, a comment that names its source and
    licence. Fails on a notice that is not MIT's, NCSA's or LLVM's."""
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
    if not re.search(r"Permission is hereby granted|NCSA|LLVM Exceptions", m.group(1)):
        sys.exit(f"{name}: a notice that is not MIT's, NCSA's or LLVM's")
    return m.group(1).rstrip() + f"\n\n/* Excerpt of {SOURCES[name]}. */\n"


def blocks(text, names):
    """The lines of the definitions of the structs and unions [names]: from the
    line of their keyword to the line of their closing brace and name."""
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
    if found != set(names):
        sys.exit(f"no definition of {sorted(set(names) - found)}")
    return out


def enum_blocks(text, names):
    """The lines of the enumerations [names], from their keyword to their
    closing brace."""
    lines = text.splitlines()
    out, found = set(), set()
    for i, l in enumerate(lines):
        m = re.match(r"\s*(?:typedef\s+)?enum\s+(\w+)\s*\{?\s*$", l)
        if m is None or m.group(1) not in names or m.group(1) in found:
            continue
        j = next(k for k in range(i, len(lines)) if lines[k].strip().startswith("}"))
        out |= set(range(i, j + 1))
        found.add(m.group(1))
    if found != set(names):
        sys.exit(f"no enumeration {sorted(set(names) - found)}")
    return out


def enum_with(text, member):
    """The lines of the enumeration that has [member], from its keyword to its
    closing brace."""
    lines = text.splitlines()
    at = next((i for i, l in enumerate(lines) if re.match(rf"\s*{member}\b", l)), None)
    if at is None:
        sys.exit(f"no enumeration with {member}")
    first = next(i for i in range(at, -1, -1) if re.match(r"\s*(typedef\s+)?enum\b", lines[i]))
    last = next(i for i in range(at, len(lines)) if lines[i].strip().startswith("}"))
    return set(range(first, last + 1))


def abi_lines(name, text):
    """The lines rig_amd_abi reads of the excerpt [name] of [text]: the macros,
    enumerators and structs it keeps and the macros a kept one refers to."""
    w = abi_wanted()[name]
    lines = text.splitlines()
    keep = set(blocks(text, w["structs"])) if w["structs"] else set()
    ds = {m.group(1) for m in DEFINE.finditer(text)}
    if name.startswith("kernel/gc_"):
        keep |= register_lines(lines, GC_REGISTERS, "GC")
    for i, l in enumerate(lines):
        m = DEFINE.match(l)
        if m and m.group(1) in w["defines"] | w["optional"]:
            keep.add(i)
        e = ENUMERATOR.match(l)
        if e and e.group(1) in w["enums"] | w["optional"]:
            keep.add(i)
        if name == ELF_H and e and e.group(1).startswith("EF_AMDGPU_MACH_AMDGCN_"):
            keep.add(i)
        p = KD_PROPERTY.match(l)
        if name == KD_H and p and p.group(1) in KD_PROPERTIES:
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
    return keep


def is_define(l, names):
    m = DEFINE.match(l)
    return m is not None and m.group(1) in names


def pci_lines(name, text):
    """The lines rig_amd_pci reads of the excerpt [name] of [text]."""
    lines = text.splitlines()
    keep = set()
    if name == DISCOVERY:
        keep |= blocks(text, set(DISCOVERY_STRUCTS))
        # The table enumeration and the pack(1) around the structs.
        first = next(i for i, l in enumerate(lines) if re.match(r"\s*typedef enum\s*\{?\s*$", l))
        last = next(i for i in range(first, len(lines)) if lines[i].strip().startswith("}"))
        keep |= set(range(first, last + 1))
        keep |= {i for i, l in enumerate(lines) if l.startswith("#pragma pack")}
        keep |= {i for i, l in enumerate(lines) if is_define(l, DISCOVERY_CONSTANTS)}
    elif name == "kernel/amdgpu_discovery.h":
        keep |= {i for i, l in enumerate(lines) if is_define(l, DISCOVERY_PLACE)}
    elif name == UCODE:
        keep |= blocks(text, set(UCODE_STRUCTS) | set(UCODE_NESTED)) | enum_blocks(text, ENUMS[name])
    elif name == "kernel/amdgpu_vm.h":
        keep |= enum_blocks(text, ["amdgpu_vm_level"])
        keep |= {i for i, l in enumerate(lines) if is_define(l, PTE_BITS + PTE_SHIFTS)}
    elif any(name == m for _, m, _, _ in SMU_TABLES):
        keep |= {i for i, l in enumerate(lines) if is_define(l, SMU_MESSAGES)}
    elif any(name in (c, f) for _, _, c, f in SMU_TABLES):
        if any(name == c for _, _, c, _ in SMU_TABLES):
            keep |= enum_with(text, "PPCLK_UCLK")
        keep |= {i for i, l in enumerate(lines) if SMU_FEATURE.search(l)}
    elif name == IH_CLIENTID:
        keep |= enum_blocks(text, IH_ENUMS)
    elif name in IH_SOURCES:
        keep |= {i for i, l in enumerate(lines) if SRCID.match(l)}
    elif name in MTYPES:
        keep |= {i for i, l in enumerate(lines) if re.match(r"\s*(MTYPE_UC|MTYPE_NC|SH_MEM_ADDRESS_MODE_64|"
                                                              r"SH_MEM_ALIGNMENT_MODE_UNALIGNED)\s*=", l)}
    elif name == "kernel/amdgpu_discovery.c":
        keep |= {i for i, l in enumerate(lines) if is_define(l, EARLY)}
    elif name == "kernel/amdgpu_virt.h":
        keep |= {i for i, l in enumerate(lines) if is_define(l, VF)}
    elif name == "kernel/mxgpu_nv.h":
        keep |= {i for i, l in enumerate(lines) if is_define(l, MAILBOX)}
        keep |= enum_blocks(text, ["idh_request", "idh_event"])
    elif name == "kernel/amdgpu_doorbell.h":
        keep |= enum_blocks(text, ["AMDGPU_NAVI10_DOORBELL_ASSIGNMENT"])
    elif name in MQDS:
        keep |= blocks(text, {MQDS[name][0]})
    elif name == PSP_GFX:
        keep |= enum_blocks(text, ENUMS[name]) | blocks(text, set(PSP_STRUCTS) | {"psp_gfx_cmd_resp"})
    elif name == PSP_H:
        keep |= enum_blocks(text, ENUMS[name])
        keep |= {i for i, l in enumerate(lines) if is_define(l, PSP_SIZES)}
    elif name.endswith(("_offset.h", "_sh_mask.h")):
        block = next(b for b, vs in REG_FILES.items() for v in vs
                     if name in (reg_header(b, v, "offset"), reg_header(b, v, "sh_mask")))
        keep |= register_lines(lines, REG_INVENTORY[block], HUBS.get(block))
    elif name == "kernel/soc15_hw_ip.h":
        keep |= {i for i, l in enumerate(lines) if DEFINE.match(l) and DEFINE.match(l).group(1).endswith("_HWID")}
    return keep


def excerpt(name, text):
    """[name]'s excerpt of [text]: its licence notice, its URL and the lines
    each library reads, in order."""
    lines = text.splitlines()
    if name.endswith(".rst"):
        return licence(name, text) + "\n" + "\n\n".join("\n".join(rst_table_lines(lines, t)) for t in RST_TABLES) + "\n"
    if name.endswith(".yaml"):
        keep = set()
        _, spans = counter_defs(text)
        for (n, _, _), (first, last) in spans.items():
            keep.add(next(i for i in range(first, -1, -1) if lines[i] == f"  - name: {n}"))
            keep.add(next(i for i in range(first, -1, -1) if lines[i].strip() == "definitions:"))
            keep |= set(range(first, last + 1))
        return licence(name, text) + "\n".join(lines[i] for i in sorted(keep)) + "\n"
    keep = (abi_lines(name, text) if name in ABI else set()) | (pci_lines(name, text) if name in PCI else set())
    return licence(name, text) + "\n" + "\n".join(lines[i] for i in sorted(keep)) + "\n"


def uapi_excerpts(texts):
    """The uapi excerpts: the definitions of UAPI_NAMES and those they need,
    whole, in each header's order."""
    keep = Model(texts).closure(set(UAPI_NAMES))
    out = {}
    for h in texts:
        if h not in keep:
            sys.exit(f"{h}: nothing read from it")
        body = []
        for it in sorted(keep[h].values(), key=lambda it: it.start):
            if body and it.start < body[-1][1]:
                continue
            body.append((it.start, it.end))
        out[h] = licence(h, texts[h]) + "\n" + "\n\n".join(texts[h][s:e] for s, e in body) + "\n"
    return out


def firmware(get):
    """firmware.tsv: each pinned image's path, BLAKE2b-256 digest and URL."""
    rows = ["# path\tBLAKE2b-256\tURL, of linux-firmware at " + FIRMWARE_COMMIT]
    for n in IMAGES:
        url = FIRMWARE_URL + "amdgpu/" + n
        rows.append(f"amdgpu/{n}\t{hashlib.blake2b(get(url), digest_size=32).hexdigest()}\t{url}")
    return "\n".join(rows) + "\n"


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
    return data

# Emission


def ml_name(c):
    return c.lower()


def ml_int(v):
    return f"0x{v:x}" if v > 9 else str(v)


def ml_version(v):
    return "(%d, %d, %d)" % v


def ml_field(v):
    return f"({v[1]}, {v[2]})" if v[0] == "bits" else f"({v[0]}, {v[1]})"


def ml_module(name):
    return name[0].upper() + name[1:]


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
    return "\n".join(l for l in lines if not l.startswith("//"))


def owners(texts):
    """AMD's copyright lines in [texts]."""
    return sorted({m.group(0).rstrip(" */") for t in texts
                   for m in re.finditer(r"Copyright [^\n]*?Advanced Micro Devices, Inc\.[^\n]*", t)})


def indent(s):
    return "\n".join(f"   {l}".rstrip() for l in s.splitlines())


def generated(out):
    """The opening of the generated file [out]: raven's licence, and where it
    comes from."""
    script = os.path.relpath(HERE / "gen.py", out.parent)
    excerpts = os.path.relpath(HEADERS, out.parent)
    return (
        "(*---------------------------------------------------------------------------\n"
        "  Copyright (c) 2026 The Raven authors. All rights reserved.\n"
        "  SPDX-License-Identifier: ISC\n"
        "  ---------------------------------------------------------------------------*)\n\n"
        f"(* Generated by {script} from the excerpts in {excerpts};\n"
        f"   do not edit. The command that regenerates this file is in {script}.\n\n"
    )


def abi_header(h):
    """The licence header of rig_amd_abi's defs.ml: raven's, then the notices
    of the sources of its values, from the excerpts [h]."""
    kind = {n: "llvm" if "LLVM Exceptions" in t else "ncsa" if "NCSA" in t else "mit" for n, t in h.items()}
    ncsa = sorted(base(n) for n in h if kind[n] == "ncsa")
    llvm = sorted(base(n) for n in h if kind[n] == "llvm")
    ncsa_notices = {notice(t).strip("\n") for n, t in h.items() if kind[n] == "ncsa"}
    if len(ncsa_notices) != 1:
        sys.exit(f"the notices of {ncsa} differ")
    return (
        generated(ABI_OUT)
        + "   The values are AMD's and LLVM's, copied from their headers and documents.\n\n"
        f"   AMD's {' and '.join(ncsa)} are under the University of Illinois/NCSA\n"
        "   Open Source License:\n\n"
        + indent(ncsa_notices.pop()) + "\n\n"
        "   AMD's other sources are under the MIT licence:\n\n"
        + indent("\n".join(owners(t for n, t in h.items() if kind[n] == "mit"))) + "\n\n" + indent(MIT) + "\n\n"
        f"   LLVM's {', '.join(llvm[:-1])} and {llvm[-1]} are under\n"
        "   the Apache License v2.0 with LLVM Exceptions (SPDX-License-Identifier:\n"
        "   Apache-2.0 WITH LLVM-exception), whose text is in rig's LICENSE-llvm. *)\n"
    )


def pci_header(h):
    """The licence header of rig_amd_pci's defs.ml: raven's, then the notice
    of AMD's headers, from the excerpts [h]."""
    bare = [n for n in PCI if "Permission is hereby granted" not in h[n]]
    if bare:
        sys.exit(f"{bare}: not under the MIT licence")
    return (
        generated(PCI_OUT)
        + "   The values are AMD's, copied from its headers under the MIT licence:\n\n"
        + indent("\n".join(owners(h[n] for n in PCI))) + "\n\n" + indent(MIT) + " *)\n"
    )


def generate_abi(h):
    """rig_amd_abi's defs.ml, from the excerpts [h], by name."""
    out = [abi_header({n: h[n] for n in ABI})]

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
        regs = registers(h[reg_header("gc", ver, "offset")], h[reg_header("gc", ver, "sh_mask")], GC_REGISTERS, "GC")
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
        gens.append((major, base(name), bases))
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

    def segment_base(major, segment):
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
        if any(r not in regs or segment_base(ver[0], regs[r][1]) is None for r in names):
            continue
        v = "_".join(map(str, ver))
        fs = [f"{f} = {ml_int(segment_base(ver[0], regs[r][1]) + regs[r][0])}" for f, r in DISPATCH_REGISTERS]
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
    kfd = constants(h[KFD_PM4], PM4_ENUMS)
    sv = constants(h[SOC15D], PM4_CONSTANTS + PM4_SOC15_ONLY + PM4_SOC15_CONSTANTS)
    nvv = constants(h[NVD], PM4_CONSTANTS + PM4_NV_CONSTANTS)
    out += ["(* PM4, the same in soc15d.h (GFX9) and nvd.h (GFX10 on), and the release's",
            "   enumerations in kfd_pm4_headers_ai.h *)", ""]
    out += [f"let {ml_name(n)} = {ml_int(same(n, [sv[n], nvv[n]]))}" for n in PM4_CONSTANTS]
    out += [f"let {ml_name(n)} = {ml_int(kfd[n])}" for n in PM4_ENUMS]
    ss = {n: shift(h[SOC15D], n) for n in sorted({g for g, _ in PM4_SHIFTS} | set(PM4_SOC15_SHIFTS))}
    ns = {n: shift(h[NVD], n) for n in sorted({n for _, n in PM4_SHIFTS} | set(PM4_NV_SHIFTS))}
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
    counters, _ = counter_defs(h[COUNTERS])
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
    for m in MASK_DEFINE.finditer(h[reg_header("gc", (9, 4, 3), "sh_mask")]):
        if m.group(1).startswith("SQ_THREAD_TRACE_WORD_"):
            masks.setdefault(m.group(1)[len("SQ_THREAD_TRACE_WORD_"):], {})[m.group(2)] = int(m.group(3), 0)

    def halfwords(word):
        if word not in masks:
            sys.exit(f"gc_9_4_3_sh_mask.h has no SQ_THREAD_TRACE_WORD_{word}")
        return 1 if max(masks[word].values()).bit_length() <= 16 else 2
    vega10 = constants(h["rocm/vega10_enum.h"], [f"SQ_THREAD_TRACE_TOKEN_{t}" for t in GFX9_TOKENS])
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
        ds = defines(h[name])
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
    hv = constants(h[HSA], HSA_CONSTANTS)
    out += [f"let {ml_name(n)} = {ml_int(hv[n])}" for n in HSA_CONSTANTS]
    out.append("")
    size, fields = layout(h[HSA], DISPATCH)
    out += ["module Dispatch = struct", f"  let sizeof = {size}"]
    out += [f"  let {f} = {ml_field(fields[f])}" for f in DISPATCH_FIELDS]
    out += ["end", ""]

    # Scratch buffer descriptors
    rv = constants(h[REGISTERS_H], SQ_CONSTANTS)
    out += ["(* The scratch buffer descriptor's words 1 and 3, by GC major: each field", "   as (bit offset, bits). *)", ""]
    out += ["type sq_buf_rsrc = {"]
    out += [f"  {ml_name(f) if f != 'TYPE' else 'type_'} : int * int;" for f in SQ_WORD1 + SQ_WORD3]
    out += [f"  {ml_name(f)} : (int * int) option;" for f in SQ_WORD3_OPTIONAL]
    out += ["}", "", "let sq_buf_rsrc = ["]
    for major, (w1, w3) in SQ_BUF_RSRC.items():
        def bits(union):
            _, fields = layout(h[REGISTERS_H], union)
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
    kd = h[KD_H]
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
    elf, machs, generic = processors(h[ELF_H], h[USAGE])
    out += ["", "(* The ELF header of a code object, and its notes' types. *)"]
    out += [f"let {ml_name(c)} = {ml_int(v)}" for c, v in elf.items()]
    out += ["", "(* LLVM's AMDGCN processors, by their EF_AMDGPU_MACH value. *)", "let processors = ["]
    out += [f"  ({ml_int(v)}, {json.dumps(n)});" for v, n in machs]
    out += ["]", "", "(* The generic processors, and the processors that run their code objects. *)", "let generic = ["]
    out += [f"  ({json.dumps(n)}, [ " + "; ".join(json.dumps(m) for m in ms) + " ]);" for n, ms in generic]
    out.append("]")
    return "\n".join(out) + "\n"


def generate_pci(h):
    """rig_amd_pci's defs.ml, from the excerpts [h], by name."""
    ml = [pci_header(h)]

    def struct(text, s, wanted, upto=None):
        size, fields = packed_layout(text, s, upto)
        ml.append(f"module {ml_module(s)} = struct")
        ml.append(f"  let sizeof = {size}")
        for f in wanted:
            if f not in fields:
                sys.exit(f"{s} has no field {f}")
            ml.append(f"  let {f} = {ml_field(fields[f])}")
        ml.extend(["end", ""])

    # Discovery
    d = h[DISCOVERY]
    ml += ["(* The discovery table *)", ""]
    for n, v in constants(d, DISCOVERY_CONSTANTS).items():
        ml.append(f"let {n.lower()} = {ml_int(v)}")
    for n, v in constants(h["kernel/amdgpu_discovery.h"], DISCOVERY_PLACE).items():
        ml.append(f"let {n.lower()} = {ml_int(v)}")
    ml.append("")
    ml.append("(* The tables the binary header lists, by index. *)")
    for n, v in enum_values(d, DISCOVERY_TABLES).items():
        ml.append(f"let {n.lower()}_table = {v}")
    ml.append("")
    ml.append("(* Layouts: each field as (byte offset, bytes), or (bit offset, bits) for a")
    ml.append("   bit field; an array's bytes are its element's. *)")
    for s, wanted in DISCOVERY_STRUCTS.items():
        struct(d, s, wanted)

    # Hardware IDs
    ids = hw_ids(h["kernel/soc15_hw_ip.h"])
    known = dict(ids)
    ml += ["(* The blocks' hardware IDs, and their names *)", ""]
    for n in HWIDS:
        ml.append(f"let {n.lower()}_hwid = {known[n]}")
    ml.append("")
    first = {}
    for n, v in ids:
        first.setdefault(v, n)
    ml.append("let hwid_name = function")
    for v, n in sorted(first.items()):
        ml.append(f"  | {v} -> {json.dumps(n)}")
    ml.append("  | _ -> \"\"")
    ml.append("")

    # Page tables
    vm = h["kernel/amdgpu_vm.h"]
    ml += ["(* Page tables: their entries, and the memory types and modes of each", "   generation *)", ""]
    for n, v in constants(vm, PTE_BITS).items():
        ml.append(f"let {n.lower()} = 0x{v:x}L")
    for n in PTE_SHIFTS:
        ml.append(f"let {n.lower().removesuffix('_shift')}_shift = {shift(vm, n)}")
    for n, v in enum_values(vm, VM_LEVELS).items():
        ml.append(f"let {n.lower()} = {v}")
    for hdr, gen_name in MTYPES.items():
        for n in ["MTYPE_UC", "MTYPE_NC"] + SH_MEM:
            m = re.search(rf"\b{n}\s*=\s*(0x[0-9a-fA-F]+|\d+)", h[hdr])
            if m is None:
                sys.exit(f"{hdr}: no {n}")
            ml.append(f"let {gen_name}_{n.lower()} = {int(m.group(1), 0)}")
    ml.append("")

    # Before discovery, and virtual functions
    ml += ["(* Before the discovery table *)", ""]
    for n, v in constants(h["kernel/amdgpu_discovery.c"], EARLY).items():
        ml.append(f"let {n.lower()} = {ml_int(v)}")
    ml += ["", "(* Virtual functions and their mailbox *)", ""]
    for n, v in {**constants(h["kernel/amdgpu_virt.h"], VF), **constants(h["kernel/mxgpu_nv.h"], MAILBOX),
                 **enum_values(h["kernel/mxgpu_nv.h"], MAILBOX_ENUMS)}.items():
        ml.append(f"let {n.lower()} = {ml_int(v)}")
    ml.append("")

    # Compute queues
    ml += ["(* Compute queues *)", ""]
    for n, v in enum_values(h["kernel/amdgpu_doorbell.h"], DOORBELLS).items():
        ml.append(f"let {n.lower()} = {ml_int(v)}")
    ml.append("")
    ml += ["(* The fields of a memory queue descriptor, as (byte offset, bytes); the",
           "   thread management words of its shader engines, in order. *)",
           "type mqd = {"]
    ml += [f"  {f} : int * int;" for f in MQD_FIELDS]
    ml += [f"  {f} : (int * int) option;" for f in MQD_OPTIONAL]
    ml += ["  compute_static_thread_mgmt : (int * int) list;", "  sizeof : int;", "}", ""]
    for hdr, (st, major) in MQDS.items():
        size, fields = packed_layout(h[hdr], st)
        ml.append(f"(* GFX{major}'s, {st} *)")
        ml.append(f"let mqd_v{major} = {{")
        for f in MQD_FIELDS:
            if f not in fields:
                sys.exit(f"{st} has no field {f}")
            ml.append(f"  {f} = {ml_field(fields[f])};")
        for f in MQD_OPTIONAL:
            ml.append(f"  {f} = " + (f"Some {ml_field(fields[f])};" if f in fields else "None;"))
        mgmt = sorted((f for f in fields if f.startswith("compute_static_thread_mgmt_se")),
                      key=lambda f: int(f[len("compute_static_thread_mgmt_se"):]))
        ml.append("  compute_static_thread_mgmt = [ " + "; ".join(ml_field(fields[f]) for f in mgmt) + " ];")
        ml.append(f"  sizeof = {size};")
        ml += ["}", ""]

    # Security processor
    g, ps = h[PSP_GFX], h[PSP_H]
    ml += ["(* The security processor *)", ""]
    for n, v in {**enum_values(g, PSP_COMMANDS), **enum_values(ps, PSP_BOOT), **constants(ps, PSP_SIZES)}.items():
        ml.append(f"let {n.lower()} = {ml_int(v)}")
    # A command buffer holds the command at the offset its first reserved
    # array's size subtracts, and the response at the one it subtracts from,
    # in a buffer of the size the second array's subtracts from.
    m = re.search(r"reserved_1\[(\d+)\s*-\s*sizeof\(union psp_gfx_commands\)\s*-\s*(\d+)\]", g)
    n = re.search(r"reserved_2\[(\d+)\s*-\s*(\d+)\s*-\s*sizeof\(struct psp_gfx_resp\)\]", g)
    if m is None or n is None or m.group(1) != n.group(2):
        sys.exit("psp_gfx_cmd_resp: no command and response offsets")
    ml += [f"let psp_command_bytes = {n.group(1)}", f"let psp_command_at = {m.group(2)}",
           f"let psp_response_at = {m.group(1)}"]
    size, fields = packed_layout(g, "psp_gfx_cmd_resp", upto="cmd")
    ml.append(f"let psp_command_id = {ml_field(fields['cmd_id'])}")
    ml.append("")
    for st, (wanted, upto) in PSP_STRUCTS.items():
        struct(g, st, wanted, upto)

    # Power manager
    ml += ["(* The power manager's messages, clocks and the feature bits of their DPM, by MP1",
           "   version, as (name, value). *)",
           "let smu_messages = function"]
    for versions, m, c, f in SMU_TABLES:
        msgs = {n: v for n, v in ((n, b) for n, b in defines(h[m]).items()) if n in SMU_MESSAGES}
        vals = [(n, evaluate(msgs[n], {})) for n in SMU_MESSAGES if n in msgs]
        clocks = enum_values(h[c], [n for n in SMU_CLOCKS if re.search(rf"\b{n}\b", h[c])])
        vals += list(clocks.items())
        features = dict((n, int(v, 0)) for n, v in SMU_FEATURE.findall(h[f]))
        vals += [(n, features[n]) for n in SMU_FEATURES if n in features]
        pats = " | ".join(f"({a}, {b}, {cc})" for a, b, cc in versions)
        ml.append(f"  | {pats} ->")
        ml.append("      [ " + "; ".join(f"({json.dumps(n)}, {ml_int(v)})" for n, v in vals) + " ]")
    ml += ["  | _ -> []", ""]

    # Interrupts
    ih = h[IH_CLIENTID]
    ml += ["(* Interrupts *)", ""]
    for soc in ("soc15", "soc21"):
        body = re.search(rf"enum\s+{soc}_ih_clientid\s*\{{(.*?)\}}", preprocess(ih), re.S).group(1)
        prefix = f"{soc.upper()}_IH_CLIENTID_"
        primary = [n.strip() for n, v in re.findall(r"(\w+)\s*=\s*(0x[0-9a-fA-F]+|\d+)", body)]
        values = enum_values(ih, primary)
        every = enum_values(ih, re.findall(r"(\w+)\s*=", body))
        for n in IH_CLIENTS:
            if n.startswith(prefix):
                ml.append(f"let {n.lower()} = {ml_int(every[n])}")
        ml.append(f"let {soc}_client_name = function")
        for n, v in sorted(values.items(), key=lambda x: x[1]):
            ml.append(f"  | {ml_int(v)} -> {json.dumps(n[len(prefix):])}")
        ml.append("  | _ -> \"\"")
        ml.append("")
    ml.append("(* Interrupt sources: block, source ID, name. *)")
    ml.append("let ih_sources = [")
    for hdr in IH_SOURCES:
        for m in SRCID.finditer(h[hdr]):
            ml.append(f"  ({json.dumps(m.group(1))}, {ml_int(int(m.group(3), 0))}, {json.dumps(m.group(2))});")
    ml += ["]", ""]

    # Registers
    ml += ["(* Registers *)", "",
           "(* The registers of each block but GC, at each version with headers: each",
           "   register's name, offset and segment in 32-bit words, and its fields as",
           "   (name, (lowest bit, highest bit)). *)",
           "let registers = ["]
    for block, vs in REG_FILES.items():
        for v in vs:
            regs = registers(h[reg_header(block, v, "offset")], h[reg_header(block, v, "sh_mask")],
                             REG_INVENTORY[block], HUBS.get(block))
            if not regs:
                sys.exit(f"{block} {v}: no register kept")
            ml.append(f"  ( {json.dumps(block)}, ({v[0]}, {v[1]}, {v[2]}), [")
            for n, (off, seg, fs) in sorted(regs.items()):
                f = "; ".join(f"({json.dumps(fn)}, ({lo}, {hi}))" for fn, lo, hi in fs)
                ml.append(f"      {{ Rig_amd_abi.Register.name = {json.dumps(n)}; offset = {ml_int(off)}; "
                          f"segment = {seg}; fields = [ {f} ] }};")
            ml.append("    ] );")
    ml += ["]", ""]

    # Firmware
    u = h[UCODE]
    ml += ["(* Firmware images *)", "", "(* The headers of images, laid out as the discovery table's. *)"]
    for s, wanted in UCODE_STRUCTS.items():
        struct(u, s, wanted)
    types = {**enum_values(u, PSP_FW_TYPES), **enum_values(h[PSP_GFX], GFX_FW_TYPES)}
    ml.append("(* The components of the PSP's own image, by type. *)")
    for n in PSP_FW_TYPES:
        ml.append(f"let {n.lower()} = {types[n]}")
    ml.append("")
    ml.append("(* The types the PSP loads the other images' pieces as. *)")
    for n in GFX_FW_TYPES:
        ml.append(f"let {n.lower()} = {types[n]}")
    ml.append("")
    ml.append(f"(* The pinned images of linux-firmware at {FIRMWARE_COMMIT[:12]}: each image's path,")
    ml.append("   and its BLAKE2b-256 digest. *)")
    ml.append("let pinned = [")
    paths = set()
    for row in h[FIRMWARE].splitlines():
        if row.startswith("#"):
            continue
        path, digest, url = row.split("\t")
        if url != FIRMWARE_URL + path:
            sys.exit(f"{path}: URL {url} is not in the pinned tree")
        paths.add(path)
        ml.append(f"  ({json.dumps(path)}, {json.dumps(digest)});")
    ml.append("]")
    want = {"amdgpu/" + n for n in IMAGES}
    if paths != want:
        sys.exit(f"{FIRMWARE}: extra {sorted(paths - want)}, missing {sorted(want - paths)}")
    return "\n".join(ml) + "\n"

# Command line


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--excerpt", action="store_true", help="make the excerpts from the pinned files")
    p.add_argument("--check", action="store_true", help="fail if a committed file differs")
    p.add_argument("--pin", action="store_true", help="record the digests of files not yet pinned")
    p.add_argument("--cache", type=pathlib.Path, default=pathlib.Path.home() / ".cache/raven/amd-gen")
    a = p.parse_args()
    if a.excerpt:
        pins = json.loads(PINS.read_text()) if PINS.exists() else {}
        a.cache.mkdir(parents=True, exist_ok=True)
        get = lambda url: fetch(url, a.cache, pins, a.pin)  # noqa: E731
        texts = {n: get(url).decode("latin-1") for n, url in SOURCES.items()}
        files = {HEADERS / n: excerpt(n, texts[n]) for n in [*ABI, *PCI] if n not in UAPI}
        files.update({HEADERS / n: t for n, t in uapi_excerpts({n: texts[n] for n in UAPI}).items()})
        files[HEADERS / FIRMWARE] = firmware(get)
        urls = set(SOURCES.values()) | {FIRMWARE_URL + "amdgpu/" + n for n in IMAGES}
        if a.pin:
            files[PINS] = json.dumps({u: d for u, d in pins.items() if u in urls}, indent=1, sort_keys=True) + "\n"
    else:
        h = {n: (HEADERS / n).read_text(encoding="latin-1") for n in [*ABI, *PCI, FIRMWARE]}
        files = {ABI_OUT: generate_abi(h), PCI_OUT: generate_pci(h)}
    stale = [f for f, text in files.items() if not f.exists() or f.read_text(encoding="latin-1") != text]
    # An excerpt no source names any more.
    if a.excerpt:
        stale += [f for f in HEADERS.rglob("*") if f.is_file() and f not in files]
    if a.check:
        if stale:
            sys.exit("stale: " + ", ".join(str(f.relative_to(LIB)) for f in stale))
        print("up to date")
        return
    for f in stale:
        if f not in files:
            f.unlink()
            continue
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(files[f], encoding="latin-1")


if __name__ == "__main__":
    main()
