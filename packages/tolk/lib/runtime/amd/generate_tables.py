#!/usr/bin/env python3
"""Emit the OCaml AMD hardware data tables (amd_*_defs.ml) next to this script.

Regenerate with:
  uv run packages/tolk/lib/runtime/amd/generate_tables.py

Values are read from the pinned reference clone at `_tinygrad/` (repository
root); no network access is needed. The curated symbol inventories below drive
emission: to grow coverage, add a symbol to the relevant list and rerun.
Output is deterministic; rerunning must produce byte-identical files.

Struct layouts are emitted as accessor functions over `bytes`: readers take
the buffer and the struct's byte position (`f b pos`), setters write a field
of a struct built at position 0 (`set_f b v`). Nested struct fields use
flattened absolute offsets, so a header embedded at offset 0 is read with the
embedded struct's own module. Every emitted accessor is cross-checked against
the reference ctypes layout before being written.
"""

import ctypes
import inspect
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "..", "..", "_tinygrad"))

from tinygrad import helpers  # noqa: E402
from tinygrad.runtime.autogen import amdgpu_kd, hsa  # noqa: E402
from tinygrad.runtime.autogen.am import (  # noqa: E402
    am, fw, navi_offsets, pm4_nv, pm4_soc15, regs, sdma_4_0_0, sdma_6_0_0,
    smu_13_0_0, smu_13_0_6, smu_13_0_12, smu_14_0_2, soc_9, soc_11, soc_12,
    vega_offsets,
)

# Curated inventories

# PM4 symbols shared by both packet flavors (RDNA `Nv` and gfx9 `Soc15`).
PM4_SHARED_INTS = [
    "PACKET3_NOP",
    "PACKET3_SET_SH_REG",
    "PACKET3_SET_SH_REG_START",
    "PACKET3_SET_SH_REG_END",
    "PACKET3_SET_UCONFIG_REG",
    "PACKET3_SET_UCONFIG_REG_START",
    "PACKET3_ACQUIRE_MEM",
    "PACKET3_RELEASE_MEM",
    "PACKET3_WAIT_REG_MEM",
    "PACKET3_DISPATCH_DIRECT",
    "PACKET3_EVENT_WRITE",
    "PACKET3_INDIRECT_BUFFER",
    "INDIRECT_BUFFER_VALID",
    "PACKET3_PRED_EXEC",
    "CACHE_FLUSH_AND_INV_TS_EVENT",
    "event_index__mec_release_mem__end_of_pipe",
    "data_sel__mec_release_mem__send_32_bit_low",
    "data_sel__mec_release_mem__send_64_bit_data",
    "data_sel__mec_release_mem__send_gpu_clock_counter",
    "int_sel__mec_release_mem__none",
    "int_sel__mec_release_mem__send_interrupt_after_write_confirm",
]
PM4_SHARED_ENCODERS = [
    "WAIT_REG_MEM_FUNCTION",
    "WAIT_REG_MEM_MEM_SPACE",
    "WAIT_REG_MEM_OPERATION",
    "WAIT_REG_MEM_ENGINE",
    "EVENT_TYPE",
    "EVENT_INDEX",
]
# RDNA-only: ACQUIRE_MEM GCR control and RELEASE_MEM field encoders / GCR flags.
PM4_NV_INTS = [
    "PACKET3_RELEASE_MEM_GCR_GLM_WB",
    "PACKET3_RELEASE_MEM_GCR_GLM_INV",
    "PACKET3_RELEASE_MEM_GCR_GLV_INV",
    "PACKET3_RELEASE_MEM_GCR_GL1_INV",
    "PACKET3_RELEASE_MEM_GCR_GL2_INV",
    "PACKET3_RELEASE_MEM_GCR_GL2_WB",
    "PACKET3_RELEASE_MEM_GCR_SEQ",
]
PM4_NV_ENCODERS = [
    "PACKET3_ACQUIRE_MEM_GCR_CNTL_GLI_INV",
    "PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_WB",
    "PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_INV",
    "PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_WB",
    "PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_INV",
    "PACKET3_ACQUIRE_MEM_GCR_CNTL_GLV_INV",
    "PACKET3_ACQUIRE_MEM_GCR_CNTL_GL1_INV",
    "PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_INV",
    "PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_WB",
    "PACKET3_RELEASE_MEM_EVENT_TYPE",
    "PACKET3_RELEASE_MEM_EVENT_INDEX",
    "PACKET3_RELEASE_MEM_DATA_SEL",
    "PACKET3_RELEASE_MEM_INT_SEL",
    "PACKET3_RELEASE_MEM_DST_SEL",
]
# gfx9-only: ACQUIRE_MEM CP_COHER control, RELEASE_MEM EOP flags and selects.
PM4_SOC15_INTS = [
    "EOP_TC_WB_ACTION_EN",
    "EOP_TC_NC_ACTION_EN",
]
PM4_SOC15_ENCODERS = [
    "PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_SH_ICACHE_ACTION_ENA",
    "PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_SH_KCACHE_ACTION_ENA",
    "PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TC_ACTION_ENA",
    "PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TCL1_ACTION_ENA",
    "PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TC_WB_ACTION_ENA",
    "DATA_SEL",
    "INT_SEL",
]

# SDMA symbols shared by both packet versions.
SDMA_INTS = [
    "SDMA_OP_COPY",
    "SDMA_OP_WRITE",
    "SDMA_OP_INDIRECT",
    "SDMA_OP_FENCE",
    "SDMA_OP_TRAP",
    "SDMA_OP_POLL_REGMEM",
    "SDMA_OP_TIMESTAMP",
    "SDMA_SUBOP_COPY_LINEAR",
    "SDMA_SUBOP_TIMESTAMP_GET_GLOBAL",
]
SDMA_ENCODERS = [
    "SDMA_PKT_COPY_LINEAR_HEADER_SUB_OP",
    "SDMA_PKT_COPY_LINEAR_COUNT_COUNT",
    "SDMA_PKT_INDIRECT_HEADER_VMID",
    "SDMA_PKT_POLL_REGMEM_HEADER_FUNC",
    "SDMA_PKT_POLL_REGMEM_HEADER_MEM_POLL",
    "SDMA_PKT_POLL_REGMEM_DW5_INTERVAL",
    "SDMA_PKT_POLL_REGMEM_DW5_RETRY_COUNT",
    "SDMA_PKT_TIMESTAMP_GET_HEADER_SUB_OP",
    "SDMA_PKT_TRAP_INT_CONTEXT_INT_CONTEXT",
]
# The fence-header mtype field only exists in the 6.0.0 packet format (its
# only user is guarded to non-gfx9 targets).
SDMA_V6_ONLY_ENCODERS = ["SDMA_PKT_FENCE_HEADER_MTYPE"]

# Register families: every family in the reference table set, in full. The
# reference tables are themselves curated (register-name patterns cover
# exactly what the driver-less and queue tiers touch), so the family dicts
# are emitted verbatim.

# Hardware IP discovery ids (kept here for the queue tier; the driver-less
# tier's full id tables live in amd_am_defs.ml).
IP_IDS = [
    "GC_HWID",
    "SDMA0_HWID",
    "NBIF_HWID",
    "GC_HWIP",
    "SDMA0_HWIP",
    "NBIF_HWIP",
]

# hsa_kernel_dispatch_packet_t fields whose byte offsets the runtime needs.
DISPATCH_PACKET_FIELDS = [
    "header",
    "setup",
    "workgroup_size_x",
    "workgroup_size_y",
    "workgroup_size_z",
    "grid_size_x",
    "grid_size_y",
    "grid_size_z",
    "private_segment_size",
    "group_segment_size",
    "kernel_object",
    "kernarg_address",
    "completion_signal",
]
AMD_QUEUE_FIELDS = ["read_dispatch_id", "write_dispatch_id"]
HSA_INTS = [
    "AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER",
    "AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR",
    "HSA_PACKET_HEADER_TYPE",
    "HSA_PACKET_HEADER_BARRIER",
    "HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE",
    "HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE",
    "HSA_FENCE_SCOPE_SYSTEM",
    "HSA_PACKET_TYPE_VENDOR_SPECIFIC",
    "HSA_PACKET_TYPE_KERNEL_DISPATCH",
    "HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS",
]

# llvm_amdhsa_kernel_descriptor_t fields whose byte offsets the loader reads.
KD_FIELDS = [
    "group_segment_fixed_size",
    "private_segment_fixed_size",
    "kernarg_size",
    "kernel_code_entry_byte_offset",
    "compute_pgm_rsrc3",
    "compute_pgm_rsrc1",
    "compute_pgm_rsrc2",
    "kernel_code_properties",
]

# Hardware IP ids the driver-less tier resolves discovery entries against.
AM_HWIP_IDS = [
    "GC_HWIP",
    "HDP_HWIP",
    "SDMA0_HWIP",
    "MMHUB_HWIP",
    "NBIO_HWIP",
    "NBIF_HWIP",
    "MP0_HWIP",
    "MP1_HWIP",
    "OSSSYS_HWIP",
    "MAX_HWIP",
]

# PSP bootloader components, in load order.
PSP_BL_IDS = [
    "PSP_BL__LOAD_KEY_DATABASE",
    "PSP_BL__LOAD_TOS_SPL_TABLE",
    "PSP_BL__LOAD_SYSDRV",
    "PSP_BL__LOAD_SOCDRV",
    "PSP_BL__LOAD_INTFDRV",
    "PSP_BL__LOAD_DBGDRV",
    "PSP_BL__LOAD_RASDRV",
    "PSP_BL__LOAD_SOSDRV",
]
PSP_FW_TYPE_IDS = [
    "PSP_FW_TYPE_PSP_KDB",
    "PSP_FW_TYPE_PSP_SPL",
    "PSP_FW_TYPE_PSP_SYS_DRV",
    "PSP_FW_TYPE_PSP_SOC_DRV",
    "PSP_FW_TYPE_PSP_INTF_DRV",
    "PSP_FW_TYPE_PSP_DBG_DRV",
    "PSP_FW_TYPE_PSP_RAS_DRV",
    "PSP_FW_TYPE_PSP_SOS",
    "PSP_FW_TYPE_PSP_TOC",
    "PSP_FW_TYPE_PSP_RL",
]
GFX_FW_TYPE_IDS = [
    "GFX_FW_TYPE_SMU",
    "GFX_FW_TYPE_P2S_TABLE",
    "GFX_FW_TYPE_SDMA0",
    "GFX_FW_TYPE_SDMA1",
    "GFX_FW_TYPE_SDMA2",
    "GFX_FW_TYPE_SDMA3",
    "GFX_FW_TYPE_SDMA_UCODE_TH0",
    "GFX_FW_TYPE_SDMA_UCODE_TH1",
    "GFX_FW_TYPE_CP_PFP",
    "GFX_FW_TYPE_CP_ME",
    "GFX_FW_TYPE_CP_MEC",
    "GFX_FW_TYPE_CP_MEC_ME1",
    "GFX_FW_TYPE_RS64_PFP",
    "GFX_FW_TYPE_RS64_ME",
    "GFX_FW_TYPE_RS64_MEC",
    "GFX_FW_TYPE_RS64_PFP_P0_STACK",
    "GFX_FW_TYPE_RS64_ME_P0_STACK",
    "GFX_FW_TYPE_RS64_MEC_P0_STACK",
    "GFX_FW_TYPE_IMU_I",
    "GFX_FW_TYPE_IMU_D",
    "GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_CNTL",
    "GFX_FW_TYPE_RLC_RESTORE_LIST_GPM_MEM",
    "GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_MEM",
    "GFX_FW_TYPE_RLC_IRAM",
    "GFX_FW_TYPE_RLC_DRAM_BOOT",
    "GFX_FW_TYPE_RLC_P",
    "GFX_FW_TYPE_RLC_V",
    "GFX_FW_TYPE_RLC_G",
    "GFX_FW_TYPE_REG_LIST",
]
GFX_CMD_IDS = [
    "GFX_CMD_ID_SETUP_TMR",
    "GFX_CMD_ID_LOAD_IP_FW",
    "GFX_CMD_ID_LOAD_TOC",
    "GFX_CMD_ID_AUTOLOAD_RLC",
    "GFX_CMD_ID_SRIOV_SPATIAL_PART",
]
PSP_INTS = [
    "PSP_1_MEG",
    "PSP_CMD_BUFFER_SIZE",
    "PSP_FENCE_BUFFER_SIZE",
    "PSP_TMR_ALIGNMENT",
    "PSP_RING_TYPE__KM",
    "GFX_CTRL_CMD_ID_DESTROY_RINGS",
]

# Page-table flags: 64-bit values (several use bit 63), emitted as int64.
PTE_FLAG_IDS = [
    "AMDGPU_PTE_VALID",
    "AMDGPU_PTE_SYSTEM",
    "AMDGPU_PTE_SNOOPED",
    "AMDGPU_PTE_EXECUTABLE",
    "AMDGPU_PTE_READABLE",
    "AMDGPU_PTE_WRITEABLE",
    "AMDGPU_PDE_PTE",
    "AMDGPU_PTE_TF",
    "AMDGPU_PTE_IS_PTE",
    "AMDGPU_PDE_PTE_GFX12",
]
VM_LEVEL_IDS = ["AMDGPU_VM_PDB2", "AMDGPU_VM_PDB1", "AMDGPU_VM_PDB0", "AMDGPU_VM_PTB"]
DOORBELL_IDS = ["AMDGPU_NAVI10_DOORBELL_MEC_RING0", "AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE0"]

# Interrupt-ring entry decoders: (name, dword index, shift, mask or None).
IH_ENTRY_FIELDS = [
    ("client_id", 0, 0, 0xFF),
    ("source_id", 0, 8, 0xFF),
    ("ring_id", 0, 16, 0xFF),
    ("vmid", 0, 24, 0xF),
    ("vmid_type", 0, 31, 0x1),
    ("pasid", 3, 0, 0xFFFF),
    ("nodeid", 3, 16, 0xFF),
    ("context_id0", 4, 0, None),
    ("context_id1", 5, 0, None),
    ("context_id2", 6, 0, None),
    ("context_id3", 7, 0, None),
]
SOC15_CLIENT_IDS = (
    ["SOC15_IH_CLIENTID_GRBM_CP", "SOC15_IH_CLIENTID_UTCL2"]
    + [f"SOC15_IH_CLIENTID_SE{i}SH" for i in range(4)]
    + [f"SOC15_IH_CLIENTID_SDMA{i}" for i in range(8)]
)
SOC21_CLIENT_IDS = ["SOC21_IH_CLIENTID_GRBM_CP", "SOC21_IH_CLIENTID_GFX"]
# Interrupt source-id name tables, one per (block prefix, ip major).
IH_SRCID_PREFIXES = [
    ("gfx_9", "GFX_9"),
    ("gfx_11", "GFX_11"),
    ("gfx_12", "GFX_12"),
    ("sdma0_4", "SDMA0_4"),
    ("sdma0_5", "SDMA0_5"),
]

# MQD fields the queue-bringup path writes. gfx9's se4..se7 dwords are
# repurposed as the multi-XCC fields, so the v9 list carries those instead.
MQD_COMMON_FIELDS = [
    "header",
    "cp_mqd_base_addr_lo",
    "cp_mqd_base_addr_hi",
    "cp_hqd_vmid",
    "cp_hqd_persistent_state",
    "cp_hqd_pipe_priority",
    "cp_hqd_queue_priority",
    "cp_hqd_quantum",
    "cp_hqd_pq_base_lo",
    "cp_hqd_pq_base_hi",
    "cp_hqd_pq_rptr_report_addr_lo",
    "cp_hqd_pq_rptr_report_addr_hi",
    "cp_hqd_pq_wptr_poll_addr_lo",
    "cp_hqd_pq_wptr_poll_addr_hi",
    "cp_hqd_pq_doorbell_control",
    "cp_hqd_pq_control",
    "cp_hqd_ib_control",
    "cp_hqd_hq_status0",
    "cp_mqd_control",
    "cp_hqd_aql_control",
    "cp_hqd_eop_base_addr_lo",
    "cp_hqd_eop_base_addr_hi",
    "cp_hqd_eop_control",
]
MQD_V9_FIELDS = MQD_COMMON_FIELDS + [
    f"compute_static_thread_mgmt_se{i}" for i in range(4)
] + ["compute_tg_chunk_size", "compute_current_logic_xcc_id", "cp_mqd_stride_size"]
MQD_V1X_FIELDS = MQD_COMMON_FIELDS + [
    f"compute_static_thread_mgmt_se{i}" for i in range(8)
]

# SMU message and clock-domain ids. A symbol present in every emitted SMU
# table is an int; one missing from some tables is an int option, matching
# the version-guarded call sites.
SMU_MODULES = [
    ("V13_0_0", smu_13_0_0),
    ("V13_0_6", smu_13_0_6),
    ("V13_0_12", smu_13_0_12),
    ("V14_0_2", smu_14_0_2),
]
SMU_IDS = [
    "PPSMC_MSG_SetDriverDramAddrHigh",
    "PPSMC_MSG_SetDriverDramAddrLow",
    "PPSMC_MSG_EnableAllSmuFeatures",
    "PPSMC_MSG_GetSmuVersion",
    "PPSMC_MSG_Mode1Reset",
    "PPSMC_MSG_GfxDriverReset",
    "PPSMC_MSG_TransferTableSmu2Dram",
    "PPSMC_MSG_GetMetricsTable",
    "PPSMC_MSG_GetDpmFreqByIndex",
    "PPSMC_MSG_SetSoftMinByFreq",
    "PPSMC_MSG_SetSoftMaxByFreq",
    "PPSMC_MSG_SetPptLimit",
    "PPSMC_MSG_QueryValidMcaCount",
    "PPSMC_MSG_QueryValidMcaCeCount",
    "PPSMC_MSG_McaBankDumpDW",
    "PPSMC_MSG_McaBankCeDumpDW",
    "PPCLK_UCLK",
    "PPCLK_FCLK",
    "PPCLK_SOCCLK",
    "PPCLK_GFXCLK",
]

HEADER = """\
(* Generated by generate_tables.py; do not edit by hand.
   Regenerate with: uv run packages/tolk/lib/runtime/amd/generate_tables.py *)
"""

ENCODER_PROBES = [0, 1, 2, 3, 5, 0x7F, 0xFF, 0x1234, 0xFFFFF, 0xFFFFFFFF]


def ml_int(v):
    if v < 0:
        raise ValueError(f"negative constant {v}")
    return str(v) if v < 10 else f"0x{v:x}"


def ml_int64(v):
    if not 0 <= v < 1 << 64:
        raise ValueError(f"constant out of 64-bit range {v}")
    return f"0x{v:x}L"


def int_let(name, v):
    return f"let {name.lower()} = {ml_int(v)}"


def encoder_let(name, fn):
    """Recover `(x [land mask]) lsl shift` from a field-encoder lambda.

    The recovered form is verified against the lambda on a probe set; any
    encoder that does not fit the shape aborts generation.
    """
    if fn(0) != 0:
        raise ValueError(f"{name}: fn(0) != 0")
    shift = fn(1).bit_length() - 1
    if fn(1) != 1 << shift:
        raise ValueError(f"{name}: fn(1) is not a power of two")
    if fn(-1) < 0:
        if not all(fn(v) == v << shift for v in ENCODER_PROBES):
            raise ValueError(f"{name}: does not fit `x lsl {shift}`")
        return f"let {name.lower()} x = x lsl {shift}"
    mask = fn(-1) >> shift
    if not all(fn(v) == (v & mask) << shift for v in ENCODER_PROBES):
        raise ValueError(f"{name}: does not fit `(x land {mask:#x}) lsl {shift}`")
    return f"let {name.lower()} x = (x land 0x{mask:x}) lsl {shift}"


def packet3_let(mod):
    got = "let packet3 op count = (3 lsl 30) lor ((op land 0xff) lsl 8) lor ((count land 0x3fff) lsl 16)"
    for op, count in [(0, 0), (0x10, 1), (0x15, 2), (0x3C, 5), (0xFF, 0x3FFF), (0x46, 0x2000)]:
        want = (3 << 30) | ((op & 0xFF) << 8) | ((count & 0x3FFF) << 16)
        if mod.PACKET3(op, count) != want:
            raise ValueError(f"{mod.__name__}: PACKET3({op}, {count}) mismatch")
    return got


def pm4_module(title, mod, ints, encoders):
    lines = [f"module {title} = struct", f"  {packet3_let(mod)}"]
    lines += [f"  {int_let(nm, getattr(mod, nm))}" for nm in ints]
    lines += [f"  {encoder_let(nm, getattr(mod, nm))}" for nm in encoders]
    lines.append("end")
    return lines


def sdma_module(title, mod, extra_encoders):
    lines = [f"module {title} = struct"]
    lines += [f"  {int_let(nm, getattr(mod, nm))}" for nm in SDMA_INTS]
    lines += [f"  {encoder_let(nm, getattr(mod, nm))}" for nm in SDMA_ENCODERS + extra_encoders]
    lines.append("end")
    return lines


def reg_entry(fam, reg_name):
    off, seg, fields = fam[reg_name]
    parts = [
        f'("{fname}", ({lo}, {hi}))'
        for fname, (lo, hi) in sorted(fields.items(), key=lambda kv: (kv[1], kv[0]))
    ]
    body = f'[ {"; ".join(parts)} ]' if parts else "[]"
    return f'  ("{reg_name}", ({ml_int(off)}, {seg}, {body}));'


def bases_rows(mod, block, nseg):
    rows = []
    for inst in range(6):
        segs = [getattr(mod, f"{block}_BASE__INST{inst}_SEG{s}", 0) for s in range(nseg)]
        rows.append("    [| " + "; ".join(ml_int(v) for v in segs) + " |];")
    return rows


def bitfield_spec(cls, fld):
    """(shift, width) of a struct bitfield, cross-checked by assignment."""
    name, _typ, off, bit_width, bit_off = next(
        f for f in cls._real_fields_ if f[0] == fld
    )
    shift = off * 8 + bit_off
    for probe in (1, (1 << bit_width) - 1):
        if int.from_bytes(bytes(cls(**{name: probe})), "little") != probe << shift:
            raise ValueError(f"{cls.__name__}.{fld}: bitfield probe mismatch")
    return shift, bit_width


def gen_pm4():
    lines = [HEADER]
    lines += pm4_module("Nv", pm4_nv, PM4_SHARED_INTS + PM4_NV_INTS,
                        PM4_SHARED_ENCODERS + PM4_NV_ENCODERS)
    lines.append("")
    lines += pm4_module("Soc15", pm4_soc15, PM4_SHARED_INTS + PM4_SOC15_INTS,
                        PM4_SHARED_ENCODERS + PM4_SOC15_ENCODERS)
    return lines


def gen_sdma():
    lines = [HEADER]
    lines += sdma_module("V4_0_0", sdma_4_0_0, [])
    lines.append("")
    lines += sdma_module("V6_0_0", sdma_6_0_0, SDMA_V6_ONLY_ENCODERS)
    return lines


def family_version(value):
    parts = value.split("_")
    split = next(i for i, p in enumerate(parts) if p.isdigit())
    return "_".join(parts[:split]), tuple(map(int, parts[split:]))


def gen_regs():
    lines = [HEADER]
    lines.append("(* Per-family register maps: name -> (offset, segment, fields as (lo, hi)). *)")
    fam_values = []
    for value in regs.__all__:
        prefix, ver = family_version(value)
        fam_values.append((prefix, ver, value))
        fam = getattr(regs, value)
        lines.append(f"let {value} = [")
        lines += [reg_entry(fam, nm) for nm in fam]
        lines.append("]")
        lines.append("")
    lines.append("let families = [")
    lines += [
        f'  ("{prefix}", ({v0}, {v1}, {v2}), {value});'
        for prefix, (v0, v1, v2), value in fam_values
    ]
    lines.append("]")
    lines.append("")
    lines.append("(* Address-space segment bases, per die instance. *)")
    for title, mod in [("Navi", navi_offsets), ("Vega", vega_offsets)]:
        lines.append(f"module {title} = struct")
        lines.append("  let gc_bases = [|")
        lines += bases_rows(mod, "GC", 6)
        lines.append("  |]")
        lines.append("")
        lines.append("  let nbio_bases = [|")
        lines += bases_rows(mod, "NBIO", 9)
        lines.append("  |]")
        lines.append("end")
        lines.append("")
    lines.append("(* Hardware IP discovery ids. *)")
    lines += [int_let(nm, getattr(am, nm)) for nm in IP_IDS]
    return lines


def gen_soc():
    lines = [HEADER]
    for title, mod in [("Soc_9", soc_9), ("Soc_11", soc_11), ("Soc_12", soc_12)]:
        lines.append(f"module {title} = struct")
        lines.append(f"  {int_let('cs_partial_flush', mod.CS_PARTIAL_FLUSH)}")
        lines.append(f"  {int_let('mtype_uc', mod.MTYPE_UC)}")
        lines.append("end")
        lines.append("")
    return lines[:-1]


def gen_hsa():
    lines = [HEADER]
    lines.append("(* hsa_kernel_dispatch_packet_t: byte offsets and total size. *)")
    lines.append("module Kernel_dispatch_packet = struct")
    lines.append(f"  let size = {ml_int(ctypes.sizeof(hsa.hsa_kernel_dispatch_packet_t))}")
    lines += [
        f"  let {nm} = {ml_int(getattr(hsa.hsa_kernel_dispatch_packet_t, nm).offset)}"
        for nm in DISPATCH_PACKET_FIELDS
    ]
    lines.append("end")
    lines.append("")
    lines.append("(* amd_queue_t: byte offsets. *)")
    lines.append("module Amd_queue = struct")
    lines += [
        f"  let {nm} = {ml_int(getattr(hsa.amd_queue_t, nm).offset)}"
        for nm in AMD_QUEUE_FIELDS
    ]
    lines.append("end")
    lines.append("")
    lines.append("(* COMPUTE_TMPRING_SIZE bitfields as (shift, width), per target generation. *)")
    lines.append("module Compute_tmpring_size = struct")
    for title, suffix in [("gfx9", ""), ("gfx11", "_GFX11"), ("gfx12", "_GFX12")]:
        cls = getattr(hsa, f"union_COMPUTE_TMPRING_SIZE{suffix}_bitfields")
        for fld in ["WAVES", "WAVESIZE"]:
            shift, width = bitfield_spec(cls, fld)
            lines.append(f"  let {title}_{fld.lower()} = ({shift}, {width})")
    lines.append("end")
    lines.append("")
    lines += [int_let(nm, getattr(hsa, nm)) for nm in HSA_INTS]
    return lines


def gen_kd():
    lines = [HEADER]
    lines.append("(* llvm_amdhsa_kernel_descriptor_t: byte offsets and total size. *)")
    kd = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t
    lines.append(f"let size = {ml_int(ctypes.sizeof(kd))}")
    lines += [f"let {nm} = {ml_int(getattr(kd, nm).offset)}" for nm in KD_FIELDS]
    return lines


# Struct accessor emission (driver-less tier)

READERS = {1: "g8", 2: "g16", 4: "g32"}
WRITERS = {1: "s8", 2: "s16", 4: "s32"}

ACCESSOR_PRELUDE = [
    "(* Little-endian byte accessors used by the struct modules below. *)",
    "let g8 = Bytes.get_uint8",
    "let g16 = Bytes.get_uint16_le",
    "let g32 b p = Int32.to_int (Bytes.get_int32_le b p) land 0xffffffff",
    "let s8 b p v = Bytes.set_uint8 b p (v land 0xff)",
    "let s16 b p v = Bytes.set_uint16_le b p (v land 0xffff)",
    "let s32 b p v = Bytes.set_int32_le b p (Int32.of_int v)",
]


def scalar_field(cls, fld):
    """(offset, byte size, bit spec or None) for a scalar field, probe-checked."""
    entry = next(f for f in cls._real_fields_ if f[0] == fld)
    if len(entry) == 5:
        name, typ, off, bit_width, bit_off = entry
        size = ctypes.sizeof(typ)
        for probe in (1, (1 << bit_width) - 1):
            raw = bytes(cls(**{name: probe}))
            if int.from_bytes(raw, "little") != probe << (off * 8 + bit_off):
                raise ValueError(f"{cls.__name__}.{fld}: bitfield probe mismatch")
        return off, size, (bit_off, bit_width)
    name, typ, off = entry
    if issubclass(typ, (ctypes.Structure, ctypes.Array)):
        raise ValueError(f"{cls.__name__}.{fld}: not a scalar field")
    size = ctypes.sizeof(typ)
    probe = (1 << (8 * size)) - 1
    raw = bytes(cls(**{name: probe}))
    want = b"\x00" * off + probe.to_bytes(size, "little") + b"\x00" * (cls.SIZE - off - size)
    if raw != want:
        raise ValueError(f"{cls.__name__}.{fld}: scalar probe mismatch")
    return off, size, None


def reader_lines(cls, fields, base=0, indent="  ", rename=None):
    lines = []
    for fld in fields:
        nested = None
        if isinstance(fld, tuple):
            fld, nested, extra = fld
        off, size, bits = scalar_field(nested or cls, fld)
        if nested is not None:
            off += extra
        off += base
        name = (rename or {}).get(fld, fld).lower()
        if bits is None:
            lines.append(f"{indent}let {name} b pos = {READERS[size]} b (pos + {ml_int(off)})")
        else:
            bit_off, width = bits
            expr = f"{READERS[size]} b (pos + {ml_int(off)})"
            if bit_off:
                expr = f"({expr} lsr {bit_off})"
            lines.append(f"{indent}let {name} b pos = {expr} land 0x{(1 << width) - 1:x}")
    return lines


def writer_lines(cls, fields, base=0, indent="  "):
    lines = []
    for fld in fields:
        nested = None
        if isinstance(fld, tuple):
            fld, nested, extra = fld
        off, size, bits = scalar_field(nested or cls, fld)
        if nested is not None:
            off += extra
        off += base
        if bits is None:
            lines.append(f"{indent}let set_{fld.lower()} b v = {WRITERS[size]} b {ml_int(off)} v")
        else:
            bit_off, width = bits
            mask = ((1 << width) - 1) << bit_off
            clear = ((1 << (8 * size)) - 1) ^ mask
            lines.append(
                f"{indent}let set_{fld.lower()} b v = {WRITERS[size]} b {ml_int(off)} "
                f"(({READERS[size]} b {ml_int(off)} land 0x{clear:x}) lor ((v land 0x{(1 << width) - 1:x}) lsl {bit_off}))"
            )
    return lines


def own_scalar_fields(cls):
    """The struct's own scalar fields, plus (name, offset) of its arrays."""
    scalars, arrays = [], []
    for entry in cls._real_fields_:
        name, typ = entry[0], entry[1]
        if name.startswith(("reserved", "padding")):
            continue
        if issubclass(typ, ctypes.Array):
            arrays.append((name, entry[2]))
        elif issubclass(typ, ctypes.Structure):
            continue
        else:
            scalars.append(name)
    return scalars, arrays


def reader_struct(title, cls, fields=None, extra=()):
    arrays = []
    if fields is None:
        fields, arrays = own_scalar_fields(cls)
    lines = [f"module {title} = struct", f"  let sizeof = {ml_int(cls.SIZE)}"]
    lines += [f"  let {name}_offset = {ml_int(off)}" for name, off in arrays]
    lines += reader_lines(cls, list(fields) + list(extra))
    lines.append("end")
    return lines


def gen_am():
    lines = [HEADER]
    lines += ACCESSOR_PRELUDE
    lines.append("")

    lines.append("(* IP discovery: signatures, table ids, hardware ids. *)")
    lines.append(int_let("binary_signature", am.BINARY_SIGNATURE))
    lines.append(int_let("discovery_table_signature", am.DISCOVERY_TABLE_SIGNATURE))
    lines.append(int_let("table_ip_discovery", am.IP_DISCOVERY))
    lines.append(int_let("table_gc", am.GC))
    lines += [int_let(nm, getattr(am, nm)) for nm in AM_HWIP_IDS]
    lines.append("")
    lines.append("(* Hardware IP id -> discovery hardware id. *)")
    lines.append("let hw_id_map = [")
    lines += [f"  ({ml_int(hwip)}, {ml_int(hwid)});" for hwip, hwid in am.hw_id_map.items()]
    lines.append("]")
    lines.append("")
    lines.append("(* Discovery hardware id -> block name. *)")
    hwid_names = {
        v: k.removesuffix("_HWID")
        for k, v in vars(am).items()
        if k.endswith("_HWID") and isinstance(v, int)
    }
    lines.append("let hwid_names = [")
    lines += [f'  ({ml_int(v)}, "{nm}");' for v, nm in hwid_names.items()]
    lines.append("]")
    lines.append("")

    lines.append("(* Discovery table layouts. *)")
    for title, cls in [
        ("Table_info", am.struct_table_info),
        ("Binary_header", am.struct_binary_header),
        ("Die_info", am.struct_die_info),
        ("Die_header", am.struct_die_header),
        ("Ip_v4", am.struct_ip_v4),
        ("Ip_discovery_header", am.struct_ip_discovery_header),
        ("Gpu_info_header", am.struct_gpu_info_header),
    ]:
        lines += reader_struct(title, cls)
        lines.append("")
    gc_info_v1 = [
        "gc_num_se", "gc_num_wgp0_per_sa", "gc_num_wgp1_per_sa", "gc_num_sa_per_se",
        "gc_max_scratch_slots_per_cu", "gc_max_waves_per_simd", "gc_lds_size",
    ]
    gc_info_v2 = [
        "gc_num_se", "gc_num_cu_per_sh", "gc_num_sh_per_se",
        "gc_max_scratch_slots_per_cu", "gc_max_waves_per_simd", "gc_lds_size",
    ]
    for minor in range(4):
        lines += reader_struct(f"Gc_info_v1_{minor}", getattr(am, f"struct_gc_info_v1_{minor}"), gc_info_v1)
        lines.append("")
    for minor in range(2):
        lines += reader_struct(f"Gc_info_v2_{minor}", getattr(am, f"struct_gc_info_v2_{minor}"), gc_info_v2)
        lines.append("")

    lines.append("(* Firmware image headers. *)")
    for title, cls in [
        ("Common_firmware_header", am.struct_common_firmware_header),
        ("Psp_firmware_header_v2_0", am.struct_psp_firmware_header_v2_0),
        ("Psp_firmware_header_v2_1", am.struct_psp_firmware_header_v2_1),
        ("Psp_fw_bin_desc", am.struct_psp_fw_bin_desc),
        ("Smc_firmware_header_v1_0", am.struct_smc_firmware_header_v1_0),
        ("Smc_firmware_header_v2_0", am.struct_smc_firmware_header_v2_0),
        ("Smc_firmware_header_v2_1", am.struct_smc_firmware_header_v2_1),
        ("Smc_soft_pptable_entry", am.struct_smc_soft_pptable_entry),
        ("Sdma_firmware_header_v1_0", am.struct_sdma_firmware_header_v1_0),
        ("Sdma_firmware_header_v2_0", am.struct_sdma_firmware_header_v2_0),
        ("Sdma_firmware_header_v3_0", am.struct_sdma_firmware_header_v3_0),
        ("Gfx_firmware_header_v1_0", am.struct_gfx_firmware_header_v1_0),
        ("Gfx_firmware_header_v2_0", am.struct_gfx_firmware_header_v2_0),
        ("Imu_firmware_header_v1_0", am.struct_imu_firmware_header_v1_0),
        ("Rlc_firmware_header_v2_0", am.struct_rlc_firmware_header_v2_0),
        ("Rlc_firmware_header_v2_1", am.struct_rlc_firmware_header_v2_1),
        ("Rlc_firmware_header_v2_2", am.struct_rlc_firmware_header_v2_2),
        ("Rlc_firmware_header_v2_3", am.struct_rlc_firmware_header_v2_3),
    ]:
        lines += reader_struct(title, cls)
        lines.append("")

    lines.append("(* PSP interface constants. *)")
    lines += [int_let(nm, getattr(am, nm)) for nm in PSP_INTS + PSP_BL_IDS + PSP_FW_TYPE_IDS + GFX_CMD_IDS + GFX_FW_TYPE_IDS]
    lines.append("")
    for value_name, enum in [("psp_fw_type_names", am.enum_psp_fw_type),
                             ("psp_gfx_fw_type_names", am.enum_psp_gfx_fw_type)]:
        lines.append(f"let {value_name} = [")
        lines += [f'  ({ml_int(v)}, "{nm}");' for v, nm in enum.items()]
        lines.append("]")
        lines.append("")

    lines.append("(* PSP ring protocol layouts. *)")
    cmd = am.struct_psp_gfx_cmd_resp
    cmd_off = next(f[2] for f in cmd._real_fields_ if f[0] == "cmd")
    resp_off = next(f[2] for f in cmd._real_fields_ if f[0] == "resp")
    lines.append("module Psp_gfx_cmd_resp = struct")
    lines.append(f"  let sizeof = {ml_int(cmd.SIZE)}")
    lines += writer_lines(cmd, ["cmd_id"])
    lines += reader_lines(am.struct_psp_gfx_resp, ["status", "tmr_size"], base=resp_off,
                          rename={"status": "resp_status", "tmr_size": "resp_tmr_size"})
    for title, arm, fields in [
        ("Cmd_load_ip_fw", am.struct_psp_gfx_cmd_load_ip_fw,
         ["fw_phy_addr_lo", "fw_phy_addr_hi", "fw_size", "fw_type"]),
        ("Cmd_setup_tmr", am.struct_psp_gfx_cmd_setup_tmr,
         ["buf_phy_addr_lo", "buf_phy_addr_hi", "buf_size",
          ("virt_phy_addr", am.struct_psp_gfx_cmd_setup_tmr_bitfield,
           next(f[2] for f in am.struct_psp_gfx_cmd_setup_tmr._real_fields_ if f[0] == "bitfield")),
          "system_phy_addr_lo", "system_phy_addr_hi"]),
        ("Cmd_load_toc", am.struct_psp_gfx_cmd_load_toc,
         ["toc_phy_addr_lo", "toc_phy_addr_hi", "toc_size"]),
        ("Cmd_spatial_part", am.struct_psp_gfx_cmd_sriov_spatial_part, ["mode"]),
    ]:
        lines.append(f"  module {title} = struct")
        lines += writer_lines(arm, fields, base=cmd_off, indent="    ")
        lines.append("  end")
    lines.append("end")
    lines.append("")
    lines.append("module Psp_gfx_rb_frame = struct")
    lines.append(f"  let sizeof = {ml_int(am.struct_psp_gfx_rb_frame.SIZE)}")
    lines += writer_lines(am.struct_psp_gfx_rb_frame,
                          ["cmd_buf_addr_lo", "cmd_buf_addr_hi", "fence_addr_lo", "fence_addr_hi", "fence_value"])
    lines.append("end")
    lines.append("")

    lines.append("(* Page-table entry flags (64-bit words; several flags use bit 63). *)")
    for nm in PTE_FLAG_IDS:
        lines.append(f"let {nm.lower()} = {ml_int64(getattr(am, nm))}")
    for nm, fn, mask, shift in [
        ("amdgpu_pte_frag", am.AMDGPU_PTE_FRAG, 0x1F, 7),
        ("amdgpu_pde_bfs", am.AMDGPU_PDE_BFS, None, 59),
    ]:
        for probe in ENCODER_PROBES[:8]:
            want = ((probe & mask) if mask is not None else probe) << shift
            if fn(probe) != want:
                raise ValueError(f"{nm}: probe mismatch")
        arg = f"(x land 0x{mask:x})" if mask is not None else "x"
        if shift + (mask or 0x1F).bit_length() < 62 and mask is not None:
            lines.append(f"let {nm} x = Int64.of_int ({arg} lsl {shift})")
        else:
            lines.append(f"let {nm} x = Int64.shift_left (Int64.of_int {arg}) {shift}")
    for nm, gen in [("vg10", "VG10"), ("nv10", "NV10"), ("gfx12", "GFX12")]:
        fn = getattr(am, f"AMDGPU_PTE_MTYPE_{gen}")
        mask = getattr(am, f"AMDGPU_PTE_MTYPE_{gen}_MASK")
        shift = getattr(am, f"AMDGPU_PTE_MTYPE_{gen}_SHIFT")(1).bit_length() - 1
        max_mtype = mask >> shift
        for flags in (0, 0xDEAD_BEEF_0000, (1 << 64) - 1):
            for mtype in (0, 1, max_mtype):
                if fn(flags, mtype) & ((1 << 64) - 1) != ((flags & ~mask) | (mtype << shift)) & ((1 << 64) - 1):
                    raise ValueError(f"mtype {gen}: probe mismatch")
        keep = (~mask) & ((1 << 64) - 1)
        lines.append(
            f"let amdgpu_pte_mtype_{nm} flags mtype = "
            f"Int64.logor (Int64.logand flags {ml_int64(keep)}) (Int64.shift_left (Int64.of_int mtype) {shift})"
        )
    lines.append("")
    lines.append("(* Page-table levels. *)")
    lines += [int_let(nm, getattr(am, nm)) for nm in VM_LEVEL_IDS]
    lines.append("")
    lines.append("(* Doorbell assignments. *)")
    lines += [int_let(nm, getattr(am, nm)) for nm in DOORBELL_IDS]
    lines.append("")

    lines.append("(* Interrupt-ring entries: decoders over the 8 dwords of an entry. *)")
    for name, idx, shift, mask in IH_ENTRY_FIELDS:
        fn = getattr(am, f"SOC15_{name.upper()}_FROM_IH_ENTRY")
        for probe in (0x12345678, 0xFFFFFFFF, 0x80000001):
            entry = [0x1000 + i for i in range(8)]
            entry[idx] = probe
            want = (probe >> shift) & mask if mask is not None else probe
            if fn(entry) != want:
                raise ValueError(f"ih {name}: probe mismatch")
        expr = f"e.({idx})"
        if shift:
            expr = f"({expr} lsr {shift})"
        if mask is not None:
            expr = f"{expr} land 0x{mask:x}"
        lines.append(f"let soc15_{name}_from_ih_entry (e : int array) = {expr}")
    lines.append("")
    lines.append("(* Interrupt client ids. *)")
    lines += [int_let(nm, getattr(am, nm)) for nm in SOC15_CLIENT_IDS + SOC21_CLIENT_IDS]
    lines.append("")
    for value_name, enum in [("soc15_ih_clientid_names", am.enum_soc15_ih_clientid),
                             ("soc21_ih_clientid_names", am.enum_soc21_ih_clientid)]:
        lines.append(f"let {value_name} = [")
        lines += [f'  ({ml_int(v)}, "{nm}");' for v, nm in enum.items()]
        lines.append("]")
        lines.append("")
    lines.append("(* Interrupt source ids, per block generation. *)")
    for value_name, prefix in IH_SRCID_PREFIXES:
        srcs = {}
        for k in dir(am):
            if k.startswith(prefix) and (off := k.find("__SRCID__")) != -1:
                srcs[getattr(am, k)] = k[off + 9:]
        lines.append(f"let {value_name}_srcids = [")
        lines += [f'  ({ml_int(v)}, "{nm}");' for v, nm in srcs.items()]
        lines.append("]")
        lines.append("")

    lines.append("(* Memory queue descriptors: setters for the queue-bringup fields. *)")
    for title, cls, fields in [
        ("V9_mqd", am.struct_v9_mqd, MQD_V9_FIELDS),
        ("V11_compute_mqd", am.struct_v11_compute_mqd, MQD_V1X_FIELDS),
        ("V12_compute_mqd", am.struct_v12_compute_mqd, MQD_V1X_FIELDS),
    ]:
        lines.append(f"module {title} = struct")
        lines.append(f"  let sizeof = {ml_int(cls.SIZE)}")
        lines += writer_lines(cls, fields)
        lines.append("end")
        lines.append("")
    return lines[:-1]


def gen_smu():
    lines = [HEADER]
    lines.append("(* SMU message and clock-domain ids, per firmware interface version.")
    lines.append("   Symbols absent from some versions are int options. *)")
    universal = {nm for nm in SMU_IDS if all(hasattr(mod, nm) for _, mod in SMU_MODULES)}
    for title, mod in SMU_MODULES:
        lines.append(f"module {title} = struct")
        for nm in SMU_IDS:
            if nm in universal:
                lines.append(f"  {int_let(nm, getattr(mod, nm))}")
            elif hasattr(mod, nm):
                lines.append(f"  let {nm.lower()} : int option = Some {ml_int(getattr(mod, nm))}")
            else:
                lines.append(f"  let {nm.lower()} : int option = None")
        lines.append("end")
        lines.append("")
    return lines[:-1]


def gen_fw():
    pin = re.search(r"kernel-firmware/linux-firmware/-/raw/([0-9a-f]{40})/",
                    inspect.getsource(helpers.fetch_fw)).group(1)
    lines = [HEADER]
    lines.append(f"(* Firmware file sha256 digests, for the linux-firmware pin {pin}. *)")
    lines.append("let hashes = [")
    for name, digest in fw.hashes.items():
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{name}: malformed digest")
        lines.append(f'  ("{name}", "{digest}");')
    lines.append("]")
    lines.append("")
    lines.append("(* The pinned linux-firmware tree the digests were taken from. *)")
    lines.append("let upstream =")
    lines.append('  "https://gitlab.com/kernel-firmware/linux-firmware/-/raw/'
                 f'{pin}/amdgpu"')
    return lines


def main():
    outputs = {
        "amd_pm4_defs.ml": gen_pm4,
        "amd_sdma_defs.ml": gen_sdma,
        "amd_regs_defs.ml": gen_regs,
        "amd_soc_defs.ml": gen_soc,
        "amd_hsa_defs.ml": gen_hsa,
        "amd_kd_defs.ml": gen_kd,
        "amd_am_defs.ml": gen_am,
        "amd_smu_defs.ml": gen_smu,
        "amd_fw_defs.ml": gen_fw,
    }
    for fname, gen in outputs.items():
        text = "\n".join(gen()) + "\n"
        with open(os.path.join(_HERE, fname), "w") as f:
            f.write(text)
        print(f"wrote {fname} ({text.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
