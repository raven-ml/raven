#!/usr/bin/env python3
"""Emit the OCaml AMD hardware data tables (amd_*_defs.ml) next to this script.

Regenerate with:
  uv run packages/tolk/lib/runtime/amd/generate_tables.py

Values are read from the pinned reference clone at `_tinygrad/` (repository
root); no network access is needed. The curated symbol inventories below drive
emission: to grow coverage, add a symbol to the relevant list and rerun.
Output is deterministic; rerunning must produce byte-identical files.
"""

import ctypes
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "..", "..", "_tinygrad"))

from tinygrad.runtime.autogen import amdgpu_kd, hsa  # noqa: E402
from tinygrad.runtime.autogen.am import (  # noqa: E402
    am, navi_offsets, pm4_nv, pm4_soc15, regs, sdma_4_0_0, sdma_6_0_0, soc_9,
    soc_11, soc_12, vega_offsets,
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

# Register families and the registers pulled from each.
GC_REGS = [
    "regCOMPUTE_PGM_LO",
    "regCOMPUTE_PGM_RSRC1",
    "regCOMPUTE_PGM_RSRC3",
    "regCOMPUTE_TMPRING_SIZE",
    "regCOMPUTE_DISPATCH_SCRATCH_BASE_LO",
    "regCOMPUTE_RESTART_X",
    "regCOMPUTE_USER_DATA_0",
    "regCOMPUTE_RESOURCE_LIMITS",
    "regCOMPUTE_START_X",
    "regCOMPUTE_DISPATCH_INITIATOR",
]
NBIO_REGS = [
    "regBIF_BX_PF0_GPU_HDP_FLUSH_REQ",
    "regBIF_BX_PF0_GPU_HDP_FLUSH_DONE",
    "regBIF_BX_PF1_GPU_HDP_FLUSH_REQ",
    "regBIF_BX_PF1_GPU_HDP_FLUSH_DONE",
]
REG_FAMILIES = [
    ("gc", (9, 4, 3), GC_REGS),
    ("gc", (11, 0, 0), GC_REGS),
    ("gc", (11, 0, 3), GC_REGS),
    ("gc", (11, 5, 0), GC_REGS),
    ("gc", (12, 0, 0), GC_REGS),
    ("nbio", (4, 3, 0), NBIO_REGS),
    ("nbio", (7, 2, 0), NBIO_REGS),
    ("nbio", (7, 7, 0), NBIO_REGS),
    ("nbio", (7, 9, 0), NBIO_REGS),
    ("nbio", (7, 11, 0), NBIO_REGS),
    ("nbif", (6, 3, 1), NBIO_REGS),
]

# Hardware IP discovery ids.
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

HEADER = """\
(* Generated by generate_tables.py; do not edit by hand.
   Regenerate with: uv run packages/tolk/lib/runtime/amd/generate_tables.py *)
"""

ENCODER_PROBES = [0, 1, 2, 3, 5, 0x7F, 0xFF, 0x1234, 0xFFFFF, 0xFFFFFFFF]


def ml_int(v):
    if v < 0:
        raise ValueError(f"negative constant {v}")
    return str(v) if v < 10 else f"0x{v:x}"


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
    return f'  ("{reg_name}", ({ml_int(off)}, {seg}, [ {"; ".join(parts)} ]));'


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


def gen_regs():
    lines = [HEADER]
    lines.append("(* Per-family register maps: name -> (offset, segment, fields as (lo, hi)). *)")
    fam_values = []
    for prefix, ver, wanted in REG_FAMILIES:
        value = f"{prefix}_{'_'.join(map(str, ver))}"
        fam_values.append((prefix, ver, value))
        fam = getattr(regs, value)
        lines.append(f"let {value} = [")
        lines += [reg_entry(fam, nm) for nm in wanted]
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


def main():
    outputs = {
        "amd_pm4_defs.ml": gen_pm4,
        "amd_sdma_defs.ml": gen_sdma,
        "amd_regs_defs.ml": gen_regs,
        "amd_soc_defs.ml": gen_soc,
        "amd_hsa_defs.ml": gen_hsa,
        "amd_kd_defs.ml": gen_kd,
    }
    for fname, gen in outputs.items():
        text = "\n".join(gen()) + "\n"
        with open(os.path.join(_HERE, fname), "w") as f:
            f.write(text)
        print(f"wrote {fname} ({text.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
