#!/usr/bin/env python3
"""Emit the OCaml NVIDIA driver tables (nv_defs.ml, nv_defs_versions.ml,
nv/nv_reg_defs.ml, nv/nv_gsp_defs.ml).

Regenerate with:
  uv run packages/tolk/lib/runtime/nv/generate_tables.py

Values are read from the pinned reference clone at `_tinygrad/` (repository
root); no network access is needed. The curated symbol inventories below drive
emission: to grow coverage, add a symbol to the relevant list and rerun.
Output is deterministic; rerunning must produce byte-identical files.

Parameter-structure layouts are emitted as per-used-field (byte offset, byte
size) pairs plus the structure's total size, read from the reference ctypes
layouts; scalar fields are probe-verified by instantiation before being
written. Array fields are emitted as offset/element-size/count triples.
Layouts and constants shared by the 570, 580, and 610 driver generations go
to nv_defs.ml (the generator aborts if any of them differ across the three
reference modules); the structures that differ go to nv_defs_versions.ml as
three values of one record type, and the generator aborts if a structure
curated as version-dependent stops differing, so a reference pin move that
shifts the delta set fails loudly instead of silently curating.
"""

import ast
import ctypes
import importlib
import inspect
import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIN = os.path.join(_HERE, "..", "..", "..", "..", "..", "_tinygrad")
sys.path.insert(0, _PIN)

from tinygrad import helpers  # noqa: E402
from tinygrad.runtime.autogen import nv, nv_570, nv_580, nv_610  # noqa: E402
import tinygrad.runtime.autogen.nv_regs as nv_regs  # noqa: E402

MODULES = [nv_570, nv_580, nv_610]

_IP_PY = os.path.join(_PIN, "tinygrad", "runtime", "support", "nv", "ip.py")

PIN_COMMIT = subprocess.run(
    ["git", "-C", _PIN, "rev-parse", "HEAD"],
    check=True, capture_output=True, text=True,
).stdout.strip()

# Curated inventories

CLASS_IDS = [
    "NV01_ROOT",
    "NV01_ROOT_CLIENT",
    "NV01_DEVICE_0",
    "NV20_SUBDEVICE_0",
    "NV01_MEMORY_VIRTUAL",
    "NV01_MEMORY_SYSTEM_OS_DESCRIPTOR",
    "NV1_MEMORY_SYSTEM",
    "NV1_MEMORY_USER",
    "FERMI_VASPACE_A",
    "FERMI_CONTEXT_SHARE_A",
    "KEPLER_CHANNEL_GROUP_A",
    "TURING_USERMODE_A",
    "HOPPER_USERMODE_A",
    "AMPERE_CHANNEL_GPFIFO_A",
    "BLACKWELL_CHANNEL_GPFIFO_A",
    "AMPERE_COMPUTE_B",
    "ADA_COMPUTE_A",
    "BLACKWELL_COMPUTE_A",
    "BLACKWELL_COMPUTE_B",
    "AMPERE_DMA_COPY_B",
    "BLACKWELL_DMA_COPY_B",
    "GT200_DEBUGGER",
]

ESCAPE_NUMBERS = [
    "NV_ESC_CARD_INFO",
    "NV_ESC_REGISTER_FD",
    "NV_ESC_RM_ALLOC",
    "NV_ESC_RM_ALLOC_MEMORY",
    "NV_ESC_RM_CONTROL",
    "NV_ESC_RM_FREE",
    "NV_ESC_RM_MAP_MEMORY",
    "NV_ESC_RM_MAP_MEMORY_DMA",
]

MEMORY_FLAGS = [
    "NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS",
    "NVOS02_FLAGS_COHERENCY_CACHED",
    "NVOS02_FLAGS_MAPPING_NO_MAP",
    "NVOS32_ATTR_PHYSICALITY_CONTIGUOUS",
    "NVOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS",
    "NVOS32_ATTR_PAGE_SIZE_HUGE",
    "NVOS32_ATTR_LOCATION_PCI",
    "NVOS32_ATTR2_GPU_CACHEABLE_YES",
    "NVOS32_ATTR2_GPU_CACHEABLE_NO",
    "NVOS32_ATTR2_PAGE_SIZE_HUGE_2MB",
    "NVOS32_ATTR2_ZBC_PREFER_NO_ZBC",
    "NVOS32_ATTR2_PROTECTION_USER_READ_ONLY",
    "NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED",
    "NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED",
    "NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE",
    "NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT",
    "NVOS32_ALLOC_FLAGS_PERSISTENT_VIDMEM",
    "NVOS32_TYPE_IMAGE",
    "NVOS32_TYPE_NOTIFIER",
    "NVOS33_FLAGS_CACHING_TYPE_WRITECOMBINED",
    "NVOS46_FLAGS_PAGE_SIZE_4KB",
    "NVOS46_FLAGS_CACHE_SNOOP_ENABLE",
    "NVOS46_FLAGS_DMA_OFFSET_FIXED_TRUE",
]

# Channel method ids and method-argument constants, per engine class family:
# host/gpfifo (NVC56F), compute (NVC6C0), copy (NVC6B5).
CHANNEL_INTS = [
    "NVC56F_SEM_ADDR_LO",
    "NVC56F_NON_STALL_INTERRUPT",
    "NVC56F_SEM_EXECUTE_OPERATION_ACQ_CIRC_GEQ",
    "NVC56F_SEM_EXECUTE_OPERATION_ACQ_AND",
    "NVC56F_SEM_EXECUTE_OPERATION_ACQ_NOR",
    "NVC56F_SEM_EXECUTE_OPERATION_RELEASE",
    "NVC56F_SEM_EXECUTE_PAYLOAD_SIZE_32BIT",
    "NVC56F_SEM_EXECUTE_PAYLOAD_SIZE_64BIT",
    "NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP_EN",
    "NVC56F_SEM_EXECUTE_RELEASE_WFI_EN",
    "NVC6C0_SET_OBJECT",
    "NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A",
    "NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A",
    "NVC6C0_SET_SHADER_LOCAL_MEMORY_A",
    "NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A",
    "NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI",
    "NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI_INSTRUCTION_TRUE",
    "NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI_GLOBAL_DATA_TRUE",
    "NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI_CONSTANT_TRUE",
    "NVC6C0_PM_TRIGGER",
    "NVC6C0_SEND_PCAS_A",
    "NVC6C0_SEND_SIGNALING_PCAS2_B",
    "NVC6B5_OFFSET_IN_UPPER",
    "NVC6B5_LINE_LENGTH_IN",
    "NVC6B5_LAUNCH_DMA",
    "NVC6B5_LAUNCH_DMA_DATA_TRANSFER_TYPE_NON_PIPELINED",
    "NVC6B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT_PITCH",
    "NVC6B5_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH",
    "NVC6B5_LAUNCH_DMA_FLUSH_ENABLE_TRUE",
    "NVC6B5_LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_FOUR_WORD_SEMAPHORE",
    "NVC6B5_SET_SEMAPHORE_A",
]
# Method-argument bit ranges, as (hi, lo) within the 32-bit argument.
CHANNEL_BIT_RANGES = [
    "NVC56F_SEM_EXECUTE_OPERATION",
    "NVC56F_SEM_EXECUTE_PAYLOAD_SIZE",
    "NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP",
    "NVC56F_SEM_EXECUTE_RELEASE_WFI",
    "NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI_INSTRUCTION",
    "NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI_GLOBAL_DATA",
    "NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI_CONSTANT",
    "NVC6B5_LAUNCH_DMA_DATA_TRANSFER_TYPE",
    "NVC6B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT",
    "NVC6B5_LAUNCH_DMA_DST_MEMORY_LAYOUT",
    "NVC6B5_LAUNCH_DMA_FLUSH_ENABLE",
    "NVC6B5_LAUNCH_DMA_SEMAPHORE_TYPE",
]

CONTROL_INTS = [
    "NV0000_CTRL_CMD_SYSTEM_GET_BUILD_VERSION_V2",
    "NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2",
    "NV0080_CTRL_CMD_GPU_GET_CLASSLIST",
    "NV2080_CTRL_CMD_GPU_GET_GID_INFO",
    "NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY",
    "NV2080_CTRL_CMD_PERF_BOOST",
    "NV2080_CTRL_PERF_BOOST_FLAGS_CMD_BOOST_TO_MAX",
    "NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_YES",
    "NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_PRIORITY_HIGH",
    "NVA06C_CTRL_CMD_GPFIFO_SCHEDULE",
    "NVA06F_CTRL_CMD_BIND",
    "NVA06F_CTRL_CMD_GPFIFO_SCHEDULE",
    "NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN",
    "NV2080_CTRL_CMD_GR_GET_INFO",
    "NV2080_CTRL_CMD_FB_FLUSH_GPU_CACHE",
    "NV2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_WRITE_BACK_YES",
    "NV2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_INVALIDATE_YES",
    "NV2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_FLUSH_MODE_FULL_CACHE",
    "NV2080_CTRL_CMD_INTERNAL_BUS_FLUSH_WITH_SYSMEMBAR",
    "NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_INFO",
    "NV83DE_CTRL_CMD_DEBUG_READ_ALL_SM_ERROR_STATES",
    "NV83DE_CTRL_CMD_DEBUG_READ_MMU_FAULT_INFO",
    "NV2080_ENGINE_TYPE_GRAPHICS",
]

ALLOCATION_FLAGS = [
    "NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES",
    "NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING",
    "NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED",
    "NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC",
]

ERROR_INTS = ["NV_ERR_NO_MEMORY"]

UVM_COMMANDS = [
    "UVM_INITIALIZE",
    "UVM_MM_INITIALIZE",
    "UVM_REGISTER_GPU",
    "UVM_REGISTER_GPU_VASPACE",
    "UVM_REGISTER_CHANNEL",
    "UVM_CREATE_EXTERNAL_RANGE",
    "UVM_MAP_EXTERNAL_ALLOCATION",
    "UVM_FREE",
]

# GR info query indexes, resolved from the request names (the LITTER_ spelling
# is used when the plain one does not exist).
GR_INFO_REQUESTS = [
    "num_gpcs",
    "num_tpc_per_gpc",
    "num_sm_per_tpc",
    "max_warps_per_sm",
    "sm_version",
]

QMD_INTS = [
    "NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR",
    "NVCEC0_QMDV05_00_QMD_TYPE_GRID_CTA",
]

QMD_PREFIXES = ["NVC6C0_QMDV03_00", "NVCEC0_QMDV05_00"]

# Version-stable parameter structures: (module title, reference name, used
# fields). Field specs: "f" scalar pair, "f/region" byte-region pair,
# "f/array" offset/element-size/count triple, "f.g" nested scalar flattened
# to an absolute pair. A rename maps a reference field name to the emitted
# one (OCaml keywords).
STRUCTS = [
    ("Nv_ioctl_card_info", "nv_ioctl_card_info_t",
     ["valid", "gpu_id", "minor_number"]),
    ("Nv_ioctl_register_fd", "nv_ioctl_register_fd_t", ["ctl_fd"]),
    ("Nvos00_parameters", "NVOS00_PARAMETERS",
     ["hRoot", "hObjectParent", "hObjectOld", "status"]),
    ("Nvos02_parameters", "NVOS02_PARAMETERS",
     ["hRoot", "hObjectParent", "hObjectNew", "hClass", "flags", "pMemory",
      "limit", "status"]),
    ("Nvos21_parameters", "NVOS21_PARAMETERS",
     ["hRoot", "hObjectParent", "hObjectNew", "hClass", "pAllocParms",
      "status"]),
    ("Nvos33_parameters", "NVOS33_PARAMETERS",
     ["hClient", "hDevice", "hMemory", "length", "flags", "status"]),
    ("Nvos54_parameters", "NVOS54_PARAMETERS",
     ["hClient", "hObject", "cmd", "params", "paramsSize", "status"]),
    ("Nv_ioctl_nvos02_parameters_with_fd", "nv_ioctl_nvos02_parameters_with_fd",
     ["params/region", "fd"]),
    ("Nv_ioctl_nvos33_parameters_with_fd", "nv_ioctl_nvos33_parameters_with_fd",
     ["params/region", "fd"]),
    ("Nv_uuid", "struct_nv_uuid", ["uuid/region"]),
    ("Nv0000_alloc_parameters", "NV0000_ALLOC_PARAMETERS", []),
    ("Nv0080_alloc_parameters", "NV0080_ALLOC_PARAMETERS",
     ["deviceId", "hClientShare", "vaMode"]),
    ("Nv2080_alloc_parameters", "NV2080_ALLOC_PARAMETERS", []),
    ("Nv_memory_virtual_allocation_params", "NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS",
     ["limit"]),
    ("Nv_memory_allocation_params", "NV_MEMORY_ALLOCATION_PARAMS",
     ["owner", "type", "flags", "attr", "attr2", "format", "size",
      "alignment", "offset", "limit"]),
    ("Nv_channel_group_allocation_parameters",
     "NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS", ["engineType"]),
    ("Nv_ctxshare_allocation_parameters", "NV_CTXSHARE_ALLOCATION_PARAMETERS",
     ["hVASpace", "flags"]),
    ("Nv83de_alloc_parameters", "NV83DE_ALLOC_PARAMETERS",
     ["hAppClient", "hClass3dObject"]),
    ("Nv0000_ctrl_system_get_build_version_v2_params",
     "NV0000_CTRL_SYSTEM_GET_BUILD_VERSION_V2_PARAMS",
     ["driverVersionBuffer/region"]),
    ("Nv0000_ctrl_gpu_get_id_info_v2_params",
     "NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS", ["gpuId", "deviceInstance"]),
    ("Nv0080_ctrl_gpu_get_classlist_params",
     "NV0080_CTRL_GPU_GET_CLASSLIST_PARAMS", ["numClasses", "classList"]),
    ("Nv2080_ctrl_gpu_get_gid_info_params",
     "NV2080_CTRL_GPU_GET_GID_INFO_PARAMS",
     ["flags", "length", "data/region"]),
    ("Nv2080_ctrl_perf_boost_params", "NV2080_CTRL_PERF_BOOST_PARAMS",
     ["flags", "duration"]),
    ("Nva06f_ctrl_bind_params", "NVA06F_CTRL_BIND_PARAMS", ["engineType"]),
    ("Nvc36f_ctrl_cmd_gpfifo_get_work_submit_token_params",
     "NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS",
     ["workSubmitToken"]),
    ("Nv2080_ctrl_gr_info", "NV2080_CTRL_GR_INFO", ["index", "data"]),
    ("Nv2080_ctrl_gr_get_info_params", "NV2080_CTRL_GR_GET_INFO_PARAMS",
     ["grInfoListSize", "grInfoList"]),
    ("Nv2080_ctrl_fb_flush_gpu_cache_params",
     "NV2080_CTRL_FB_FLUSH_GPU_CACHE_PARAMS", ["flags"]),
    ("Nv83de_ctrl_debug_read_all_sm_error_states_params",
     "NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS",
     ["hTargetChannel", "numSMsToRead", "smErrorStateArray/array",
      "mmuFault.valid"]),
    ("Nv83de_sm_error_state_registers", "struct_NV83DE_SM_ERROR_STATE_REGISTERS",
     ["hwwGlobalEsr", "hwwWarpEsr", "hwwWarpEsrPc64"]),
    ("Nv83de_ctrl_debug_read_mmu_fault_info_params",
     "NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_PARAMS",
     ["count", "mmuFaultInfoList/array"]),
    ("Nv83de_ctrl_debug_read_mmu_fault_info_entry",
     "struct_NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_ENTRY",
     ["faultAddress", "faultType", "accessType"]),
    ("Uvm_initialize_params", "UVM_INITIALIZE_PARAMS", ["rmStatus"]),
    ("Uvm_mm_initialize_params", "UVM_MM_INITIALIZE_PARAMS",
     ["uvmFd", "rmStatus"]),
    ("Uvm_register_gpu_params", "UVM_REGISTER_GPU_PARAMS",
     ["gpu_uuid/region", "rmCtrlFd", "rmStatus"]),
    ("Uvm_register_gpu_vaspace_params", "UVM_REGISTER_GPU_VASPACE_PARAMS",
     ["gpuUuid/region", "rmCtrlFd", "hClient", "hVaSpace", "rmStatus"]),
    ("Uvm_register_channel_params", "UVM_REGISTER_CHANNEL_PARAMS",
     ["gpuUuid/region", "rmCtrlFd", "hClient", "hChannel", "base", "length",
      "rmStatus"]),
    ("Uvm_create_external_range_params", "UVM_CREATE_EXTERNAL_RANGE_PARAMS",
     ["base", "length", "rmStatus"]),
    ("Uvm_map_external_allocation_params", "UVM_MAP_EXTERNAL_ALLOCATION_PARAMS",
     ["base", "length", "perGpuAttributes/array", "gpuAttributesCount",
      "rmCtrlFd", "hClient", "hMemory", "rmStatus"]),
    ("Uvm_gpu_mapping_attributes", "UvmGpuMappingAttributes",
     ["gpuUuid/region", "gpuMappingType"]),
]
RENAMES = {"type": "typ", "function": "func"}

# The driver-less tier reads this structure through an RPC layer that is
# pinned to the 570-generation layouts, so it is emitted from that generation
# only and exempt from the cross-version stability check.
PINNED_570_STRUCTS = [
    ("Nv2080_ctrl_internal_static_gr_get_info_params",
     "NV2080_CTRL_INTERNAL_STATIC_GR_GET_INFO_PARAMS", ["engineInfo/array"]),
    ("Nv2080_ctrl_internal_static_gr_info",
     "struct_NV2080_CTRL_INTERNAL_STATIC_GR_INFO", ["infoList/array"]),
    ("Nv2080_ctrl_internal_gr_info", "struct_NV2080_CTRL_INTERNAL_GR_INFO",
     ["data"]),
]

# Version-dependent structures: (record field in t, record type name,
# reference name, used fields). "f?" marks a field absent from some
# generations, emitted as an option.
VERSIONED_STRUCTS = [
    ("nva06c_ctrl_gpfifo_schedule_params", "schedule_params",
     "NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS", ["bEnable"]),
    ("nva06f_ctrl_gpfifo_schedule_params", "schedule_params",
     "NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS", ["bEnable"]),
    ("nvos46_parameters", "nvos46_parameters", "NVOS46_PARAMETERS",
     ["hClient", "hDevice", "hDma", "hMemory", "length", "flags",
      "dmaOffset", "status"]),
    ("nv_channelgpfifo_allocation_parameters",
     "channelgpfifo_allocation_parameters",
     "NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS",
     ["hObjectError", "hObjectBuffer", "gpFifoOffset", "gpFifoEntries",
      "hContextShare", "hUserdMemory", "userdOffset", "engineType"]),
    ("nv_vaspace_allocation_parameters", "vaspace_allocation_parameters",
     "NV_VASPACE_ALLOCATION_PARAMETERS", ["vaBase", "vaSize", "flags"]),
    ("uvm_free_params", "uvm_free_params", "UVM_FREE_PARAMS",
     ["base", "length?", "rmStatus"]),
]
# hUserdMemory and userdOffset are 8-element arrays of which only element 0
# is written; their pairs are element 0.
VERSIONED_ELEM0_ARRAYS = {"hUserdMemory", "userdOffset"}

HEADER = f"""\
(* Generated by generate_tables.py; do not edit by hand.
   Regenerate with: uv run packages/tolk/lib/runtime/nv/generate_tables.py
   Data source: the pinned reference clone at _tinygrad, commit
   {PIN_COMMIT}. *)
"""


def ml_int(v):
    if v < 0:
        raise ValueError(f"negative constant {v}")
    return str(v) if v < 10 else f"0x{v:x}"


def int_let(name, v, indent=""):
    return f"{indent}let {name.lower()} = {ml_int(v)}"


def emit_name(field):
    return RENAMES.get(field, field).lower()


def struct_entry(cls, field):
    entry = next((f for f in cls._real_fields_ if f[0] == field), None)
    if entry is None:
        raise ValueError(f"{cls.__name__}: no field {field}")
    if len(entry) != 3:
        raise ValueError(f"{cls.__name__}.{field}: bitfields are unsupported")
    return entry


def scalar_pair(cls, field):
    """(offset, size) of a scalar field, probe-checked by instantiation."""
    name, typ, off = struct_entry(cls, field)
    if issubclass(typ, (ctypes.Structure, ctypes.Array)):
        raise ValueError(f"{cls.__name__}.{field}: not a scalar field")
    size = ctypes.sizeof(typ)
    try:
        raw = bytes(cls(**{name: (1 << (8 * size)) - 1}))
    except (TypeError, ValueError, OverflowError):
        raw = bytes(cls(**{name: -1}))
    want = b"\x00" * off + b"\xff" * size + b"\x00" * (cls.SIZE - off - size)
    if raw != want:
        raise ValueError(f"{cls.__name__}.{field}: scalar probe mismatch")
    return off, size


def region_pair(cls, field):
    """(offset, byte length) of an embedded structure or byte array."""
    name, typ, off = struct_entry(cls, field)
    if not issubclass(typ, (ctypes.Structure, ctypes.Array)):
        raise ValueError(f"{cls.__name__}.{field}: not a region field")
    return off, ctypes.sizeof(typ)


def array_spec(cls, field):
    """(offset, element size, count) of an array field."""
    name, typ, off = struct_entry(cls, field)
    if not issubclass(typ, ctypes.Array):
        raise ValueError(f"{cls.__name__}.{field}: not an array field")
    return off, ctypes.sizeof(typ._type_), typ._length_


def bitfield_bytepair(cls, field):
    """(offset, byte length) of a low-aligned bitfield.

    Only bit_off 0 bitfields are representable as a plain (offset, size) pair;
    they occupy the low ceil(width/8) bytes of their storage, so writing that
    many bytes sets the field for any value it can hold (the higher bits in
    those bytes belong to reserved fields, cleared to zero). A pin move that
    shifts such a field off a byte boundary aborts generation loudly.
    """
    entry = next((f for f in cls._real_fields_ if f[0] == field), None)
    if entry is None or len(entry) != 5:
        raise ValueError(f"{cls.__name__}.{field}: not a bitfield")
    _name, _typ, off, bit_width, bit_off = entry
    if bit_off != 0:
        raise ValueError(
            f"{cls.__name__}.{field}: bit offset {bit_off} is not byte-aligned")
    return off, (bit_width + 7) // 8


def field_lines(cls, fields, indent="  "):
    lines = []
    for spec in fields:
        if spec.endswith("/array"):
            field = spec.removesuffix("/array")
            off, elem, count = array_spec(cls, field)
            name = emit_name(field)
            lines.append(f"{indent}let {name}_offset = {ml_int(off)}")
            lines.append(f"{indent}let {name}_elem_size = {ml_int(elem)}")
            lines.append(f"{indent}let {name}_count = {ml_int(count)}")
        elif spec.endswith("/region"):
            field = spec.removesuffix("/region")
            off, size = region_pair(cls, field)
            lines.append(
                f"{indent}let {emit_name(field)} = ({ml_int(off)}, {ml_int(size)})")
        elif spec.endswith("/bits"):
            outer, inner = spec.removesuffix("/bits").split(".")
            outer_off, _ = region_pair(cls, outer)
            _, typ, _ = struct_entry(cls, outer)
            off, size = bitfield_bytepair(typ, inner)
            lines.append(
                f"{indent}let {emit_name(outer)}_{emit_name(inner)} = "
                f"({ml_int(outer_off + off)}, {ml_int(size)})")
        elif "." in spec:
            outer, inner = spec.split(".")
            outer_off, _ = region_pair(cls, outer)
            _, typ, _ = struct_entry(cls, outer)
            off, size = scalar_pair(typ, inner)
            lines.append(
                f"{indent}let {emit_name(outer)}_{emit_name(inner)} = "
                f"({ml_int(outer_off + off)}, {ml_int(size)})")
        else:
            off, size = scalar_pair(cls, spec)
            lines.append(
                f"{indent}let {emit_name(spec)} = ({ml_int(off)}, {ml_int(size)})")
    return lines


def struct_module(title, cls, fields):
    lines = [f"module {title} = struct", f"  let sizeof = {ml_int(cls.SIZE)}"]
    lines += field_lines(cls, fields)
    lines.append("end")
    return lines


def used_layout(mod, name, fields):
    """The comparable layout of a structure's used fields in one module."""
    cls = getattr(mod, name)
    out = {"@sizeof": cls.SIZE}
    for spec in fields:
        field = spec.rstrip("?")
        if field.endswith("/array"):
            base = field.removesuffix("/array")
            if getattr(cls, base, None) is None:
                out[field] = None
            else:
                out[field] = array_spec(cls, base)
        elif field.endswith("/region"):
            out[field] = region_pair(cls, field.removesuffix("/region"))
        elif "." in field:
            outer, inner = field.split(".")
            outer_off, _ = region_pair(cls, outer)
            _, typ, _ = struct_entry(cls, outer)
            off, size = scalar_pair(typ, inner)
            out[field] = (outer_off + off, size)
        elif next((f for f in cls._real_fields_ if f[0] == field), None) is None:
            out[field] = None
        elif field in VERSIONED_ELEM0_ARRAYS:
            out[field] = array_spec(cls, field)
        else:
            out[field] = scalar_pair(cls, field)
    return out


def assert_stable(kind, name, values):
    if not all(v == values[0] for v in values[1:]):
        raise ValueError(f"{kind} {name} differs across 570/580/610: {values}")


def qmd_table(mod, pref):
    """Field name -> (hi, lo), indexed entries expanded for slots 0..7."""
    table = {name[len(pref) + 1:]: dt for name, dt in mod.__dict__.items()
             if name.startswith(pref) and isinstance(dt, tuple)}
    table.update({name[len(pref) + 1:] + f"_{i}": dt(i)
                  for name, dt in mod.__dict__.items()
                  for i in range(8) if name.startswith(pref) and callable(dt)})
    return table


def pfault_tables(mod):
    fault = {dt: name for name, dt in mod.__dict__.items()
             if name.startswith("NV_PFAULT_FAULT_TYPE_")}
    access = {dt: name.split("_")[-1] for name, dt in mod.__dict__.items()
              if name.startswith("NV_PFAULT_ACCESS_TYPE_")}
    return fault, access


def assoc_lines(pairs):
    return [f'  ({ml_int(k)}, "{v}");' for k, v in sorted(pairs)]


def gr_info_indexes():
    names = []
    for req in GR_INFO_REQUESTS:
        plain = "NV2080_CTRL_GR_INFO_INDEX_" + req.upper()
        litter = "NV2080_CTRL_GR_INFO_INDEX_LITTER_" + req.upper()
        names.append(plain if hasattr(nv_570, plain) else litter)
    return names


def gen_defs():
    int_sections = [
        ("RM class ids.", CLASS_IDS),
        ("Escape ioctl numbers.", ESCAPE_NUMBERS),
        ("Memory allocation and mapping flags (NVOS02/32/33/46).",
         MEMORY_FLAGS),
        ("Channel method ids and method-argument values: host (NVC56F), "
         "compute (NVC6C0), copy (NVC6B5).", CHANNEL_INTS),
        ("Control commands and their flag values.", CONTROL_INTS),
        ("Allocation flags.", ALLOCATION_FLAGS),
        ("Driver status codes checked by name.", ERROR_INTS),
        ("UVM ioctl command numbers.", UVM_COMMANDS),
        ("GR info query indexes.", gr_info_indexes()),
        ("QMD constants.", QMD_INTS),
    ]
    for _, names in int_sections:
        for nm in names:
            assert_stable("int", nm, [getattr(m, nm) for m in MODULES])
    for nm in CHANNEL_BIT_RANGES:
        assert_stable("bit range", nm, [getattr(m, nm) for m in MODULES])

    lines = [HEADER]
    for comment, names in int_sections:
        lines.append(f"(* {comment} *)")
        lines += [int_let(nm, getattr(nv_570, nm)) for nm in names]
        lines.append("")

    lines.append("(* Method-argument bit ranges, as (hi, lo) inclusive. *)")
    for nm in CHANNEL_BIT_RANGES:
        hi, lo = getattr(nv_570, nm)
        lines.append(f"let {nm.lower()} = ({hi}, {lo})")
    lines.append("")

    gpput = [next(f[2] for f in m.AmpereAControlGPFifo._real_fields_
                  if f[0] == "GPPut") for m in MODULES]
    assert_stable("offset", "AmpereAControlGPFifo.GPPut", gpput)
    lines.append("(* Byte offset of the GPPut register in the GPFIFO"
                 " control page. *)")
    lines.append(f"let ampere_a_control_gpfifo_gpput = {ml_int(gpput[0])}")
    lines.append("")

    lines.append("(* Parameter-structure layouts: per-field (byte offset,"
                 " byte size) pairs\n   and total size; arrays as"
                 " offset/element-size/count. *)")
    for title, name, fields in STRUCTS:
        assert_stable("struct", name,
                      [used_layout(m, name, fields) for m in MODULES])
        lines += struct_module(title, getattr(nv_570, name), fields)
        lines.append("")

    lines.append("(* Read only on the driver-less tier, whose RPC layer is"
                 " pinned to the\n   570-generation layouts; emitted from"
                 " that generation. *)")
    for title, name, fields in PINNED_570_STRUCTS:
        lines += struct_module(title, getattr(nv_570, name), fields)
        lines.append("")

    fault, access = pfault_tables(nv_570)
    for other in MODULES[1:]:
        if pfault_tables(other) != (fault, access):
            raise ValueError("MMU fault name tables differ across versions")
    lines.append("(* MMU fault decode tables: fault type and access type"
                 " names by id. *)")
    lines.append("let nv_pfault_fault_type = [")
    lines += assoc_lines(fault.items())
    lines.append("]")
    lines.append("")
    lines.append("let nv_pfault_access_type = [")
    lines += assoc_lines(access.items())
    lines.append("]")
    lines.append("")

    lines.append("(* QMD bitfield tables: field name -> (hi, lo) bit"
                 " positions within the\n   descriptor, with the per-slot"
                 " fields expanded for slots 0..7. *)")
    for pref in QMD_PREFIXES:
        tables = [qmd_table(m, pref) for m in MODULES]
        assert_stable("qmd table", pref, tables)
        if not tables[0]:
            raise ValueError(f"empty QMD table for {pref}")
        lines.append(f"let {pref.lower()}_fields = [")
        lines += [f'  ("{nm}", ({hi}, {lo}));'
                  for nm, (hi, lo) in sorted(tables[0].items())]
        lines.append("]")
        lines.append("")
    return lines[:-1]


def versioned_value(mod):
    fields = {}
    for record_field, _, name, used in VERSIONED_STRUCTS:
        cls = getattr(mod, name)
        parts = [f"sizeof = {ml_int(cls.SIZE)}"]
        for spec in used:
            optional = spec.endswith("?")
            field = spec.rstrip("?")
            entry = next((f for f in cls._real_fields_ if f[0] == field), None)
            if entry is None:
                if not optional:
                    raise ValueError(f"{mod.__name__}: {name}.{field} missing")
                parts.append(f"{emit_name(field)} = None")
                continue
            if field in VERSIONED_ELEM0_ARRAYS:
                off, size, _count = array_spec(cls, field)
            else:
                off, size = scalar_pair(cls, field)
            pair = f"({ml_int(off)}, {ml_int(size)})"
            parts.append(f"{emit_name(field)} = "
                         + (f"Some {pair}" if optional else pair))
        fields[record_field] = parts
    return fields


def gen_versions():
    for _, _, name, used in VERSIONED_STRUCTS:
        layouts = [used_layout(m, name, used) for m in MODULES]
        if all(l == layouts[0] for l in layouts[1:]):
            raise ValueError(
                f"{name} no longer differs across 570/580/610; move it to"
                " nv_defs.ml and drop it from VERSIONED_STRUCTS")

    lines = [HEADER]
    lines.append("""\
(* Parameter-structure layouts that differ across the supported driver
   generations, plus the per-generation status-code names, as three values
   of one record type. Fields are (byte offset, byte size) pairs. *)

type field = int * int

type schedule_params = { sizeof : int; benable : field }

type nvos46_parameters = {
  sizeof : int;
  hclient : field;
  hdevice : field;
  hdma : field;
  hmemory : field;
  length : field;
  flags : field;
  dmaoffset : field;
  status : field;
}

type channelgpfifo_allocation_parameters = {
  sizeof : int;
  hobjecterror : field;
  hobjectbuffer : field;
  gpfifooffset : field;
  gpfifoentries : field;
  hcontextshare : field;
  huserdmemory : field;  (* element 0 of 8 *)
  userdoffset : field;  (* element 0 of 8 *)
  enginetype : field;
}

type vaspace_allocation_parameters = {
  sizeof : int;
  vabase : field;
  vasize : field;
  flags : field;
}

type uvm_free_params = {
  sizeof : int;
  base : field;
  length : field option;  (* absent from the 610 layout *)
  rmstatus : field;
}

type t = {
  nva06c_ctrl_gpfifo_schedule_params : schedule_params;
  nva06f_ctrl_gpfifo_schedule_params : schedule_params;
  nvos46_parameters : nvos46_parameters;
  nv_channelgpfifo_allocation_parameters :
    channelgpfifo_allocation_parameters;
  nv_vaspace_allocation_parameters : vaspace_allocation_parameters;
  uvm_free_params : uvm_free_params;
  nv_status_codes : (int * string) list;
}
""")
    for mod, tag in zip(MODULES, ["v570", "v580", "v610"]):
        lines.append(f"let {tag} : t = {{")
        for record_field, parts in versioned_value(mod).items():
            lines.append(f"  {record_field} =")
            lines.append("    { " + ";\n      ".join(parts) + " };")
        lines.append("  nv_status_codes = [")
        lines += ["    " + line.strip()
                  for line in assoc_lines(mod.nv_status_codes.items())]
        lines.append("  ];")
        lines.append("}")
        lines.append("")
    return lines[:-1]


# GSP driver-less tier (nv_reg_defs.ml, nv_gsp_defs.ml)
#
# The GSP tier is pinned to the 570-generation layouts (ip.py imports nv_570
# unconditionally and the nv module carries no version dispatch), so both
# modules are emitted from a single reference generation. The census below is
# checked against the true `nv.`-qualified usage in ip.py on every run, so a
# reference pin that adds, drops, or renames a used symbol fails loudly.

# GSP structures used by ip.py, as (module title, reference name, used fields).
# Field specs follow field_lines: "f" scalar, "f/region" byte region, "f/array"
# offset/element-size/count, "outer.inner/bits" a low-aligned nested bitfield.
GSP_STRUCTS = [
    ("Msgq_tx_header", "msgqTxHeader",
     ["version", "size", "entryOff", "msgSize", "msgCount", "writePtr", "flags",
      "rxHdrOff"]),
    ("Rpc_message_header", "rpc_message_header_v",
     ["signature", "rpc_result", "rpc_result_private", "header_version",
      "function", "length"]),
    ("Gsp_msg_queue_element", "GSP_MSG_QUEUE_ELEMENT",
     ["elemCount", "seqNum", "checkSum"]),
    ("Gsp_fw_wpr_meta", "GspFwWprMeta",
     ["magic", "revision", "sysmemAddrOfRadix3Elf", "sizeOfRadix3Elf",
      "sysmemAddrOfBootloader", "sizeOfBootloader", "bootloaderCodeOffset",
      "bootloaderDataOffset", "bootloaderManifestOffset", "sysmemAddrOfSignature",
      "sizeOfSignature", "gspFwRsvdStart", "nonWprHeapOffset", "nonWprHeapSize",
      "gspFwWprStart", "gspFwHeapOffset", "gspFwHeapSize", "gspFwOffset",
      "bootBinOffset", "frtsOffset", "frtsSize", "gspFwWprEnd", "fbSize",
      "vgaWorkspaceOffset", "vgaWorkspaceSize", "pmuReservedSize"]),
    ("Nvfw_bin_hdr", "struct_nvfw_bin_hdr",
     ["header_offset", "data_offset", "data_size"]),
    ("Nvfw_hs_header_v2", "struct_nvfw_hs_header_v2",
     ["header_offset", "patch_loc", "patch_sig", "sig_prod_offset",
      "sig_prod_size", "num_sig"]),
    ("Nvfw_hs_load_header_v2", "struct_nvfw_hs_load_header_v2",
     ["os_data_offset", "os_data_size"]),
    ("Nvfw_hs_load_header_v2_app", "struct_nvfw_hs_load_header_v2_app",
     ["offset", "size"]),
    ("Rpc_run_cpu_sequencer", "rpc_run_cpu_sequencer_v17_00",
     ["cmdIndex", "regSaveArea/array"]),
    ("Packed_registry_table", "PACKED_REGISTRY_TABLE", ["size", "numEntries"]),
    ("Packed_registry_entry", "PACKED_REGISTRY_ENTRY",
     ["nameOffset", "type", "data", "length"]),
    ("Nvdm_payload_cot", "NVDM_PAYLOAD_COT",
     ["version", "size", "frtsVidmemOffset", "frtsVidmemSize",
      "gspBootArgsSysmemOffset", "gspFmcSysmemOffset", "hash384/array",
      "signature/array", "publicKey/array"]),
    ("Libos_memory_region_init_argument", "LibosMemoryRegionInitArgument",
     ["kind", "loc", "size", "id8", "pa"]),
    ("Gsp_arguments_cached", "GSP_ARGUMENTS_CACHED",
     ["bDmemStack", "messageQueueInitArguments/region"]),
    ("Message_queue_init_arguments", "MESSAGE_QUEUE_INIT_ARGUMENTS",
     ["sharedMemPhysAddr", "pageTableEntryCount", "cmdQueueOffset",
      "statQueueOffset"]),
    ("Fwseclic_read_vbios_desc", "FWSECLIC_READ_VBIOS_DESC",
     ["version", "size", "flags"]),
    ("Fwseclic_frts_region_desc", "FWSECLIC_FRTS_REGION_DESC",
     ["version", "size", "frtsRegionOffset4K", "frtsRegionSize",
      "frtsRegionMediaType"]),
    ("Fwseclic_frts_cmd", "FWSECLIC_FRTS_CMD",
     ["readVbiosDesc/region", "frtsRegionDesc/region"]),
    ("Bit_header_v1_00", "BIT_HEADER_V1_00",
     ["Signature", "TokenEntries", "HeaderSize", "TokenSize"]),
    ("Bit_token_v1_00", "BIT_TOKEN_V1_00",
     ["TokenId", "DataVersion", "DataSize", "DataPtr"]),
    ("Bit_data_falcon_data_v2", "BIT_DATA_FALCON_DATA_V2",
     ["FalconUcodeTablePtr"]),
    ("Falcon_ucode_table_hdr_v1", "FALCON_UCODE_TABLE_HDR_V1",
     ["EntryCount", "HeaderSize", "EntrySize"]),
    ("Falcon_ucode_table_entry_v1", "FALCON_UCODE_TABLE_ENTRY_V1",
     ["ApplicationID", "DescPtr"]),
    ("Falcon_ucode_desc_header", "FALCON_UCODE_DESC_HEADER", ["vDesc"]),
    ("Falcon_ucode_desc_v3", "FALCON_UCODE_DESC_V3",
     ["StoredSize", "IMEMLoadSize", "InterfaceOffset", "PKCDataOffset",
      "IMEMPhysBase", "IMEMVirtBase", "DMEMPhysBase", "DMEMLoadSize",
      "EngineIdMask", "UcodeId"]),
    ("Falcon_application_interface_header_v1",
     "FALCON_APPLICATION_INTERFACE_HEADER_V1", ["entryCount"]),
    ("Falcon_application_interface_entry_v1",
     "FALCON_APPLICATION_INTERFACE_ENTRY_V1", ["id", "dmemOffset"]),
    ("Falcon_application_interface_dmem_mapper_v3",
     "FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3",
     ["init_cmd", "cmd_in_buffer_offset"]),
    ("Gsp_fmc_boot_params", "GSP_FMC_BOOT_PARAMS",
     ["bootGspRmParams/region", "gspRmParams/region"]),
    ("Gsp_acr_boot_gsp_rm_params", "GSP_ACR_BOOT_GSP_RM_PARAMS",
     ["gspRmDescOffset", "gspRmDescSize", "target", "bIsGspRmBoot"]),
    ("Gsp_rm_params", "GSP_RM_PARAMS", ["bootArgsOffset", "target"]),
    ("Gsp_system_info", "GspSystemInfo",
     ["gpuPhysAddr", "gpuPhysFbAddr", "gpuPhysInstAddr", "pciConfigMirrorBase",
      "pciConfigMirrorSize", "nvDomainBusDeviceFunc", "bIsPassthru",
      "PCIDeviceID", "PCISubDeviceID", "PCIRevisionID", "maxUserVa"]),
    ("Rm_riscv_ucode_desc", "RM_RISCV_UCODE_DESC",
     ["monitorCodeOffset", "monitorDataOffset", "manifestOffset"]),
    ("Rpc_alloc_memory", "rpc_alloc_memory_v",
     ["hClient", "hDevice", "hMemory", "hClass", "flags", "pteAdjust", "format",
      "length", "pageCount", "pteDesc.idr/bits", "pteDesc.length/bits"]),
    ("Pte_desc_pte_pde", "struct_pte_desc_pte_pde", ["pte"]),
    ("Rpc_gsp_rm_alloc", "rpc_gsp_rm_alloc_v",
     ["hClient", "hParent", "hObject", "hClass", "flags", "paramsSize"]),
    ("Rpc_gsp_rm_control", "rpc_gsp_rm_control_v",
     ["hClient", "hObject", "cmd", "flags", "paramsSize"]),
    ("Rpc_set_page_directory", "rpc_set_page_directory_v",
     ["hClient", "hDevice", "pasid", "params/region"]),
    ("Nv0080_ctrl_dma_set_page_directory_params",
     "struct_NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS_v1E_05",
     ["physAddress", "numEntries", "flags", "hVASpace", "pasid", "subDeviceId",
      "chId"]),
    ("Rpc_unloading_guest_driver", "rpc_unloading_guest_driver_v",
     ["bInPMTransition", "bGc6Entering", "newLevel"]),
]

# GSP scalar constants used by ip.py.
GSP_CONSTS = [
    "NV_VGPU_MSG_SIGNATURE_VALID",
    "NV_VGPU_MSG_RESULT_RPC_PENDING",
    "NV_VGPU_PTEDESC_IDR_NONE",
    "NV_VGPU_MSG_FUNCTION_ALLOC_MEMORY",
    "NV_VGPU_MSG_FUNCTION_CONTINUATION_RECORD",
    "NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC",
    "NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL",
    "NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY",
    "NV_VGPU_MSG_FUNCTION_GSP_SET_SYSTEM_INFO",
    "NV_VGPU_MSG_FUNCTION_SET_REGISTRY",
    "NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER",
    "NV_VGPU_MSG_EVENT_GSP_INIT_DONE",
    "NV_VGPU_MSG_EVENT_GSP_RUN_CPU_SEQUENCER",
    "NV_VGPU_MSG_EVENT_OS_ERROR_LOG",
    "NV_VGPU_MSG_EVENT_MMU_FAULT_QUEUED",
    "LIBOS_MEMORY_REGION_CONTIGUOUS",
    "LIBOS_MEMORY_REGION_LOC_SYSMEM",
    "LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2",
    "GSP_DMA_TARGET_COHERENT_SYSTEM",
    "GSP_FW_WPR_META_REVISION",
    "GSP_FW_WPR_META_MAGIC",
    "REGISTRY_TABLE_ENTRY_TYPE_DWORD",
    "NVDM_TYPE_COT",
    "PCI_ROM_IMAGE_BLOCK_SIZE",
    "OFFSETOF_PCI_EXP_ROM_PCI_DATA_STRUCT_PTR",
    "OFFSETOF_PCI_DATA_STRUCT_IMAGE_LEN",
    "OFFSETOF_PCI_DATA_STRUCT_CODE_TYPE",
    "NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_BASE",
    "NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_EXT",
    "BIT_TOKEN_FALCON_DATA",
    "BIT_DATA_FALCON_DATA_V2_SIZE_4",
    "FALCON_UCODE_ENTRY_APPID_FWSEC_PROD",
    "FALCON_UCODE_DESC_V3_SIZE_44",
    "FALCON_APPLICATION_INTERFACE_ENTRY_ID_DMEMMAPPER",
]
# Constants wider than a 63-bit OCaml int, emitted as int64 literals.
GSP_CONSTS64 = {"GSP_FW_WPR_META_MAGIC"}

# The GSP firmware files ip.py fetches, in load order. booter_load/bootloader
# carry a per-chip sha; gsp is always taken from the ga102 dir (its per-chip
# .fwsignature sections are selected at load time); fmc is fetched with one
# sha from the COT-boot (gb202) dir.
FW_ORDER = ["booter_load-570.144.bin", "bootloader-570.144.bin",
            "gsp-570.144.bin", "fmc-570.144.bin"]
FW_SINGLE_DIR = {"fmc-570.144.bin": "gb202"}

# ip.py also drives the RM control surface through the driver-generation
# module (its nv_gpu alias for the pinned 570 generation). Most of those
# symbols are already emitted by nv_defs.ml or nv_defs_versions.ml; the rest
# are curated here and emitted into nv_gsp_defs.ml from the 570 layouts. The
# census below asserts that this bucket is exactly the uncovered remainder of
# ip.py's nv_gpu usage, in both directions.

GSP_GPU_STRUCTS = [
    ("Nv2080_ctrl_gpu_promote_ctx_params", "NV2080_CTRL_GPU_PROMOTE_CTX_PARAMS",
     ["entryCount", "engineType", "hChanClient", "hObject", "promoteEntry/array"]),
    ("Nv2080_ctrl_gpu_promote_ctx_buffer_entry",
     "NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ENTRY",
     ["gpuPhysAddr", "gpuVirtAddr", "size", "physAttr", "bufferId",
      "bInitialize", "bNonmapped"]),
    ("Nv90f1_ctrl_vaspace_copy_server_reserved_pdes_params",
     "struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS",
     ["pageSize", "virtAddrLo", "virtAddrHi", "numLevelsToCopy", "levels/array"]),
    ("Nv90f1_ctrl_vaspace_copy_server_reserved_pdes_params_level",
     "struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_level",
     ["physAddress", "size", "aperture", "pageShift"]),
    ("Nv_memory_desc_params", "NV_MEMORY_DESC_PARAMS",
     ["base", "size", "addressSpace", "cacheAttrib"]),
    ("Nv2080_ctrl_internal_static_kgr_get_context_buffers_info_params",
     "NV2080_CTRL_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO_PARAMS",
     ["engineContextBuffersInfo/array"]),
    ("Nvb0cc_ctrl_internal_permissions_init_params",
     "NVB0CC_CTRL_INTERNAL_PERMISSIONS_INIT_PARAMS",
     ["bAdminProfilingPermitted", "bDevProfilingPermitted",
      "bCtxProfilingPermitted", "bVideoMemoryProfilingPermitted",
      "bSysMemoryProfilingPermitted"]),
    # The versioned record deliberately carries only the fields the
    # kernel-driver tier writes; the GSP tier builds the whole structure, so
    # the full 570 layout is emitted here.
    ("Nv_channelgpfifo_allocation_parameters",
     "NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS",
     ["hObjectError", "hObjectBuffer", "gpFifoOffset", "gpFifoEntries",
      "flags", "hContextShare", "hVASpace", "hUserdMemory/array",
      "userdOffset/array", "engineType", "cid", "subDeviceId",
      "hObjectEccError", "instanceMem/region", "userdMem/region",
      "ramfcMem/region", "mthdbufMem/region", "hPhysChannelGroup",
      "internalFlags", "errorNotifierMem/region", "eccErrorNotifierMem/region",
      "ProcessID", "SubProcessID", "encryptIv/array", "decryptIv/array",
      "hmacNonce/array", "tpcConfigID"]),
]

# Element layouts of the KGR context-buffers array, reached through the
# parameter structure rather than by name, so they sit outside the census.
GSP_GPU_NESTED_STRUCTS = [
    ("Nv2080_ctrl_internal_static_gr_context_buffers_info",
     "struct_NV2080_CTRL_INTERNAL_STATIC_GR_CONTEXT_BUFFERS_INFO",
     ["engine/array"]),
    ("Nv2080_ctrl_internal_engine_context_buffer_info",
     "struct_NV2080_CTRL_INTERNAL_ENGINE_CONTEXT_BUFFER_INFO",
     ["size", "alignment"]),
]

GSP_GPU_CONSTS = [
    "NV1_ROOT",
    "NV01_MEMORY_LIST_SYSTEM",
    "NV2080_CTRL_CMD_GPU_PROMOTE_CTX",
    "NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO",
    "NV90F1_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES",
    "NVB0CC_CTRL_CMD_POWER_REQUEST_FEATURES",
    "NVB0CC_CTRL_CMD_INTERNAL_PERMISSIONS_INIT",
    "NVB0CC_CTRL_CMD_ALLOC_PMA_STREAM",
    "NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS",
    "NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_PATCH",
    "NVOS02_FLAGS_ALLOC_USER_READ_ONLY_YES",
]


def ip_used_symbols():
    """Every `nv.`-qualified symbol referenced in ip.py."""
    src = open(_IP_PY).read()
    return set(re.findall(r"\bnv\.([A-Za-z_][A-Za-z0-9_]*)", src))


def ip_used_gpu_symbols():
    """Every `nv_gpu.`-qualified symbol referenced in ip.py."""
    src = open(_IP_PY).read()
    return set(re.findall(r"\bnv_gpu\.([A-Za-z_][A-Za-z0-9_]*)", src))


def nv_defs_covered_symbols():
    """The reference names nv_defs.ml and nv_defs_versions.ml already emit.

    The versioned NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS record carries only
    a field subset, so it does not count as covered here.
    """
    names = set()
    for lst in [CLASS_IDS, ESCAPE_NUMBERS, MEMORY_FLAGS, CHANNEL_INTS,
                CHANNEL_BIT_RANGES, CONTROL_INTS, ALLOCATION_FLAGS, ERROR_INTS,
                UVM_COMMANDS, QMD_INTS]:
        names |= set(lst)
    names |= {name for _, name, _ in STRUCTS}
    names |= {name for _, name, _ in PINNED_570_STRUCTS}
    names |= {name for _, _, name, _ in VERSIONED_STRUCTS}
    names.discard("NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS")
    return names


def gsp_const_line(mod, name):
    v = getattr(mod, name)
    if not isinstance(v, int) or v < 0:
        raise ValueError(f"{name}: not a non-negative int constant")
    if name in GSP_CONSTS64:
        return f"let {name.lower()} = 0x{v:x}L"
    if v.bit_length() > 62:
        raise ValueError(f"{name}: exceeds OCaml int; add it to GSP_CONSTS64")
    return int_let(name, v)


def gen_gsp_defs():
    emitted = {name for _, name, _ in GSP_STRUCTS} | set(GSP_CONSTS) \
        | {"rpc_fns", "rpc_events"}
    used = ip_used_symbols()
    if emitted != used:
        raise ValueError(
            "GSP symbol census drift against ip.py:\n"
            f"  no longer used (drop): {sorted(emitted - used)}\n"
            f"  newly used (add): {sorted(used - emitted)}")

    emitted_gpu = {name for _, name, _ in GSP_GPU_STRUCTS} | set(GSP_GPU_CONSTS)
    uncovered_gpu = ip_used_gpu_symbols() - nv_defs_covered_symbols()
    if emitted_gpu != uncovered_gpu:
        raise ValueError(
            "GSP nv_gpu symbol census drift against ip.py:\n"
            f"  no longer used (drop): {sorted(emitted_gpu - uncovered_gpu)}\n"
            f"  newly used (add): {sorted(uncovered_gpu - emitted_gpu)}")

    lines = [HEADER]
    lines.append("(* Scalar constants: RPC framing, message and event function"
                 " ids, libos and\n   GSP DMA constants, VBIOS/FALCON/BIT"
                 " parsing offsets. *)")
    lines += [gsp_const_line(nv, nm) for nm in GSP_CONSTS]
    lines.append("")

    lines.append("(* Structure layouts: per used field (byte offset, byte size)"
                 " and total size;\n   arrays as offset/element-size/count. *)")
    for title, name, fields in GSP_STRUCTS:
        lines += struct_module(title, getattr(nv, name), fields)
        lines.append("")

    lines.append("(* RM control surface of the golden-image bring-up, emitted"
                 " from the pinned\n   570 generation: class and control ids,"
                 " flag values, and the parameter\n   layouts not already"
                 " carried by the shared or versioned tables. *)")
    lines += [gsp_const_line(nv_570, nm) for nm in GSP_GPU_CONSTS]
    lines.append("")
    for title, name, fields in GSP_GPU_STRUCTS + GSP_GPU_NESTED_STRUCTS:
        lines += struct_module(title, getattr(nv_570, name), fields)
        lines.append("")

    lines.append("(* RPC function and event id -> name, for debug output. *)")
    for value_name, table in [("rpc_fns", nv.rpc_fns), ("rpc_events", nv.rpc_events)]:
        lines.append(f"let {value_name} = [")
        lines += assoc_lines(table.items())
        lines.append("]")
        lines.append("")

    lines.append("(* Firmware sha256 digests, pinned to the 570.144 GSP images,"
                 " by driver dir. *)")
    fw = firmware_hashes()
    lines.append("let firmware = [")
    for fname in FW_ORDER:
        entries = "; ".join(f'("{chip}", "{sha}")' for chip, sha in fw[fname])
        lines.append(f'  ("{fname}", [ {entries} ]);')
    lines.append("]")
    lines.append("")
    lines.append("(* The pinned linux-firmware tree the digests were taken from;"
                 "\n   a file lives at <upstream>/nvidia/<dir>/gsp/<name>. *)")
    lines.append(f'let upstream =\n  "{linux_firmware_upstream()}"')
    return lines


def firmware_hashes():
    """{filename: [(chip dir, sha256), ...]} extracted from ip.py's fetch_fw
    calls without executing them."""
    tree = ast.parse(open(_IP_PY).read())

    def const_str(node):
        return node.value if isinstance(node, ast.Constant) \
            and isinstance(node.value, str) else None

    def sha_dict(node):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Dict):
            return {const_str(k): const_str(v)
                    for k, v in zip(node.value.keys, node.value.values)}
        return None

    fw = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        local = {}
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name):
                d = sha_dict(node.value)
                if d is not None:
                    local[node.targets[0].id] = d
        for call in ast.walk(fn):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id == "fetch_fw" and len(call.args) == 3):
                continue
            path_node, name_node, sha_node = call.args
            fname = const_str(name_node)
            if fname is None:
                continue
            fixed_dir = None
            if isinstance(path_node, ast.Constant):
                m = re.match(r"nvidia/(\w+)/gsp", path_node.value)
                fixed_dir = m.group(1) if m else None
            if isinstance(sha_node, ast.Constant):
                entry = {fixed_dir or FW_SINGLE_DIR[fname]: sha_node.value}
            elif sha_dict(sha_node) is not None:
                entry = sha_dict(sha_node)
            elif isinstance(sha_node, ast.Name) and sha_node.id in local:
                entry = dict(local[sha_node.id])
            else:
                raise ValueError(f"{fname}: cannot resolve sha argument")
            fw.setdefault(fname, {}).update(entry)

    if set(fw) != set(FW_ORDER):
        raise ValueError(f"firmware file set drift: {sorted(fw)} vs {FW_ORDER}")
    out = {}
    for fname, chips in fw.items():
        for chip, sha in chips.items():
            if not re.fullmatch(r"[0-9a-f]{64}", sha or ""):
                raise ValueError(f"{fname}/{chip}: malformed sha256 {sha!r}")
        out[fname] = list(chips.items())
    return out


def linux_firmware_upstream():
    pin = re.search(r"kernel-firmware/linux-firmware/-/raw/([0-9a-f]{40})/",
                    inspect.getsource(helpers.fetch_fw)).group(1)
    return f"https://gitlab.com/kernel-firmware/linux-firmware/-/raw/{pin}"


def reg_entry_ml(value):
    """One nv_regs entry as an OCaml `entry` value."""
    if isinstance(value, int):
        return f"Const {ml_int(value)}"
    base, off, fields = value
    parts = "; ".join(f'("{k}", ({lo}, {hi}))' for k, (lo, hi) in fields.items())
    fields_ml = f"[ {parts} ]" if fields else "[]"
    if base is None and off is None:
        return f"Group {{ fields = {fields_ml} }}"
    if callable(off):
        o0, o1, o2 = off(0), off(1), off(2)
        stride = o1 - o0
        if o2 != o0 + 2 * stride:
            raise ValueError(f"non-affine offset lambda: {o0}, {o1}, {o2}")
        off_ml = f"Indexed {{ base = {ml_int(o0)}; stride = {ml_int(stride)} }}"
    else:
        off_ml = f"Fixed {ml_int(off)}"
    return f"Reg {{ base = {ml_int(base)}; off = {off_ml}; fields = {fields_ml} }}"


def gen_reg_defs():
    lines = [HEADER]
    lines.append("""\
(* Per-family, per-architecture NVIDIA register maps. Each entry is a raw
   constant, a register (block base offset, register offset, and named bitfield
   ranges as inclusive (lo, hi) bit positions), or a bitfield-only group. An
   indexed register's offset is the affine form base + stride * i. *)

type off = Fixed of int | Indexed of { base : int; stride : int }

type field = string * (int * int)

type entry =
  | Const of int
  | Reg of { base : int; off : off; fields : field list }
  | Group of { fields : field list }
""")

    bindings = []
    for family in nv_regs.__all__:
        mod = importlib.import_module(f"tinygrad.runtime.autogen.nv_regs.{family}")
        arches = [(k, v) for k, v in vars(mod).items()
                  if isinstance(v, dict) and not k.startswith("__")]
        if not arches:
            raise ValueError(f"nv_regs.{family}: no register dicts")
        family_bindings = []
        for arch, table in arches:
            binding = f"{family}_{arch}"
            family_bindings.append((arch, binding))
            lines.append(f"let {binding} : (string * entry) list = [")
            lines += [f'  ("{name}", {reg_entry_ml(value)});'
                      for name, value in table.items()]
            lines.append("]")
            lines.append("")
        bindings.append((family, family_bindings))

    lines.append("(* family -> arch -> entries, mirroring nvdev.py include(). *)")
    lines.append("let families "
                 ": (string * (string * (string * entry) list) list) list = [")
    for family, family_bindings in bindings:
        archs = "; ".join(f'("{arch}", {binding})'
                          for arch, binding in family_bindings)
        lines.append(f'  ("{family}", [ {archs} ]);')
    lines.append("]")
    return lines


def main():
    outputs = {
        "nv_defs.ml": gen_defs,
        "nv_defs_versions.ml": gen_versions,
        os.path.join("nv", "nv_reg_defs.ml"): gen_reg_defs,
        os.path.join("nv", "nv_gsp_defs.ml"): gen_gsp_defs,
    }
    for fname, gen in outputs.items():
        text = "\n".join(gen()) + "\n"
        path = os.path.join(_HERE, fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
        print(f"wrote {fname} ({text.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
