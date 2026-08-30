#!/usr/bin/env python3
"""Emit the OCaml NVIDIA driver tables (nv_defs.ml, nv_defs_versions.ml).

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

import ctypes
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIN = os.path.join(_HERE, "..", "..", "..", "..", "..", "_tinygrad")
sys.path.insert(0, _PIN)

from tinygrad.runtime.autogen import nv_570, nv_580, nv_610  # noqa: E402

MODULES = [nv_570, nv_580, nv_610]

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
RENAMES = {"type": "typ"}

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


def main():
    outputs = {
        "nv_defs.ml": gen_defs,
        "nv_defs_versions.ml": gen_versions,
    }
    for fname, gen in outputs.items():
        text = "\n".join(gen()) + "\n"
        with open(os.path.join(_HERE, fname), "w") as f:
            f.write(text)
        print(f"wrote {fname} ({text.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
