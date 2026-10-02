#!/usr/bin/env python3
"""Generates nv_defs.ml, the NVIDIA definitions nx.nv.device reads.

Run from the repository root:

  uv run --with libclang==18.1.1 packages/nx/lib/nv/device/gen/gen.py
  uv run --with libclang==18.1.1 packages/nx/lib/nv/device/gen/gen.py --check

Every input is pinned by URL and SHA-256 in pins.json: NVIDIA's open kernel
modules at each driver release the kernel interface supports, two headers of
Linux's nouveau driver, and the firmware files of linux-firmware at one commit.
Downloads are kept in --cache. The output is deterministic: --check generates
into a temporary directory and fails if the committed file differs.

The GSP firmware is release 570.144, so everything the driver-less interface
reads comes from that tree: the GSP's messages, the resource manager's
parameters it forwards, registers, page-table formats and class methods. The
kernel interface speaks to the installed driver, whose parameter layouts and
bit fields differ between releases: those that differ are emitted once per
release, the others once, and the script fails if the split changes.

Struct layouts come from libclang, for x86_64 Linux; the script checks that
aarch64 Linux lays them out the same. Structures upstream defines only inside
.c files are cut out of them by name. The VBIOS structures are laid out by
their format strings, which describe them as the ROM packs them. It emits only
what the runtime and the libraries that submit work to it read, such as the
method constants of tolk's pushbuffers: the inventories below name it.
"""

import glob
import hashlib
import json
import pathlib
import re
import sys
import tarfile

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent
sys.path.insert(0, str(HERE.parents[2] / "device" / "gen"))
from devgen import Unit, fetch, key, layout, main, ml_int, stub_dir  # noqa: E402

GITHUB = "https://github.com/NVIDIA/open-gpu-kernel-modules/archive/"
# The driver releases, the first of which is the GSP firmware's.
RELEASES = {
    570: GITHUB + "refs/tags/570.144.tar.gz",
    580: GITHUB + "2af9f1f0f7de4988432d4ae875b5858ffdb09cc2.tar.gz",
    610: GITHUB + "refs/tags/610.43.03.tar.gz",
    615: GITHUB + "refs/tags/615.71.09.tar.gz",
}
GSP = 570
NVFW = "https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/drivers/gpu/drm/nouveau/include/nvfw/"
NVFW_FILES = ["fw.h", "hs.h"]
NVFW_TAG = "v6.18"
FIRMWARE_COMMIT = "0a6871b19abf5d6e024b5d208b101ae53e7fa0de"
FIRMWARE_RAW = f"https://gitlab.com/kernel-firmware/linux-firmware/-/raw/{FIRMWARE_COMMIT}/nvidia/{{name}}"
FIRMWARE = [
    "ga102/gsp/booter_load-570.144.bin", "ga102/gsp/bootloader-570.144.bin", "ga102/gsp/gsp-570.144.bin",
    "ad102/gsp/booter_load-570.144.bin", "ad102/gsp/bootloader-570.144.bin",
    "gb202/gsp/bootloader-570.144.bin", "gb202/gsp/fmc-570.144.bin",
]

# The tree's parts the headers below include.
KEEP = ["src/common/", "src/nvidia/inc/", "src/nvidia/interface/", "src/nvidia/arch/", "src/nvidia/generated/",
        "kernel-open/common/", "kernel-open/nvidia-uvm/", "src/nvidia/src/kernel/gpu/gsp/", "version.mk"]

# The resource manager's interface

RM_HEADERS = [
    "kernel-open/common/inc/nvmisc.h",
    *[f"src/common/sdk/nvidia/inc/class/cl{s}.h" for s in [
        "0000", "0070", "0080", "2080", "9067", "90f1", "a06c", "c56f", "c86f", "c96f", "c761", "83de", "c6c0", "cdc0"]],
    *[f"kernel-open/nvidia-uvm/{s}.h" for s in ["clc6b5", "clc7b5", "uvm_ioctl", "uvm_linux_ioctl"]],
    *[f"src/nvidia/arch/nvalloc/unix/include/nv{s}.h" for s in [
        "_escape", "-ioctl", "-ioctl-numbers", "-unix-nvos-params-wrappers"]],
    *[f"src/common/sdk/nvidia/inc/{s}.h" for s in [
        "alloc/alloc_channel", "nvos", "ctrl/ctrlc36f", "ctrl/ctrla06c", "ctrl/ctrl90f1"]],
    *[f"src/common/sdk/nvidia/inc/ctrl/ctrl{s}/*.h" for s in ["0000", "0080", "2080", "83de"]],
    "kernel-open/common/inc/nvstatus.h", "src/nvidia/generated/g_allclasses.h",
]
RM_INCLUDES = ["src/common/inc", "kernel-open/nvidia-uvm", "kernel-open/common/inc", "src/common/sdk/nvidia/inc",
               "src/nvidia/arch/nvalloc/unix/include", "src/common/sdk/nvidia/inc/ctrl"]

RM_CONSTANTS = [
    # classes
    "NV01_ROOT", "NV01_ROOT_CLIENT", "NV01_DEVICE_0", "NV20_SUBDEVICE_0", "NV01_MEMORY_VIRTUAL",
    "NV01_MEMORY_SYSTEM_OS_DESCRIPTOR", "NV1_MEMORY_SYSTEM", "NV1_MEMORY_USER", "FERMI_VASPACE_A",
    "FERMI_CONTEXT_SHARE_A", "KEPLER_CHANNEL_GROUP_A", "TURING_USERMODE_A", "HOPPER_USERMODE_A",
    "AMPERE_CHANNEL_GPFIFO_A", "BLACKWELL_CHANNEL_GPFIFO_A", "AMPERE_COMPUTE_B", "ADA_COMPUTE_A",
    "BLACKWELL_COMPUTE_A", "BLACKWELL_COMPUTE_B", "AMPERE_DMA_COPY_B", "BLACKWELL_DMA_COPY_B", "GT200_DEBUGGER",
    # escapes
    "NV_IOCTL_MAGIC", "NV_ESC_CARD_INFO", "NV_ESC_REGISTER_FD", "NV_ESC_RM_ALLOC", "NV_ESC_RM_ALLOC_MEMORY",
    "NV_ESC_RM_CONTROL", "NV_ESC_RM_FREE", "NV_ESC_RM_MAP_MEMORY", "NV_ESC_RM_MAP_MEMORY_DMA",
    # memory
    "NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS", "NVOS02_FLAGS_COHERENCY_CACHED", "NVOS02_FLAGS_MAPPING_NO_MAP",
    "NVOS32_ATTR_PHYSICALITY_CONTIGUOUS", "NVOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS",
    "NVOS32_ATTR_PAGE_SIZE_HUGE", "NVOS32_ATTR_LOCATION_VIDMEM", "NVOS32_ATTR_LOCATION_PCI",
    "NVOS32_ATTR2_GPU_CACHEABLE_YES", "NVOS32_ATTR2_GPU_CACHEABLE_NO", "NVOS32_ATTR2_PAGE_SIZE_HUGE_2MB",
    "NVOS32_ATTR2_ZBC_PREFER_NO_ZBC", "NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED",
    "NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED", "NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE",
    "NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT", "NVOS32_ALLOC_FLAGS_PERSISTENT_VIDMEM", "NVOS32_TYPE_IMAGE",
    "NVOS32_TYPE_NOTIFIER", "NVOS33_FLAGS_CACHING_TYPE_WRITECOMBINED", "NVOS46_FLAGS_PAGE_SIZE_4KB",
    "NVOS46_FLAGS_CACHE_SNOOP_ENABLE", "NVOS46_FLAGS_DMA_OFFSET_FIXED_TRUE",
    # channel methods
    "NVC56F_SEM_ADDR_LO", "NVC56F_SEM_ADDR_HI", "NVC56F_SEM_PAYLOAD_LO", "NVC56F_SEM_PAYLOAD_HI",
    "NVC56F_SEM_EXECUTE", "NVC56F_NON_STALL_INTERRUPT", "NVC56F_SEM_EXECUTE_OPERATION_ACQ_CIRC_GEQ",
    "NVC56F_SEM_EXECUTE_OPERATION_RELEASE", "NVC56F_SEM_EXECUTE_PAYLOAD_SIZE_64BIT",
    "NVC56F_SEM_EXECUTE_RELEASE_WFI_EN", "NVC56F_GP_ENTRY1_LEVEL_SUBROUTINE",
    "NVC6C0_SET_OBJECT", "NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A", "NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A",
    "NVC6C0_SET_SHADER_LOCAL_MEMORY_A", "NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A",
    "NVC6B5_OFFSET_IN_UPPER", "NVC6B5_LINE_LENGTH_IN", "NVC6B5_LAUNCH_DMA", "NVC6B5_SET_SEMAPHORE_A",
    "NVC6B5_LAUNCH_DMA_DATA_TRANSFER_TYPE_NONE", "NVC6B5_LAUNCH_DMA_DATA_TRANSFER_TYPE_NON_PIPELINED",
    "NVC6B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT_PITCH", "NVC6B5_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH",
    "NVC6B5_LAUNCH_DMA_FLUSH_ENABLE_TRUE", "NVC6B5_LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_ONE_WORD_SEMAPHORE",
    "NVC6B5_LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_FOUR_WORD_SEMAPHORE",
    # controls
    "NV0000_CTRL_CMD_SYSTEM_GET_BUILD_VERSION_V2", "NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2",
    "NV0080_CTRL_CMD_GPU_GET_CLASSLIST", "NV2080_CTRL_CMD_GPU_GET_GID_INFO",
    "NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY", "NV2080_CTRL_CMD_PERF_BOOST",
    "NV2080_CTRL_PERF_BOOST_FLAGS_CMD_BOOST_TO_MAX", "NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_YES",
    "NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_PRIORITY_HIGH", "NVA06C_CTRL_CMD_GPFIFO_SCHEDULE",
    "NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN", "NV2080_CTRL_CMD_GR_GET_INFO",
    "NV2080_CTRL_CMD_FB_FLUSH_GPU_CACHE", "NV2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_WRITE_BACK_YES",
    "NV2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_INVALIDATE_YES", "NV2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_FLUSH_MODE_FULL_CACHE",
    "NV2080_CTRL_CMD_FB_GET_INFO_V2", "NV2080_CTRL_FB_INFO_INDEX_HEAP_SIZE", "NV2080_CTRL_FB_INFO_INDEX_BAR1_SIZE",
    "NV2080_CTRL_CMD_INTERNAL_BUS_FLUSH_WITH_SYSMEMBAR", "NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_INFO",
    "NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO", "NV2080_CTRL_CMD_GPU_PROMOTE_CTX",
    "NV2080_CTRL_CMD_FIFO_GET_DEVICE_INFO_TABLE", "NV90F1_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES",
    "NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS",
    "NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_PATCH",
    "NV83DE_CTRL_CMD_DEBUG_READ_ALL_SM_ERROR_STATES", "NV83DE_CTRL_CMD_DEBUG_READ_MMU_FAULT_INFO",
    "NV2080_ENGINE_TYPE_GRAPHICS",
    "NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_GPCS", "NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_TPC_PER_GPC",
    "NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_SM_PER_TPC", "NV2080_CTRL_GR_INFO_INDEX_MAX_WARPS_PER_SM",
    "NV2080_CTRL_GR_INFO_INDEX_SM_VERSION",
    # allocations
    "NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES", "NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING",
    "NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED", "NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC",
    # statuses
    "NV_OK", "NV_ERR_NO_MEMORY",
    # UVM
    "UVM_INITIALIZE", "UVM_MM_INITIALIZE", "UVM_REGISTER_GPU", "UVM_UNREGISTER_GPU", "UVM_REGISTER_GPU_VASPACE",
    "UVM_UNREGISTER_GPU_VASPACE", "UVM_ENABLE_PEER_ACCESS",
    "UVM_REGISTER_CHANNEL", "UVM_CREATE_EXTERNAL_RANGE", "UVM_MAP_EXTERNAL_ALLOCATION", "UVM_UNMAP_EXTERNAL",
    "UVM_FREE", "UvmGpuMappingTypeReadWriteAtomic",
]

# Bit fields of 32-bit words, "hi:lo" in the headers: (lowest bit, bits).
RM_FIELDS = [
    "NVOS02_FLAGS_PHYSICALITY", "NVOS02_FLAGS_COHERENCY", "NVOS02_FLAGS_MAPPING", "NVOS32_ATTR_PHYSICALITY",
    "NVOS32_ATTR_PAGE_SIZE", "NVOS32_ATTR_LOCATION", "NVOS32_ATTR2_GPU_CACHEABLE", "NVOS32_ATTR2_PAGE_SIZE_HUGE",
    "NVOS32_ATTR2_ZBC", "NVOS33_FLAGS_CACHING_TYPE", "NVOS46_FLAGS_PAGE_SIZE", "NVOS46_FLAGS_CACHE_SNOOP",
    "NVOS46_FLAGS_DMA_OFFSET_FIXED",
    "NVC56F_SEM_ADDR_LO_OFFSET", "NVC56F_SEM_EXECUTE_OPERATION", "NVC56F_SEM_EXECUTE_PAYLOAD_SIZE",
    "NVC56F_SEM_EXECUTE_RELEASE_WFI", "NVC56F_GP_ENTRY0_GET", "NVC56F_GP_ENTRY1_GET_HI", "NVC56F_GP_ENTRY1_LEVEL",
    "NVC56F_GP_ENTRY1_LENGTH",
    "NVC6B5_LAUNCH_DMA_DATA_TRANSFER_TYPE", "NVC6B5_LAUNCH_DMA_FLUSH_ENABLE", "NVC6B5_LAUNCH_DMA_SEMAPHORE_TYPE",
    "NVC6B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT", "NVC6B5_LAUNCH_DMA_DST_MEMORY_LAYOUT",
    "NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT", "NV2080_CTRL_PERF_BOOST_FLAGS_CMD", "NV2080_CTRL_PERF_BOOST_FLAGS_CUDA",
    "NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_PRIORITY", "NV2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_WRITE_BACK",
    "NV2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_INVALIDATE", "NV2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_FLUSH_MODE",
]

# The bit fields whose ranges differ between the releases.
RM_FIELDS_PER_RELEASE = {"NV2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_WRITE_BACK",
                         "NV2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_INVALIDATE",
                         "NV2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_FLUSH_MODE"}

# Structs, by C name: the module they become and the fields read ("a__b" for a
# nested field). An array field is (offset, bytes of an element, count).
RM_STRUCTS = {
    "nv_ioctl_card_info_t": ("Card_info", ["valid", "pci_info__domain", "pci_info__bus", "pci_info__slot",
                                           "gpu_id", "minor_number"]),
    "nv_ioctl_register_fd_t": ("Register_fd", ["ctl_fd"]),
    "NVOS00_PARAMETERS": ("Nvos00", ["hRoot", "hObjectParent", "hObjectOld", "status"]),
    "NVOS02_PARAMETERS": ("Nvos02", ["hRoot", "hObjectParent", "hObjectNew", "hClass", "flags", "pMemory",
                                     "limit", "status"]),
    "NVOS21_PARAMETERS": ("Nvos21", ["hRoot", "hObjectParent", "hObjectNew", "hClass", "pAllocParms",
                                     "paramsSize", "status"]),
    "NVOS33_PARAMETERS": ("Nvos33", ["hClient", "hDevice", "hMemory", "offset", "length", "pLinearAddress",
                                     "status", "flags"]),
    "NVOS46_PARAMETERS": ("Nvos46", ["hClient", "hDevice", "hDma", "hMemory", "offset", "length", "flags",
                                     "dmaOffset", "status"]),
    "NVOS54_PARAMETERS": ("Nvos54", ["hClient", "hObject", "cmd", "flags", "params", "paramsSize", "status"]),
    "nv_ioctl_nvos02_parameters_with_fd": ("Nvos02_with_fd", ["params", "fd"]),
    "nv_ioctl_nvos33_parameters_with_fd": ("Nvos33_with_fd", ["params", "fd"]),
    "NV0000_ALLOC_PARAMETERS": ("Nv0000_alloc", ["hClient"]),
    "NV0080_ALLOC_PARAMETERS": ("Nv0080_alloc", ["deviceId", "hClientShare", "vaMode"]),
    "NV2080_ALLOC_PARAMETERS": ("Nv2080_alloc", ["subDeviceId"]),
    "NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS": ("Memory_virtual_alloc", ["offset", "limit", "hVASpace"]),
    "NV_MEMORY_ALLOCATION_PARAMS": ("Memory_alloc", ["owner", "type", "flags", "attr", "attr2", "format", "size",
                                                     "alignment", "offset", "limit"]),
    "NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS": ("Channel_group_alloc", ["engineType"]),
    "NV_CTXSHARE_ALLOCATION_PARAMETERS": ("Ctxshare_alloc", ["hVASpace", "flags"]),
    "NV_MEMORY_DESC_PARAMS": ("Memory_desc", ["base", "size", "addressSpace", "cacheAttrib"]),
    "NV83DE_ALLOC_PARAMETERS": ("Nv83de_alloc", ["hAppClient", "hClass3dObject"]),
    "NV0000_CTRL_SYSTEM_GET_BUILD_VERSION_V2_PARAMS": ("Build_version", ["driverVersionBuffer"]),
    "NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS": ("Id_info", ["gpuId", "deviceInstance"]),
    "NV0080_CTRL_GPU_GET_CLASSLIST_PARAMS": ("Classlist", ["numClasses", "classList"]),
    "NV2080_CTRL_GPU_GET_GID_INFO_PARAMS": ("Gid_info", ["flags", "length", "data"]),
    "NV2080_CTRL_PERF_BOOST_PARAMS": ("Perf_boost", ["flags", "duration"]),
    "NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS": ("Work_submit_token", ["workSubmitToken"]),
    "NV2080_CTRL_GR_INFO": ("Gr_info", ["index", "data"]),
    "NV2080_CTRL_GR_GET_INFO_PARAMS": ("Gr_get_info", ["grInfoListSize", "grInfoList"]),
    "NV2080_CTRL_FB_FLUSH_GPU_CACHE_PARAMS": ("Flush_gpu_cache", ["flags"]),
    "NV2080_CTRL_FB_INFO": ("Fb_info", ["index", "data"]),
    "NV2080_CTRL_FB_GET_INFO_V2_PARAMS": ("Fb_get_info", ["fbInfoListSize", "fbInfoList"]),
    "NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS": ("Sm_error_states", [
        "hTargetChannel", "numSMsToRead", "smErrorStateArray", "mmuFault__valid"]),
    "NV83DE_SM_ERROR_STATE_REGISTERS": ("Sm_error_state", ["hwwGlobalEsr", "hwwWarpEsr", "hwwWarpEsrPc64"]),
    "NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_PARAMS": ("Mmu_fault_info", ["count", "mmuFaultInfoList"]),
    "NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_ENTRY": ("Mmu_fault_entry", ["faultAddress", "faultType", "accessType"]),
    "NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_PARAMS": ("Device_info_table", ["numEntries", "entries"]),
    "NV2080_CTRL_FIFO_DEVICE_ENTRY": ("Device_entry", ["engineData"]),
    "NV2080_CTRL_INTERNAL_STATIC_GR_GET_INFO_PARAMS": ("Static_gr_info", ["engineInfo"]),
    "NV2080_CTRL_INTERNAL_STATIC_GR_INFO": ("Gr_info_list", ["infoList"]),
    "NV2080_CTRL_INTERNAL_GR_INFO": ("Internal_gr_info", ["data"]),
    "NV2080_CTRL_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO_PARAMS": ("Context_buffers_info", [
        "engineContextBuffersInfo"]),
    "NV2080_CTRL_INTERNAL_STATIC_GR_CONTEXT_BUFFERS_INFO": ("Context_buffers", ["engine"]),
    "NV2080_CTRL_INTERNAL_ENGINE_CONTEXT_BUFFER_INFO": ("Context_buffer", ["size", "alignment"]),
    "NV2080_CTRL_GPU_PROMOTE_CTX_PARAMS": ("Promote_ctx", [
        "engineType", "hClient", "ChID", "hChanClient", "hObject", "hVirtMemory", "virtAddress", "size",
        "entryCount", "promoteEntry"]),
    "NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ENTRY": ("Promote_entry", [
        "gpuPhysAddr", "gpuVirtAddr", "size", "physAttr", "bufferId", "bInitialize", "bNonmapped"]),
    "NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS": ("Reserved_pdes", [
        "hSubDevice", "subDeviceId", "pageSize", "virtAddrLo", "virtAddrHi", "numLevelsToCopy", "levels",
        "levels__physAddress", "levels__size", "levels__aperture", "levels__pageShift"]),
    "Nvc56fControl": ("Userd", ["GPGet", "GPPut"]),
    "UVM_INITIALIZE_PARAMS": ("Uvm_initialize", ["flags", "rmStatus"]),
    "UVM_MM_INITIALIZE_PARAMS": ("Uvm_mm_initialize", ["uvmFd", "rmStatus"]),
    "UVM_REGISTER_GPU_PARAMS": ("Uvm_register_gpu", ["gpu_uuid", "rmCtrlFd", "hClient", "hSmcPartRef", "rmStatus"]),
    "UVM_REGISTER_GPU_VASPACE_PARAMS": ("Uvm_register_gpu_vaspace", [
        "gpuUuid", "rmCtrlFd", "hClient", "hVaSpace", "rmStatus"]),
    "UVM_UNREGISTER_GPU_PARAMS": ("Uvm_unregister_gpu", ["gpu_uuid", "rmStatus"]),
    "UVM_UNREGISTER_GPU_VASPACE_PARAMS": ("Uvm_unregister_gpu_vaspace", ["gpuUuid", "rmStatus"]),
    "UVM_ENABLE_PEER_ACCESS_PARAMS": ("Uvm_enable_peer_access", ["gpuUuidA", "gpuUuidB", "rmStatus"]),
    "UVM_REGISTER_CHANNEL_PARAMS": ("Uvm_register_channel", [
        "gpuUuid", "rmCtrlFd", "hClient", "hChannel", "base", "length", "rmStatus"]),
    "UVM_CREATE_EXTERNAL_RANGE_PARAMS": ("Uvm_create_external_range", ["base", "length", "rmStatus"]),
    "UVM_MAP_EXTERNAL_ALLOCATION_PARAMS": ("Uvm_map_external_allocation", [
        "base", "length", "offset", "perGpuAttributes", "gpuAttributesCount", "rmCtrlFd", "hClient", "hMemory",
        "rmStatus"]),
    "UvmGpuMappingAttributes": ("Uvm_gpu_mapping", ["gpuUuid", "gpuMappingType"]),
    "UVM_UNMAP_EXTERNAL_PARAMS": ("Uvm_unmap_external", ["base", "length", "gpuUuid", "rmStatus"]),
    "UVM_FREE_PARAMS": ("Uvm_free", ["base", "length", "rmStatus"]),
}

# The structs only the GSP is sent, which are its release's.
RM_GSP_ONLY = {
    "NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_PARAMS", "NV2080_CTRL_FIFO_DEVICE_ENTRY",
    "NV2080_CTRL_INTERNAL_STATIC_GR_GET_INFO_PARAMS", "NV2080_CTRL_INTERNAL_STATIC_GR_INFO",
    "NV2080_CTRL_INTERNAL_GR_INFO", "NV2080_CTRL_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO_PARAMS",
    "NV2080_CTRL_INTERNAL_STATIC_GR_CONTEXT_BUFFERS_INFO", "NV2080_CTRL_INTERNAL_ENGINE_CONTEXT_BUFFER_INFO",
    "NV2080_CTRL_GPU_PROMOTE_CTX_PARAMS", "NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ENTRY",
    "NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS", "NV_MEMORY_DESC_PARAMS"}

# The structs whose layouts differ between the releases.
RM_PER_RELEASE = {"NV2080_CTRL_FB_GET_INFO_V2_PARAMS", "NVOS46_PARAMETERS", "NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS", "NV_VASPACE_ALLOCATION_PARAMETERS",
                  "NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS", "UVM_FREE_PARAMS", "NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS"}
RM_STRUCTS.update({
    "NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS": ("Gpfifo_alloc", [
        "gpFifoOffset", "gpFifoEntries", "flags", "hContextShare", "hVASpace", "hUserdMemory", "userdOffset",
        "engineType", "cid", "hObjectError", "hObjectBuffer", "instanceMem", "userdMem", "ramfcMem", "mthdbufMem",
        "errorNotifierMem", "internalFlags"]),
    "NV_VASPACE_ALLOCATION_PARAMETERS": ("Vaspace_alloc", ["index", "flags", "vaSize", "vaBase"]),
    "NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS": ("Group_schedule", ["bEnable"]),
})

# The GSP's interface, from the firmware's release

GSP_HEADERS = [
    *[f"src/nvidia/inc/kernel/gpu/{s}.h" for s in [
        "fsp/kern_fsp_cot_payload", "gsp/gsp_init_args", "gsp/gsp_static_config"]],
    *[f"src/nvidia/arch/nvalloc/common/inc/{s}.h" for s in [
        "gsp/gspifpub", "gsp/gsp_fw_wpr_meta", "rmRiscvUcode", "fsp/fsp_nvdm_format", "rmgspseq"]],
    *[f"src/nvidia/inc/kernel/vgpu/{s}.h" for s in ["rpc_headers", "rpc_global_enums"]],
    "src/common/uproc/os/common/include/libos_init_args.h", "src/common/shared/msgq/inc/msgq/msgq_priv.h",
    "src/nvidia/generated/g_rpc-structures.h", "src/nvidia/generated/g_rpc-message-header.h",
]
GSP_INCLUDES = ["src/nvidia/generated", "src/common/inc", "src/nvidia/inc", "src/nvidia/interface",
                "src/nvidia/inc/kernel", "src/nvidia/inc/libraries", "src/nvidia/arch/nvalloc/common/inc",
                "kernel-open/nvidia-uvm", "kernel-open/common/inc", "src/common/sdk/nvidia/inc",
                "src/nvidia/arch/nvalloc/unix/include", "src/common/sdk/nvidia/inc/ctrl"]
GSP_DEFINES = ["RPC_MESSAGE_STRUCTURES", "RPC_STRUCTURES", "RPC_GENERIC_UNION"]

GSP_CONSTANTS = [
    "NV_VGPU_MSG_SIGNATURE_VALID", "NV_VGPU_MSG_RESULT_RPC_PENDING", "NV_VGPU_MSG_FUNCTION_CONTINUATION_RECORD",
    "NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC", "NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL",
    "NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY", "NV_VGPU_MSG_FUNCTION_GSP_SET_SYSTEM_INFO",
    "NV_VGPU_MSG_FUNCTION_SET_REGISTRY", "NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER",
    "NV_VGPU_MSG_EVENT_GSP_INIT_DONE", "NV_VGPU_MSG_EVENT_GSP_RUN_CPU_SEQUENCER", "NV_VGPU_MSG_EVENT_OS_ERROR_LOG",
    "NV_VGPU_MSG_EVENT_MMU_FAULT_QUEUED", "GSP_FW_WPR_META_REVISION", "GSP_FW_WPR_META_MAGIC",
    "LIBOS_MEMORY_REGION_CONTIGUOUS", "LIBOS_MEMORY_REGION_LOC_SYSMEM", "LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2",
    "GSP_DMA_TARGET_COHERENT_SYSTEM", "NVDM_TYPE_COT",
    "GSP_SEQ_BUF_OPCODE_REG_WRITE", "GSP_SEQ_BUF_OPCODE_REG_MODIFY", "GSP_SEQ_BUF_OPCODE_REG_POLL",
    "GSP_SEQ_BUF_OPCODE_DELAY_US", "GSP_SEQ_BUF_OPCODE_REG_STORE", "GSP_SEQ_BUF_OPCODE_CORE_RESET",
    "GSP_SEQ_BUF_OPCODE_CORE_START", "GSP_SEQ_BUF_OPCODE_CORE_WAIT_FOR_HALT", "GSP_SEQ_BUF_OPCODE_CORE_RESUME",
]

GSP_STRUCTS = {
    "msgqTxHeader": ("Msgq_tx_header", ["version", "size", "msgSize", "msgCount", "writePtr", "flags", "rxHdrOff",
                                        "entryOff"]),
    "msgqRxHeader": ("Msgq_rx_header", ["readPtr"]),
    "GSP_MSG_QUEUE_ELEMENT": ("Queue_element", ["checkSum", "seqNum", "elemCount", "rpc"]),
    "rpc_message_header_v": ("Rpc_header", ["header_version", "signature", "length", "function", "rpc_result",
                                            "rpc_result_private", "sequence", "rpc_message_data"]),
    "rpc_gsp_rm_alloc_v": ("Rpc_rm_alloc", ["hClient", "hParent", "hObject", "hClass", "status", "paramsSize",
                                            "flags", "params"]),
    "rpc_gsp_rm_control_v": ("Rpc_rm_control", ["hClient", "hObject", "cmd", "status", "paramsSize", "flags",
                                                "params"]),
    "rpc_set_page_directory_v": ("Rpc_set_page_directory", ["hClient", "hDevice", "pasid", "params"]),
    "NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS_v1E_05": ("Set_page_directory", [
        "physAddress", "numEntries", "flags", "hVASpace", "chId", "subDeviceId", "pasid"]),
    "rpc_unloading_guest_driver_v": ("Rpc_unloading", ["bInPMTransition", "bGc6Entering", "newLevel"]),
    "rpc_run_cpu_sequencer_v17_00": ("Rpc_cpu_sequencer", ["bufferSizeDWord", "cmdIndex", "regSaveArea",
                                                           "commandBuffer"]),
    "rpc_os_error_log_v17_00": ("Rpc_os_error_log", ["exceptType", "runlistId", "chid", "errString"]),
    "GspSystemInfo": ("System_info", ["gpuPhysAddr", "gpuPhysFbAddr", "gpuPhysInstAddr", "nvDomainBusDeviceFunc",
                                      "maxUserVa", "pciConfigMirrorBase", "pciConfigMirrorSize", "PCIDeviceID",
                                      "PCISubDeviceID", "PCIRevisionID", "bIsPassthru", "hostPageSize"]),
    "MESSAGE_QUEUE_INIT_ARGUMENTS": ("Queue_init_args", ["sharedMemPhysAddr", "pageTableEntryCount",
                                                         "cmdQueueOffset", "statQueueOffset"]),
    "GSP_ARGUMENTS_CACHED": ("Gsp_arguments", ["messageQueueInitArguments", "bDmemStack"]),
    "LibosMemoryRegionInitArgument": ("Libos_region", ["id8", "pa", "size", "kind", "loc"]),
    "GspFwWprMeta": ("Wpr_meta", [
        "magic", "revision", "sysmemAddrOfRadix3Elf", "sizeOfRadix3Elf", "sysmemAddrOfBootloader",
        "sizeOfBootloader", "bootloaderCodeOffset", "bootloaderDataOffset", "bootloaderManifestOffset",
        "sysmemAddrOfSignature", "sizeOfSignature", "gspFwRsvdStart", "nonWprHeapOffset", "nonWprHeapSize",
        "gspFwWprStart", "gspFwHeapOffset", "gspFwHeapSize", "gspFwOffset", "bootBinOffset", "frtsOffset",
        "frtsSize", "gspFwWprEnd", "fbSize", "vgaWorkspaceOffset", "vgaWorkspaceSize", "pmuReservedSize"]),
    "RM_RISCV_UCODE_DESC": ("Riscv_ucode_desc", ["monitorDataOffset", "monitorCodeOffset", "manifestOffset"]),
    "GSP_FMC_BOOT_PARAMS": ("Fmc_boot_params", ["bootGspRmParams", "gspRmParams"]),
    "GSP_ACR_BOOT_GSP_RM_PARAMS": ("Acr_boot_params", ["target", "gspRmDescSize", "gspRmDescOffset",
                                                       "bIsGspRmBoot"]),
    "GSP_RM_PARAMS": ("Rm_params", ["target", "bootArgsOffset"]),
    "NVDM_PAYLOAD_COT": ("Cot_payload", ["version", "size", "gspFmcSysmemOffset", "frtsSysmemOffset",
                                         "frtsSysmemSize", "frtsVidmemOffset", "frtsVidmemSize", "hash384",
                                         "publicKey", "signature", "gspBootArgsSysmemOffset"]),
}

# Definitions upstream makes inside .c files and generated headers too
# entangled to parse whole, cut out by name: (file, typedefs, macros).
CUTS = [
    ("src/nvidia/src/kernel/gpu/gsp/kernel_gsp_fwsec.c",
     ["BIT_HEADER_V1_00", "BIT_TOKEN_V1_00", "BIT_DATA_FALCON_DATA_V2", "FALCON_UCODE_TABLE_HDR_V1",
      "FALCON_UCODE_TABLE_ENTRY_V1", "FALCON_UCODE_DESC_HEADER", "FALCON_UCODE_DESC_V3"],
     ["BIT_HEADER_SIGNATURE", "BIT_TOKEN_FALCON_DATA", "BIT_HEADER_V1_00_FMT", "BIT_TOKEN_V1_00_SIZE_6",
      "BIT_TOKEN_V1_00_SIZE_8", "BIT_TOKEN_V1_00_FMT_SIZE_6", "BIT_TOKEN_V1_00_FMT_SIZE_8",
      "BIT_DATA_FALCON_DATA_V2_4_FMT", "BIT_DATA_FALCON_DATA_V2_SIZE_4", "FALCON_UCODE_TABLE_HDR_V1_6_FMT",
      "FALCON_UCODE_TABLE_ENTRY_V1_6_FMT", "FALCON_UCODE_ENTRY_APPID_FWSEC_PROD", "FALCON_UCODE_DESC_HEADER_FORMAT",
      "FALCON_UCODE_DESC_V3_44_FMT", "FALCON_UCODE_DESC_V3_SIZE_44", "BCRT30_RSA3K_SIG_SIZE",
      "NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_VERSION", "NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_VERSION_V3",
      "NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_SIZE"]),
    ("src/nvidia/src/kernel/gpu/gsp/arch/turing/kernel_gsp_frts_tu102.c",
     ["FALCON_APPLICATION_INTERFACE_HEADER_V1", "FALCON_APPLICATION_INTERFACE_ENTRY_V1",
      "FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3", "FWSECLIC_READ_VBIOS_DESC", "FWSECLIC_FRTS_REGION_DESC",
      "FWSECLIC_FRTS_CMD"],
     ["FALCON_APPLICATION_INTERFACE_ENTRY_ID_DMEMMAPPER", "FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3_CMD_FRTS",
      "FWSECLIC_READ_VBIOS_STRUCT_FLAGS", "FWSECLIC_FRTS_REGION_MEDIA_FB", "FWSECLIC_FRTS_REGION_SIZE_1MB_IN_4K"]),
    ("src/nvidia/src/kernel/gpu/gsp/arch/turing/kernel_gsp_vbios_tu102.c",
     [], ["NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_BASE", "NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_EXT"]),
    ("src/nvidia/generated/g_os_nvoc.h",
     ["PACKED_REGISTRY_ENTRY", "PACKED_REGISTRY_TABLE"], ["REGISTRY_TABLE_ENTRY_TYPE_DWORD"]),
    ("src/nvidia/inc/kernel/gpu/gsp/message_queue_priv.h", ["GSP_MSG_QUEUE_ELEMENT"], []),
]
CUT_HEADERS = ["src/nvidia/inc/kernel/platform/pci_exp_table.h"]
CUT_CONSTANTS = [
    "BIT_HEADER_SIGNATURE", "BIT_TOKEN_FALCON_DATA", "BIT_TOKEN_V1_00_SIZE_6", "BIT_TOKEN_V1_00_SIZE_8",
    "BIT_DATA_FALCON_DATA_V2_SIZE_4", "FALCON_UCODE_ENTRY_APPID_FWSEC_PROD", "FALCON_UCODE_DESC_V3_SIZE_44",
    "BCRT30_RSA3K_SIG_SIZE", "NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_VERSION_V3",
    "FALCON_APPLICATION_INTERFACE_ENTRY_ID_DMEMMAPPER", "FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3_CMD_FRTS",
    "FWSECLIC_READ_VBIOS_STRUCT_FLAGS", "FWSECLIC_FRTS_REGION_MEDIA_FB", "FWSECLIC_FRTS_REGION_SIZE_1MB_IN_4K",
    "NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_BASE", "NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_EXT",
    "REGISTRY_TABLE_ENTRY_TYPE_DWORD", "OFFSETOF_PCI_EXP_ROM_PCI_DATA_STRUCT_PTR",
    "OFFSETOF_PCI_DATA_STRUCT_IMAGE_LEN", "OFFSETOF_PCI_DATA_STRUCT_CODE_TYPE", "PCI_ROM_IMAGE_BLOCK_SIZE",
]
CUT_FIELDS = ["NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_VERSION", "NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_SIZE"]
CUT_STRUCTS = {
    "PACKED_REGISTRY_ENTRY": ("Registry_entry", ["nameOffset", "type", "data", "length"]),
    "PACKED_REGISTRY_TABLE": ("Registry_table", ["size", "numEntries", "entries"]),
    "GSP_MSG_QUEUE_ELEMENT": ("Queue_element", ["checkSum", "seqNum", "elemCount", "rpc"]),
    "FALCON_APPLICATION_INTERFACE_HEADER_V1": ("App_interface_header", ["headerSize", "entrySize", "entryCount"]),
    "FALCON_APPLICATION_INTERFACE_ENTRY_V1": ("App_interface_entry", ["id", "dmemOffset"]),
    "FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3": ("Dmem_mapper", ["cmd_in_buffer_offset", "init_cmd"]),
    "FWSECLIC_READ_VBIOS_DESC": ("Read_vbios_desc", ["version", "size", "gfwImageOffset", "gfwImageSize",
                                                     "flags"]),
    "FWSECLIC_FRTS_REGION_DESC": ("Frts_region_desc", ["version", "size", "frtsRegionOffset4K", "frtsRegionSize",
                                                       "frtsRegionMediaType"]),
    "FWSECLIC_FRTS_CMD": ("Frts_cmd", ["readVbiosDesc", "frtsRegionDesc"]),
}
# ROM structures: (C name, module, format macro).
ROM_STRUCTS = [
    ("BIT_HEADER_V1_00", "Bit_header", "BIT_HEADER_V1_00_FMT"),
    ("BIT_TOKEN_V1_00", "Bit_token_6", "BIT_TOKEN_V1_00_FMT_SIZE_6"),
    ("BIT_TOKEN_V1_00", "Bit_token_8", "BIT_TOKEN_V1_00_FMT_SIZE_8"),
    ("BIT_DATA_FALCON_DATA_V2", "Falcon_data", "BIT_DATA_FALCON_DATA_V2_4_FMT"),
    ("FALCON_UCODE_TABLE_HDR_V1", "Ucode_table_header", "FALCON_UCODE_TABLE_HDR_V1_6_FMT"),
    ("FALCON_UCODE_TABLE_ENTRY_V1", "Ucode_table_entry", "FALCON_UCODE_TABLE_ENTRY_V1_6_FMT"),
    ("FALCON_UCODE_DESC_HEADER", "Ucode_desc_header", "FALCON_UCODE_DESC_HEADER_FORMAT"),
    ("FALCON_UCODE_DESC_V3", "Ucode_desc_v3", "FALCON_UCODE_DESC_V3_44_FMT"),
]
NVFW_STRUCTS = {
    "nvfw_bin_hdr": ("Bin_header", ["bin_magic", "header_offset", "data_offset", "data_size"]),
    "nvfw_hs_header_v2": ("Hs_header", ["sig_prod_offset", "sig_prod_size", "patch_loc", "patch_sig", "num_sig",
                                        "header_offset"]),
    "nvfw_hs_load_header_v2": ("Hs_load_header", ["os_data_offset", "os_data_size", "app"]),
}

# Registers

SWREF = "src/common/inc/swref/published"
HWREF = "kernel-open/nvidia-uvm/hwref"
ARCH_DIR = {"tu102": "turing", "ga100": "ampere", "ga102": "ampere", "gh100": "hopper", "gb202": "blackwell"}
# (header, arch) in the order a runtime includes them; the ga102 addenda are
# appended to their headers.
REG_FILES = [
    ("nv_ref", None), ("dev_fb", "tu102"), ("dev_gc6_island", "ga102"), ("dev_vm", "tu102"), ("dev_vm", "gh100"),
    ("dev_gsp", "ga102"), ("dev_falcon_v4", "ga102"), ("dev_falcon_v4", "gh100"), ("dev_riscv_pri", "ga102"),
    ("dev_fbif_v4", "ga102"), ("dev_falcon_second_pri", "ga102"), ("dev_sec_pri", "ga102"), ("dev_bus", "tu102"),
    ("dev_fsp_pri", "gh100"), ("dev_therm", "gb202"),
]
ADDENDA = {("dev_gc6_island", "ga102"), ("dev_falcon_v4", "ga102")}
# The offsets of the register groups in their unit's range: a falcon's
# registers are relative to the falcon.
REG_BASES = {"NV_PRISCV_RISCV": 0x1000, "NV_PFALCON_FBIF": 0x600, "NV_PFALCON2_FALCON": 0x1000,
             "NV_VIRTUAL_FUNCTION": 0xb80000}
REG_PREFIXES = ["NV_PFALCON_FALCON", "NV_PGSP_FALCON", "NV_PSEC_FALCON", "NV_PRISCV_RISCV", "NV_PGC6_AON",
                "NV_PFSP", "NV_PGC6_BSI", "NV_PFALCON_FBIF", "NV_PFALCON2_FALCON", "NV_PBUS", "NV_PFB", "NV_PMC",
                "NV_PGSP_QUEUE", "NV_VIRTUAL_FUNCTION", "NV_THERM"]
REGISTERS = [
    "NV_PMC_BOOT_0", "NV_PMC_BOOT_42", "NV_PFB_PRI_MMU_WPR2_ADDR_LO", "NV_PFB_PRI_MMU_WPR2_ADDR_HI", "NV_PGC6_AON_SECURE_SCRATCH_GROUP_42",
    "NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK", "NV_PGC6_AON_SECURE_SCRATCH_GROUP_05",
    "NV_PGC6_BSI_SECURE_SCRATCH_14", "NV_THERM_I2CS_SCRATCH", "NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE",
    "NV_VIRTUAL_FUNCTION_PRIV_FUNC_BAR1_BLOCK_LOW_ADDR", "NV_PBUS_BAR1_BLOCK", "NV_PGSP_QUEUE_HEAD",
    "NV_PGSP_FALCON_MAILBOX0", "NV_PGSP_FALCON_MAILBOX1", "NV_PGSP_FALCON_ENGINE", "NV_PSEC_FALCON_ENGINE",
    "NV_PFALCON_FALCON_OS", "NV_PFALCON_FALCON_DMATRFCMD", "NV_PFALCON_FALCON_DMATRFBASE",
    "NV_PFALCON_FALCON_DMATRFBASE1", "NV_PFALCON_FALCON_DMATRFMOFFS", "NV_PFALCON_FALCON_DMATRFFBOFFS",
    "NV_PFALCON_FALCON_CPUCTL", "NV_PFALCON_FALCON_BOOTVEC", "NV_PFALCON_FALCON_MAILBOX0", "NV_PFALCON_FALCON_MAILBOX1",
    "NV_PFALCON_FALCON_DMACTL", "NV_PFALCON_FALCON_HWCFG2", "NV_PFALCON_FALCON_RM", "NV_PFALCON_FBIF_TRANSCFG",
    "NV_PFALCON_FBIF_CTL", "NV_PFALCON2_FALCON_BROM_PARAADDR", "NV_PFALCON2_FALCON_BROM_ENGIDMASK",
    "NV_PFALCON2_FALCON_BROM_CURR_UCODE_ID", "NV_PFALCON2_FALCON_MOD_SEL", "NV_PRISCV_RISCV_CPUCTL",
    "NV_PRISCV_RISCV_BCR_CTRL", "NV_PFSP_EMEMC", "NV_PFSP_EMEMD", "NV_PFSP_QUEUE_HEAD", "NV_PFSP_QUEUE_TAIL",
    "NV_PFSP_MSGQ_HEAD", "NV_PFSP_MSGQ_TAIL",
]
REG_VALUES = ["NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B", "NV_PFALCON_FBIF_TRANSCFG_MEM_TYPE_PHYSICAL",
              "NV_PFALCON2_FALCON_MOD_SEL_ALGO_RSA3K", "NV_PFALCON_FALCON_CPUCTL_ALIAS"]
MMU = {2: "tu102", 3: "gh100"}

# Parsing


def sources(cache, pins, pin):
    """The trees of each release, and the directory of the nouveau headers, each
    under the digest of the URL it comes from, so that a moved pin never reads a
    tree extracted for another."""
    trees = {}
    for rel, url in RELEASES.items():
        tar = fetch(cache, url, pins, pin)
        root = cache / "src" / key(url)
        if not root.exists():
            partial = root.with_suffix(".partial")
            with tarfile.open(tar) as t:
                top = t.getnames()[0].split("/")[0]
                members = [m for m in t.getmembers()
                           if any(m.name[len(top) + 1:].startswith(k) for k in KEEP)]
                for m in members:
                    m.name = m.name[len(top) + 1:]
                t.extractall(partial, members=members, filter="data")
            partial.rename(root)
        trees[rel] = root
    nvfw = cache / "src" / key(NVFW + NVFW_TAG)
    nvfw.mkdir(parents=True, exist_ok=True)
    for f in NVFW_FILES:
        # The headers include the kernel's own, which only functions need.
        text = fetch(cache, f"{NVFW}{f}?h={NVFW_TAG}", pins, pin).read_text()
        (nvfw / f).write_text(text.replace("#include <core/os.h>", ""))
    return trees, nvfw


def version(tree):
    return re.search(r"NVIDIA_VERSION = (\S+)", (tree / "version.mk").read_text()).group(1)


def headers(tree, names):
    out = []
    for n in names:
        found = sorted(glob.glob(str(tree / n)))
        if not found:
            sys.exit(f"{tree}: no {n}")
        out += [pathlib.Path(p) for p in found]
    return out


def nvtypes(tree):
    return f'#include "{tree}/src/common/sdk/nvidia/inc/nvtypes.h"\n'


def field_ranges(files, names):
    """The "hi:lo" bit ranges of [names], as (lowest bit, bits)."""
    out = {}
    for f in files:
        for m in re.finditer(r"^\s*#define\s+(\w+)\s+\(?\s*(\d+)\s*:\s*(\d+)\s*\)?\s*(?:/[/*].*)?$",
                             pathlib.Path(f).read_text(errors="replace"), re.M):
            if m.group(1) in names and m.group(1) not in out:
                hi, lo = int(m.group(2)), int(m.group(3))
                out[m.group(1)] = (lo, hi - lo + 1)
    missing = [n for n in names if n not in out]
    if missing:
        sys.exit(f"no bit ranges for {missing}")
    return out


KEYWORDS = {"type", "function", "method", "val", "open", "end", "include", "module", "object", "private"}


def snake(name):
    """The OCaml value for the C name [name]: NV01_ROOT is nv01_root and
    hObjectParent h_object_parent; nested fields a__b become a_b."""
    if name.upper() == name:
        s = name.lower()
    else:
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name.replace("__", "_"))
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s).lower()
    return s + "_" if s in KEYWORDS else s


def arrays(ci, cursor):
    """The element counts of the array fields, and the fields of their
    elements that are structs, relative to an element, by path."""
    counts, elements = {}, {}

    def walk(t, prefix):
        for f in t.get_fields():
            ft = f.type.get_canonical()
            anonymous = not f.spelling or "(anonymous" in f.spelling
            path = prefix if anonymous else (prefix + "__" if prefix else "") + f.spelling
            if ft.kind in (ci.TypeKind.CONSTANTARRAY, ci.TypeKind.INCOMPLETEARRAY):
                counts[path] = ft.get_array_size() if ft.kind == ci.TypeKind.CONSTANTARRAY else 0
                et = ft.get_array_element_type().get_canonical()
                if et.kind == ci.TypeKind.RECORD:
                    elements.update({f"{path}__{k}": v for k, v in layout(ci, et.get_declaration())[1].items()
                                     if v[0] != "bits"})
            elif ft.kind == ci.TypeKind.RECORD:
                walk(ft, path)

    walk(cursor.type.get_canonical(), "")
    return counts, elements


def struct_fields(ci, unit, cname):
    c = unit.struct(cname)
    if c is None:
        sys.exit(f"no struct {cname}")
    size, fields = layout(ci, c)
    counts, elements = arrays(ci, c)
    fields = {k: (v[0], v[1], counts[k]) if k in counts else v for k, v in fields.items() if v[0] != "bits"}
    return size, {**elements, **fields}


def emit_struct(out, module, size, fields, wanted, indent=""):
    out.append(f"{indent}module {module} = struct")
    out.append(f"{indent}  let sizeof = {size}")
    for f in wanted:
        if f not in fields:
            sys.exit(f"{module} has no field {f}; it has {sorted(fields)}")
        out.append(f"{indent}  let {snake(f)} = {ml_tuple(fields[f])}")
    out.append(f"{indent}end")
    out.append("")


def ml_const(v):
    """An OCaml constant: an [int], or an [int64] when it does not fit one."""
    return f"0x{v:x}L" if v >= 1 << 62 else ml_int(v)


def ml_tuple(v):
    return "(" + ", ".join(ml_int(x) for x in v) + ")"


def blocks(text):
    """The struct and union definitions of [text] at the start of a line, as
    (tag, the name after the closing brace, text)."""
    out = []
    for m in re.finditer(r"^(typedef\s+)?(struct|union)\s*(\w*)\s*\{", text, re.M):
        depth, i = 0, m.end() - 1
        while True:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            if depth == 0:
                break
            i += 1
        end = re.match(r"\s*(\w*)\s*;", text[i + 1:])
        if end:
            out.append((m.group(3), end.group(1), text[m.start():i + 1 + end.end()]))
    return out


def cut(tree, path, types, macros):
    """The C text of the typedefs [types] and the macros [macros] of [path]."""
    text = (tree / path).read_text()
    defs = blocks(text)
    out = []
    for n in types:
        named = [b for tag, after, b in defs if after == n]
        tagged = [b for tag, after, b in defs if tag == n and after == ""]
        typedef = re.findall(rf"^typedef\s+struct\s+{n}\s+{n}\s*;", text, re.M)
        if len(named) == 1:
            out.append(named[0])
        elif len(tagged) == 1 and len(typedef) == 1:
            out += [tagged[0], typedef[0]]
        else:
            sys.exit(f"{path}: no single definition of {n}")
    for n in macros:
        m = re.findall(rf"^#define\s+{n}\s+.*$", text, re.M)
        if len(m) != 1:
            sys.exit(f"{path}: {n} is defined {len(m)} times")
        out.append(re.sub(r"//.*$", "", m[0]))
    return "\n".join(out) + "\n"


def rom_layout(fields, fmt):
    """Offsets of [fields], in order, packed as the format string [fmt] says:
    counts of b (byte), w (word) and d (double word)."""
    sizes = []
    for n, k in re.findall(r"(\d+)([bwdq])", fmt):
        sizes += [{"b": 1, "w": 2, "d": 4, "q": 8}[k]] * int(n)
    if len(sizes) != len(fields):
        sys.exit(f"format {fmt} has {len(sizes)} items for {len(fields)} fields")
    out, off = {}, 0
    for f, s in zip(fields, sizes):
        out[f] = (off, s)
        off += s
    return off, out


def registers(tree, name, arch):
    """tinygrad's rule over a published register header: (registers as
    (offset, stride, fields), values)."""
    if name == "nv_ref":
        path = tree / SWREF / "nv_ref.h"
    elif name == "dev_mmu":
        path = tree / HWREF / ARCH_DIR[arch] / arch / f"{name}.h"
    else:
        path = tree / SWREF / ARCH_DIR[arch] / arch / f"{name}.h"
    text = path.read_text()
    if (name, arch) in ADDENDA:
        text += (path.parent / f"{name}_addendum.h").read_text()
    lines = text.splitlines()
    bitfields = {}
    for line in lines:
        m = re.match(r"#define\s+(\w+)\s+([0-9\+\-\*\(\)]+):([0-9\+\-\*\(\)]+)", line)
        if m:
            hi, lo = eval(m.group(2)), eval(m.group(3))  # noqa: S307 (arithmetic of the pinned header)
            bitfields[m.group(1)] = (lo, hi - lo + 1)

    def base(n):
        return next((REG_BASES.get(p, 0) for p in REG_PREFIXES if n.startswith(p)), None)

    def fields(n):
        return [(k[len(n) + 1:].lower(), v) for k, v in bitfields.items() if k.startswith(n + "_")]
    regs, values = {}, {}
    for line in lines:
        m = re.match(r"#define\s+(\w+)\s*\(\s*(\w+)\s*\)\s*(.+)", line)
        if m and base(m.group(1)) is not None:
            expr = re.sub(r"\s*/\*.*\*/", "", m.group(3))
            at = [eval(expr.replace(m.group(2), str(i))) for i in (0, 1)]  # noqa: S307
            regs[m.group(1)] = (base(m.group(1)) + at[0], at[1] - at[0], fields(m.group(1)))
            continue
        m = re.match(r"#define\s+(\w+)\s+(0x[0-9A-Fa-f]+|\d+)(?![^\n]*:)", line)
        if m:
            n, v = m.group(1), int(m.group(2), 0)
            if base(n) is None or any(n.startswith(r + "_") for r in regs):
                values[n] = v
            else:
                regs[n] = (base(n) + v, 0, fields(n))
    return regs, values, bitfields

# Generation


def generate(cache, pins, pin, outdir):
    import clang.cindex as ci
    trees, nvfw = sources(cache, pins, pin)
    stub = stub_dir("nv")
    gsp = trees[GSP]
    out = ["(* Generated by gen.py; do not edit. The inputs and the command that",
           "   regenerates this file are in gen.py; their digests are in pins.json. *)", ""]
    out.append(f"(* The driver releases whose layouts are below: {', '.join(version(t) for t in trees.values())}. *)")
    out.append("let releases = [ " + "; ".join(str(r) for r in trees) + " ]")
    out.append("")

    # The resource manager's interface, per release
    units, arms = {}, {}
    for rel, tree in trees.items():
        hs, incs = headers(tree, RM_HEADERS), [tree / i for i in RM_INCLUDES]
        units[rel] = Unit(ci, hs, incs, stub, extra=nvtypes(tree))
        arms[rel] = Unit(ci, hs, incs, stub, extra=nvtypes(tree), target="aarch64-unknown-linux-gnu")
    values = {}
    for rel, u in units.items():
        v = {**u.enums()}
        v.update(u.macros([c for c in RM_CONSTANTS if c not in v]))
        missing = [c for c in RM_CONSTANTS if c not in v]
        if missing:
            sys.exit(f"{rel}: undefined constants {missing}")
        values[rel] = v
    for c in RM_CONSTANTS:
        if len({values[r][c] for r in trees}) != 1:
            sys.exit(f"{c} differs between releases")
    out.append("(* Constants of the resource manager's interface, the same in every release. *)")
    for c in RM_CONSTANTS:
        out.append(f"let {snake(c)} = {ml_const(values[GSP][c])}")
    out.append("")
    rm_files = {rel: headers(t, RM_HEADERS) + [pathlib.Path(p) for p in glob.glob(
        str(t / "src/common/sdk/nvidia/inc/**/*.h"), recursive=True)] for rel, t in trees.items()}
    ranges = {rel: field_ranges(rm_files[rel], RM_FIELDS) for rel in trees}
    differ = {f for f in RM_FIELDS if len({ranges[r][f] for r in trees}) != 1}
    if differ != RM_FIELDS_PER_RELEASE:
        sys.exit(f"the bit fields that differ between releases are {sorted(differ)}, not {sorted(RM_FIELDS_PER_RELEASE)}")
    out.append("(* Bit fields of their words: (lowest bit, bits). *)")
    for f in RM_FIELDS:
        if f not in RM_FIELDS_PER_RELEASE:
            out.append(f"let {f.lower()} = {ml_tuple(ranges[GSP][f])}")
    out.append("")

    layouts = {}
    for cname in RM_STRUCTS:
        per = {}
        for rel, u in units.items():
            per[rel] = struct_fields(ci, u, cname)
            if struct_fields(ci, arms[rel], cname) != per[rel]:
                sys.exit(f"{cname} differs on aarch64 in {rel}")
        layouts[cname] = per
    differ = {c for c, per in layouts.items() if c not in RM_GSP_ONLY
              if any(per[r][0] != per[GSP][0] or any(per[r][1].get(f) != per[GSP][1].get(f) for f in RM_STRUCTS[c][1])
                     for r in trees)}
    if differ != RM_PER_RELEASE:
        sys.exit(f"the structs that differ between releases are {sorted(differ)}, not {sorted(RM_PER_RELEASE)}")
    out.append("(* Structs: each field is (byte offset, bytes), an array's (offset, bytes")
    out.append("   of an element, elements). *)")
    for cname, (module, wanted) in RM_STRUCTS.items():
        if cname not in RM_PER_RELEASE:
            size, fields = layouts[cname][GSP]
            emit_struct(out, module, size, fields, wanted)

    # the structs whose layouts differ, and the statuses, per release
    out.append("(* What differs between the releases. *)")
    out.append("module type RELEASE = sig")
    for cname in sorted(RM_PER_RELEASE):
        module, wanted = RM_STRUCTS[cname]
        out.append(f"  module {module} : sig")
        out.append("    val sizeof : int")
        for f in wanted:
            opt = "" if all(f in layouts[cname][r][1] for r in trees) else " option"
            arity = len(next(layouts[cname][r][1][f] for r in trees if f in layouts[cname][r][1]))
            out.append(f"    val {snake(f)} : ({' * '.join(['int'] * arity)}){opt}")
        out.append("  end")
        out.append("")
    for f in RM_FIELDS:
        if f in RM_FIELDS_PER_RELEASE:
            out.append(f"  val {f.lower()} : int * int")
    out.append("")
    out.append("  val statuses : (int * string) list")
    out.append("end")
    out.append("")
    for rel, tree in trees.items():
        out.append(f"module R{rel} : RELEASE = struct")
        for cname in sorted(RM_PER_RELEASE):
            module, wanted = RM_STRUCTS[cname]
            size, fields = layouts[cname][rel]
            out.append(f"  module {module} = struct")
            out.append(f"    let sizeof = {size}")
            for f in wanted:
                opt = not all(f in layouts[cname][r][1] for r in trees)
                if f in fields:
                    out.append(f"    let {snake(f)} = {'Some ' if opt else ''}{ml_tuple(fields[f])}")
                else:
                    out.append(f"    let {snake(f)} = None")
            out.append("  end")
            out.append("")
        for f in RM_FIELDS:
            if f in RM_FIELDS_PER_RELEASE:
                out.append(f"  let {f.lower()} = {ml_tuple(ranges[rel][f])}")
        out.append("")
        codes = re.findall(r'NV_STATUS_CODE\(\s*(\w+)\s*,\s*(0x[0-9A-Fa-f]+)\s*,\s*"([^"]*)"\s*\)',
                           (tree / "kernel-open/common/inc/nvstatuscodes.h").read_text())
        out.append("  let statuses = [")
        for name, code, _ in codes:
            out.append(f"    ({ml_int(int(code, 16))}, {json.dumps(name)});")
        out.append("  ]")
        out.append("end")
        out.append("")
    out.append("let release = function")
    for rel in trees:
        out.append(f"  | {rel} -> (module R{rel} : RELEASE)")
    out.append("  | r -> invalid_arg (Printf.sprintf \"Nv_defs.release %d\" r)")
    out.append("")

    # MMU faults
    fault = (gsp / HWREF / "ampere/ga100/dev_fault.h").read_text()
    for kind in ("FAULT_TYPE", "ACCESS_TYPE"):
        names = re.findall(rf"#define\s+NV_PFAULT_{kind}_(\w+)\s+(0x[0-9A-Fa-f]+|\d+)", fault)
        out.append(f"let fault_{kind.lower()}s = [")
        for n, v in names:
            out.append(f"  ({ml_int(int(v, 0))}, {json.dumps(n)});")
        out.append("]")
        out.append("")

    # The GSP's interface
    gu = Unit(ci, headers(gsp, GSP_HEADERS), [gsp / i for i in GSP_INCLUDES], stub, extra=nvtypes(gsp),
              defines=GSP_DEFINES)
    ga = Unit(ci, headers(gsp, GSP_HEADERS), [gsp / i for i in GSP_INCLUDES], stub, extra=nvtypes(gsp),
              defines=GSP_DEFINES, target="aarch64-unknown-linux-gnu")
    gv = {**gu.enums()}
    gv.update(gu.macros([c for c in GSP_CONSTANTS if c not in gv]))
    missing = [c for c in GSP_CONSTANTS if c not in gv]
    if missing:
        sys.exit(f"undefined GSP constants {missing}")
    out.append("(* The GSP's messages and boot structures. *)")
    for c in GSP_CONSTANTS:
        out.append(f"let {c.lower()} = {ml_const(gv[c])}")
    out.append("")
    for cname, (module, wanted) in GSP_STRUCTS.items():
        if cname in CUT_STRUCTS:
            continue
        size, fields = struct_fields(ci, gu, cname)
        if struct_fields(ci, ga, cname) != (size, fields):
            sys.exit(f"{cname} differs on aarch64")
        emit_struct(out, module, size, fields, wanted)

    # definitions cut out of their files
    text = nvtypes(gsp) + "".join(f'#include "{h}"\n' for h in headers(gsp, CUT_HEADERS))
    text += '#include "gpu/vbios/bios_types.h"\n#include "g_rpc-structures.h"\n#include "g_rpc-message-header.h"\n'
    text += "".join(cut(gsp, f, t, m) for f, t, m in CUTS)
    cuth = stub / "nv_cuts.h"
    cuth.write_text(text)
    incs = [gsp / i for i in GSP_INCLUDES]
    cu = Unit(ci, [cuth], incs, stub, defines=GSP_DEFINES)
    ca = Unit(ci, [cuth], incs, stub, defines=GSP_DEFINES, target="aarch64-unknown-linux-gnu")
    cv = {**cu.enums()}
    cv.update(cu.macros([c for c in CUT_CONSTANTS if c not in cv]))
    missing = [c for c in CUT_CONSTANTS if c not in cv]
    if missing:
        sys.exit(f"undefined constants {missing}")
    out.append("(* The VBIOS, its falcon ucode tables, and the registry. *)")
    for c in CUT_CONSTANTS:
        out.append(f"let {c.lower()} = {ml_const(cv[c])}")
    for f, v in field_ranges([cuth], CUT_FIELDS).items():
        out.append(f"let {f.lower()} = {ml_tuple(v)}")
    out.append("")
    for cname, (module, wanted) in CUT_STRUCTS.items():
        size, fields = struct_fields(ci, cu, cname)
        if struct_fields(ci, ca, cname) != (size, fields):
            sys.exit(f"{cname} differs on aarch64")
        emit_struct(out, module, size, fields, wanted)
    fmts = dict(re.findall(r'#define\s+(\w+)\s+"([^"]+)"', text))
    out.append("(* Structures of the ROM, packed as their format strings say. *)")
    for cname, module, fmt in ROM_STRUCTS:
        c = cu.struct(cname)
        names = [f.spelling for f in c.type.get_canonical().get_fields()]
        size, fields = rom_layout(names, fmts[fmt])
        emit_struct(out, module, size, fields, names)

    # the firmware containers
    nu = Unit(ci, [nvfw / f for f in NVFW_FILES], [], stub)
    for cname, (module, wanted) in NVFW_STRUCTS.items():
        size, fields = struct_fields(ci, nu, cname)
        emit_struct(out, module, size, fields, wanted)

    # registers
    out.append("(* Registers: (name, (offset, stride of an indexed register, fields as (name,")
    out.append("   (lowest bit, bits)))), by header and architecture. A falcon's are relative")
    out.append("   to the falcon. *)")
    out.append("let registers = [")
    found, vals = set(), {}
    for name, arch in REG_FILES:
        regs, values_, _ = registers(gsp, name, arch)
        keep = [(n, r) for n, r in regs.items() if n in REGISTERS]
        found |= {n for n, _ in keep}
        vals.update({n: v for n, v in values_.items() if n in REG_VALUES and n not in vals})
        out.append(f"  ( {json.dumps(name)}, {json.dumps(arch or '')}, [")
        for n, (off, stride, fs) in keep:
            fields = "; ".join(f"({json.dumps(k)}, {ml_tuple(v)})" for k, v in fs)
            out.append(f"      ({json.dumps(n)}, ({ml_int(off)}, {ml_int(stride)}, [ {fields} ]));")
        out.append("    ] );")
    out.append("]")
    out.append("")
    missing = [n for n in REGISTERS if n not in found] + [n for n in REG_VALUES if n not in vals]
    if missing:
        sys.exit(f"no registers {missing}")
    for n in REG_VALUES:
        out.append(f"let {n.lower()} = {ml_int(vals[n])}")
    out.append("")

    # page-table entries
    out.append("(* Page-table entries of each MMU version: fields as (lowest bit, bits); a")
    out.append("   dual PDE's are of its 128 bits. *)")
    out.append("let mmu = [")
    for ver, arch in MMU.items():
        _, _, bits = registers(gsp, "dev_mmu", arch)
        out.append(f"  ( {ver}, [")
        for kind in ("PTE", "PDE", "DUAL_PDE"):
            pre = f"NV_MMU_VER{ver}_{kind}_"
            fs = [(k[len(pre):].lower(), v) for k, v in bits.items() if k.startswith(pre)]
            out.append(f"      ( {json.dumps(kind.lower())}, [ " + "; ".join(
                f"({json.dumps(k)}, {ml_tuple(v)})" for k, v in fs) + " ] );")
        out.append("    ] );")
    out.append("]")
    out.append("")

    # firmware
    out.append("(* The linux-firmware commit of the firmware below. *)")
    out.append(f"let firmware_commit = {json.dumps(FIRMWARE_COMMIT)}")
    out.append("")
    out.append("(* The SHA-256 of each firmware file, under nvidia/. *)")
    out.append("let firmware_sha256 = [")
    for n in FIRMWARE:
        digest = hashlib.sha256(fetch(cache, FIRMWARE_RAW.format(name=n), pins, pin).read_bytes()).hexdigest()
        out.append(f"  ({json.dumps(n)}, {json.dumps(digest)});")
    out.append("]")
    (outdir / "nv_defs.ml").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main(__doc__, generate, ["nv_defs.ml"], HERE, OUT, "nv-gen")
