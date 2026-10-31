# /// script
# requires-python = ">=3.10"
# ///
"""Generates the tables of NVIDIA's GPUs from NVIDIA's sources:

- lib/nv/defs.ml: the resource manager's classes and the parameter
  layouts the driver allocates channels and reads faults with, and the
  names of the errors and faults it reports, per release of NVIDIA's
  driver where they differ;
- lib/nv/nvidia/defs.ml: the escapes of NVIDIA's kernel driver and the
  layouts of their parameters, the resource manager's classes and controls
  a path opens a GPU and sends the driver's commands with, the unified
  memory driver's ioctls, and each release's status names, per release
  where they differ;
- lib/nv/pci/defs.ml: what booting an NVIDIA GPU over PCI reads. The
  structures the GSP's boot takes, the VBIOS's structures, the firmware
  containers, the registers per family, and the firmware images, pinned.
  They come from release 570.144, the GSP firmware's release, and from
  Linux's nouveau driver.
- lib/nv/pci/gsp_defs.ml: the GSP's formats, from the same excerpts. Its
  messages and the bodies of the RM's calls, the CPU sequences, the WPR
  metadata, the registry, the names of the channel errors, and the
  page-table entries.
- lib/nv/abi/defs.ml: the methods of the host, compute and copy classes,
  the fields of their words and the values they take, the layout of a
  channel's ring entries and method headers, and the layouts of launch
  descriptors, versions 3 and 5, from release 570.144 and open-gpu-doc.

Run from the worktree root:

  uv run packages/rig/lib/nv/gen/gen.py
  uv run packages/rig/lib/nv/gen/gen.py --check
  uv run packages/rig/lib/nv/gen/gen.py --excerpt [--check]

The inputs are excerpts, in headers/: headers/ORIGIN/NAME is cut from a
file of ORIGIN, the release of open-gpu-kernel-modules (570, 580, 610,
615), open-gpu-doc or Linux, one excerpt of each file. Each is the file's
licence notice, the URL it is cut from, and the definitions the libraries
read, verbatim and in the file's order, with the definitions they depend
on, as a 64-bit Linux build compiles them. A few small files are kept
whole, under the repository's MIT notice where they carry none. The
newest release's excerpts hold everything rig_nv and rig_nv_nvidia read;
an older release's hold only what differs between releases.
headers/firmware.tsv lists the firmware images of linux-firmware at
FIRMWARE_COMMIT: each image's path, BLAKE2b-256 digest and URL; the page
of rig firmware, in packages/rig/bin/help.ml, names FIRMWARE_COMMIT too.

--excerpt makes the excerpts and firmware.tsv from the upstream files,
each pinned in pins.json by URL and SHA-256 and checked against its pin,
and fails on a file whose notice is not the MIT licence. It checks that
what defs.ml takes once is the same in every release, that the structs
and tables that differ are exactly those it takes per release, and that
the excerpts generate what the upstream files do. Downloads are kept in
--cache, and --pin records the digests of files not yet pinned.
Generating reads the excerpts and firmware.tsv alone, offline, and
requires firmware.tsv to list exactly the images of FIRMWARE. --check
generates into memory and fails if a committed file differs.

Text is read and written as latin-1, one character per byte, so every byte
of a header round-trips into its excerpt as upstream wrote it.

Struct layouts follow the C rules of the 64-bit Linux ABIs, x86_64 and
aarch64 alike: a field at the next multiple of its alignment, a struct's
size a multiple of its largest alignment, NV_DECLARE_ALIGNED and
NV_ALIGN_BYTES raising a field's alignment, and #pragma pack(1) removing
the padding. The VBIOS's structures are packed as their format strings
say. A register's address is its define, plus its unit's base for the
falcon's units, whose headers give offsets within the unit. Every value is
NVIDIA's. Where two classes rig_nv_abi supports both define a name (the
copy classes 0xc7b5 and 0xc9b5, the launch descriptors of 0xc7c0 and
0xc9c0), the script fails unless they agree. Every table is a literal,
which the compiler lays out as static data.
"""

import argparse
import dataclasses
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
# The command that regenerates the files, run from the repository root.
COMMAND = "uv run packages/rig/lib/nv/gen/gen.py"

# Origins

REPO = "https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/"
# The driver releases, by the branch the RM reports: the tree each is read
# from, and its version.
RELEASES = {
    570: ("570.144", "570.144"),
    580: ("2af9f1f0f7de4988432d4ae875b5858ffdb09cc2", "580.105.08"),
    610: ("610.43.03", "610.43.03"),
    615: ("615.71.09", "615.71.09"),
}
NEWEST = max(RELEASES)
DOC = "https://raw.githubusercontent.com/NVIDIA/open-gpu-doc/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/"
LINUX = "https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/{}?h=v6.18"
FIRMWARE_COMMIT = "0a6871b19abf5d6e024b5d208b101ae53e7fa0de"
FIRMWARE_ORIGIN = f"https://gitlab.com/kernel-firmware/linux-firmware/-/raw/{FIRMWARE_COMMIT}/"
FIRMWARE_LIST = "firmware.tsv"


def tree(r):
    """The URL of release [r]'s tree."""
    return REPO + RELEASES[r][0] + "/"


def origin(url):
    """The directory of headers/ that holds the excerpts of [url]'s tree."""
    for r in RELEASES:
        if url.startswith(tree(r)):
            return str(r)
    if url.startswith(DOC):
        return "open-gpu-doc"
    if url.startswith(LINUX.split("{")[0]):
        return "linux"
    sys.exit(f"{url}: no origin")


SDK = "src/common/sdk/nvidia/inc/"
UNIX = "src/nvidia/arch/nvalloc/unix/include/"
UVM = "kernel-open/nvidia-uvm/"
COMMON = "kernel-open/common/inc/"

# The MIT licence, under which NVIDIA's headers and nouveau's are copied.
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


# rig_nv and rig_nv_nvidia

@dataclasses.dataclass
class Defs:
    """A library's defs.ml and what it reads from its excerpts."""

    out: pathlib.Path
    # Each excerpt and the header it is cut from, in each release's tree. A
    # name is read from the first header that defines it.
    sources: dict
    # Constants, the same in every release.
    constants: list
    # Bit fields of 32-bit words, "hi:lo" in the headers: (lowest bit, bits).
    fields: list
    # Structs, by C name: the module they become and the fields read ("a__b"
    # for a field b of a struct a; "a?" for a field a release may lack).
    structs: dict
    # The structs whose layouts differ between the releases.
    per_release: set
    # Name tables: (OCaml name, header, the defines read, as a pattern whose
    # group 1 is the name an entry carries), and those that differ between
    # the releases.
    tables: list = ()
    tables_per_release: frozenset = frozenset()
    # The statuses read by name from STATUS_HEADER, the same in every
    # release. A library that reads them also gets each release's table of
    # every status.
    statuses: list = ()
    # The headers some releases lack, by the first release that has each.
    since: dict = dataclasses.field(default_factory=dict)

    def headers(self, r):
        """The headers of release [r], by name, with their URLs."""
        return {h: tree(r) + p for h, p in self.sources.items() if r >= self.since.get(h, 0)}


NV = Defs(
    out=LIB / "defs.ml",
    sources={
        "g_allclasses.h": "src/nvidia/generated/g_allclasses.h",
        "nvlimits.h": SDK + "nvlimits.h",
        "nvos.h": SDK + "nvos.h",
        "alloc_channel.h": SDK + "alloc/alloc_channel.h",
        "nvgputypes.h": SDK + "nvgputypes.h",
        "nverror.h": SDK + "nverror.h",
        "cl83de.h": SDK + "class/cl83de.h",
        "clc56f.h": SDK + "class/clc56f.h",
        "cl2080_notification.h": SDK + "class/cl2080_notification.h",
        "ctrla06c.h": SDK + "ctrl/ctrla06c.h",
        "ctrla06fgpfifo.h": SDK + "ctrl/ctrla06f/ctrla06fgpfifo.h",
        "ctrl83dedebug.h": SDK + "ctrl/ctrl83de/ctrl83dedebug.h",
        "dev_fault.h": UVM + "hwref/ampere/ga100/dev_fault.h",
    },
    constants=[
        # classes
        "KEPLER_CHANNEL_GROUP_A", "FERMI_CONTEXT_SHARE_A", "GT200_DEBUGGER",
        # allocations
        "NV2080_ENGINE_TYPE_GRAPHICS", "NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC",
        # a channel's error notifier
        "NV_CHANNELGPFIFO_NOTIFICATION_TYPE_ERROR", "NV_CHANNELGPFIFO_NOTIFICATION_TYPE__SIZE_1",
    ],
    fields=[
        "NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT",
    ],
    structs={
        "NV_CTXSHARE_ALLOCATION_PARAMETERS": ("Ctxshare_alloc", ["hVASpace", "flags"]),
        "NV83DE_ALLOC_PARAMETERS": ("Nv83de_alloc", ["hAppClient", "hClass3dObject"]),
        "NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS": ("Sm_error_states", [
            "hTargetChannel", "numSMsToRead", "smErrorStateArray", "mmuFault__valid"]),
        "NV83DE_SM_ERROR_STATE_REGISTERS": ("Sm_error_state", ["hwwGlobalEsr", "hwwWarpEsr", "hwwWarpEsrPc64"]),
        "NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_PARAMS": ("Mmu_fault_info", ["mmuFaultInfoList", "count"]),
        "NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_ENTRY": ("Mmu_fault_entry", [
            "faultAddress", "faultType", "accessType"]),
        "Nvc56fControl": ("Userd", ["GPGet", "GPPut"]),
        "NvNotification": ("Notification", ["info32", "info16", "status"]),
        "NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS": ("Channel_group_alloc", ["engineType"]),
        "NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS": ("Gpfifo_alloc", [
            "hObjectError", "hObjectBuffer", "gpFifoOffset", "gpFifoEntries", "hContextShare", "hUserdMemory",
            "userdOffset"]),
        "NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS": ("Group_schedule", ["bEnable"]),
    },
    per_release={"NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS", "NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS",
                 "NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS"},
    # The errors leave out the recovery levels and the count, which are no
    # error. Later releases add errors, and give some numbers new names.
    tables=[
        ("robust_channel_errors", "nverror.h", r"ROBUST_CHANNEL_(?!ERROR_RECOVERY_LEVEL_|LAST_ERROR)(\w+)"),
        ("fault_types", "dev_fault.h", r"NV_PFAULT_FAULT_TYPE_(\w+)"),
        ("access_types", "dev_fault.h", r"NV_PFAULT_ACCESS_TYPE_(\w+)"),
    ],
    tables_per_release={"robust_channel_errors"},
)

NVIDIA = Defs(
    out=LIB / "nvidia/defs.ml",
    sources={
        "g_allclasses.h": "src/nvidia/generated/g_allclasses.h",
        "nvmisc.h": COMMON + "nvmisc.h",
        "nvlimits.h": SDK + "nvlimits.h",
        "nvos.h": SDK + "nvos.h",
        "cl0080.h": SDK + "class/cl0080.h",
        "cl2080.h": SDK + "class/cl2080.h",
        "ctrlxxxx.h": SDK + "ctrl/ctrlxxxx.h",
        "ctrl0000system.h": SDK + "ctrl/ctrl0000/ctrl0000system.h",
        "ctrl0000gpu.h": SDK + "ctrl/ctrl0000/ctrl0000gpu.h",
        "ctrl0080gpu.h": SDK + "ctrl/ctrl0080/ctrl0080gpu.h",
        "ctrl2080gpu.h": SDK + "ctrl/ctrl2080/ctrl2080gpu.h",
        "ctrl0080gr.h": SDK + "ctrl/ctrl0080/ctrl0080gr.h",
        "ctrl2080gr.h": SDK + "ctrl/ctrl2080/ctrl2080gr.h",
        "ctrl2080fb.h": SDK + "ctrl/ctrl2080/ctrl2080fb.h",
        "ctrl2080rc.h": SDK + "ctrl/ctrl2080/ctrl2080rc.h",
        "ctrla06c.h": SDK + "ctrl/ctrla06c.h",
        "ctrlc36f.h": SDK + "ctrl/ctrlc36f.h",
        "ctrl2080perf.h": SDK + "ctrl/ctrl2080/ctrl2080perf.h",
        "ctrl83dedebug.h": SDK + "ctrl/ctrl83de/ctrl83dedebug.h",
        "nv-ioctl.h": COMMON + "nv-ioctl.h",
        "nv-ioctl-numbers.h": COMMON + "nv-ioctl-numbers.h",
        "nv_escape.h": UNIX + "nv_escape.h",
        "nv-unix-nvos-params-wrappers.h": UNIX + "nv-unix-nvos-params-wrappers.h",
        "nvCpuUuid.h": COMMON + "nvCpuUuid.h",
        "uvm_types.h": UVM + "uvm_types.h",
        "nv_uvm_user_types.h": COMMON + "nv_uvm_user_types.h",
        "uvm_ioctl.h": UVM + "uvm_ioctl.h",
        "uvm_linux_ioctl.h": UVM + "uvm_linux_ioctl.h",
        "nvstatuscodes.h": COMMON + "nvstatuscodes.h",
        "tu102_dev_mmu.h": UVM + "hwref/turing/tu102/dev_mmu.h",
    },
    constants=[
        # classes
        "NV01_ROOT_CLIENT", "NV01_DEVICE_0", "NV20_SUBDEVICE_0",
        "NV01_MEMORY_SYSTEM_OS_DESCRIPTOR", "NV1_MEMORY_SYSTEM", "NV1_MEMORY_USER", "FERMI_VASPACE_A",
        "TURING_USERMODE_A", "HOPPER_USERMODE_A", "AMPERE_CHANNEL_GPFIFO_A", "BLACKWELL_CHANNEL_GPFIFO_A",
        "AMPERE_COMPUTE_B", "ADA_COMPUTE_A", "BLACKWELL_COMPUTE_B", "AMPERE_DMA_COPY_B", "BLACKWELL_DMA_COPY_B",
        # escapes
        "NV_IOCTL_MAGIC", "NV_ESC_CARD_INFO", "NV_ESC_REGISTER_FD", "NV_ESC_RM_ALLOC", "NV_ESC_RM_ALLOC_MEMORY",
        "NV_ESC_RM_CONTROL", "NV_ESC_RM_FREE", "NV_ESC_RM_MAP_MEMORY",
        "NV_ESC_EXPORT_TO_DMABUF_FD", "NV_DMABUF_EXPORT_MAPPING_TYPE_DEFAULT",
        # memory
        "NV_MAX_DEVICES", "NV_MMU_PTE_KIND_GENERIC_MEMORY", "NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS",
        "NVOS02_FLAGS_COHERENCY_CACHED", "NVOS02_FLAGS_MAPPING_NO_MAP",
        "NVOS32_ATTR_PHYSICALITY_CONTIGUOUS", "NVOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS",
        "NVOS32_ATTR_PAGE_SIZE_HUGE", "NVOS32_ATTR_LOCATION_VIDMEM", "NVOS32_ATTR_LOCATION_PCI",
        "NVOS32_ATTR2_GPU_CACHEABLE_YES", "NVOS32_ATTR2_GPU_CACHEABLE_NO", "NVOS32_ATTR2_PAGE_SIZE_HUGE_2MB",
        "NVOS32_ATTR2_ZBC_PREFER_NO_ZBC", "NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED",
        "NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED", "NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE",
        "NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT", "NVOS32_ALLOC_FLAGS_PERSISTENT_VIDMEM", "NVOS32_TYPE_IMAGE",
        "NVOS32_TYPE_NOTIFIER", "NVOS33_FLAGS_CACHING_TYPE_CACHED", "NVOS33_FLAGS_CACHING_TYPE_UNCACHED",
        "NVOS33_FLAGS_CACHING_TYPE_WRITECOMBINED",
        # controls
        "NV0000_CTRL_CMD_SYSTEM_GET_BUILD_VERSION_V2", "NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2",
        "NV0080_CTRL_CMD_GPU_GET_CLASSLIST", "NV2080_CTRL_CMD_GPU_GET_GID_INFO",
        "NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY", "NV2080_CTRL_CMD_GR_GET_INFO",
        "NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_GPCS", "NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_TPC_PER_GPC",
        "NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_SM_PER_TPC", "NV2080_CTRL_GR_INFO_INDEX_MAX_WARPS_PER_SM",
        "NV2080_CTRL_GR_INFO_INDEX_SM_VERSION", "NV2080_CTRL_CMD_FB_GET_INFO_V2",
        "NV2080_CTRL_FB_INFO_INDEX_HEAP_SIZE", "NV2080_CTRL_FB_INFO_INDEX_BAR1_SIZE",
        "NV2080_CTRL_CMD_RC_GET_WATCHDOG_INFO", "NV2080_CTRL_RC_GET_WATCHDOG_INFO_FLAGS_RUNNING",
        "NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN", "NVA06C_CTRL_CMD_GPFIFO_SCHEDULE",
        "NV2080_CTRL_CMD_PERF_BOOST", "NV83DE_CTRL_CMD_DEBUG_READ_ALL_SM_ERROR_STATES",
        "NV83DE_CTRL_CMD_DEBUG_READ_MMU_FAULT_INFO", "NV2080_CTRL_PERF_BOOST_FLAGS_CMD_BOOST_TO_MAX",
        "NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_YES", "NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_PRIORITY_HIGH",
        "NV2080_CTRL_PERF_BOOST_DURATION_INFINITE",
        # allocations
        "NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES", "NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING",
        "NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED",
        # the unified memory driver
        "UVM_INITIALIZE", "UVM_MM_INITIALIZE", "UVM_REGISTER_GPU", "UVM_UNREGISTER_GPU", "UVM_REGISTER_GPU_VASPACE",
        "UVM_UNREGISTER_GPU_VASPACE", "UVM_ENABLE_PEER_ACCESS", "UVM_REGISTER_CHANNEL", "UVM_UNREGISTER_CHANNEL",
        "UVM_CREATE_EXTERNAL_RANGE", "UVM_MAP_EXTERNAL_ALLOCATION", "UVM_UNMAP_EXTERNAL", "UVM_FREE",
        "UvmGpuMappingTypeReadWriteAtomic",
    ],
    fields=[
        "NVOS02_FLAGS_PHYSICALITY", "NVOS02_FLAGS_COHERENCY", "NVOS02_FLAGS_MAPPING", "NVOS32_ATTR_PHYSICALITY",
        "NVOS32_ATTR_PAGE_SIZE", "NVOS32_ATTR_LOCATION", "NVOS32_ATTR2_GPU_CACHEABLE", "NVOS32_ATTR2_PAGE_SIZE_HUGE",
        "NVOS32_ATTR2_ZBC", "NVOS33_FLAGS_CACHING_TYPE", "NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT",
        "NV2080_CTRL_PERF_BOOST_FLAGS_CMD", "NV2080_CTRL_PERF_BOOST_FLAGS_CUDA",
        "NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_PRIORITY",
    ],
    structs={
        "nv_ioctl_card_info_t": ("Card_info", ["valid", "pci_info__domain", "pci_info__bus", "pci_info__slot",
                                               "gpu_id", "minor_number"]),
        "nv_ioctl_register_fd_t": ("Register_fd", ["ctl_fd"]),
        "nv_ioctl_export_to_dma_buf_fd_t": ("Export_dma_buf", [
            "fd", "hClient", "totalObjects", "numObjects", "index", "totalSize", "mappingType", "handles",
            "offsets", "sizes", "status"]),
        "NVOS00_PARAMETERS": ("Nvos00", ["hRoot", "hObjectParent", "hObjectOld", "status"]),
        "NVOS02_PARAMETERS": ("Nvos02", ["hRoot", "hObjectParent", "hObjectNew", "hClass", "flags", "pMemory",
                                         "limit", "status"]),
        "NVOS21_PARAMETERS": ("Nvos21", ["hRoot", "hObjectParent", "hObjectNew", "hClass", "pAllocParms",
                                         "paramsSize", "status"]),
        "NVOS33_PARAMETERS": ("Nvos33", ["hClient", "hDevice", "hMemory", "offset", "length", "pLinearAddress",
                                         "status", "flags"]),
        "NVOS54_PARAMETERS": ("Nvos54", ["hClient", "hObject", "cmd", "flags", "params", "paramsSize", "status"]),
        "nv_ioctl_nvos02_parameters_with_fd": ("Nvos02_with_fd", ["params", "fd"]),
        "nv_ioctl_nvos33_parameters_with_fd": ("Nvos33_with_fd", ["params", "fd"]),
        "NV0080_ALLOC_PARAMETERS": ("Nv0080_alloc", ["deviceId", "hClientShare", "vaMode"]),
        "NV2080_ALLOC_PARAMETERS": ("Nv2080_alloc", ["subDeviceId"]),
        "NV_MEMORY_ALLOCATION_PARAMS": ("Memory_alloc", ["owner", "type", "flags", "attr", "attr2", "format", "size",
                                                         "alignment", "offset", "limit"]),
        "NV0000_CTRL_SYSTEM_GET_BUILD_VERSION_V2_PARAMS": ("Build_version", ["driverVersionBuffer"]),
        "NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS": ("Id_info", ["gpuId", "deviceInstance"]),
        "NV0080_CTRL_GPU_GET_CLASSLIST_PARAMS": ("Classlist", ["numClasses", "classList"]),
        "NV2080_CTRL_GPU_GET_GID_INFO_PARAMS": ("Gid_info", ["flags", "length", "data"]),
        "NV2080_CTRL_GR_INFO": ("Gr_info", ["index", "data"]),
        "NV2080_CTRL_GR_GET_INFO_PARAMS": ("Gr_get_info", ["grInfoListSize", "grInfoList"]),
        "NV2080_CTRL_FB_INFO": ("Fb_info", ["index", "data"]),
        "UVM_INITIALIZE_PARAMS": ("Uvm_initialize", ["flags", "rmStatus"]),
        "UVM_MM_INITIALIZE_PARAMS": ("Uvm_mm_initialize", ["uvmFd", "rmStatus"]),
        "UVM_REGISTER_GPU_PARAMS": ("Uvm_register_gpu", [
            "gpu_uuid", "rmCtrlFd", "hClient", "hSmcPartRef", "rmStatus"]),
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
        "NV2080_CTRL_FB_GET_INFO_V2_PARAMS": ("Fb_get_info", ["fbInfoListSize", "fbInfoList"]),
        "NV2080_CTRL_RC_GET_WATCHDOG_INFO_PARAMS": ("Watchdog_info", ["watchdogStatusFlags"]),
        "NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS": ("Work_submit_token", ["workSubmitToken"]),
        "NV2080_CTRL_PERF_BOOST_PARAMS": ("Perf_boost", ["flags", "duration"]),
        "NV_VASPACE_ALLOCATION_PARAMETERS": ("Vaspace_alloc", ["index", "flags", "vaSize", "vaBase"]),
        "UVM_FREE_PARAMS": ("Uvm_free", ["base", "length?", "rmStatus"]),
        "UVM_UNREGISTER_CHANNEL_PARAMS": ("Uvm_unregister_channel", [
            "gpuUuid?", "hClient", "hChannel", "rmStatus"]),
    },
    per_release={"NV2080_CTRL_FB_GET_INFO_V2_PARAMS", "NV_VASPACE_ALLOCATION_PARAMETERS",
                 "UVM_FREE_PARAMS", "UVM_UNREGISTER_CHANNEL_PARAMS"},
    statuses=["NV_OK", "NV_ERR_NO_MEMORY", "NV_ERR_INSUFFICIENT_RESOURCES", "NV_ERR_INVALID_ADDRESS",
              "NV_ERR_INVALID_ARGUMENT", "NV_ERR_NOT_SUPPORTED"],
    # The unified memory driver's mapping types moved to nv_uvm_user_types.h
    # from uvm_types.h.
    since={"nv_uvm_user_types.h": 580},
)

LIBRARIES = [NV, NVIDIA]

# A status: its name, value and description, in nvstatuscodes.h.
STATUS = re.compile(r'^[ \t]*NV_STATUS_CODE\(\s*(\w+)\s*,\s*(0[xX][0-9A-Fa-f]+)\s*,\s*"[^"]*"\s*\)[^\n]*$', re.M)
STATUS_HEADER = "nvstatuscodes.h"


# rig_nv_pci

PCI_OUT = LIB / "pci/defs.ml"
GSP_OUT = LIB / "pci/gsp_defs.ml"

# Release 570.144, the GSP firmware's, which rig_nv_abi reads too.
R570 = tree(570)
GSP_INC = "src/nvidia/arch/nvalloc/common/inc/"
SWREF = "src/common/inc/swref/published/"
HWREF = "kernel-open/nvidia-uvm/hwref/"

# Each excerpt and the file it is cut from. A name is read from the first
# file that defines it.
PCI_SOURCES = {
    "msgq_priv.h": R570 + "src/common/shared/msgq/inc/msgq/msgq_priv.h",
    "message_queue_priv.h": R570 + "src/nvidia/inc/kernel/gpu/gsp/message_queue_priv.h",
    "g_rpc-message-header.h": R570 + "src/nvidia/generated/g_rpc-message-header.h",
    "g_rpc-structures.h": R570 + "src/nvidia/generated/g_rpc-structures.h",
    "g_sdk-structures.h": R570 + "src/nvidia/generated/g_sdk-structures.h",
    "rpc_headers.h": R570 + "src/nvidia/inc/kernel/vgpu/rpc_headers.h",
    "gsp_fw_wpr_meta.h": R570 + GSP_INC + "gsp/gsp_fw_wpr_meta.h",
    "rmRiscvUcode.h": R570 + GSP_INC + "rmRiscvUcode.h",
    "nverror.h": R570 + "src/common/sdk/nvidia/inc/nverror.h",
    "pci_exp_table.h": R570 + "src/nvidia/inc/kernel/platform/pci_exp_table.h",
    "kernel_gsp_fwsec.c": R570 + "src/nvidia/src/kernel/gpu/gsp/kernel_gsp_fwsec.c",
    "kernel_gsp_frts_tu102.c": R570 + "src/nvidia/src/kernel/gpu/gsp/arch/turing/kernel_gsp_frts_tu102.c",
    "kernel_gsp_vbios_tu102.c": R570 + "src/nvidia/src/kernel/gpu/gsp/arch/turing/kernel_gsp_vbios_tu102.c",
    "tu102_dev_mmu.h": R570 + HWREF + "turing/tu102/dev_mmu.h",
    "gh100_dev_mmu.h": R570 + HWREF + "hopper/gh100/dev_mmu.h",
    "gsp_init_args.h": R570 + "src/nvidia/inc/kernel/gpu/gsp/gsp_init_args.h",
    "gsp_static_config.h": R570 + "src/nvidia/inc/kernel/gpu/gsp/gsp_static_config.h",
    "gspifpub.h": R570 + GSP_INC + "gsp/gspifpub.h",
    "libos_init_args.h": R570 + "src/common/uproc/os/common/include/libos_init_args.h",
    "rmgspseq.h": R570 + GSP_INC + "rmgspseq.h",
    "fsp_nvdm_format.h": R570 + GSP_INC + "fsp/fsp_nvdm_format.h",
    "fsp_mctp_format.h": R570 + GSP_INC + "fsp/fsp_mctp_format.h",
    "kern_fsp_cot_payload.h": R570 + "src/nvidia/inc/kernel/gpu/fsp/kern_fsp_cot_payload.h",
    "g_os_nvoc.h": R570 + "src/nvidia/generated/g_os_nvoc.h",
    "g_mem_desc_nvoc.h": R570 + "src/nvidia/generated/g_mem_desc_nvoc.h",
    "nv_memory_type.h": R570 + "src/nvidia/inc/kernel/os/nv_memory_type.h",
    "g_allclasses.h": R570 + "src/nvidia/generated/g_allclasses.h",
    "nvlimits.h": R570 + SDK + "nvlimits.h",
    "nvmisc.h": R570 + "kernel-open/common/inc/nvmisc.h",
    "nvos.h": R570 + SDK + "nvos.h",
    "alloc_channel.h": R570 + SDK + "alloc/alloc_channel.h",
    "cl0000.h": R570 + SDK + "class/cl0000.h",
    "cl0080.h": R570 + SDK + "class/cl0080.h",
    "cl2080.h": R570 + SDK + "class/cl2080.h",
    "cl2080_notification.h": R570 + SDK + "class/cl2080_notification.h",
    "ctrl0080fifo.h": R570 + SDK + "ctrl/ctrl0080/ctrl0080fifo.h",
    "ctrl2080fifo.h": R570 + SDK + "ctrl/ctrl2080/ctrl2080fifo.h",
    "ctrl2080gpu.h": R570 + SDK + "ctrl/ctrl2080/ctrl2080gpu.h",
    "ctrl0080gr.h": R570 + SDK + "ctrl/ctrl0080/ctrl0080gr.h",
    "ctrl2080gr.h": R570 + SDK + "ctrl/ctrl2080/ctrl2080gr.h",
    "ctrl2080internal.h": R570 + SDK + "ctrl/ctrl2080/ctrl2080internal.h",
    "ctrl2080nvd.h": R570 + SDK + "ctrl/ctrl2080/ctrl2080nvd.h",
    "ctrl2080perf.h": R570 + SDK + "ctrl/ctrl2080/ctrl2080perf.h",
    "ctrl90f1.h": R570 + SDK + "ctrl/ctrl90f1.h",
    "ctrla06c.h": R570 + SDK + "ctrl/ctrla06c.h",
    "ctrl83dedebug.h": R570 + SDK + "ctrl/ctrl83de/ctrl83dedebug.h",
    "g_chipset_nvoc.h": R570 + "src/nvidia/generated/g_chipset_nvoc.h",
    "gpu_acpi_data.h": R570 + "src/nvidia/inc/kernel/gpu/gpu_acpi_data.h",
    "ctrl0073system.h": R570 + "src/common/sdk/nvidia/inc/ctrl/ctrl0073/ctrl0073system.h",
    "ga100_dev_ctrl.h": R570 + SWREF + "ampere/ga100/dev_ctrl.h",
    "gb202_dev_vm.h": R570 + SWREF + "blackwell/gb202/dev_vm.h",
    "fw.h": LINUX.format("drivers/gpu/drm/nouveau/include/nvfw/fw.h"),
    "hs.h": LINUX.format("drivers/gpu/drm/nouveau/include/nvfw/hs.h"),
}

# Files kept whole: the function and event numbers are X-macro lines, which
# no definition holds.
WHOLE = {
    "rpc_global_enums.h": R570 + "src/nvidia/inc/kernel/vgpu/rpc_global_enums.h",
}

# The register headers, by excerpt: (directory, header). An addendum adds to
# the header before it.
REGISTER_SOURCES = {
    "nv_ref.h": "nv_ref.h",
    "tu102_dev_fb.h": "turing/tu102/dev_fb.h",
    "ga100_dev_fb.h": "ampere/ga100/dev_fb.h",
    "gb10b_dev_fbhub.h": "blackwell/gb10b/dev_fbhub.h",
    "tu102_dev_vm.h": "turing/tu102/dev_vm.h",
    "tu102_dev_bus.h": "turing/tu102/dev_bus.h",
    "ga102_dev_gc6_island.h": "ampere/ga102/dev_gc6_island.h",
    "ga102_dev_gc6_island_addendum.h": "ampere/ga102/dev_gc6_island_addendum.h",
    "ga102_dev_gsp.h": "ampere/ga102/dev_gsp.h",
    "ga102_dev_falcon_v4.h": "ampere/ga102/dev_falcon_v4.h",
    "ga102_dev_falcon_v4_addendum.h": "ampere/ga102/dev_falcon_v4_addendum.h",
    "ga102_dev_riscv_pri.h": "ampere/ga102/dev_riscv_pri.h",
    "ga102_dev_fbif_v4.h": "ampere/ga102/dev_fbif_v4.h",
    "ga102_dev_falcon_second_pri.h": "ampere/ga102/dev_falcon_second_pri.h",
    "ga102_dev_sec_pri.h": "ampere/ga102/dev_sec_pri.h",
    "gh100_dev_falcon_v4.h": "hopper/gh100/dev_falcon_v4.h",
    "gh100_dev_vm.h": "hopper/gh100/dev_vm.h",
    "gh100_dev_fsp_pri.h": "hopper/gh100/dev_fsp_pri.h",
    "gb202_dev_therm.h": "blackwell/gb202/dev_therm.h",
    "gb202_dev_therm_addendum.h": "blackwell/gb202/dev_therm_addendum.h",
    "tu102_dev_ext_devices.h": "turing/tu102/dev_ext_devices.h",
}
PCI_SOURCES.update({k: R570 + SWREF + v for k, v in REGISTER_SOURCES.items()})

# The register headers of each family, in the order the first that defines
# a name gives it.
FAMILIES = {
    "Legacy": ["nv_ref.h", "tu102_dev_fb.h", "ga100_dev_fb.h", "ga102_dev_gc6_island.h", "ga102_dev_gc6_island_addendum.h",
               "tu102_dev_vm.h", "ga102_dev_gsp.h", "ga102_dev_falcon_v4.h", "ga102_dev_falcon_v4_addendum.h",
               "ga102_dev_riscv_pri.h", "ga102_dev_fbif_v4.h", "ga102_dev_falcon_second_pri.h",
               "ga102_dev_sec_pri.h", "tu102_dev_bus.h", "tu102_dev_ext_devices.h"],
    "Blackwell": ["nv_ref.h", "tu102_dev_fb.h", "ga102_dev_gc6_island.h", "ga102_dev_gc6_island_addendum.h",
                  "gh100_dev_vm.h", "tu102_dev_vm.h", "ga102_dev_gsp.h", "gh100_dev_falcon_v4.h",
                  "gh100_dev_fsp_pri.h", "tu102_dev_bus.h", "gb202_dev_therm.h",
                  "gb202_dev_therm_addendum.h", "ga102_dev_riscv_pri.h", "gb10b_dev_fbhub.h"],
}

# The units whose headers give offsets within the unit: a falcon's units, at
# their offset within the falcon, and the virtual function's registers, at
# their offset in BAR 0 (NV_VIRTUAL_FUNCTION_FULL_PHYS_OFFSET, dev_vm.h).
UNIT_BASES = {"NV_PRISCV_RISCV": 0x1000, "NV_PFALCON_FBIF": 0x600, "NV_PFALCON2_FALCON": 0x1000,
              "NV_VIRTUAL_FUNCTION": 0xb80000}

# The registers read and written, with every bit field of each, per family
# where a family has them.
REGISTERS = {
    "Legacy": [
        "NV_PMC_BOOT_0", "NV_PMC_BOOT_42", "NV_PFB_PRI_MMU_WPR2_ADDR_LO", "NV_PFB_PRI_MMU_WPR2_ADDR_HI",
        "NV_PGC6_AON_SECURE_SCRATCH_GROUP_42", "NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK",
        "NV_PGC6_AON_SECURE_SCRATCH_GROUP_05", "NV_PGC6_BSI_SECURE_SCRATCH_14",
        "NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE", "NV_PBUS_BAR1_BLOCK", "NV_PGSP_QUEUE_HEAD",
        "NV_PGSP_FALCON_MAILBOX0", "NV_PGSP_FALCON_MAILBOX1", "NV_PGSP_FALCON_ENGINE", "NV_PSEC_FALCON_ENGINE",
        "NV_PFALCON_FALCON_OS", "NV_PFALCON_FALCON_DMATRFCMD", "NV_PFALCON_FALCON_DMATRFBASE",
        "NV_PFALCON_FALCON_DMATRFBASE1", "NV_PFALCON_FALCON_DMATRFMOFFS", "NV_PFALCON_FALCON_DMATRFFBOFFS",
        "NV_PFALCON_FALCON_CPUCTL", "NV_PFALCON_FALCON_CPUCTL_ALIAS", "NV_PFALCON_FALCON_BOOTVEC",
        "NV_PFALCON_FALCON_MAILBOX0", "NV_PFALCON_FALCON_MAILBOX1", "NV_PFALCON_FALCON_DMACTL",
        "NV_PFALCON_FALCON_HWCFG2", "NV_PFALCON_FALCON_RM", "NV_PFALCON_FBIF_TRANSCFG", "NV_PFALCON_FBIF_CTL",
        "NV_PFALCON2_FALCON_BROM_PARAADDR", "NV_PFALCON2_FALCON_BROM_ENGIDMASK",
        "NV_PFALCON2_FALCON_BROM_CURR_UCODE_ID", "NV_PFALCON2_FALCON_MOD_SEL", "NV_PRISCV_RISCV_CPUCTL",
        "NV_PRISCV_RISCV_BCR_CTRL", "NV_PROM_DATA", "NV_PBUS_VBIOS_SCRATCH",
        "NV_PFB_NISO_FLUSH_SYSMEM_ADDR", "NV_PFB_NISO_FLUSH_SYSMEM_ADDR_HI",
    ],
    "Blackwell": [
        "NV_PMC_BOOT_0", "NV_PMC_BOOT_42", "NV_PFB_PRI_MMU_WPR2_ADDR_LO", "NV_PFB_PRI_MMU_WPR2_ADDR_HI",
        "NV_PGC6_AON_SECURE_SCRATCH_GROUP_42", "NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE",
        "NV_VIRTUAL_FUNCTION_PRIV_FUNC_BAR1_BLOCK_LOW_ADDR", "NV_PBUS_BAR1_BLOCK", "NV_PGSP_QUEUE_HEAD",
        "NV_PFALCON_FALCON_HWCFG2", "NV_THERM_I2CS_SCRATCH", "NV_PFSP_EMEMC", "NV_PFSP_EMEMD",
        "NV_PFSP_QUEUE_HEAD", "NV_PFSP_QUEUE_TAIL", "NV_PFSP_MSGQ_HEAD", "NV_PFSP_MSGQ_TAIL",
        "NV_PRISCV_RISCV_CPUCTL", "NV_PFB_FBHUB0_PCIE_FLUSH_SYSMEM_ADDR_LO",
        "NV_PFB_FBHUB0_PCIE_FLUSH_SYSMEM_ADDR_HI",
    ],
}

# Values of register fields, from the register headers.
REGISTER_VALUES = [
    "NV_PMC_BOOT_42_ARCHITECTURE_GA100", "NV_PMC_BOOT_42_ARCHITECTURE_AD100", "NV_PMC_BOOT_42_ARCHITECTURE_GB200",
    "NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B", "NV_PFALCON_FBIF_TRANSCFG_MEM_TYPE_PHYSICAL",
    "NV_PFALCON2_FALCON_MOD_SEL_ALGO_RSA3K",
    "NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK_READ_PROTECTION_LEVEL0_ENABLE",
    "NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_0_GFW_BOOT_PROGRESS_COMPLETED",
    "NV_PGC6_BSI_SECURE_SCRATCH_14_BOOT_STAGE_3_HANDOFF_VALUE_DONE", "NV_PFALCON_FALCON_HWCFG2_MEM_SCRUBBING_DONE",
    "NV_PRISCV_RISCV_CPUCTL_ACTIVE_STAT_ACTIVE", "NV_PRISCV_RISCV_BCR_CTRL_VALID_TRUE",
    "NV_PFB_NISO_FLUSH_SYSMEM_ADDR_SHIFT", "NV_PFB_NISO_FLUSH_SYSMEM_ADDR_HI_MASK",
    "NV_PFB_FBHUB0_PCIE_FLUSH_SYSMEM_ADDR_HI_ADR_MASK",
    "NV_PRISCV_RISCV_BCR_CTRL_CORE_SELECT_RISCV", "NV_THERM_I2CS_SCRATCH_FSP_BOOT_COMPLETE_STATUS_SUCCESS",
]

# The falcons' register ranges, whose first address is a falcon's base.
UNITS = ["NV_PGSP", "NV_PSEC"]

# The GSP's formats, rig_nv_abi's, are the constants, fields, functions,
# events, structs and tables named GSP_ below, and the page-table entries.

# Page-table entries, versions 2 and 3: every field of each entry and the
# values named below.
MMU = {"V2": ("tu102_dev_mmu.h", "NV_MMU_VER2"), "V3": ("gh100_dev_mmu.h", "NV_MMU_VER3")}
MMU_ENTRIES = ["PTE", "PDE", "DUAL_PDE"]
MMU_VALUES = [
    "PTE_APERTURE_VIDEO_MEMORY", "PTE_APERTURE_PEER_MEMORY", "PTE_APERTURE_SYSTEM_COHERENT_MEMORY",
    "PTE_APERTURE_SYSTEM_NON_COHERENT_MEMORY", "PDE_APERTURE_INVALID", "PDE_APERTURE_VIDEO_MEMORY",
    "PDE_APERTURE_SYSTEM_COHERENT_MEMORY", "PDE_APERTURE_SYSTEM_NON_COHERENT_MEMORY",
    "DUAL_PDE_APERTURE_SMALL_VIDEO_MEMORY",
]
MMU_V3_VALUES = [
    "PTE_PCF_REGULAR_RW_ATOMIC_CACHED_ACE", "PTE_PCF_REGULAR_RW_ATOMIC_UNCACHED_ACE",
    "PDE_PCF_VALID_CACHED_ATS_NOT_ALLOWED", "DUAL_PDE_PCF_SMALL_VALID_CACHED_ATS_NOT_ALLOWED",
]
MMU_KIND = "NV_MMU_PTE_KIND_GENERIC_MEMORY"

# Constants, from the excerpts.
GSP_CONSTANTS = [
    "NV_VGPU_MSG_SIGNATURE_VALID", "NV_VGPU_MSG_RESULT_RPC_PENDING", "NV_VGPU_MSG_HEADER_VERSION_MAJOR_TOT",
    "NV_VGPU_MSG_HEADER_VERSION_MINOR_TOT", "GSP_FW_WPR_META_REVISION", "GSP_FW_WPR_META_MAGIC",
    "REGISTRY_TABLE_ENTRY_TYPE_DWORD", "GSP_SEQ_BUF_OPCODE_REG_WRITE", "GSP_SEQ_BUF_OPCODE_REG_MODIFY",
    "GSP_SEQ_BUF_OPCODE_REG_POLL", "GSP_SEQ_BUF_OPCODE_DELAY_US", "GSP_SEQ_BUF_OPCODE_REG_STORE",
    "GSP_SEQ_BUF_OPCODE_CORE_RESET", "GSP_SEQ_BUF_OPCODE_CORE_START", "GSP_SEQ_BUF_OPCODE_CORE_WAIT_FOR_HALT",
    "GSP_SEQ_BUF_OPCODE_CORE_RESUME",
]
PCI_CONSTANTS = [
    "BIT_HEADER_SIGNATURE", "BIT_TOKEN_FALCON_DATA", "BIT_TOKEN_V1_00_SIZE_6",
    "BIT_TOKEN_V1_00_SIZE_8", "BIT_DATA_FALCON_DATA_V2_SIZE_4", "FALCON_UCODE_ENTRY_APPID_FWSEC_PROD",
    "FALCON_UCODE_DESC_V3_SIZE_44", "BCRT30_RSA3K_SIG_SIZE", "NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_VERSION_V3",
    "FALCON_APPLICATION_INTERFACE_ENTRY_ID_DMEMMAPPER", "FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3_CMD_FRTS",
    "FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3_CMD_SB", "NV_VBIOS_FWSECLIC_SCRATCH_INDEX_15",
    "NV_VBIOS_FWSECLIC_SB_ERR_CODE_NONE",
    "FWSECLIC_READ_VBIOS_STRUCT_FLAGS", "FWSECLIC_FRTS_REGION_MEDIA_FB", "FWSECLIC_FRTS_REGION_SIZE_1MB_IN_4K",
    "NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_BASE", "NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_EXT",
    "OFFSETOF_PCI_EXP_ROM_PCI_DATA_STRUCT_PTR", "OFFSETOF_PCI_DATA_STRUCT_IMAGE_LEN",
    "OFFSETOF_PCI_DATA_STRUCT_CODE_TYPE", "PCI_ROM_IMAGE_BLOCK_SIZE", "BIT_HEADER_ID", "BIT_HEADER_SIZE_OFFSET",
    "FALCON_UCODE_TABLE_HDR_V1_VERSION", "FALCON_UCODE_TABLE_HDR_V1_SIZE_6", "FALCON_UCODE_TABLE_ENTRY_V1_SIZE_6",
    "FALCON_UCODE_ENTRY_APPID_FIRMWARE_SEC_LIC", "PCI_DATA_STRUCT_SIGNATURE", "PCI_DATA_STRUCT_SIGNATURE_NV",
    "PCI_DATA_STRUCT_SIGNATURE_NV2", "OFFSETOF_PCI_DATA_STRUCT_LEN", "OFFSETOF_PCI_DATA_STRUCT_LAST_IMAGE",
    "NV_PCI_DATA_EXT_SIG", "NV_PCI_DATA_EXT_REV_10", "NV_PCI_DATA_EXT_REV_11", "OFFSETOF_PCI_DATA_EXT_STRUCT_SIG",
    "OFFSETOF_PCI_DATA_EXT_STRUCT_LEN", "OFFSETOF_PCI_DATA_EXT_STRUCT_REV",
    "OFFSETOF_PCI_DATA_EXT_STRUCT_SUBIMAGE_LEN", "OFFSETOF_PCI_DATA_EXT_STRUCT_LAST_IMAGE",
]

# The GSP's boot and the resource manager it runs, from release 570.144.
PCI_CONSTANTS += [
    "LIBOS_MEMORY_REGION_CONTIGUOUS", "LIBOS_MEMORY_REGION_LOC_SYSMEM", "LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2",
    "GSP_DMA_TARGET_COHERENT_SYSTEM", "NVDM_TYPE_COT", "ADDR_SYSMEM", "ADDR_FBMEM", "NV_MEMORY_CACHED",
    "NV01_ROOT", "NV01_DEVICE_0", "NV20_SUBDEVICE_0", "FERMI_VASPACE_A", "AMPERE_CHANNEL_GPFIFO_A",
    "BLACKWELL_CHANNEL_GPFIFO_A", "AMPERE_COMPUTE_B", "ADA_COMPUTE_A", "BLACKWELL_COMPUTE_B", "AMPERE_DMA_COPY_B",
    "BLACKWELL_DMA_COPY_B",
    "NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_INFO", "NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO",
    "NV2080_CTRL_CMD_GPU_PROMOTE_CTX", "NV2080_CTRL_CMD_FIFO_GET_DEVICE_INFO_TABLE",
    "NV90F1_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES",
    "NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS",
    "NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_PM",
    "NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_PATCH",
    "NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_GPCS", "NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_TPC_PER_GPC",
    "NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_SM_PER_TPC", "NV2080_CTRL_GR_INFO_INDEX_MAX_WARPS_PER_SM",
    "NV2080_CTRL_GR_INFO_INDEX_SM_VERSION", "NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES",
    "NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING", "NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED",
    "MCTP_MSG_HEADER_TYPE_VENDOR_PCI", "MCTP_MSG_HEADER_VENDOR_ID_NV", "NV2080_ENGINE_TYPE_GRAPHICS",
    "KEPLER_CHANNEL_GROUP_A", "NV2080_CTRL_CMD_INTERNAL_PERF_BOOST_SET_2X",
    "NV2080_CTRL_PERF_BOOST_FLAGS_CMD_BOOST_TO_MAX", "NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_YES",
    "NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_PRIORITY_HIGH", "NV2080_CTRL_PERF_BOOST_DURATION_INFINITE",
    "NVA06C_CTRL_CMD_GPFIFO_SCHEDULE", "NV83DE_CTRL_CMD_DEBUG_READ_ALL_SM_ERROR_STATES",
    "NV83DE_CTRL_CMD_DEBUG_READ_MMU_FAULT_INFO",
    "NVOS04_FLAGS_CHANNEL_USERD_INDEX_PAGE_FIXED_TRUE", "NV_VIRTUAL_FUNCTION_DOORBELL_RUNLIST_DOORBELL_ENABLE",
]

# Bit fields of 32-bit words, "hi:lo": (lowest bit, bits).
GSP_FIELDS = ["NV_VGPU_MSG_HEADER_VERSION_MAJOR", "NV_VGPU_MSG_HEADER_VERSION_MINOR"]
PCI_FIELDS = ["NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_VERSION", "NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_SIZE",
          "MCTP_HEADER_SOM", "MCTP_HEADER_EOM", "MCTP_MSG_HEADER_TYPE", "MCTP_MSG_HEADER_VENDOR_ID",
          "MCTP_MSG_HEADER_NVDM_TYPE", "NV_VBIOS_FWSECLIC_SB_ERR_CODE",
          "NVOS04_FLAGS_CHANNEL_USERD_INDEX_VALUE", "NVOS04_FLAGS_CHANNEL_USERD_INDEX_FIXED",
          "NVOS04_FLAGS_CHANNEL_USERD_INDEX_PAGE_VALUE", "NVOS04_FLAGS_CHANNEL_USERD_INDEX_PAGE_FIXED",
          "NV_CTRL_VF_DOORBELL_VECTOR", "NV_CTRL_VF_DOORBELL_RUNLIST_ID", "NV_VIRTUAL_FUNCTION_DOORBELL_VECTOR",
          "NV_VIRTUAL_FUNCTION_DOORBELL_RUNLIST_ID", "NV_VIRTUAL_FUNCTION_DOORBELL_RUNLIST_DOORBELL",
          "NV2080_CTRL_PERF_BOOST_FLAGS_CMD", "NV2080_CTRL_PERF_BOOST_FLAGS_CUDA",
          "NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_PRIORITY"]

# The function and event numbers read, from rpc_global_enums.h.
GSP_FUNCTIONS = ["CONTINUATION_RECORD"]
GSP_EVENTS = ["RC_TRIGGERED", "MMU_FAULT_QUEUED", "OS_ERROR_LOG", "GSP_POST_NOCAT_RECORD"]
PCI_FUNCTIONS = ["GSP_RM_ALLOC", "GSP_RM_CONTROL", "SET_PAGE_DIRECTORY", "GSP_SET_SYSTEM_INFO", "SET_REGISTRY",
                 "UNLOADING_GUEST_DRIVER", "FREE"]
PCI_EVENTS = ["GSP_INIT_DONE", "GSP_RUN_CPU_SEQUENCER", "UCODE_LIBOS_PRINT"]

# Structs, by C name: the module they become and the fields read ("a__b" for
# a field b of a struct a).
GSP_STRUCTS = {
    "GSP_SEQ_BUF_PAYLOAD_REG_WRITE": ("Seq_reg_write", ["addr", "val"]),
    "GSP_SEQ_BUF_PAYLOAD_REG_MODIFY": ("Seq_reg_modify", ["addr", "mask", "val"]),
    "GSP_SEQ_BUF_PAYLOAD_REG_POLL": ("Seq_reg_poll", ["addr", "mask", "val", "timeout", "error"]),
    "GSP_SEQ_BUF_PAYLOAD_DELAY_US": ("Seq_delay_us", ["val"]),
    "GSP_SEQ_BUF_PAYLOAD_REG_STORE": ("Seq_reg_store", ["addr", "index"]),
    "GSP_MSG_QUEUE_ELEMENT": ("Queue_element", ["authTagBuffer", "aadBuffer", "checkSum", "seqNum",
                                                "elemCount", "rpc"]),
    "rpc_message_header_v": ("Rpc_header", ["header_version", "signature", "length", "function",
                                            "rpc_result", "rpc_result_private", "sequence"]),
    "rpc_rc_triggered_v17_02": ("Rpc_rc_triggered", ["nv2080EngineType", "chid", "exceptType", "scope"]),
    "rpc_os_error_log_v17_00": ("Rpc_os_error_log", ["exceptType", "errString"]),
    "NV2080CtrlNocatJournalInsertRecord": ("Nocat_record", ["source", "errorCode"]),
    "GspFwWprMeta": ("Wpr_meta", [
        "magic", "revision", "sysmemAddrOfRadix3Elf", "sizeOfRadix3Elf", "sysmemAddrOfBootloader",
        "sizeOfBootloader", "bootloaderCodeOffset", "bootloaderDataOffset", "bootloaderManifestOffset",
        "sysmemAddrOfSignature", "sizeOfSignature", "gspFwRsvdStart", "nonWprHeapOffset", "nonWprHeapSize",
        "gspFwWprStart", "gspFwHeapOffset", "gspFwHeapSize", "gspFwOffset", "bootBinOffset", "frtsOffset",
        "frtsSize", "gspFwWprEnd", "fbSize", "vgaWorkspaceOffset", "vgaWorkspaceSize", "pmuReservedSize"]),
    "rpc_gsp_rm_alloc_v03_00": ("Rpc_rm_alloc", ["hClient", "hParent", "hObject", "hClass", "status",
                                                 "paramsSize", "flags", "params"]),
    "rpc_gsp_rm_control_v03_00": ("Rpc_rm_control", ["hClient", "hObject", "cmd", "status", "paramsSize",
                                                     "flags", "params"]),
    "rpc_set_page_directory_v1E_05": ("Rpc_set_page_directory", ["hClient", "hDevice", "pasid", "params"]),
    "rpc_free_v03_00": ("Rpc_free", ["params"]),
    "NVOS00_PARAMETERS_v03_00": ("Free_params", ["hRoot", "hObjectParent", "hObjectOld", "status"]),
    "NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS_v1E_05": ("Set_page_directory", [
        "physAddress", "numEntries", "flags", "hVASpace", "chId", "subDeviceId", "pasid"]),
    "rpc_unloading_guest_driver_v1F_07": ("Rpc_unloading", ["bInPMTransition", "bGc6Entering", "newLevel"]),
    "rpc_run_cpu_sequencer_v17_00": ("Rpc_cpu_sequencer", ["bufferSizeDWord", "cmdIndex", "regSaveArea",
                                                           "commandBuffer"]),
    "PACKED_REGISTRY_ENTRY": ("Registry_entry", ["nameOffset", "type", "data", "length"]),
    "PACKED_REGISTRY_TABLE": ("Registry_table", ["size", "numEntries", "entries"]),
}
PCI_STRUCTS = {
    "msgqTxHeader": ("Msgq_tx_header", ["version", "size", "msgSize", "msgCount", "writePtr", "flags",
                                        "rxHdrOff", "entryOff"]),
    "msgqRxHeader": ("Msgq_rx_header", ["readPtr"]),
    "RM_RISCV_UCODE_DESC": ("Riscv_ucode_desc", ["monitorDataOffset", "monitorCodeOffset", "manifestOffset"]),
    "FALCON_APPLICATION_INTERFACE_HEADER_V1": ("App_interface_header", ["headerSize", "entrySize",
                                                                       "entryCount"]),
    "FALCON_APPLICATION_INTERFACE_ENTRY_V1": ("App_interface_entry", ["id", "dmemOffset"]),
    "FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3": ("Dmem_mapper", ["cmd_in_buffer_offset", "init_cmd"]),
    "FWSECLIC_READ_VBIOS_DESC": ("Read_vbios_desc", ["version", "size", "gfwImageOffset", "gfwImageSize",
                                                     "flags"]),
    "FWSECLIC_FRTS_REGION_DESC": ("Frts_region_desc", []),
    "FWSECLIC_FRTS_CMD": ("Frts_cmd", [
        "readVbiosDesc__version", "readVbiosDesc__size", "readVbiosDesc__gfwImageOffset",
        "readVbiosDesc__gfwImageSize", "readVbiosDesc__flags", "frtsRegionDesc__version",
        "frtsRegionDesc__size", "frtsRegionDesc__frtsRegionOffset4K", "frtsRegionDesc__frtsRegionSize",
        "frtsRegionDesc__frtsRegionMediaType"]),
    "GSP_ARGUMENTS_CACHED": ("Gsp_arguments", ["messageQueueInitArguments", "bDmemStack"]),
    "MESSAGE_QUEUE_INIT_ARGUMENTS": ("Queue_init_args", ["sharedMemPhysAddr", "pageTableEntryCount",
                                                         "cmdQueueOffset", "statQueueOffset"]),
    "LibosMemoryRegionInitArgument": ("Libos_region", ["id8", "pa", "size", "kind", "loc"]),
    "GspSystemInfo": ("System_info", ["gpuPhysAddr", "gpuPhysFbAddr", "gpuPhysInstAddr",
                                      "nvDomainBusDeviceFunc", "maxUserVa", "pciConfigMirrorBase",
                                      "pciConfigMirrorSize", "PCIDeviceID", "PCISubDeviceID", "PCIRevisionID",
                                      "bIsPassthru"]),
    "GSP_FMC_BOOT_PARAMS": ("Fmc_boot_params", ["bootGspRmParams", "gspRmParams"]),
    "GSP_ACR_BOOT_GSP_RM_PARAMS": ("Acr_boot_params", ["target", "gspRmDescSize", "gspRmDescOffset",
                                                       "bIsGspRmBoot"]),
    "GSP_RM_PARAMS": ("Rm_params", ["target", "bootArgsOffset"]),
    "NVDM_PAYLOAD_COT": ("Cot_payload", ["version", "size", "gspFmcSysmemOffset", "frtsVidmemOffset",
                                         "frtsVidmemSize", "hash384", "publicKey", "signature",
                                         "gspBootArgsSysmemOffset"]),
    "NV0000_ALLOC_PARAMETERS": ("Nv0000_alloc", ["hClient"]),
    "NV0080_ALLOC_PARAMETERS": ("Nv0080_alloc", ["deviceId", "hClientShare", "vaMode"]),
    "NV2080_ALLOC_PARAMETERS": ("Nv2080_alloc", ["subDeviceId"]),
    "NV_VASPACE_ALLOCATION_PARAMETERS": ("Vaspace_alloc", ["index", "flags", "vaSize", "vaBase"]),
    "NV_MEMORY_DESC_PARAMS": ("Memory_desc", ["base", "size", "addressSpace", "cacheAttrib"]),
    "NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS": ("Gpfifo_alloc", [
        "gpFifoOffset", "gpFifoEntries", "flags", "hContextShare", "hVASpace", "hUserdMemory", "userdOffset",
        "engineType", "cid", "hObjectError", "hObjectBuffer", "instanceMem", "userdMem", "ramfcMem",
        "mthdbufMem", "errorNotifierMem", "internalFlags"]),
    "NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS": ("Channel_group_alloc", ["engineType"]),
    "NV2080_CTRL_INTERNAL_PERF_BOOST_SET_PARAMS_2X": ("Perf_boost_2x", ["flags", "duration"]),
    "NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_PARAMS": ("Device_info_table", ["numEntries", "entries"]),
    "NV2080_CTRL_FIFO_DEVICE_ENTRY": ("Device_entry", ["engineData"]),
    "NV2080_CTRL_INTERNAL_STATIC_GR_GET_INFO_PARAMS": ("Static_gr_info", ["engineInfo"]),
    "NV2080_CTRL_INTERNAL_STATIC_GR_INFO": ("Gr_info_list", ["infoList"]),
    "NV2080_CTRL_INTERNAL_GR_INFO": ("Internal_gr_info", ["data"]),
    "NV2080_CTRL_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO_PARAMS": ("Context_buffers_info", [
        "engineContextBuffersInfo"]),
    "NV2080_CTRL_INTERNAL_STATIC_GR_CONTEXT_BUFFERS_INFO": ("Context_buffers", ["engine"]),
    "NV2080_CTRL_INTERNAL_ENGINE_CONTEXT_BUFFER_INFO": ("Context_buffer", ["size", "alignment"]),
    "NV2080_CTRL_GPU_PROMOTE_CTX_PARAMS": ("Promote_ctx", ["engineType", "hChanClient", "hObject",
                                                         "entryCount", "promoteEntry"]),
    "NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ENTRY": ("Promote_entry", [
        "gpuPhysAddr", "gpuVirtAddr", "size", "physAttr", "bufferId", "bInitialize", "bNonmapped"]),
    "NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS": ("Reserved_pdes", [
        "pageSize", "virtAddrLo", "virtAddrHi", "numLevelsToCopy", "levels", "levels__physAddress",
        "levels__size", "levels__aperture", "levels__pageShift"]),
    "nvfw_bin_hdr": ("Bin_header", ["bin_magic", "header_offset", "data_offset", "data_size"]),
    "nvfw_hs_header_v2": ("Hs_header", ["sig_prod_offset", "sig_prod_size", "patch_loc", "patch_sig",
                                        "meta_data_offset", "num_sig", "header_offset"]),
    "nvfw_hs_load_header_v2": ("Hs_load_header", ["os_data_offset", "os_data_size", "num_apps", "app"]),}

# Types read only as the elements of a flexible array that ends a struct,
# which adds no bytes: they stay undefined, the array at a multiple of 8, where
# any alignment of theirs up to 8 puts it. rpc_generic_union is the union of
# every RPC's parameters.
FLEXIBLE = {"rpc_generic_union"}

# The VBIOS's structures, packed as their format strings say: (C name,
# module, format macro).
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

# Name tables: (OCaml name, header, the defines read, as a pattern whose
# group 1 is the name an entry carries). The errors leave out the recovery
# levels and the count, which are no error.
GSP_TABLES = [
    ("robust_channel_errors", "nverror.h", r"ROBUST_CHANNEL_(?!ERROR_RECOVERY_LEVEL_|LAST_ERROR)(\w+)"),
]

# The firmware images, by path under a firmware directory, per family: the
# GSP's image (one file, with a signature section per family), its
# bootloader, the image that starts it, and on Ampere and Ada the booter
# that takes the boot's protected memory down at its stop.
FIRMWARE = {
    "Ampere": ["nvidia/ga102/gsp/gsp-570.144.bin", "nvidia/ga102/gsp/bootloader-570.144.bin",
               "nvidia/ga102/gsp/booter_load-570.144.bin", "nvidia/ga102/gsp/booter_unload-570.144.bin"],
    "Ada": ["nvidia/ga102/gsp/gsp-570.144.bin", "nvidia/ad102/gsp/bootloader-570.144.bin",
            "nvidia/ad102/gsp/booter_load-570.144.bin", "nvidia/ad102/gsp/booter_unload-570.144.bin"],
    "Blackwell": ["nvidia/ga102/gsp/gsp-570.144.bin", "nvidia/gb202/gsp/bootloader-570.144.bin",
                  "nvidia/gb202/gsp/fmc-570.144.bin"],
}


# rig_nv_abi

ABI_OUT = LIB / "abi/defs.ml"
CLASSES = R570 + SDK + "class/"

# Each excerpt and the header it is cut from.
ABI_SOURCES = {
    "clc56f.h": CLASSES + "clc56f.h",  # AMPERE_CHANNEL_GPFIFO_A: the host
    "clc7c0.h": CLASSES + "clc7c0.h",  # AMPERE_COMPUTE_B
    "clc9c0.h": CLASSES + "clc9c0.h",  # ADA_COMPUTE_A
    "clcec0.h": CLASSES + "clcec0.h",  # BLACKWELL_COMPUTE_B
    "clc7b5.h": R570 + UVM + "clc7b5.h",  # AMPERE_DMA_COPY_B
    "clc9b5.h": R570 + UVM + "clc9b5.h",  # BLACKWELL_DMA_COPY_A
    "clc7c0qmd.h": DOC + "classes/compute/clc7c0qmd.h",
    "clc9c0qmd.h": DOC + "classes/compute/clc9c0qmd.h",
    "clcec0qmd.h": DOC + "classes/compute/clcec0qmd.h",
}

# The classes the library supports, by header.
CLASS_IDS = [("clc7c0.h", "AMPERE_COMPUTE_B"), ("clc9c0.h", "ADA_COMPUTE_A"), ("clcec0.h", "BLACKWELL_COMPUTE_B")]

# Constants and bit fields ("hi:lo" in the headers, (lowest bit, bits) here),
# by header. The copy class's names hold for 0xc9b5 under its prefix.
HOST = "clc56f.h"
HOST_CONSTANTS = [
    "NVC56F_SET_OBJECT", "NVC56F_SEM_ADDR_LO", "NVC56F_NON_STALL_INTERRUPT",
    "NVC56F_SEM_EXECUTE_OPERATION_ACQ_CIRC_GEQ", "NVC56F_SEM_EXECUTE_OPERATION_RELEASE",
    "NVC56F_SEM_EXECUTE_PAYLOAD_SIZE_64BIT", "NVC56F_SEM_EXECUTE_RELEASE_WFI_EN",
    "NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP_EN", "NVC56F_GP_ENTRY1_LEVEL_SUBROUTINE", "NVC56F_DMA_SEC_OP_INC_METHOD",
]
HOST_FIELDS = [
    "NVC56F_SEM_EXECUTE_OPERATION", "NVC56F_SEM_EXECUTE_PAYLOAD_SIZE", "NVC56F_SEM_EXECUTE_RELEASE_WFI",
    "NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP", "NVC56F_GP_ENTRY0_GET", "NVC56F_GP_ENTRY1_GET_HI",
    "NVC56F_GP_ENTRY1_LEVEL", "NVC56F_GP_ENTRY1_LENGTH", "NVC56F_DMA_METHOD_ADDRESS", "NVC56F_DMA_METHOD_SUBCHANNEL",
    "NVC56F_DMA_METHOD_COUNT", "NVC56F_DMA_SEC_OP",
]
COMPUTE = "clc7c0.h"
COMPUTE_CONSTANTS = [
    "NVC7C0_SET_SHADER_SHARED_MEMORY_WINDOW_A", "NVC7C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A",
    "NVC7C0_SET_SHADER_LOCAL_MEMORY_A", "NVC7C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A",
    "NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI", "NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI_INSTRUCTION_TRUE",
    "NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI_GLOBAL_DATA_TRUE", "NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI_CONSTANT_TRUE",
    "NVC7C0_SEND_PCAS_A", "NVC7C0_SEND_SIGNALING_PCAS2_B", "NVC7C0_WAIT_FOR_IDLE",
    "NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_PREFETCH_SCHEDULE",
]
COMPUTE_FIELDS = [
    "NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI_INSTRUCTION", "NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI_GLOBAL_DATA",
    "NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI_CONSTANT",
    "NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION",
]
COPY, LATER_COPY = "clc7b5.h", "clc9b5.h"
COPY_CONSTANTS = [
    "NVC7B5_SET_SEMAPHORE_A", "NVC7B5_LAUNCH_DMA", "NVC7B5_OFFSET_IN_UPPER", "NVC7B5_OFFSET_OUT_UPPER",
    "NVC7B5_LINE_LENGTH_IN", "NVC7B5_SET_REMAP_CONST_B",
    "NVC7B5_LAUNCH_DMA_DATA_TRANSFER_TYPE_NON_PIPELINED", "NVC7B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT_PITCH",
    "NVC7B5_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH", "NVC7B5_LAUNCH_DMA_FLUSH_ENABLE_TRUE",
    "NVC7B5_LAUNCH_DMA_FLUSH_TYPE_SYS", "NVC7B5_LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_ONE_WORD_SEMAPHORE",
    "NVC7B5_LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_FOUR_WORD_SEMAPHORE",
    "NVC7B5_LAUNCH_DMA_SEMAPHORE_PAYLOAD_SIZE_TWO_WORD", "NVC7B5_LAUNCH_DMA_MULTI_LINE_ENABLE_TRUE",
    "NVC7B5_LAUNCH_DMA_REMAP_ENABLE_TRUE", "NVC7B5_SET_REMAP_COMPONENTS_DST_X_CONST_B",
    "NVC7B5_SET_REMAP_COMPONENTS_COMPONENT_SIZE_ONE", "NVC7B5_SET_REMAP_COMPONENTS_COMPONENT_SIZE_FOUR",
    "NVC7B5_SET_REMAP_COMPONENTS_NUM_DST_COMPONENTS_ONE",
]
COPY_FIELDS = [
    "NVC7B5_LAUNCH_DMA_DATA_TRANSFER_TYPE", "NVC7B5_LAUNCH_DMA_FLUSH_ENABLE", "NVC7B5_LAUNCH_DMA_FLUSH_TYPE",
    "NVC7B5_LAUNCH_DMA_SEMAPHORE_TYPE", "NVC7B5_LAUNCH_DMA_SEMAPHORE_PAYLOAD_SIZE",
    "NVC7B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT", "NVC7B5_LAUNCH_DMA_DST_MEMORY_LAYOUT",
    "NVC7B5_LAUNCH_DMA_MULTI_LINE_ENABLE", "NVC7B5_LAUNCH_DMA_REMAP_ENABLE",
    "NVC7B5_SET_REMAP_COMPONENTS_DST_X", "NVC7B5_SET_REMAP_COMPONENTS_COMPONENT_SIZE",
    "NVC7B5_SET_REMAP_COMPONENTS_NUM_DST_COMPONENTS",
]

# The methods an encoder writes after one header, which must be consecutive.
# Only the first is emitted.
RUNS = [
    (HOST, ["NVC56F_SEM_ADDR_LO", "NVC56F_SEM_ADDR_HI", "NVC56F_SEM_PAYLOAD_LO", "NVC56F_SEM_PAYLOAD_HI",
            "NVC56F_SEM_EXECUTE"]),
    (COMPUTE, ["NVC7C0_SET_SHADER_SHARED_MEMORY_WINDOW_A", "NVC7C0_SET_SHADER_SHARED_MEMORY_WINDOW_B"]),
    (COMPUTE, ["NVC7C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A", "NVC7C0_SET_SHADER_LOCAL_MEMORY_WINDOW_B"]),
    (COMPUTE, ["NVC7C0_SET_SHADER_LOCAL_MEMORY_A", "NVC7C0_SET_SHADER_LOCAL_MEMORY_B"]),
    (COMPUTE, ["NVC7C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A", "NVC7C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_B",
               "NVC7C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_C"]),
    (COPY, ["NVC7B5_SET_SEMAPHORE_A", "NVC7B5_SET_SEMAPHORE_B", "NVC7B5_SET_SEMAPHORE_PAYLOAD",
            "NVC7B5_SET_SEMAPHORE_PAYLOAD_UPPER"]),
    (COPY, ["NVC7B5_OFFSET_IN_UPPER", "NVC7B5_OFFSET_IN_LOWER", "NVC7B5_OFFSET_OUT_UPPER", "NVC7B5_OFFSET_OUT_LOWER",
            "NVC7B5_PITCH_IN", "NVC7B5_PITCH_OUT", "NVC7B5_LINE_LENGTH_IN", "NVC7B5_LINE_COUNT"]),
    (COPY, ["NVC7B5_OFFSET_OUT_UPPER", "NVC7B5_OFFSET_OUT_LOWER"]),
    (COPY, ["NVC7B5_SET_REMAP_CONST_B", "NVC7B5_SET_REMAP_COMPONENTS"]),
]

# Launch descriptors: version 3, which 0xc7c0 and 0xc9c0 read, and version 5,
# which 0xcec0 reads. Each field the encoder writes is named in OCaml by the
# first name, and by the name each version gives it; None where the version
# has none. An (i) name is the field of index i: a constant bank's, or
# release k's at i = k. A name's suffix _SHIFTEDn says its field holds the value
# shifted right by n.
QMD = [("clc7c0qmd.h", "NVC7C0_QMDV03_00", 3), ("clcec0qmd.h", "NVCEC0_QMDV05_00", 5)]
QMD_SAME = ("clc9c0qmd.h", "NVC9C0_QMDV03_00")
QMD_FIELDS = [
    ("qmd_major_version", "QMD_MAJOR_VERSION", "QMD_MAJOR_VERSION"),
    ("qmd_type", None, "QMD_TYPE"),
    ("sm_global_caching_enable", "SM_GLOBAL_CACHING_ENABLE", None),
    ("qmd_group_id", "QMD_GROUP_ID", "QMD_GROUP_ID"),
    ("grid_width", "CTA_RASTER_WIDTH", "GRID_WIDTH"),
    ("grid_height", "CTA_RASTER_HEIGHT", "GRID_HEIGHT"),
    ("grid_depth", "CTA_RASTER_DEPTH", "GRID_DEPTH"),
    ("cta_thread_dimension0", "CTA_THREAD_DIMENSION0", "CTA_THREAD_DIMENSION0"),
    ("cta_thread_dimension1", "CTA_THREAD_DIMENSION1", "CTA_THREAD_DIMENSION1"),
    ("cta_thread_dimension2", "CTA_THREAD_DIMENSION2", "CTA_THREAD_DIMENSION2"),
    ("register_count", "REGISTER_COUNT_V", "REGISTER_COUNT"),
    ("barrier_count", "BARRIER_COUNT", "BARRIER_COUNT"),
    ("shared_memory_size", "SHARED_MEMORY_SIZE", "SHARED_MEMORY_SIZE_SHIFTED7"),
    ("min_sm_config_shared_mem_size", "MIN_SM_CONFIG_SHARED_MEM_SIZE", "MIN_SM_CONFIG_SHARED_MEM_SIZE"),
    ("target_sm_config_shared_mem_size", "TARGET_SM_CONFIG_SHARED_MEM_SIZE", "TARGET_SM_CONFIG_SHARED_MEM_SIZE"),
    ("max_sm_config_shared_mem_size", "MAX_SM_CONFIG_SHARED_MEM_SIZE", "MAX_SM_CONFIG_SHARED_MEM_SIZE"),
    ("shader_local_memory_high_size", "SHADER_LOCAL_MEMORY_HIGH_SIZE", "SHADER_LOCAL_MEMORY_HIGH_SIZE_SHIFTED4"),
    ("program_address_lower", "PROGRAM_ADDRESS_LOWER", "PROGRAM_ADDRESS_LOWER_SHIFTED4"),
    ("program_address_upper", "PROGRAM_ADDRESS_UPPER", "PROGRAM_ADDRESS_UPPER_SHIFTED4"),
    ("program_prefetch_addr_lower_shifted", "PROGRAM_PREFETCH_ADDR_LOWER_SHIFTED", "PROGRAM_PREFETCH_ADDR_LOWER_SHIFTED"),
    ("program_prefetch_addr_upper_shifted", "PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED", "PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED"),
    ("program_prefetch_size", "PROGRAM_PREFETCH_SIZE", "PROGRAM_PREFETCH_SIZE"),
    ("sass_version", "SASS_VERSION", "SASS_VERSION"),
    ("api_visible_call_limit", "API_VISIBLE_CALL_LIMIT", "API_VISIBLE_CALL_LIMIT"),
    ("sampler_index", "SAMPLER_INDEX", "SAMPLER_INDEX"),
    ("cwd_membar_type", "CWD_MEMBAR_TYPE", "CWD_MEMBAR_TYPE"),
    ("invalidate_texture_header_cache", "INVALIDATE_TEXTURE_HEADER_CACHE", "INVALIDATE_TEXTURE_HEADER_CACHE"),
    ("invalidate_texture_sampler_cache", "INVALIDATE_TEXTURE_SAMPLER_CACHE", "INVALIDATE_TEXTURE_SAMPLER_CACHE"),
    ("invalidate_texture_data_cache", "INVALIDATE_TEXTURE_DATA_CACHE", "INVALIDATE_TEXTURE_DATA_CACHE"),
    ("invalidate_shader_data_cache", "INVALIDATE_SHADER_DATA_CACHE", "INVALIDATE_SHADER_DATA_CACHE"),
    ("dependent_qmd0_pointer", "DEPENDENT_QMD0_POINTER", "DEPENDENT_QMD0_POINTER"),
    ("dependent_qmd0_action", "DEPENDENT_QMD0_ACTION", "DEPENDENT_QMD0_ACTION"),
    ("dependent_qmd0_prefetch", "DEPENDENT_QMD0_PREFETCH", "DEPENDENT_QMD0_PREFETCH"),
    ("dependent_qmd0_enable", "DEPENDENT_QMD0_ENABLE", "DEPENDENT_QMD0_ENABLE"),
]
QMD_BANK_FIELDS = [
    ("constant_buffer_addr_lower", "CONSTANT_BUFFER_ADDR_LOWER(i)", "CONSTANT_BUFFER_ADDR_LOWER_SHIFTED6(i)"),
    ("constant_buffer_addr_upper", "CONSTANT_BUFFER_ADDR_UPPER(i)", "CONSTANT_BUFFER_ADDR_UPPER_SHIFTED6(i)"),
    ("constant_buffer_size_shifted4", "CONSTANT_BUFFER_SIZE_SHIFTED4(i)", "CONSTANT_BUFFER_SIZE_SHIFTED4(i)"),
    ("constant_buffer_valid", "CONSTANT_BUFFER_VALID(i)", "CONSTANT_BUFFER_VALID(i)"),
    ("constant_buffer_invalidate", "CONSTANT_BUFFER_INVALIDATE(i)", "CONSTANT_BUFFER_INVALIDATE(i)"),
]
QMD_RELEASES = 2
QMD_RELEASE_FIELDS = [
    ("enable", "RELEASE{k}_ENABLE", "RELEASE_ENABLE(i)"),
    ("structure_size", "RELEASE{k}_STRUCTURE_SIZE", "RELEASE_STRUCTURE_SIZE(i)"),
    ("payload64b", "RELEASE{k}_PAYLOAD64B", "RELEASE_PAYLOAD64B(i)"),
    ("membar_type", "RELEASE{k}_MEMBAR_TYPE", "RELEASE_MEMBAR_TYPE(i)"),
    ("address_lower", "RELEASE{k}_ADDRESS_LOWER", "RELEASE_SEMAPHORE{k}_ADDR_LOWER"),
    ("address_upper", "RELEASE{k}_ADDRESS_UPPER", "RELEASE_SEMAPHORE{k}_ADDR_UPPER"),
    ("payload_lower", "RELEASE{k}_PAYLOAD_LOWER", "RELEASE_SEMAPHORE{k}_PAYLOAD_LOWER"),
    ("payload_upper", "RELEASE{k}_PAYLOAD_UPPER", "RELEASE_SEMAPHORE{k}_PAYLOAD_UPPER"),
]
# The values the encoder writes into fields, the same in each version that
# names them.
QMD_VALUES = [
    ("qmd_type_grid_cta", None, "QMD_TYPE_GRID_CTA"),
    ("cwd_membar_type_l1_sysmembar", "CWD_MEMBAR_TYPE_L1_SYSMEMBAR", "CWD_MEMBAR_TYPE_L1_SYSMEMBAR"),
    ("api_visible_call_limit_no_check", "API_VISIBLE_CALL_LIMIT_NO_CHECK", "API_VISIBLE_CALL_LIMIT_NO_CHECK"),
    ("sampler_index_via_header_index", "SAMPLER_INDEX_VIA_HEADER_INDEX", "SAMPLER_INDEX_VIA_HEADER_INDEX"),
    ("dependent_qmd0_action_qmd_schedule", "DEPENDENT_QMD0_ACTION_QMD_SCHEDULE", "DEPENDENT_QMD0_ACTION_QMD_SCHEDULE"),
    ("release_structure_size_semaphore_four_words", "RELEASE0_STRUCTURE_SIZE_SEMAPHORE_FOUR_WORDS",
     "RELEASE_STRUCTURE_SIZE_SEMAPHORE_FOUR_WORDS"),
    ("release_membar_type_fe_none", "RELEASE0_MEMBAR_TYPE_FE_NONE", "RELEASE_MEMBAR_TYPE_FE_NONE"),
    ("release_structure_size_semaphore_two_words", "RELEASE0_STRUCTURE_SIZE_SEMAPHORE_TWO_WORDS",
     "RELEASE_STRUCTURE_SIZE_SEMAPHORE_TWO_WORDS"),
]
# The fields only one version has.
OPTIONAL = {"qmd_type", "sm_global_caching_enable"}
# The shifts the encoder applies to its values: (OCaml name, the field each
# version names it by).
QMD_SHIFTS = [
    ("program_address_shift", "PROGRAM_ADDRESS_LOWER", "PROGRAM_ADDRESS_LOWER_SHIFTED4"),
    ("constant_buffer_addr_shift", "CONSTANT_BUFFER_ADDR_LOWER(i)", "CONSTANT_BUFFER_ADDR_LOWER_SHIFTED6(i)"),
    ("shader_local_memory_shift", "SHADER_LOCAL_MEMORY_HIGH_SIZE", "SHADER_LOCAL_MEMORY_HIGH_SIZE_SHIFTED4"),
    ("shared_memory_shift", "SHARED_MEMORY_SIZE", "SHARED_MEMORY_SIZE_SHIFTED7"),
]


# The excerpts

def key(name, url):
    """The excerpt [name] of [url], as its path under headers/."""
    return f"{origin(url)}/{name}"


def excerpt_sources():
    """{excerpt: URL} of every excerpt, one of each file, and of the files
    kept whole."""
    out = {}
    named = [(h, u) for d in LIBRARIES for r in RELEASES for h, u in d.headers(r).items()]
    named += [*PCI_SOURCES.items(), *WHOLE.items(), *ABI_SOURCES.items()]
    for h, url in named:
        k = key(h, url)
        if out.setdefault(k, url) != url:
            sys.exit(f"{k} is cut from both {out[k]} and {url}")
    if len(set(out.values())) != len(out):
        sys.exit("a file has two excerpts")
    return out


SOURCES = excerpt_sources()


# C headers

COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(\w+)(\([\w\s,]*\))?((?:[^\n]*\\\n)*[^\n]*)$", re.M)
TYPEDEF = re.compile(r"^[ \t]*typedef\b", re.M)
TAGGED = re.compile(r"^[ \t]*(struct|union|enum)[ \t]+(\w+)\s*\{", re.M)
IDENT = re.compile(r"\b[A-Za-z_]\w*\b")
PRAGMA = re.compile(r"^[ \t]*#[ \t]*pragma[ \t]+pack[ \t]*\(([^)]*)\)[^\n]*$", re.M)
DIRECTIVE = re.compile(r"^[ \t]*#[ \t]*(if|ifdef|ifndef|elif|else|endif|define|undef)\b(.*)$")

# The macros a 64-bit Linux build defines that the headers' conditionals
# test, and those that make the RPC headers define their structures; every
# other name a conditional tests is undefined.
PLATFORM = {"__linux__", "__LP64__", "RPC_MESSAGE_STRUCTURES", "RPC_STRUCTURES", "SDK_STRUCTURES"}

# The scalar types, (bytes, alignment), in the 64-bit Linux ABIs.
SCALARS = {
    **{t: (1, 1) for t in ["NvU8", "NvS8", "NvV8", "NvBool", "char", "NvChar", "u8"]},
    **{t: (2, 2) for t in ["NvU16", "NvS16", "NvV16", "u16"]},
    **{t: (4, 4) for t in ["NvU32", "NvS32", "NvV32", "NvHandle", "NV_STATUS", "NvF32", "int", "unsigned", "u32"]},
    **{t: (8, 8) for t in ["NvU64", "NvS64", "NvP64", "NvLength", "NvUPtr", "NvF64", "long", "size_t", "u64"]},
}
KEYWORDS = {"const", "volatile", "struct", "union", "enum", "unsigned", "signed"}


def blank(text):
    """[text] with its comments blanked out, newlines kept, so offsets hold."""
    return COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), text)


def matching(text, i):
    """The index of the brace that closes the one at [i]."""
    depth = 0
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return j
    sys.exit("unbalanced braces")


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
    stack, out, pending = [], [], ""
    for line in b.split("\n"):
        live = all(taken for taken, _ in stack)
        out.append(live)
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        line, pending = pending + line, ""
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
    for m in DEFINE.finditer(b):
        params = [p.strip() for p in m.group(2)[1:-1].split(",")] if m.group(2) else None
        body = m.group(3).replace("\\\n", " ").strip()
        out.append(Item("define", [m.group(1)], m.start(), m.end(), params=params, body=body))
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
                out += constants(it)
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
                out += constants(it)
    for m in PRAGMA.finditer(b):
        out.append(Item("pragma", [], m.start(), m.end(), packed=m.group(1).strip() == "1"))
    return [it for it in out if lines[b.count("\n", 0, it.start)]]


def constants(enum):
    """The constants of [enum], each an item over the enum's span."""
    entries = [e.split("=", 1) for e in (e.strip() for e in enum.body.split(",")) if e]
    enum.entries = [(e[0].strip(), e[1].strip() if len(e) > 1 else None) for e in entries]
    return [Item("constant", [n], enum.start, enum.end, enum=enum) for n, _ in enum.entries]


class Model:
    """The definitions of a release's headers, by name, from the first header
    that defines each."""

    def __init__(self, texts):
        self.texts, self.where, self.layouts, self.pragmas = texts, {}, {}, {}
        for h, text in texts.items():
            for it in items(text):
                it.header = h
                if it.kind == "pragma":
                    self.pragmas.setdefault(h, []).append(it)
                    continue
                for n in it.names:
                    if n in self.where and self.where[n].header == h:
                        old = self.where[n]
                        if text[old.start:old.end] != text[it.start:it.end] and it.kind == "define":
                            sys.exit(f"{h}: {n} is defined twice")
                    old = self.where.setdefault(n, it)
                    # [typedef struct X X;] names the struct defined by its tag.
                    if old.kind == "alias" and old.target == n and it.kind in ("struct", "union"):
                        self.where[n] = it

    def item(self, name):
        if name not in self.where:
            sys.exit(f"no definition of {name}")
        return self.where[name]

    # Values

    def expand(self, expr, depth=0):
        if depth > 64:
            sys.exit(f"recursive macro in {expr}")
        expr = re.sub(r"'(\\?.)'", lambda m: str(ord(m.group(1)[-1])), expr)
        out, i = [], 0
        for m in IDENT.finditer(expr):
            out.append(expr[i:m.start()])
            i = m.end()
            it = self.where.get(m.group())
            if not it or it.kind != "define":
                out.append(m.group())
            elif it.params is None:
                out.append("(" + self.expand(it.body, depth + 1) + ")")
            else:
                j = expr.index("(", m.end())
                close, level = j, 0
                for close in range(j, len(expr)):
                    level += {"(": 1, ")": -1}.get(expr[close], 0)
                    if level == 0:
                        break
                args = [a.strip() for a in expr[j + 1:close].split(",")]
                body = it.body
                for p, a in zip(it.params, args):
                    body = re.sub(rf"\b{p}\b", f"({a})", body)
                out.append("(" + self.expand(body, depth + 1) + ")")
                i = close + 1
        out.append(expr[i:])
        return "".join(out)

    def value(self, name):
        it = self.item(name)
        if it.kind == "constant":
            v = -1
            for n, expr in it.enum.entries:
                v = self.evaluate(expr, n) if expr else v + 1
                if n == name:
                    return v
        return self.evaluate(it.body, name)

    def evaluate(self, expr, name):
        e = self.expand(expr)
        e = re.sub(r"\(\s*(?:const\s+)?(?:Nv[US]\d+|NvV32|NvLength|unsigned(?:\s+(?:long\s+long|long|int|char))?|"
                   r"int|long)\s*\)", "", e)
        e = re.sub(r"\b(0[xX][0-9a-fA-F]+|\d+)[uUlL]+\b", r"\1", e)
        if re.search(r"[A-Za-z_][A-Za-z_0-9]*", re.sub(r"\b0[xX][0-9a-fA-F]+\b", "", e)):
            sys.exit(f"{name}: not a constant: {e}")
        return eval(e.replace("/", "//"), {"__builtins__": {}})

    def bits(self, name):
        parts = self.item(name).body.split(":")
        if len(parts) != 2 or not all(re.fullmatch(r"[\d()+\-* ]+", p) for p in parts):
            sys.exit(f"{name} is not a bit field")
        hi, lo = (eval(p, {"__builtins__": {}}) for p in parts)
        return (lo, hi - lo + 1)

    # Layouts

    def scalar_or_type(self, name):
        if name in SCALARS:
            return SCALARS[name][0], SCALARS[name][1], {}
        return self.layout(name)

    def layout(self, name):
        """(bytes, alignment, fields) of the type [name]: fields by path, an
        array's as (offset, bytes of an element, elements), others as
        (offset, bytes)."""
        if name in self.layouts:
            return self.layouts[name]
        it = self.item(name)
        if it.kind == "alias":
            r = self.scalar_or_type(it.target)
        elif it.kind == "enum":
            r = (4, 4, {})
        else:
            r = self.aggregate(it.kind, it.body, name, packed=self.packed(it))
        self.layouts[name] = r
        return r

    def members(self, body, where):
        """The members of a struct's body: (name, type, counts, alignment),
        type a name or an inline (kind, body)."""
        out, i, level, start = [], 0, 0, 0
        for i, c in enumerate(body):
            level += {"{": 1, "}": -1}.get(c, 0)
            if c == ";" and level == 0:
                out.append(self.member(body[start:i].strip(), where))
                start = i + 1
        if body[start:].strip():
            sys.exit(f"{where}: trailing {body[start:].strip()}")
        return [m for m in out if m]

    def member(self, s, where):
        if not s:
            return None
        align = 1
        m = re.fullmatch(r"NV_DECLARE_ALIGNED\((.*),\s*(\d+)\s*\)", s, re.S)
        if m:
            s, align = m.group(1).strip(), int(m.group(2))
        m = re.fullmatch(r"(.*?)\s*NV_ALIGN_BYTES\((\d+)\)", s, re.S)
        if m:
            s, align = m.group(1).strip(), max(align, int(m.group(2)))
        m = re.fullmatch(r"(?:volatile\s+)?(struct|union)\s*\w*\s*(\{.*\})\s*(\w*)\s*((?:\[[^\]]*\]\s*)*)", s, re.S)
        if m:
            ty = (m.group(1), m.group(2)[1:-1])
            name, dims = m.group(3), m.group(4)
        else:
            if ":" in s:
                sys.exit(f"{where}: a bit field: {s}")
            m = re.fullmatch(r"((?:\w+\s+)*?\w+)\s*(\*?)\s*(\w+)\s*((?:\[[^\]]*\]\s*)*)", s, re.S)
            if not m:
                sys.exit(f"{where}: cannot read {s!r}")
            words = [w for w in m.group(1).split() if w not in KEYWORDS - {"unsigned"}]
            ty = "NvP64" if m.group(2) else " ".join(words)
            ty = {"unsigned int": "int", "unsigned char": "char", "unsigned long": "long"}.get(ty, ty)
            name, dims = m.group(3), m.group(4)
        counts = [self.count(d, where) for d in re.findall(r"\[([^\]]*)\]", dims)]
        return (name, ty, counts, align)

    def count(self, expr, where):
        if not expr.strip():
            return 0
        e = self.expand(expr)
        e = re.sub(r"\b(0[xX][0-9a-fA-F]+|\d+)[uUlL]+\b", r"\1", e)
        if re.search(r"[A-Za-z_]", re.sub(r"\b0[xX][0-9a-fA-F]+\b", "", e)):
            sys.exit(f"{where}: not a count: {expr}")
        return eval(e.replace("/", "//"), {"__builtins__": {}})

    def packed(self, it):
        """Whether [it] lies under a [#pragma pack(1)] of its header, which
        lays its members out without padding."""
        before = [p for p in self.pragmas.get(it.header, []) if p.start < it.start]
        return bool(before) and before[-1].packed

    def aggregate(self, kind, body, where, packed=False):
        offset, align, fields = 0, 1, {}
        for name, ty, counts, falign in self.members(body, where):
            if isinstance(ty, tuple):
                size, a, sub = self.aggregate(ty[0], ty[1], where, packed)
            elif ty in FLEXIBLE and counts == [0]:
                if offset % 8:
                    sys.exit(f"{where}: {name} is not at a multiple of 8")
                size, a, sub = 0, 1, {}
            else:
                size, a, sub = self.scalar_or_type(ty)
            a = 1 if packed else max(a, falign)
            n = 1
            for c in counts:
                n *= c
            at = 0 if kind == "union" else (offset + a - 1) // a * a
            path = name
            if counts:
                fields[path] = (at, size, n)
                fields.update({f"{path}__{k}": v for k, v in sub.items()})
            else:
                prefix = f"{path}__" if path else ""
                if path:
                    fields[path] = (at, size)
                fields.update({prefix + k: (v[0] + at, *v[1:]) for k, v in sub.items()})
            offset = max(offset, at + size * n) if kind == "union" else at + size * n
            align = max(align, a)
        return (offset + align - 1) // align * align, align, fields

    # Excerpts

    def deps(self, name):
        """The names [name]'s definition refers to."""
        it = self.item(name)
        if it.kind == "define":
            return set(IDENT.findall(it.body))
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

    def spans(self, names):
        """The spans of the definitions [names] need, by header, with the
        pragmas that pack them."""
        return {h: {(it.start, it.end) for it in its.values()} | {(p.start, p.end) for p in self.pragmas.get(h, [])}
                for h, its in self.closure(names).items()}


def table_names(model, header, pattern):
    """The defines of [header] whose names [pattern] matches and whose values
    are numbers."""
    return {n for n, it in model.where.items()
            if it.header == header and it.kind == "define" and it.params is None and re.fullmatch(pattern, n)
            and re.fullmatch(r"\(?\s*(0[xX][0-9a-fA-F]+|\d+)[uUlL]*\s*\)?", it.body)}


def table(model, header, pattern):
    """The entries (value, name) of [header]'s table of the defines [pattern]
    matches, by value."""
    return tuple(sorted((model.value(n), re.fullmatch(pattern, n).group(1))
                        for n in table_names(model, header, pattern)))


def notice(name, text):
    """The file's first comment, its licence notice, which must be the MIT
    licence."""
    end = text.index("*/") + 2
    if not re.search(r"Permission is hereby granted|SPDX-License-Identifier: MIT\b", text[:end]):
        sys.exit(f"{name}: its notice is not the MIT licence")
    return text[:end] + "\n"


def cut(name, text, spans):
    """The excerpt [name] of [text]: its licence notice, its URL and the
    spans [spans], in order, a span that starts inside the one before it left
    out."""
    body = []
    for s, e in sorted(spans):
        if body and s < body[-1][1]:
            continue
        body.append((s, e))
    return (notice(name, text) + f"\n/* Excerpt of {SOURCES[name]}. */\n\n"
            + "\n\n".join(text[s:e] for s, e in body) + "\n")


def whole(url, text):
    """[text], kept whole, under a notice if it carries none: the repository's
    COPYING licenses each of its files that notes nothing else under the MIT
    licence."""
    if "Permission is hereby granted" in text:
        return text
    return (f"/*\n * Copied whole from\n * {url},\n"
            " * which carries no notice. The repository's COPYING: \"Except where noted\n"
            " * otherwise, the individual files within this package are licensed as MIT:\"\n *\n"
            " * Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n *\n"
            + "".join(f" * {l}".rstrip() + "\n" for l in MIT.splitlines()) + " */\n\n" + text)


def release_texts(d, r, texts, every=False):
    """{header: text} of library [d]'s headers in release [r] among [texts],
    by excerpt; [every] fails on a missing one."""
    out = {}
    for h, url in d.headers(r).items():
        k = key(h, url)
        if k in texts:
            out[h] = texts[k]
        elif every:
            sys.exit(f"{k}: no excerpt")
    return out


def release_spans(d, r, texts):
    """{header: spans} of release [r]'s [texts] that library [d] reads: all of
    it from the newest release, only what differs between releases from an
    older one; and the status lines of STATUS_HEADER when [d] reads statuses."""
    model = Model(texts)
    if r == NEWEST:
        names = set(d.constants) | set(d.fields) | set(d.structs)
        tables = d.tables
    else:
        names = set(d.per_release)
        tables = [t for t in d.tables if t[0] in d.tables_per_release]
    for _, h, pattern in tables:
        names |= table_names(model, h, pattern)
    spans = model.spans(names)
    if d.statuses:
        lines = active(blank(texts[STATUS_HEADER]))
        spans.setdefault(STATUS_HEADER, set()).update(
            (m.start(), m.end()) for m in STATUS.finditer(texts[STATUS_HEADER])
            if lines[texts[STATUS_HEADER].count("\n", 0, m.start())])
    if r == NEWEST:
        bare = [h for h in texts if h not in spans]
        if bare:
            sys.exit(f"{bare}: nothing read from them")
    return spans


def register_names(model, regs):
    """The registers [regs], their bit fields and the named values, as the
    names of their defines in [model]."""
    names = set()
    for r in regs:
        names.add(r)
        names |= {n for n, it in model.where.items()
                  if n.startswith(r + "_") and it.kind == "define" and it.params is None and ":" in it.body}
    return names


def mmu_names(model, prefix):
    return {n for n, it in model.where.items()
            if any(n.startswith(f"{prefix}_{e}_") for e in MMU_ENTRIES) and it.kind == "define"
            and it.params is None and ":" in it.body}


def gsp_wanted(model):
    """The names the GSP's formats read."""
    names = set(GSP_CONSTANTS) | set(GSP_FIELDS) | set(GSP_STRUCTS) | {MMU_KIND}
    for _, prefix in MMU.values():
        names |= mmu_names(model, prefix) | {f"{prefix}_{v}" for v in MMU_VALUES}
    names |= {f"{MMU['V3'][1]}_{v}" for v in MMU_V3_VALUES}
    for _, h, pattern in GSP_TABLES:
        names |= table_names(model, h, pattern)
    return names


def pci_wanted(model):
    """The names read from the files other than the register headers."""
    names = set(PCI_CONSTANTS) | set(PCI_FIELDS) | set(PCI_STRUCTS) | gsp_wanted(model)
    return names | {c for c, _, _ in ROM_STRUCTS} | {f for _, _, f in ROM_STRUCTS}


def family_wanted(family):
    """The names read from the register headers of [family]."""
    def names(model):
        out = register_names(model, REGISTERS[family]) | {"NV_VIRTUAL_FUNCTION_FULL_PHYS_OFFSET"}
        out |= {u for u in UNITS if u in model.where}
        return out | {v for v in REGISTER_VALUES if v in model.where}
    return names


def pci_spans(texts):
    """{file: spans} of the files [texts] that rig_nv_pci reads. Each family's
    register headers are read as that family reads them, the first that
    defines a name giving it."""
    others = [h for h in PCI_SOURCES if h not in REGISTER_SOURCES]
    groups = [(others, pci_wanted)] + [(hs, family_wanted(f)) for f, hs in FAMILIES.items()]
    keep = {}
    for hs, want in groups:
        model = Model({h: texts[h] for h in hs})
        for h, spans in model.spans(want(model)).items():
            keep.setdefault(h, set()).update(spans)
    bare = [h for h in PCI_SOURCES if h not in keep]
    if bare:
        sys.exit(f"{bare}: nothing read from them")
    return keep


def abi_wanted():
    """The names read from each header of rig_nv_abi, by header, an indexed
    name without its (i)."""
    w = {h: set() for h in ABI_SOURCES}
    for h, n in CLASS_IDS:
        w[h].add(n)
    w[HOST] |= set(HOST_CONSTANTS + HOST_FIELDS)
    w[COMPUTE] |= set(COMPUTE_CONSTANTS + COMPUTE_FIELDS)
    w[COPY] |= set(COPY_CONSTANTS + COPY_FIELDS)
    w[LATER_COPY] |= {later(n) for n in COPY_CONSTANTS + COPY_FIELDS}
    for h, names in RUNS:
        w[h] |= set(names)
        if h == COPY:
            w[LATER_COPY] |= {later(n) for n in names}
    for v, (h, prefix, _) in enumerate(QMD):
        w[h] |= {f"{prefix}_{n}" for n in qmd_names(v)}
        if v == 0:
            w[QMD_SAME[0]] |= {f"{QMD_SAME[1]}_{n}" for n in qmd_names(v)}
    return {h: {n.removesuffix("(i)") for n in names} for h, names in w.items()}


def abi_spans(texts):
    """{header: spans} of the headers [texts] that rig_nv_abi reads, with the
    field that ends each launch descriptor."""
    out = {}
    for h, names in abi_wanted().items():
        model = Model({h: texts[h]})
        for q, prefix, _ in QMD + [(*QMD_SAME, 3)]:
            if q == h:
                names = names | {last_field(model, prefix)}
        out.update(model.spans(names))
    return out


def excerpts(texts):
    """{excerpt: text} of the upstream files [texts], by excerpt: one of each
    file, holding what every library reads of it."""
    spans = {}

    def add(found, urls):
        for h, s in found.items():
            spans.setdefault(key(h, urls[h]), set()).update(s)

    for d in LIBRARIES:
        for r in RELEASES:
            urls = d.headers(r)
            add(release_spans(d, r, release_texts(d, r, texts, every=True)), urls)
    add(pci_spans({h: texts[key(h, u)] for h, u in PCI_SOURCES.items()}), PCI_SOURCES)
    add(abi_spans({h: texts[key(h, u)] for h, u in ABI_SOURCES.items()}), ABI_SOURCES)
    out = {k: cut(k, texts[k], s) for k, s in spans.items()}
    out.update({key(h, u): whole(u, texts[key(h, u)]) for h, u in WHOLE.items()})
    return out


# Generation

ML_KEYWORDS = {"type", "function", "method", "val", "open", "end", "include", "module", "object", "private"}


def snake(name):
    """The OCaml value for the C name [name]: NV01_ROOT is nv01_root and
    hObjectParent h_object_parent; nested fields a__b become a_b."""
    if name.upper() == name:
        s = name.lower()
    else:
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name.replace("__", "_"))
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s).lower()
    return s + "_" if s in ML_KEYWORDS else s


def ml_int(v):
    """[v] as an OCaml literal: an int64 one past OCaml's 63-bit ints."""
    if v > 2**62 - 1:
        return f"0x{v:x}L"
    return f"0x{v:x}" if v > 9 else str(v)


def ml_tuple(v):
    return "(" + ", ".join(ml_int(x) for x in v) + ")"


def emit_struct(out, module, layout, indent=""):
    """Emits [module], of [layout]: (bytes, [(field, its layout, or None if a
    field "a?" is missing)])."""
    size, fields = layout
    out.append(f"{indent}module {module} = struct")
    out.append(f"{indent}  let sizeof = {size}")
    for f, v in fields:
        if f.endswith("?"):
            value = f"Some {ml_tuple(v)}" if v else "None"
        else:
            value = ml_tuple(v)
        out.append(f"{indent}  let {snake(f.rstrip('?'))} = {value}")
    out.append(f"{indent}end")


def emit_table(out, name, entries, indent=""):
    out.append(f"{indent}let {name} = [")
    for v, n in entries:
        out.append(f"{indent}  ({ml_int(v)}, {json.dumps(n)});")
    out.append(f"{indent}]")


def joined(items):
    """The lines of [items], a blank line between two."""
    out = []
    for item in items:
        if out:
            out.append("")
        out += item
    return out


def header(out, texts, credit):
    """The comment that opens the file [out]: the script that generates it,
    [credit], and the owners the notices of the files [texts] name."""
    owners = sorted({m.group(1).strip() for t in texts
                     for m in re.finditer(r"Copyright \(c\) ([^\n]*?NVIDIA[^\n.]*)", t[:t.index("*/")], re.I)})
    script = os.path.relpath(HERE / "gen.py", out.parent)
    excerpts = os.path.relpath(HEADERS, out.parent)
    notice = "\n".join(f"   {l}".rstrip() for l in MIT.splitlines())
    return (
        "(*---------------------------------------------------------------------------\n"
        "  Copyright (c) 2026 The Raven authors. All rights reserved.\n"
        "  SPDX-License-Identifier: ISC\n"
        "  ---------------------------------------------------------------------------*)\n\n"
        f"(* Generated by {script} from the excerpts in {excerpts}; do not\n"
        "   edit. Regenerate it from the repository root with\n"
        f"   [{COMMAND}].\n\n"
        + credit + "\n\n"
        + "\n".join(f"   Copyright (c) {o}" for o in owners) + "\n\n" + notice + " *)\n\n"
    )


# The credit of each defs.ml, before the owners of the files it copies from.
NVIDIA_CREDIT = "   The values are NVIDIA's, copied from its headers under the MIT licence:"
PCI_CREDIT = ("   The values are NVIDIA's, copied under the MIT licence from its headers and\n"
              "   from the headers of Linux's nouveau driver (SPDX-License-Identifier: MIT):")


def struct(d, model, cname, where):
    """(bytes, [(field, its layout, or None if the struct lacks it)])."""
    size, _, fields = model.layout(cname)
    module, names = d.structs[cname]
    out = []
    for f in names:
        name = f.rstrip("?")
        if name not in fields and not f.endswith("?"):
            sys.exit(f"{where}: {cname} has no field {name}; it has {sorted(fields)}")
        out.append((f, fields.get(name)))
    return size, out


def release_tables(d, models):
    """{name: {release: entries}} of [d]'s tables that differ between
    releases, and the statuses, from each release's [models]."""
    tables = {name: {r: table(m, h, pattern) for r, m in models.items()}
              for name, h, pattern in d.tables if name in d.tables_per_release}
    if d.statuses:
        tables["statuses"] = {r: tuple((int(v, 16), n) for n, v in STATUS.findall(m.texts[STATUS_HEADER]))
                              for r, m in models.items()}
    return tables


def check_releases(d, models):
    """Fails unless what [d]'s defs.ml takes once is the same in every
    release's whole [models], and the structs and tables that differ between
    them are exactly those it takes per release."""
    def same(what, values):
        if len(set(values.values())) != 1:
            sys.exit(f"{what} differs between releases: {values}")

    for c in d.constants:
        same(c, {r: m.value(c) for r, m in models.items()})
    if d.statuses:
        for name in d.statuses:
            same(name, {r: int(dict(STATUS.findall(m.texts[STATUS_HEADER]))[name], 16) for r, m in models.items()})
    for f in d.fields:
        same(f, {r: m.bits(f) for r, m in models.items()})
    layouts = {c: {r: repr(struct(d, m, c, r)) for r, m in models.items()} for c in d.structs}
    differ = {c for c, per in layouts.items() if len(set(per.values())) != 1}
    if differ != d.per_release:
        sys.exit(f"the structs that differ between releases are {sorted(differ)}, not {sorted(d.per_release)}")
    tables = {name: {r: table(m, h, pattern) for r, m in models.items()} for name, h, pattern in d.tables}
    differ = {name for name, per in tables.items() if len(set(per.values())) != 1}
    if differ != d.tables_per_release:
        sys.exit(f"the tables that differ between releases are {sorted(differ)}, "
                 f"not {sorted(d.tables_per_release)}")


def generate(d, texts):
    """[d]'s defs.ml, from [texts] by excerpt: what it takes once from the
    newest release, what differs from each release."""
    models = {r: Model(release_texts(d, r, texts, every=r == NEWEST)) for r in RELEASES}
    newest = models[NEWEST]
    out = []
    out.append("(* The releases whose layouts are below: "
               + ", ".join(v for _, v in RELEASES.values()) + ". *)")
    out.append("let releases = [ " + "; ".join(str(r) for r in RELEASES) + " ]")
    out.append("")
    out.append("(* Constants, the same in every release. *)")
    for c in d.constants:
        out.append(f"let {snake(c)} = {ml_int(newest.value(c))}")
    out.append("")
    if d.statuses:
        named = dict(STATUS.findall(newest.texts[STATUS_HEADER]))
        out.append("(* Statuses, the same in every release. *)")
        for name in d.statuses:
            out.append(f"let {snake(name)} = {ml_int(int(named[name], 16))}")
        out.append("")
    out.append("(* Bit fields of their words: (lowest bit, bits). *)")
    for f in d.fields:
        out.append(f"let {snake(f)} = {ml_tuple(newest.bits(f))}")
    out.append("")

    out.append("(* Structs: each field is (byte offset, bytes), an array's (offset, bytes")
    out.append("   of an element, elements). *)")
    for c, (module, _) in d.structs.items():
        if c not in d.per_release:
            emit_struct(out, module, struct(d, newest, c, NEWEST))
            out.append("")
    for name, h, pattern in d.tables:
        if name not in d.tables_per_release:
            out.append(f"(* {h}'s {name.replace('_', ' ')}, by value. *)")
            emit_table(out, name, table(newest, h, pattern))
            out.append("")

    # The signature and each release's module: one item per struct and
    # table that differs, a blank line between two.
    layouts = {c: {r: struct(d, m, c, r) for r, m in models.items()} for c in d.per_release}
    tables = release_tables(d, models)
    sig = []
    for c, (module, _) in d.structs.items():
        if c in d.per_release:
            item = [f"  module {module} : sig", "    val sizeof : int"]
            arities = {f: len(v) for r in RELEASES for f, v in layouts[c][r][1] if v}
            for f, _ in layouts[c][min(RELEASES)][1]:
                ty = " * ".join(["int"] * arities[f])
                item.append(f"    val {snake(f.rstrip('?'))} : " + (f"({ty}) option" if f.endswith("?") else ty))
            sig.append(item + ["  end"])
    for name, h, _ in d.tables:
        if name in d.tables_per_release:
            sig.append([f"  (* {h}'s {name.replace('_', ' ')}, by value. *)", f"  val {name} : (int * string) list"])
    if d.statuses:
        sig.append([f"  (* {STATUS_HEADER}'s statuses: (value, name). *)", "  val statuses : (int * string) list"])
    out.append("(* What differs between the releases. *)")
    out.append("module type RELEASE = sig")
    out += joined(sig)
    out.append("end")
    out.append("")
    for r in RELEASES:
        body = []
        for c, (module, _) in d.structs.items():
            if c in d.per_release:
                item = []
                emit_struct(item, module, layouts[c][r], indent="  ")
                body.append(item)
        for name in tables:
            item = []
            emit_table(item, name, tables[name][r], indent="  ")
            body.append(item)
        out.append(f"module R{r} : RELEASE = struct")
        out += joined(body)
        out.append("end")
        out.append("")
    out.append("(* The layouts of release [r], if it is one of {!releases}. *)")
    out.append("let release = function")
    for r in RELEASES:
        out.append(f"  | {r} -> Some (module R{r} : RELEASE)")
    out.append("  | _ -> None")
    read = [t for m in models.values() for t in m.texts.values()]
    return header(d.out, read, NVIDIA_CREDIT) + "\n".join(out) + "\n"


# Registers


def registers(model, regs):
    """The registers [regs] of a family's [model]: name -> (address, stride
    of an indexed register or None, fields as name -> (lowest bit, bits))."""
    out = {}
    for r in regs:
        it = model.item(r)
        base = next((b for p, b in UNIT_BASES.items() if r.startswith(p + "_")), 0)
        if it.params:
            if len(it.params) != 1:
                sys.exit(f"{r}: more than one index")
            at = [model.evaluate(re.sub(rf"\b{it.params[0]}\b", str(i), it.body), r) for i in (0, 1)]
            address, stride = base + at[0], at[1] - at[0]
        else:
            address, stride = base + model.value(r), None
        fields = {n[len(r) + 1:]: model.bits(n) for n in sorted(register_names(model, [r]) - {r})}
        out[r] = (address, stride, fields)
    return out


def emit_register(out, name, reg, seen, indent=""):
    """Emits the register [name] and those of its fields not in [seen]: a
    field of a register may also be named as a field of a register whose name
    it extends, such as CPUCTL_ALIAS_EN."""
    address, stride, fields = reg
    if stride is None:
        out.append(f"{indent}let {snake(name)} = {ml_int(address)}")
    else:
        out.append(f"{indent}let {snake(name)} i = {ml_int(address)} + (i * {ml_int(stride)})")
    for f, v in fields.items():
        n = snake(name + "_" + f)
        if n not in seen:
            seen.add(n)
            out.append(f"{indent}let {n} = {ml_tuple(v)}")


def register_sig(out, name, regs):
    """The signature of the register [name] that differs between the
    families' [regs]: its address and the fields every family has."""
    strides = {r[1] is None for r in regs}
    if len(strides) != 1:
        sys.exit(f"{name} is indexed in one family alone")
    out.append(f"  val {snake(name)} : {'int' if strides.pop() else 'int -> int'}")
    for f in sorted(set.intersection(*(set(r[2]) for r in regs))):
        out.append(f"  val {snake(name + '_' + f)} : int * int")


# The VBIOS's structures


def rom_layout(fields, fmt):
    """(bytes, fields at their offsets) of [fields], in order, packed as the
    format string [fmt] says: counts of b (byte), w (word), d (double word)
    and q (quad word)."""
    sizes = []
    for n, k in re.findall(r"(\d+)([bwdq])", fmt):
        sizes += [{"b": 1, "w": 2, "d": 4, "q": 8}[k]] * int(n)
    if len(sizes) != len(fields):
        sys.exit(f"format {fmt} has {len(sizes)} items for {len(fields)} fields")
    out, off = [], 0
    for f, s in zip(fields, sizes):
        out.append((f, (off, s)))
        off += s
    return off, out


def rom_fields(model, cname):
    it = model.item(cname)
    while it.kind == "alias":
        it = model.item(it.target)
    return [name for name, _, _, _ in model.members(it.body, cname)]


# Function and event numbers


def rpc_numbers(text):
    """The functions and events of rpc_global_enums.h, by name."""
    functions = {m.group(1): int(m.group(2), 0)
                 for m in re.finditer(r"^\s*X\(\s*\w+\s*,\s*(\w+)\s*,\s*(\w+)\s*\)", text, re.M)}
    events = {m.group(1): int(m.group(2), 0)
              for m in re.finditer(r"^\s*E\(\s*(\w+)\s*,\s*(\w+)\s*\)", text, re.M)}
    return functions, events


def pci_texts(texts):
    """{file: text} of rig_nv_pci's files among [texts], by excerpt."""
    return {h: texts[key(h, u)] for h, u in [*PCI_SOURCES.items(), *WHOLE.items()]}


def pci_model(texts):
    return Model({h: texts[h] for h in PCI_SOURCES if h not in REGISTER_SOURCES})


def emit_gsp(out, model, texts, constants, fields, functions, events, structs):
    """Emits [constants], the bit [fields], the numbers of the GSP's
    [functions] and [events], and the [structs] of [model]."""
    out.append("(* Constants. *)")
    for c in constants:
        out.append(f"let {snake(c)} = {ml_int(model.value(c))}")
    out.append("")
    out.append("(* Bit fields of 32-bit words: (lowest bit, bits). *)")
    for f in fields:
        out.append(f"let {snake(f)} = {ml_tuple(model.bits(f))}")
    out.append("")

    numbers = rpc_numbers(texts["rpc_global_enums.h"])
    out.append("(* The GSP's functions and events (rpc_global_enums.h). *)")
    for f in functions:
        out.append(f"let nv_vgpu_msg_function_{f.lower()} = {ml_int(numbers[0][f])}")
    for e in events:
        out.append(f"let nv_vgpu_msg_event_{e.lower()} = {ml_int(numbers[1][e])}")
    out.append("")

    out.append("(* Structs: each field is (byte offset, bytes), an array's (offset, bytes")
    out.append("   of an element, elements). *)")
    for c, (module, names) in structs.items():
        size, _, layout = model.layout(c)
        for f in names:
            if f not in layout:
                sys.exit(f"{c} has no field {f}; it has {sorted(layout)}")
        emit_struct(out, module, (size, [(f, layout[f]) for f in names]))
        out.append("")


def generate_gsp(texts):
    texts = pci_texts(texts)
    model = pci_model(texts)
    out = []
    emit_gsp(out, model, texts, GSP_CONSTANTS, GSP_FIELDS, GSP_FUNCTIONS, GSP_EVENTS, GSP_STRUCTS)

    for name, h, pattern in GSP_TABLES:
        out.append(f"(* {h}'s {name.replace('_', ' ')}, by value. *)")
        emit_table(out, name, table(model, h, pattern))
        out.append("")

    out.append("(* Page-table entries: fields as (lowest bit, bits) of the entry, which is")
    out.append("   16 bytes for a dual entry, and values. *)")
    out.append(f"let {snake(MMU_KIND)} = {ml_int(model.value(MMU_KIND))}")
    for _, prefix in MMU.values():
        for n in sorted(mmu_names(model, prefix)):
            out.append(f"let {snake(n)} = {ml_tuple(model.bits(n))}")
        values = MMU_VALUES + (MMU_V3_VALUES if prefix == MMU["V3"][1] else [])
        for v in values:
            out.append(f"let {snake(prefix + '_' + v)} = {ml_int(model.value(prefix + '_' + v))}")
    read = [texts[h] for h in model.closure(gsp_wanted(model))] + [texts["rpc_global_enums.h"]]
    return header(GSP_OUT, read, NVIDIA_CREDIT) + "\n".join(out) + "\n"


def generate_pci(texts):
    firmware = texts[FIRMWARE_LIST]
    texts = pci_texts(texts)
    model = pci_model(texts)
    out = []
    emit_gsp(out, model, texts, PCI_CONSTANTS, PCI_FIELDS, PCI_FUNCTIONS, PCI_EVENTS, PCI_STRUCTS)

    out.append("(* The VBIOS's structures, packed: each field is (byte offset, bytes). *)")
    for c, module, fmt in ROM_STRUCTS:
        fmt_text = model.item(fmt).body.strip().strip('"')
        size, fields = rom_layout(rom_fields(model, c), fmt_text)
        emit_struct(out, module, (size, fields))
        out.append("")

    families = {f: registers(Model({h: texts[h] for h in hs}), REGISTERS[f]) for f, hs in FAMILIES.items()}
    vf = Model({"tu102_dev_vm.h": texts["tu102_dev_vm.h"]}).item("NV_VIRTUAL_FUNCTION_FULL_PHYS_OFFSET").body
    if int(vf.split(":")[1].split()[0], 16) != UNIT_BASES["NV_VIRTUAL_FUNCTION"]:
        sys.exit(f"NV_VIRTUAL_FUNCTION_FULL_PHYS_OFFSET is {vf}")
    legacy, blackwell = families["Legacy"], families["Blackwell"]
    common = [r for r in legacy if r in blackwell and legacy[r] == blackwell[r]]
    differ = [r for r in legacy if r in blackwell and legacy[r] != blackwell[r]]
    out.append("(* Registers: each is its address in BAR 0, or a function of its index, and")
    out.append("   each of its fields (lowest bit, bits). Those of both families, then what")
    out.append("   differs: Ampere and Ada's, and Blackwell's. *)")
    seen = set()
    for r in common:
        emit_register(out, r, legacy[r], seen)
    both = Model({h: texts[h] for h in dict.fromkeys(FAMILIES["Legacy"] + FAMILIES["Blackwell"])})
    for v in REGISTER_VALUES:
        out.append(f"let {snake(v)} = {ml_int(both.value(v))}")
    for u in UNITS:
        out.append(f"let {snake(u)} = {ml_int(int(both.item(u).body.split(':')[1].split()[0], 16))}")
    out.append("")
    out.append("(* The registers whose address or fields differ between the families. *)")
    out.append("module type FAMILY = sig")
    for r in differ:
        register_sig(out, r, [legacy[r], blackwell[r]])
    out.append("end")
    out.append("")
    for f, regs in families.items():
        out.append(f"module {f} = struct")
        own = set(seen)
        for r, reg in regs.items():
            if r not in common:
                emit_register(out, r, reg, own, indent="  ")
        out.append("end")
        out.append("")

    digests = {}
    for row in firmware.splitlines():
        if row.startswith("#"):
            continue
        path, digest, url = row.split("\t")
        if url != FIRMWARE_ORIGIN + path:
            sys.exit(f"{path}: URL {url} is not in the pinned tree")
        digests[path] = digest
    want = {p for ps in FIRMWARE.values() for p in ps}
    if set(digests) != want:
        sys.exit(f"firmware.tsv: extra {sorted(set(digests) - want)}, missing {sorted(want - set(digests))}")
    out.append("(* The firmware images of each family, GSP first, and every image's")
    out.append("   BLAKE2b-256 digest. *)")
    for f, paths in FIRMWARE.items():
        out.append(f"let firmware_{f.lower()} = [ " + "; ".join(json.dumps(p) for p in paths) + " ]")
    out.append("")
    out.append("let pinned = [")
    for p in sorted({p for ps in FIRMWARE.values() for p in ps}):
        out.append(f"  ({json.dumps(p)}, {json.dumps(digests[p])});")
    out.append("]")
    return header(PCI_OUT, texts.values(), PCI_CREDIT) + "\n".join(out) + "\n"


# Classes and launch descriptors


def later(name):
    return name.replace("NVC7B5_", "NVC9B5_")


def qmd_names(v):
    """The names version [v] (0 for 3, 1 for 5) reads, without its prefix."""
    names = [f[1 + v] for f in QMD_FIELDS + QMD_BANK_FIELDS + QMD_VALUES if f[1 + v]]
    for f in QMD_RELEASE_FIELDS:
        names += [f[1 + v].format(k=k) for k in range(QMD_RELEASES)]
    return names


def mw(body, i=None):
    """A descriptor field "MW(hi:lo)", whose bounds may use i, as (lowest bit, bits)."""
    m = re.fullmatch(r"MW\((.+):(.+)\)", body.replace(" ", ""))
    if not m or not re.fullmatch(r"[\d()+*i]+", m.group(1) + m.group(2)):
        sys.exit(f"not a descriptor field: {body}")
    env = {"__builtins__": {}, "i": i}
    hi, lo = eval(m.group(1), env), eval(m.group(2), env)
    return (lo, hi - lo + 1)


def shift(name):
    m = re.search(r"_SHIFTED(\d+)(\(i\))?$", name)
    return int(m.group(1)) if m else 0


def last_field(model, prefix):
    """The name of [prefix]'s field that ends last, which sets the descriptor's size."""
    fields = [(mw(it.body)[0] + mw(it.body)[1], n) for n, it in model.where.items()
              if n.startswith(prefix + "_") and it.kind == "define" and it.params is None
              and it.body.startswith("MW(")]
    return max(fields)[1]


def ml_field(f):
    return f"{{ lo = {ml_int(f[0])}; bits = {ml_int(f[1])} }}"


def ml_option(f):
    return f"Some {ml_field(f)}" if f else "None"


def generate_abi(texts):
    texts = {h: texts[key(h, u)] for h, u in ABI_SOURCES.items()}
    models = {h: Model({h: t}) for h, t in texts.items()}

    def body(h, n):
        """The body of [h]'s macro [n], an indexed one named with its (i)."""
        return models[h].item(n.removesuffix("(i)")).body

    out = []

    def let(name, value):
        out.append(f"let {name.lower()} = {value}")

    out.append(qmd_types())
    out.append("\n(* Classes *)\n")
    for h, n in CLASS_IDS:
        let(n, ml_int(models[h].value(n)))

    for h, names in RUNS:
        first = models[h].value(names[0])
        for k, n in enumerate(names):
            if models[h].value(n) != first + 4 * k:
                sys.exit(f"{h}: {n} does not follow {names[k - 1]}")

    for name in COPY_CONSTANTS + COPY_FIELDS + [n for h, ns in RUNS if h == COPY for n in ns]:
        if body(COPY, name) != body(LATER_COPY, later(name)):
            sys.exit(f"{name} differs in {LATER_COPY}")

    sections = [("Host methods (clc56f.h)", HOST, HOST_CONSTANTS, HOST_FIELDS),
                ("Compute methods (clc7c0.h)", COMPUTE, COMPUTE_CONSTANTS, COMPUTE_FIELDS),
                ("Copy methods (clc7b5.h)", COPY, COPY_CONSTANTS, COPY_FIELDS)]
    for title, h, constants, fields in sections:
        out.append(f"\n(* {title} *)\n")
        for n in constants:
            let(n, ml_int(models[h].value(n)))
        for n in fields:
            let(n, ml_field(models[h].bits(n)))

    out.append("\n(* Launch descriptors *)\n")
    same_h, same_prefix = QMD_SAME
    for n in qmd_names(0):
        if body(QMD[0][0], f"{QMD[0][1]}_{n}") != body(same_h, f"{same_prefix}_{n}"):
            sys.exit(f"{n} differs in {same_h}")
    for name, *versions in QMD_VALUES:
        values = {models[h].value(f"{prefix}_{n}") for (h, prefix, _), n in zip(QMD, versions) if n}
        if len(values) != 1:
            sys.exit(f"{name} differs between versions")
        let(name, ml_int(values.pop()))
    for v, (h, prefix, version) in enumerate(QMD):

        def field(n, i=None):
            return mw(body(h, f"{prefix}_{n}"), i)

        last = last_field(models[h], prefix)
        size = field(last[len(prefix) + 1:])
        out.append("")
        out.append(f"(* Version {version}, {prefix} ({h}). *)")
        out.append(f"let qmd_v{version} =")
        out.append("  {")
        out.append(f"    version = {version};")
        out.append(f"    bytes = {ml_int((size[0] + size[1]) // 8)};")
        for name, *names in QMD_FIELDS:
            n = names[v]
            if name in OPTIONAL:
                out.append(f"    {name} = {ml_option(field(n) if n else None)};")
            else:
                out.append(f"    {name} = {ml_field(field(n))};")
        for name, *names in QMD_BANK_FIELDS:
            n = names[v]
            first, second = field(n, 0), field(n, 1)
            out.append(f"    {name} = {{ first = {ml_field(first)}; stride = {second[0] - first[0]} }};")
        for name, *names in QMD_SHIFTS:
            out.append(f"    {name} = {shift(names[v])};")
        out.append("    releases =")
        for k in range(QMD_RELEASES):
            out.append(f"      {'(' if k == 0 else ','} {{")
            for f, *names in QMD_RELEASE_FIELDS:
                out.append(f"          {f} = {ml_field(field(names[v].format(k=k), k))};")
            out.append("        }")
        out.append("      );")
        out.append("  }")
    return header(ABI_OUT, texts.values(), NVIDIA_CREDIT) + "\n".join(out) + "\n"


def qmd_types():
    fields = ["  version : int;", "  bytes : int;"]
    for name, *_ in QMD_FIELDS:
        fields.append(f"  {name} : field{' option' if name in OPTIONAL else ''};")
    fields += [f"  {name} : banked;" for name, *_ in QMD_BANK_FIELDS]
    fields += [f"  {name} : int;" for name, *_ in QMD_SHIFTS]
    fields.append("  releases : release * release;")
    release = "\n".join(f"  {f} : field;" for f, *_ in QMD_RELEASE_FIELDS)
    return (
        "(* A bit field of a word or a descriptor: its lowest bit and its bits. *)\n"
        "type field = { lo : int; bits : int }\n\n"
        "(* A field for each constant bank: bank i's starts [i * stride] bits after\n"
        "   bank 0's. *)\n"
        "type banked = { first : field; stride : int }\n\n"
        "(* The fields of a descriptor's release. *)\n"
        f"type release = {{\n{release}\n}}\n\n"
        "(* A descriptor version's layout, by NVIDIA's names of its fields. Where the\n"
        "   versions' names differ by a suffix _SHIFTEDn, which says the field holds\n"
        "   its value shifted right by n, the name drops it and a [_shift] holds n. A\n"
        "   field the version lacks is [None]. [bytes] is the descriptor's size, the\n"
        "   end of its last field. *)\n"
        "type qmd = {\n" + "\n".join(fields) + "\n}"
    )


def body(text):
    """[text], a generated file, without its opening comment."""
    return text.split(" *)\n\n", 1)[1]


def generate_all(texts):
    """{file: text} of every defs.ml, from [texts] by excerpt."""
    files = {d.out: generate(d, texts) for d in LIBRARIES}
    files[PCI_OUT] = generate_pci(texts)
    files[GSP_OUT] = generate_gsp(texts)
    files[ABI_OUT] = generate_abi(texts)
    return files


def firmware(get):
    """firmware.tsv: each image's path, BLAKE2b-256 digest and URL."""
    rows = ["# path\tBLAKE2b-256\tURL, of linux-firmware at " + FIRMWARE_COMMIT]
    for p in sorted({p for ps in FIRMWARE.values() for p in ps}):
        rows.append(f"{p}\t{hashlib.blake2b(get(FIRMWARE_ORIGIN + p), digest_size=32).hexdigest()}\t{FIRMWARE_ORIGIN + p}")
    return "\n".join(rows) + "\n"


def download(url, cache, pins, pin):
    path = cache / hashlib.sha256(url.encode()).hexdigest()[:16]
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        # git.kernel.org refuses urllib's default User-Agent.
        req = urllib.request.Request(url, headers={"User-Agent": "raven-gen"})
        with urllib.request.urlopen(req, timeout=300) as r:
            path.write_bytes(r.read())
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if url not in pins:
        if not pin:
            sys.exit(f"{url} is not pinned; run with --pin to record {digest}")
        pins[url] = digest
    if pins[url] != digest:
        sys.exit(f"{url}: SHA-256 {digest}, pinned {pins[url]}")
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--check", action="store_true", help="fail if a committed file differs")
    p.add_argument("--excerpt", action="store_true", help="make the excerpts from the pinned files")
    p.add_argument("--pin", action="store_true", help="record the digests of files not yet pinned")
    p.add_argument("--cache", type=pathlib.Path, default=pathlib.Path.home() / ".cache/raven/nv-gen")
    a = p.parse_args()
    if a.excerpt:
        pins = json.loads(PINS.read_text()) if PINS.exists() else {}
        get = lambda url: download(url, a.cache, pins, a.pin)  # noqa: E731
        texts = {k: get(url).decode("latin-1") for k, url in SOURCES.items()}
        for d in LIBRARIES:
            check_releases(d, {r: Model(release_texts(d, r, texts, every=True)) for r in RELEASES})
        made = excerpts(texts)
        made[FIRMWARE_LIST] = firmware(get)
        whole_files = {key(h, u): made[key(h, u)] for h, u in WHOLE.items()}
        upstream = generate_all({**texts, **whole_files, FIRMWARE_LIST: made[FIRMWARE_LIST]})
        for f, text in generate_all(made).items():
            if body(text) != body(upstream[f]):
                sys.exit(f"{f.relative_to(LIB)}: the excerpts generate other tables than the files they are cut from")
        files = {HEADERS / k: t for k, t in made.items()}
        urls = set(SOURCES.values()) | {FIRMWARE_ORIGIN + p for ps in FIRMWARE.values() for p in ps}
        if a.pin:
            files[PINS] = json.dumps({u: d for u, d in sorted(pins.items()) if u in urls}, indent=1) + "\n"
    else:
        texts = {k: (HEADERS / k).read_text(encoding="latin-1") for k in [*SOURCES, FIRMWARE_LIST]
                 if (HEADERS / k).exists()}
        files = generate_all(texts)
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
