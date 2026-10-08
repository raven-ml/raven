# /// script
# requires-python = ">=3.10"
# ///
"""Generates defs.ml, the tables of rig_nv_pci: what booting an NVIDIA
GPU over PCI reads from NVIDIA's sources. The GSP's messages and the
structures its boot takes, the VBIOS's structures, the firmware
containers, the registers per family, the page-table entries, the names of
the channel errors, and the firmware images, pinned.

Run from the worktree root:

  uv run dev/rig/lib/nv/pci/gen/gen.py
  uv run dev/rig/lib/nv/pci/gen/gen.py --check
  uv run dev/rig/lib/nv/pci/gen/gen.py --excerpt [--check]

The inputs are excerpts in headers/: each is a header's licence notice and
the definitions this script reads, verbatim and in the header's order, with
the definitions they depend on; a few small headers are kept whole, under
the repository's MIT notice where they carry none. They come from NVIDIA's
open-gpu-kernel-modules at release 570.144, the GSP firmware's release, and
from Linux's nouveau driver. headers/firmware.tsv lists the firmware
images of linux-firmware at FIRMWARE_COMMIT: each image's path, BLAKE2b-256
digest and URL; the page of rig firmware, in dev/rig/bin/help.ml, names
FIRMWARE_COMMIT too. --excerpt makes the excerpts and firmware.tsv from the
upstream files, each pinned in pins.json by URL and SHA-256 and checked
against its pin; downloads are kept in --cache, and --pin records the
digests of files not yet pinned. Generating reads the excerpts and
firmware.tsv alone, offline, and requires firmware.tsv to list exactly the
images of FIRMWARE. --check generates into memory and fails if a committed
file differs.

Text is read and written as latin-1, one character per byte, so every byte
of a header round-trips into its excerpt as upstream wrote it.

Struct layouts follow the C rules of the 64-bit Linux ABIs, x86_64 and
aarch64 alike. The VBIOS's structures are packed as their format strings
say. A register's address is its define, plus its unit's base for the
falcon's units, whose headers give offsets within the unit. Every value is
NVIDIA's.
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
FIRMWARE_LIST = HEADERS / "firmware.tsv"
OUT = HERE.parent / "defs.ml"

KERNEL = "https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/570.144/"
LINUX = "https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/{}?h=v6.18"
FIRMWARE_COMMIT = "0a6871b19abf5d6e024b5d208b101ae53e7fa0de"
ORIGIN = f"https://gitlab.com/kernel-firmware/linux-firmware/-/raw/{FIRMWARE_COMMIT}/"

GSP_INC = "src/nvidia/arch/nvalloc/common/inc/"
SDK = "src/common/sdk/nvidia/inc/"
SWREF = "src/common/inc/swref/published/"
HWREF = "kernel-open/nvidia-uvm/hwref/"

# Each excerpt and the file it is cut from. A name is read from the first
# file that defines it.
SOURCES = {
    "msgq_priv.h": KERNEL + "src/common/shared/msgq/inc/msgq/msgq_priv.h",
    "message_queue_priv.h": KERNEL + "src/nvidia/inc/kernel/gpu/gsp/message_queue_priv.h",
    "g_rpc-message-header.h": KERNEL + "src/nvidia/generated/g_rpc-message-header.h",
    "g_rpc-structures.h": KERNEL + "src/nvidia/generated/g_rpc-structures.h",
    "g_sdk-structures.h": KERNEL + "src/nvidia/generated/g_sdk-structures.h",
    "rpc_headers.h": KERNEL + "src/nvidia/inc/kernel/vgpu/rpc_headers.h",
    "gsp_fw_wpr_meta.h": KERNEL + GSP_INC + "gsp/gsp_fw_wpr_meta.h",
    "rmRiscvUcode.h": KERNEL + GSP_INC + "rmRiscvUcode.h",
    "nverror.h": KERNEL + "src/common/sdk/nvidia/inc/nverror.h",
    "pci_exp_table.h": KERNEL + "src/nvidia/inc/kernel/platform/pci_exp_table.h",
    "kernel_gsp_fwsec.c": KERNEL + "src/nvidia/src/kernel/gpu/gsp/kernel_gsp_fwsec.c",
    "kernel_gsp_frts_tu102.c": KERNEL + "src/nvidia/src/kernel/gpu/gsp/arch/turing/kernel_gsp_frts_tu102.c",
    "kernel_gsp_vbios_tu102.c": KERNEL + "src/nvidia/src/kernel/gpu/gsp/arch/turing/kernel_gsp_vbios_tu102.c",
    "tu102_dev_mmu.h": KERNEL + HWREF + "turing/tu102/dev_mmu.h",
    "gh100_dev_mmu.h": KERNEL + HWREF + "hopper/gh100/dev_mmu.h",
    "gsp_init_args.h": KERNEL + "src/nvidia/inc/kernel/gpu/gsp/gsp_init_args.h",
    "gsp_static_config.h": KERNEL + "src/nvidia/inc/kernel/gpu/gsp/gsp_static_config.h",
    "gspifpub.h": KERNEL + GSP_INC + "gsp/gspifpub.h",
    "libos_init_args.h": KERNEL + "src/common/uproc/os/common/include/libos_init_args.h",
    "rmgspseq.h": KERNEL + GSP_INC + "rmgspseq.h",
    "fsp_nvdm_format.h": KERNEL + GSP_INC + "fsp/fsp_nvdm_format.h",
    "fsp_mctp_format.h": KERNEL + GSP_INC + "fsp/fsp_mctp_format.h",
    "kern_fsp_cot_payload.h": KERNEL + "src/nvidia/inc/kernel/gpu/fsp/kern_fsp_cot_payload.h",
    "g_os_nvoc.h": KERNEL + "src/nvidia/generated/g_os_nvoc.h",
    "g_mem_desc_nvoc.h": KERNEL + "src/nvidia/generated/g_mem_desc_nvoc.h",
    "nv_memory_type.h": KERNEL + "src/nvidia/inc/kernel/os/nv_memory_type.h",
    "g_allclasses.h": KERNEL + "src/nvidia/generated/g_allclasses.h",
    "nvlimits.h": KERNEL + SDK + "nvlimits.h",
    "nvmisc.h": KERNEL + "kernel-open/common/inc/nvmisc.h",
    "nvos.h": KERNEL + SDK + "nvos.h",
    "alloc_channel.h": KERNEL + SDK + "alloc/alloc_channel.h",
    "cl0000.h": KERNEL + SDK + "class/cl0000.h",
    "cl0080.h": KERNEL + SDK + "class/cl0080.h",
    "cl2080.h": KERNEL + SDK + "class/cl2080.h",
    "cl2080_notification.h": KERNEL + SDK + "class/cl2080_notification.h",
    "ctrl0080fifo.h": KERNEL + SDK + "ctrl/ctrl0080/ctrl0080fifo.h",
    "ctrl2080fifo.h": KERNEL + SDK + "ctrl/ctrl2080/ctrl2080fifo.h",
    "ctrl2080gpu.h": KERNEL + SDK + "ctrl/ctrl2080/ctrl2080gpu.h",
    "ctrl0080gr.h": KERNEL + SDK + "ctrl/ctrl0080/ctrl0080gr.h",
    "ctrl2080gr.h": KERNEL + SDK + "ctrl/ctrl2080/ctrl2080gr.h",
    "ctrl2080internal.h": KERNEL + SDK + "ctrl/ctrl2080/ctrl2080internal.h",
    "ctrl90f1.h": KERNEL + SDK + "ctrl/ctrl90f1.h",
    "ctrlc36f.h": KERNEL + SDK + "ctrl/ctrlc36f.h",
    "g_chipset_nvoc.h": KERNEL + "src/nvidia/generated/g_chipset_nvoc.h",
    "gpu_acpi_data.h": KERNEL + "src/nvidia/inc/kernel/gpu/gpu_acpi_data.h",
    "ctrl0073system.h": KERNEL + "src/common/sdk/nvidia/inc/ctrl/ctrl0073/ctrl0073system.h",
    "fw.h": LINUX.format("drivers/gpu/drm/nouveau/include/nvfw/fw.h"),
    "hs.h": LINUX.format("drivers/gpu/drm/nouveau/include/nvfw/hs.h"),
}

# Files kept whole: the function and event numbers are X-macro lines, which
# no definition holds.
WHOLE = {
    "rpc_global_enums.h": KERNEL + "src/nvidia/inc/kernel/vgpu/rpc_global_enums.h",
}

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

# The register headers, by excerpt: (directory, header). An addendum adds to
# the header before it.
REGISTER_SOURCES = {
    "nv_ref.h": "nv_ref.h",
    "tu102_dev_fb.h": "turing/tu102/dev_fb.h",
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
SOURCES.update({k: KERNEL + SWREF + v for k, v in REGISTER_SOURCES.items()})

# The register headers of each family, in the order the first that defines
# a name gives it.
FAMILIES = {
    "Legacy": ["nv_ref.h", "tu102_dev_fb.h", "ga102_dev_gc6_island.h", "ga102_dev_gc6_island_addendum.h",
               "tu102_dev_vm.h", "ga102_dev_gsp.h", "ga102_dev_falcon_v4.h", "ga102_dev_falcon_v4_addendum.h",
               "ga102_dev_riscv_pri.h", "ga102_dev_fbif_v4.h", "ga102_dev_falcon_second_pri.h",
               "ga102_dev_sec_pri.h", "tu102_dev_bus.h", "tu102_dev_ext_devices.h"],
    "Blackwell": ["nv_ref.h", "tu102_dev_fb.h", "ga102_dev_gc6_island.h", "ga102_dev_gc6_island_addendum.h",
                  "gh100_dev_vm.h", "tu102_dev_vm.h", "ga102_dev_gsp.h", "gh100_dev_falcon_v4.h",
                  "gh100_dev_fsp_pri.h", "tu102_dev_bus.h", "gb202_dev_therm.h",
                  "gb202_dev_therm_addendum.h"],
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
        "NV_PRISCV_RISCV_BCR_CTRL", "NV_PROM_DATA",
    ],
    "Blackwell": [
        "NV_PMC_BOOT_0", "NV_PMC_BOOT_42", "NV_PFB_PRI_MMU_WPR2_ADDR_LO", "NV_PFB_PRI_MMU_WPR2_ADDR_HI",
        "NV_PGC6_AON_SECURE_SCRATCH_GROUP_42", "NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE",
        "NV_VIRTUAL_FUNCTION_PRIV_FUNC_BAR1_BLOCK_LOW_ADDR", "NV_PBUS_BAR1_BLOCK", "NV_PGSP_QUEUE_HEAD",
        "NV_PFALCON_FALCON_HWCFG2", "NV_THERM_I2CS_SCRATCH", "NV_PFSP_EMEMC", "NV_PFSP_EMEMD",
        "NV_PFSP_QUEUE_HEAD", "NV_PFSP_QUEUE_TAIL", "NV_PFSP_MSGQ_HEAD", "NV_PFSP_MSGQ_TAIL",
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
    "NV_PRISCV_RISCV_BCR_CTRL_CORE_SELECT_RISCV", "NV_THERM_I2CS_SCRATCH_FSP_BOOT_COMPLETE_STATUS_SUCCESS",
]

# The falcons' register ranges, whose first address is a falcon's base.
UNITS = ["NV_PGSP", "NV_PSEC"]

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
CONSTANTS = [
    "NV_VGPU_MSG_SIGNATURE_VALID", "NV_VGPU_MSG_RESULT_RPC_PENDING", "NV_VGPU_MSG_HEADER_VERSION_MAJOR_TOT",
    "NV_VGPU_MSG_HEADER_VERSION_MINOR_TOT", "GSP_FW_WPR_META_REVISION",
    "GSP_FW_WPR_META_MAGIC", "BIT_HEADER_SIGNATURE", "BIT_TOKEN_FALCON_DATA", "BIT_TOKEN_V1_00_SIZE_6",
    "BIT_TOKEN_V1_00_SIZE_8", "BIT_DATA_FALCON_DATA_V2_SIZE_4", "FALCON_UCODE_ENTRY_APPID_FWSEC_PROD",
    "FALCON_UCODE_DESC_V3_SIZE_44", "BCRT30_RSA3K_SIG_SIZE", "NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_VERSION_V3",
    "FALCON_APPLICATION_INTERFACE_ENTRY_ID_DMEMMAPPER", "FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3_CMD_FRTS",
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
CONSTANTS += [
    "LIBOS_MEMORY_REGION_CONTIGUOUS", "LIBOS_MEMORY_REGION_LOC_SYSMEM", "LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2",
    "GSP_DMA_TARGET_COHERENT_SYSTEM", "NVDM_TYPE_COT", "REGISTRY_TABLE_ENTRY_TYPE_DWORD", "ADDR_SYSMEM",
    "ADDR_FBMEM", "NV_MEMORY_CACHED", "GSP_SEQ_BUF_OPCODE_REG_WRITE", "GSP_SEQ_BUF_OPCODE_REG_MODIFY", "GSP_SEQ_BUF_OPCODE_REG_POLL",
    "GSP_SEQ_BUF_OPCODE_DELAY_US", "GSP_SEQ_BUF_OPCODE_REG_STORE", "GSP_SEQ_BUF_OPCODE_CORE_RESET",
    "GSP_SEQ_BUF_OPCODE_CORE_START", "GSP_SEQ_BUF_OPCODE_CORE_WAIT_FOR_HALT", "GSP_SEQ_BUF_OPCODE_CORE_RESUME",
    "NV01_ROOT", "NV01_DEVICE_0", "NV20_SUBDEVICE_0", "FERMI_VASPACE_A", "AMPERE_CHANNEL_GPFIFO_A",
    "BLACKWELL_CHANNEL_GPFIFO_A", "AMPERE_COMPUTE_B", "ADA_COMPUTE_A", "BLACKWELL_COMPUTE_B", "AMPERE_DMA_COPY_B",
    "BLACKWELL_DMA_COPY_B", "NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN",
    "NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_INFO", "NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO",
    "NV2080_CTRL_CMD_GPU_PROMOTE_CTX", "NV2080_CTRL_CMD_FIFO_GET_DEVICE_INFO_TABLE",
    "NV90F1_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES",
    "NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS",
    "NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_PATCH",
    "NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_GPCS", "NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_TPC_PER_GPC",
    "NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_SM_PER_TPC", "NV2080_CTRL_GR_INFO_INDEX_MAX_WARPS_PER_SM",
    "NV2080_CTRL_GR_INFO_INDEX_SM_VERSION", "NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES",
    "NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING", "NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED",
    "MCTP_MSG_HEADER_TYPE_VENDOR_PCI", "MCTP_MSG_HEADER_VENDOR_ID_NV", "NV2080_ENGINE_TYPE_GRAPHICS",
]

# Bit fields of 32-bit words, "hi:lo": (lowest bit, bits).
FIELDS = ["NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_VERSION", "NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_SIZE",
          "NV_VGPU_MSG_HEADER_VERSION_MAJOR", "NV_VGPU_MSG_HEADER_VERSION_MINOR", "MCTP_HEADER_SOM",
          "MCTP_HEADER_EOM", "MCTP_MSG_HEADER_TYPE", "MCTP_MSG_HEADER_VENDOR_ID", "MCTP_MSG_HEADER_NVDM_TYPE"]

# The function and event numbers read, from rpc_global_enums.h.
FUNCTIONS = ["CONTINUATION_RECORD", "GSP_RM_ALLOC", "GSP_RM_CONTROL", "SET_PAGE_DIRECTORY",
             "GSP_SET_SYSTEM_INFO", "SET_REGISTRY", "UNLOADING_GUEST_DRIVER", "FREE"]
EVENTS = ["GSP_INIT_DONE", "GSP_RUN_CPU_SEQUENCER", "RC_TRIGGERED", "MMU_FAULT_QUEUED", "OS_ERROR_LOG",
          "UCODE_LIBOS_PRINT"]

# Structs, by C name: the module they become and the fields read ("a__b" for
# a field b of a struct a).
STRUCTS = {
    "msgqTxHeader": ("Msgq_tx_header", ["version", "size", "msgSize", "msgCount", "writePtr", "flags",
                                        "rxHdrOff", "entryOff"]),
    "msgqRxHeader": ("Msgq_rx_header", ["readPtr"]),
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
    "GspFwWprMeta": ("Wpr_meta", [
        "magic", "revision", "sysmemAddrOfRadix3Elf", "sizeOfRadix3Elf", "sysmemAddrOfBootloader",
        "sizeOfBootloader", "bootloaderCodeOffset", "bootloaderDataOffset", "bootloaderManifestOffset",
        "sysmemAddrOfSignature", "sizeOfSignature", "gspFwRsvdStart", "nonWprHeapOffset", "nonWprHeapSize",
        "gspFwWprStart", "gspFwHeapOffset", "gspFwHeapSize", "gspFwOffset", "bootBinOffset", "frtsOffset",
        "frtsSize", "gspFwWprEnd", "fbSize", "vgaWorkspaceOffset", "vgaWorkspaceSize", "pmuReservedSize"]),
    "RM_RISCV_UCODE_DESC": ("Riscv_ucode_desc", ["monitorDataOffset", "monitorCodeOffset", "manifestOffset"]),
    "FALCON_APPLICATION_INTERFACE_HEADER_V1": ("App_interface_header", ["headerSize", "entrySize",
                                                                       "entryCount"]),
    "FALCON_APPLICATION_INTERFACE_ENTRY_V1": ("App_interface_entry", ["id", "dmemOffset"]),
    "FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3": ("Dmem_mapper", ["cmd_in_buffer_offset", "init_cmd"]),
    "FWSECLIC_READ_VBIOS_DESC": ("Read_vbios_desc", []),
    "FWSECLIC_FRTS_REGION_DESC": ("Frts_region_desc", []),
    "FWSECLIC_FRTS_CMD": ("Frts_cmd", [
        "readVbiosDesc__version", "readVbiosDesc__size", "readVbiosDesc__gfwImageOffset",
        "readVbiosDesc__gfwImageSize", "readVbiosDesc__flags", "frtsRegionDesc__version",
        "frtsRegionDesc__size", "frtsRegionDesc__frtsRegionOffset4K", "frtsRegionDesc__frtsRegionSize",
        "frtsRegionDesc__frtsRegionMediaType"]),
    "rpc_gsp_rm_alloc_v03_00": ("Rpc_rm_alloc", ["hClient", "hParent", "hObject", "hClass", "status",
                                                 "paramsSize", "flags", "params"]),
    "rpc_gsp_rm_control_v03_00": ("Rpc_rm_control", ["hClient", "hObject", "cmd", "status", "paramsSize",
                                                     "flags", "params"]),
    "rpc_set_page_directory_v1E_05": ("Rpc_set_page_directory", ["hClient", "hDevice", "pasid", "params"]),
    "NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS_v1E_05": ("Set_page_directory", [
        "physAddress", "numEntries", "flags", "hVASpace", "chId", "subDeviceId", "pasid"]),
    "rpc_unloading_guest_driver_v1F_07": ("Rpc_unloading", ["bInPMTransition", "bGc6Entering", "newLevel"]),
    "rpc_run_cpu_sequencer_v17_00": ("Rpc_cpu_sequencer", ["bufferSizeDWord", "cmdIndex", "regSaveArea",
                                                           "commandBuffer"]),
    "GSP_ARGUMENTS_CACHED": ("Gsp_arguments", ["messageQueueInitArguments", "bDmemStack"]),
    "MESSAGE_QUEUE_INIT_ARGUMENTS": ("Queue_init_args", ["sharedMemPhysAddr", "pageTableEntryCount",
                                                         "cmdQueueOffset", "statQueueOffset"]),
    "LibosMemoryRegionInitArgument": ("Libos_region", ["id8", "pa", "size", "kind", "loc"]),
    "GspSystemInfo": ("System_info", ["gpuPhysAddr", "gpuPhysFbAddr", "gpuPhysInstAddr",
                                      "nvDomainBusDeviceFunc", "maxUserVa", "pciConfigMirrorBase",
                                      "pciConfigMirrorSize", "PCIDeviceID", "PCISubDeviceID", "PCIRevisionID",
                                      "bIsPassthru"]),
    "PACKED_REGISTRY_ENTRY": ("Registry_entry", ["nameOffset", "type", "data", "length"]),
    "PACKED_REGISTRY_TABLE": ("Registry_table", ["size", "numEntries", "entries"]),
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
    "NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS": ("Work_submit_token", ["workSubmitToken"]),
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
    "nvfw_hs_load_header_v2": ("Hs_load_header", ["os_data_offset", "os_data_size", "num_apps", "app"]),
}

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
TABLES = [
    ("robust_channel_errors", "nverror.h", r"ROBUST_CHANNEL_(?!ERROR_RECOVERY_LEVEL_|LAST_ERROR)(\w+)"),
]

# The firmware images, by path under a firmware directory, per family: the
# GSP's image (one file, with a signature section per family), its
# bootloader, and the image that starts it.
FIRMWARE = {
    "Ampere": ["nvidia/ga102/gsp/gsp-570.144.bin", "nvidia/ga102/gsp/bootloader-570.144.bin",
               "nvidia/ga102/gsp/booter_load-570.144.bin"],
    "Ada": ["nvidia/ga102/gsp/gsp-570.144.bin", "nvidia/ad102/gsp/bootloader-570.144.bin",
            "nvidia/ad102/gsp/booter_load-570.144.bin"],
    "Blackwell": ["nvidia/ga102/gsp/gsp-570.144.bin", "nvidia/gb202/gsp/bootloader-570.144.bin",
                  "nvidia/gb202/gsp/fmc-570.144.bin"],
}

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




def licence(text):
    """The file's first comment, its licence notice."""
    end = text.index("*/") + 2
    return text[:end] + "\n"


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


def wanted(model):
    """The names read from the files other than the register headers."""
    names = set(CONSTANTS) | set(FIELDS) | set(STRUCTS)
    names |= {c for c, _, _ in ROM_STRUCTS} | {f for _, _, f in ROM_STRUCTS} | {MMU_KIND}
    for _, prefix in MMU.values():
        names |= mmu_names(model, prefix) | {f"{prefix}_{v}" for v in MMU_VALUES}
    names |= {f"{MMU['V3'][1]}_{v}" for v in MMU_V3_VALUES}
    for _, h, pattern in TABLES:
        names |= table_names(model, h, pattern)
    return names


def family_wanted(family):
    """The names read from the register headers of [family]."""
    def names(model):
        out = register_names(model, REGISTERS[family]) | {"NV_VIRTUAL_FUNCTION_FULL_PHYS_OFFSET"}
        out |= {u for u in UNITS if u in model.where}
        return out | {v for v in REGISTER_VALUES if v in model.where}
    return names


def table_names(model, header, pattern):
    """The defines of [header] whose names [pattern] matches and whose values
    are numbers."""
    return {n for n, it in model.where.items()
            if it.header == header and it.kind == "define" and it.params is None and re.fullmatch(pattern, n)
            and re.fullmatch(r"\(?\s*(0[xX][0-9a-fA-F]+|\d+)[uUlL]*\s*\)?", it.body)}


def excerpts(texts):
    """The excerpts of the files [texts]: the definitions read from each,
    with those they need. Each family's register headers are read as that
    family reads them, the first that defines a name giving it."""
    others = [h for h in SOURCES if h not in REGISTER_SOURCES]
    groups = [(others, wanted)] + [(hs, family_wanted(f)) for f, hs in FAMILIES.items()]
    keep = {}
    for hs, want in groups:
        model = Model({h: texts[h] for h in hs})
        for h, its in model.closure(want(model)).items():
            keep.setdefault(h, {}).update(its)
            keep[h].update({p.start: p for p in model.pragmas.get(h, [])})
    out = {}
    for h in SOURCES:
        if h not in keep:
            sys.exit(f"{h}: nothing read from it")
        spans = sorted(keep[h].values(), key=lambda it: it.start)
        body = []
        for it in spans:
            if body and it.start < body[-1][1]:
                continue
            body.append((it.start, it.end))
        out[h] = licence(texts[h]) + "\n" + "\n\n".join(texts[h][s:e] for s, e in body) + "\n"
    return out


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


def emit_struct(out, module, size, fields, indent=""):
    out.append(f"{indent}module {module} = struct")
    out.append(f"{indent}  let sizeof = {size}")
    for f, v in fields:
        out.append(f"{indent}  let {snake(f)} = {ml_tuple(v)}")
    out.append(f"{indent}end")


def generate():
    texts = {h: (HEADERS / h).read_text(encoding="latin-1") for h in list(SOURCES) + list(WHOLE)}
    model = Model({h: texts[h] for h in SOURCES if h not in REGISTER_SOURCES})
    out = []

    out.append("(* Constants. *)")
    for c in CONSTANTS:
        out.append(f"let {snake(c)} = {ml_int(model.value(c))}")
    out.append("")
    out.append("(* Bit fields of 32-bit words: (lowest bit, bits). *)")
    for f in FIELDS:
        out.append(f"let {snake(f)} = {ml_tuple(model.bits(f))}")
    out.append("")

    functions, events = rpc_numbers(texts["rpc_global_enums.h"])
    out.append("(* The GSP's functions and events (rpc_global_enums.h). *)")
    for f in FUNCTIONS:
        out.append(f"let nv_vgpu_msg_function_{f.lower()} = {ml_int(functions[f])}")
    for e in EVENTS:
        out.append(f"let nv_vgpu_msg_event_{e.lower()} = {ml_int(events[e])}")
    out.append("")

    out.append("(* Structs: each field is (byte offset, bytes), an array's (offset, bytes")
    out.append("   of an element, elements). *)")
    for c, (module, names) in STRUCTS.items():
        size, _, fields = model.layout(c)
        for f in names:
            if f not in fields:
                sys.exit(f"{c} has no field {f}; it has {sorted(fields)}")
        emit_struct(out, module, size, [(f, fields[f]) for f in names])
        out.append("")

    out.append("(* The VBIOS's structures, packed: each field is (byte offset, bytes). *)")
    for c, module, fmt in ROM_STRUCTS:
        fmt_text = model.item(fmt).body.strip().strip('"')
        size, fields = rom_layout(rom_fields(model, c), fmt_text)
        emit_struct(out, module, size, fields)
        out.append("")

    for name, h, pattern in TABLES:
        entries = sorted((model.value(n), re.fullmatch(pattern, n).group(1)) for n in table_names(model, h, pattern))
        out.append(f"(* {h}'s {name.replace('_', ' ')}, by value. *)")
        out.append(f"let {name} = [")
        out += [f"  ({ml_int(v)}, {json.dumps(n)});" for v, n in entries]
        out.append("]")
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
    for row in FIRMWARE_LIST.read_text().splitlines():
        if row.startswith("#"):
            continue
        path, digest, url = row.split("\t")
        if url != ORIGIN + path:
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
    return header(texts) + "\n".join(out) + "\n"


def header(texts):
    owners = sorted({m.group(1).strip() for t in texts.values()
                     for m in re.finditer(r"Copyright \(c\) ([^\n]*?NVIDIA[^\n.]*)", t, re.I)})
    notice = "\n".join(f"   {l}".rstrip() for l in MIT.splitlines())
    return (
        "(*---------------------------------------------------------------------------\n"
        "  Copyright (c) 2026 The Raven authors. All rights reserved.\n"
        "  SPDX-License-Identifier: ISC\n"
        "  ---------------------------------------------------------------------------*)\n\n"
        "(* Generated by gen/gen.py from the excerpts in gen/headers; do not edit.\n"
        "   The command that regenerates this file is in gen/gen.py.\n\n"
        "   The values are NVIDIA's, copied under the MIT licence from its headers and\n"
        "   from the headers of Linux's nouveau driver (SPDX-License-Identifier: MIT):\n\n"
        + "\n".join(f"   Copyright (c) {o}" for o in owners) + "\n\n" + notice + " *)\n\n"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--check", action="store_true", help="fail if a committed file differs")
    p.add_argument("--excerpt", action="store_true", help="make the excerpts from the pinned files")
    p.add_argument("--pin", action="store_true", help="record the digests of files not yet pinned")
    p.add_argument("--cache", type=pathlib.Path, default=pathlib.Path.home() / ".cache/raven/nv-pci-gen")
    a = p.parse_args()
    if a.excerpt:
        pins = json.loads(PINS.read_text()) if PINS.exists() else {}
        get = lambda url: download(url, a.cache, pins, a.pin)  # noqa: E731
        texts = {h: get(url).decode("latin-1") for h, url in SOURCES.items()}
        files = {HEADERS / h: t for h, t in excerpts(texts).items()}
        files.update({HEADERS / h: whole(url, get(url).decode("latin-1")) for h, url in WHOLE.items()})
        images = sorted({p for ps in FIRMWARE.values() for p in ps})
        rows = ["# path\tBLAKE2b-256\tURL, of linux-firmware at " + FIRMWARE_COMMIT]
        for p in images:
            rows.append(f"{p}\t{hashlib.blake2b(get(ORIGIN + p), digest_size=32).hexdigest()}\t{ORIGIN + p}")
        files[FIRMWARE_LIST] = "\n".join(rows) + "\n"
        if a.pin:
            files[PINS] = json.dumps(dict(sorted(pins.items())), indent=1) + "\n"
    else:
        files = {OUT: generate()}
    stale = [f for f, text in files.items() if not f.exists() or f.read_text(encoding="latin-1") != text]
    if a.check:
        if stale:
            sys.exit("stale: " + ", ".join(str(f.relative_to(HERE.parent)) for f in stale))
        return
    for f in stale:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(files[f], encoding="latin-1")


if __name__ == "__main__":
    main()
