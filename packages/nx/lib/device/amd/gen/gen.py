#!/usr/bin/env python3
"""Generates amd_defs.ml and kfd_ioctl.h, the AMD definitions nx.amd.device reads.

Run from the repository root:

  uv run --with libclang==18.1.1 packages/nx/lib/device/amd/gen/gen.py
  uv run --with libclang==18.1.1 packages/nx/lib/device/amd/gen/gen.py --check

Every input is pinned in pins.json by URL and SHA-256: the source archives and
files below, and the firmware files of the linux-firmware commit below. Every
run checks each input it reads against its pin. Downloads, and the source trees
extracted from them, are kept in --cache under the digest of their URL. Only
--pin reaches beyond the pinned URLs: it lists the firmware of the commit and
records the digests of new inputs. The output is deterministic: --check
generates into a temporary directory, with no network once the cache holds the
pinned inputs, and fails if the committed files differ.

Struct layouts come from libclang, for x86_64 Linux; the script checks that
aarch64 Linux lays them out the same. It emits what the runtime reads, as the
inventories below name it, and the digests of the firmware of the GPU
generations the runtime supports.
"""

import hashlib
import json
import pathlib
import re
import sys
import tarfile
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent
sys.path.insert(0, str(HERE.parents[1] / "gen"))
from devgen import Unit, fetch, key, layout, main, ml_field, ml_int, ml_name, ml_version, stub_dir, struct_module  # noqa: E402

KERNEL = ("https://github.com/ROCm/ROCK-Kernel-Driver/archive/"
          "33970e1351f5e511029602454979f3de7e22260f.tar.gz")
ROCM = "https://raw.githubusercontent.com/ROCm/rocm-systems/cccc350dc620e61ae2554978b62ab3532dc10bd9/"
LLVM = "https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-20.1.0/"
FIRMWARE_COMMIT = "0a6871b19abf5d6e024b5d208b101ae53e7fa0de"
FIRMWARE_TREE = ("https://gitlab.com/api/v4/projects/kernel-firmware%2Flinux-firmware/repository/tree"
                 f"?path=amdgpu&ref={FIRMWARE_COMMIT}&per_page=100&page={{page}}")
FIRMWARE_RAW = f"https://gitlab.com/kernel-firmware/linux-firmware/-/raw/{FIRMWARE_COMMIT}/amdgpu/{{name}}"

ROCM_FILES = [
    "projects/rocr-runtime/runtime/hsa-runtime/core/inc/registers.h",
    "projects/rocr-runtime/runtime/hsa-runtime/inc/amd_hsa_queue.h",
    "projects/rocr-runtime/runtime/hsa-runtime/inc/amd_hsa_kernel_code.h",
    "projects/rocr-runtime/runtime/hsa-runtime/inc/amd_hsa_common.h",
    "projects/rocr-runtime/runtime/hsa-runtime/inc/hsa.h",
    "projects/aqlprofile/linux/vega10_enum.h",
    "projects/aqlprofile/linux/soc21_enum.h",
    "projects/aqlprofile/linux/soc24_enum.h",
]
LLVM_FILES = ["llvm/include/llvm/Support/AMDHSAKernelDescriptor.h"]

# Register blocks and the versions whose headers exist.
REG_FILES = {
    "gc": [(9, 4, 3), (11, 0, 0), (11, 0, 3), (11, 5, 0), (12, 0, 0)],
    "mmhub": [(1, 8, 0), (3, 0, 0), (3, 0, 1), (3, 0, 2), (3, 3, 0), (4, 1, 0)],
    "nbio": [(4, 3, 0), (7, 2, 0), (7, 7, 0), (7, 9, 0), (7, 11, 0)],
    "nbif": [(6, 3, 1)],
    "mp": [(11, 0, 0), (13, 0, 0), (14, 0, 2)],
    "hdp": [(4, 4, 2), (6, 0, 0), (7, 0, 0)],
    "osssys": [(4, 4, 2), (6, 0, 0), (6, 1, 0), (7, 0, 0)],
    "sdma": [(4, 4, 2)],
}

# The registers the runtime reads and writes, by block: names that match.
VM = r"reg(GC|MM)"
REG_INVENTORY = {
    "gc": [
        r"regGRBM_(CNTL|GFX_CNTL|GFX_INDEX|SOFT_RESET)",
        r"regSCRATCH_REG[0-35-7]",
        r"regRLC_(SPARE_INT|CNTL|SRM_CNTL|SPM_MC_CNTL|CP_SCHEDULERS|RLCS_BOOTLOAD_STATUS|SAFE_MODE|CGCG_CGLS_CTRL|CGTT_MGCG_OVERRIDE)",
        r"regCP_(HQD_ACTIVE|HQD_DEQUEUE_REQUEST|HQD_EOP_CONTROL|HQD_IB_CONTROL|HQD_PERSISTENT_STATE|HQD_PQ_CONTROL|"
        r"HQD_PQ_DOORBELL_CONTROL|HQD_PQ_WPTR_HI|MQD_BASE_ADDR|MQD_CONTROL|STAT|MEC_CNTL|MEC_RS64_CNTL|ME_CNTL|"
        r"MEC_DOORBELL_RANGE_(LOWER|UPPER)|RB_WPTR_POLL_CNTL|INT_CNTL|(PFP|ME|MEC_RS64)_PRGRM_CNTR_START(_HI)?)",
        r"regSH_MEM_(CONFIG|BASES)", r"regSPI_COMPUTE_QUEUE_RESET", r"regTCP_(CNTL|UTCL1_CNTL2)",
        r"regGB_ADDR_CONFIG", r"regCOMPUTE_TMPRING_SIZE",
        r"regSDMA[01]_RLC_CGCG_CTRL", r"regSDMA0_(F32_CNTL|MCU_CNTL|WATCHDOG_CNTL|UTCL1_CNTL|UTCL1_PAGE|CNTL)",
        r"regSDMA0_QUEUE0_(RB_CNTL|RB_BASE(_HI)?|RB_RPTR(_HI)?|RB_WPTR(_HI)?|RB_RPTR_ADDR_(LO|HI)|"
        r"RB_WPTR_POLL_ADDR_(LO|HI)|DOORBELL|DOORBELL_OFFSET|MINOR_PTR_UPDATE|IB_CNTL)",
    ],
    "vm": [  # the VM registers of the GC and MM hubs, in the gc and mmhub blocks
        VM + r"VM_CONTEXT0_(CNTL|PAGE_TABLE_(START|END|BASE)_ADDR_(LO32|HI32))",
        VM + r"VM_INVALIDATE_ENG17_(REQ|ACK|SEM)",
        VM + r"VM_INVALIDATE_ENG\d+_ADDR_RANGE_(LO32|HI32)",
        VM + r"VM_L2_(CNTL[2-5]?|PROTECTION_FAULT_(CNTL2?|STATUS(_LO32)?|DEFAULT_ADDR_(LO32|HI32)|ADDR_(LO32|HI32))|"
        r"CONTEXT1_IDENTITY_APERTURE_(LOW|HIGH)_ADDR_(LO32|HI32)|CONTEXT_IDENTITY_PHYSICAL_OFFSET_(LO32|HI32)|"
        r"BANK_SELECT_RESERVED_CID2)",
        VM + r"MC_VM_(AGP_(BASE|BOT|TOP)|SYSTEM_APERTURE_(LOW|HIGH)_ADDR|SYSTEM_APERTURE_DEFAULT_ADDR_(LSB|MSB)|"
        r"MX_L1_TLB_CNTL|FB_LOCATION_(BASE|TOP)|XGMI_LFB_(CNTL|SIZE))",
        r"regMM_ATC_L2_MISC_CG",
    ],
    "nbio": [
        r"regBIF_BX_PF0_RSMU_(INDEX|DATA)",
        r"regBIF_BX0_(PCIE_INDEX2(_HI)?|PCIE_DATA2|REMAP_HDP_MEM_FLUSH_CNTL|BIF_DOORBELL_INT_CNTL)",
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
REG_INVENTORY["gc"] += REG_INVENTORY["vm"]
REG_INVENTORY["mmhub"] = REG_INVENTORY["vm"]
REG_INVENTORY["nbif"] = REG_INVENTORY["nbio"]

# The GC registers a VF reaches through the RLC gateway: its range per segment
# ends at the last of these, as in the reference driver.
RLCG_PATTERNS = [
    "GCVM", "GCMC_VM", "CP_(HQD|MQD|MEC|ME_CNTL|PERFMON|RB_WPTR_POLL_CNTL|INT_CNTL|STAT|PFP_PRGRM|ME_PRGRM|COHER_START)",
    "COMPUTE_", "(SQ|GL2C|TCC)_PERFCOUNTER", "SQ_THREAD_TRACE", "SPI_(CONFIG_CNTL|COMPUTE_QUEUE_RESET)", "GRBM",
    "SH_MEM", "RLC", "TCP", "GB_ADDR_CONFIG",
    "SDMA[01]_(WATCHDOG_CNTL|UTCL1_(CNTL|PAGE)|MCU_CNTL|F32_CNTL|CNTL|QUEUE0_|RLC_CGCG_CTRL)", "SCRATCH_REG[0-35-7]",
]

AMD = "drivers/gpu/drm/amd"

# Constants the runtime reads, by C name: macros and enumerators.
CONSTANTS = [
    # IP blocks and their hardware ids
    "GC_HWIP", "HDP_HWIP", "SDMA0_HWIP", "MMHUB_HWIP", "NBIO_HWIP", "MP0_HWIP", "MP1_HWIP", "OSSSYS_HWIP", "NBIF_HWIP",
    "GC_HWID", "SDMA0_HWID", "NBIF_HWID",
    # discovery
    "BINARY_SIGNATURE", "DISCOVERY_TABLE_SIGNATURE", "HARVEST_TABLE_SIGNATURE", "IP_DISCOVERY", "GC", "HARVEST_INFO",
    # VF mailbox
    "mmRCC_IOV_FUNC_IDENTIFIER", "NV_MAIBOX_CONTROL_TRN_OFFSET_BYTE", "NV_MAILBOX_POLL_ACK_TIMEDOUT",
    "NV_MAILBOX_POLL_MSG_TIMEDOUT", "mmMAILBOX_MSGBUF_TRN_DW0", "mmMAILBOX_MSGBUF_RCV_DW0",
    "IDH_REQ_GPU_INIT_ACCESS", "IDH_REQ_GPU_FINI_ACCESS", "IDH_READY_TO_ACCESS_GPU",
    # PSP
    "PSP_1_MEG", "PSP_CMD_BUFFER_SIZE", "PSP_FENCE_BUFFER_SIZE", "PSP_TMR_ALIGNMENT", "PSP_RING_TYPE__KM",
    "PSP_BL__LOAD_KEY_DATABASE", "PSP_BL__LOAD_TOS_SPL_TABLE", "PSP_BL__LOAD_SYSDRV", "PSP_BL__LOAD_SOCDRV",
    "PSP_BL__LOAD_INTFDRV", "PSP_BL__LOAD_DBGDRV", "PSP_BL__LOAD_RASDRV", "PSP_BL__LOAD_SOSDRV",
    "PSP_FW_TYPE_PSP_SOS", "PSP_FW_TYPE_PSP_SYS_DRV", "PSP_FW_TYPE_PSP_KDB", "PSP_FW_TYPE_PSP_TOC", "PSP_FW_TYPE_PSP_SPL",
    "PSP_FW_TYPE_PSP_RL", "PSP_FW_TYPE_PSP_SOC_DRV", "PSP_FW_TYPE_PSP_INTF_DRV", "PSP_FW_TYPE_PSP_DBG_DRV",
    "PSP_FW_TYPE_PSP_RAS_DRV",
    "GFX_CMD_ID_LOAD_IP_FW", "GFX_CMD_ID_SETUP_TMR", "GFX_CMD_ID_LOAD_TOC", "GFX_CMD_ID_AUTOLOAD_RLC",
    "GFX_CMD_ID_SRIOV_SPATIAL_PART", "GFX_CTRL_CMD_ID_DESTROY_RINGS",
    "GFX_FW_TYPE_SMU", "GFX_FW_TYPE_P2S_TABLE", "GFX_FW_TYPE_SDMA0", "GFX_FW_TYPE_SDMA1", "GFX_FW_TYPE_SDMA2",
    "GFX_FW_TYPE_SDMA3", "GFX_FW_TYPE_SDMA_UCODE_TH0", "GFX_FW_TYPE_SDMA_UCODE_TH1", "GFX_FW_TYPE_CP_PFP",
    "GFX_FW_TYPE_CP_ME", "GFX_FW_TYPE_CP_MEC", "GFX_FW_TYPE_CP_MEC_ME1",
    "GFX_FW_TYPE_RS64_PFP", "GFX_FW_TYPE_RS64_ME", "GFX_FW_TYPE_RS64_MEC", "GFX_FW_TYPE_RS64_PFP_P0_STACK",
    "GFX_FW_TYPE_RS64_ME_P0_STACK", "GFX_FW_TYPE_RS64_MEC_P0_STACK", "GFX_FW_TYPE_IMU_I", "GFX_FW_TYPE_IMU_D",
    "GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_CNTL", "GFX_FW_TYPE_RLC_RESTORE_LIST_GPM_MEM",
    "GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_MEM", "GFX_FW_TYPE_RLC_IRAM", "GFX_FW_TYPE_RLC_DRAM_BOOT", "GFX_FW_TYPE_RLC_P",
    "GFX_FW_TYPE_RLC_V", "GFX_FW_TYPE_RLC_G", "GFX_FW_TYPE_REG_LIST",
    # page tables
    "AMDGPU_VM_PDB2", "AMDGPU_VM_PDB1", "AMDGPU_VM_PDB0", "AMDGPU_VM_PTB",
    # doorbells
    "AMDGPU_NAVI10_DOORBELL_MEC_RING0", "AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE0", "AMDGPU_DOORBELL_KIQ",
    # interrupt clients
    "SOC15_IH_CLIENTID_GRBM_CP", "SOC15_IH_CLIENTID_UTCL2", "SOC21_IH_CLIENTID_GRBM_CP", "SOC21_IH_CLIENTID_GFX",
    # PM4 packets of a VF's KIQ
    "PACKET3_WRITE_DATA", "PACKET3_WAIT_REG_MEM", "WR_CONFIRM",
    # KFD
    "KFD_IOC_ALLOC_MEM_FLAGS_VRAM", "KFD_IOC_ALLOC_MEM_FLAGS_GTT", "KFD_IOC_ALLOC_MEM_FLAGS_USERPTR",
    "KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE", "KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE", "KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC",
    "KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE", "KFD_IOC_ALLOC_MEM_FLAGS_COHERENT", "KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED",
    "KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP", "KFD_MMIO_REMAP_HDP_MEM_FLUSH_CNTL",
    "KFD_IOC_QUEUE_TYPE_COMPUTE", "KFD_IOC_QUEUE_TYPE_SDMA", "KFD_IOC_QUEUE_TYPE_COMPUTE_AQL",
    "KFD_IOC_EVENT_SIGNAL", "KFD_IOC_EVENT_MEMORY", "KFD_IOC_EVENT_HW_EXCEPTION",
    # AQL queues and kernels
    "AMD_QUEUE_PROPERTIES_IS_PTR64", "AMD_QUEUE_PROPERTIES_ENABLE_PROFILING",
    "AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR", "AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER",
    "AMD_KERNEL_CODE_PROPERTIES_ENABLE_WAVEFRONT_SIZE32",
    "SQ_SEL_X", "SQ_SEL_Y", "SQ_SEL_Z", "SQ_SEL_W", "SQ_RSRC_BUF", "BUF_FORMAT_32_UINT", "BUF_NUM_FORMAT_UINT",
    "BUF_DATA_FORMAT_32",
]
# 64-bit page-table flags.
PTE_CONSTANTS = ["AMDGPU_PTE_VALID", "AMDGPU_PTE_SYSTEM", "AMDGPU_PTE_SNOOPED", "AMDGPU_PTE_EXECUTABLE",
                 "AMDGPU_PTE_READABLE", "AMDGPU_PTE_WRITEABLE", "AMDGPU_PTE_TF", "AMDGPU_PDE_PTE",
                 "AMDGPU_PDE_PTE_GFX12", "AMDGPU_PTE_IS_PTE"]

# Structs, by C name: the module they become and the fields read, or None for
# all of them. Nested fields are joined with "__".
MQD_FIELDS = ["header", "cp_mqd_base_addr_lo", "cp_mqd_base_addr_hi", "cp_hqd_pipe_priority", "cp_hqd_queue_priority",
              "cp_hqd_quantum", "cp_hqd_persistent_state", "cp_hqd_pq_base_lo", "cp_hqd_pq_base_hi",
              "cp_hqd_pq_rptr_report_addr_lo", "cp_hqd_pq_rptr_report_addr_hi", "cp_hqd_pq_wptr_poll_addr_lo",
              "cp_hqd_pq_wptr_poll_addr_hi", "cp_hqd_pq_doorbell_control", "cp_hqd_pq_control", "cp_hqd_ib_control",
              "cp_hqd_hq_status0", "cp_mqd_control", "cp_hqd_vmid", "cp_hqd_aql_control", "cp_hqd_eop_base_addr_lo",
              "cp_hqd_eop_base_addr_hi", "cp_hqd_eop_control"]
MQD_OPTIONAL = ["compute_tg_chunk_size", "compute_current_logic_xcc_id", "cp_mqd_stride_size"]
MQDS = {9: "v9_mqd", 11: "v11_compute_mqd", 12: "v12_compute_mqd"}

STRUCTS = {
    "binary_header": ("Binary_header", ["binary_signature", "table_list"]),
    "table_info": ("Table_info", ["offset"]),
    "ip_discovery_header": ("Ip_discovery_header", ["signature", "num_dies", "base_addr_64_bit", "die_info"]),
    "die_info": ("Die_info", ["die_offset"]),
    "die_header": ("Die_header", ["num_ips"]),
    "ip_v4": ("Ip_v4", ["hw_id", "instance_number", "num_base_address", "major", "minor", "revision"]),
    "gc_info_v1_0": ("Gc_info_v1_0", ["header__version_major", "gc_num_se", "gc_num_wgp0_per_sa", "gc_num_wgp1_per_sa",
                                      "gc_num_sa_per_se", "gc_max_scratch_slots_per_cu", "gc_max_waves_per_simd",
                                      "gc_lds_size"]),
    "gc_info_v2_0": ("Gc_info_v2_0", ["gc_num_se", "gc_num_cu_per_sh", "gc_num_sh_per_se",
                                      "gc_max_scratch_slots_per_cu", "gc_max_waves_per_simd", "gc_lds_size"]),
    "common_firmware_header": ("Common_firmware_header", ["header_version_major", "header_version_minor",
                                                          "ucode_array_offset_bytes", "ucode_size_bytes"]),
    "psp_firmware_header_v2_0": ("Psp_firmware_header_v2_0", ["psp_fw_bin_count", "psp_fw_bin"]),
    "psp_firmware_header_v2_1": ("Psp_firmware_header_v2_1", ["psp_aux_fw_bin_index", "psp_fw_bin"]),
    "psp_fw_bin_desc": ("Psp_fw_bin_desc", ["fw_type", "offset_bytes", "size_bytes"]),
    "smc_firmware_header_v2_1": ("Smc_firmware_header_v2_1", ["pptable_count", "pptable_entry_offset"]),
    "smc_soft_pptable_entry": ("Smc_soft_pptable_entry", ["id", "ppt_offset_bytes", "ppt_size_bytes"]),
    "sdma_firmware_header_v2_0": ("Sdma_firmware_header_v2_0", ["ctx_ucode_size_bytes", "ctl_ucode_offset",
                                                                "ctl_ucode_size_bytes"]),
    "sdma_firmware_header_v3_0": ("Sdma_firmware_header_v3_0", ["ucode_size_bytes"]),
    "gfx_firmware_header_v1_0": ("Gfx_firmware_header_v1_0", ["jt_offset", "jt_size"]),
    "gfx_firmware_header_v2_0": ("Gfx_firmware_header_v2_0", ["ucode_size_bytes", "data_offset_bytes",
                                                              "data_size_bytes", "ucode_start_addr_lo",
                                                              "ucode_start_addr_hi"]),
    "imu_firmware_header_v1_0": ("Imu_firmware_header_v1_0", ["imu_iram_ucode_size_bytes",
                                                              "imu_dram_ucode_size_bytes"]),
    "rlc_firmware_header_v2_1": ("Rlc_firmware_header_v2_1", [
        f"save_restore_list_{m}_{k}_bytes" for m in ("cntl", "gpm", "srm") for k in ("offset", "size")]),
    "rlc_firmware_header_v2_2": ("Rlc_firmware_header_v2_2", [
        f"rlc_{m}_ucode_{k}_bytes" for m in ("iram", "dram") for k in ("offset", "size")]),
    "rlc_firmware_header_v2_3": ("Rlc_firmware_header_v2_3", [
        f"rlc{m}_ucode_{k}_bytes" for m in ("p", "v") for k in ("offset", "size")]),
    "psp_gfx_cmd_resp": ("Psp_gfx_cmd_resp", [
        "cmd_id", "resp__status", "resp__tmr_size",
        "cmd__cmd_load_ip_fw__fw_phy_addr_lo", "cmd__cmd_load_ip_fw__fw_phy_addr_hi",
        "cmd__cmd_load_ip_fw__fw_size", "cmd__cmd_load_ip_fw__fw_type",
        "cmd__cmd_setup_tmr__buf_phy_addr_lo", "cmd__cmd_setup_tmr__buf_phy_addr_hi", "cmd__cmd_setup_tmr__buf_size",
        "cmd__cmd_setup_tmr__bitfield__virt_phy_addr", "cmd__cmd_setup_tmr__system_phy_addr_lo",
        "cmd__cmd_setup_tmr__system_phy_addr_hi",
        "cmd__cmd_load_toc__toc_phy_addr_lo", "cmd__cmd_load_toc__toc_phy_addr_hi", "cmd__cmd_load_toc__toc_size",
        "cmd__cmd_spatial_part__mode"]),
    "psp_gfx_rb_frame": ("Psp_gfx_rb_frame", ["cmd_buf_addr_lo", "cmd_buf_addr_hi", "fence_addr_lo",
                                              "fence_addr_hi", "fence_value"]),
    "amd_queue_s": ("Amd_queue", ["queue_properties", "read_dispatch_id_field_base_byte_offset", "max_cu_id",
                                  "max_wave_id", "read_dispatch_id", "write_dispatch_id", "compute_tmpring_size",
                                  "scratch_resource_descriptor", "scratch_backing_memory_location",
                                  "scratch_wave64_lane_byte_size"]),
    "kernel_descriptor_t": ("Kernel_descriptor", None),
}

SQ_BUF_RSRC = {  # by GC major: the unions of words 1 and 3
    9: ("SQ_BUF_RSRC_WORD1", "SQ_BUF_RSRC_WORD3"),
    11: ("SQ_BUF_RSRC_WORD1_GFX11", "SQ_BUF_RSRC_WORD3_GFX11"),
    12: ("SQ_BUF_RSRC_WORD1_GFX11", "SQ_BUF_RSRC_WORD3_GFX12"),
}
SQ_WORD1 = ["BASE_ADDRESS_HI", "SWIZZLE_ENABLE"]
SQ_WORD3 = ["DST_SEL_X", "DST_SEL_Y", "DST_SEL_Z", "DST_SEL_W", "ADD_TID_ENABLE", "TYPE"]
SQ_WORD3_OPTIONAL = ["NUM_FORMAT", "DATA_FORMAT", "ELEMENT_SIZE", "INDEX_STRIDE", "FORMAT", "OOB_SELECT"]

SMU = {  # version: headers
    (13, 0, 0): ["smu_v13_0_0_ppsmc", "smu13_driver_if_v13_0_0"],
    (13, 0, 6): ["smu_v13_0_6_ppsmc", "smu_v13_0_6_pmfw", "smu13_driver_if_v13_0_6"],
    (13, 0, 12): ["smu_v13_0_12_ppsmc", "smu_v13_0_12_pmfw", "smu13_driver_if_v13_0_6"],
    (14, 0, 2): ["smu_v14_0_0_pmfw", "smu_v14_0_2_ppsmc", "smu14_driver_if_v14_0"],
}
SMU_NAMES = ["PPSMC_MSG_" + m for m in (
    "SetDriverDramAddrHigh", "SetDriverDramAddrLow", "EnableAllSmuFeatures", "GetSmuVersion", "GfxDriverReset",
    "Mode1Reset", "GetDpmFreqByIndex", "SetSoftMinByFreq", "SetSoftMaxByFreq", "QueryValidMcaCount",
    "McaBankDumpDW", "QueryValidMcaCeCount", "McaBankCeDumpDW")] + ["PPCLK_UCLK", "PPCLK_FCLK", "PPCLK_SOCCLK",
                                                                    "PPCLK_GFXCLK"]
SDMA_PKT = {(4, 0, 0): "vega10_sdma_pkt_open", (5, 0, 0): "navi10_sdma_pkt_open", (6, 0, 0): "sdma_v6_0_0_pkt_open"}
SDMA_OPS = ["SDMA_OP_COPY", "SDMA_OP_FENCE", "SDMA_OP_TRAP", "SDMA_OP_POLL_REGMEM", "SDMA_OP_TIMESTAMP",
            "SDMA_SUBOP_COPY_LINEAR", "SDMA_SUBOP_TIMESTAMP_GET_GLOBAL"]
SDMA_FIELDS = ["SDMA_PKT_COPY_LINEAR_HEADER_sub_op", "SDMA_PKT_POLL_REGMEM_HEADER_func",
               "SDMA_PKT_POLL_REGMEM_HEADER_mem_poll", "SDMA_PKT_POLL_REGMEM_DW5_interval",
               "SDMA_PKT_POLL_REGMEM_DW5_retry_count", "SDMA_PKT_FENCE_HEADER_mtype",
               "SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_sub_op"]
SDMA_OPTIONAL = ["SDMA_PKT_FENCE_HEADER_mtype"]  # GFX9 engines take no memory type

# The headers' bit fields as a little-endian processor lays them out.
DEFINES = ["LITTLEENDIAN_CPU"]

# The firmware of the generations the runtime supports: GC 9.4.3, 9.4.4 and
# 9.5.0, 11 and 12, with their PSP 13 and 14, SMU 13 and 14, and SDMA 4.4, 6
# and 7.
FIRMWARE = re.compile(r"(psp_1[34]_.*_sos|smu_1[34]_.*|sdma_(4_4|6|7)_.*|"
                      r"gc_(9_4_[34]|9_5_0|11_\d+_\d+|12_\d+_\d+)_(sjt_)?(pfp|me|mec|imu|rlc))\.bin")

# Sources


def sources(cache, pins, pin):
    """The source trees, each under the digest of the URL it comes from, so that
    a moved pin never reads a tree extracted for another."""
    root = cache / "src"
    tar = fetch(cache, KERNEL, pins, pin)
    kernel = root / key(KERNEL)
    if not kernel.exists():
        partial = kernel.with_suffix(".partial")
        with tarfile.open(tar) as t:
            top = t.getnames()[0].split("/")[0]
            members = [m for m in t.getmembers()
                       if m.name.startswith(f"{top}/{AMD}/") or m.name == f"{top}/include/uapi/linux/kfd_ioctl.h"]
            for m in members:
                m.name = m.name[len(top) + 1:]
            t.extractall(partial, members=members, filter="data")
        partial.rename(kernel)
    trees = []
    for base, files in ((ROCM, ROCM_FILES), (LLVM, LLVM_FILES)):
        tree = root / key(base)
        for f in files:
            data = fetch(cache, base + f, pins, pin).read_bytes()
            dst = tree / f
            if not dst.exists() or dst.read_bytes() != data:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)
        trees.append(tree)
    return kernel, trees[0], trees[1]


def firmware(cache, pins, pin):
    """The SHA-256 of every firmware file the runtime may load: the files of the
    commit's tree under --pin, and otherwise those pins.json names."""
    names = []
    if pin:
        page = 1
        while True:
            with urllib.request.urlopen(FIRMWARE_TREE.format(page=page), timeout=60) as r:
                entries = json.loads(r.read())
            if not entries:
                break
            names += [e["name"] for e in entries if e["type"] == "blob"]
            page += 1
    else:
        prefix = FIRMWARE_RAW.format(name="")
        names = [u[len(prefix):] for u in pins if u.startswith(prefix)]
    names = [n for n in names if FIRMWARE.fullmatch(n)]
    out = {}
    for n in sorted(names):
        url = FIRMWARE_RAW.format(name=n)
        out[n] = hashlib.sha256(fetch(cache, url, pins, pin).read_bytes()).hexdigest()
    return out

# Registers


def split_name(name):
    pos = next((i for i, c in enumerate(name) if c.isupper()), len(name))
    return name[:pos], name[pos:]


def registers(kernel, prefix, ver):
    stem = prefix if prefix != "osssys" else "oss"
    base = kernel / AMD / "include/asic_reg" / stem / f"{prefix}_{'_'.join(map(str, ver))}"
    if prefix == "mp" and ver == (11, 0, 0):
        base = base.with_name("mp_11_0")

    def normalize(reg):
        s = split_name(reg)
        if prefix in ("gc", "mmhub") and s[1].startswith(("VM_", "MC_VM_")):
            return s[0] + prefix.upper()[:2] + s[1]
        return reg

    def extract(lines, pat):
        return ((normalize(m.group(1)), int(m.group(2), 0)) for l in lines if (m := re.match(pat, l)))

    offset = pathlib.Path(f"{base}_offset.h").read_text().splitlines()
    masks = pathlib.Path(f"{base}_sh_mask.h").read_text().splitlines()
    defs = dict(extract(offset, r"#define\s+((?:mm|reg)\S+)\s+(0x[\da-fA-F]+|\d+)"))
    fields = {}
    for name, mask in extract(masks, r"#define\s+(\S+)_MASK\s+(0x[\da-fA-F]+|\d+)"):
        reg, field = name.split("__")[0], name.split("__")[1].lower()
        fields.setdefault(reg, []).append((field, (mask & -mask).bit_length() - 1, mask.bit_length() - 1))
    return {reg: (off, defs[f"{reg}_BASE_IDX"], fields.get(split_name(reg)[1], []))
            for reg, off in defs.items() if f"{reg}_BASE_IDX" in defs}


def rlcg_extent(regs):
    ext = {}
    for name, (off, seg, _) in regs.items():
        if any(re.match("(mm|reg)" + p, name) for p in RLCG_PATTERNS):
            ext[seg] = max(ext.get(seg, 0), off)
    return sorted(ext.items())

def generate(cache, pins, pin, outdir):
    import clang.cindex as ci
    kernel, rocm, llvm = sources(cache, pins, pin)
    amd = kernel / AMD
    stub = stub_dir("amd")
    out = ["(* Generated by gen.py; do not edit. The inputs and the command that",
           "   regenerates this file are in gen.py; their digests are in pins.json. *)", ""]

    # Registers
    regs = {(p, v): registers(kernel, p, v) for p, vs in REG_FILES.items() for v in vs}
    out.append("(* The registers of each block version: (name, offset, segment, fields as")
    out.append("   (name, lowest bit, highest bit)). *)")
    out.append("let registers = [")
    for (prefix, ver), rs in regs.items():
        pats = [re.compile(p) for p in REG_INVENTORY[prefix]]
        keep = [(n, r) for n, r in rs.items() if any(p.fullmatch(n) for p in pats)]
        out.append(f"  ( {json.dumps(prefix)}, {ml_version(ver)}, [")
        for n, (off, seg, fields) in keep:
            fs = "; ".join(f"({json.dumps(f)}, {lo}, {hi})" for f, lo, hi in fields)
            out.append(f"      ({json.dumps(n)}, {ml_int(off)}, {seg}, [ {fs} ]);")
        out.append("    ] );")
    out.append("]")
    out.append("")
    out.append("(* The block versions with definitions: registers, SMU messages and SDMA packets. *)")
    fams = [(p, v) for p, v in regs] + [("smu", v) for v in SMU] + [("sdma_pkt", v) for v in SDMA_PKT]
    out.append("let families = [ " + "; ".join(f"({json.dumps(p)}, {ml_version(v)})" for p, v in fams) + " ]")
    out.append("")
    out.append("(* The extent of the GC registers a VF reaches through the RLC gateway, per segment. *)")
    out.append("let gc_rlcg_extent = function")
    for (p, v), rs in regs.items():
        if p == "gc":
            out.append(f"  | {ml_version(v)} -> [ " + "; ".join(f"({s}, {ml_int(o)})" for s, o in rlcg_extent(rs)) + " ]")
    out.append("  | _ -> []")
    out.append("")

    # Constants and structs of the kernel driver
    kheaders = [amd / h for h in [
        "include/discovery.h", "include/soc15_hw_ip.h", "include/soc15_ih_clientid.h", "include/v9_structs.h",
        "include/v10_structs.h", "include/v11_structs.h", "include/v12_structs.h", "amdgpu/amdgpu_ucode.h",
        "amdgpu/psp_gfx_if.h", "amdgpu/amdgpu_psp.h", "amdgpu/amdgpu_vm.h", "amdgpu/amdgpu_doorbell.h",
        "amdgpu/mxgpu_nv.h", "amdgpu/amdgpu_virt.h", "amdgpu/soc15d.h", "amdgpu/amdgpu.h"]]
    kheaders.append(kernel / "include/uapi/linux/kfd_ioctl.h")
    incs = [amd / "include", amd / "amdgpu", amd / "include/asic_reg", kernel / "include/uapi"]
    ku = Unit(ci, kheaders, incs, stub, defines=DEFINES)
    rheaders = [rocm / "projects/rocr-runtime/runtime/hsa-runtime/core/inc/registers.h",
                rocm / "projects/rocr-runtime/runtime/hsa-runtime/inc/amd_hsa_queue.h",
                rocm / "projects/rocr-runtime/runtime/hsa-runtime/inc/amd_hsa_kernel_code.h"]
    ru = Unit(ci, rheaders, [rocm / "projects/rocr-runtime/runtime/hsa-runtime/inc"], stub, defines=DEFINES)
    lu = Unit(ci, [llvm / LLVM_FILES[0]], [], stub, cpp=True, defines=DEFINES)
    values = {**ku.enums(), **ru.enums()}
    values.update(ku.macros([c for c in CONSTANTS + PTE_CONSTANTS if c not in values]))
    values.update(ru.macros([c for c in CONSTANTS if c not in values]))
    missing = [c for c in CONSTANTS + PTE_CONSTANTS if c not in values]
    if missing:
        sys.exit(f"undefined constants: {missing}")
    out.append("(* Constants *)")
    for c in CONSTANTS:
        v = values[c]
        if v >= 1 << 63 and v >= 0xffffffff80000000:
            v &= 0xffffffff  # an int shifted into its sign bit, as the 32-bit fields take it
        out.append(f"let {ml_name(c)} = {ml_int(v)}")
    for c in PTE_CONSTANTS:
        out.append(f"let {ml_name(c)} = 0x{values[c]:x}L")
    out.append("")

    # the hardware ids of IP blocks, in the order the kernel lists them
    src = (amd / "amdgpu/amdgpu_discovery.c").read_text()
    body = re.search(r"static int hw_id_map\[MAX_HWIP\] = \{(.*?)\};", src, re.S).group(1)
    pairs = re.findall(r"\[(\w+)\]\s*=\s*(\w+)", body)
    ids = ku.macros([hwid for _, hwid in pairs if hwid not in values])
    ids.update(values)
    out.append("(* The hardware id of each IP block, in the kernel's order. *)")
    out.append("let hw_id_map = [ " + "; ".join(f"({ml_int(values[ip])}, {ml_int(ids[hw])})" for ip, hw in pairs) + " ]")
    out.append("")

    # interrupt clients and sources
    for soc in ("soc15", "soc21"):
        names = sorted(((v, n) for n, v in values.items() if n.startswith(f"{soc.upper()}_IH_CLIENTID_")
                        and not n.endswith("_MAX")), key=lambda x: (x[0], x[1]))
        out.append(f"let {soc}_ih_clients = [ " + "; ".join(
            f"({ml_int(v)}, {json.dumps(n[len(soc) + 13:])})" for v, n in names) + " ]")
    out.append("let soc15_ih_clientid_se_sh = [ " + "; ".join(
        ml_int(values[f"SOC15_IH_CLIENTID_SE{i}SH"]) for i in range(4)) + " ]")
    out.append("let soc15_ih_clientid_sdma = [ " + "; ".join(
        ml_int(values[f"SOC15_IH_CLIENTID_SDMA{i}"]) for i in range(8)) + " ]")
    sources_ = []
    for f in ["gfx/irqsrcs_gfx_9_0.h", "gfx/irqsrcs_gfx_11_0_0.h", "gfx/irqsrcs_gfx_12_0_0.h",
              "sdma0/irqsrcs_sdma0_4_0.h", "sdma0/irqsrcs_sdma0_5_0.h"]:
        for m in re.finditer(r"#define\s+(\w+?)__SRCID__(\w+)\s+(0x[0-9a-fA-F]+|\d+)",
                             (amd / "include/ivsrcid" / f).read_text()):
            sources_.append((m.group(1), int(m.group(3), 0), m.group(2)))
    out.append("(* Interrupt sources: (block, id, name). *)")
    out.append("let ih_sources = [")
    for b, v, n in sources_:
        out.append(f"  ({json.dumps(b)}, {ml_int(v)}, {json.dumps(n)});")
    out.append("]")
    out.append("")

    # memory types and shader memory modes, per SOC generation
    for major, f in [(9, "vega10_enum.h"), (11, "soc21_enum.h"), (12, "soc24_enum.h")]:
        text = (rocm / "projects/aqlprofile/linux" / f).read_text()

        def enum(n):
            return int(re.search(rf"^\s*{n}\s*=\s*(0x[0-9a-fA-F]+|\d+)", text, re.M).group(1), 0)
        out.append(f"let soc{major}_mtype_uc = {enum('MTYPE_UC')}")
        out.append(f"let soc{major}_sh_mem_address_mode_64 = {enum('SH_MEM_ADDRESS_MODE_64')}")
        out.append(f"let soc{major}_sh_mem_alignment_mode_unaligned = {enum('SH_MEM_ALIGNMENT_MODE_UNALIGNED')}")
    out.append("let sh_mem_address_mode_64 = function 9 -> soc9_sh_mem_address_mode_64 | 11 -> "
               "soc11_sh_mem_address_mode_64 | _ -> soc12_sh_mem_address_mode_64")
    out.append("let sh_mem_alignment_mode_unaligned = function 9 -> soc9_sh_mem_alignment_mode_unaligned | 11 -> "
               "soc11_sh_mem_alignment_mode_unaligned | _ -> soc12_sh_mem_alignment_mode_unaligned")
    out.append("")

    # structs
    out.append("(* Structs: each field is (byte offset, bytes), an array's (offset, bytes")
    out.append("   of an element); a bit field's [_bits] is (bit offset, bits). *)")
    arm = Unit(ci, kheaders, incs, stub, target="aarch64-unknown-linux-gnu", defines=DEFINES)
    for cname, (module, wanted) in STRUCTS.items():
        unit = ku if ku.struct(cname) else ru if ru.struct(cname) else lu
        c = unit.struct(cname)
        if c is None:
            sys.exit(f"no struct {cname}")
        size, fields = layout(ci, c)
        if unit is ku:
            other = arm.struct(cname)
            if other is None or layout(ci, other) != (size, fields):
                sys.exit(f"struct {cname} differs on aarch64")
        struct_module(out, module, size, fields, wanted)
    out.append("module type MQD = sig")
    out.append("  val sizeof : int")
    for f in MQD_FIELDS:
        out.append(f"  val {f} : int * int")
    for f in MQD_OPTIONAL:
        out.append(f"  val {f} : (int * int) option")
    out.append("  val compute_static_thread_mgmt : (int * int) list")
    out.append("end")
    out.append("")
    for major, cname in MQDS.items():
        size, fields = layout(ci, ku.struct(cname))
        out.append(f"module Mqd_v{major} : MQD = struct")
        out.append(f"  let sizeof = {size}")
        for f in MQD_FIELDS:
            out.append(f"  let {f} = {ml_field(fields[f])}")
        for f in MQD_OPTIONAL:
            out.append(f"  let {f} = " + (f"Some {ml_field(fields[f])}" if f in fields else "None"))
        ses = 8 if major >= 10 else 4
        out.append("  let compute_static_thread_mgmt = [ " + "; ".join(
            ml_field(fields[f"compute_static_thread_mgmt_se{i}"]) for i in range(ses)) + " ]")
        out.append("end")
        out.append("")
    out.append("let mqd = function")
    for major in MQDS:
        out.append(f"  | {major} -> (module Mqd_v{major} : MQD)")
    out.append("  | m -> failwith (Printf.sprintf \"no MQD layout for GC %d\" m)")
    out.append("")

    # scratch buffer descriptors
    out.append("module type SQ_BUF_RSRC = sig")
    for f in SQ_WORD1 + SQ_WORD3:
        out.append(f"  val {ml_name(f) if f != 'TYPE' else 'type_'} : int * int")
    for f in SQ_WORD3_OPTIONAL:
        out.append(f"  val {ml_name(f)} : (int * int) option")
    out.append("end")
    out.append("")
    for major, (w1, w3) in SQ_BUF_RSRC.items():
        def bits(union):
            c = ru.struct(union)
            if c is None:
                sys.exit(f"no union {union}")
            _, fields = layout(ci, c)
            return {k.split("__")[-1]: v for k, v in fields.items() if v[0] == "bits"}
        b1, b3 = bits(w1), bits(w3)
        out.append(f"module Sq_buf_rsrc_v{major} : SQ_BUF_RSRC = struct")
        for f in SQ_WORD1:
            out.append(f"  let {ml_name(f)} = ({b1[f][1]}, {b1[f][2]})")
        for f in SQ_WORD3:
            out.append(f"  let {ml_name(f) if f != 'TYPE' else 'type_'} = ({b3[f][1]}, {b3[f][2]})")
        for f in SQ_WORD3_OPTIONAL:
            out.append(f"  let {ml_name(f)} = " + (f"Some ({b3[f][1]}, {b3[f][2]})" if f in b3 else "None"))
        out.append("end")
        out.append("")
    out.append("let sq_buf_rsrc = function")
    for major in SQ_BUF_RSRC:
        out.append(f"  | {major} -> (module Sq_buf_rsrc_v{major} : SQ_BUF_RSRC)")
    out.append("  | m -> failwith (Printf.sprintf \"no buffer descriptor layout for GC %d\" m)")
    out.append("")

    # SMU messages
    out.append("(* The SMU messages and clocks of each SMU version. *)")
    out.append("let smu_messages = function")
    for ver, hs in SMU.items():
        u = Unit(ci, [amd / "pm/swsmu/inc/pmfw_if" / f"{h}.h" for h in hs], incs + [amd / "pm/swsmu/inc/pmfw_if"], stub, defines=DEFINES)
        vals = {**u.enums()}
        vals.update(u.macros([n for n in SMU_NAMES if n not in vals]))
        out.append(f"  | {ml_version(ver)} -> [ " + "; ".join(
            f"({json.dumps(n)}, {ml_int(vals[n])})" for n in SMU_NAMES if n in vals) + " ]")
    out.append("  | _ -> []")
    out.append("")

    # SDMA packets
    out.append("module type SDMA = sig")
    for n in SDMA_OPS:
        out.append(f"  val {ml_name(n[5:])} : int")
    for n in SDMA_FIELDS:
        opt = " option" if n in SDMA_OPTIONAL else ""
        out.append(f"  val {ml_name(n[9:])} : (int * int){opt}  (* mask, shift *)")
    out.append("end")
    out.append("")
    for ver, h in SDMA_PKT.items():
        u = Unit(ci, [amd / "amdgpu" / f"{h}.h"], incs, stub, defines=DEFINES)
        want = SDMA_OPS + [f"{n}_{k}" for n in SDMA_FIELDS for k in ("mask", "shift")]
        vals = u.macros(want)
        out.append(f"module Sdma_v{ver[0]} : SDMA = struct")
        for n in SDMA_OPS:
            out.append(f"  let {ml_name(n[5:])} = {ml_int(vals[n])}")
        for n in SDMA_FIELDS:
            if n + "_mask" in vals:
                v = f"({ml_int(vals[n + '_mask'])}, {vals[n + '_shift']})"
                out.append(f"  let {ml_name(n[9:])} = {'Some ' + v if n in SDMA_OPTIONAL else v}")
            elif n in SDMA_OPTIONAL:
                out.append(f"  let {ml_name(n[9:])} = None")
            else:
                sys.exit(f"{h} has no {n}")
        out.append("end")
        out.append("")
    out.append("let sdma = function")
    for ver in SDMA_PKT:
        out.append(f"  | {ml_version(ver)} -> (module Sdma_v{ver[0]} : SDMA)")
    out.append("  | _ -> failwith \"no SDMA packet definitions\"")
    out.append("")

    # firmware
    out.append("(* The linux-firmware commit of the firmware below. *)")
    out.append(f"let firmware_commit = {json.dumps(FIRMWARE_COMMIT)}")
    out.append("")
    out.append("(* The SHA-256 of each firmware file. *)")
    out.append("let firmware_sha256 = [")
    for n, d in firmware(cache, pins, pin).items():
        out.append(f"  ({json.dumps(n)}, {json.dumps(d)});")
    out.append("]")

    (outdir / "amd_defs.ml").write_text("\n".join(out) + "\n")
    header = (kernel / "include/uapi/linux/kfd_ioctl.h").read_text()
    if "#include <drm/drm.h>" not in header:
        sys.exit("kfd_ioctl.h no longer includes drm/drm.h")
    (outdir / "kfd_ioctl.h").write_text(
        "/* Copied by gen.py from the Linux kernel's include/uapi/linux/kfd_ioctl.h\n"
        "   (ROCK-Kernel-Driver 33970e1351f5); do not edit. It includes\n"
        "   linux/types.h in place of drm/drm.h, which it needs only for those\n"
        "   types, so that building needs no DRM headers. */\n\n"
        + header.replace("#include <drm/drm.h>", "#include <linux/types.h>"))


if __name__ == "__main__":
    main(__doc__, generate, ["amd_defs.ml", "kfd_ioctl.h"], HERE, OUT, "amd-gen")
