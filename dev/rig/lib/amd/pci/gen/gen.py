# /// script
# requires-python = ">=3.10"
# ///
"""Generates defs.ml, the tables of rig_amd_pci: the layouts of the
discovery table and its blocks' hardware IDs, the registers of each block
but GC (whose are rig_amd_abi's) at each version with headers, the layouts of firmware
images' headers and the types the security processor loads them as, and the
pinned firmware images.

Run from the worktree root:

  uv run dev/rig/lib/amd/pci/gen/gen.py
  uv run dev/rig/lib/amd/pci/gen/gen.py --check
  uv run dev/rig/lib/amd/pci/gen/gen.py --excerpt [--check]

The inputs are excerpts of the Linux kernel's amdgpu headers, in headers/:
each is a header's licence notice and the lines this script reads, verbatim
and in the header's order (a #define, an enumeration, a struct's definition).
headers/firmware.tsv lists the pinned firmware images of linux-firmware at
FIRMWARE_COMMIT: each image's path, BLAKE2b-256 digest and URL. --excerpt
makes it from the images, each checked against its SHA-256 pin.
SOURCES names the header each comes from, at its commit. --excerpt makes them
from the upstream headers, each pinned in pins.json by URL and SHA-256 and
checked against its pin; downloads are kept in --cache, and --pin records the
digests of headers not yet pinned. Generating reads the excerpts alone,
offline. --check generates into memory and fails if a committed file differs.

Text is read and written as latin-1, one character per byte, so every byte of
a header round-trips into its excerpt. Every table is a literal, which the
compiler lays out as static data: nothing is built when a program starts.
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

# Sources, each at the commit in its URL

KERNEL = ("https://raw.githubusercontent.com/ROCm/ROCK-Kernel-Driver/33970e1351f5e511029602454979f3de7e22260f/"
          "drivers/gpu/drm/amd/")

SOURCES = {
    "discovery.h": KERNEL + "include/discovery.h",  # the discovery table
    "amdgpu_discovery.h": KERNEL + "amdgpu/amdgpu_discovery.h",  # where it lies
    "soc15_hw_ip.h": KERNEL + "include/soc15_hw_ip.h",  # the blocks' hardware IDs
    "amdgpu_ucode.h": KERNEL + "amdgpu/amdgpu_ucode.h",  # firmware images' headers
    "psp_gfx_if.h": KERNEL + "amdgpu/psp_gfx_if.h",  # the types the PSP loads images as
    "amdgpu_vm.h": KERNEL + "amdgpu/amdgpu_vm.h",  # page-table entries
    "amdgpu_psp.h": KERNEL + "amdgpu/amdgpu_psp.h",  # the security processor's bootloader and ring
    "vega10_enum.h": KERNEL + "include/vega10_enum.h",  # memory types, GFX9
    "soc21_enum.h": KERNEL + "include/soc21_enum.h",  # GFX11
    "soc24_enum.h": KERNEL + "include/soc24_enum.h",  # GFX12
    "soc15_ih_clientid.h": KERNEL + "include/soc15_ih_clientid.h",  # interrupt clients
    **{f"irqsrcs_{b}.h": KERNEL + f"include/ivsrcid/{d}/irqsrcs_{b}.h"  # interrupt sources
       for d, b in (("gfx", "gfx_9_0"), ("gfx", "gfx_11_0_0"), ("gfx", "gfx_12_0_0"), ("sdma0", "sdma0_4_0"),
                    ("sdma0", "sdma0_5_0"))},
}

# The power manager's messages and clocks, by the MP1 versions amdgpu drives
# with each message header (amdgpu_smu.c's smu_set_funcs): the header of the
# messages, and the header of the clocks.
SMU_TABLES = [
    ([(13, 0, 0), (13, 0, 10)], "smu_v13_0_0_ppsmc.h", "smu13_driver_if_v13_0_0.h"),
    ([(13, 0, 7)], "smu_v13_0_7_ppsmc.h", "smu13_driver_if_v13_0_7.h"),
    ([(13, 0, 6), (13, 0, 14)], "smu_v13_0_6_ppsmc.h", "smu13_driver_if_v13_0_6.h"),
    ([(13, 0, 12)], "smu_v13_0_12_ppsmc.h", "smu13_driver_if_v13_0_6.h"),
    ([(14, 0, 2), (14, 0, 3)], "smu_v14_0_2_ppsmc.h", "smu14_driver_if_v14_0.h"),
]
SMU_MESSAGES = ["PPSMC_MSG_" + m for m in (
    "SetDriverDramAddrHigh", "SetDriverDramAddrLow", "EnableAllSmuFeatures", "GetSmuVersion", "GfxDriverReset",
    "Mode1Reset", "GetDpmFreqByIndex", "SetSoftMinByFreq", "SetSoftMaxByFreq", "QueryValidMcaCount",
    "McaBankDumpDW", "QueryValidMcaCeCount", "McaBankCeDumpDW")]
SMU_CLOCKS = ["PPCLK_UCLK", "PPCLK_FCLK", "PPCLK_SOCCLK", "PPCLK_GFXCLK"]
for _, _m, _c in SMU_TABLES:
    SOURCES[_m] = KERNEL + "pm/swsmu/inc/pmfw_if/" + _m
    SOURCES[_c] = KERNEL + "pm/swsmu/inc/pmfw_if/" + _c

# Interrupts: the clients' enumerations, and the source headers of the blocks
# whose interrupts the library reads.
IH_ENUMS = ["soc15_ih_clientid", "soc21_ih_clientid"]
IH_CLIENTS = ["SOC15_IH_CLIENTID_GRBM_CP", "SOC15_IH_CLIENTID_UTCL2"] + [
    f"SOC15_IH_CLIENTID_SE{i}SH" for i in range(4)] + [f"SOC15_IH_CLIENTID_SDMA{i}" for i in range(8)] + [
    "SOC21_IH_CLIENTID_GRBM_CP", "SOC21_IH_CLIENTID_GFX"]
IH_SOURCES = ["irqsrcs_gfx_9_0.h", "irqsrcs_gfx_11_0_0.h", "irqsrcs_gfx_12_0_0.h", "irqsrcs_sdma0_4_0.h",
              "irqsrcs_sdma0_5_0.h"]
SRCID = re.compile(r"^\s*#\s*define\s+(\w+?)__SRCID__(\w+)\s+(0x[0-9a-fA-F]+|\d+)\b", re.M)

# Page-table entries: their bits, the first bit of their fields, the levels
# of the tables, and the uncached memory type of each generation.
PTE_BITS = ["AMDGPU_PTE_VALID", "AMDGPU_PTE_SYSTEM", "AMDGPU_PTE_SNOOPED", "AMDGPU_PTE_EXECUTABLE",
            "AMDGPU_PTE_READABLE", "AMDGPU_PTE_WRITEABLE", "AMDGPU_PTE_TF", "AMDGPU_PDE_PTE", "AMDGPU_PDE_PTE_GFX12",
            "AMDGPU_PTE_IS_PTE"]
PTE_SHIFTS = ["AMDGPU_PTE_FRAG", "AMDGPU_PDE_BFS", "AMDGPU_PTE_MTYPE_VG10_SHIFT", "AMDGPU_PTE_MTYPE_NV10_SHIFT",
              "AMDGPU_PTE_MTYPE_GFX12_SHIFT"]
VM_LEVELS = ["AMDGPU_VM_PDB2", "AMDGPU_VM_PDB1", "AMDGPU_VM_PDB0", "AMDGPU_VM_PTB"]
MTYPES = {"vega10_enum.h": "soc15", "soc21_enum.h": "soc21", "soc24_enum.h": "soc24"}

# The register headers of each block but GC, by the stem of their names, at
# each version with headers the GPUs the library boots program.
REG_FILES = {
    "mmhub": [(1, 8, 0), (3, 0, 0), (3, 0, 1), (3, 0, 2), (4, 1, 0)],
    "nbio": [(4, 3, 0), (7, 2, 0), (7, 7, 0), (7, 9, 0)],
    "nbif": [(6, 3, 1)],
    "mp": [(11, 0, 0), (13, 0, 0), (14, 0, 2)],
    "hdp": [(4, 4, 2), (6, 0, 0), (7, 0, 0)],
    "osssys": [(4, 4, 2), (6, 0, 0), (7, 0, 0)],
    "sdma": [(4, 4, 2)],
}
REG_DIRS = {"osssys": "oss"}


def reg_header(prefix, ver, kind):
    """The header of [prefix]'s registers at [ver]: MP 11.0's is unversioned."""
    stem = "mp_11_0" if (prefix, ver) == ("mp", (11, 0, 0)) else f"{prefix}_{'_'.join(map(str, ver))}"
    return f"{stem}_{kind}.h"


for _p, _vs in REG_FILES.items():
    for _v in _vs:
        for _k in ("offset", "sh_mask"):
            SOURCES[reg_header(_p, _v, _k)] = KERNEL + f"include/asic_reg/{REG_DIRS.get(_p, _p)}/" + reg_header(_p, _v, _k)

# The registers the library reads and writes, by block: names that match.
VM = r"regMM"
REG_INVENTORY = {
    "mmhub": [
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
REG_INVENTORY["nbif"] = REG_INVENTORY["nbio"]

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

# What the library reads

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
                "PSP_FW_TYPE_PSP_DBG_DRV", "PSP_FW_TYPE_PSP_RAS_DRV"]
GFX_FW_TYPES = ["GFX_FW_TYPE_" + n for n in (
    "CP_ME", "CP_PFP", "CP_MEC", "CP_MEC_ME1", "RLC_V", "RLC_G", "SDMA0", "SDMA1", "SDMA2", "SDMA3", "SMU",
    "RLC_RESTORE_LIST_GPM_MEM", "RLC_RESTORE_LIST_SRM_MEM", "RLC_RESTORE_LIST_SRM_CNTL", "RLC_P", "RLC_IRAM",
    "RLC_DRAM_BOOT", "P2S_TABLE", "REG_LIST", "IMU_I", "IMU_D", "SDMA_UCODE_TH0", "SDMA_UCODE_TH1", "RS64_PFP",
    "RS64_ME", "RS64_MEC", "RS64_PFP_P0_STACK", "RS64_ME_P0_STACK", "RS64_MEC_P0_STACK")]
ENUMS = {"amdgpu_ucode.h": ["psp_fw_type"],
         "psp_gfx_if.h": ["psp_gfx_fw_type", "psp_gfx_crtl_cmd_id", "psp_gfx_cmd_id"],
         "amdgpu_psp.h": ["psp_bootloader_cmd", "psp_ring_type"]}

# The security processor: its commands, ring frames and bootloader steps.
PSP_COMMANDS = ["GFX_CTRL_CMD_ID_DESTROY_RINGS", "GFX_CMD_ID_SETUP_TMR", "GFX_CMD_ID_LOAD_IP_FW", "GFX_CMD_ID_LOAD_TOC",
                "GFX_CMD_ID_AUTOLOAD_RLC", "GFX_CMD_ID_SRIOV_SPATIAL_PART"]
PSP_BOOT = ["PSP_BL__LOAD_KEY_DATABASE", "PSP_BL__LOAD_TOS_SPL_TABLE", "PSP_BL__LOAD_SYSDRV", "PSP_BL__LOAD_SOCDRV",
            "PSP_BL__LOAD_INTFDRV", "PSP_BL__LOAD_DBGDRV", "PSP_BL__LOAD_RASDRV", "PSP_BL__LOAD_SOSDRV",
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

# The configuration the headers are read under: a little-endian processor.
DEFINED = {"LITTLEENDIAN_CPU"}

# Reading C headers

DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(\w+)(\([\w, ]*\))?[ \t]*(.*?)[ \t]*(?:/\*.*?\*/|//.*)?[ \t]*$", re.M)
ENUMERATOR = re.compile(r"^[ \t]*(\w+)[ \t]*(?:=[ \t]*([^,/\n]+?))?[ \t]*,?[ \t]*(?:/\*.*|//.*)?$", re.M)


def arg_defines(text):
    """{name: (parameter, body)} of [text]'s #define lines of one parameter."""
    return {m.group(1): (m.group(2).strip("() "), m.group(3)) for m in DEFINE.finditer(text)
            if m.group(2) is not None and "," not in m.group(2)}


def shift(text, name):
    """The first bit [name(1)] sets, for an argument macro [name]."""
    ds = arg_defines(text)
    if name not in ds:
        sys.exit(f"undefined argument macro {name}")
    param, body = ds[name]
    body = re.sub(rf"\b{param}\b", "1", body)
    return (evaluate(body, defines(text)) & 0xffff_ffff_ffff_ffff).bit_length() - 1


def defines(text):
    """{name: body} of [text]'s #define lines without parameters."""
    return {m.group(1): m.group(3) for m in DEFINE.finditer(text) if m.group(2) is None}


def evaluate(expr, names):
    """The integer of the C constant expression [expr]: numbers, names of
    [names], and integer operators."""
    e = re.sub(r"\((unsigned|uint32_t|uint64_t|int)\)", "", expr)
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
    """{name: value} of the macros [wanted] of [text]."""
    names = defines(text)
    out = {}
    for n in wanted:
        if n not in names:
            sys.exit(f"undefined constant {n}")
        out[n] = evaluate(names[n], names)
    return out


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


SCALARS = {"uint8_t": 1, "uint16_t": 2, "uint32_t": 4, "uint64_t": 8}


def packed_layout(text, name, upto=None):
    """(sizeof, {field: (byte offset, bytes) or ("bits", bit offset, bits)}) of
    the struct [name] of [text], laid out under #pragma pack(1): every member
    at the bit where the one before ends, a bit field's run ending on a byte,
    a union's members at its start. Nested members are named by their own
    names; an array's bytes are its element's; a flexible array takes none."""
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

REG_DEFINE = re.compile(r"^\s*#\s*define\s+((?:mm|reg)\w+)\s+(0x[\da-fA-F]+|\d+)\b", re.M)
MASK_DEFINE = re.compile(r"^\s*#\s*define\s+(\w+?)__(\w+)_MASK\s+(0x[\da-fA-F]+L?|\d+)\b", re.M)


def split_name(name):
    """(prefix, name) of a register's macro: "reg" or "mm", and the rest."""
    pos = next((i for i, c in enumerate(name) if c.isupper()), len(name))
    return name[:pos], name[pos:]


def normalize(prefix, reg):
    """An MMHUB VM register is named with its hub, as GC's are: regVM_L2_CNTL is
    regMMVM_L2_CNTL."""
    p, rest = split_name(reg)
    if prefix == "mmhub" and rest.startswith(("VM_", "MC_VM_")):
        return p + "MM" + rest
    return reg


def kept(prefix, reg):
    return any(re.fullmatch(r, normalize(prefix, reg)) for r in REG_INVENTORY[prefix])


def block_registers(prefix, offsets, masks):
    """{name: (offset, segment, [(field, lowest bit, highest bit)])} of the
    registers of [prefix] kept, from its offset and mask headers."""
    defs = {m.group(1): int(m.group(2), 0) for m in REG_DEFINE.finditer(offsets)}
    fields = {}
    for m in MASK_DEFINE.finditer(masks):
        mask = int(m.group(3).rstrip("L"), 0)
        fields.setdefault(m.group(1), []).append(
            (m.group(2).lower(), (mask & -mask).bit_length() - 1, mask.bit_length() - 1))
    out = {}
    for reg, off in defs.items():
        if reg.endswith("_BASE_IDX") or f"{reg}_BASE_IDX" not in defs or not kept(prefix, reg):
            continue
        out[normalize(prefix, reg)] = (off, defs[f"{reg}_BASE_IDX"], fields.get(split_name(reg)[1], []))
    return out


def hw_ids(text):
    """[(name, id)] of soc15_hw_ip.h's hardware IDs, in the header's order,
    aliases resolved."""
    names = defines(text)
    return [(n[:-len("_HWID")], evaluate(b, names)) for n, b in names.items() if n.endswith("_HWID")]

# Excerpts


def licence(name, text):
    """The header's leading comment, its licence notice."""
    m = re.match(r"\s*((?:/\*.*?\*/\s*|//[^\n]*\n)+)", text, re.S)
    if m is None:
        sys.exit(f"{name}: no licence notice")
    return m.group(1).rstrip() + "\n"


def blocks(text, names):
    """The lines of the definitions of the structs [names], from the line of
    their keyword to the line of their closing brace and name."""
    lines = text.splitlines()
    out, found = set(), set()
    for i, l in enumerate(lines):
        m = re.match(r"\s*(?:typedef\s+)?struct\s+(\w+)\s*(\{|$)", l)
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


def excerpt(name, text):
    """[name]'s excerpt of [text]: its licence notice and the lines this script
    reads, in order."""
    lines = text.splitlines()
    keep = set()
    if name == "discovery.h":
        keep |= blocks(text, set(DISCOVERY_STRUCTS))
        # The table enumeration and the pack(1) around the structs.
        first = next(i for i, l in enumerate(lines) if re.match(r"\s*typedef enum\s*\{?\s*$", l))
        last = next(i for i in range(first, len(lines)) if lines[i].strip().startswith("}"))
        keep |= set(range(first, last + 1))
        keep |= {i for i, l in enumerate(lines) if l.startswith("#pragma pack")}
        keep |= {i for i, l in enumerate(lines) if DEFINE.match(l) and DEFINE.match(l).group(1) in DISCOVERY_CONSTANTS}
    elif name == "amdgpu_discovery.h":
        keep |= {i for i, l in enumerate(lines) if DEFINE.match(l) and DEFINE.match(l).group(1) in DISCOVERY_PLACE}
    elif name == "amdgpu_ucode.h":
        keep |= blocks(text, set(UCODE_STRUCTS) | set(UCODE_NESTED)) | enum_blocks(text, ENUMS[name])
    elif name == "amdgpu_vm.h":
        keep |= enum_blocks(text, ["amdgpu_vm_level"])
        keep |= {i for i, l in enumerate(lines) if DEFINE.match(l) and DEFINE.match(l).group(1) in PTE_BITS + PTE_SHIFTS}
    elif any(name == m for _, m, _ in SMU_TABLES):
        keep |= {i for i, l in enumerate(lines) if DEFINE.match(l) and DEFINE.match(l).group(1) in SMU_MESSAGES}
    elif any(name == c for _, _, c in SMU_TABLES):
        keep |= enum_with(text, "PPCLK_UCLK")
    elif name == "soc15_ih_clientid.h":
        keep |= enum_blocks(text, IH_ENUMS)
    elif name in IH_SOURCES:
        keep |= {i for i, l in enumerate(lines) if SRCID.match(l)}
    elif name in MTYPES:
        keep |= {i for i, l in enumerate(lines) if re.match(r"\s*MTYPE_UC\s*=", l)}
    elif name == "psp_gfx_if.h":
        keep |= enum_blocks(text, ENUMS[name]) | blocks(text, set(PSP_STRUCTS) | {"psp_gfx_cmd_resp"})
    elif name == "amdgpu_psp.h":
        keep |= enum_blocks(text, ENUMS[name])
        keep |= {i for i, l in enumerate(lines) if DEFINE.match(l) and DEFINE.match(l).group(1) in PSP_SIZES}
    elif name.endswith(("_offset.h", "_sh_mask.h")):
        prefix = next(p for p, vs in REG_FILES.items() for v in vs
                      if name in (reg_header(p, v, "offset"), reg_header(p, v, "sh_mask")))
        for i, l in enumerate(lines):
            r = REG_DEFINE.match(l)
            if r and kept(prefix, r.group(1).removesuffix("_BASE_IDX")):
                keep.add(i)
            k = MASK_DEFINE.match(l)
            if k and kept(prefix, "reg" + k.group(1)):
                keep.add(i)
    elif name == "soc15_hw_ip.h":
        keep |= {i for i, l in enumerate(lines) if DEFINE.match(l) and DEFINE.match(l).group(1).endswith("_HWID")}
    return licence(name, text) + "\n" + "\n".join(lines[i] for i in sorted(keep)) + "\n"


def firmware(cache, pins, pin):
    """firmware.tsv: each pinned image's path, BLAKE2b-256 digest and URL."""
    rows = ["# path\tBLAKE2b-256\tURL, of linux-firmware at " + FIRMWARE_COMMIT]
    for n in IMAGES:
        url = FIRMWARE_URL + "amdgpu/" + n
        data = fetch(url, cache, pins, pin, text=False)
        rows.append(f"amdgpu/{n}\t{hashlib.blake2b(data, digest_size=32).hexdigest()}\t{url}")
    return "\n".join(rows) + "\n"


def fetch(url, cache, pins, pin, text=True):
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
    return data.decode("latin-1") if text else data

# Emission


def ml_int(v):
    return f"0x{v:x}" if v > 9 else str(v)


def ml_field(v):
    return f"({v[1]}, {v[2]})" if v[0] == "bits" else f"({v[0]}, {v[1]})"


def ml_module(name):
    return name[0].upper() + name[1:]


def generate(h):
    """defs.ml, from the excerpts [h], by name."""
    out = ["(* Generated by gen/gen.py from the excerpts in gen/headers; do not edit. *)", ""]

    # Discovery
    d = h["discovery.h"]
    out += ["(* The discovery table *)", ""]
    for n, v in constants(d, DISCOVERY_CONSTANTS).items():
        out.append(f"let {n.lower()} = {ml_int(v)}")
    for n, v in constants(h["amdgpu_discovery.h"], DISCOVERY_PLACE).items():
        out.append(f"let {n.lower()} = {ml_int(v)}")
    out.append("")
    out.append("(* The tables the binary header lists, by index. *)")
    for n, v in enum_values(d, DISCOVERY_TABLES).items():
        out.append(f"let {n.lower()}_table = {v}")
    out.append("")
    out.append("(* Layouts: each field as (byte offset, bytes), or (bit offset, bits) for a")
    out.append("   bit field; an array's bytes are its element's. *)")
    for s, wanted in DISCOVERY_STRUCTS.items():
        size, fields = packed_layout(d, s)
        out.append(f"module {ml_module(s)} = struct")
        out.append(f"  let sizeof = {size}")
        for f in wanted:
            if f not in fields:
                sys.exit(f"{s} has no field {f}")
            out.append(f"  let {f} = {ml_field(fields[f])}")
        out.append("end")
        out.append("")

    # Hardware IDs
    ids = hw_ids(h["soc15_hw_ip.h"])
    out += ["(* The blocks' hardware IDs, and their names *)", ""]
    known = dict(ids)
    for n in HWIDS:
        out.append(f"let {n.lower()}_hwid = {known[n]}")
    out.append("")
    first = {}
    for n, v in ids:
        first.setdefault(v, n)
    out.append("let hwid_name = function")
    for v, n in sorted(first.items()):
        out.append(f"  | {v} -> {json.dumps(n)}")
    out.append("  | _ -> \"\"")
    out.append("")

    # Page tables
    vm = h["amdgpu_vm.h"]
    out += ["(* Page-table entries *)", ""]
    for n, v in constants(vm, PTE_BITS).items():
        out.append(f"let {n.lower()} = 0x{v:x}L")
    for n in PTE_SHIFTS:
        out.append(f"let {n.lower().removesuffix('_shift')}_shift = {shift(vm, n)}")
    for n, v in enum_values(vm, VM_LEVELS).items():
        out.append(f"let {n.lower()} = {v}")
    for hdr, gen_name in MTYPES.items():
        m = re.search(r"MTYPE_UC\s*=\s*(0x[0-9a-fA-F]+|\d+)", h[hdr])
        if m is None:
            sys.exit(f"{hdr}: no MTYPE_UC")
        out.append(f"let {gen_name}_mtype_uc = {int(m.group(1), 0)}")
    out.append("")

    # Security processor
    g, ps = h["psp_gfx_if.h"], h["amdgpu_psp.h"]
    out += ["(* The security processor *)", ""]
    for n, v in {**enum_values(g, PSP_COMMANDS), **enum_values(ps, PSP_BOOT), **constants(ps, PSP_SIZES)}.items():
        out.append(f"let {n.lower()} = {ml_int(v)}")
    # A command buffer holds the command at the offset its first reserved
    # array's size subtracts, and the response at the one it subtracts from,
    # in a buffer of the size the second array's subtracts from.
    m = re.search(r"reserved_1\[(\d+)\s*-\s*sizeof\(union psp_gfx_commands\)\s*-\s*(\d+)\]", g)
    n = re.search(r"reserved_2\[(\d+)\s*-\s*(\d+)\s*-\s*sizeof\(struct psp_gfx_resp\)\]", g)
    if m is None or n is None or m.group(1) != n.group(2):
        sys.exit("psp_gfx_cmd_resp: no command and response offsets")
    out += [f"let psp_command_bytes = {n.group(1)}", f"let psp_command_at = {m.group(2)}",
            f"let psp_response_at = {m.group(1)}"]
    size, fields = packed_layout(g, "psp_gfx_cmd_resp", upto="cmd")
    out.append(f"let psp_command_id = {ml_field(fields['cmd_id'])}")
    out.append("")
    for st, (wanted, upto) in PSP_STRUCTS.items():
        size, fields = packed_layout(g, st, upto)
        out.append(f"module {ml_module(st)} = struct")
        out.append(f"  let sizeof = {size}")
        for f in wanted:
            if f not in fields:
                sys.exit(f"{st} has no field {f}")
            out.append(f"  let {f} = {ml_field(fields[f])}")
        out += ["end", ""]

    # Power manager
    out += ["(* The power manager's messages and clocks, by MP1 version, as (name, value). *)",
            "let smu_messages = function"]
    for versions, m, c in SMU_TABLES:
        msgs = {n: v for n, v in ((n, b) for n, b in defines(h[m]).items()) if n in SMU_MESSAGES}
        vals = [(n, evaluate(msgs[n], {})) for n in SMU_MESSAGES if n in msgs]
        clocks = enum_values(h[c], [n for n in SMU_CLOCKS if re.search(rf"\b{n}\b", h[c])])
        vals += list(clocks.items())
        pats = " | ".join(f"({a}, {b}, {cc})" for a, b, cc in versions)
        out.append(f"  | {pats} ->")
        out.append("      [ " + "; ".join(f"({json.dumps(n)}, {ml_int(v)})" for n, v in vals) + " ]")
    out += ["  | _ -> []", ""]

    # Interrupts
    ih = h["soc15_ih_clientid.h"]
    out += ["(* Interrupts *)", ""]
    for soc in ("soc15", "soc21"):
        body = re.search(rf"enum\s+{soc}_ih_clientid\s*\{{(.*?)\}}", preprocess(ih), re.S).group(1)
        prefix = f"{soc.upper()}_IH_CLIENTID_"
        primary = [n.strip() for n, v in re.findall(r"(\w+)\s*=\s*(0x[0-9a-fA-F]+|\d+)", body)]
        values = enum_values(ih, primary)
        every = enum_values(ih, re.findall(r"(\w+)\s*=", body))
        for n in IH_CLIENTS:
            if n.startswith(prefix):
                out.append(f"let {n.lower()} = {ml_int(every[n])}")
        out.append(f"let {soc}_client_name = function")
        for n, v in sorted(values.items(), key=lambda x: x[1]):
            out.append(f"  | {ml_int(v)} -> {json.dumps(n[len(prefix):])}")
        out.append("  | _ -> \"\"")
        out.append("")
    out.append("(* Interrupt sources: block, source ID, name. *)")
    out.append("let ih_sources = [")
    for hdr in IH_SOURCES:
        for m in SRCID.finditer(h[hdr]):
            out.append(f"  ({json.dumps(m.group(1))}, {ml_int(int(m.group(3), 0))}, {json.dumps(m.group(2))});")
    out += ["]", ""]

    # Registers
    out += ["(* Registers *)", "",
            "(* The registers of each block but GC, at each version with headers: each",
            "   register's name, offset and segment in 32-bit words, and its fields as",
            "   (name, (lowest bit, highest bit)). *)",
            "let registers = ["]
    for prefix, vs in REG_FILES.items():
        for v in vs:
            regs = block_registers(prefix, h[reg_header(prefix, v, "offset")], h[reg_header(prefix, v, "sh_mask")])
            if not regs:
                sys.exit(f"{prefix} {v}: no register kept")
            out.append(f"  ( {json.dumps(prefix)}, ({v[0]}, {v[1]}, {v[2]}), [")
            for n, (off, seg, fs) in sorted(regs.items()):
                f = "; ".join(f"({json.dumps(fn)}, ({lo}, {hi}))" for fn, lo, hi in fs)
                out.append(f"      {{ Rig_amd_abi.Register.name = {json.dumps(n)}; offset = {ml_int(off)}; "
                           f"segment = {seg}; fields = [ {f} ] }};")
            out.append("    ] );")
    out += ["]", ""]

    # Firmware
    u = h["amdgpu_ucode.h"]
    out += ["(* Firmware images *)", "", "(* The headers of images, laid out as the discovery table's. *)"]
    for s, wanted in UCODE_STRUCTS.items():
        size, fields = packed_layout(u, s)
        out.append(f"module {ml_module(s)} = struct")
        out.append(f"  let sizeof = {size}")
        for f in wanted:
            if f not in fields:
                sys.exit(f"{s} has no field {f}")
            out.append(f"  let {f} = {ml_field(fields[f])}")
        out.append("end")
        out.append("")
    out.append("(* The components of the PSP's own image, by type. *)")
    for n, v in enum_values(u, PSP_FW_TYPES).items():
        out.append(f"let {n.lower()} = {v}")
    out.append("")
    out.append("(* The types the PSP loads the other images' pieces as. *)")
    for n, v in enum_values(h["psp_gfx_if.h"], GFX_FW_TYPES).items():
        out.append(f"let {n.lower()} = {v}")
    out.append("")
    out.append(f"(* The pinned images of linux-firmware at {FIRMWARE_COMMIT[:12]}: each image's path,")
    out.append("   and its BLAKE2b-256 digest; [origin ^ path] is its URL. *)")
    out.append(f"let origin = {json.dumps(FIRMWARE_URL)}")
    out.append("let pinned = [")
    for row in h[FIRMWARE].splitlines():
        if row.startswith("#"):
            continue
        path, digest, url = row.split("\t")
        if url != FIRMWARE_URL + path:
            sys.exit(f"{path}: URL {url} is not in the pinned tree")
        out.append(f"  ({json.dumps(path)}, {json.dumps(digest)});")
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
        files[HEADERS / FIRMWARE] = firmware(a.cache, pins, a.pin)
        urls = set(SOURCES.values()) | {FIRMWARE_URL + "amdgpu/" + n for n in IMAGES}
        if a.pin:
            PINS.write_text(json.dumps({u: d for u, d in pins.items() if u in urls}, indent=1,
                                       sort_keys=True) + "\n")
    else:
        h = {n: (HEADERS / n).read_text(encoding="latin-1") for n in [*SOURCES, FIRMWARE]}
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
