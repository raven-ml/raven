# /// script
# requires-python = ">=3.10"
# dependencies = ["libclang==18.1.1"]
# ///
"""Generates defs.ml, the tables of device_amd_abi: the GC registers of each GC
version and the bases of their segments, the constants of PM4, SDMA and AQL
packets and of thread traces, the layouts of AQL's dispatch packet, of the
scratch buffer descriptor and of the kernel descriptor, and LLVM's AMDGPU
processors.

Run from the worktree root:

  uv run dev/device/amd/abi/gen/gen.py
  uv run dev/device/amd/abi/gen/gen.py --check

Every input is pinned in pins.json by URL and SHA-256, and checked against its
pin on every run. Downloads, and the source trees extracted from them, are kept
in --cache under the digest of their URL. --pin records the digests of inputs
not yet pinned. The output is deterministic: --check generates into a
temporary directory, with no network once the cache holds the pinned inputs,
and fails if the committed file differs.

Values come from the headers through libclang, parsed for x86_64 Linux. Every
table is a literal, which the compiler lays out as static data: nothing is
built when a program starts.
"""

import argparse
import hashlib
import json
import pathlib
import re
import sys
import tarfile
import tempfile
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "defs.ml"

KERNEL = ("https://github.com/ROCm/ROCK-Kernel-Driver/archive/"
          "33970e1351f5e511029602454979f3de7e22260f.tar.gz")
ROCM = "https://raw.githubusercontent.com/ROCm/rocm-systems/cccc350dc620e61ae2554978b62ab3532dc10bd9/"
LLVM = "https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-20.1.0/"
PAL = "https://raw.githubusercontent.com/GPUOpen-Drivers/pal/c5e800072a32f68b6ccc4422936d96167c6e0728/"

RUNTIME = "projects/rocr-runtime/runtime/hsa-runtime/"
ROCM_FILES = [RUNTIME + "core/inc/registers.h", RUNTIME + "inc/amd_hsa_kernel_code.h",
              RUNTIME + "inc/amd_hsa_common.h", RUNTIME + "inc/hsa.h",
              "projects/aqlprofile/linux/vega10_enum.h", "projects/aqlprofile/linux/soc21_enum.h",
              "projects/aqlprofile/linux/soc24_enum.h"]
LLVM_FILES = ["llvm/include/llvm/Support/AMDHSAKernelDescriptor.h", "llvm/include/llvm/BinaryFormat/ELF.h",
              "llvm/docs/AMDGPUUsage.rst"]
PAL_FILES = ["src/core/hw/gfxip/gfx9/chip/gfx9_plus_merged_f32_mec_pm4_packets.h",
             "src/core/hw/gfxip/gfx12/chip/gfx12_merged_f32_mec_pm4_packets.h"]

AMD = "drivers/gpu/drm/amd"

# The GC versions whose register headers exist, and the registers kept of them:
# those of compute queues, dispatches, performance counters, thread traces, and
# of bringing the GPU up (its queues, its memory hub, its firmware engines).
GC_VERSIONS = [(9, 4, 3), (11, 0, 0), (11, 0, 3), (11, 5, 0), (12, 0, 0)]
VM = r"reg(GC|MM)"
GC_REGISTERS = [
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
    r"regCOMPUTE_(DISPATCH_INITIATOR|START_X|PGM_LO|DISPATCH_SCRATCH_BASE_LO|PGM_RSRC1|RESOURCE_LIMITS|RESTART_X|"
    r"PGM_RSRC3|USER_DATA_0|PERFCOUNT_ENABLE|THREAD_TRACE_ENABLE)",
    r"regCP_PERFMON_CNTL(_1)?", r"regSQ_PERFCOUNTER_(CTRL2?|MASK)", r"reg(GRBM|GL2C|TCC|SQ)_PERFCOUNTER\d+_(SELECT|LO|HI)",
    r"regSQ_THREAD_TRACE_\w+", r"regSPI_CONFIG_CNTL",
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
# The bases of the GC's segments in PM4's register space, by the GC major
# version from which they hold.
GC_BASES = {9: "vega20_ip_offset.h", 10: "sienna_cichlid_ip_offset.h"}

# PM4: the same in soc15d.h (GFX9) and nvd.h (GFX10 on), and the release's
# enumerations in kfd_pm4_headers_ai.h.
PM4_CONSTANTS = [
    "PACKET_TYPE3", "PACKET3_SET_SH_REG", "PACKET3_SET_SH_REG_START", "PACKET3_SET_SH_REG_END", "PACKET3_SET_UCONFIG_REG",
    "PACKET3_SET_UCONFIG_REG_START", "PACKET3_PRED_EXEC", "PACKET3_WAIT_REG_MEM", "PACKET3_ACQUIRE_MEM",
    "PACKET3_RELEASE_MEM", "PACKET3_DISPATCH_DIRECT", "PACKET3_EVENT_WRITE", "PACKET3_INDIRECT_BUFFER", "PACKET3_COPY_DATA",
    "PACKET3_WRITE_DATA", "INDIRECT_BUFFER_VALID", "CACHE_FLUSH_AND_INV_TS_EVENT", "WR_ONE_ADDR", "WR_CONFIRM",
    "PACKET3_WAIT_REG_MEM__FUNCTION__EQUAL_TO_THE_REFERENCE_VALUE",
    "PACKET3_WAIT_REG_MEM__FUNCTION__GREATER_THAN_OR_EQUAL_REFERENCE_VALUE",
    "event_index__mec_release_mem__end_of_pipe", "data_sel__mec_release_mem__send_32_bit_low",
    "data_sel__mec_release_mem__send_64_bit_data", "int_sel__mec_release_mem__none",
    "int_sel__mec_release_mem__send_interrupt_after_write_confirm",
]
# soc15d.h alone: the destinations of WRITE_DATA, and the source and destination
# of COPY_DATA.
PM4_SOC15_ONLY = ["PACKET3_WRITE_DATA__DST_SEL__MEM_MAPPED_REGISTER", "PACKET3_WRITE_DATA__DST_SEL__MEMORY",
                  "PACKET3_COPY_DATA__SRC_SEL__PERFCOUNTERS", "PACKET3_COPY_DATA__SRC_SEL__GPU_CLOCK_COUNT",
                  "PACKET3_COPY_DATA__DST_SEL__TC_L2", "PACKET3_COPY_DATA__COUNT_SEL__64_BITS_OF_DATA",
                  "PACKET3_COPY_DATA__WR_CONFIRM__WAIT_FOR_CONFIRMATION"]
# Fields as the shift of their first bit, from the argument macros of both
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
    f"PACKET3_RELEASE_MEM_GCR_{f}" for f in ("GLV_INV", "GL1_INV", "GL2_INV", "GLM_WB", "GLM_INV", "GL2_WB", "SEQ")]
PM4_SOC15_SHIFTS = [f"PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_{f}" for f in
                    ("SH_ICACHE_ACTION_ENA", "SH_KCACHE_ACTION_ENA", "TC_ACTION_ENA", "TCL1_ACTION_ENA", "TC_WB_ACTION_ENA")]
PM4_SOC15_CONSTANTS = ["EOP_TC_WB_ACTION_EN", "EOP_TC_NC_ACTION_EN"]

# WAIT_REG_MEM64, which the kernel's headers name but do not lay out: PAL's
# layout of it, (field, byte offset or (bit offset, bits)), which the encoder
# writes in this order.
WAIT_REG_MEM64 = "PM4_MEC_WAIT_REG_MEM64"
WAIT_REG_MEM64_LAYOUT = [
    ("ordinal2__bitfields__function", ("bits", 32, 3)), ("ordinal2__bitfields__mem_space", ("bits", 36, 2)),
    ("ordinal3__bitfieldsA__mem_poll_addr_lo", ("bits", 67, 29)), ("ordinal4__mem_poll_addr_hi", (12, 4)),
    ("ordinal5__reference", (16, 4)), ("ordinal6__reference_hi", (20, 4)), ("ordinal7__mask", (24, 4)),
    ("ordinal8__mask_hi", (28, 4)), ("ordinal9__bitfields__poll_interval", ("bits", 256, 16))]

# The events and thread trace values of the SOC enumerations, the same in each
# that defines them.
SOC_EVENTS = ["CS_PARTIAL_FLUSH", "THREAD_TRACE_MARKER", "THREAD_TRACE_FINISH"]
SOC_TRACE = ["SQ_TT_RT_FREQ_4096_CLK", "SQ_TT_WTYPE_INCLUDE_CS_BIT", "SQ_TT_TOKEN_MASK_SQDEC_BIT",
             "SQ_TT_TOKEN_MASK_SHDEC_BIT", "SQ_TT_TOKEN_MASK_GFXUDEC_BIT", "SQ_TT_TOKEN_MASK_COMP_BIT",
             "SQ_TT_TOKEN_MASK_CONTEXT_BIT", "SQ_TT_TOKEN_EXCLUDE_VMEMEXEC_SHIFT", "SQ_TT_TOKEN_EXCLUDE_ALUEXEC_SHIFT",
             "SQ_TT_TOKEN_EXCLUDE_VALUINST_SHIFT", "SQ_TT_TOKEN_EXCLUDE_IMMEDIATE_SHIFT", "SQ_TT_TOKEN_EXCLUDE_INST_SHIFT"]

# SDMA packets: the same in each version's header but for the fence's memory
# type, from version 5.
SDMA_PKT = {(4, 0, 0): "vega10_sdma_pkt_open", (5, 0, 0): "navi10_sdma_pkt_open", (6, 0, 0): "sdma_v6_0_0_pkt_open"}
SDMA_OPS = ["SDMA_OP_COPY", "SDMA_OP_FENCE", "SDMA_OP_TRAP", "SDMA_OP_POLL_REGMEM", "SDMA_OP_TIMESTAMP",
            "SDMA_SUBOP_COPY_LINEAR", "SDMA_SUBOP_TIMESTAMP_GET_GLOBAL"]
SDMA_FIELDS = ["SDMA_PKT_COPY_LINEAR_HEADER_sub_op", "SDMA_PKT_POLL_REGMEM_HEADER_func",
               "SDMA_PKT_POLL_REGMEM_HEADER_mem_poll", "SDMA_PKT_POLL_REGMEM_DW5_interval",
               "SDMA_PKT_POLL_REGMEM_DW5_retry_count", "SDMA_PKT_FENCE_HEADER_mtype",
               "SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_sub_op"]
SDMA_OPTIONAL = ["SDMA_PKT_FENCE_HEADER_mtype"]  # GFX9 engines take no memory type

# AQL, from hsa.h.
HSA_CONSTANTS = ["HSA_PACKET_HEADER_TYPE", "HSA_PACKET_HEADER_BARRIER", "HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE",
                 "HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE", "HSA_FENCE_SCOPE_SYSTEM", "HSA_PACKET_TYPE_VENDOR_SPECIFIC",
                 "HSA_PACKET_TYPE_KERNEL_DISPATCH", "HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS"]
DISPATCH_FIELDS = ["header", "setup", "workgroup_size_x", "workgroup_size_y", "workgroup_size_z", "grid_size_x",
                   "grid_size_y", "grid_size_z", "private_segment_size", "group_segment_size", "kernel_object",
                   "kernarg_address"]

# The scratch buffer descriptor, by GC major: the unions of its words 1 and 3,
# and the values its fields take.
SQ_BUF_RSRC = {9: ("SQ_BUF_RSRC_WORD1", "SQ_BUF_RSRC_WORD3"),
               11: ("SQ_BUF_RSRC_WORD1_GFX11", "SQ_BUF_RSRC_WORD3_GFX11"),
               12: ("SQ_BUF_RSRC_WORD1_GFX11", "SQ_BUF_RSRC_WORD3_GFX12")}
SQ_WORD1 = ["BASE_ADDRESS_HI", "SWIZZLE_ENABLE"]
SQ_WORD3 = ["DST_SEL_X", "DST_SEL_Y", "DST_SEL_Z", "DST_SEL_W", "ADD_TID_ENABLE", "TYPE"]
SQ_WORD3_OPTIONAL = ["NUM_FORMAT", "DATA_FORMAT", "ELEMENT_SIZE", "INDEX_STRIDE", "FORMAT", "OOB_SELECT"]
SQ_CONSTANTS = ["SQ_SEL_X", "SQ_SEL_Y", "SQ_SEL_Z", "SQ_SEL_W", "SQ_RSRC_BUF", "BUF_FORMAT_32_UINT",
                "BUF_NUM_FORMAT_UINT", "BUF_DATA_FORMAT_32"]

# The kernel descriptor's fields and code properties, ELF.h's values of the
# AMDGPU header: its machine, its ABI version, and the fields of its flags.
KD_FIELDS = ["group_segment_fixed_size", "private_segment_fixed_size", "kernarg_size", "kernel_code_entry_byte_offset",
             "compute_pgm_rsrc3", "compute_pgm_rsrc1", "compute_pgm_rsrc2", "kernel_code_properties"]
KD_CONSTANTS = ["AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER",
                "AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR", "AMD_KERNEL_CODE_PROPERTIES_ENABLE_WAVEFRONT_SIZE32"]
ELF_CONSTANTS = ["EM_AMDGPU", "ELFABIVERSION_AMDGPU_HSA_V6", "EF_AMDGPU_MACH",
                 "EF_AMDGPU_GENERIC_VERSION", "EF_AMDGPU_GENERIC_VERSION_OFFSET"]

# The headers' bit fields as a little-endian processor lays them out.
DEFINES = ["LITTLEENDIAN_CPU"]

# Inputs


def key(url):
    """The digest of [url] that names what the cache keeps of it."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


USED = set()


def fetch(cache, url, pins, pin):
    """The path of [url]'s contents in [cache], verified against [pins]."""
    USED.add(url)
    path = cache / (key(url) + "-" + url.rsplit("/", 1)[-1])
    if not path.exists():
        print(f"fetching {url}", file=sys.stderr)
        req = urllib.request.Request(url, headers={"User-Agent": "raven-gen"})
        with urllib.request.urlopen(req, timeout=120) as r:
            path.write_bytes(r.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if url in pins and pins[url] != digest:
        sys.exit(f"{url}: SHA-256 {digest}, expected {pins[url]}")
    if url not in pins:
        if not pin:
            sys.exit(f"{url} is not pinned; run with --pin to record {digest}")
        pins[url] = digest
    return path


def sources(cache, pins, pin):
    """The kernel's tree, and the trees of the files of ROCm, LLVM and PAL, each
    under the digest of the URL it comes from."""
    root = cache / "src"
    tar = fetch(cache, KERNEL, pins, pin)
    kernel = root / key(KERNEL)
    if not kernel.exists():
        partial = kernel.with_suffix(".partial")
        with tarfile.open(tar) as t:
            top = t.getnames()[0].split("/")[0]
            members = [m for m in t.getmembers() if m.name.startswith(f"{top}/{AMD}/")]
            for m in members:
                m.name = m.name[len(top) + 1:]
            t.extractall(partial, members=members, filter="data")
        partial.rename(kernel)
    trees = []
    for base, files in ((ROCM, ROCM_FILES), (LLVM, LLVM_FILES), (PAL, PAL_FILES)):
        tree = root / key(base)
        for f in files:
            data = fetch(cache, base + f, pins, pin).read_bytes()
            dst = tree / f
            if not dst.exists() or dst.read_bytes() != data:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)
        trees.append(tree)
    return kernel / AMD, *trees

# C definitions


PRELUDE = """
typedef unsigned char uint8_t; typedef unsigned short uint16_t; typedef unsigned int uint32_t;
typedef unsigned long long uint64_t; typedef signed char int8_t; typedef short int16_t; typedef int int32_t;
typedef long long int64_t; typedef unsigned long uintptr_t; typedef long intptr_t; typedef unsigned long size_t;
typedef uint8_t u8; typedef uint16_t u16; typedef uint32_t u32; typedef uint64_t u64;
typedef int8_t s8; typedef int16_t s16; typedef int32_t s32; typedef int64_t s64;
typedef uint8_t __u8; typedef uint16_t __u16; typedef uint32_t __u32; typedef uint64_t __u64;
typedef int8_t __s8; typedef int16_t __s16; typedef int32_t __s32; typedef int64_t __s64;
typedef uint16_t __le16; typedef uint32_t __le32; typedef uint64_t __le64;
#define __packed __attribute__((packed))
#define __user
#define BIT(n) (1UL << (n))
#define BIT_ULL(n) (1ULL << (n))
"""
C_PRELUDE = "#define bool _Bool\n#define true 1\n#define false 0\n"


class Unit:
    """A translation unit of headers, parsed as C (or C++) for x86_64 Linux.
    Includes that cannot be found are replaced by empty files in [stub]."""

    def __init__(self, ci, headers, includes, stub, cpp=False):
        self.ci, self.includes, self.stub, self.cpp = ci, includes, stub, cpp
        self.src = PRELUDE + ("" if cpp else C_PRELUDE) + "".join(f'#include "{h}"\n' for h in headers)
        self.tu = self.parse(self.src)

    def parse(self, src):
        args = ["-x", "c++" if self.cpp else "c", "-target", "x86_64-unknown-linux-gnu", "-nostdinc",
                "-ferror-limit=0", "-I", str(self.stub)] + [f"-D{d}" for d in DEFINES] \
            + [f"-I{i}" for i in self.includes]
        index = self.ci.Index.create()
        for _ in range(100):
            tu = index.parse("probe.c", args=args, unsaved_files=[("probe.c", src)])
            missing = [m.group(1) for d in tu.diagnostics if (m := re.search(r"'([^']+)' file not found", d.spelling))]
            if not missing:
                return tu
            for m in missing:
                p = self.stub / m
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("")
        sys.exit("too many missing includes")

    def enums(self):
        return {c.spelling: c.enum_value for c in self.tu.cursor.walk_preorder()
                if c.kind == self.ci.CursorKind.ENUM_CONSTANT_DECL}

    def macros(self, names):
        """Evaluates macros through enumerators, a 32-bit half at a time."""
        probe = "".join(f"#ifdef {n}\nenum {{ __probe_hi_{n} = (int)((unsigned long long)({n}) >> 32), "
                        f"__probe_lo_{n} = (int)((unsigned long long)({n}) & 0xffffffffULL) }};\n#endif\n"
                        for n in names)
        tu = self.parse(self.src + probe)
        vals = {c.spelling: c.enum_value for c in tu.cursor.walk_preorder()
                if c.kind == self.ci.CursorKind.ENUM_CONSTANT_DECL and c.spelling.startswith("__probe_")}
        out = {}
        for n in names:
            hi, lo = vals.get(f"__probe_hi_{n}"), vals.get(f"__probe_lo_{n}")
            if hi is not None and lo is not None:
                out[n] = ((hi & 0xffffffff) << 32) | (lo & 0xffffffff)
        return out

    def constants(self, names):
        vals = {**self.enums(), **self.macros(names)}
        missing = [n for n in names if n not in vals]
        if missing:
            sys.exit(f"undefined constants: {missing}")
        return vals

    def shifts(self, names):
        """The shift of each argument macro [name(x)]: the first bit of [name(1)]."""
        probe = "".join(f"enum {{ __shift_{n} = (int)({n}(1)) }};\n" for n in names)
        tu = self.parse(self.src + probe)
        vals = {c.spelling[len("__shift_"):]: c.enum_value for c in tu.cursor.walk_preorder()
                if c.kind == self.ci.CursorKind.ENUM_CONSTANT_DECL and c.spelling.startswith("__shift_")}
        missing = [n for n in names if n not in vals]
        if missing:
            sys.exit(f"undefined argument macros: {missing}")
        return {n: (vals[n] & 0xffffffff).bit_length() - 1 for n in names}

    def struct(self, name):
        """The definition of the struct or union [name], or of the typedef [name]
        names."""
        kinds = (self.ci.CursorKind.STRUCT_DECL, self.ci.CursorKind.UNION_DECL)
        for c in self.tu.cursor.walk_preorder():
            if c.kind in kinds and c.spelling == name and c.is_definition():
                return c
        for c in self.tu.cursor.walk_preorder():
            if c.kind == self.ci.CursorKind.TYPEDEF_DECL and c.spelling == name:
                d = c.underlying_typedef_type.get_canonical().get_declaration()
                if d.kind in kinds and d.is_definition():
                    return d
        sys.exit(f"no struct {name}")


def layout(ci, cursor):
    """(sizeof, {path: (byte offset, bytes) or ("bits", bit offset, bits)}),
    nested fields joined with "__". An array's bytes are its element's."""
    fields = {}

    def walk(t, base, prefix):
        for f in t.get_fields():
            off = base + f.get_field_offsetof()
            ft = f.type.get_canonical()
            anonymous = not f.spelling or "(anonymous" in f.spelling
            path = prefix if anonymous else (prefix + "__" if prefix else "") + f.spelling
            if f.is_bitfield():
                fields[path] = ("bits", off, f.get_bitfield_width())
            elif ft.kind == ci.TypeKind.RECORD:
                if not anonymous:
                    fields[path] = (off // 8, ft.get_size())
                walk(ft, off, path)
            elif ft.kind in (ci.TypeKind.CONSTANTARRAY, ci.TypeKind.INCOMPLETEARRAY):
                fields[path] = (off // 8, ft.get_array_element_type().get_canonical().get_size())
            else:
                fields[path] = (off // 8, ft.get_size())

    walk(cursor.type.get_canonical(), 0, "")
    return cursor.type.get_size(), fields


def same(what, values):
    if len(set(values)) != 1:
        sys.exit(f"{what} differs: {values}")
    return values[0]

# Registers


def split_name(name):
    pos = next((i for i, c in enumerate(name) if c.isupper()), len(name))
    return name[:pos], name[pos:]


def gc_registers(amd, ver):
    """The registers of GC [ver]: {name: (offset, segment, [(field, lo, hi)])}.
    The VM registers of the GC's hub take its prefix, as regGCVM_CONTEXT0_CNTL."""
    base = amd / "include/asic_reg/gc" / f"gc_{'_'.join(map(str, ver))}"

    def normalize(reg):
        s = split_name(reg)
        return s[0] + "GC" + s[1] if s[1].startswith(("VM_", "MC_VM_")) else reg

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


def processors(llvm):
    """LLVM's AMDGCN processors by their [EF_AMDGPU_MACH] value, and the
    processors each generic one lists, from ELF.h and AMDGPUUsage.rst, which
    must agree."""
    header = (llvm / LLVM_FILES[1]).read_text()
    enums = {m.group(1): int(m.group(2), 0)
             for m in re.finditer(r"^\s*(E[A-Z0-9_]+)\s*=\s*(0x[0-9a-fA-F]+|\d+),", header, re.M)}
    missing = [c for c in ELF_CONSTANTS if c not in enums]
    if missing:
        sys.exit(f"ELF.h lacks {missing}")
    lines = (llvm / LLVM_FILES[2]).read_text().splitlines()
    code = re.compile(r"``([^`]+)``")
    machs = []
    for name, value, desc in rst_table(lines, "amdgpu-ef-amdgpu-mach-table"):
        m = code.fullmatch(name[0])
        if not (m and m.group(1).startswith("EF_AMDGPU_MACH_AMDGCN_")):
            continue
        if enums.get(m.group(1)) != int(value[0], 16):
            sys.exit(f"{m.group(1)}: ELF.h and AMDGPUUsage.rst differ")
        machs.append((int(value[0], 16), code.match(desc[0]).group(1)))
    names = {n for _, n in machs}
    generic = []
    for row in rst_table(lines, "amdgpu-generic-processor-table"):
        name = code.fullmatch(row[0][0]).group(1)
        members = [code.search(l).group(1) for l in row[2] if l.startswith("- ")]
        if name not in names or not set(members) <= names:
            sys.exit(f"{name}: a generic processor or member without an EF_AMDGPU_MACH value")
        generic.append((name, members))
    return {c: enums[c] for c in ELF_CONSTANTS}, machs, generic

# Emission


def ml_name(c):
    return c.lower()


def ml_int(v):
    return f"0x{v:x}" if v > 9 else str(v)


def ml_version(v):
    return "(%d, %d, %d)" % v


def ml_field(v):
    return f"({v[1]}, {v[2]})" if v[0] == "bits" else f"({v[0]}, {v[1]})"


def struct_module(out, module, size, fields, wanted):
    out.append(f"module {module} = struct")
    out.append(f"  let sizeof = {size}")
    for f in wanted:
        if f not in fields:
            sys.exit(f"{module} has no field {f}")
        out.append(f"  let {f} = {ml_field(fields[f])}")
    out.append("end")
    out.append("")


def generate(cache, pins, pin, outfile):
    import clang.cindex as ci
    amd, rocm, llvm, pal = sources(cache, pins, pin)
    stub = pathlib.Path(tempfile.mkdtemp(prefix="device-amd-abi-gen-stub-"))
    incs = [amd / "include", amd / "amdgpu", amd / "include/asic_reg"]
    out = ["(* Generated by gen/gen.py; do not edit. The inputs and the command that",
           "   regenerates this file are in gen/gen.py; their digests in gen/pins.json. *)", ""]

    # Registers
    out += ["(* Registers *)", "",
            "type register = {", "  name : string;", "  offset : int;", "  segment : int;",
            "  fields : (string * (int * int)) list;", "}", "",
            "(* The registers of each GC version, their fields as (name, (lowest bit,",
            "   highest bit)). *)", "let gc_registers = ["]
    pats = [re.compile(p) for p in GC_REGISTERS]
    for ver in GC_VERSIONS:
        out.append(f"  ( {ml_version(ver)}, [")
        for n, (off, seg, fields) in gc_registers(amd, ver).items():
            if any(p.fullmatch(n) for p in pats):
                fs = "; ".join(f"({json.dumps(f)}, ({lo}, {hi}))" for f, lo, hi in fields)
                out.append(f"      {{ name = {json.dumps(n)}; offset = {ml_int(off)}; segment = {seg}; fields = [ {fs} ] }};")
        out.append("    ] );")
    out += ["]", "", "(* The bases of the GC's register segments in PM4's register space, by the",
            "   GC major version from which they hold, from segment 0 to the last with a",
            "   base. *)", "let gc_bases = ["]
    for major, h in GC_BASES.items():
        segs = [f"GC_BASE__INST0_SEG{i}" for i in range(6)]
        bv = Unit(ci, [amd / "include" / h], incs, stub).macros(segs)
        bases = [bv.get(n, 0) for n in segs]
        while bases and bases[-1] == 0:  # the headers write 0 for no base
            bases.pop()
        if 0 in bases:
            sys.exit(f"{h}: a GC segment without a base before one with")
        out.append(f"  ({major}, [ " + "; ".join(ml_int(b) for b in bases) + " ]);")
    out += ["]", ""]

    # PM4
    soc = Unit(ci, [amd / "amdkfd/kfd_pm4_headers_ai.h", amd / "amdgpu/soc15d.h"], incs, stub)
    nv = Unit(ci, [amd / "amdkfd/kfd_pm4_headers_ai.h", amd / "amdgpu/nvd.h"], incs, stub)
    sv = soc.constants(PM4_CONSTANTS + PM4_SOC15_ONLY + PM4_SOC15_CONSTANTS)
    nvv = nv.constants(PM4_CONSTANTS + PM4_NV_CONSTANTS)
    out += ["(* PM4, the same in soc15d.h (GFX9) and nvd.h (GFX10 on) *)", ""]
    out += [f"let {ml_name(n)} = {ml_int(same(n, [sv[n], nvv[n]]))}" for n in PM4_CONSTANTS]
    ss = soc.shifts(sorted({g for g, _ in PM4_SHIFTS} | set(PM4_SOC15_SHIFTS)))
    ns = nv.shifts(sorted({n for _, n in PM4_SHIFTS} | set(PM4_NV_SHIFTS)))
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
    for f in PAL_FILES:
        size, fields = layout(ci, Unit(ci, [pal / f], [], stub, cpp=True).struct(WAIT_REG_MEM64))
        if size != 36 or any(fields.get(name) != at for name, at in WAIT_REG_MEM64_LAYOUT):
            sys.exit(f"{f}: {WAIT_REG_MEM64} is not laid out as the encoder writes it")
    if ss["WAIT_REG_MEM_FUNCTION"] != 0 or ss["WAIT_REG_MEM_MEM_SPACE"] != 4:
        sys.exit(f"{WAIT_REG_MEM64}'s control word differs from WAIT_REG_MEM's")

    # Events and thread traces
    socs = [(rocm / "projects/aqlprofile/linux" / f).read_text() for f in ("vega10_enum.h", "soc21_enum.h", "soc24_enum.h")]

    def soc_enum(n):
        found = [int(m.group(1), 0) for t in socs if (m := re.search(rf"^\s*{n}\s*=\s*(0x[0-9a-fA-F]+|\d+)", t, re.M))]
        if not found:
            sys.exit(f"no SOC enumeration defines {n}")
        return same(n, found)
    out += ["(* Events and thread trace values, the same in each SOC enumeration that",
            "   defines them *)", ""]
    out += [f"let {ml_name(n)} = {ml_int(soc_enum(n))}" for n in SOC_EVENTS + SOC_TRACE]
    out.append("")

    # SDMA
    out += ["(* SDMA: (mask, shift) for a field *)", ""]
    want = SDMA_OPS + [f"{n}_{k}" for n in SDMA_FIELDS for k in ("mask", "shift")]
    svals = {ver: Unit(ci, [amd / "amdgpu" / f"{h}.h"], incs, stub).macros(want) for ver, h in SDMA_PKT.items()}
    for n in SDMA_OPS:
        out.append(f"let {ml_name(n)} = {ml_int(same(n, [v[n] for v in svals.values()]))}")
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
    hu = Unit(ci, [rocm / RUNTIME / "inc/hsa.h"], [rocm / RUNTIME / "inc"], stub)
    hv = hu.constants(HSA_CONSTANTS)
    out += [f"let {ml_name(n)} = {ml_int(hv[n])}" for n in HSA_CONSTANTS]
    out.append("")
    size, fields = layout(ci, hu.struct("hsa_kernel_dispatch_packet_t"))
    struct_module(out, "Dispatch", size, fields, DISPATCH_FIELDS)

    # Scratch buffer descriptors
    ru = Unit(ci, [rocm / RUNTIME / "core/inc/registers.h"], [rocm / RUNTIME / "inc"], stub)
    rv = ru.constants(SQ_CONSTANTS)
    out += ["(* The scratch buffer descriptor's words 1 and 3, by GC major: each field", "   as (bit offset, bits). *)", ""]
    out += ["type sq_buf_rsrc = {"]
    out += [f"  {ml_name(f) if f != 'TYPE' else 'type_'} : int * int;" for f in SQ_WORD1 + SQ_WORD3]
    out += [f"  {ml_name(f)} : (int * int) option;" for f in SQ_WORD3_OPTIONAL]
    out += ["}", "", "let sq_buf_rsrc = ["]
    for major, (w1, w3) in SQ_BUF_RSRC.items():
        def bits(union):
            _, fields = layout(ci, ru.struct(union))
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
    lu = Unit(ci, [llvm / LLVM_FILES[0]], [], stub, cpp=True)
    out += ["(* The kernel descriptor: each field is (byte offset, bytes). *)"]
    size, fields = layout(ci, lu.struct("kernel_descriptor_t"))
    struct_module(out, "Kernel_descriptor", size, fields, KD_FIELDS)
    ku = Unit(ci, [rocm / RUNTIME / "inc/amd_hsa_kernel_code.h"], [rocm / RUNTIME / "inc"], stub)
    kv = ku.constants(KD_CONSTANTS)
    out.append("(* Its code properties. *)")
    out += [f"let {ml_name(c)} = {ml_int(kv[c])}" for c in KD_CONSTANTS]
    elf, machs, generic = processors(llvm)
    out += ["", "(* The ELF header of a code object. *)"]
    out += [f"let {ml_name(c)} = {ml_int(v)}" for c, v in elf.items()]
    out += ["", "(* LLVM's AMDGCN processors, by their EF_AMDGPU_MACH value. *)", "let processors = ["]
    out += [f"  ({ml_int(v)}, {json.dumps(n)});" for v, n in machs]
    out += ["]", "", "(* The generic processors, and the processors that run their code objects. *)", "let generic = ["]
    out += [f"  ({json.dumps(n)}, [ " + "; ".join(json.dumps(m) for m in ms) + " ]);" for n, ms in generic]
    out.append("]")
    outfile.write_text("\n".join(out) + "\n")

# Command line


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=pathlib.Path, default=pathlib.Path.home() / ".cache/raven/amd-gen")
    ap.add_argument("--check", action="store_true", help="fail if the committed file differs")
    ap.add_argument("--pin", action="store_true", help="record the digests of inputs not yet pinned")
    args = ap.parse_args()
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    pins_file = HERE / "pins.json"
    pins = json.loads(pins_file.read_text()) if pins_file.exists() else {}
    if args.check:
        with tempfile.TemporaryDirectory() as d:
            generated = pathlib.Path(d) / OUT.name
            generate(cache, pins, False, generated)
            if generated.read_text() != OUT.read_text():
                sys.exit(f"{OUT.name} differs from what gen.py generates")
        print("up to date")
        return
    generate(cache, pins, args.pin, OUT)
    if args.pin:
        pins = {u: d for u, d in pins.items() if u in USED}
        pins_file.write_text(json.dumps(pins, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
