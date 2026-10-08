# /// script
# requires-python = ">=3.10"
# ///
"""Generates defs.ml, the tables of device_nv_abi: the methods of the host,
compute and copy classes, the fields of their words and the values they take,
the layout of a channel's ring entries and method headers, and the layouts of
launch descriptors, versions 3 and 5.

Run from the worktree root:

  uv run dev/device/nv/abi/gen/gen.py
  uv run dev/device/nv/abi/gen/gen.py --check
  uv run dev/device/nv/abi/gen/gen.py --excerpt [--check]

The inputs are excerpts of NVIDIA's headers, in headers/: each is a header's
licence notice and the #define lines this script reads, verbatim and in the
header's order. --excerpt makes them from the upstream headers, each pinned in
pins.json by URL and SHA-256 and checked against its pin; downloads are kept
in --cache, and --pin records the digests of headers not yet pinned.
Generating reads the excerpts alone, offline. --check generates
into memory and fails if a committed file differs.

Every value is NVIDIA's. Where two classes the library supports both define a
name (the copy classes 0xc7b5 and 0xc9b5, the launch descriptors of 0xc7c0 and
0xc9c0), the script fails unless they agree. Every table is a literal, which
the compiler lays out as static data: nothing is built when a program starts.
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

KERNEL = "https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/570.144/"
DOC = "https://raw.githubusercontent.com/NVIDIA/open-gpu-doc/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/compute/"
CLASSES = KERNEL + "src/common/sdk/nvidia/inc/class/"
UVM = KERNEL + "kernel-open/nvidia-uvm/"

# Each excerpt and the header it is cut from.
SOURCES = {
    "clc56f.h": UVM + "clc56f.h",  # AMPERE_CHANNEL_GPFIFO_A: the host
    "clc7c0.h": CLASSES + "clc7c0.h",  # AMPERE_COMPUTE_B
    "clc9c0.h": CLASSES + "clc9c0.h",  # ADA_COMPUTE_A
    "clcec0.h": CLASSES + "clcec0.h",  # BLACKWELL_COMPUTE_B
    "clc7b5.h": UVM + "clc7b5.h",  # AMPERE_DMA_COPY_B
    "clc9b5.h": UVM + "clc9b5.h",  # BLACKWELL_DMA_COPY_A
    "clc7c0qmd.h": DOC + "clc7c0qmd.h",
    "clc9c0qmd.h": DOC + "clc9c0qmd.h",
    "clcec0qmd.h": DOC + "clcec0qmd.h",
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
    "NVC7C0_SEND_PCAS_A", "NVC7C0_SEND_SIGNALING_PCAS2_B",
    "NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_PREFETCH_SCHEDULE",
]
COMPUTE_FIELDS = [
    "NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI_INSTRUCTION", "NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI_GLOBAL_DATA",
    "NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI_CONSTANT",
    "NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION",
]
COPY, LATER_COPY = "clc7b5.h", "clc9b5.h"
COPY_CONSTANTS = [
    "NVC7B5_SET_SEMAPHORE_A", "NVC7B5_LAUNCH_DMA", "NVC7B5_OFFSET_IN_UPPER", "NVC7B5_LINE_LENGTH_IN",
    "NVC7B5_LAUNCH_DMA_DATA_TRANSFER_TYPE_NON_PIPELINED", "NVC7B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT_PITCH",
    "NVC7B5_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH", "NVC7B5_LAUNCH_DMA_FLUSH_ENABLE_TRUE",
    "NVC7B5_LAUNCH_DMA_FLUSH_TYPE_SYS", "NVC7B5_LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_ONE_WORD_SEMAPHORE",
    "NVC7B5_LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_FOUR_WORD_SEMAPHORE",
    "NVC7B5_LAUNCH_DMA_SEMAPHORE_PAYLOAD_SIZE_TWO_WORD",
]
COPY_FIELDS = [
    "NVC7B5_LAUNCH_DMA_DATA_TRANSFER_TYPE", "NVC7B5_LAUNCH_DMA_FLUSH_ENABLE", "NVC7B5_LAUNCH_DMA_FLUSH_TYPE",
    "NVC7B5_LAUNCH_DMA_SEMAPHORE_TYPE", "NVC7B5_LAUNCH_DMA_SEMAPHORE_PAYLOAD_SIZE",
    "NVC7B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT", "NVC7B5_LAUNCH_DMA_DST_MEMORY_LAYOUT",
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
    (COPY, ["NVC7B5_OFFSET_IN_UPPER", "NVC7B5_OFFSET_IN_LOWER", "NVC7B5_OFFSET_OUT_UPPER", "NVC7B5_OFFSET_OUT_LOWER"]),
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
RELEASES = 2
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

# Reading headers

DEFINE = re.compile(r"^#define[ \t]+(\w+(?:\(i\))?)[ \t]+(.*?)[ \t]*(?://.*)?$", re.M)


def defines(text):
    """The #define lines of [text], as {name: body}; an indexed name keeps its (i)."""
    return {m.group(1): m.group(2) for m in DEFINE.finditer(text)}


def number(body):
    m = re.fullmatch(r"\(?\s*(0[xX][0-9A-Fa-f]+|\d+)\s*\)?", body)
    if not m:
        sys.exit(f"not a number: {body}")
    return int(m.group(1), 0)


def bits(body):
    """A field "hi:lo" as (lowest bit, bits)."""
    m = re.fullmatch(r"(\d+):(\d+)", body)
    if not m:
        sys.exit(f"not a bit field: {body}")
    hi, lo = int(m.group(1)), int(m.group(2))
    return (lo, hi - lo + 1)


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


def lookup(defs, name, header):
    if name not in defs:
        sys.exit(f"{header}: no {name}")
    return defs[name]


# The names each excerpt holds

def wanted():
    """The names read from each header, by header."""
    w = {h: set() for h in SOURCES}
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
    return w


def later(name):
    return name.replace("NVC7B5_", "NVC9B5_")


def qmd_names(v):
    """The names version [v] (0 for 3, 1 for 5) reads, without its prefix."""
    names = [f[1 + v] for f in QMD_FIELDS + QMD_BANK_FIELDS + QMD_VALUES if f[1 + v]]
    for f in QMD_RELEASE_FIELDS:
        names += [f[1 + v].format(k=k) for k in range(RELEASES)]
    return names


def last_field(text, prefix):
    """The name of [prefix]'s field that ends last, which sets the descriptor's size."""
    fields = [(mw(b)[0] + mw(b)[1], n) for n, b in defines(text).items()
              if n.startswith(prefix + "_") and b.startswith("MW(") and "(i)" not in n]
    return max(fields)[1]


# Excerpts

def licence(text):
    """The header's first comment, its licence notice."""
    end = text.index("*/") + 2
    return text[:end] + "\n"


def excerpt(name, text):
    keep = wanted()[name]
    for h, prefix, _ in QMD + [(*QMD_SAME, 3)]:
        if h == name:
            keep = keep | {last_field(text, prefix)}
    lines = [m.group(0) for m in DEFINE.finditer(text) if m.group(1) in keep]
    found = {DEFINE.match(line).group(1) for line in lines}
    if found != keep:
        sys.exit(f"{name}: no {sorted(keep - found)}")
    return licence(text) + "\n" + "\n".join(lines) + "\n"


def fetch(url, cache, pins, pin):
    path = cache / hashlib.sha256(url.encode()).hexdigest()[:16]
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            path.write_bytes(r.read())
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if url not in pins:
        if not pin:
            sys.exit(f"{url} is not pinned; run with --pin to record {digest}")
        pins[url] = digest
    if pins[url] != digest:
        sys.exit(f"{url}: SHA-256 {digest}, pinned {pins[url]}")
    return data.decode()


# Generation

def ml_int(n):
    return str(n) if n < 10 else hex(n)


def ml_field(f):
    return f"{{ lo = {ml_int(f[0])}; bits = {ml_int(f[1])} }}"


def ml_ident(name):
    return name.lower()


def ml_option(f):
    return f"Some {ml_field(f)}" if f else "None"


def generate():
    texts = {h: (HEADERS / h).read_text() for h in SOURCES}
    defs = {h: defines(t) for h, t in texts.items()}
    out = []

    def let(name, value):
        out.append(f"let {ml_ident(name)} = {value}")

    out.append(qmd_types())
    out.append("\n(* Classes *)\n")
    for h, n in CLASS_IDS:
        let(n, ml_int(number(lookup(defs[h], n, h))))

    for h, names in RUNS:
        first = number(lookup(defs[h], names[0], h))
        for k, n in enumerate(names):
            if number(lookup(defs[h], n, h)) != first + 4 * k:
                sys.exit(f"{h}: {n} does not follow {names[k - 1]}")

    for name in COPY_CONSTANTS + COPY_FIELDS + [n for h, ns in RUNS if h == COPY for n in ns]:
        if defs[COPY][name] != lookup(defs[LATER_COPY], later(name), LATER_COPY):
            sys.exit(f"{name} differs in {LATER_COPY}")

    sections = [("Host methods (clc56f.h)", HOST, HOST_CONSTANTS, HOST_FIELDS),
                ("Compute methods (clc7c0.h)", COMPUTE, COMPUTE_CONSTANTS, COMPUTE_FIELDS),
                ("Copy methods (clc7b5.h)", COPY, COPY_CONSTANTS, COPY_FIELDS)]
    for title, h, constants, fields in sections:
        out.append(f"\n(* {title} *)\n")
        for n in constants:
            let(n, ml_int(number(lookup(defs[h], n, h))))
        for n in fields:
            let(n, ml_field(bits(lookup(defs[h], n, h))))

    out.append("\n(* Launch descriptors *)\n")
    same_h, same_prefix = QMD_SAME
    for n in qmd_names(0):
        a = lookup(defs[QMD[0][0]], f"{QMD[0][1]}_{n}", QMD[0][0])
        b = lookup(defs[same_h], f"{same_prefix}_{n}", same_h)
        if a != b:
            sys.exit(f"{n} differs in {same_h}")
    for name, *versions in QMD_VALUES:
        values = {number(lookup(defs[h], f"{prefix}_{n}", h))
                  for (h, prefix, _), n in zip(QMD, versions) if n}
        if len(values) != 1:
            sys.exit(f"{name} differs between versions")
        let(name, ml_int(values.pop()))
    for v, (h, prefix, version) in enumerate(QMD):
        d = defs[h]

        def field(n, i=None):
            return mw(lookup(d, f"{prefix}_{n}", h), i)

        last = last_field(texts[h], prefix)
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
        for k in range(RELEASES):
            out.append(f"      {'(' if k == 0 else ','} {{")
            for f, *names in QMD_RELEASE_FIELDS:
                out.append(f"          {f} = {ml_field(field(names[v].format(k=k), k))};")
            out.append("        }")
        out.append("      );")
        out.append("  }")
    return header(texts) + "\n".join(out) + "\n"


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


def header(texts):
    owners = []
    for h in SOURCES:
        m = re.search(r"Copyright \(c\) ([^\n]*?NVIDIA[^\n.]*)", texts[h])
        owners.append(f"   {h}: Copyright (c) {m.group(1).strip()}")
    notice = (
        "   Permission is hereby granted, free of charge, to any person obtaining a\n"
        "   copy of this software and associated documentation files (the \"Software\"),\n"
        "   to deal in the Software without restriction, including without limitation\n"
        "   the rights to use, copy, modify, merge, publish, distribute, sublicense,\n"
        "   and/or sell copies of the Software, and to permit persons to whom the\n"
        "   Software is furnished to do so, subject to the following conditions:\n\n"
        "   The above copyright notice and this permission notice shall be included in\n"
        "   all copies or substantial portions of the Software.\n\n"
        "   THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n"
        "   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
        "   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL\n"
        "   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
        "   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING\n"
        "   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER\n"
        "   DEALINGS IN THE SOFTWARE."
    )
    return (
        "(*---------------------------------------------------------------------------\n"
        "  Copyright (c) 2026 The Raven authors. All rights reserved.\n"
        "  SPDX-License-Identifier: ISC\n"
        "  ---------------------------------------------------------------------------*)\n\n"
        "(* Generated by gen/gen.py from the excerpts in gen/headers; do not edit.\n"
        "   The command that regenerates this file is in gen/gen.py.\n\n"
        "   The values are NVIDIA's, copied from its headers under the MIT licence:\n\n"
        + "\n".join(owners) + "\n\n" + notice + " *)\n\n"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--check", action="store_true", help="fail if a committed file differs")
    p.add_argument("--excerpt", action="store_true", help="make the excerpts from the pinned headers")
    p.add_argument("--pin", action="store_true", help="record the digests of headers not yet pinned")
    p.add_argument("--cache", type=pathlib.Path, default=pathlib.Path.home() / ".cache/raven/nv-gen")
    a = p.parse_args()
    if a.excerpt:
        pins = json.loads(PINS.read_text()) if PINS.exists() else {}
        files = {HEADERS / h: excerpt(h, fetch(url, a.cache, pins, a.pin)) for h, url in SOURCES.items()}
        files[PINS] = json.dumps(dict(sorted(pins.items())), indent=1) + "\n"
    else:
        files = {OUT: generate()}
    stale = [f for f, text in files.items() if not f.exists() or f.read_text() != text]
    if a.check:
        if stale:
            sys.exit("stale: " + ", ".join(str(f.relative_to(HERE.parent)) for f in stale))
        return
    for f in stale:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(files[f])


if __name__ == "__main__":
    main()
