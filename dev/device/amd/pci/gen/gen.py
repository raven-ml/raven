# /// script
# requires-python = ">=3.10"
# ///
"""Generates defs.ml, the tables of device_amd_pci: the layouts of the
discovery table and its blocks' hardware IDs, the layouts of firmware
images' headers and the types the security processor loads them as, and the
pinned firmware images.

Run from the worktree root:

  uv run dev/device/amd/pci/gen/gen.py
  uv run dev/device/amd/pci/gen/gen.py --check
  uv run dev/device/amd/pci/gen/gen.py --excerpt [--check]

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
}

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
ENUMS = {"amdgpu_ucode.h": ["psp_fw_type"], "psp_gfx_if.h": ["psp_gfx_fw_type"]}

# The blocks a boot programs, by their hardware IDs' names.
HWIDS = ["GC", "SDMA0", "SDMA1", "SDMA2", "SDMA3", "MP0", "MP1", "MMHUB", "OSSSYS", "NBIF", "HDP"]

# The configuration the headers are read under: a little-endian processor.
DEFINED = {"LITTLEENDIAN_CPU"}

# Reading C headers

DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(\w+)(\([\w, ]*\))?[ \t]*(.*?)[ \t]*(?:/\*.*?\*/|//.*)?[ \t]*$", re.M)
ENUMERATOR = re.compile(r"^[ \t]*(\w+)[ \t]*(?:=[ \t]*([^,/\n]+?))?[ \t]*,?[ \t]*(?:/\*.*|//.*)?$", re.M)


def defines(text):
    """{name: body} of [text]'s #define lines without parameters."""
    return {m.group(1): m.group(3) for m in DEFINE.finditer(text) if m.group(2) is None}


def evaluate(expr, names):
    """The integer of the C constant expression [expr]: numbers, names of
    [names], and integer operators."""
    e = re.sub(r"\b(0[xX][0-9a-fA-F]+|\d+)[uUlL]*\b", r"\1", expr)

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
            nxt = evaluate(value.strip(), {}) if value else nxt
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


def packed_layout(text, name):
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
                i = after + 1
                continue
            j = i
            while tokens[j] != ";":
                j += 1
            decl = tokens[i:j]
            i = j + 1
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
            if ftype in SCALARS:
                fbytes = SCALARS[ftype]
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
    elif name == "psp_gfx_if.h":
        keep |= enum_blocks(text, ENUMS[name])
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
    out.append(f"(* The pinned images of linux-firmware at {FIRMWARE_COMMIT[:12]}: path, BLAKE2b-256")
    out.append("   digest and URL. *)")
    out.append("let pinned = [")
    for row in h[FIRMWARE].splitlines():
        if row.startswith("#"):
            continue
        path, digest, url = row.split("\t")
        out.append(f"  ({json.dumps(path)}, {json.dumps(digest)}, {json.dumps(url)});")
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
