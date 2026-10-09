# /// script
# requires-python = ">=3.10"
# ///
"""Generates defs.ml of rig.mlx5.abi, rig.mlx5 and rig.mlx5.uverbs: the
layouts of ConnectX's work entries, completions and doorbell records and the
rules of its access regions (abi/defs.ml), the driver data of the mlx5
driver's objects (defs.ml), and the kernel's verbs ioctl, its commands and
their answers (uverbs/defs.ml).

Run from the worktree root:

  uv run dev/rig/lib/mlx5/gen/gen.py
  uv run dev/rig/lib/mlx5/gen/gen.py --check
  uv run dev/rig/lib/mlx5/gen/gen.py --excerpt [--check]

The inputs are excerpts of rdma-core's and Linux's headers, in headers/: each
is a header's licence notice, the URL it is cut from, and the lines this script
reads, verbatim and in the header's order (a #define, a whole enumeration, a
whole structure). SOURCES names the header each comes from, at its version.
Every header is under the GPL version 2 or the OpenIB.org BSD licence, at the
user's choice; raven takes the BSD licence, whose notice each defs.ml carries.
--excerpt makes the excerpts from the upstream headers, each pinned in
pins.json by URL and SHA-256 and checked against its pin; downloads are kept
in --cache, and --pin records the digests of headers not yet pinned.
Generating reads the excerpts alone, offline. --check generates into memory
and fails if a committed file differs.

A structure is laid out as the C compilers of Linux's 64-bit hosts lay it out:
each member at the next multiple of its alignment, its size's for integers, 8
for __aligned_u64, 1 in a packed structure.
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
LIB = HERE.parent

# Sources, each at the commit or tag in its URL

RDMA_CORE = "https://raw.githubusercontent.com/linux-rdma/rdma-core/v56.0/"
LINUX = "https://raw.githubusercontent.com/torvalds/linux/v6.12/"

SOURCES = {
    "mlx5dv.h": RDMA_CORE + "providers/mlx5/mlx5dv.h",  # work entries, completions
    "mlx5.h": RDMA_CORE + "providers/mlx5/mlx5.h",  # doorbell records, access regions
    "wqe.h": RDMA_CORE + "providers/mlx5/wqe.h",  # the inline segment
    "verbs.h": RDMA_CORE + "libibverbs/verbs.h",  # queue pair states, events
    "device.h": LINUX + "include/linux/mlx5/device.h",  # doorbell registers
    "mlx5_ib.h": LINUX + "drivers/infiniband/hw/mlx5/mlx5_ib.h",
    "mlx5-abi.h": LINUX + "include/uapi/rdma/mlx5-abi.h",  # driver data
    "rdma_user_ioctl_cmds.h": LINUX + "include/uapi/rdma/rdma_user_ioctl_cmds.h",
    "ib_user_ioctl_cmds.h": LINUX + "include/uapi/rdma/ib_user_ioctl_cmds.h",
    "ib_user_ioctl_verbs.h": LINUX + "include/uapi/rdma/ib_user_ioctl_verbs.h",
    "ib_user_verbs.h": LINUX + "include/uapi/rdma/ib_user_verbs.h",
}

# What each library reads: (header, C name) of constants, and (header, C name,
# OCaml module) of structures.

ABI_CONSTANTS = [
    ("mlx5dv.h", n) for n in [
        "MLX5_SEND_WQE_BB", "MLX5_OPCODE_RDMA_WRITE", "MLX5_OPCODE_RDMA_READ", "MLX5_WQE_CTRL_CQ_UPDATE",
        "MLX5_WQE_CTRL_FENCE", "MLX5_INLINE_SEG", "MLX5_RCV_DBR", "MLX5_SND_DBR", "MLX5_CQE_OWNER_MASK",
        "MLX5_CQE_REQ", "MLX5_CQE_REQ_ERR", "MLX5_CQE_RESP_ERR", "MLX5_CQE_INVALID",
        "MLX5_CQE_SYNDROME_LOCAL_LENGTH_ERR", "MLX5_CQE_SYNDROME_LOCAL_QP_OP_ERR",
        "MLX5_CQE_SYNDROME_LOCAL_PROT_ERR", "MLX5_CQE_SYNDROME_WR_FLUSH_ERR", "MLX5_CQE_SYNDROME_MW_BIND_ERR",
        "MLX5_CQE_SYNDROME_BAD_RESP_ERR", "MLX5_CQE_SYNDROME_LOCAL_ACCESS_ERR",
        "MLX5_CQE_SYNDROME_REMOTE_INVAL_REQ_ERR", "MLX5_CQE_SYNDROME_REMOTE_ACCESS_ERR",
        "MLX5_CQE_SYNDROME_REMOTE_OP_ERR", "MLX5_CQE_SYNDROME_TRANSPORT_RETRY_EXC_ERR",
        "MLX5_CQE_SYNDROME_RNR_RETRY_EXC_ERR", "MLX5_CQE_SYNDROME_REMOTE_ABORTED_ERR", "MLX5_CQ_DB_REQ_NOT",
        "MLX5_CQ_DOORBELL"]
] + [
    ("mlx5.h", n) for n in ["MLX5_BF_OFFSET", "MLX5_CQ_SET_CI", "MLX5_CQ_ARM_DB", "MLX5_IB_MMAP_CMD_SHIFT",
                            "MLX5_ADAPTER_PAGE_SIZE"]
] + [
    ("device.h", "MLX5_BFREGS_PER_UAR"), ("device.h", "MLX5_NON_FP_BFREGS_PER_UAR"),
    ("mlx5-abi.h", "MLX5_IB_MMAP_NC_PAGE"),
]
ABI_STRUCTS = [
    ("mlx5dv.h", "mlx5_wqe_ctrl_seg", "Ctrl_seg"), ("mlx5dv.h", "mlx5_wqe_raddr_seg", "Raddr_seg"),
    ("mlx5dv.h", "mlx5_wqe_data_seg", "Data_seg"), ("wqe.h", "mlx5_wqe_inline_seg", "Inline_seg"),
    ("mlx5dv.h", "mlx5_cqe64", "Cqe64"), ("mlx5dv.h", "mlx5_err_cqe", "Err_cqe"),
]

MLX5_CONSTANTS = [
    ("mlx5-abi.h", "MLX5_LIB_CAP_4K_UAR"), ("device.h", "MLX5_NON_FP_BFREGS_PER_UAR"),
    ("mlx5.h", "MLX5_CQE_VERSION_V0"), ("mlx5_ib.h", "MLX5_IB_DEFAULT_UIDX"),
]
MLX5_STRUCTS = [
    ("mlx5-abi.h", "mlx5_ib_alloc_ucontext_req_v2", "Ucontext_req"),
    ("mlx5-abi.h", "mlx5_ib_alloc_ucontext_resp", "Ucontext_resp"),
    ("mlx5-abi.h", "mlx5_ib_create_cq", "Create_cq"), ("mlx5-abi.h", "mlx5_ib_create_cq_resp", "Create_cq_resp"),
    ("mlx5-abi.h", "mlx5_ib_create_qp", "Create_qp"), ("mlx5-abi.h", "mlx5_ib_create_qp_resp", "Create_qp_resp"),
]
# The structures the excerpts keep for a kept one's members.
NESTED = {"mlx5-abi.h": ["mlx5_ib_create_qp_dci_streams"], "ib_user_verbs.h": ["ib_uverbs_qp_dest"]}

UVERBS_CONSTANTS = [
    ("rdma_user_ioctl_cmds.h", n) for n in ["RDMA_IOCTL_MAGIC", "UVERBS_ATTR_F_MANDATORY"]
] + [
    ("ib_user_ioctl_cmds.h", n) for n in [
        "UVERBS_OBJECT_DEVICE", "UVERBS_OBJECT_MR", "UVERBS_METHOD_INVOKE_WRITE", "UVERBS_ATTR_CORE_IN",
        "UVERBS_ATTR_CORE_OUT", "UVERBS_ATTR_WRITE_CMD", "UVERBS_ATTR_UHW_IN", "UVERBS_ATTR_UHW_OUT",
        "UVERBS_METHOD_REG_DMABUF_MR", "UVERBS_ATTR_REG_DMABUF_MR_HANDLE", "UVERBS_ATTR_REG_DMABUF_MR_PD_HANDLE",
        "UVERBS_ATTR_REG_DMABUF_MR_OFFSET", "UVERBS_ATTR_REG_DMABUF_MR_LENGTH", "UVERBS_ATTR_REG_DMABUF_MR_IOVA",
        "UVERBS_ATTR_REG_DMABUF_MR_FD", "UVERBS_ATTR_REG_DMABUF_MR_ACCESS_FLAGS",
        "UVERBS_ATTR_REG_DMABUF_MR_RESP_LKEY", "UVERBS_ATTR_REG_DMABUF_MR_RESP_RKEY"]
] + [
    ("ib_user_ioctl_verbs.h", n) for n in [
        "RDMA_DRIVER_MLX5", "IB_UVERBS_ACCESS_LOCAL_WRITE", "IB_UVERBS_ACCESS_REMOTE_WRITE",
        "IB_UVERBS_ACCESS_REMOTE_READ", "IB_UVERBS_ACCESS_RELAXED_ORDERING", "IB_UVERBS_QPT_RC"]
] + [
    ("ib_user_verbs.h", "IB_USER_VERBS_CMD_" + n) for n in [
        "GET_CONTEXT", "QUERY_DEVICE", "QUERY_PORT", "ALLOC_PD", "REG_MR", "DEREG_MR", "CREATE_COMP_CHANNEL",
        "CREATE_CQ", "DESTROY_CQ", "CREATE_QP", "MODIFY_QP", "DESTROY_QP"]
] + [
    ("verbs.h", n) for n in [
        "IBV_QPS_INIT", "IBV_QPS_RTR", "IBV_QPS_RTS", "IBV_QP_STATE", "IBV_QP_ACCESS_FLAGS", "IBV_QP_PKEY_INDEX",
        "IBV_QP_PORT", "IBV_QP_AV", "IBV_QP_PATH_MTU", "IBV_QP_TIMEOUT", "IBV_QP_RETRY_CNT", "IBV_QP_RNR_RETRY",
        "IBV_QP_RQ_PSN", "IBV_QP_MAX_QP_RD_ATOMIC", "IBV_QP_MIN_RNR_TIMER", "IBV_QP_SQ_PSN",
        "IBV_QP_MAX_DEST_RD_ATOMIC", "IBV_QP_DEST_QPN", "IBV_MTU_256", "IBV_MTU_4096", "IBV_PORT_ACTIVE",
        "IBV_LINK_LAYER_INFINIBAND", "IBV_LINK_LAYER_ETHERNET", "IBV_EVENT_CQ_ERR", "IBV_EVENT_QP_FATAL",
        "IBV_EVENT_QP_REQ_ERR", "IBV_EVENT_QP_ACCESS_ERR", "IBV_EVENT_DEVICE_FATAL", "IBV_EVENT_PORT_ACTIVE",
        "IBV_EVENT_PORT_ERR"]
]
UVERBS_STRUCTS = [
    ("rdma_user_ioctl_cmds.h", "ib_uverbs_attr", "Attr"), ("rdma_user_ioctl_cmds.h", "ib_uverbs_ioctl_hdr", "Ioctl_hdr"),
] + [
    ("ib_user_verbs.h", "ib_uverbs_" + n, n.capitalize()) for n in [
        "get_context", "get_context_resp", "query_device", "query_device_resp", "query_port", "query_port_resp",
        "alloc_pd", "alloc_pd_resp", "reg_mr", "reg_mr_resp", "dereg_mr", "create_comp_channel",
        "create_comp_channel_resp", "create_cq", "create_cq_resp", "destroy_cq", "destroy_cq_resp", "create_qp",
        "create_qp_resp", "qp_dest", "modify_qp", "destroy_qp", "destroy_qp_resp", "async_event_desc",
        "comp_event_desc"]
]

OUTPUTS = {
    LIB / "abi" / "defs.ml": (ABI_CONSTANTS, ABI_STRUCTS),
    LIB / "defs.ml": (MLX5_CONSTANTS, MLX5_STRUCTS),
    LIB / "uverbs" / "defs.ml": (UVERBS_CONSTANTS, UVERBS_STRUCTS),
}

# The BSD licence every header offers, which defs.ml carries with the headers'
# copyright lines.
OPENIB_BSD = """\
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 - Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

 - Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

# Reading C headers

DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(\w+)[ \t]+(.*?)[ \t]*(?:/\*.*?\*/|//.*)?[ \t]*$", re.M)
ENUM = re.compile(r"^[ \t]*enum\b[ \t]*\w*[ \t]*\{", re.M)
STRUCT = re.compile(r"^[ \t]*struct[ \t]+(\w+)[ \t]*\{", re.M)


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def block_end(text, start):
    """The index past the [;] that ends the braced block opened at or after
    [start]."""
    i, depth = text.index("{", start), 0
    while True:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text.index(";", i) + 1
        i += 1


def enums(text):
    """{name: expression} of [text]'s enumerators, an implicit one as its
    predecessor's plus one."""
    text = strip_comments(text)
    out = {}
    for m in ENUM.finditer(text):
        body = text[text.index("{", m.start()) + 1:block_end(text, m.start())].rsplit("}", 1)[0]
        prev = None
        for item in body.split(","):
            item = item.strip()
            if not item:
                continue
            name, _, expr = (s.strip() for s in item.partition("="))
            expr = expr or (f"({prev}) + 1" if prev else "0")
            out[name] = expr
            prev = name
    return out


def defines(text):
    return {m.group(1): m.group(2) for m in DEFINE.finditer(strip_comments(text)) if m.group(2)}


def evaluate(expr, names, seen=()):
    """The integer of the C constant expression [expr] over [names]."""
    e = re.sub(r"\(\s*(unsigned|unsigned int|unsigned long|__u64|__u32|u64|u32|int)\s*\)", "", expr)
    e = re.sub(r"\b(0[xX][0-9a-fA-F]+|\d+)[uUlL]*\b", r"\1", e)

    def name(m):
        n = m.group(0)
        if n.startswith(("0x", "0X")) or n.isdigit():
            return n
        if n not in names or n in seen:
            sys.exit(f"{expr}: {n} is not defined")
        return str(evaluate(names[n], names, seen + (n,)))
    e = re.sub(r"\b[A-Za-z_]\w*\b", name, e)
    if not re.fullmatch(r"[0-9a-fA-FxX ()|&<>~+\-*\t]*", e):
        sys.exit(f"{expr}: not a constant expression")
    return eval(e, {"__builtins__": {}})


def constants(text, wanted):
    names = {**enums(text), **defines(text)}
    missing = [n for n in wanted if n not in names]
    if missing:
        sys.exit(f"no {missing}")
    return {n: evaluate(names[n], names) for n in wanted}


# Structures

SIZES = {
    **dict.fromkeys(["__u8", "__s8", "uint8_t", "int8_t", "u8", "char"], 1),
    **dict.fromkeys(["__u16", "__s16", "__be16", "uint16_t", "int16_t"], 2),
    **dict.fromkeys(["__u32", "__s32", "__be32", "uint32_t", "int32_t", "int"], 4),
    **dict.fromkeys(["__u64", "__s64", "__be64", "uint64_t", "int64_t", "__aligned_u64"], 8),
}
TOKEN = re.compile(r"\w+|[{}\[\];(),:*]")


def tokens(text):
    return TOKEN.findall(strip_comments(text))


def struct_tokens(text, name):
    m = re.search(rf"^[ \t]*struct[ \t]+{name}[ \t]*\{{", text, re.M)
    if not m:
        sys.exit(f"no struct {name}")
    return tokens(text[m.start():block_end(text, m.start())])


def skip_parens(ts, i):
    depth = 0
    while True:
        if ts[i] == "(":
            depth += 1
        elif ts[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1


def attributes(ts, i):
    """[(packed, aligned), i] of the __attribute__s at [i]."""
    packed, aligned = False, 1
    while i < len(ts) and ts[i] == "__attribute__":
        j = skip_parens(ts, i + 1)
        inner = ts[i + 1:j]
        packed |= "__packed__" in inner or "packed" in inner
        for k, t in enumerate(inner):
            if t in ("__aligned__", "aligned"):
                aligned = max(aligned, int(inner[k + 2], 0))
        i = j
    return packed, aligned, i


class Layout:
    """A structure or union laid out: its size, alignment and fields, each
    (offset, bytes) by name, nested ones as [outer__inner]."""

    def __init__(self, size, align, fields):
        self.size, self.align, self.fields = size, align, fields


def parse_members(ts, i, known, packed):
    """The members of the braced list at [ts[i]] = "{": [(name or None,
    layout)], and the index past its "}"."""
    assert ts[i] == "{"
    i += 1
    members = []
    while ts[i] != "}":
        if ts[i] in ("union", "struct") and ts[i + 1] == "{":
            kind = ts[i]
            inner, i = parse_members(ts, i + 1, known, packed)
            lay = aggregate(inner, kind == "union", packed)
            name = None
            if ts[i] != ";":
                name, i = ts[i], i + 1
            members.append((name, lay))
            i += 1
            continue
        if ts[i] == "RDMA_UAPI_PTR":
            j = skip_parens(ts, i + 1)
            members.append((ts[j - 2], scalar(8, packed)))
            i = j + 1
            continue
        if ts[i] == "struct":
            ty, i = ts[i + 1], i + 2
            base = known.get(ty)
        else:
            while ts[i] in ("const", "volatile", "enum"):
                i += 1
            ty, i = ts[i], i + 1
            base = scalar(SIZES[ty], packed, 8 if ty == "__aligned_u64" else None) if ty in SIZES else None
        name, i = ts[i], i + 1
        count = 1
        if ts[i] == "[":
            count = 0 if ts[i + 1] == "]" else int(ts[i + 1], 0)
            i = i + (2 if ts[i + 1] == "]" else 3)
        if ts[i] == ":":
            sys.exit(f"{name}: a bit field")
        if ts[i] != ";":
            sys.exit(f"{name}: one declarator per member")
        i += 1
        if base is None:
            members.append((name, None))  # a type the excerpts do not define
            continue
        members.append((name, Layout(base.size * count, base.align, base.fields if count == 1 else {})))
    return members, i + 1


def scalar(size, packed, align=None):
    return Layout(size, 1 if packed else (align or size), {})


def aggregate(members, union, packed, aligned=1):
    fields, offset, size, align = {}, 0, 0, aligned
    for name, lay in members:
        if lay is None:
            if not union:
                sys.exit(f"{name}: a member of unknown type")
            continue
        at = 0 if union else -(-offset // lay.align) * lay.align
        align = max(align, lay.align)
        if name is not None:
            fields[name] = (at, lay.size)
        for f, (o, n) in lay.fields.items():
            fields[f if name is None else f"{name}__{f}"] = (at + o, n)
        size = max(size, at + lay.size) if union else at + lay.size
        offset = size
    return Layout(-(-size // align) * align, align, fields)


def layout(text, name, known):
    ts = struct_tokens(text, name)
    i = ts.index("{")
    members, i = parse_members(ts, i, known, packed=False)
    packed, aligned, _ = attributes(ts, i)
    if packed:
        members, _ = parse_members(ts, ts.index("{"), known, packed=True)
    return aggregate(members, False, packed, aligned)


def layouts(h, wanted):
    """{C name: layout} of the structures [wanted], [(header, C name)], with
    the nested ones their header keeps."""
    known = {}
    for header in sorted({hd for hd, _ in wanted}):
        for n in NESTED.get(header, []):
            known[n] = layout(h[header], n, known)
    for header, n in wanted:
        known[n] = layout(h[header], n, known)
    return known


# Excerpts

def licence(text):
    m = re.match(r"(?:\s*/\*.*?\*/)+", text, re.S)
    if not m or "OpenIB.org BSD" not in m.group(0) and "Linux-OpenIB" not in m.group(0):
        sys.exit("a header without the OpenIB.org BSD licence")
    return m.group(0).strip("\n") + "\n"


def wanted_by_header():
    out = {n: (set(), set()) for n in SOURCES}
    for cs, ss in OUTPUTS.values():
        for h, c in cs:
            out[h][0].add(c)
        for h, s, _ in ss:
            out[h][1].add(s)
    for h, ns in NESTED.items():
        out[h][1].update(ns)
    return out


def excerpt(name, url, text):
    """[name]'s excerpt of [text]: its licence notice, its URL, and the
    #defines, enumerations and structures this script reads, in order."""
    consts, structs = wanted_by_header()[name]
    spans = []
    for m in STRUCT.finditer(text):
        if m.group(1) in structs:
            spans.append((m.start(), block_end(text, m.start())))
    for m in ENUM.finditer(text):
        end = block_end(text, m.start())
        if set(enums(text[m.start():end])) & consts:
            spans.append((m.start(), end))
    defs = defines(text)
    kept = strip_comments("\n".join(text[a:b] for a, b in spans))
    want = {c for c in consts if c in defs} | {r for r in re.findall(r"\b[A-Za-z_]\w*\b", kept) if r in defs}
    while True:  # the #defines a kept line refers to
        more = {r for c in want for r in re.findall(r"\b[A-Za-z_]\w*\b", defs[c]) if r in defs} - want
        if not more:
            break
        want |= more
    for m in DEFINE.finditer(text):
        if m.group(1) in want:
            spans.append((m.start(), m.end()))
    found = set(defines("\n".join(text[a:b] for a, b in spans))) | set(
        enums("\n".join(text[a:b] for a, b in spans)))
    found |= {m.group(1) for a, b in spans for m in STRUCT.finditer(text[a:b])}
    missing = (consts | structs) - found
    if missing:
        sys.exit(f"{name}: no {sorted(missing)}")
    body = "\n\n".join(text[a:b].strip("\n") for a, b in sorted(spans))
    return licence(text) + f"\n/* Excerpt of {url}. */\n\n" + body + "\n"


def fetch(url, cache, pins, pin):
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
    return data.decode("latin-1")


# Emission

def ml_int(v):
    return str(v) if -10 <= v <= 10 else hex(v)


def ml_field(f):
    f = f.lower()
    keywords = ("type", "method", "object", "val", "open", "to", "done", "end", "begin", "class", "match", "or",
                "of", "new", "in", "and", "do", "for", "fun", "if", "let", "mod", "rec", "then", "with")
    return f + "_" if f in keywords else f


def copyrights(h, headers):
    lines = []
    for n in headers:
        for line in licence(h[n]).splitlines():
            line = line.strip(" */")
            if line.startswith("Copyright") and line not in lines:
                lines.append(line)
    return lines


def notice(h, headers, out):
    rel = lambda p: pathlib.Path(*([".."] * (len(out.parent.relative_to(LIB).parts))), *p.relative_to(LIB).parts)
    gen = rel(HERE / "gen.py")
    text = [
        "(*---------------------------------------------------------------------------",
        "  Copyright (c) 2026 The Raven authors. All rights reserved.",
        "  SPDX-License-Identifier: ISC",
        "  ---------------------------------------------------------------------------*)",
        "",
        f"(* Generated by {gen} from the excerpts in {rel(HEADERS)}; do not",
        f"   edit. The command that regenerates this file is in {gen}.",
        "",
        "   The values are copied from rdma-core's and Linux's headers, which",
        "   offer the GNU General Public License version 2 or the OpenIB.org BSD",
        "   licence; raven takes the BSD licence:",
        "",
    ]
    text += ["   " + line for line in copyrights(h, headers)]
    text += [""] + [("   " + line).rstrip() for line in OPENIB_BSD.splitlines()]
    text[-1] += " *)"
    return text


def generate(h, out, consts, structs):
    headers = sorted({n for n, _ in consts} | {n for n, _, _ in structs})
    lines = notice(h, headers, out) + ["", "(* Constants *)", ""]
    for n, c in consts:
        lines.append(f"let {c.lower()} = {ml_int(constants(h[n], [c])[c])}")
    known = layouts(h, [(n, s) for n, s, _ in structs])
    lines += ["", "(* Structures: each field is (byte offset, bytes). *)"]
    for _, s, m in structs:
        lay = known[s]
        lines += ["", f"module {m} = struct", f"  let sizeof = {lay.size}"]
        lines += [f"  let {ml_field(f)} = ({o}, {b})" for f, (o, b) in lay.fields.items()]
        lines.append("end")
    return "\n".join(lines) + "\n"


def check_layouts(h):
    """The facts the encoders take for granted, which the headers state."""
    k = layouts(h, [(n, s) for n, s, _ in ABI_STRUCTS])
    if k["mlx5_cqe64"].size != 64 or k["mlx5_err_cqe"].size != 64:
        sys.exit("a completion is not 64 bytes")
    for a, b in [("sop_drop_qpn", "s_wqe_opcode_qpn"), ("wqe_counter", "wqe_counter"), ("op_own", "op_own")]:
        if k["mlx5_cqe64"].fields[a] != k["mlx5_err_cqe"].fields[b]:
            sys.exit(f"mlx5_cqe64.{a} is not where mlx5_err_cqe.{b} is")
    sizes = [k[s].size for s in ("mlx5_wqe_ctrl_seg", "mlx5_wqe_raddr_seg", "mlx5_wqe_data_seg")]
    if sizes != [16, 16, 16] or k["mlx5_wqe_inline_seg"].size != 4:
        sys.exit("a work entry segment is not of the size the encoder writes")


# Command line


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--excerpt", action="store_true", help="make the excerpts from the pinned headers")
    p.add_argument("--check", action="store_true", help="fail if a committed file differs")
    p.add_argument("--pin", action="store_true", help="record the digests of headers not yet pinned")
    p.add_argument("--cache", type=pathlib.Path, default=pathlib.Path.home() / ".cache/raven/mlx5-gen")
    a = p.parse_args()
    if a.excerpt:
        pins = json.loads(PINS.read_text()) if PINS.exists() else {}
        a.cache.mkdir(parents=True, exist_ok=True)
        files = {HEADERS / n: excerpt(n, url, fetch(url, a.cache, pins, a.pin)) for n, url in SOURCES.items()}
        if a.pin:
            PINS.write_text(json.dumps({u: d for u, d in pins.items() if u in SOURCES.values()}, indent=1,
                                       sort_keys=True) + "\n")
    else:
        h = {n: (HEADERS / n).read_text(encoding="latin-1") for n in SOURCES}
        check_layouts(h)
        files = {out: generate(h, out, cs, ss) for out, (cs, ss) in OUTPUTS.items()}
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
