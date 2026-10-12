# /// script
# requires-python = ">=3.10"
# ///
"""Generates defs.ml of rig.bnxt: the requests of a Broadcom NetXtreme NIC's
firmware (HWRM), the commands of its RoCE engine (RCFW) and their completions,
its work entries and completions, and their constants.

Run from the worktree root:

  uv run packages/rig/lib/bnxt/gen/gen.py
  uv run packages/rig/lib/bnxt/gen/gen.py --check
  uv run packages/rig/lib/bnxt/gen/gen.py --excerpt [--check]

The inputs are excerpts of DPDK's and Linux's headers, in headers/: each is a
header's licence notice, the URL it is cut from, and the lines this script
reads, verbatim and in the header's order (a #define, a whole enumeration, a
whole structure). SOURCES names the header each comes from, at its version.
DPDK's header is under the BSD-3-Clause licence. Linux's are under the GPL
version 2 or a BSD licence, at the user's choice; raven takes the BSD licence.
Linux's own firmware header (include/linux/bnxt/hsi.h) is under the GPL alone,
so the firmware's requests come from DPDK's copy of the same interface.
--excerpt makes the excerpts from the upstream headers, each pinned in
pins.json by URL and SHA-256 and checked against its pin; downloads are kept
in --cache, and --pin records the digests of headers not yet pinned.
Generating reads the excerpts alone, offline. --check generates into memory
and fails if a committed file differs.

A structure is laid out as the C compilers of Linux's 64-bit hosts lay it out:
each member at the next multiple of its alignment, its size's for integers, 1
in a packed structure.
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

# Sources, each at the tag in its URL

DPDK = "https://raw.githubusercontent.com/DPDK/dpdk/v24.11/"
LINUX = "https://raw.githubusercontent.com/torvalds/linux/v6.18/"

# Each header, what the library reads of it, and its licence. FreeBSD's copy
# of the firmware interface (sys/dev/bnxt/bnxt_en/hsi_struct_def.h) is not
# used: below its BSD notice it says it is Broadcom's proprietary and
# confidential information, which may not be copied.
SOURCES = {
    # The firmware's requests; BSD-3-Clause.
    "hsi_struct_def_dpdk.h": DPDK + "drivers/net/bnxt/hsi_struct_def_dpdk.h",
    # The RoCE engine's commands, entries and completions; GPL-2.0 or BSD.
    "roce_hsi.h": LINUX + "drivers/infiniband/hw/bnxt_re/roce_hsi.h",
    # The command queue's registers; GPL-2.0 or BSD.
    "qplib_rcfw.h": LINUX + "drivers/infiniband/hw/bnxt_re/qplib_rcfw.h",
    # The doorbell's valid bit and epoch; GPL-2.0 or BSD.
    "qplib_res.h": LINUX + "drivers/infiniband/hw/bnxt_re/qplib_res.h",
}

HWRM = ["ver_get", "func_reset", "func_qcaps", "func_drv_rgtr", "func_drv_unrgtr", "func_qcfg",
        "func_backing_store_qcaps_v2", "func_backing_store_cfg_v2", "ring_alloc", "stat_ctx_alloc",
        "vnic_alloc", "vnic_cfg", "cfa_l2_filter_alloc"]
RCFW = ["initialize_fw", "add_gid", "create_cq", "create_qp", "modify_qp", "register_mr", "deregister_mr"]

# What the library reads: (header, C name) of constants, and (header, C name,
# OCaml module) of structures.

CONSTANTS = [
    ("hsi_struct_def_dpdk.h", n) for n in [
        *[f"HWRM_{r.upper()}" for r in HWRM], "HWRM_MAX_REQ_LEN",
        "HWRM_FUNC_BACKING_STORE_CFG_V2_INPUT_FLAGS_BS_CFG_ALL_DONE",
        "HWRM_RING_ALLOC_INPUT_RING_TYPE_NQ", "HWRM_RING_ALLOC_INPUT_RING_TYPE_L2_CMPL",
        "HWRM_RING_ALLOC_INPUT_RING_TYPE_RX", "HWRM_RING_ALLOC_INPUT_INT_MODE_MSIX",
        "HWRM_RING_ALLOC_INPUT_ENABLES_NQ_RING_ID_VALID", "HWRM_RING_ALLOC_INPUT_ENABLES_RX_BUF_SIZE_VALID",
        "HWRM_VNIC_CFG_INPUT_ENABLES_MRU", "HWRM_VNIC_CFG_INPUT_ENABLES_DEFAULT_RX_RING_ID",
        "HWRM_VNIC_CFG_INPUT_ENABLES_DEFAULT_CMPL_RING_ID",
        "HWRM_CFA_L2_FILTER_ALLOC_INPUT_FLAGS_PATH_RX", "HWRM_CFA_L2_FILTER_ALLOC_INPUT_ENABLES_L2_ADDR",
        "HWRM_CFA_L2_FILTER_ALLOC_INPUT_ENABLES_L2_ADDR_MASK", "HWRM_CFA_L2_FILTER_ALLOC_INPUT_ENABLES_DST_ID"]
] + [
    ("roce_hsi.h", n) for n in [
        *[f"CMDQ_BASE_OPCODE_{c.upper()}" for c in RCFW],
        "CMDQ_INIT_CMDQ_SIZE_SFT", "CMDQ_INITIALIZE_FW_FLAGS_HW_REQUESTER_RETX_SUPPORTED",
        "CREQ_BASE_V", "CREQ_BASE_TYPE_QP_EVENT", "CREQ_QP_EVENT_EVENT_QP_ERROR_NOTIFICATION",
        "PTU_PTE_VALID", "PTU_PTE_LAST", "PTU_PTE_NEXT_TO_LAST",
        "CMDQ_CREATE_QP_TYPE_RC", "CMDQ_MODIFY_QP_QP_TYPE_RC", "CMDQ_MODIFY_QP_NETWORK_TYPE_ROCEV2_IPV4",
        "CMDQ_MODIFY_QP_PATH_MTU_MTU_4096", "CMDQ_MODIFY_QP_NEW_STATE_INIT", "CMDQ_MODIFY_QP_NEW_STATE_RTR",
        "CMDQ_MODIFY_QP_NEW_STATE_RTS", "CMDQ_MODIFY_QP_ACCESS_LOCAL_WRITE",
        "CMDQ_MODIFY_QP_ACCESS_REMOTE_WRITE",
        *[f"CMDQ_MODIFY_QP_MODIFY_MASK_{m}" for m in [
            "STATE", "ACCESS", "PKEY", "DGID", "SGID_INDEX", "HOP_LIMIT", "DEST_MAC", "PATH_MTU", "RQ_PSN",
            "MIN_RNR_TIMER", "MAX_DEST_RD_ATOMIC", "DEST_QP_ID", "TIMEOUT", "RETRY_CNT", "RNR_RETRY",
            "MAX_RD_ATOMIC", "SQ_PSN"]],
        "CMDQ_REGISTER_MR_FLAGS_ALLOC_MR", "CMDQ_REGISTER_MR_LVL_SFT", "CMDQ_REGISTER_MR_LOG2_PG_SIZE_SFT",
        "CMDQ_REGISTER_MR_ACCESS_LOCAL_WRITE", "CMDQ_REGISTER_MR_ACCESS_REMOTE_WRITE",
        "SQ_BASE_WQE_TYPE_SEND", "SQ_SEND_FLAGS_SIGNAL_COMP", "RQ_WQE_WQE_TYPE_RCV",
        "SQ_MSN_SEARCH_START_IDX_SFT", "SQ_MSN_SEARCH_NEXT_PSN_SFT", "CQ_BASE_TOGGLE",
        "CQ_BASE_CQE_TYPE_MASK", "CQ_BASE_CQE_TYPE_REQ", "CQ_BASE_CQE_TYPE_RES_RC",
        "CQ_BASE_CQE_TYPE_TERMINAL",
        *[f"CQ_REQ_STATUS_{s}" for s in [
            "BAD_RESPONSE_ERR", "LOCAL_LENGTH_ERR", "LOCAL_QP_OPERATION_ERR", "LOCAL_PROTECTION_ERR",
            "MEMORY_MGT_OPERATION_ERR", "REMOTE_INVALID_REQUEST_ERR", "REMOTE_ACCESS_ERR",
            "REMOTE_OPERATION_ERR", "RNR_NAK_RETRY_CNT_ERR", "TRANSPORT_RETRY_CNT_ERR",
            "WORK_REQUEST_FLUSHED_ERR"]],
        *[f"CQ_RES_RC_STATUS_{s}" for s in [
            "LOCAL_ACCESS_ERROR", "LOCAL_LENGTH_ERR", "LOCAL_PROTECTION_ERR", "LOCAL_QP_OPERATION_ERR",
            "MEMORY_MGT_OPERATION_ERR", "REMOTE_INVALID_REQUEST_ERR", "WORK_REQUEST_FLUSHED_ERR",
            "HW_FLUSH_ERR"]]]
] + [
    ("qplib_rcfw.h", n) for n in ["RCFW_COMM_BASE_OFFSET", "RCFW_PF_VF_COMM_PROD_OFFSET",
                                  "RCFW_COMM_TRIG_OFFSET", "RCFW_CMDQ_TRIG_VAL", "FIRMWARE_FIRST_FLAG"]
] + [
    ("qplib_res.h", n) for n in ["BNXT_QPLIB_DBR_VALID", "BNXT_QPLIB_DBR_EPOCH_SHIFT"]
]


def module(c):
    return c[0].upper() + c[1:]


STRUCTS = [
    ("hsi_struct_def_dpdk.h", s, module(s)) for s in [
        "hwrm_cmd_hdr", "hwrm_resp_hdr", "hwrm_func_qcaps_input", "hwrm_func_qcaps_output",
        "hwrm_func_qcfg_input", "hwrm_func_qcfg_output", "hwrm_func_backing_store_qcaps_v2_input",
        "hwrm_func_backing_store_qcaps_v2_output", "hwrm_func_backing_store_cfg_v2_input",
        "hwrm_ring_alloc_input", "hwrm_ring_alloc_output", "hwrm_stat_ctx_alloc_input",
        "hwrm_stat_ctx_alloc_output", "hwrm_vnic_alloc_output", "hwrm_vnic_cfg_input",
        "hwrm_cfa_l2_filter_alloc_input"]
] + [
    ("roce_hsi.h", s, module(s)) for s in [
        "cmdq_init", "cmdq_base", "creq_base", *[f"cmdq_{c}" for c in RCFW], "creq_create_qp_resp",
        "sq_send_hdr", "sq_sge", "rq_wqe_hdr", "cq_base", "cq_res_rc"]
]

# Reading C headers

DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(\w+)((?:[^\n]*\\\n)*[^\n]*)$", re.M)
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
    """{name: expression} of [text]'s #defines, continued lines joined."""
    out = {}
    for m in DEFINE.finditer(strip_comments(text)):
        value = re.sub(r"\\\n", " ", m.group(2)).strip()
        if value:
            out[m.group(1)] = value
    return out


def evaluate(expr, names, seen=()):
    """The integer of the C constant expression [expr] over [names]."""
    e = re.sub(r"\bUINT(8|16|32|64)_C\b", "", expr)
    e = re.sub(r"\(\s*(unsigned|unsigned int|unsigned long|uint32_t|uint64_t|u64|u32|int)\s*\)", "", e)
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
    **dict.fromkeys(["u8", "uint8_t", "char"], 1),
    **dict.fromkeys(["__le16", "__be16", "u16", "uint16_t"], 2),
    **dict.fromkeys(["__le32", "__be32", "u32", "uint32_t"], 4),
    **dict.fromkeys(["__le64", "u64", "uint64_t"], 8),
}
TOKEN = re.compile(r"\w+|[{}\[\];(),:*]")


def struct_tokens(text, name):
    """The tokens of the structure [name], without the #defines its body
    holds."""
    m = re.search(rf"^[ \t]*struct[ \t]+{name}[ \t]*\{{", text, re.M)
    if not m:
        sys.exit(f"no struct {name}")
    body = strip_comments(text[m.start():block_end(text, m.start())])
    return TOKEN.findall(DEFINE.sub("", body))


def layout(text, name):
    """(size, {field: (offset, bytes)}) of the structure [name]; an array is
    one field of all its elements' bytes."""
    ts = struct_tokens(text, name)
    packed = "__rte_packed" in ts[ts.index("}"):]
    i, offset, align, fields = ts.index("{") + 1, 0, 1, {}
    while ts[i] != "}":
        ty, name_, i = ts[i], ts[i + 1], i + 2
        if ty not in SIZES:
            sys.exit(f"{name}.{name_}: a member of type {ty}")
        count = 1
        if ts[i] == "[":
            count, i = int(ts[i + 1], 0), i + 3
        if ts[i] != ";":
            sys.exit(f"{name}.{name_}: not one plain member")
        i += 1
        a = 1 if packed else SIZES[ty]
        offset = -(-offset // a) * a
        fields[name_] = (offset, SIZES[ty] * count)
        offset += SIZES[ty] * count
        align = max(align, a)
    return -(-offset // align) * align, fields


# Excerpts

def licence(text):
    m = re.match(r"\s*/\*.*?\*/", text, re.S)
    if not m or "BSD-3-Clause" not in m.group(0) and "BSD license below" not in m.group(0):
        sys.exit("a header without a BSD licence")
    return m.group(0).strip("\n") + "\n"


def wanted_by_header():
    out = {n: (set(), set()) for n in SOURCES}
    for h, c in CONSTANTS:
        out[h][0].add(c)
    for h, s, _ in STRUCTS:
        out[h][1].add(s)
    return out


def excerpt(name, url, text):
    """[name]'s excerpt of [text]: its licence notice, its URL, and the
    #defines, enumerations and structures this script reads, in order. A
    #define inside a kept structure stays there."""
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
    inside = lambda p: any(a <= p < b for a, b in spans)
    spans += [(m.start(), m.end()) for m in DEFINE.finditer(text) if m.group(1) in want and not inside(m.start())]
    joined = "\n".join(text[a:b] for a, b in spans)
    found = set(defines(joined)) | set(enums(joined)) | {m.group(1) for m in STRUCT.finditer(joined)}
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
    keywords = ("type", "method", "object", "val", "open", "to", "done", "end", "begin", "class", "match", "or",
                "of", "new", "in", "and", "do", "for", "fun", "if", "let", "mod", "rec", "then", "with", "private",
                "struct", "sig", "when", "while", "function", "module", "external", "include", "exception",
                "virtual", "constraint", "inherit", "initializer", "lazy", "mutable", "nonrec", "try", "downto",
                "else", "functor", "assert", "land", "lor", "lxor", "lsl", "lsr", "asr")
    return f + "_" if f in keywords else f


def notice(h):
    """The header of defs.ml: each source's copyright lines and licence, as
    the licence asks."""
    text = [
        "(*---------------------------------------------------------------------------",
        "  Copyright (c) 2026 The Raven authors. All rights reserved.",
        "  SPDX-License-Identifier: ISC",
        "  ---------------------------------------------------------------------------*)",
        "",
        "(* Generated by gen/gen.py from the excerpts in gen/headers; do not",
        "   edit. The command that regenerates this file is in gen/gen.py.",
        "",
        "   The values are copied from DPDK's header, under the BSD-3-Clause",
        "   licence, and from Linux's, which offer the GNU General Public License",
        "   version 2 or the BSD licence below; raven takes the BSD licence.",
        "",
    ]
    seen = []
    for n in SOURCES:
        lines = [line.strip().lstrip("/*").strip() for line in licence(h[n]).splitlines()]
        if n == "hsi_struct_def_dpdk.h":
            keep = [line for line in lines if line.startswith(("SPDX", "Copyright", "All rights"))]
        else:  # the copyright lines, then the BSD licence
            copyright = next(i for i, line in enumerate(lines) if line.startswith("Copyright"))
            bsd = next(i for i, line in enumerate(lines) if line.startswith("Redistribution"))
            end = next(i for i, line in enumerate(lines) if "POSSIBILITY OF SUCH DAMAGE" in line)
            keep = lines[copyright:copyright + 2] + [""] + lines[bsd:end + 1]
        if keep not in seen:
            seen.append(keep)
    for i, keep in enumerate(seen):
        if i:
            text.append("")
        text += [("   " + line).rstrip() for line in keep]
    text[-1] += " *)"
    return text


def generate(h):
    lines = notice(h) + ["", "(* Constants *)", ""]
    for n, c in CONSTANTS:
        lines.append(f"let {c.lower()} = {ml_int(constants(h[n], [c])[c])}")
    lines += ["", "(* Structures: each field is (byte offset, bytes), an array's of all its",
              "   elements. *)"]
    for n, s, m in STRUCTS:
        size, fields = layout(h[n], s)
        lines += ["", f"module {m} = struct", f"  let sizeof = {size}"]
        lines += [f"  let {ml_field(f)} = ({o}, {b})" for f, (o, b) in fields.items()]
        lines.append("end")
    return "\n".join(lines) + "\n"


def check_layouts(h):
    """The facts the library takes for granted, which the headers state."""
    size = lambda s: layout(h["roce_hsi.h"], s)[0]
    if size("cq_base") != 32 or size("cq_res_rc") != 32:
        sys.exit("a completion is not 32 bytes")
    if size("cmdq_base") > 16 or size("creq_base") != 16:
        sys.exit("a command's base or a completion of one is not 16 bytes")
    if size("sq_send_hdr") + size("sq_sge") != 48 or size("rq_wqe_hdr") != size("sq_send_hdr"):
        sys.exit("a work entry's header and one SGE are not 48 bytes")
    for r in ["hwrm_resp_hdr", "hwrm_cmd_hdr"]:
        if layout(h["hsi_struct_def_dpdk.h"], r)[0] not in (8, 16):
            sys.exit(f"{r} is not of the size the library reads")


# Command line


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--excerpt", action="store_true", help="make the excerpts from the pinned headers")
    p.add_argument("--check", action="store_true", help="fail if a committed file differs")
    p.add_argument("--pin", action="store_true", help="record the digests of headers not yet pinned")
    p.add_argument("--cache", type=pathlib.Path, default=pathlib.Path.home() / ".cache/raven/bnxt-gen")
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
        files = {LIB / "defs.ml": generate(h)}
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
