# /// script
# requires-python = ">=3.10"
# ///
"""Generates defs.ml, the tables of device_nv: the resource manager's
classes, controls and parameter layouts the driver allocates channels and
reads faults with, per release of NVIDIA's driver where they differ, and the
names of the errors and faults it reports.

Run from the worktree root:

  uv run dev/device/nv/gen/gen.py
  uv run dev/device/nv/gen/gen.py --check
  uv run dev/device/nv/gen/gen.py --excerpt [--check]

The inputs are excerpts of NVIDIA's headers, in headers/RELEASE/: each is a
header's licence notice and the definitions this script reads, verbatim and
in the header's order, with the definitions they depend on. --excerpt makes
them from the upstream headers of each release, each pinned in pins.json by
URL and SHA-256 and checked against its pin; downloads are kept in --cache,
and --pin records the digests of headers not yet pinned. Generating reads the
excerpts alone, offline. --check generates into memory and fails if a
committed file differs.

Text is read and written as latin-1, one character per byte, so every byte
of a header round-trips into its excerpt as upstream wrote it.

Struct layouts follow the C rules of the 64-bit Linux ABIs, x86_64 and
aarch64 alike: a field at the next multiple of its alignment, a struct's
size a multiple of its largest alignment, NV_DECLARE_ALIGNED and
NV_ALIGN_BYTES raising a field's alignment. Every value is NVIDIA's.
Definitions the same in every release are emitted once; those that differ
are emitted per release, and the script fails if the split changes. Every
table is a literal, which the compiler lays out as static data.
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

KERNEL = "https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/"
# The driver releases, by the branch the RM reports: the tree each is read
# from, and its version.
RELEASES = {
    570: ("570.144", "570.144"),
    580: ("2af9f1f0f7de4988432d4ae875b5858ffdb09cc2", "580.105.08"),
    610: ("610.43.03", "610.43.03"),
    615: ("615.71.09", "615.71.09"),
}
SDK = "src/common/sdk/nvidia/inc/"

# Each excerpt and the header it is cut from, in each release's tree. A name
# is read from the first header that defines it.
SOURCES = {
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
    "ctrlc36f.h": SDK + "ctrl/ctrlc36f.h",
    "ctrl2080perf.h": SDK + "ctrl/ctrl2080/ctrl2080perf.h",
    "ctrl83dedebug.h": SDK + "ctrl/ctrl83de/ctrl83dedebug.h",
    "dev_fault.h": "kernel-open/nvidia-uvm/hwref/ampere/ga100/dev_fault.h",
}

# Constants, the same in every release.
CONSTANTS = [
    # classes
    "KEPLER_CHANNEL_GROUP_A", "FERMI_CONTEXT_SHARE_A", "GT200_DEBUGGER",
    # controls
    "NVA06C_CTRL_CMD_GPFIFO_SCHEDULE", "NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN", "NV2080_CTRL_CMD_PERF_BOOST",
    "NV2080_CTRL_PERF_BOOST_FLAGS_CMD_BOOST_TO_MAX", "NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_YES",
    "NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_PRIORITY_HIGH", "NV2080_CTRL_PERF_BOOST_DURATION_INFINITE",
    "NV83DE_CTRL_CMD_DEBUG_READ_ALL_SM_ERROR_STATES", "NV83DE_CTRL_CMD_DEBUG_READ_MMU_FAULT_INFO",
    # allocations
    "NV2080_ENGINE_TYPE_GRAPHICS", "NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC",
    # a channel's error notifier
    "NV_CHANNELGPFIFO_NOTIFICATION_TYPE_ERROR", "NV_CHANNELGPFIFO_NOTIFICATION_TYPE__SIZE_1",
]

# Bit fields of 32-bit words, "hi:lo" in the headers: (lowest bit, bits).
FIELDS = [
    "NV2080_CTRL_PERF_BOOST_FLAGS_CMD", "NV2080_CTRL_PERF_BOOST_FLAGS_CUDA",
    "NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_PRIORITY", "NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT",
]

# Structs, by C name: the module they become and the fields read ("a__b" for
# a field b of a struct a).
STRUCTS = {
    "NV_CTXSHARE_ALLOCATION_PARAMETERS": ("Ctxshare_alloc", ["hVASpace", "flags"]),
    "NV83DE_ALLOC_PARAMETERS": ("Nv83de_alloc", ["hAppClient", "hClass3dObject"]),
    "NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS": ("Work_submit_token", ["workSubmitToken"]),
    "NV2080_CTRL_PERF_BOOST_PARAMS": ("Perf_boost", ["flags", "duration"]),
    "NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS": ("Sm_error_states", [
        "hTargetChannel", "numSMsToRead", "smErrorStateArray", "mmuFault__valid"]),
    "NV83DE_SM_ERROR_STATE_REGISTERS": ("Sm_error_state", ["hwwGlobalEsr", "hwwWarpEsr", "hwwWarpEsrPc64"]),
    "NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_PARAMS": ("Mmu_fault_info", ["mmuFaultInfoList", "count"]),
    "NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_ENTRY": ("Mmu_fault_entry", ["faultAddress", "faultType", "accessType"]),
    "Nvc56fControl": ("Userd", ["GPGet", "GPPut"]),
    "NvNotification": ("Notification", ["info32", "info16", "status"]),
    "NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS": ("Channel_group_alloc", ["engineType"]),
    "NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS": ("Gpfifo_alloc", [
        "hObjectError", "hObjectBuffer", "gpFifoOffset", "gpFifoEntries", "hContextShare", "hUserdMemory",
        "userdOffset"]),
    "NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS": ("Group_schedule", ["bEnable"]),
}

# The tables that differ between the releases: later releases add errors.
TABLES_PER_RELEASE = {"robust_channel_errors"}

# The structs whose layouts differ between the releases.
PER_RELEASE = {"NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS", "NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS",
               "NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS"}

# Name tables: (OCaml name, header, the defines read, as a pattern whose
# group 1 is the name an entry carries). The errors leave out the recovery
# levels and the count, which are no error.
TABLES = [
    ("robust_channel_errors", "nverror.h", r"ROBUST_CHANNEL_(?!ERROR_RECOVERY_LEVEL_|LAST_ERROR)(\w+)"),
    ("fault_types", "dev_fault.h", r"NV_PFAULT_FAULT_TYPE_(\w+)"),
    ("access_types", "dev_fault.h", r"NV_PFAULT_ACCESS_TYPE_(\w+)"),
]

# C headers

COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(\w+)(\([\w\s,]*\))?((?:[^\n]*\\\n)*[^\n]*)$", re.M)
TYPEDEF = re.compile(r"^[ \t]*typedef\b", re.M)
TAGGED = re.compile(r"^[ \t]*(struct|union|enum)[ \t]+(\w+)\s*\{", re.M)
IDENT = re.compile(r"\b[A-Za-z_]\w*\b")
DIRECTIVE = re.compile(r"^[ \t]*#[ \t]*(if|ifdef|ifndef|elif|else|endif|define|undef)\b(.*)$")

# The macros a 64-bit Linux build defines that the headers' conditionals
# test; every other name a conditional tests is undefined.
PLATFORM = {"__linux__", "__LP64__"}

# The scalar types, (bytes, alignment), in the 64-bit Linux ABIs.
SCALARS = {
    **{t: (1, 1) for t in ["NvU8", "NvS8", "NvV8", "NvBool", "char", "NvChar"]},
    **{t: (2, 2) for t in ["NvU16", "NvS16", "NvV16"]},
    **{t: (4, 4) for t in ["NvU32", "NvS32", "NvV32", "NvHandle", "NV_STATUS", "NvF32", "int", "unsigned"]},
    **{t: (8, 8) for t in ["NvU64", "NvS64", "NvP64", "NvLength", "NvUPtr", "NvF64", "long", "size_t"]},
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
    stack, out = [], []
    for line in b.split("\n"):
        live = all(taken for taken, _ in stack)
        out.append(live)
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
        self.texts, self.where, self.layouts = texts, {}, {}
        for h, text in texts.items():
            for it in items(text):
                it.header = h
                for n in it.names:
                    if n in self.where and self.where[n].header == h:
                        old = self.where[n]
                        if text[old.start:old.end] != text[it.start:it.end] and it.kind == "define":
                            sys.exit(f"{h}: {n} is defined twice")
                    self.where.setdefault(n, it)

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
        m = re.fullmatch(r"\(?\s*(\d+)\s*:\s*(\d+)\s*\)?", self.item(name).body)
        if not m:
            sys.exit(f"{name} is not a bit field")
        hi, lo = int(m.group(1)), int(m.group(2))
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
            r = self.aggregate(it.kind, it.body, name)
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
        e = self.expand(expr)
        e = re.sub(r"\b(0[xX][0-9a-fA-F]+|\d+)[uUlL]+\b", r"\1", e)
        if re.search(r"[A-Za-z_]", re.sub(r"\b0[xX][0-9a-fA-F]+\b", "", e)):
            sys.exit(f"{where}: not a count: {expr}")
        return eval(e.replace("/", "//"), {"__builtins__": {}})

    def aggregate(self, kind, body, where):
        offset, align, fields = 0, 1, {}
        for name, ty, counts, falign in self.members(body, where):
            if isinstance(ty, tuple):
                size, a, sub = self.aggregate(ty[0], ty[1], where)
            else:
                size, a, sub = self.scalar_or_type(ty)
            a = max(a, falign)
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
    """The header's first comment, its licence notice."""
    end = text.index("*/") + 2
    return text[:end] + "\n"


def wanted(model):
    names = set(CONSTANTS) | set(FIELDS) | set(STRUCTS)
    for _, h, pattern in TABLES:
        names |= table_names(model, h, pattern)
    return names


def table_names(model, header, pattern):
    """The defines of [header] whose names [pattern] matches and whose values
    are numbers."""
    return {n for n, it in model.where.items()
            if it.header == header and it.kind == "define" and it.params is None and re.fullmatch(pattern, n)
            and re.fullmatch(r"\(?\s*(0[xX][0-9a-fA-F]+|\d+)[uUlL]*\s*\)?", it.body)}


def excerpts(texts):
    model = Model(texts)
    keep = model.closure(wanted(model))
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
    return data.decode("latin-1")


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
    return f"0x{v:x}" if v > 9 else str(v)


def ml_tuple(v):
    return "(" + ", ".join(ml_int(x) for x in v) + ")"


def struct(model, cname, where):
    size, _, fields = model.layout(cname)
    module, names = STRUCTS[cname]
    for f in names:
        if f not in fields:
            sys.exit(f"{where}: {cname} has no field {f}; it has {sorted(fields)}")
    return size, [(f, fields[f]) for f in names]


def emit_struct(out, module, layout, indent=""):
    size, fields = layout
    out.append(f"{indent}module {module} = struct")
    out.append(f"{indent}  let sizeof = {size}")
    for f, v in fields:
        out.append(f"{indent}  let {snake(f)} = {ml_tuple(v)}")
    out.append(f"{indent}end")


def emit_table(out, name, entries, indent=""):
    out.append(f"{indent}let {name} = [")
    for v, n in entries:
        out.append(f"{indent}  ({ml_int(v)}, {json.dumps(n)});")
    out.append(f"{indent}]")


def generate():
    models = {r: Model({h: (HEADERS / str(r) / h).read_text(encoding="latin-1") for h in SOURCES})
              for r in RELEASES}
    out = []

    def same(what, values):
        if len(set(values.values())) != 1:
            sys.exit(f"{what} differs between releases: {values}")
        return values[min(RELEASES)]

    out.append("(* The releases whose layouts are below: "
               + ", ".join(v for _, v in RELEASES.values()) + ". *)")
    out.append("let releases = [ " + "; ".join(str(r) for r in RELEASES) + " ]")
    out.append("")
    out.append("(* Constants, the same in every release. *)")
    for c in CONSTANTS:
        v = same(c, {r: m.value(c) for r, m in models.items()})
        out.append(f"let {snake(c)} = {ml_int(v)}")
    out.append("")
    out.append("(* Bit fields of their words: (lowest bit, bits). *)")
    for f in FIELDS:
        v = same(f, {r: m.bits(f) for r, m in models.items()})
        out.append(f"let {snake(f)} = {ml_tuple(v)}")
    out.append("")

    layouts = {c: {r: struct(m, c, r) for r, m in models.items()} for c in STRUCTS}
    differ = {c for c, per in layouts.items() if len({repr(v) for v in per.values()}) != 1}
    if differ != PER_RELEASE:
        sys.exit(f"the structs that differ between releases are {sorted(differ)}, not {sorted(PER_RELEASE)}")
    out.append("(* Structs: each field is (byte offset, bytes), an array's (offset, bytes")
    out.append("   of an element, elements). *)")
    for c, (module, _) in STRUCTS.items():
        if c not in PER_RELEASE:
            emit_struct(out, module, layouts[c][min(RELEASES)])
            out.append("")

    tables = {name: {r: tuple(sorted((m.value(n), re.fullmatch(pattern, n).group(1))
                                     for n in table_names(m, h, pattern))) for r, m in models.items()}
              for name, h, pattern in TABLES}
    differ = {name for name, per in tables.items() if len(set(per.values())) != 1}
    if differ != TABLES_PER_RELEASE:
        sys.exit(f"the tables that differ between releases are {sorted(differ)}, not {sorted(TABLES_PER_RELEASE)}")
    for name, h, _ in TABLES:
        if name not in TABLES_PER_RELEASE:
            out.append(f"(* {h}'s {name.replace('_', ' ')}, by value. *)")
            emit_table(out, name, tables[name][min(RELEASES)])
            out.append("")

    out.append("(* What differs between the releases. *)")
    out.append("module type RELEASE = sig")
    for c, (module, names) in STRUCTS.items():
        if c in PER_RELEASE:
            out.append(f"  module {module} : sig")
            out.append("    val sizeof : int")
            for f, v in layouts[c][min(RELEASES)][1]:
                out.append(f"    val {snake(f)} : {' * '.join(['int'] * len(v))}")
            out.append("  end")
            out.append("")
    for name, h, _ in TABLES:
        if name in TABLES_PER_RELEASE:
            out.append(f"  (* {h}'s {name.replace('_', ' ')}, by value. *)")
            out.append(f"  val {name} : (int * string) list")
            out.append("")
    out[-1] = "end"
    out.append("")
    for r in RELEASES:
        out.append(f"module R{r} : RELEASE = struct")
        body = []
        for c, (module, _) in STRUCTS.items():
            if c in PER_RELEASE:
                emit_struct(body, module, layouts[c][r], indent="  ")
                body.append("")
        for name, _, _ in TABLES:
            if name in TABLES_PER_RELEASE:
                emit_table(body, name, tables[name][r], indent="  ")
                body.append("")
        out += body[:-1]
        out.append("end")
        out.append("")
    out.append("(* The layouts of release [r], if it is one of {!releases}. *)")
    out.append("let release = function")
    for r in RELEASES:
        out.append(f"  | {r} -> Some (module R{r} : RELEASE)")
    out.append("  | _ -> None")
    return header(models.values()) + "\n".join(out) + "\n"


def header(models):
    owners = sorted({re.search(r"Copyright \(c\) ([^\n]*?NVIDIA[^\n.]*)", t, re.I).group(1).strip()
                     for m in models for t in m.texts.values()})
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
        + "\n".join(f"   Copyright (c) {o}" for o in owners) + "\n\n" + notice + " *)\n\n"
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
        files = {}
        for r, (ref, _) in RELEASES.items():
            texts = {h: fetch(KERNEL + ref + "/" + path, a.cache, pins, a.pin) for h, path in SOURCES.items()}
            files.update({HEADERS / str(r) / h: t for h, t in excerpts(texts).items()})
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
