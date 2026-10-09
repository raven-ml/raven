# /// script
# requires-python = ">=3.10"
# ///
"""Generates defs.ml, the tables of rig_amd_amdgpu: the requests of KFD
(/dev/kfd) and of amdgpu's render node the library makes, their numbers,
the layouts of their parameters and the flags they take.

Run from the worktree root:

  uv run dev/rig/lib/amd/amdgpu/gen/gen.py
  uv run dev/rig/lib/amd/amdgpu/gen/gen.py --check
  uv run dev/rig/lib/amd/amdgpu/gen/gen.py --excerpt [--check]

The inputs are excerpts of Linux's uapi headers, in headers/: each is a
header's licence notice and the definitions this script reads, verbatim and
in the header's order, with the definitions they depend on, as a 64-bit
Linux build compiles them. --excerpt makes them from the upstream headers,
each pinned in pins.json by URL and SHA-256 and checked against its pin;
downloads are kept in --cache, and --pin records the digests of headers not
yet pinned. Generating reads the excerpts alone, offline. --check generates
into memory and fails if a committed file differs.

Text is read and written as latin-1, one character per byte, so every byte
of a header round-trips into its excerpt as upstream wrote it.

Struct layouts follow the C rules of the 64-bit Linux ABIs, x86_64 and
aarch64 alike: a field at the next multiple of its alignment, a struct's
size a multiple of its largest alignment. A request number is Linux's
_IOC encoding of asm-generic/ioctl.h, which both ABIs use: the direction
in bits 31:30 (1 write, 2 read), the parameters' size in 29:16, the type in
15:8 and the number in 7:0. Every table is a literal, which the compiler
lays out as static data.
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

KERNEL = "https://raw.githubusercontent.com/ROCm/ROCK-Kernel-Driver/33970e1351f5e511029602454979f3de7e22260f/"

# Each excerpt and the header it is cut from. A name is read from the first
# header that defines it.
SOURCES = {
    "kfd_ioctl.h": "include/uapi/linux/kfd_ioctl.h",
    "amdgpu_drm.h": "include/uapi/drm/amdgpu_drm.h",
    "drm.h": "include/uapi/drm/drm.h",
}

# Constants.
CONSTANTS = [
    # KFD's memory
    "KFD_IOC_ALLOC_MEM_FLAGS_VRAM", "KFD_IOC_ALLOC_MEM_FLAGS_GTT", "KFD_IOC_ALLOC_MEM_FLAGS_USERPTR",
    "KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP", "KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE", "KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE",
    "KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC", "KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE", "KFD_IOC_ALLOC_MEM_FLAGS_COHERENT",
    "KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED",
    # KFD's events
    "KFD_IOC_EVENT_SIGNAL", "KFD_IOC_EVENT_MEMORY", "KFD_IOC_EVENT_HW_EXCEPTION",
    # KFD's queues
    "KFD_IOC_QUEUE_TYPE_COMPUTE", "KFD_IOC_QUEUE_TYPE_COMPUTE_AQL", "KFD_IOC_QUEUE_TYPE_SDMA",
    "KFD_MAX_QUEUE_PERCENTAGE",
    # the render node
    "AMDGPU_INFO_DEV_INFO", "AMDGPU_CTX_OP_ALLOC_CTX", "AMDGPU_CTX_OP_FREE_CTX",
    "AMDGPU_CTX_OP_SET_STABLE_PSTATE", "AMDGPU_CTX_STABLE_PSTATE_STANDARD",
]

# Requests, whose numbers encode their parameters' size.
REQUESTS = [
    "AMDKFD_IOC_GET_VERSION", "AMDKFD_IOC_CREATE_QUEUE", "AMDKFD_IOC_DESTROY_QUEUE", "AMDKFD_IOC_CREATE_EVENT",
    "AMDKFD_IOC_DESTROY_EVENT", "AMDKFD_IOC_RESET_EVENT", "AMDKFD_IOC_WAIT_EVENTS", "AMDKFD_IOC_ACQUIRE_VM",
    "AMDKFD_IOC_ALLOC_MEMORY_OF_GPU", "AMDKFD_IOC_FREE_MEMORY_OF_GPU", "AMDKFD_IOC_MAP_MEMORY_TO_GPU",
    "AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU", "AMDKFD_IOC_RUNTIME_ENABLE",
    "DRM_IOCTL_AMDGPU_INFO", "DRM_IOCTL_AMDGPU_CTX",
]

# Structs and unions, by C tag: the module they become and the fields read
# ("a__b" for a field b of a member a).
STRUCTS = {
    "kfd_ioctl_get_version_args": ("Get_version", ["major_version", "minor_version"]),
    "kfd_ioctl_create_queue_args": ("Create_queue", [
        "ring_base_address", "write_pointer_address", "read_pointer_address", "doorbell_offset", "ring_size",
        "gpu_id", "queue_type", "queue_percentage", "queue_priority", "queue_id", "eop_buffer_address",
        "eop_buffer_size", "ctx_save_restore_address", "ctx_save_restore_size", "ctl_stack_size"]),
    "kfd_ioctl_destroy_queue_args": ("Destroy_queue", ["queue_id"]),
    "kfd_ioctl_create_event_args": ("Create_event", ["event_page_offset", "event_type", "auto_reset",
                                                     "event_id"]),
    "kfd_ioctl_destroy_event_args": ("Destroy_event", ["event_id"]),
    "kfd_ioctl_reset_event_args": ("Reset_event", ["event_id"]),
    "kfd_ioctl_wait_events_args": ("Wait_events", ["events_ptr", "num_events", "wait_for_all", "timeout"]),
    "kfd_event_data": ("Event_data", [
        "memory_exception_data__failure__NotPresent", "memory_exception_data__failure__ReadOnly",
        "memory_exception_data__failure__NoExecute", "memory_exception_data__failure__imprecise",
        "memory_exception_data__va", "memory_exception_data__gpu_id", "memory_exception_data__ErrorType",
        "hw_exception_data__reset_type", "hw_exception_data__reset_cause", "hw_exception_data__memory_lost",
        "hw_exception_data__gpu_id", "event_id"]),
    "kfd_ioctl_acquire_vm_args": ("Acquire_vm", ["drm_fd", "gpu_id"]),
    "kfd_ioctl_alloc_memory_of_gpu_args": ("Alloc_memory_of_gpu", ["va_addr", "size", "handle", "mmap_offset",
                                                                   "gpu_id", "flags"]),
    "kfd_ioctl_free_memory_of_gpu_args": ("Free_memory_of_gpu", ["handle"]),
    "kfd_ioctl_map_memory_to_gpu_args": ("Map_memory_to_gpu", ["handle", "device_ids_array_ptr", "n_devices",
                                                               "n_success"]),
    "kfd_ioctl_runtime_enable_args": ("Runtime_enable", []),
    "drm_amdgpu_info": ("Info", ["return_pointer", "return_size", "query"]),
    "drm_amdgpu_info_device": ("Info_device", ["gpu_counter_freq", "cu_bitmap"]),
    "drm_amdgpu_ctx": ("Ctx", ["in__op", "in__flags", "in__ctx_id", "out__alloc__ctx_id"]),
}

# Structs laid out alike, by C tag: the first is emitted for both, and the
# script fails if their layouts differ.
SAME = {"kfd_ioctl_unmap_memory_from_gpu_args": "kfd_ioctl_map_memory_to_gpu_args"}

# C headers

COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(\w+)(\([\w\s,]*\))?((?:[^\n]*\\\n)*[^\n]*)$", re.M)
TYPEDEF = re.compile(r"^[ \t]*typedef\b", re.M)
TAGGED = re.compile(r"^[ \t]*(struct|union|enum)[ \t]+(\w+)\s*\{", re.M)
IDENT = re.compile(r"\b[A-Za-z_]\w*\b")
DIRECTIVE = re.compile(r"^[ \t]*#[ \t]*(if|ifdef|ifndef|elif|else|endif|define|undef)\b(.*)$")
SIZEOF = re.compile(r"\bsizeof\s*\(")

# The macros a 64-bit Linux build of a program defines that the headers'
# conditionals test; every other name a conditional tests is undefined.
PLATFORM = {"__linux__", "__LP64__"}

# The scalar types, (bytes, alignment), in the 64-bit Linux ABIs.
SCALARS = {
    **{t: (1, 1) for t in ["__u8", "__s8", "uint8_t", "int8_t", "char"]},
    **{t: (2, 2) for t in ["__u16", "__s16", "uint16_t", "int16_t"]},
    **{t: (4, 4) for t in ["__u32", "__s32", "uint32_t", "int32_t", "int"]},
    **{t: (8, 8) for t in ["__u64", "__s64", "uint64_t", "int64_t", "long", "__aligned_u64"]},
}
KEYWORDS = {"const", "volatile", "struct", "union", "enum", "unsigned", "signed"}

# Linux's _IOC encoding (asm-generic/ioctl.h): the macros, by name, as their
# direction (_IOC_WRITE 1, _IOC_READ 2) and whether they take a type.
IOC = {"_IO": (0, False), "_IOW": (1, True), "_IOR": (2, True), "_IOWR": (3, True)}
IOC_SIZE_BITS = 14


def blank(text):
    """[text] with its comments blanked out, newlines kept, so offsets hold."""
    return COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), text)


def matching(text, i, pair="{}"):
    """The index of the bracket that closes the one at [i]."""
    depth = 0
    for j in range(i, len(text)):
        depth += {pair[0]: 1, pair[1]: -1}.get(text[j], 0)
        if depth == 0:
            return j
    sys.exit("unbalanced brackets")


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


def arguments(expr, i):
    """The arguments of the call whose opening parenthesis is at [i], and the
    index after its closing one."""
    close = matching(expr, i, "()")
    args, level, start = [], 0, i + 1
    for j in range(i + 1, close):
        level += {"(": 1, ")": -1}.get(expr[j], 0)
        if expr[j] == "," and level == 0:
            args.append(expr[start:j].strip())
            start = j + 1
    args.append(expr[start:close].strip())
    return args, close + 1


class Model:
    """The definitions of the headers, by name, from the first header that
    defines each."""

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
        """[expr] with its macros expanded, _IOC's as their numbers and sizeof
        as the size of its type."""
        if depth > 64:
            sys.exit(f"recursive macro in {expr}")
        expr = re.sub(r"'(\\?.)'", lambda m: str(ord(m.group(1)[-1])), expr)
        out, i = [], 0
        for m in IDENT.finditer(expr):
            if m.start() < i:
                continue
            out.append(expr[i:m.start()])
            i = m.end()
            name = m.group()
            it = self.where.get(name)
            if name == "sizeof":
                j = expr.index("(", m.end())
                (ty,), i = arguments(expr, j)
                out.append(str(self.layout(self.type_name(ty))[0]))
            elif name in IOC:
                direction, typed = IOC[name]
                args, i = arguments(expr, expr.index("(", m.end()))
                type_, nr = (self.evaluate(a, name) for a in args[:2])
                size = self.layout(self.type_name(args[2]))[0] if typed else 0
                if size >= 1 << IOC_SIZE_BITS:
                    sys.exit(f"{expr}: parameters of {size} bytes")
                out.append(str((direction << 30) | (size << 16) | (type_ << 8) | nr))
            elif not it or it.kind != "define":
                out.append(name)
            elif it.params is None:
                out.append("(" + self.expand(it.body, depth + 1) + ")")
            else:
                args, i = arguments(expr, expr.index("(", m.end()))
                body = it.body
                for p, a in zip(it.params, args):
                    body = re.sub(rf"\b{p}\b", lambda _: f"({a})", body)
                out.append("(" + self.expand(body, depth + 1) + ")")
        out.append(expr[i:])
        return "".join(out)

    def type_name(self, ty):
        """The name of the type [ty], a C type name in parentheses or not."""
        words = [w for w in re.sub(r"[()]", " ", ty).split() if w not in KEYWORDS]
        if len(words) != 1:
            sys.exit(f"not a type: {ty}")
        return words[0]

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
        e = re.sub(r"\(\s*(?:const\s+)?(?:__[us]\d+|unsigned(?:\s+(?:long\s+long|long|int|char))?|int|long)\s*\)",
                   "", e)
        e = re.sub(r"\b(0[xX][0-9a-fA-F]+|\d+)[uUlL]+\b", r"\1", e)
        if re.search(r"[A-Za-z_][A-Za-z_0-9]*", re.sub(r"\b0[xX][0-9a-fA-F]+\b", "", e)):
            sys.exit(f"{name}: not a constant: {e}")
        return eval(e.replace("/", "//"), {"__builtins__": {}})

    # Layouts

    def scalar_or_type(self, name):
        if name in SCALARS:
            return SCALARS[name][0], SCALARS[name][1], {}
        return self.layout(name)

    def layout(self, name):
        """(bytes, alignment, fields) of the type [name]: fields by path, an
        array's as (offset, bytes of an element, elements), others as
        (offset, bytes)."""
        if name in SCALARS:
            return self.scalar_or_type(name)
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
        """The members of a struct's body: (name, type, counts), type a name
        or an inline (kind, body)."""
        out, level, start = [], 0, 0
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
        m = re.fullmatch(r"(struct|union)\s*\w*\s*(\{.*\})\s*(\w*)\s*((?:\[[^\]]*\]\s*)*)", s, re.S)
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
            ty = "__u64" if m.group(2) else " ".join(words)
            ty = {"unsigned int": "int", "unsigned char": "char", "unsigned long": "long"}.get(ty, ty)
            name, dims = m.group(3), m.group(4)
        counts = [self.count(d, where) for d in re.findall(r"\[([^\]]*)\]", dims)]
        return (name, ty, counts)

    def count(self, expr, where):
        if expr.strip() == "":
            return 0  # a flexible array member
        return self.evaluate(expr, where)

    def aggregate(self, kind, body, where):
        offset, align, fields = 0, 1, {}
        for name, ty, counts in self.members(body, where):
            if isinstance(ty, tuple):
                size, a, sub = self.aggregate(ty[0], ty[1], where)
            else:
                size, a, sub = self.scalar_or_type(ty)
            n = 1
            for c in counts:
                n *= c
            at = 0 if kind == "union" else (offset + a - 1) // a * a
            if counts:
                fields[name] = (at, size, n)
            else:
                prefix = f"{name}__" if name else ""
                if name:
                    fields[name] = (at, size)
                fields.update({prefix + k: (v[0] + at, *v[1:]) for k, v in sub.items()})
            offset = max(offset, at + size * n) if kind == "union" else at + size * n
            align = max(align, a)
        return (offset + align - 1) // align * align, align, fields

    # Excerpts

    def deps(self, name):
        """The names [name]'s definition refers to."""
        it = self.item(name)
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
    """The header's leading comments, to the one that holds its licence
    notice."""
    end = text.index("*/", text.index("Permission is hereby granted")) + 2
    return text[:end] + "\n"


def excerpts(texts):
    model = Model(texts)
    keep = model.closure(set(CONSTANTS) | set(REQUESTS) | set(STRUCTS) | set(SAME))
    out = {}
    for h in texts:
        if h not in keep:
            sys.exit(f"{h}: nothing read from it")
        body = []
        for it in sorted(keep[h].values(), key=lambda it: it.start):
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

ML_KEYWORDS = {"type", "function", "method", "val", "open", "end", "include", "module", "object", "private", "in"}


def snake(name):
    """The OCaml value for the C name [name]: KFD_IOC_EVENT_SIGNAL is
    kfd_ioc_event_signal and NotPresent not_present; nested fields a__b
    become a_b."""
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


def generate():
    model = Model({h: (HEADERS / h).read_text(encoding="latin-1") for h in SOURCES})
    out = [header(model.texts)]
    for c, like in SAME.items():
        if model.layout(c) != model.layout(like):
            sys.exit(f"{c} is not laid out as {like}")
    out.append("(* Constants. *)")
    out += [f"let {snake(c)} = {ml_int(model.value(c))}" for c in CONSTANTS]
    out.append("")
    out.append("(* Request numbers. *)")
    out += [f"let {snake(r)} = {ml_int(model.value(r))}" for r in REQUESTS]
    out.append("")
    out.append("(* Structs: each field is (byte offset, bytes), an array's (offset, bytes")
    out.append("   of an element, elements). *)")
    for c, (module, names) in STRUCTS.items():
        size, _, fields = model.layout(c)
        out.append(f"module {module} = struct")
        out.append(f"  let sizeof = {size}")
        for f in names:
            if f not in fields:
                sys.exit(f"{c} has no field {f}; it has {sorted(fields)}")
            out.append(f"  let {snake(f)} = {ml_tuple(fields[f])}")
        out.append("end")
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


def notice(text):
    """The lines of [text]'s leading comments, without their markers."""
    lines = []
    for l in licence(text).splitlines():
        l = re.sub(r"^\s*(/\*+|\*/|\*(?!/))?", "", l).removesuffix("*/").rstrip()
        lines.append(l.removeprefix(" "))
    return "\n".join(lines).strip("\n")


def header(texts):
    indent = lambda s: "\n".join(f"   {l}".rstrip() for l in s.splitlines())
    notices = "\n\n".join(f"   {h}:\n\n" + indent(notice(t)) for h, t in texts.items())
    return (
        "(*---------------------------------------------------------------------------\n"
        "  Copyright (c) 2026 The Raven authors. All rights reserved.\n"
        "  SPDX-License-Identifier: ISC\n"
        "  ---------------------------------------------------------------------------*)\n\n"
        "(* Generated by gen/gen.py from the excerpts in gen/headers; do not edit.\n"
        "   The command that regenerates this file is in gen/gen.py.\n\n"
        "   The values are Linux's, copied from its uapi headers under their\n"
        "   notices:\n\n" + notices + " *)\n\n"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--check", action="store_true", help="fail if a committed file differs")
    p.add_argument("--excerpt", action="store_true", help="make the excerpts from the pinned headers")
    p.add_argument("--pin", action="store_true", help="record the digests of headers not yet pinned")
    p.add_argument("--cache", type=pathlib.Path, default=pathlib.Path.home() / ".cache/raven/amdgpu-gen")
    a = p.parse_args()
    if a.excerpt:
        pins = json.loads(PINS.read_text()) if PINS.exists() else {}
        texts = {h: fetch(KERNEL + src, a.cache, pins, a.pin) for h, src in SOURCES.items()}
        files = {HEADERS / h: t for h, t in excerpts(texts).items()}
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
