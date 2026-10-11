# /// script
# requires-python = ">=3.10"
# ///
"""Makes headers/, the excerpts of Linux's uapi headers that
rig_amd_amdgpu_stubs.c compiles against: the requests of KFD (/dev/kfd) and
of amdgpu's render node the library makes, the structures of their
parameters and the flags they take.

Run from the worktree root:

  uv run packages/rig/lib/amd/amdgpu/gen/gen.py
  uv run packages/rig/lib/amd/amdgpu/gen/gen.py --check

Each excerpt is a header's licence notice and the definitions NAMES lists,
verbatim and in the header's order, with the definitions they depend on, as a
64-bit Linux build compiles them. The headers are pinned in pins.json by URL
and SHA-256 and checked against their pins; downloads are kept in --cache,
and --pin records the digests of headers not yet pinned. --check makes the
excerpts into memory and fails if a committed file differs.

The stubs compile against the excerpts, never the build machine's headers,
which may predate a definition they use.

Text is read and written as latin-1, one character per byte, so every byte
of a header round-trips into its excerpt as upstream wrote it.
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

KERNEL = "https://raw.githubusercontent.com/ROCm/ROCK-Kernel-Driver/33970e1351f5e511029602454979f3de7e22260f/"

# Each excerpt and the header it is cut from. A name is read from the first
# header that defines it.
SOURCES = {
    "kfd_ioctl.h": "include/uapi/linux/kfd_ioctl.h",
    "amdgpu_drm.h": "include/uapi/drm/amdgpu_drm.h",
    "drm.h": "include/uapi/drm/drm.h",
}

# The definitions the stubs use.
NAMES = [
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
    # requests
    "AMDKFD_IOC_GET_VERSION", "AMDKFD_IOC_CREATE_QUEUE", "AMDKFD_IOC_DESTROY_QUEUE", "AMDKFD_IOC_CREATE_EVENT",
    "AMDKFD_IOC_DESTROY_EVENT", "AMDKFD_IOC_RESET_EVENT", "AMDKFD_IOC_WAIT_EVENTS", "AMDKFD_IOC_ACQUIRE_VM",
    "AMDKFD_IOC_ALLOC_MEMORY_OF_GPU", "AMDKFD_IOC_FREE_MEMORY_OF_GPU", "AMDKFD_IOC_MAP_MEMORY_TO_GPU",
    "AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU", "AMDKFD_IOC_RUNTIME_ENABLE",
    "DRM_IOCTL_AMDGPU_INFO", "DRM_IOCTL_AMDGPU_CTX",
    # structures
    "kfd_ioctl_get_version_args", "kfd_ioctl_create_queue_args", "kfd_ioctl_destroy_queue_args",
    "kfd_ioctl_create_event_args", "kfd_ioctl_destroy_event_args", "kfd_ioctl_reset_event_args",
    "kfd_ioctl_wait_events_args", "kfd_event_data", "kfd_ioctl_acquire_vm_args",
    "kfd_ioctl_alloc_memory_of_gpu_args", "kfd_ioctl_free_memory_of_gpu_args",
    "kfd_ioctl_map_memory_to_gpu_args", "kfd_ioctl_unmap_memory_from_gpu_args", "kfd_ioctl_runtime_enable_args",
    "drm_amdgpu_info", "drm_amdgpu_info_device", "drm_amdgpu_ctx",
]

# C headers

COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(\w+)(\([\w\s,]*\))?((?:[^\n]*\\\n)*[^\n]*)$", re.M)
TYPEDEF = re.compile(r"^[ \t]*typedef\b", re.M)
TAGGED = re.compile(r"^[ \t]*(struct|union|enum)[ \t]+(\w+)\s*\{", re.M)
IDENT = re.compile(r"\b[A-Za-z_]\w*\b")
DIRECTIVE = re.compile(r"^[ \t]*#[ \t]*(if|ifdef|ifndef|elif|else|endif|define|undef)\b(.*)$")

# The macros a 64-bit Linux build of a program defines that the headers'
# conditionals test; every other name a conditional tests is undefined.
PLATFORM = {"__linux__", "__LP64__"}


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
        body = m.group(3).replace("\\\n", " ").strip()
        out.append(Item("define", [m.group(1)], m.start(), m.end(), body=body))
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
    return [Item("constant", [e[0].strip()], enum.start, enum.end, enum=enum) for e in entries]


class Model:
    """The definitions of the headers, by name, from the first header that
    defines each."""

    def __init__(self, texts):
        self.where = {}
        for h, text in texts.items():
            for it in items(text):
                it.header = h
                for n in it.names:
                    if n in self.where and self.where[n].header == h:
                        old = self.where[n]
                        if text[old.start:old.end] != text[it.start:it.end] and it.kind == "define":
                            sys.exit(f"{h}: {n} is defined twice")
                    self.where.setdefault(n, it)

    def deps(self, name):
        """The names [name]'s definition refers to."""
        it = self.where[name]
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


# Excerpts

def licence(text):
    """The header's leading comments, to the one that holds its licence
    notice."""
    end = text.index("*/", text.index("Permission is hereby granted")) + 2
    return text[:end] + "\n"


def excerpts(texts):
    keep = Model(texts).closure(set(NAMES))
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


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--check", action="store_true", help="fail if a committed file differs")
    p.add_argument("--pin", action="store_true", help="record the digests of headers not yet pinned")
    p.add_argument("--cache", type=pathlib.Path, default=pathlib.Path.home() / ".cache/raven/amdgpu-gen")
    a = p.parse_args()
    pins = json.loads(PINS.read_text()) if PINS.exists() else {}
    texts = {h: fetch(KERNEL + src, a.cache, pins, a.pin) for h, src in SOURCES.items()}
    files = {HEADERS / h: t for h, t in excerpts(texts).items()}
    if a.pin:
        files[PINS] = json.dumps(dict(sorted(pins.items())), indent=1) + "\n"
    stale = [f for f, text in files.items() if not f.exists() or f.read_text(encoding="latin-1") != text]
    if a.check:
        if stale:
            sys.exit("stale: " + ", ".join(str(f.relative_to(HERE)) for f in stale))
        return
    for f in stale:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(files[f], encoding="latin-1")


if __name__ == "__main__":
    main()
