"""The machinery the device definitions generators share.

A vendor's gen.py names its pinned inputs and what the runtime reads of them;
this module fetches and verifies the inputs, lays C definitions out with
libclang, writes OCaml, and runs the command line: generate, --check (generate
into a temporary directory and fail if the committed files differ) and --pin
(record the digests of inputs not yet pinned).
"""

import argparse
import hashlib
import json
import pathlib
import re
import sys
import tempfile
import urllib.request

# Fetching


# The URLs this run read, whose pins --pin keeps.
USED = set()


def key(url):
    """The digest of [url] that names what the cache keeps of it."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def fetch(cache, url, pins, pin):
    """The path of [url]'s contents in [cache], verified against [pins]."""
    USED.add(url)
    path = cache / (key(url) + "-" + url.rsplit("/", 1)[-1])
    if not path.exists():
        print(f"fetching {url}", file=sys.stderr)
        req = urllib.request.Request(url, headers={"User-Agent": "raven-gen"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        path.write_bytes(data)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if url in pins and pins[url] != digest:
        sys.exit(f"{url}: SHA-256 {digest}, pinned {pins[url]}")
    if url not in pins:
        if not pin:
            sys.exit(f"{url} is not pinned; run with --pin to record {digest}")
        pins[url] = digest
    return path

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
#define bool _Bool
#define true 1
#define false 0
#define __packed __attribute__((packed))
#define __user
#define BIT(n) (1UL << (n))
#define BIT_ULL(n) (1ULL << (n))
"""


def stub_dir(vendor):
    """A directory for the empty headers that stand in for missing includes."""
    return pathlib.Path(tempfile.mkdtemp(prefix=f"nx-{vendor}-gen-stub-"))


class Unit:
    """A translation unit of headers, parsed as C (or C++) for x86_64 Linux.
    Includes that cannot be found are replaced by empty files in [stub]."""

    def __init__(self, ci, headers, includes, stub, target="x86_64-unknown-linux-gnu", cpp=False, extra="",
                 defines=()):
        self.ci = ci
        self.headers, self.includes, self.stub, self.target, self.cpp = headers, includes, stub, target, cpp
        self.defines = list(defines)
        self.src = PRELUDE + extra + "".join(f'#include "{h}"\n' for h in headers)
        self.tu = self.parse(self.src)

    def parse(self, src):
        args = ["-x", "c++" if self.cpp else "c", "-target", self.target, "-nostdinc", "-ferror-limit=0", "-I", str(self.stub)] \
            + [f"-D{d}" for d in self.defines] + [f"-I{i}" for i in self.includes]
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
        return None


def layout(ci, cursor):
    """(sizeof, {path: (byte offset, width) or ("bits", bit offset, width)}).
    An array's width is its element's."""
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

# Emission


def ml_name(c):
    return c.lower()


def ml_int(v):
    return f"0x{v:x}" if v > 9 else str(v)


def ml_field(v):
    if v[0] == "bits":
        return f"({v[1]}, {v[2]})"
    return f"({v[0]}, {v[1]})"


def ml_version(v):
    return "(%d, %d, %d)" % v


def struct_module(out, module, size, fields, wanted, sig=None):
    out.append(f"module {module}{' : ' + sig if sig else ''} = struct")
    out.append(f"  let sizeof = {size}")
    for f in (wanted if wanted is not None else sorted(fields)):
        if f not in fields:
            sys.exit(f"{module} has no field {f}")
        v = fields[f]
        suffix = "_bits" if v[0] == "bits" else ""
        out.append(f"  let {f}{suffix} = {ml_field(v)}")
    out.append("end")
    out.append("")

# Command line


def main(doc, generate, outputs, here, out, cache):
    """Runs a generator. [generate cache pins pin outdir] writes [outputs]
    into [outdir]; [here] holds pins.json, [out] the committed outputs, and
    [cache] is the default download cache."""
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=pathlib.Path, default=pathlib.Path.home() / ".cache/raven" / cache)
    ap.add_argument("--check", action="store_true", help="fail if the committed files differ")
    ap.add_argument("--pin", action="store_true", help="record the digests of inputs not yet pinned")
    args = ap.parse_args()
    args.cache = args.cache.resolve()
    args.cache.mkdir(parents=True, exist_ok=True)
    pins_file = here / "pins.json"
    pins = json.loads(pins_file.read_text()) if pins_file.exists() else {}
    if args.check:
        with tempfile.TemporaryDirectory() as d:
            generate(args.cache, pins, False, pathlib.Path(d))
            for f in outputs:
                if (pathlib.Path(d) / f).read_text() != (out / f).read_text():
                    sys.exit(f"{f} differs from what gen.py generates")
        print("up to date")
        return
    generate(args.cache, pins, args.pin, out)
    if args.pin:
        pins = {u: d for u, d in pins.items() if u in USED}
        pins_file.write_text(json.dumps(pins, indent=1, sort_keys=True) + "\n")
