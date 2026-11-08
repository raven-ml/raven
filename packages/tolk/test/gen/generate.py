# /// script
# requires-python = "==3.12.*"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Generate the goldens of tolk from tinygrad.

    uv run packages/tolk/test/gen/generate.py [--check] [MODULE...]

Each generator file `gen/<path>.py` runs in a fresh interpreter against
the tinygrad checkout, which must be clean and at TINYGRAD, with
`gen/tinygrad.patch` applied to a copy of it: the patch makes tinygrad's
rewrites keep IEEE and modular values, and its constants a NaN's bits, as
tolk's do, and its constant arithmetic compute what the README's CPython rows
say, with sin, log2, exp2 and pow correctly rounded by mpmath, so that the
goldens are the same on every host's C library. A float NaN's payload is
placed by C's conversion to and from a double, which keeps a quiet NaN's sign
and payload on x86 and Arm (not on RISC-V, which makes every converted NaN
canonical). It also removes `spec_hcq` from `uop/spec.py`, which tolk does
not port (see the README).
Moving TINYGRAD means re-applying it. MODULE, such as `dtype` or `uop/op`,
limits the run to those generators; by default all run.

The script pins CPython 3.12, whose `struct` packs a float beyond float32's
range as an infinity, which tinygrad's `bitcast` inherits; 3.13.16 raises
`OverflowError` there, so the goldens would depend on the interpreter.

A program tinygrad refuses, by raising while one is built or scheduled, loses
its goldens and no others: the run lists it with the error, and its goldens,
gone from the output, read as removed (a refused program fails --check).

Without --check, the goldens and the manifest are written, goldens that a
generator no longer declares are removed, and each change is listed: review
them with `git diff` before committing. With --check nothing is written: each
golden that would change is printed as a diff, and the run fails if any would.
"""

import argparse
import difflib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

TINYGRAD = "c22edff14c79310a9b2df62a78819cf77c8289fa"

# The generators recorded from unmodified tinygrad: the others run against a
# copy of the checkout with PATCH applied, until each part has moved here.
UNMODIFIED = {
    "codegen/opt/opt",
    "dtype",
    "helpers",
    "renderer/tc",
    "runtime/support/c",
    "runtime/support/support_memory",
    "setting",
    "support",
    "support/programs",
    "uop/divandmod",
    "uop/movement",
    "uop/op",
    "uop/render",
    "uop/uop_weak",
    "codegen/gpudims",
    "uop/spec",
    "runtime/support/compiler_cpu",
    "uop/symbolic",
    "renderer/renderer",
    "schedule/prepare",
    "device",
    "uop/ops",
    "codegen/late/gater",
    "codegen/decomp/decomp_op",
    "engine/realize",
    "schedule/memory",
    "codegen/opt/search",
    "schedule/indexing",
    "codegen/decomp/transcendental",
    "codegen/simplify",
    "codegen/decomp/decomp_dtype",
    "codegen/late/coalesce",
    "codegen/late/linearizer",
    "schedule/rangeify",
    "engine/jit",
    "schedule/schedule",
    "renderer/cstyle",
    "codegen/opt/postrange",
}

HERE = Path(__file__).resolve().parent
TEST = HERE.parent
MANIFEST = HERE / "manifest"
PATCH = HERE / "tinygrad.patch"
# The manifest's first line, naming the commit its goldens come from.
HEADER = f"# tinygrad {TINYGRAD}\n"

# The child runs the generator, then makes each golden in a process of its own,
# forked from the generator's, so that no golden sees the state another left in
# tinygrad (buffer numbering, caches). It writes the goldens to a file, out of
# reach of anything tinygrad prints.
#
# No golden compiles a program for a GPU, so the child gives every machine
# stand-ins for what a GPU renderer's compiler needs from the machine, when
# tinygrad first imports it: then, and not before, since a generator sets
# tinygrad's environment before importing it. A stand-in keeps tinygrad's cache
# key, the one compiler field a golden records, and compiles nothing. The same
# stand-ins on every machine keep the goldens the same on all of them.
#
# - A Metal renderer makes its compiler from `runtime/ops_metal.py`, which loads
#   the Objective-C runtime and Metal's frameworks, which macOS alone has; on
#   macOS the compiler's code generation service makes the forked processes
#   crash at random. The child replaces the module.
# - A CUDA renderer's `NVRTCCompiler` asks NVRTC its version off macOS, and on
#   macOS starts a compile server in a docker container, which can outlive the
#   generator and hold its output open. The child replaces its `__init__`, and
#   `Compiler.server`, which other compilers call on macOS, starts no server.
CHILD = """
import importlib.abc, importlib.machinery, json, runpy, sys, golden

class MetalCompilerModule(importlib.abc.Loader):
    def create_module(self, spec): return None
    def exec_module(self, module):
        from tinygrad.device import Compiler, CompileError
        class MetalCompiler(Compiler):
            def __init__(self): super().__init__("compile_metal_direct")
            def compile(self, src): raise CompileError("no golden compiles for Metal")
        module.MetalCompiler = MetalCompiler

def no_server(device):
    device.Compiler.server = lambda self, *args: None

def no_nvrtc(compiler_cuda):
    from tinygrad.device import Compiler, CompileError
    def __init__(self, arch, ptx=True, cache_key="cuda"):
        self.ptx, self.arch = ptx, arch
        Compiler.__init__(self, f"compile_{cache_key}_{arch}")
    def compile(self, src): raise CompileError("no golden compiles with NVRTC")
    compiler_cuda.NVRTCCompiler.__init__, compiler_cuda.NVRTCCompiler.compile = __init__, compile

PATCHED = {"tinygrad.device": no_server, "tinygrad.runtime.support.compiler_cuda": no_nvrtc}

class Stubs(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "tinygrad.runtime.ops_metal":
            return importlib.machinery.ModuleSpec(name, MetalCompilerModule())
        if name not in PATCHED: return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        exec_module = spec.loader.exec_module
        def stubbed(module):
            exec_module(module)
            PATCHED[name](module)
        spec.loader.exec_module = stubbed
        return spec

sys.meta_path.insert(0, Stubs())
runpy.run_path(sys.argv[1])
goldens = []
for name, body in golden.GOLDENS:
    text = golden.attempt(name, lambda: golden.forked(body))
    if text is not None: goldens.append((name, text))
with open(sys.argv[2], "w") as out:
    json.dump({"goldens": goldens, "refused": golden.REFUSED}, out)
"""


def git(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


def default_tinygrad():
    # The checkout sits in the main working tree, which every worktree shares.
    common = git(HERE, "rev-parse", "--path-format=absolute", "--git-common-dir")
    return Path(common).parent / "_tinygrad_next"


def check_checkout(tinygrad):
    head = git(tinygrad, "rev-parse", "HEAD")
    if head != TINYGRAD:
        sys.exit(f"{tinygrad} is at {head}, not at {TINYGRAD}")
    if git(tinygrad, "status", "--porcelain", "--untracked-files=no"):
        sys.exit(f"{tinygrad} has local changes")


def patched(tinygrad, scratch):
    """A copy of the checkout's HEAD in `scratch`, with PATCH applied."""
    tree = Path(scratch) / "tinygrad"
    tree.mkdir()
    archive = subprocess.run(["git", "-C", str(tinygrad), "archive", "HEAD"], capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(tree)], input=archive.stdout, check=True)
    subprocess.run(["git", "init", "-q", str(tree)], check=True)
    subprocess.run(["git", "-C", str(tree), "apply", str(PATCH)], check=True)
    return tree


def generators(modules):
    found = {path.relative_to(HERE).with_suffix("").as_posix(): path
             for path in sorted(HERE.rglob("*.py"))
             if path.parent != HERE or path.name not in ("generate.py", "golden.py", "graph.py", "gen_test.py")}
    unknown = [module for module in modules if module not in found]
    if unknown:
        sys.exit(f"no generator for {', '.join(unknown)}; generators: {', '.join(found)}")
    return {module: found[module] for module in (modules or found)}


def child_env(tinygrad):
    env = {key: os.environ[key] for key in ("PATH", "HOME", "TMPDIR", "LANG") if key in os.environ}
    # Scrub everything tinygrad reads (DEBUG, DEV, BEAM, ...) and fix string
    # hashing, so that a golden depends on tinygrad alone.
    env.update(PYTHONPATH=os.pathsep.join([str(tinygrad), str(HERE)]), PYTHONHASHSEED="0",
               PYTHONDONTWRITEBYTECODE="1", NO_COLOR="1")
    return env


def self_test(tinygrad):
    """Fail if the generators' machinery does not do what gen_test.py states."""
    result = subprocess.run([sys.executable, str(HERE / "gen_test.py")], cwd=tinygrad, env=child_env(tinygrad),
                            capture_output=True, text=True)
    if result.returncode:
        sys.exit(f"gen/gen_test.py failed:\n{result.stdout}{result.stderr}")


def run(module, generator, tinygrad):
    """The files of the goldens `generator` declares, by path under test/, and
    the programs tinygrad refused, as (name, reason)."""
    env = child_env(tinygrad)
    with tempfile.TemporaryDirectory(prefix="tolk-gen-") as scratch:
        out = Path(scratch) / "goldens.json"
        result = subprocess.run([sys.executable, "-c", CHILD, str(generator), str(out)], cwd=scratch,
                                env=env, capture_output=True, text=True)
        if result.returncode:
            sys.exit(f"gen/{module}.py failed:\n{result.stdout}{result.stderr}")
        declared = json.loads(out.read_text())
    files = {}
    for name, body in declared["goldens"]:
        path = f"{module}/{name}.golden"
        if path in files:
            sys.exit(f"gen/{module}.py declares {name} twice")
        files[path] = body + "\n"
    if not files:
        sys.exit(f"gen/{module}.py declares no golden")
    return files, declared["refused"]


def read(path):
    return path.read_text(encoding="utf-8") if path.is_file() else None


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="write nothing; fail if a golden would change")
    parser.add_argument("--tinygrad", type=Path, default=None, help="the tinygrad checkout")
    parser.add_argument("modules", nargs="*", metavar="MODULE")
    args = parser.parse_args()
    tinygrad = (args.tinygrad or default_tinygrad()).resolve()
    check_checkout(tinygrad)
    selected = generators(args.modules)
    manifest = read(MANIFEST) or HEADER
    if args.modules and not manifest.startswith(HEADER):
        sys.exit("the goldens are from another tinygrad commit: regenerate them all")
    goldens = {}
    with tempfile.TemporaryDirectory(prefix="tolk-tinygrad-") as scratch:
        tree = patched(tinygrad, scratch)
        self_test(tree)
        for module, generator in selected.items():
            files, refused = run(module, generator, tinygrad if module in UNMODIFIED else tree)
            goldens.update(files)
            for name, reason in refused:
                print(f"refused {module} {name}: {reason}")
    # A golden belongs to the generator of its directory. A full run owns every
    # golden, including those of a generator that no longer exists.
    recorded = set(manifest.splitlines()[1:])
    removed = sorted(path for path in recorded if path not in goldens
                     and (not args.modules or Path(path).parent.as_posix() in selected))
    kept = recorded - set(removed)
    files = {path: body for path, body in goldens.items() if read(TEST / path) != body}
    listed = HEADER + "".join(f"{path}\n" for path in sorted(kept | set(goldens)))
    if args.check:
        for path, body in files.items():
            before = read(TEST / path) or ""
            sys.stdout.writelines(difflib.unified_diff(
                before.splitlines(keepends=True), body.splitlines(keepends=True),
                fromfile=f"committed/{path}", tofile=f"generated/{path}"))
        for path in removed:
            print(f"would remove {path}")
        if listed != manifest:
            print("would rewrite gen/manifest")
        if files or removed or listed != manifest:
            sys.exit(1)
        print(f"{len(goldens)} goldens match tinygrad {TINYGRAD}")
        return
    for path, body in files.items():
        write(TEST / path, body)
        print(f"wrote {path}")
    for path in removed:
        (TEST / path).unlink(missing_ok=True)
        print(f"removed {path}")
    write(MANIFEST, listed)
    if not files and not removed:
        print(f"no golden changed ({len(goldens)} generated)")


if __name__ == "__main__":
    main()
