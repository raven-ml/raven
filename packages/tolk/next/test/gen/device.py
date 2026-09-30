"""Goldens of tinygrad/device.py: the renderer a device picks for a DEV
setting, and the compiled programs of `UOp.to_elf`.

The table `renderers` gives, for each DEV setting, device and architecture
the device reports, the target `Compiled._select_renderer` renders for and its
outcome: the renderer it picks (`ok`), the error of `select_by_name` when the
target names a renderer the device lacks (`no renderer`), or the failure of
the renderer itself on the target (`renderer fails`), whose text is Python's.
A device lists the renderers of its `Compiled.__init__` (`runtime/ops_*.py`)
that tolk.next ports.

Each graph golden `<case>_program` is a program that `to_program` compiles for
a C-family renderer. For each, the table `elfs` gives what `to_elf` makes of
it, `signatures` each parameter of its signature, and `layouts` where
`TinyELF.iter_sig` packs them. The renderer's compiler is the empty `Compiler`,
which takes the source as the binary, so that the goldens depend on no
toolchain. The table `reprs` holds the repr of compiled programs of chosen
fields.
"""

import tinygrad.runtime.support.compiler_amd as compiler_amd
import tinygrad.runtime.support.compiler_cuda as compiler_cuda
import tinygrad.runtime.ops_metal as ops_metal

# Making a renderer makes its compiler, which loads its library in tinygrad and
# nothing in tolk.next (D15): comgr and NVRTC are absent where the goldens are
# generated, and on macOS NVRTC starts a compile server in docker. Metal's code
# generation service makes the forked processes crash at random.
compiler_amd.c.DLL._loaded_.add(compiler_amd.comgr.dll.nm)
compiler_cuda.NVRTCCompiler.__init__ = lambda self, arch, ptx=True, cache_key="cuda": None
ops_metal.MetalCompiler.support.MTLCodeGenServiceCreate = lambda name: None

from golden import graph, table
from graph import kernels
from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.device import Compiled, Compiler, TinyELF
from tinygrad.helpers import DEV, Context, Target
from tinygrad.renderer.cstyle import ClangRenderer, CUDARenderer, HIPRenderer, MetalRenderer
from tinygrad.uop.ops import KernelInfo, UOp

# Renderers

RENDERERS = {"CPU": [ClangRenderer], "METAL": [MetalRenderer], "CUDA": [CUDARenderer], "NV": [CUDARenderer],
             "AMD": [HIPRenderer]}

# The architecture each device reports, and none.
ARCHES = {"CPU": "arm64,apple-m1", "METAL": "Apple9", "CUDA": "sm_89", "NV": "sm_80", "AMD": "gfx1100"}

DEVS = [
    "",
    "CPU", "METAL", "CUDA", "NV", "AMD",
    "CPU:CLANG", "METAL:METAL", "CUDA:CUDA", "NV:CUDA", "AMD:HIP",
    ":CLANG", ":HIP", "::gfx942",
    "CPU::x86_64,znver2", "METAL::Apple6", "CUDA::sm_75", "NV::sm_90", "AMD::gfx950",
    "CPU:TYPO", "CPU:CLANGJIT", "CPU:LLVM", "CUDA:PTX", "NV:CUD", "AMD:HIPP", "METAL:METL",
    "CPU::sparc,v9", "CPU::arm64", "CPU::riscv64,native", "METAL::AppleX", "CUDA::sm_x", "CUDA::gfx1100",
    "AMD:HIP;CPU:CLANG", "PCI:1+AMD", "USB+AMD::gfx1201;CPU::x86_64,x86-64", "QCOM;NV::sm_86",
]


def select(dev, device, arch):
    """The renderer `Compiled._select_renderer` picks under DEV=`dev` for
    `device` reporting `arch`, on a device that holds nothing else."""
    d = object.__new__(Compiled)
    d.device, d.arch, d.renderers, d.cached_renderer = device, arch or None, RENDERERS[device], {}
    with Context(DEV=dev):
        target = DEV.target(device, **({"arch": arch} if arch else {}))
        try:
            r = d._select_renderer()
        except Exception as e:
            if str(e).startswith(f"{device} has no renderer"): return target, "", "no renderer", str(e)
            # Each device has one renderer, whose own failure is raised as it is.
            return target, d._renderer_name(RENDERERS[device][0]), "renderer fails", ""
    return r.target, d._renderer_name(type(r)), "ok", ""


@table
def renderers():
    rows = []
    for dev in DEVS:
        for device in RENDERERS:
            for arch in ("", ARCHES[device]):
                target, name, outcome, error = select(dev, device, arch)
                rows.append((dev, device, arch, repr(target), name, outcome, error))
    return ["dev", "device", "arch", "target", "renderer", "outcome", "error"], rows


# Programs

CLANG = Target("CPU", "CLANG", "x86_64,x86-64")
METAL = Target("METAL", "METAL", "Apple9")


def renderer(target):
    r = {"CPU": ClangRenderer, "METAL": MetalRenderer}[target.device](target)
    r.compiler = Compiler()
    return r


def elementwise():
    a, b = Tensor.empty(16, device="CPU"), Tensor.empty(16, device="CPU", dtype=dtypes.half)
    (kernel,) = kernels(a + b.float())
    return kernel


def sparse():
    """Buffers in the sparse slots 2 and 7, and a variable."""
    out, inp = UOp.param(2, dtypes.int, 1), UOp.param(7, dtypes.int, 1)
    n = UOp.variable("increment", 0, 100, dtype=dtypes.int)
    return out.index(UOp.const(0, dtypes.int)).store(inp.index(UOp.const(0, dtypes.int)).load() + n) \
        .sink(arg=KernelInfo(name="sparse"))


def scalars():
    """A named buffer and scalars of three widths, under a name to escape."""
    a = UOp.param(0, dtypes.long, 4, name="acc")
    b = UOp.param(1, dtypes.char, 4)
    small = UOp.variable("small", 0, 8, dtype=dtypes.char)
    wide = UOp.variable("wide", 0, 2**40, dtype=dtypes.long)
    mid = UOp.variable("mid", 0, 1000, dtype=dtypes.int)
    r = UOp.range(4, 0)
    value = b.index(r).load().cast(dtypes.long) + small.cast(dtypes.long) + wide + mid.cast(dtypes.long)
    return a.index(r).store(value).end(r).sink(arg=KernelInfo(name="r:scalars 2"))


def no_buffer_read():
    """A program of one buffer, written only."""
    a = UOp.param(3, dtypes.float, 8)
    r = UOp.range(8, 0)
    return a.index(r).store(r.cast(dtypes.float)).end(r).sink(arg=KernelInfo(name="fill"))


PROGRAMS = {
    "elementwise": (elementwise, CLANG),
    "elementwise_metal": (elementwise, METAL),
    "sparse": (sparse, CLANG),
    "scalars": (scalars, CLANG),
    "fill": (no_buffer_read, METAL),
}


def program(case):
    make, target = PROGRAMS[case]
    return to_program(make(), renderer(target))


@table
def elfs():
    rows = []
    for case in PROGRAMS:
        elf = program(case).to_elf()
        rows.append((case, elf.name, repr(elf.target), len(elf.signature), len(elf.lib)))
    return ["program", "name", "target", "params", "lib_bytes"], rows


@table
def signatures():
    """Each parameter of each program's signature, and its repr."""
    rows = []
    for case in PROGRAMS:
        for position, param in enumerate(program(case).to_elf().signature):
            name, slot, dt, shape = param
            rows.append((case, position, repr(name), slot, dt, repr(shape), repr(param)))
    return ["program", "position", "name", "slot", "dtype", "shape", "repr"], rows


@table
def layouts():
    """Where `iter_sig` packs each program's signature from three offsets."""
    rows = []
    for case in PROGRAMS:
        sig = program(case).to_elf().signature
        for offset in (0, 3, 8):
            for position, (at, dt) in enumerate(TinyELF.iter_sig(sig, offset)):
                rows.append((case, offset, position, at, dt))
    return ["program", "offset", "position", "at", "dtype"], rows


@table
def reprs():
    """The repr of compiled programs of chosen fields."""
    sig = ((None, 0, dtypes.float, (16,)), ("n", 1, dtypes.int, ()))
    elfs = {
        "some": TinyELF(b"\x7fELF\x00'\"", "k", CLANG, sig, b"key\n"),
        "none": TinyELF(b"", "E_4", Target("AMD", "HIP", "gfx1100", "PCI", "1"), sig[:1], None),
        "empty": TinyELF(b"lib", "k", Target(), (), None),
    }
    return ["case", "repr"], [(case, repr(elf)) for case, elf in elfs.items()]


def declare(case):
    def fn(): return program(case)
    fn.__name__ = f"{case}_program"
    graph(fn)


for case in PROGRAMS:
    declare(case)
