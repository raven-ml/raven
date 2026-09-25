"""Generate AMD operation packets with reference HCQ2 builders, without hardware.

The real committed hsaco supplies program metadata. Retired compute write/poll
and owner-specific timeline helpers are not copied into this driver.
"""
import argparse
import importlib
import os
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tinygrad", type=Path, default=HERE.parents[4] / "_tinygrad_target")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.tinygrad.resolve()))
for key in ("DEBUG", "VIZ", "PROFILE", "SQTT", "PMC", "WAVES_PER_SH", "IOCTL", "DEV"):
    os.environ.pop(key, None)

from tinygrad.runtime.ops_amd import AMDComputeQueue, AMDSDMAQueue, AMDDevice, amd_build_program
from tinygrad.runtime.support.amd import AMDIP, import_module, import_soc
from tinygrad.dtype import dtypes
from tinygrad.helpers import Target
from tinygrad.uop.ops import UOp, Ops, KernelInfo, ProgramInfo

DEVS = ("AMD",)
ADDRESSES = {}


def allocation(size, dtype, address):
    buf = UOp.placeholder((size,), dtype, device=DEVS)
    ADDRESSES[buf.getaddr(DEVS)] = UOp.const(address, dtypes.uint64)
    return buf


SIGNAL = allocation(2, dtypes.uint64, 0x400000)
WRITE = allocation(2, dtypes.uint32, 0x600000)
SRC = allocation(1, dtypes.uint8, 0x10000000)
DST = allocation(1, dtypes.uint8, 0x20000000)
BUFFERS = [allocation(32, dtypes.int32, address) for address in (0x900000, 0xA00000, 0xB00000)]
PARAMS = [UOp.param(i, dtypes.int32, shape=(32,), device="AMD") for i in range(3)]
VAR = UOp.variable("n", 1, 32, dtype=dtypes.int32)
FORMAL = VAR.replace(op=Ops.PARAM)
BINARY = (HERE.parents[1] / "fixtures/amd/simple_add_gfx1100.hsaco").read_bytes()
PROGRAM = UOp(Ops.PROGRAM, src=(UOp(Ops.SINK, arg=KernelInfo(name="simple_add")),
    UOp(Ops.LINEAR, src=tuple(PARAMS + [FORMAL])), UOp(Ops.SOURCE, arg=""), UOp(Ops.BINARY, arg=BINARY)),
    arg=ProgramInfo(global_size=(4, 3, 2), local_size=(8, 4, 1), globals=(0, 1, 2), vars=(FORMAL,), target=Target("AMD", arch="gfx1100")))
CALL = PROGRAM.call(*BUFFERS, VAR.bind(32))


class FakeDev:
    """AMDDevice stand-in: the queue builders read exactly these fields.

    Module/IP wiring follows AMDDevice.__init__ for the selected chip.
    Driver construction and publication are outside this encoding fixture.
    """

    def __init__(self, target, xccs, gc_ver, nbio_ver, sdma_ver):
        self.target, self.xccs = target, xccs
        gfx9 = target[0] == 9
        ip_off = importlib.import_module(
            f"tinygrad.runtime.autogen.am.{'vega' if gfx9 else 'navi'}_offsets")
        self.soc = import_soc(target)
        self.pm4 = importlib.import_module(
            f"tinygrad.runtime.autogen.am.pm4_{'soc15' if gfx9 else 'nv'}")
        self.sdma = import_module('sdma', min(sdma_ver, (6, 0, 0)))
        self.gc = AMDIP('gc', gc_ver, bases={
            i: tuple(getattr(ip_off, f'GC_BASE__INST{i}_SEG{s}', 0) for s in range(6))
            for i in range(6)})
        self.nbio = AMDIP('nbio' if target[0] < 12 else 'nbif', nbio_ver, bases={
            i: tuple(getattr(ip_off, f'NBIO_BASE__INST{i}_SEG{s}', 0) for s in range(9))
            for i in range(6)})
        self.max_copy_size = 0x40000000 if (4, 4, 2) <= sdma_ver < (5, 0, 0) or sdma_ver >= (5, 2, 0) else 0x400000
        self.sqtt_enabled = False
        self.tmpring_size = lambda size: AMDDevice.tmpring_size(self, size)
        self.iface = SimpleNamespace(props={"lds_size_in_kb": 64, "max_slots_scratch_cu": 4})
        self.cu_cnt, self.se_cnt = 8, 2
        self.pmc_enabled = False


# IP versions as the KFD topology reports them for each chip; the module
# resolution (import_module / import_asic_regs) then picks the same autogen
# tables AMDDevice.__init__ would.
CHIPS = {
    'gfx1100': FakeDev(target=(11, 0, 0), xccs=1, gc_ver=(11, 0, 0),
                       nbio_ver=(4, 3, 0), sdma_ver=(6, 0, 0)),
    'gfx942': FakeDev(target=(9, 4, 2), xccs=8, gc_ver=(9, 4, 3),
                      nbio_ver=(7, 9, 0), sdma_ver=(4, 4, 2)),
}


def queue(cls, dev):
    q = object.__new__(cls)
    q.dev, q.devs, q.blob, q.patches = dev, DEVS, bytearray(), []
    q.pm4, q.gc, q.soc, q.nbio, q.target = dev.pm4, dev.gc, dev.soc, dev.nbio, dev.target
    q.profiled, q.sdma, q.max_copy_size = [], dev.sdma, dev.max_copy_size
    return q


def words(blob, patches):
    result = bytearray(blob)
    for offset, value in patches:
        # Kernarg and scratch allocations are symbolic input addresses. Bind
        # their GETADDR leaves; no packet fields are rewritten or normalized.
        bindings = dict(ADDRESSES)
        for node in value.toposort():
            if node.op is Ops.GETADDR:
                base = node.src[0]
                if base.op is Ops.LINEAR and base.arg == "kernargs":
                    bindings[node] = UOp.const(0x300000, dtypes.uint64)
                elif base.tag == "scratch":
                    bindings[node] = UOp.const(0x200000, dtypes.uint64)
        scalar = value.substitute(bindings).ssimplify()
        assert isinstance(scalar, int), (offset, value, scalar)
        width = value.dtype.itemsize
        result[offset:offset + width] = (scalar & ((1 << (8 * width)) - 1)).to_bytes(width, "little")
    assert len(result) % 4 == 0
    return struct.unpack(f"<{len(result) // 4}I", result)


def streams(dev):
    out = {}
    def build(name, cls, action):
        q = queue(cls, dev)
        action(q)
        out[name] = words(q.blob, q.patches)
    c32 = lambda value: UOp.const(value, dtypes.uint32)
    c64 = lambda value: UOp.const(value, dtypes.uint64)
    _, image = amd_build_program(dev, PROGRAM, DEVS)
    ADDRESSES[image.getaddr(DEVS)] = c64(0x100000)
    build("exec", AMDComputeQueue, lambda q: q.exec(CALL, PROGRAM))
    build("signal", AMDComputeQueue, lambda q: q.signal(SIGNAL, c32(0x42)))
    build("wait", AMDComputeQueue, lambda q: q.wait(SIGNAL, c32(0x42)))
    build("timestamp", AMDComputeQueue, lambda q: q.timestamp(SIGNAL))
    build("memory_barrier", AMDComputeQueue, lambda q: q.memory_barrier())
    build("sdma_copy_small", AMDSDMAQueue, lambda q: q.copy(DST, SRC, 0x1000))
    build("sdma_copy_large", AMDSDMAQueue, lambda q: q.copy(DST, SRC, 2 * dev.max_copy_size + 0x400))
    build("sdma_signal", AMDSDMAQueue, lambda q: q.signal(SIGNAL, c32(0x42)))
    build("sdma_wait", AMDSDMAQueue, lambda q: q.wait(SIGNAL, c32(0x42)))
    build("sdma_timestamp", AMDSDMAQueue, lambda q: q.timestamp(SIGNAL))
    build("sdma_write32", AMDSDMAQueue, lambda q: q.write(WRITE, c32(0x12345678)))
    build("sdma_write64", AMDSDMAQueue, lambda q: q.write(WRITE, c64(0x1122334455667788)))
    return out


args.output.mkdir(parents=True, exist_ok=True)
for chip, dev in CHIPS.items():
    DEVS = (f"AMD:{chip}",)
    for buf, address in [(SIGNAL, 0x400000), (WRITE, 0x600000), (SRC, 0x10000000), (DST, 0x20000000), *zip(BUFFERS, (0x900000, 0xA00000, 0xB00000))]:
        ADDRESSES[buf.getaddr(DEVS)] = UOp.const(address, dtypes.uint64)
    for name, stream in streams(dev).items():
        path = args.output / f"{name}_{chip}.expected"
        path.write_text("".join(f"{word:08x}\n" for word in stream))
        print(f"wrote {path} ({len(stream)} dwords)")
