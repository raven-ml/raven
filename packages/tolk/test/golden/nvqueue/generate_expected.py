"""Render NV operation packets and QMDs through the reference HCQ2 builders.

No GPU is opened. The committed cubin supplies real program metadata; symbolic
allocation addresses are bound to fixed fixture addresses after encoding.
Retired raw write/poll helpers are deliberately absent. No expected/actual file
is read, and output must be placed explicitly for independent review.
"""
import argparse
from pathlib import Path
import os
import struct
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tinygrad", type=Path, default=HERE.parents[4] / "_tinygrad_target")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.tinygrad.resolve()))
for key in ("DEBUG", "VIZ", "PROFILE", "PMA", "IOCTL", "DEV"):
    os.environ.pop(key, None)

from tinygrad.dtype import dtypes
from tinygrad.helpers import Target, data64, round_up
from tinygrad.runtime.ops_nv import QMD, NVComputeQueue, NVCopyQueue, nv_build_program, nv_gpu, nvm
from tinygrad.uop.ops import UOp, Ops, KernelInfo, ProgramInfo

DEVS = ("NV",)
GLOBAL_SIZE, LOCAL_SIZE = (4, 3, 2), (8, 4, 1)
SIGNAL_VALUE, DMA_SIGNAL_VALUE = 0x100000042, 0x42
SHARED_WINDOW, LOCAL_WINDOW = 0x729400000000, 0x729300000000
IMAGE_ADDRESS, QMD_ADDRESS = 0x100000, 0x300000
ADDRESSES = {}


def allocation(size, dtype, address):
    buf = UOp.placeholder((size,), dtype, device=DEVS)
    ADDRESSES[buf.getaddr(DEVS)] = UOp.const(address, dtypes.uint64)
    return buf


SIGNAL = allocation(2, dtypes.uint64, 0x400000)
SRC = allocation(1, dtypes.uint8, 0x10000000)
DST = allocation(1, dtypes.uint8, 0x20000000)
BUFFERS = [allocation(32, dtypes.float32, address) for address in (0x900000, 0xA00000, 0xB00000)]
PARAMS = [UOp.param(i, dtypes.float32, shape=(32,), device="NV") for i in range(3)]
VAR = UOp.variable("n", 1, 32, dtype=dtypes.int32)
FORMAL = VAR.replace(op=Ops.PARAM)
BINARY = (HERE.parents[1] / "fixtures/nv/simple_add_sm89.cubin").read_bytes()
PROGRAM = UOp(Ops.PROGRAM, src=(UOp(Ops.SINK, arg=KernelInfo(name="simple_add")),
    UOp(Ops.LINEAR, src=tuple(PARAMS + [FORMAL])), UOp(Ops.SOURCE, arg=""), UOp(Ops.BINARY, arg=BINARY)),
    arg=ProgramInfo(global_size=GLOBAL_SIZE, local_size=LOCAL_SIZE, globals=(0, 1, 2), vars=(FORMAL,), target=Target("NV", arch="sm_89")))
CALL = PROGRAM.call(*BUFFERS, VAR.bind(32))


def queue(cls, dev):
    # Isolate encoding from HWQueue's device lookup and publication. All packet,
    # program parsing, descriptor, argument and chaining methods are upstream.
    q = object.__new__(cls)
    q.dev, q.devs, q.blob, q.patches = dev, DEVS, bytearray(), []
    if cls is NVComputeQueue:
        data, image = nv_build_program(dev, PROGRAM, DEVS)
        ADDRESSES[image.getaddr(DEVS)] = UOp.const(IMAGE_ADDRESS, dtypes.uint64)
        q.qmd_sz = round_up(QMD(dev).sz * 4, 256)
        q.stride = q.qmd_sz + data.kernargs_size
        q.qmd_buf = allocation(2 * q.stride, dtypes.uint8, QMD_ADDRESS)
        q.qmds, q.prev_qmd = [], None
    return q


def words(blob, patches):
    result = bytearray(blob)
    for offset, value in patches:
        scalar = value.substitute(ADDRESSES).ssimplify()
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
        return q
    c64 = lambda value: UOp.const(value, dtypes.uint64)
    # Exact raw setup inputs used by target NVDevice.fifos and SLM growth.
    build("setup", NVComputeQueue, lambda q: q.q(
        *nvm(1, nv_gpu.NVC6C0_SET_OBJECT, dev.iface.compute_class),
        *nvm(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A, *data64(LOCAL_WINDOW)),
        *nvm(1, nv_gpu.NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A, *data64(SHARED_WINDOW))))
    build("setup_local_mem", NVComputeQueue, lambda q: q.q(
        *nvm(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_A, *data64(0x800000)),
        *nvm(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A, *data64(0x30000), 0xff)))
    build("memory_barrier", NVComputeQueue, lambda q: q.memory_barrier())
    build("wait", NVComputeQueue, lambda q: q.wait(SIGNAL, c64(0x42)))
    build("timestamp", NVComputeQueue, lambda q: q.timestamp(SIGNAL))
    build("signal_no_qmd", NVComputeQueue, lambda q: q.signal(SIGNAL, c64(SIGNAL_VALUE)))
    q = build("exec", NVComputeQueue, lambda q: q.exec(CALL, PROGRAM))
    out["exec_qmd"] = words(q.qmds[0].mv, q.qmds[0].patches.items())[:QMD(dev).sz]
    q = build("exec_chained", NVComputeQueue, lambda q: (q.exec(CALL, PROGRAM), q.exec(CALL, PROGRAM)))
    for i, qmd in enumerate(q.qmds):
        out[f"exec_chained_qmd{i}"] = words(qmd.mv, qmd.patches.items())[:QMD(dev).sz]
    q = build("signal_after_exec", NVComputeQueue, lambda q: (q.exec(CALL, PROGRAM), q.signal(SIGNAL, c64(SIGNAL_VALUE))))
    out["signal_after_exec_qmd"] = words(q.qmds[0].mv, q.qmds[0].patches.items())[:QMD(dev).sz]
    data, _ = nv_build_program(dev, PROGRAM, DEVS)
    out["qmd_init"] = words(data.qmd.mv, data.qmd.patches.items())
    build("dma_setup", NVCopyQueue, lambda q: q.q(*nvm(4, nv_gpu.NVC6C0_SET_OBJECT, dev.iface.dma_class)))
    build("dma_copy_small", NVCopyQueue, lambda q: q.copy(DST, SRC, 0x1000))
    build("dma_copy_large", NVCopyQueue, lambda q: q.copy(DST, SRC, 2 * (1 << 31) + 0x400))
    build("dma_signal", NVCopyQueue, lambda q: q.signal(SIGNAL, c64(DMA_SIGNAL_VALUE)))
    build("dma_wait", NVCopyQueue, lambda q: q.wait(SIGNAL, c64(0x42)))
    build("dma_timestamp", NVCopyQueue, lambda q: q.timestamp(SIGNAL))
    return out


args.output.mkdir(parents=True, exist_ok=True)
for chip, compute, dma, sass in [
    ("ada", nv_gpu.ADA_COMPUTE_A, nv_gpu.AMPERE_DMA_COPY_B, 0x89),
    ("blackwell", nv_gpu.BLACKWELL_COMPUTE_B, nv_gpu.BLACKWELL_DMA_COPY_B, 0xA4),
]:
    # The same sm89 ELF tests both descriptor layouts; this does not assert the
    # binary is executable on Blackwell. Give each descriptor its own device key.
    DEVS = (f"NV:{chip}",)
    dev = SimpleNamespace(iface=SimpleNamespace(compute_class=compute, dma_class=dma), renderer=None,
                          sass_version=sass, slm_per_thread=0x240, pma_enabled=False,
                          shared_mem_window=SHARED_WINDOW, local_mem_window=LOCAL_WINDOW)
    dev._ensure_has_local_memory = lambda size: setattr(dev, "slm_per_thread", max(dev.slm_per_thread, size))
    # Existing operands are single-device fixture addresses, independent of the
    # descriptor layout; publish their address bindings for this fixture device.
    for buf, address in [(SIGNAL, 0x400000), (SRC, 0x10000000), (DST, 0x20000000), *zip(BUFFERS, (0x900000, 0xA00000, 0xB00000))]:
        ADDRESSES[buf.getaddr(DEVS)] = UOp.const(address, dtypes.uint64)
    for name, stream in streams(dev).items():
        path = args.output / f"{name}_{chip}.expected"
        path.write_text("".join(f"{word:08x}\n" for word in stream))
        print(f"wrote {path} ({len(stream)} dwords)")
