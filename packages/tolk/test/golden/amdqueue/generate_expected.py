"""Golden PM4/SDMA dword streams from the tinygrad reference queue builders.

Drives the reference clone's AMDComputeQueue / AMDCopyQueue (no device, no
memory: a stub object provides the handful of fields the builders read) and
dumps each queue's accumulated dwords to one .expected file per stream per
chip. The OCaml port's generate_actual.ml must mirror the CONFIG block and
the stream table below exactly. See README for the full contract.

Regenerate (from anywhere):
  uv run python packages/tolk/test/golden/amdqueue/generate_expected.py
"""

import importlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "..", "..", "_tinygrad"))

# Scrub every env var the builders (or their imports) consult, so no
# profiling/debug packets sneak into the goldens and re-runs are identical.
for _v in ("DEBUG", "VIZ", "PROFILE", "SQTT", "PMC", "WAVES_PER_SH", "IOCTL",
          "AMD_AQL", "AMD_SDMA_BIND", "SQTT_ITRACE_SE_MASK", "SQTT_LIMIT_SE",
          "SQTT_SIMD_SEL", "SQTT_TOKEN_EXCLUDE", "DEV"):
    os.environ.pop(_v, None)

from tinygrad.runtime.ops_amd import AMDComputeQueue, AMDCopyQueue  # noqa: E402
from tinygrad.runtime.support.amd import AMDIP, import_module, import_soc  # noqa: E402

# CONFIG: every fake constant fed to the builders. The OCaml generator must
# use these same values.
PROG_ADDR = 0x100000              # AMDProgram.prog_addr (256-byte aligned)
SCRATCH_VA = 0x200000             # dev.scratch.va_addr
SCRATCH_SIZE = 0x80000            # dev.scratch.size (divisible by 8 xccs)
KERNARG_VA = 0x300000             # args_state.buf.va_addr
SIGNAL_VALUE_ADDR = 0x400000      # signal.value_addr
SIGNAL_TIMESTAMP_ADDR = 0x400008  # signal.timestamp_addr
MAILBOX_PTR = 0x500000            # dev.queue_event_mailbox_ptr (timeline)
EVENT_ID = 0x2A                   # dev.queue_event.event_id (timeline)
WRITE_VA = 0x600000               # buffer for write32/write64
POLL_VA = 0x700000                # buffer for poll_bit
COPY_SRC_VA = 0x10000000
COPY_DST_VA = 0x20000000
TMPRING_SIZE = 0x00200008         # dev.tmpring_size, written verbatim
RSRC1 = 0x00001111                # AMDProgram.rsrc1 (post-init value)
RSRC2 = 0x00002222
RSRC3 = 0x00003333
WAVE32 = True                     # only affects gfx11+ (cs_w32_en)
SIGNAL_VALUE = 0x42
WAIT_VALUE = 0x42
WRITE32_VALUE = 0x12345678
WRITE64_VALUE = 0x1122334455667788
POLL_MASK = 0x1                   # poll_bit: val == mask (all set)
GLOBAL_SIZE = (4, 3, 2)
LOCAL_SIZE = (8, 4, 1)
COPY_SMALL = 0x1000               # single SDMA_OP_COPY
COPY_LARGE_EXTRA = 0x400          # copy_large = 2*max_copy_size + this


class FakeBuf:
    def __init__(self, va_addr, size):
        self.va_addr, self.size = va_addr, size


class FakeQueueEvent:
    def __init__(self, event_id):
        self.event_id = event_id


class FakeSignal:
    """The fields of AMDSignal that signal()/wait()/timestamp() read."""

    def __init__(self, owner=None, is_timeline=False):
        self.value_addr = SIGNAL_VALUE_ADDR
        self.timestamp_addr = SIGNAL_TIMESTAMP_ADDR
        self.owner, self.is_timeline = owner, is_timeline


class FakeArgsState:
    """CLikeArgsState stand-in: no symbolic args, fixed kernarg address."""

    def __init__(self):
        self.buf = FakeBuf(KERNARG_VA, 0x1000)
        self.bind_data = ()


class FakeProgram:
    """The fields of AMDProgram that AMDComputeQueue.exec reads.

    enable_dispatch_ptr stays 0: the dispatch-ptr path writes an hsa packet
    through a live buffer view, which a stub cannot provide (and tolk's port
    does not need).
    """

    def __init__(self, dev, private_segment):
        self.dev = dev
        self.prog_addr = PROG_ADDR
        self.rsrc1, self.rsrc2, self.rsrc3 = RSRC1, RSRC2, RSRC3
        self.wave32 = WAVE32
        self.enable_private_segment_sgpr = 1 if private_segment else 0
        self.enable_dispatch_ptr = 0


class FakeDev:
    """AMDDevice stand-in: the queue builders read exactly these fields.

    Module/IP wiring replicates AMDDevice.__init__ (ops_amd.py) for the
    given chip. is_am() is False (KFD-style device), which is what routes
    signal() down the queue-event mailbox path for timeline signals.
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
        self.max_copy_size = 0x40000000 if sdma_ver[0] >= 5 else 0x400000
        self.sqtt_enabled = False
        self.scratch = FakeBuf(SCRATCH_VA, SCRATCH_SIZE)
        self.tmpring_size = TMPRING_SIZE
        self.queue_event_mailbox_ptr = MAILBOX_PTR
        self.queue_event = FakeQueueEvent(EVENT_ID)

    def is_am(self):
        return False


# IP versions as the KFD topology reports them for each chip; the module
# resolution (import_module / import_asic_regs) then picks the same autogen
# tables AMDDevice.__init__ would.
CHIPS = {
    'gfx1100': FakeDev(target=(11, 0, 0), xccs=1, gc_ver=(11, 0, 0),
                       nbio_ver=(4, 3, 0), sdma_ver=(6, 0, 0)),
    'gfx942': FakeDev(target=(9, 4, 2), xccs=8, gc_ver=(9, 4, 3),
                      nbio_ver=(7, 9, 0), sdma_ver=(4, 4, 2)),
}


def compute_streams(dev):
    args = FakeArgsState()
    prog = FakeProgram(dev, private_segment=False)
    sig = FakeSignal()
    sig_tl = FakeSignal(owner=dev, is_timeline=True)
    wbuf = FakeBuf(WRITE_VA, 0x1000)
    pbuf = FakeBuf(POLL_VA, 0x1000)

    out = {}

    def build(name, fn):
        q = AMDComputeQueue(dev)
        fn(q)
        out[name] = list(q._q)

    build('exec', lambda q: q.exec(prog, args, GLOBAL_SIZE, LOCAL_SIZE))
    # exec asserts xccs == 1 when the private-segment sgpr (scratch) is
    # enabled, so the scratch variant exists only for single-xcc chips.
    if dev.xccs == 1:
        prog_scratch = FakeProgram(dev, private_segment=True)
        build('exec_scratch',
              lambda q: q.exec(prog_scratch, args, GLOBAL_SIZE, LOCAL_SIZE))
    build('signal', lambda q: q.signal(sig, SIGNAL_VALUE))
    build('signal_timeline', lambda q: q.signal(sig_tl, SIGNAL_VALUE))
    build('wait', lambda q: q.wait(sig, WAIT_VALUE))
    build('timestamp', lambda q: q.timestamp(sig))
    build('write32', lambda q: q.write(wbuf, WRITE32_VALUE))
    build('write64', lambda q: q.write(wbuf, WRITE64_VALUE, b64=True))
    build('poll_bit', lambda q: q.poll_bit(pbuf, POLL_MASK, POLL_MASK))
    build('memory_barrier', lambda q: q.memory_barrier())
    return out


def sdma_streams(dev):
    sig = FakeSignal()
    sig_tl = FakeSignal(owner=dev, is_timeline=True)
    src = FakeBuf(COPY_SRC_VA, 0)
    dst = FakeBuf(COPY_DST_VA, 0)
    wbuf = FakeBuf(WRITE_VA, 0x1000)

    out = {}

    def build(name, fn):
        q = AMDCopyQueue(dev, max_copy_size=dev.max_copy_size)
        fn(q)
        out[name] = list(q._q)

    build('sdma_copy_small', lambda q: q.copy(dst, src, COPY_SMALL))
    build('sdma_copy_large',
          lambda q: q.copy(dst, src, 2 * dev.max_copy_size + COPY_LARGE_EXTRA))
    build('sdma_signal', lambda q: q.signal(sig, SIGNAL_VALUE))
    build('sdma_signal_timeline', lambda q: q.signal(sig_tl, SIGNAL_VALUE))
    build('sdma_wait', lambda q: q.wait(sig, WAIT_VALUE))
    build('sdma_timestamp', lambda q: q.timestamp(sig))
    build('sdma_write32', lambda q: q.write(wbuf, WRITE32_VALUE))
    build('sdma_write64', lambda q: q.write(wbuf, WRITE64_VALUE, b64=True))
    return out


def main():
    for chip, dev in CHIPS.items():
        for name, dwords in {**compute_streams(dev), **sdma_streams(dev)}.items():
            path = os.path.join(_HERE, f"{name}_{chip}.expected")
            with open(path, "w") as f:
                for v in dwords:
                    assert isinstance(v, int) and 0 <= v < 2**32, (name, chip, v)
                    f.write(f"{v:08x}\n")
            print(f"wrote {os.path.relpath(path, _HERE)} ({len(dwords)} dwords)")


if __name__ == "__main__":
    main()
