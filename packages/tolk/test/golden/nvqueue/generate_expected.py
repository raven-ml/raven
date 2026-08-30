"""Golden NV method-stream dwords from the tinygrad reference queue builders.

Drives the reference clone's NVComputeQueue / NVCopyQueue / QMD (no device,
no driver: a stub object provides the handful of fields the builders read)
and dumps each queue's accumulated dwords to one .expected file per stream
per chip. exec writes a QMD image through the kernarg buffer's CPU view, so
kernarg buffers are backed by anonymous mmap while keeping fixed fake GPU
VAs; the QMD images are dumped as separate *_qmd*.expected files. The OCaml
port's generate_actual.ml must mirror the CONFIG block and the stream table
below exactly. See README for the full contract.

Regenerate (from anywhere):
  uv run python packages/tolk/test/golden/nvqueue/generate_expected.py
"""

import mmap
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "..", "..", "_tinygrad"))

# Scrub every env var consulted on the import path or by the builders, so no
# profiling/debug state sneaks into the goldens and re-runs are identical:
# DEBUG/VIZ/PROFILE are helpers ContextVars read at import (PMA's default
# derives from VIZ), PMA gates the PM_TRIGGER packet in exec, IOCTL makes
# ops_nv import the ioctl sniffer, DEV swaps in the MOCK file interface.
for _v in ("DEBUG", "VIZ", "PROFILE", "PMA", "IOCTL", "DEV"):
    os.environ.pop(_v, None)

from tinygrad.helpers import hi32, lo32, round_up  # noqa: E402
from tinygrad.runtime.autogen import nv_570 as nv_gpu  # noqa: E402
from tinygrad.runtime.ops_nv import QMD, NVComputeQueue, NVCopyQueue  # noqa: E402
from tinygrad.runtime.support.hcq import (  # noqa: E402
    FileIOInterface,
    HCQBuffer,
    MMIOInterface,
)

# CONFIG: every fake constant fed to the builders. The OCaml generator must
# use these same values.
PROG_ADDR = 0x100000                # program address (256-byte aligned)
PROG_SZ = 0x1800                    # program size (prefetch = min(sz>>8, 0x1ff))
REGS_USAGE = 32                     # register count
SHMEM_USAGE = 0x480                 # shared memory size (128-byte multiple)
SLM_PER_THREAD = 0x240              # dev.slm_per_thread (32-byte multiple)
CONSTBUFS = {0: (0x110000, 0x160), 3: (0x118000, 0x200)}  # idx: (addr, size)
KERNARG_VA = 0x300000               # args_state.buf.va_addr (64-byte aligned)
KERNARG2_VA = 0x301000              # second kernarg buffer (chained exec)
KERNARG_SIZE = 0x1000
SIGNAL_VALUE_ADDR = 0x400000        # signal.value_addr
WRITE_VA = 0x600000                 # buffer for write32/write64
POLL_VA = 0x700000                  # buffer for poll_bit
LOCAL_MEM_VA = 0x800000             # dev shader_local_mem buffer address
LOCAL_MEM_TPC_BYTES = 0x30000       # per-TPC local memory bytes
COPY_SRC_VA = 0x10000000
COPY_DST_VA = 0x20000000
SHARED_MEM_WINDOW = 0x729400000000  # dev.shared_mem_window (_setup_gpfifos)
LOCAL_MEM_WINDOW = 0x729300000000   # dev.local_mem_window (_setup_gpfifos)
SIGNAL_VALUE = 0x100000042          # 64-bit: pins the hi/lo payload split
DMA_SIGNAL_VALUE = 0x42             # copy signal payload is a single dword
WAIT_VALUE = 0x42
WRITE32_VALUE = 0x12345678
WRITE64_VALUE = 0x1122334455667788
POLL_MASK = 0x1
GLOBAL_SIZE = (4, 3, 2)
LOCAL_SIZE = (8, 4, 1)
COPY_SMALL = 0x1000                 # single LAUNCH_DMA
COPY_LARGE = 2 * (1 << 31) + 0x400  # 3 chunks (copy chunks at 1 << 31)


class FakeIface:
    """The class fields of NVKIface that the builders and QMD read."""

    def __init__(self, compute_class, dma_class):
        self.compute_class, self.dma_class = compute_class, dma_class


class FakeDev:
    """NVDevice stand-in: the queue builders and QMD read exactly these
    fields.

    pma_enabled is False: the PM_TRIGGER packet in exec is profiler state,
    not builder output. sass_version replicates the NVDevice derivation
    ((sm_version & 0xf00) >> 4) | (sm_version & 0xf); the window addresses
    are the fixed values NVDevice._setup_gpfifos assigns.
    """

    def __init__(self, compute_class, dma_class, sm_version):
        self.iface = FakeIface(compute_class, dma_class)
        self.pma_enabled = False
        self.slm_per_thread = SLM_PER_THREAD
        self.shared_mem_window = SHARED_MEM_WINDOW
        self.local_mem_window = LOCAL_MEM_WINDOW
        self.sass_version = ((sm_version & 0xF00) >> 4) | (sm_version & 0xF)


class FakeSignal:
    """The one field of NVSignal the builders read."""

    value_addr = SIGNAL_VALUE_ADDR


class FakeArgsState:
    """NVArgsState stand-in: no symbolic args, mmap-backed kernarg buffer.

    exec copies the program QMD into the buffer at the kernarg offset and
    patches it through the CPU view, so the backing memory must be real;
    the GPU VA stays the fake constant.
    """

    def __init__(self, va):
        self.buf = backed_buf(va, KERNARG_SIZE)
        self.bind_data = ()


def backed_buf(va, size):
    addr = FileIOInterface.anon_mmap(
        0, size, mmap.PROT_READ | mmap.PROT_WRITE,
        mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, 0)
    return HCQBuffer(va, size, view=MMIOInterface(addr, size, fmt="B"))


def prog_qmd(dev):
    # REPLICATED BLOCK -- re-sync on every _tinygrad pin move.
    # This inlines the QMD-image construction of NVProgram.__init__
    # (_tinygrad/tinygrad/runtime/ops_nv.py, the `qmd = {...}` dicts through
    # the constbuf loop; lines 292-313 at the current pin), specialized to
    # the non-NAK path with the fixed program descriptor above. If the
    # reference block changes, update this copy and regenerate the goldens.
    if dev.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A:
        qmd = {'qmd_major_version': 5, 'qmd_type': nv_gpu.NVCEC0_QMDV05_00_QMD_TYPE_GRID_CTA,
               'program_address_upper_shifted4': hi32(PROG_ADDR >> 4),
               'program_address_lower_shifted4': lo32(PROG_ADDR >> 4), 'register_count': REGS_USAGE,
               'shared_memory_size_shifted7': SHMEM_USAGE >> 7,
               'shader_local_memory_high_size_shifted4': dev.slm_per_thread >> 4}
    else:
        qmd = {'qmd_major_version': 3, 'sm_global_caching_enable': 1,
               'program_address_upper': hi32(PROG_ADDR), 'program_address_lower': lo32(PROG_ADDR),
               'shared_memory_size': SHMEM_USAGE, 'register_count_v': REGS_USAGE,
               'shader_local_memory_high_size': dev.slm_per_thread}

    smem_cfg = min(shmem_conf * 1024 for shmem_conf in [32, 64, 100] if shmem_conf * 1024 >= SHMEM_USAGE) // 4096 + 1

    out = QMD(dev, **qmd, qmd_group_id=0x3f, invalidate_texture_header_cache=1, invalidate_texture_sampler_cache=1,
      invalidate_texture_data_cache=1, invalidate_shader_data_cache=1, api_visible_call_limit=1, sampler_index=1, barrier_count=1,
      cwd_membar_type=nv_gpu.NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR, constant_buffer_invalidate_0=1,
      min_sm_config_shared_mem_size=smem_cfg, target_sm_config_shared_mem_size=smem_cfg, max_sm_config_shared_mem_size=0x1a,
      program_prefetch_size=min(PROG_SZ >> 8, 0x1ff), sass_version=dev.sass_version,
      program_prefetch_addr_upper_shifted=PROG_ADDR >> 40, program_prefetch_addr_lower_shifted=PROG_ADDR >> 8)

    for i, (addr, sz) in CONSTBUFS.items():
        out.set_constant_buf_addr(i, addr)
        out.write(**{f'constant_buffer_size_shifted4_{i}': sz, f'constant_buffer_valid_{i}': 1})
    # END REPLICATED BLOCK
    return out


class FakeProgram:
    """The fields of NVProgram that NVComputeQueue.exec reads."""

    def __init__(self, dev):
        self.dev, self.constbufs, self.qmd = dev, CONSTBUFS, prog_qmd(dev)


def qmd_dwords(buf, qmd_sz):
    off = round_up(CONSTBUFS[0][1], 1 << 8)  # exec's kernarg-page QMD offset
    return buf.cpu_view().view(offset=off, size=qmd_sz * 4, fmt="I")[:]


# CHIP configs: the ada/blackwell class pairs are what NVKIface.setup_usermode
# resolves on those chips; sm_version is what _query_gpu_info reports
# (0x809 = sm_89, 0xa04 = sm_120).
CHIPS = {
    'ada': FakeDev(nv_gpu.ADA_COMPUTE_A, nv_gpu.AMPERE_DMA_COPY_B, sm_version=0x809),
    'blackwell': FakeDev(nv_gpu.BLACKWELL_COMPUTE_B, nv_gpu.BLACKWELL_DMA_COPY_B, sm_version=0xA04),
}


def compute_streams(dev):
    prog = FakeProgram(dev)
    qmd_sz = prog.qmd.sz
    sig = FakeSignal()
    wbuf = HCQBuffer(WRITE_VA, 0x1000)
    pbuf = HCQBuffer(POLL_VA, 0x1000)

    out = {}

    def build(name, fn):
        q = NVComputeQueue()
        fn(q)
        out[name] = list(q._q)
        return q

    # The two setup shapes the device ever issues: class + memory windows at
    # gpfifo setup, then the local-memory pair on slm growth.
    build('setup', lambda q: q.setup(compute_class=dev.iface.compute_class,
                                     local_mem_window=dev.local_mem_window,
                                     shared_mem_window=dev.shared_mem_window))
    build('setup_local_mem', lambda q: q.setup(local_mem=LOCAL_MEM_VA,
                                               local_mem_tpc_bytes=LOCAL_MEM_TPC_BYTES))
    build('memory_barrier', lambda q: q.memory_barrier())
    build('wait', lambda q: q.wait(sig, WAIT_VALUE))
    build('timestamp', lambda q: q.timestamp(sig))
    build('signal_no_qmd', lambda q: q.signal(sig, SIGNAL_VALUE))
    build('write32', lambda q: q.write(wbuf, WRITE32_VALUE))
    build('write64', lambda q: q.write(wbuf, WRITE64_VALUE, b64=True))
    build('poll_bit_set', lambda q: q.poll_bit(pbuf, POLL_MASK, POLL_MASK))
    build('poll_bit_clear', lambda q: q.poll_bit(pbuf, 0, POLL_MASK))

    # exec: SEND_PCAS packets + the QMD image patched into the kernarg page.
    args = FakeArgsState(KERNARG_VA)
    build('exec', lambda q: q.exec(prog, args, GLOBAL_SIZE, LOCAL_SIZE))
    out['exec_qmd'] = qmd_dwords(args.buf, qmd_sz)

    # exec_chained: the second exec emits no packets; it links itself into
    # the first QMD's dependent_qmd0 fields.
    args1, args2 = FakeArgsState(KERNARG_VA), FakeArgsState(KERNARG2_VA)
    build('exec_chained', lambda q: q.exec(prog, args1, GLOBAL_SIZE, LOCAL_SIZE)
                                     .exec(prog, args2, GLOBAL_SIZE, LOCAL_SIZE))
    out['exec_chained_qmd0'] = qmd_dwords(args1.buf, qmd_sz)
    out['exec_chained_qmd1'] = qmd_dwords(args2.buf, qmd_sz)

    # signal_after_exec: the signal emits no packets; it patches the active
    # QMD's release0 semaphore fields.
    args3 = FakeArgsState(KERNARG_VA)
    build('signal_after_exec', lambda q: q.exec(prog, args3, GLOBAL_SIZE, LOCAL_SIZE)
                                          .signal(sig, SIGNAL_VALUE))
    out['signal_after_exec_qmd'] = qmd_dwords(args3.buf, qmd_sz)

    # qmd_init: the Program-built QMD image for the fixed descriptor.
    out['qmd_init'] = list(memoryview(prog.qmd.mv).cast('I'))
    return out


def dma_streams(dev):
    sig = FakeSignal()
    src = HCQBuffer(COPY_SRC_VA, 0)
    dst = HCQBuffer(COPY_DST_VA, 0)

    out = {}

    def build(name, fn):
        q = NVCopyQueue()
        fn(q)
        out[name] = list(q._q)

    build('dma_setup', lambda q: q.setup(copy_class=dev.iface.dma_class))
    build('dma_copy_small', lambda q: q.copy(dst, src, COPY_SMALL))
    build('dma_copy_large', lambda q: q.copy(dst, src, COPY_LARGE))
    build('dma_signal', lambda q: q.signal(sig, DMA_SIGNAL_VALUE))
    build('dma_wait', lambda q: q.wait(sig, WAIT_VALUE))
    build('dma_timestamp', lambda q: q.timestamp(sig))
    return out


def main():
    for chip, dev in CHIPS.items():
        for name, dwords in {**compute_streams(dev), **dma_streams(dev)}.items():
            path = os.path.join(_HERE, f"{name}_{chip}.expected")
            with open(path, "w") as f:
                for v in dwords:
                    assert isinstance(v, int) and 0 <= v < 2**32, (name, chip, v)
                    f.write(f"{v:08x}\n")
            print(f"wrote {os.path.relpath(path, _HERE)} ({len(dwords)} dwords)")


if __name__ == "__main__":
    main()
