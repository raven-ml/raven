"""Generate the eight frozen-target NV release fixtures, without a GPU.

Other fixtures retain their documented old-pin provenance; this generator does
not emulate the retired HCQBuffer/ArgsState launch API. See README.
"""

import argparse
from pathlib import Path
import os
import struct
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
TARGET = "471a3aeb6924257d5e9bf321f5ff0a519163f18e"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tinygrad", type=Path, default=HERE.parents[4] / "_tinygrad_target",
                    help=f"tinygrad source tree at {TARGET}")
parser.add_argument("--output", type=Path, default=HERE, help="directory for the eight .expected files")
args = parser.parse_args()
if not (args.tinygrad / "tinygrad/runtime/ops_nv.py").is_file():
    parser.error(f"not a tinygrad source tree: {args.tinygrad}")
sys.path.insert(0, str(args.tinygrad.resolve()))
for key in ("DEBUG", "VIZ", "PROFILE", "PMA", "IOCTL", "DEV"):
    os.environ.pop(key, None)

from tinygrad.dtype import dtypes  # noqa: E402
from tinygrad.runtime.ops_nv import QMD, NVComputeQueue, NVCopyQueue, nv_gpu  # noqa: E402
from tinygrad.uop.ops import UOp  # noqa: E402

# Fixed inputs shared with generate_actual.ml. The descriptor's addresses retain
# the legacy fixture layout so its release fields are compared independently of
# the new per-submission QMD/argument arena (covered by runtime tests).
PROG_ADDR, PROG_SZ = 0x100000, 0x1800
REGS_USAGE, SHMEM_USAGE, SLM_PER_THREAD = 32, 0x480, 0x240
CONSTBUFS = {0: (0x110000, 0x160), 3: (0x118000, 0x200)}
KERNARG_VA, SIGNAL_VALUE_ADDR = 0x300000, 0x400000
SIGNAL_VALUE, DMA_SIGNAL_VALUE = 0x100000042, 0x42
GLOBAL_SIZE, LOCAL_SIZE = (4, 3, 2), (8, 4, 1)
DEVS = ("NV",)
SIGNAL = UOp.placeholder((2,), dtypes.uint64, device=DEVS)
ADDRESSES = {SIGNAL.getaddr(DEVS): UOp.const(SIGNAL_VALUE_ADDR, dtypes.uint64)}
CHIPS = {
    "ada": SimpleNamespace(iface=SimpleNamespace(compute_class=nv_gpu.ADA_COMPUTE_A), sass_version=0x89),
    "blackwell": SimpleNamespace(iface=SimpleNamespace(compute_class=nv_gpu.BLACKWELL_COMPUTE_B), sass_version=0xA4),
}


def launched_qmd(dev):
    # REPLICATED TEMPLATE: NVProgramData.__init__ in the frozen ops_nv.py,
    # CUDA path, specialized to the fixed descriptor inputs. Address and grid
    # setters are the same target QMD methods that NVComputeQueue.exec calls.
    qmd = QMD(dev)
    if qmd.ver == 5:
        qmd.write(qmd_major_version=5, qmd_type=nv_gpu.NVCEC0_QMDV05_00_QMD_TYPE_GRID_CTA,
                  register_count=REGS_USAGE, shared_memory_size_shifted7=SHMEM_USAGE >> 7,
                  shader_local_memory_high_size_shifted4=SLM_PER_THREAD >> 4)
    else:
        qmd.write(qmd_major_version=3, sm_global_caching_enable=1,
                  shared_memory_size=SHMEM_USAGE, register_count_v=REGS_USAGE,
                  shader_local_memory_high_size=SLM_PER_THREAD)
    smem_cfg = min(n * 1024 for n in (32, 64, 100) if n * 1024 >= SHMEM_USAGE) // 4096 + 1
    qmd.write(qmd_group_id=0x3F, invalidate_texture_header_cache=1, invalidate_texture_sampler_cache=1,
              invalidate_texture_data_cache=1, invalidate_shader_data_cache=1, api_visible_call_limit=1,
              sampler_index=1, barrier_count=1, cwd_membar_type=nv_gpu.NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR,
              constant_buffer_invalidate_0=1, min_sm_config_shared_mem_size=smem_cfg,
              target_sm_config_shared_mem_size=smem_cfg, max_sm_config_shared_mem_size=0x1A,
              program_prefetch_size=min(PROG_SZ >> 8, 0x1FF), sass_version=dev.sass_version)
    qmd.set_program_addr(PROG_ADDR)
    for index, (address, size) in CONSTBUFS.items():
        qmd.write(**{f"constant_buffer_size_shifted4_{index}": size, f"constant_buffer_valid_{index}": 1})
        qmd.set_constant_buf_addr(index, KERNARG_VA if index == 0 else address)
    qmd.write(**dict(zip(qmd.grid, GLOBAL_SIZE)), **{f"cta_thread_dimension{j}": n for j, n in enumerate(LOCAL_SIZE)})
    return qmd


def queue(cls):
    # Unit-test only the release builders: bypass constructor device lookup and
    # program arena allocation. HWQueue.q and every release method stay intact.
    result = object.__new__(cls)
    result.devs, result.blob, result.patches = DEVS, bytearray(), []
    if cls is NVComputeQueue:
        result.prev_qmd = None
    return result


def dwords(blob, patches):
    result = bytearray(blob)
    for offset, value in patches:
        scalar = value.substitute(ADDRESSES).ssimplify()
        assert isinstance(scalar, int), (offset, value, scalar)
        # Match the fixed-width store that applies a target QMD/command patch.
        width = value.dtype.itemsize
        result[offset:offset + width] = (scalar & ((1 << (8 * width)) - 1)).to_bytes(width, "little")
    assert len(result) % 4 == 0
    return struct.unpack(f"<{len(result) // 4}I", result)


def release_streams(dev):
    compute = queue(NVComputeQueue)
    compute.signal(SIGNAL, UOp.const(SIGNAL_VALUE, dtypes.uint64))
    timestamp = queue(NVComputeQueue)
    timestamp.timestamp(SIGNAL)
    copy = queue(NVCopyQueue)
    copy.signal(SIGNAL, UOp.const(DMA_SIGNAL_VALUE, dtypes.uint32))
    launched = queue(NVComputeQueue)
    launched.prev_qmd = qmd = launched_qmd(dev)
    launched.signal(SIGNAL, UOp.const(SIGNAL_VALUE, dtypes.uint64))
    assert not launched.blob and not launched.patches, "release must patch the active descriptor"
    return {
        "signal_no_qmd": dwords(compute.blob, compute.patches),
        "timestamp": dwords(timestamp.blob, timestamp.patches),
        "dma_signal": dwords(copy.blob, copy.patches),
        "signal_after_exec_qmd": dwords(qmd.mv, qmd.patches.items()),
    }


args.output.mkdir(parents=True, exist_ok=True)
for chip, dev in CHIPS.items():
    for name, words in release_streams(dev).items():
        path = args.output / f"{name}_{chip}.expected"
        path.write_text("".join(f"{word:08x}\n" for word in words))
        print(f"wrote {path} ({len(words)} dwords)")
