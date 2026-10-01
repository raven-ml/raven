"""The devices of the generators of batches: the CPU host, and CPU:1, CPU:2
and CPU:3 with the NULL device's queues (runtime/ops_null.py's NullQueue),
whose host is the CPU. Each compiles for Clang on x86_64, and no program
compiles: a program's binary is its source's bytes.

tinygrad is changed as tolk differs from it:
- a device's signal word is one word, as hcq2_d1.py applies it;
- NullQueue writes a variable by value, as a variable has no address, and
  takes an address on its first device, as tolk names one device.

Importing the module patches tinygrad.
"""

import hcq2_d1  # noqa: F401
from tinygrad import dtypes
from tinygrad.device import Compiler
from tinygrad.helpers import DEV, to_tuple
from tinygrad.runtime.ops_cpu import CPUDevice
from tinygrad.runtime.ops_null import NullQueue, EXEC, TIMESTAMP
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat
import tinygrad.engine.realize as realize
import tinygrad.runtime.support.hcq2 as hcq2

DEV.value = "CPU::x86_64,x86-64"
Compiler.compile_cached = lambda self, src: src.encode()
# Workers compile in processes of their own, with tinygrad's compilers.
realize.get_worker_pool = lambda: None

QUEUED = ("CPU:1", "CPU:2", "CPU:3")
hcq2.all_devices_in = lambda d, c: all(x in QUEUED for x in to_tuple(d))


def cmd(self, op, *args):
    words = [(a if a.is_variable else a.getaddr(self.devs[0])) if isinstance(a, UOp) else UOp.const(a, dtypes.uint64)
             for a in (op, *args, 0, 0, 0)]
    self.q(*words[:4])


def exec_(self, call, prg):
    args = [a.getaddr(self.devs[0]) for a in realize.get_call_arg_uops(call)] + \
           [v.cast(dtypes.uint64) for v in realize.get_call_var_uops(call, prg)]
    kernargs = UOp(Ops.LINEAR, src=tuple(hcq2.pack_args(hcq2.layout_args(args), 8 * max(len(args), 1))), arg="kernargs")
    self.cmd(EXEC, kernargs, len(args), self.event(self.devs[0], prg.src[0].arg.function_name, prg.key))


NullQueue.cmd, NullQueue.exec = cmd, exec_
NullQueue.timestamp = lambda self, signal: self.cmd(TIMESTAMP, signal.getaddr(self.devs[0]) + UOp.const(8, dtypes.uint64))
CPUDevice.pm_encode = PatternMatcher([(UPat(Ops.CUSTOM_FUNCTION, arg=f"submit_cpu_{q}", name="submit"),
                                       lambda submit: hcq2.encode_submit(NullQueue(submit))) for q in ("compute", "copy")])
