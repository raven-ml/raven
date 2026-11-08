"""Tests of the generators' machinery, run by generate.py against the tinygrad
it generates from: `unplaced_kernels` keeps each graph's meaning while a kernel
body's parameters lose the device their calls name; `forked` returns a child's
value, and reports its failure in the parent."""

import os

from golden import REFUSED, Refused, attempt, forked
from graph import unplaced_kernels, write
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import KernelInfo, Ops, UOp


def param(slot, device="CPU"):
    return UOp.param(slot, dtypes.float, 4, device)


def kernel(*params, arg=KernelInfo()):
    # a body storing the sum of its parameters into the first
    value = params[0]
    for p in params[1:]: value = value + p
    return UOp.sink(params[0].store(value), arg=arg)


def lines(sink):
    return write(sink).splitlines()


def params_of(sink):
    return [line for line in lines(sink) if " Ops.PARAM " in line or " Ops.REFPARAM " in line]


def expect(actual, expected, what):
    if actual != expected:
        raise AssertionError(f"{what}:\n  expected {expected!r}\n  got      {actual!r}")


def test_tensor_level_param_keeps_device():
    expect(params_of(UOp.sink(param(0))),
           ['1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")'], "a tensor-level PARAM")


def test_kernel_body_loses_device():
    expect(params_of(kernel(param(0), param(1))),
           ["1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)",
            "2 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)"], "a kernel body's PARAMs")


def test_shared_param_is_split_by_position():
    # the same PARAM node in the kernel body and as its call's argument
    p = param(0)
    graph = lines(UOp.sink(kernel(p).call(p)))
    params = [line for line in graph if " Ops.PARAM " in line]
    expect(sorted(line.split(" ", 2)[2] for line in params),
           ['dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)',
            'dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")'],
           "the shared PARAM as body and argument")


def test_two_kernels_share_a_param():
    # kernels on one buffer: each body's device-less PARAM is the same node
    p = param(0)
    k1, k2 = kernel(p), kernel(p, param(1, "CPU"))
    params = params_of(UOp.sink(k1, k2))
    expect(len(params), 2, "PARAMs of two kernels sharing the first")


def test_function_body_keeps_device():
    # a CALL of a function body, a SINK with no KernelInfo, is no kernel
    p = param(0)
    f = UOp.sink(p.store(p + p))
    params = params_of(UOp.sink(f.call(p)))
    expect(all('device="CPU"' in line for line in params) and len(params) > 0, True, "a function body's PARAMs")


def test_copy_body_loses_device():
    # a copy kernel's body is the STORE of its two parameters
    dst, src = param(0, "CPU:1"), param(1, "CPU")
    body = dst.store(src)
    params = params_of(UOp.sink(body.call(dst, src)))
    expect(sum('device=' in line for line in params), 2, "the call's arguments keep their devices")
    expect(sum('device=' not in line for line in params), 2, "the copy body's PARAMs lose theirs")


def test_one_device_store_body_keeps_devices():
    # a STORE over one device is no copy: a hand-built stand-in keeps its PARAMs
    dst, src = param(0), param(1)
    params = params_of(UOp.sink(dst.store(src).call(dst, src)))
    expect(all('device="CPU"' in line for line in params) and len(params) == 2, True, "a one-device STORE body")


def test_disk_store_body_loses_device():
    dst, src = param(0, "DISK:/tmp/x"), param(1, "DISK:/tmp/x")
    params = params_of(UOp.sink(dst.store(src).call(dst, src)))
    expect(sum('device=' not in line for line in params), 2, "a copy to a disk loses its body's devices")


def test_function_store_body_is_only_a_copy_body():
    # a function's body holding the STORE inside its own SINK is a function body
    p = param(0)
    f = UOp.sink(p.store(p))
    params = params_of(UOp.sink(f.call(p)))
    expect(len(params), 1, "one shared PARAM")
    expect('device="CPU"' in params[0], True, "a function body's PARAM keeps its device")


def test_linear_of_a_kernel_loses_device():
    # the linearized form lists the kernel's PARAMs beside its SINK
    p = param(0)
    k = kernel(p)
    linear = UOp(Ops.LINEAR, src=(p, k))
    params = params_of(linear)
    expect(len(params), 1, "one PARAM")
    expect('device=' in params[0], False, "the PARAM of a linearized kernel")


def test_linear_of_calls_keeps_device():
    # a schedule's LINEAR is of calls and their arguments
    p = param(0)
    linear = UOp(Ops.LINEAR, src=(kernel(p).call(p),))
    expect(sum('device=' in line for line in params_of(linear)), 1, "the argument keeps its device")


def test_refparam_loses_device():
    p = param(0)
    ref = UOp(Ops.REFPARAM, src=(p.src[0], p.src[0]), arg=p.arg)
    params = params_of(UOp.sink(ref, arg=KernelInfo()))
    expect(params, ["1 Ops.REFPARAM dtypes.f32 [0, 0] ParamArg(slot=0, dtype=dtypes.f32)"], "a kernel's REFPARAM")


def test_program_args_are_followed():
    # a PROGRAM holds the parameters of its kernel in its arg
    p = param(0)
    program = UOp(Ops.PROGRAM, src=(p,), arg=(p,))
    expect(lines(UOp.sink(program)),
           ["0 Ops.CONST dtypes.weakint [] 4",
            "1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)",
            "2 Ops.PROGRAM dtypes.void [1] (%1,)",
            "3 Ops.SINK dtypes.void [2]"], "a PROGRAM's parameters")


def test_identity_without_devices():
    # a graph with no device in a kernel is itself
    sink = kernel(UOp.param(0, dtypes.float, 4), UOp.param(1, dtypes.float, 4))
    if unplaced_kernels(sink) is not sink:
        raise AssertionError("a graph with no device in a kernel was rebuilt")


def test_forked_returns_the_value():
    expect(forked(lambda: (1, "a")), (1, "a"), "a child's value")


def refusal(fn):
    try:
        forked(fn)
    except Refused as e:
        return str(e)
    raise AssertionError("no refusal")


def test_forked_refuses_on_a_raise():
    expect("ZeroDivisionError" in refusal(lambda: 1 / 0), True, "the reason names the error")


def test_forked_refuses_on_a_system_exit():
    def leave(): raise SystemExit(3)
    expect("SystemExit" in refusal(leave), True, "the reason names the exit")


def test_forked_ends_the_run_on_a_signal():
    def die(): os.kill(os.getpid(), 9)
    try:
        forked(die)
    except SystemExit as e:
        expect("signal 9" in str(e), True, "the run ends naming the signal")
        return
    raise AssertionError("a killed child did not end the run")


def test_attempt_records_a_refusal():
    def refused(): raise ValueError("no")
    before = len(REFUSED)
    expect(attempt("p", refused, default=[]), [], "the default")
    expect((len(REFUSED) - before, REFUSED[-1][0]), (1, "p"), "the recorded name")
    expect(attempt("q", lambda: 1), 1, "an accepted program's value")
    expect(len(REFUSED) - before, 1, "an accepted program records nothing")


def main():
    tests = [v for name, v in globals().items() if name.startswith("test_")]
    for test in tests: test()
    print(f"{len(tests)} tests")


if __name__ == "__main__":
    main()
