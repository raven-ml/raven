"""PyTorch's, JAX's and tinygrad's probe for run.py.

usage: python frameworks.py STACK SCENARIO DEVICE [ARG...]

STACK is torch, jax or tinygrad, run with an interpreter that has it; DEVICE
is the framework's name for the GPU. Each scenario prints one line of
KEY=VALUE fields; compare/README.md says what each one measures.
"""
import array
import ctypes
import mmap
import os
import sys
import time

COUNT = 64  # The floats an addition adds.
WARMUPS, SAMPLES, IDLE = 3, 5, 0.02  # The micro rows' samples.
CHUNK = 4 << 20  # The bytes of each checked piece of a loaded file.

# Each stack opens its device in its constructor, and has:
#   floats(value)      an array of COUNT floats on the device, all [value]
#   load(path)         a byte array on the device, holding the file's bytes
#   get(a, start, n)   the bytes of elements [start, start + n) of [a]
#   add(x, y)          x + y, queued
#   wait(a)            returns once [a] and the work before it completed


def mapped(path):
    with open(path, 'rb') as file:
        return mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_COPY)


class Torch:
    def __init__(self, name):
        import torch
        self.torch, self.device = torch, torch.device(name)
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device.index or 0)
            torch.cuda.init()
            self.sync = lambda: torch.cuda.synchronize(self.device)
        elif self.device.type == 'mps':
            self.sync = torch.mps.synchronize
        else:
            raise SystemExit(f'torch: {name} is not a GPU')
        self.sync()

    def floats(self, value):
        return self.torch.tensor([value] * COUNT, dtype=self.torch.float32, device=self.device)

    def load(self, path):
        return self.torch.frombuffer(mapped(path), dtype=self.torch.uint8).to(self.device)

    def get(self, a, start, n):
        piece = a[start:start + n].cpu()
        return ctypes.string_at(piece.data_ptr(), n * piece.element_size())

    def add(self, x, y):
        return x + y

    def wait(self, a):
        self.sync()


class Jax:
    def __init__(self, name):
        import jax
        import numpy
        self.jax, self.numpy = jax, numpy
        self.device = jax.devices(name)[0]
        # The Metal plugin cannot donate buffers.
        donate = (0,) if self.device.platform != 'METAL' else ()
        self.add = jax.jit(lambda x, y: x + y, donate_argnums=donate)

    def floats(self, value):
        return self.jax.device_put(self.numpy.full(COUNT, value, self.numpy.float32), self.device)

    def load(self, path):
        return self.jax.device_put(self.numpy.frombuffer(mapped(path), self.numpy.uint8), self.device)

    def get(self, a, start, n):
        return self.numpy.asarray(a[start:start + n]).tobytes()

    def wait(self, a):
        a.block_until_ready()


class Tinygrad:
    def __init__(self, name):
        if name == 'NV':
            os.environ['DEV'] = 'NVK+NV'  # NV through the kernel driver, never PCI.
        import pathlib
        from tinygrad import Device, Tensor, dtypes
        self.Tensor, self.Path, self.dtypes, self.name = Tensor, pathlib.Path, dtypes, name
        self.device = Device[name]

    def floats(self, value):
        return self.Tensor([value] * COUNT, dtype=self.dtypes.float32, device=self.name).realize()

    def load(self, path):
        return self.Tensor(self.Path(path)).to(self.name).realize()

    def get(self, a, start, n):
        return bytes(a[start:start + n].data())

    def add(self, x, y):
        return (x + y).realize()

    def wait(self, a):
        self.device.synchronize()


STACKS = {'torch': Torch, 'jax': Jax, 'tinygrad': Tinygrad}


def check(stack, a, expected):
    actual = array.array('f', stack.get(a, 0, COUNT))
    if list(actual) != [expected] * COUNT:
        raise SystemExit(f'addition gave {actual[0]}, expected {expected}')


# Scenarios

def first(stack):
    """The first kernel: one addition, checked after it completed; the time
    it completed."""
    z = stack.add(stack.floats(41.0), stack.floats(1.0))
    stack.wait(z)
    completed = time.monotonic_ns()
    check(stack, z, 42.0)
    return completed


def start(stack):
    print(f'completed_ns={first(stack)}')


def fit(stack):
    first(stack)
    print(f'pid={os.getpid()}', flush=True)
    sys.stdin.readline()


def load(stack, *paths):
    """The timer covers opening each file and its completed copy into a device
    array; the check reads every byte back after it."""
    started = time.monotonic_ns()
    loaded = [stack.load(path) for path in paths]
    for a in loaded:
        stack.wait(a)
    elapsed = time.monotonic_ns() - started
    total = 0
    for path, a in zip(paths, loaded):
        source = mapped(path)
        for at in range(0, len(source), CHUNK):
            n = min(CHUNK, len(source) - at)
            if stack.get(a, at, n) != source[at:at + n]:
                raise SystemExit(f'{path} differs in the chunk at {at}')
        total += len(source)
    print(f'bytes={total} load_ns={elapsed}')


def add(stack, calls):
    """[calls] dependent additions and a wait on the last."""
    calls, ones, times = int(calls), stack.floats(1.0), []
    for sample in range(WARMUPS + SAMPLES):
        x = stack.floats(41.0)
        stack.wait(x)
        time.sleep(IDLE)
        started = time.monotonic_ns()
        for _ in range(calls):
            x = stack.add(x, ones)
        stack.wait(x)
        elapsed = time.monotonic_ns() - started
        check(stack, x, 41.0 + calls)
        if sample >= WARMUPS:
            times.append(elapsed)
    print('samples_ns=' + ','.join(map(str, times)))


SCENARIOS = {'start': start, 'fit': fit, 'load': load, 'add': add}

if __name__ == '__main__':
    if len(sys.argv) < 4 or sys.argv[1] not in STACKS or sys.argv[2] not in SCENARIOS:
        raise SystemExit(__doc__)
    name, scenario, device, *args = sys.argv[1:]
    SCENARIOS[scenario](STACKS[name](device), *args)
