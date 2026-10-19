# /// script
# requires-python = ">=3.10"
# ///
"""Compare rig with the vendor's toolkit, PyTorch, JAX and tinygrad on this machine's GPU.

usage: uv run packages/rig/bench/compare/run.py SCENARIO
           [--torch PYTHON] [--jax PYTHON] [--tinygrad PYTHON] [FILE...]

SCENARIO is ship, start, load (of the checkpoint files FILE...), fit, micro
or vendor. run.py builds the probes, runs each stack interleaved over the
rounds, prints a table and writes results/GPU-DATE.json. README.md says
what each scenario measures, what it needs and what a result holds.
"""
import argparse
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESULTS = HERE / 'results'
KERNELS = ROOT / 'packages/rig/examples/01-add/kernels'  # The add kernel, as each driver loads it.
ROUNDS = 5
TIMEOUT = 600  # Seconds a probe may take.
CHUNK = 4 << 20  # The read row's buffer.

# Each platform's rig drivers, native API and each framework's device names.
PLATFORMS = {
    'metal': {'rig': ['metal'], 'native': 'metal', 'torch': ['mps'], 'jax': ['METAL'], 'tinygrad': ['METAL']},
    'nvidia': {'rig': ['cuda', 'nv'], 'native': 'cuda', 'torch': ['cuda'], 'jax': ['gpu'], 'tinygrad': ['CUDA', 'NV']},
    'amd': {'rig': ['amd'], 'native': 'hip', 'torch': ['cuda'], 'jax': ['gpu'], 'tinygrad': ['AMD']},
}
FRAMEWORKS = ('torch', 'jax', 'tinygrad')

# proc_pid_rusage's struct rusage_info_v4, and the index of its
# ri_phys_footprint as 64-bit words: after the 16-byte UUID and 7 counters.
RUSAGE_INFO_V4, PHYS_FOOTPRINT = 4, 9

# Prefixes a system's Python shares with the rest of the system.
SYSTEM_PREFIXES = {Path('/'), Path('/usr'), Path('/usr/local')}

# A loaded library outside an installation is the GPU vendor's if its name
# says so; the rest are the operating system's.
VENDOR = re.compile(r'lib(cuda|nv|amdhip|hsa|hiprtc|amd_comgr|roc)')


def platform():
    if sys.platform == 'darwin':
        return 'metal'
    if Path('/dev/nvidiactl').exists():
        return 'nvidia'
    if Path('/dev/kfd').exists():
        return 'amd'
    raise SystemExit('run.py: no Metal, NVIDIA or AMD GPU on this machine')


def output(*command):
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def machine(gpu):
    """The machine by its GPU, the GPU's driver and the system, which name no host."""
    system = ' '.join(os.uname()[i] for i in (0, 2, 4))
    if gpu == 'metal':
        name, driver = output('sysctl', '-n', 'machdep.cpu.brand_string'), f"macOS {output('sw_vers', '-productVersion')}"
    elif gpu == 'nvidia':
        name, driver = output('nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader').splitlines()[0].split(', ')
    else:
        # GPU 0 is KFD's first node with a target, which it gives as
        # major * 10000 + minor * 100 + step, beside the PCI ids lspci names.
        nodes = sorted(Path('/sys/class/kfd/kfd/topology/nodes').glob('*/properties'), key=lambda p: int(p.parent.name))
        found = (dict(line.split(maxsplit=1) for line in node.read_text().splitlines()) for node in nodes)
        node = next(n for n in found if int(n['gfx_target_version']))
        v = int(node['gfx_target_version'])
        ids = f"{int(node['vendor_id']):x}:{int(node['device_id']):x}"
        product = shlex.split(output('lspci', '-mm', '-d', ids).splitlines()[0])[3]
        marketed = re.search(r'\[(.*)\]', product)
        name = f"{marketed[1] if marketed else product} (gfx{v // 10000}{v // 100 % 100:x}{v % 100:x})"
        # amdgpu is the kernel's, unless a module of its own says its version.
        module = Path('/sys/module/amdgpu/version')
        driver = module.read_text().strip() if module.exists() else f'Linux {os.uname().release}'
    return {'gpu': name, 'driver': driver, 'system': system}


def slug(name):
    """[name] as a file name: lower case, words joined by hyphens."""
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')


def shown(path, roots):
    """[path] as a record shows it, which names no user. A path under the home
    directory is shown relative to the innermost of [roots] that holds it, each
    a name and a directory: the working directory, named '.', and each
    framework's environment, named after the framework."""
    if not path.is_relative_to(Path.home()):
        return str(path)
    holders = [(name, root) for name, root in roots if path.is_relative_to(root)]
    if not holders:
        raise SystemExit(f'run.py: {path} lies under the home directory, outside the working directory '
                         'and the frameworks\' environments; run from a directory that holds it')
    name, root = max(holders, key=lambda holder: len(holder[1].parts))
    return str(Path(name, path.relative_to(root)))


class Stack:
    """A probe: [head(scenario)] is its command up to the scenario's arguments."""

    def __init__(self, name, device, head, image=None, installation=(), version=None, env=None):
        self.name, self.device, self.head, self.image = name, device, head, image
        self.installation, self.version, self.env = list(installation), version, env or {}

    def kernel(self):
        """The arguments that name the image of [add], if the probe takes one."""
        return [self.image] if self.image else []

    def command(self, scenario, *args):
        return [*self.head(scenario), *args]

    def run(self, scenario, *args, env=None):
        """The fields of the probe's last line, after it exited with 0."""
        child = subprocess.run(self.command(scenario, *map(str, args)), capture_output=True, text=True,
                               env=os.environ | self.env | (env or {}), timeout=TIMEOUT)
        if child.returncode != 0:
            sys.stderr.write(child.stderr)
            raise SystemExit(f'run.py: {self.name} {self.device} {scenario} exited with {child.returncode}')
        return fields(child.stdout), child.stderr


# A probe's line of fields; a framework may print other lines (jax-metal does).
FIELDS = re.compile(r'\w+=\S+( \w+=\S+)*')


def fields(stdout):
    """The fields of a probe's last line of them."""
    lines = [line for line in stdout.splitlines() if FIELDS.fullmatch(line)]
    if not lines:
        raise SystemExit(f'run.py: a probe printed no fields: {stdout!r}')
    found = dict(field.split('=', 1) for field in lines[-1].split())
    return {k: int(v) if re.fullmatch(r'\d+', v) else v for k, v in found.items()}


def commit():
    """The tree's commit, +dirty if rig differs from it beyond the results."""
    head = output('git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD')
    dirty = output('git', '-C', str(ROOT), 'status', '--porcelain', '--', 'packages/rig',
                   f':!{RESULTS.relative_to(ROOT)}')
    return head + ('+dirty' if dirty else '')


def stacks(gpu, frameworks, work):
    """The probes on this machine, built into [work]."""
    table = PLATFORMS[gpu]
    # Timings come from a release build: the dev profile's -opaque removes
    # inlining across modules. A running dune server takes the build and
    # ignores --profile, so its warning stops the run.
    build = subprocess.run(['dune', 'build', '--profile', 'release', './packages/rig/bench/compare/rig/rig.exe'],
                           cwd=ROOT, capture_output=True, text=True)
    sys.stderr.write(build.stderr)
    if build.returncode != 0 or 'forwarded to a running Dune instance' in build.stderr:
        raise SystemExit('run.py: rig.exe needs a release build, in a tree without a running dune')
    rig = ROOT / '_build/default/packages/rig/bench/compare/rig/rig.exe'
    native = work / 'native'
    cuda = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda'))
    rocm = Path(os.environ.get('ROCM_PATH', '/opt/rocm'))
    flags, libraries = ['-O2', '-Wall', '-Wextra', '-Werror'], []
    if gpu == 'metal':
        flags += ['-x', 'objective-c']
        libraries = ['-framework', 'Foundation', '-framework', 'Metal']
        image = KERNELS / 'add.metallib'
    elif gpu == 'nvidia':
        flags += ['-DCUDA', f'-I{cuda}/include']
        libraries = [f'-L{cuda}/lib64/stubs', '-lcuda']
        arch = output('nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader').splitlines()[0]
        image = work / 'add.cubin'
        subprocess.run([cuda / 'bin/ptxas', f"-arch=sm_{arch.replace('.', '')}",
                        KERNELS / 'add.ptx', '-o', image], check=True)
    else:
        flags += ['-D__HIP_PLATFORM_AMD__', f'-I{rocm}/include']
        libraries = [f'-L{rocm}/lib', f'-Wl,-rpath,{rocm}/lib', '-lamdhip64']
        image = KERNELS / 'add_gfx1201.hsaco'
    subprocess.run(['cc', *flags, HERE / 'native.c', '-o', native, *libraries], check=True)
    probes = [Stack('rig', driver, lambda s, d=driver: [rig, s, d], image, [rig], commit()) for driver in table['rig']]
    probes.append(Stack('native', table['native'], lambda s: [native, s], image, [native]))
    for name, python in frameworks.items():
        *prefixes, version = output(python, '-c', 'import importlib.metadata, sys; print(sys.prefix); '
                                    f'print(sys.base_prefix); print(importlib.metadata.version("{name}"))').splitlines()
        for device in table[name]:
            probes.append(Stack(name, device, lambda s, p=python, n=name, d=device: [p, HERE / 'frameworks.py', n, s, d],
                               installation=map(Path, prefixes), version=version, env=caches(work / 'cache')))
    return probes


def caches(root):
    """The compiler caches of tinygrad and CUDA, under [root]."""
    (root / 'cuda').mkdir(parents=True, exist_ok=True)
    return {'CACHEDB': str(root / 'tinygrad.db'), 'CUDA_CACHE_PATH': str(root / 'cuda')}


def row(stack, samples, **extra):
    return {'stack': stack.name, 'device': stack.device, 'version': stack.version, 'samples': samples, **extra}


# Scenarios

def ship(probes, files):
    if os.environ.get('PYTHONPATH'):
        raise SystemExit('run.py: ship counts an installation; unset PYTHONPATH')
    for stack in probes:
        if any(root in SYSTEM_PREFIXES for root in stack.installation):
            raise SystemExit(f'run.py: {stack.name} runs on the system Python, whose prefix holds the whole system; '
                             'ship needs an environment on a standalone Python, as uv venv makes')
    roots = [('.', Path.cwd().resolve()),
             *((stack.name, stack.installation[0].resolve()) for stack in probes if stack.name in FRAMEWORKS)]
    rows = []
    for stack in probes:
        paths = set()
        installed = 0
        for root in stack.installation:
            for path in [root] if root.is_file() else root.rglob('*'):
                if path.is_file() and not path.is_symlink() and path.resolve() not in paths:
                    paths.add(path.resolve())
                    installed += path.stat().st_size
        with tempfile.TemporaryDirectory() as fresh:
            env = caches(Path(fresh)) | ({'LD_DEBUG': 'libs'} if sys.platform == 'linux' else {})
            _, stderr = stack.run('start', *stack.kernel(), env=env)
        vendor, system = [], []
        for name in sorted(set(re.findall(r'calling init: (\S+)', stderr))):
            path = Path(name).resolve(strict=True)
            if path in paths:
                continue
            if VENDOR.match(path.name):
                vendor.append({'path': shown(path, roots), 'bytes': path.stat().st_size,
                               'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            else:
                system.append(shown(path, roots))
        vendor_bytes = sum(lib['bytes'] for lib in vendor)
        rows.append(row(stack, [installed + vendor_bytes], installed=installed, vendor=vendor_bytes,
                        vendor_libraries=vendor, os_libraries=system))
    return {'unit': 'bytes', 'rows': rows}


def start(probes, files):
    def sample(stack):
        started = time.monotonic_ns()
        found, _ = stack.run('start', *stack.kernel())
        return found['completed_ns'] - started
    for stack in probes:
        sample(stack)
    samples = {stack: [] for stack in probes}
    for _ in range(ROUNDS):
        for stack in probes:
            samples[stack].append(sample(stack))
    return {'unit': 'ns', 'rows': [row(stack, s) for stack, s in samples.items()]}


def fit(probes, files):
    gpu = platform()
    samples = {stack: [] for stack in probes}
    for _ in range(ROUNDS):
        for stack in probes:
            child = subprocess.Popen(stack.command('fit', *stack.kernel()), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, env=os.environ | stack.env)
            line = next((line for line in child.stdout if FIELDS.fullmatch(line.strip())), None)
            if line is None:
                raise SystemExit(f'run.py: {stack.name} {stack.device} fit exited with {child.wait()}')
            found = fields(line)
            samples[stack].append(counter(gpu, found['pid']))
            child.communicate('\n', timeout=TIMEOUT)
            if child.returncode != 0:
                raise SystemExit(f'run.py: {stack.name} {stack.device} fit exited with {child.returncode}')
    return {'unit': 'bytes', 'rows': [row(stack, s) for stack, s in samples.items()]}


def counter(gpu, pid):
    """The GPU memory of process [pid], in bytes. Unified memory has none of
    its own: on the Mac it is the process's physical footprint."""
    if gpu == 'metal':
        libc = ctypes.CDLL('/usr/lib/libSystem.B.dylib', use_errno=True)
        usage = (ctypes.c_uint64 * 64)()
        if libc.proc_pid_rusage(pid, RUSAGE_INFO_V4, usage):
            raise OSError(ctypes.get_errno(), f'proc_pid_rusage of process {pid}')
        return usage[PHYS_FOOTPRINT]
    if gpu == 'nvidia':
        query = output('nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits')
        for line in query.splitlines():
            found, mib = (int(field) for field in line.split(','))
            if found == pid:
                return mib << 20
        raise SystemExit(f'run.py: nvidia-smi lists no process {pid}')
    counts = list(Path(f'/sys/class/kfd/kfd/proc/{pid}').glob('vram_*'))
    if not counts:
        raise SystemExit(f'run.py: KFD lists no process {pid}')
    return sum(int(path.read_text()) for path in counts)


def page_cache(files, case):
    """Drops [files] from the page cache (cold) or reads them into it (warm)."""
    buffer = bytearray(CHUNK)
    for path in files:
        with open(path, 'rb', buffering=0) as file:
            if case == 'cold':
                os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            else:
                while file.readinto(buffer):
                    pass


def read(files):
    """Nanoseconds to read [files] into a reused host buffer."""
    buffer = bytearray(CHUNK)
    started = time.monotonic_ns()
    for path in files:
        with open(path, 'rb', buffering=0) as file:
            while file.readinto(buffer):
                pass
    return time.monotonic_ns() - started


def load(probes, files):
    if not files:
        raise SystemExit('run.py: load takes the checkpoint files')
    paths = [str(Path(f).resolve()) for f in files]
    cases = ['cold', 'warm'] if sys.platform == 'linux' else ['warm']
    native = next(stack for stack in probes if stack.name == 'native')
    reader = Stack('read', 'host', None)
    pcie = Stack('floor', 'pcie', native.head)
    size = sum(Path(p).stat().st_size for p in paths)
    samples, sizes = {}, {}
    for _ in range(ROUNDS):
        for case in cases:
            for stack in [reader, *probes]:
                page_cache(paths, case)
                if stack is reader:
                    elapsed = read(paths)
                else:
                    found, _ = stack.run('load', *paths)
                    elapsed = found['load_ns']
                samples.setdefault((stack, case), []).append(elapsed)
                sizes[stack] = size
        if sys.platform == 'linux':
            found, _ = pcie.run('pcie')
            samples.setdefault((pcie, None), []).append(found['copy_ns'])
            sizes[pcie] = found['bytes']
    rows = [row(stack, s, case=case, bytes=sizes[stack]) for (stack, case), s in samples.items()]
    return {'unit': 'ns', 'rows': rows}


def micro(probes, files):
    samples = {}
    for _ in range(ROUNDS):
        for calls in (1, 64):
            for stack in probes:
                found, _ = stack.run('add', calls, *stack.kernel())
                samples.setdefault((stack, calls), []).extend(int(t) for t in str(found['samples_ns']).split(','))
    return {'unit': 'ns', 'rows': [row(stack, s, case=f'add{calls}') for (stack, calls), s in samples.items()]}


# The vendor scenario's cases: each its name, the native probe's arguments,
# and whether it needs a bus to copy across, which the Mac's unified memory
# lacks. A case the probe answers [unsupported] for, such as in-queue waits
# on a device whose queues wait on no word, is left out.
MIB = 1 << 20
VENDOR_CASES = [
    ('release-1', ['release', 1], False),
    ('release-100', ['release', 100], False),
    ('waits-4', ['waits', 4], False),
    ('copy-h2d-256M', ['copy', 'h2d', 256 * MIB], True),
    ('copy-d2h-256M', ['copy', 'd2h', 256 * MIB], True),
    ('copy-d2d-256M', ['copy', 'd2d', 256 * MIB], False),
    ('copy-d2d-16B', ['copy', 'd2d', 16], False),
    ('alloc-64K', ['alloc', 64 << 10], False),
    ('alloc-64M', ['alloc', 64 * MIB], False),
    ('map-host-256M', ['map_host', 256 * MIB], False),
]


def vendor(probes, files):
    native = next(stack for stack in probes if stack.name == 'native')
    cases = [c for c in VENDOR_CASES if platform() != 'metal' or not c[2]]
    samples, unsupported = {}, set()
    for _ in range(ROUNDS):
        for name, args, _ in cases:
            if name in unsupported:
                continue
            found, _ = native.run(*args)
            if 'unsupported' in found:
                unsupported.add(name)
                print(f'run.py: native {native.device} has no {name}', file=sys.stderr)
                continue
            samples.setdefault(name, []).extend(int(t) for t in str(found['samples_ns']).split(','))
    return {'unit': 'ns', 'rows': [row(native, s, case=name) for name, s in samples.items()]}


SCENARIOS = {'ship': ship, 'start': start, 'load': load, 'fit': fit, 'micro': micro, 'vendor': vendor}


def measure(scenario, frameworks, files):
    gpu = platform()
    found = machine(gpu)
    with tempfile.TemporaryDirectory() as work:
        probes = stacks(gpu, frameworks, Path(work))
        utc = datetime.datetime.now(datetime.timezone.utc)
        record = {'utc': utc.isoformat(timespec='seconds'), 'commit': commit(), 'profile': 'release',
                  'load_average': os.getloadavg(), 'rounds': 1 if scenario == 'ship' else ROUNDS,
                  **SCENARIOS[scenario](probes, files)}
    path = RESULTS / f"{slug(found['gpu'])}-{utc.strftime('%Y%m%d')}.json"
    results = json.loads(path.read_text()) if path.exists() else {'date': utc.date().isoformat()}
    results['machine'] = found
    results.setdefault('scenarios', {})[scenario] = record
    path.write_text(json.dumps(results, indent=1) + '\n')
    print(table(scenario, record))
    print(f'\nwrote {path.relative_to(ROOT)}')


# Table

def median(r):
    return statistics.median(r['samples'])


def calls(r):
    return int(r['case'].removeprefix('add'))


def mib(n):
    return f'{n / (1 << 20):,.1f}'


def ms(n):
    return f'{n / 1e6:,.2f}'


def us(n):
    return f'{n / 1e3:,.1f}'


# Each scenario's columns between the stack's and the samples': a header, and
# what a row shows under it.
COLUMNS = {
    'ship': [('installed MiB', lambda r: mib(r['installed'])), ('vendor MiB', lambda r: mib(r['vendor'])),
             ('total MiB', lambda r: mib(median(r)))],
    'start': [('median ms', lambda r: ms(median(r))), ('min ms', lambda r: ms(min(r['samples']))),
              ('max ms', lambda r: ms(max(r['samples'])))],
    'load': [('cache', lambda r: r['case'] or ''), ('MiB', lambda r: mib(r['bytes'])),
             ('median ms', lambda r: ms(median(r))), ('GB/s', lambda r: f"{r['bytes'] / median(r):.2f}"),
             ('min ms', lambda r: ms(min(r['samples'])))],
    'fit': [('median MiB', lambda r: mib(median(r)))],
    'micro': [('calls', calls), ('median us', lambda r: us(median(r))), ('us/call', lambda r: us(median(r) / calls(r))),
              ('min us', lambda r: us(min(r['samples'])))],
    'vendor': [('case', lambda r: r['case']), ('median us', lambda r: us(median(r))),
               ('min us', lambda r: us(min(r['samples'])))],
}


def table(scenario, record):
    """A plain table of a scenario's record. x rig divides a row's median by
    the least of rig's in the same case."""
    rows = sorted(record['rows'], key=calls) if scenario == 'micro' else record['rows']

    def times_rig(r):
        rig = [median(x) for x in rows if x['stack'] == 'rig' and x.get('case') == r.get('case')]
        return f'{median(r) / min(rig):.2f}' if rig else ''
    columns = [('stack', lambda r: f"{r['stack']} ({r['device']})"), ('version', lambda r: r['version'] or ''),
               *COLUMNS[scenario], ('samples', lambda r: len(r['samples'])), ('x rig', times_rig)]
    cells = [[h for h, _ in columns]] + [[str(f(r)) for _, f in columns] for r in rows]
    widths = [max(len(line[i]) for line in cells) for i in range(len(columns))]
    left = {'stack', 'version', 'cache', 'case'}
    return '\n'.join('  '.join(c.ljust(w) if h in left else c.rjust(w) for c, w, (h, _) in zip(line, widths, columns))
                     for line in cells)


def main():
    parser = argparse.ArgumentParser(description=__doc__, usage=argparse.SUPPRESS,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('scenario', choices=SCENARIOS)
    for name in FRAMEWORKS:
        parser.add_argument(f'--{name}', metavar='PYTHON', help=f'an interpreter that has {name}')
    parser.add_argument('files', nargs='*', metavar='FILE', help="load's checkpoint files")
    args = parser.parse_args()
    frameworks = {name: getattr(args, name) for name in FRAMEWORKS if getattr(args, name)}
    measure(args.scenario, frameworks, args.files)


if __name__ == '__main__':
    main()
