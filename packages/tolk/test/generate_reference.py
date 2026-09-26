#!/usr/bin/env python3
"""Generate a complete reference corpus without overwriting expectations.

Each driver runs in a fresh workspace against an archived Git revision. The
output manifest records that revision, the environment, and every output hash.
Missing or unexpected files fail generation, even if a driver exits zero.

Example:
    python packages/tolk/test/generate_reference.py --revision HEAD \
        --output _reference/baseline --check-cpu-sources

The optional CPU source check requires clang and compiles each rendered C
translation unit without linking or executing it.
"""

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

# Generate and retain these target outputs, but do not imply exact comparison
# with a deliberately different implementation (see Tolk's DIVERGENCES.md).
REFERENCE_ONLY = {
    "golden/amdqueue": {
        "exec_gfx942.expected": "Resident multi-XCC scratch partitioning differs from upstream",
    },
}


def git(reference, *args):
    return subprocess.check_output(["git", "-C", str(reference), *args])


def check_cpu_sources(output, compiler, env):
    """Compile CPU source fixtures, preserving separate tensor-kernel units."""
    version = subprocess.check_output(
        [compiler, "--version"], env=env, text=True, stderr=subprocess.STDOUT,
    ).strip()
    command = [compiler, "-x", "c", "-fsyntax-only", "-"]
    sources = sorted(output.glob("parity/*/stage7_cpu.expected"))
    sources += sorted(output.glob("golden/codegen/clang_*.expected"))
    sources += sorted(output.glob("golden/cstyle/clang_*.expected"))
    files = {}
    for source in sources:
        name = source.relative_to(output).as_posix()
        # helpers.stage7_tensor joins independently compiled kernels with this
        # delimiter. Do not combine their repeated names or typedefs into one TU.
        units = source.read_text().split("\n---\n")
        results = []
        for index, unit in enumerate(units):
            result = subprocess.run(
                command, input=unit, env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            results.append({
                "unit": index, "exit_code": result.returncode,
                "diagnostics": result.stdout,
            })
        files[name] = results
    return {"command": command, "compiler_version": version, "files": files}


def generate(reference, revision, output, suite, check_cpu=False):
    compiler = shutil.which("clang") if check_cpu else None
    if check_cpu and compiler is None:
        raise RuntimeError("--check-cpu-sources requires clang on PATH")
    revision = git(reference, "rev-parse", f"{revision}^{{commit}}").decode().strip()
    drivers = []
    if suite in ("all", "golden"):
        drivers.extend(sorted(HERE.glob("golden/*/generate_expected.py")))
    if suite in ("all", "parity"):
        drivers.extend(sorted(HERE.glob("parity/*/main.py")))
    if not drivers:
        raise RuntimeError("no reference drivers found")
    output.mkdir(parents=True, exist_ok=False)
    # Do not inherit BEAM, DEV, renderer flags, or Python import overrides.
    # Keep only OS paths/locales needed by the interpreter and native compilers.
    env = {key: os.environ[key] for key in
           ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "SYSTEMROOT")
           if key in os.environ}
    env.update(NO_COLOR="1", NUM_CPU_THREADS="8", PYTHONHASHSEED="0",
               PYTHONNOUSERSITE="1")
    manifest = {
        "revision": revision,
        "python": sys.version,
        "python_executable": sys.executable,
        "python_packages": dict(sorted(
            (dist.metadata["Name"], dist.version)
            for dist in importlib.metadata.distributions()
            if dist.metadata["Name"])),
        "environment": {key: env[key] for key in
                        ("NO_COLOR", "NUM_CPU_THREADS", "PYTHONHASHSEED",
                         "PYTHONNOUSERSITE")},
        "driver_sources": {},
        "fixture_inputs": {},
        "drivers": [],
    }
    failures = []
    with tempfile.TemporaryDirectory(prefix="tolk-reference-") as tmp:
        workspace = Path(tmp)
        with tarfile.open(fileobj=io.BytesIO(git(reference, "archive", revision))) as archive:
            archive.extractall(workspace / "_tinygrad", filter="data")
        # Preserve driver-relative imports and clone paths, but never copy old
        # expectations: a skipped case must not look like generated output.
        for source in HERE.rglob("*.py"):
            manifest["driver_sources"][source.relative_to(HERE).as_posix()] = (
                hashlib.sha256(source.read_bytes()).hexdigest())
            destination = workspace / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        # Queue references parse real compiled program metadata. Stage and hash
        # those inputs, without copying any existing comparison expectations.
        for source in sorted((HERE / "fixtures").rglob("*")):
            if source.suffix not in {".cubin", ".hsaco"}:
                continue
            manifest["fixture_inputs"][source.relative_to(HERE).as_posix()] = (
                hashlib.sha256(source.read_bytes()).hexdigest())
            destination = workspace / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        for driver in drivers:
            name = driver.parent.relative_to(HERE).as_posix()
            print(f"Generating {name}", flush=True)
            staged = workspace / driver.relative_to(ROOT)
            destination = output / name
            destination.mkdir(parents=True)
            reference_only = REFERENCE_ONLY.get(name, {})
            # Reviewed local semantics may have a separate .tolk.expected snapshot.
            # It is never generated by upstream and must not enter its inventory.
            expected = {p.name for p in driver.parent.glob("*.expected")
                        if not p.name.endswith(".tolk.expected")} | set(reference_only)
            if not expected:
                failures.append(f"{name}: no expected-file inventory")
            command = [sys.executable, str(staged)]
            if name.startswith("golden/"):
                command += ["--tinygrad", str(workspace / "_tinygrad"), "--output", str(staged.parent)]
            result = subprocess.run(
                command, cwd=workspace, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            (destination / "generation.log").write_text(result.stdout)
            actual = {p.name for p in staged.parent.glob("*.expected")}
            missing, extra = sorted(expected - actual), sorted(actual - expected)
            files = {}
            for filename in sorted(actual):
                source = staged.parent / filename
                files[filename] = hashlib.sha256(source.read_bytes()).hexdigest()
                shutil.copyfile(source, destination / filename)
            manifest["drivers"].append({
                "name": name, "exit_code": result.returncode,
                "missing": missing, "unexpected": extra, "reference_only": reference_only, "files": files,
            })
            if result.returncode or missing or extra:
                failures.append(
                    f"{name}: exit={result.returncode}, missing={missing}, unexpected={extra}"
                )
    if check_cpu:
        checks = check_cpu_sources(output, compiler, env)
        manifest["cpu_source_checks"] = checks
        for name, units in checks["files"].items():
            for result in units:
                if result["exit_code"]:
                    failures.append(
                        f"{name}: CPU source unit {result['unit']} failed clang syntax checking"
                    )
    manifest["complete"] = not failures
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if failures:
        raise RuntimeError("incomplete reference corpus:\n" + "\n".join(failures))
    count = sum(len(driver["files"]) for driver in manifest["drivers"])
    print(f"Generated {count} files from {len(drivers)} drivers at {revision}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "_tinygrad")
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("all", "golden", "parity"), default="all")
    parser.add_argument("--check-cpu-sources", action="store_true",
                        help="require clang syntax checking of rendered CPU sources")
    args = parser.parse_args()
    generate(args.reference.resolve(), args.revision, args.output.resolve(), args.suite,
             check_cpu=args.check_cpu_sources)


if __name__ == "__main__":
    main()
