# /// script
# requires-python = "==3.13.*"
# dependencies = ["duckdb==1.5.6", "numpy==2.5.3", "polars==1.44.2", "pyarrow==25.0.1"]
# [tool.uv]
# exclude-newer = "2026-10-01T00:00:00Z"
# ///
"""Measure the DuckDB and Polars baselines into a thumper file.

usage: uv run baseline.py WORKLOAD SIZE DATA [--runs N] [--output FILE]
                          [--allow-load]

WORKLOAD is h2o (its group-by and join questions, SIZE 1e7 or 1e8) or tpch
(SIZE is the scale factor, 1 or 10), read from the data generated under DATA.
The CI sizes, 1e6 and 0.1, also run, as a smoke test of the harness.
Every question runs once to warm up, then N times (default 5); each timed run
is one sample of the case WORKLOAD-ID/QUESTION/ENGINE, e.g.
groupby/1e7/q01/duckdb or tpch/sf1/q07/polars.

The rows go into this machine's section of FILE (default
h2o-baselines.thumper or tpch-baselines.thumper next to this script),
replacing the rows of the same cases and keeping the others. A section pins
the engines' versions, Polars' engine, the thread count, the memory, the
Python version and the data's generator, and a run whose pins differ from the
section's is refused before it measures: delete the section to re-baseline
under new pins.

A run refuses to start on a loaded machine, as thumper's gate does (1-minute
load average at least half the cores). --allow-load measures anyway, and only
into an --output other than the committed baselines.

A case during which the machine swapped pages in or out is not recorded. A
case whose spread, (max - min) / median, is over 5% is recorded but listed.
Either makes the run exit nonzero.
"""

import argparse
import os
import platform
import re
import statistics
import subprocess
from pathlib import Path

import duckdb
import numpy as np

import h2o
import h2o_data
import thumper
import tpch
from workload import ENGINES, check_engines, check_tables, timed, versions

HERE = Path(__file__).resolve().parent


def swapped_pages() -> int:
    """[swapped_pages ()] is the number of pages the system has swapped in or
    out since it booted."""
    system = platform.system()
    if system == "Darwin":
        vm_stat = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, check=True
        ).stdout
        counts = re.findall(r"^Swap(?:ins|outs):\s+(\d+)", vm_stat, re.MULTILINE)
        return sum(map(int, counts))
    if system == "Linux":
        with open("/proc/vmstat") as vmstat:
            return sum(
                int(value)
                for name, value in map(str.split, vmstat)
                if name in ("pswpin", "pswpout")
            )
    raise SystemExit(f"cannot read swap activity on {system}")


def measure(workload, runs: int) -> tuple[dict[str, list[float]], list[str], list[str]]:
    """[measure workload runs] is the samples of every case that ran without
    swapping, the cases that swapped, and the cases whose spread is over 5%."""
    samples, swapped, noisy = {}, [], []
    for engine_class in ENGINES:
        engine = engine_class(workload)
        for question in workload.questions:
            id = f"{workload.id}/{question.name}/{engine.name}"
            before = swapped_pages()
            timed(engine, question)
            times = [timed(engine, question) for _ in range(runs)]
            median = statistics.median(times)
            spread = (max(times) - min(times)) / median
            if swapped_pages() > before:
                swapped.append(id)
                print(f"{id:<28} swapped, not recorded", flush=True)
                continue
            samples[id] = times
            flag = ""
            if spread > 0.05:
                noisy.append(id)
                flag = "  spread over 5%"
            print(f"{id:<28} {median:10.4f} s  spread {spread:6.1%}{flag}", flush=True)
        # Free this engine's tables before the next one loads its own.
        del engine
    return samples, swapped, noisy


def section(path: Path, suite: str, data: str) -> tuple[dict, thumper.Section]:
    """[section path suite data] is the baseline at [path] and this machine's
    section in it, created if absent. It refuses a section pinned otherwise
    than this run, whose data comes from [data]."""
    memory = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    pins = [
        f"engines: {versions()}",
        "polars engine: auto",
        f"threads: {os.cpu_count()}",
        f"memory: {memory / 2**30:.0f} GiB",
        f"python: {platform.python_version()}",
        f"data: {data}",
    ]
    sections = thumper.read(path, suite)
    key = thumper.machine_key()
    found = sections.setdefault(
        key, thumper.Section(thumper.host_description(), "none", pins)
    )
    if found.annotations != pins:
        raise SystemExit(
            f"{path}: machine {key} pins\n  {found.annotations}\nbut this run is\n"
            f"  {pins}\nDelete the section to re-baseline under the new pins."
        )
    return sections, found


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("workload", choices=["h2o", "tpch"])
    parser.add_argument("size")
    parser.add_argument("data", type=Path)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-load", action="store_true")
    args = parser.parse_args()
    if args.runs < 3:
        raise SystemExit("--runs must be at least 3, thumper's minimum sample count")
    suite = f"{args.workload}-baselines"
    committed = HERE / f"{suite}.thumper"
    output = (args.output or committed).resolve()
    if args.allow_load and output == committed:
        raise SystemExit(
            "--allow-load needs an --output other than the committed baselines"
        )
    if args.workload == "h2o":
        workloads = h2o.workloads(args.data, args.size)
        data = f"numpy {np.__version__} · seed {h2o_data.SEED}"
    else:
        workloads = [tpch.workload(args.data, args.size)]
        data = f"dbgen of duckdb {duckdb.__version__}"
    for workload in workloads:
        check_tables(workload)
    check_engines()
    cores, average = os.cpu_count(), os.getloadavg()[0]
    if average / cores >= 0.5 and not args.allow_load:
        raise SystemExit(
            f"the machine is loaded (1-minute load {average:.1f} on {cores} cores);"
            " timings would not gate"
        )
    sections, found = section(output, suite, data)
    swapped, noisy = [], []
    for workload in workloads:
        samples, workload_swapped, workload_noisy = measure(workload, args.runs)
        swapped += workload_swapped
        noisy += workload_noisy
        for id, times in samples.items():
            found.rows[(id, "wall_time")] = thumper.sampled_row(id, "wall_time", times)
    thumper.write(output, suite, sections)
    print(f"wrote {output}")
    if swapped or noisy:
        raise SystemExit(
            "".join(f"swapped, not recorded: {id}\n" for id in swapped)
            + "".join(f"spread over 5%: {id}\n" for id in noisy)
            + "These cases are not fit to gate; re-run them on a quieter machine."
        )


if __name__ == "__main__":
    main()
