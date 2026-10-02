"""Thumper baseline files (format v1), written for engines measured from Python.

The format and the machine key follow thumper's baseline-format.md, so a
baseline section written here and talon's own thumper section on the same
machine carry the same key. A section's [# ocaml:] line, which the grammar
requires, reads [none]; the engines that produced the rows are pinned by the
section's annotations instead.
"""

import os
import platform
import re
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

HEADER = "# thumper baseline v1"
SUITE = "# suite: "
MACHINE = "# machine: "
HOST = "# host: "
OCAML = "# ocaml: "

WORD = re.compile(r"[A-Za-z0-9._-]+")


def fingerprint(s: str) -> str:
    """[fingerprint s] is the FNV-1a 64-bit hash of [s]'s bytes, as 16 hex
    digits."""
    h = 0xCBF29CE484222325
    for byte in s.encode():
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def cpu_model() -> str:
    system = platform.system()
    if system == "Darwin":
        for sysctl in ["sysctl", "/usr/sbin/sysctl"]:
            try:
                return subprocess.run(
                    [sysctl, "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
            except (OSError, subprocess.CalledProcessError):
                continue
    elif system == "Linux":
        with open("/proc/cpuinfo") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return ""


def machine_key() -> str:
    override = os.environ.get("THUMPER_MACHINE")
    if override is not None:
        if not WORD.fullmatch(override):
            raise SystemExit(
                f"THUMPER_MACHINE={override!r} must be non-empty over [A-Za-z0-9._-]"
            )
        return override
    system = platform.system().lower() or "unknown"
    return fingerprint(f"{socket.gethostname()}:{system}:{cpu_model()}")


def host_description() -> str:
    system = platform.system().lower() or "unknown"
    system = "macos" if system == "darwin" else system
    parts = [socket.gethostname(), cpu_model(), platform.machine(), system]
    return " · ".join(re.sub(r"[\x00-\x1f\x7f]", " ", p) or "unknown" for p in parts)


def e3(x: float) -> str:
    return f"{x:.3e}"


def sampled_row(id: str, metric: str, samples: list[float]) -> str:
    """[sampled_row id metric samples] is the row of [samples], one call per
    sample. The median is derived from the printed samples, as thumper checks
    it."""
    if not all(WORD.fullmatch(s) for s in id.split("/")) or not WORD.fullmatch(metric):
        raise ValueError(f"invalid case id {id!r} or metric {metric!r}")
    if len(samples) < 3:
        raise ValueError(f"{id}: {len(samples)} samples, thumper needs at least 3")
    texts = [e3(s) for s in sorted(samples)]
    stored = [float(t) for t in texts]
    n = len(stored)
    if n % 2:
        median = e3(stored[n // 2])
    else:
        median = e3(stored[n // 2 - 1] / 2 + stored[n // 2] / 2)
    return f"{id}\t{metric}\tbatch=1\tn={n}\t{median}\t{' '.join(texts)}"


@dataclass
class Section:
    host: str
    ocaml: str
    annotations: list[str]
    rows: dict[tuple[str, str], str] = field(default_factory=dict)


def read(path: Path, suite: str) -> dict[str, Section]:
    """[read path suite] is the sections of the baseline at [path], or none if
    the file does not exist or is empty. It accepts the canonical layout
    only."""
    if not path.exists() or path.stat().st_size == 0:
        return {}
    lines = path.read_text().split("\n")
    if lines[-1] != "" or lines[:2] != [HEADER, SUITE + suite]:
        raise SystemExit(f"{path}: not a thumper baseline of suite {suite}")
    sections: dict[str, Section] = {}
    section = None
    for line in lines[2:-1]:
        if line.startswith(MACHINE):
            section = Section(host="", ocaml="", annotations=[])
            sections[line.removeprefix(MACHINE)] = section
        elif section is None or line == "":
            continue
        elif line.startswith(HOST):
            section.host = line.removeprefix(HOST)
        elif line.startswith(OCAML):
            section.ocaml = line.removeprefix(OCAML)
        elif line.startswith("# "):
            section.annotations.append(line.removeprefix("# "))
        else:
            id, metric, _ = line.split("\t", 2)
            section.rows[(id, metric)] = line
    return sections


def write(path: Path, suite: str, sections: dict[str, Section]) -> None:
    """[write path suite sections] writes the canonical file atomically:
    sections in key order, rows in (id, metric) order."""
    out = [HEADER, SUITE + suite]
    for key in sorted(sections):
        s = sections[key]
        out += ["", MACHINE + key, HOST + s.host, OCAML + s.ocaml]
        out += ["# " + a for a in s.annotations]
        if s.rows:
            out += [""] + [s.rows[k] for k in sorted(s.rows)]
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(out) + "\n")
    os.replace(tmp, path)
