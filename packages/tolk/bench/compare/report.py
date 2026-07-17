#!/usr/bin/env python3
"""Join the tolk and reference timing JSONs into a per-stage comparison.

Reads tolk.json / tinygrad.json (timing rows) and tolk.verify.json /
tinygrad.verify.json (per-workload kernel count and first-kernel source)
from a directory, prints the human comparison table, writes compare.tsv,
and runs the same-graph cross-check.

The tolk side times rangeify and schedule separately; the reference side
reports the two combined as schedule_linear. The report shows tolk's two
stages, then a synthetic tolk schedule_linear (rangeify + schedule) joined
against the reference — that row is the headline for stages 2-3.

Same-graph check: per workload, kernel counts must match and the first
kernel's rendered source must be byte-identical across sides. A mismatch
means the two sides are not compiling the same graph; the report fails
loudly rather than presenting a meaningless comparison.

Run from the repo root:  uv run packages/tolk/bench/compare/report.py [dir]
"""

import json
import os
import re
import sys

# tolk stage -> None (tolk-only) or the reference stage it joins against.
DISPLAY_STAGES = [
    ("rangeify", None),
    ("schedule", None),
    ("schedule_linear", "schedule_linear"),
    ("codegen", "codegen"),
    ("linearize", "linearize"),
    ("render", "render"),
    ("compile", "compile"),
]


def index(rows):
    return {(r["workload"], r["size"], r["stage"]): r for r in rows}


def workload_order(rows):
    seen = []
    for r in rows:
        key = (r["workload"], r["size"])
        if key not in seen:
            seen.append(key)
    return seen


def synth_schedule_linear(tolk, workload, size):
    rangeify = tolk.get((workload, size, "rangeify"))
    schedule = tolk.get((workload, size, "schedule"))
    if rangeify is None or schedule is None:
        return None
    return {
        "ms_median": rangeify["ms_median"] + schedule["ms_median"],
        "ms_min": rangeify["ms_min"] + schedule["ms_min"],
        "n_kernels": rangeify["n_kernels"],
    }


def build_table(tolk_rows, tg_rows):
    tolk = index(tolk_rows)
    tg = index(tg_rows)
    table = []
    for workload, size in workload_order(tolk_rows):
        for stage, tg_stage in DISPLAY_STAGES:
            if stage == "schedule_linear":
                t = synth_schedule_linear(tolk, workload, size)
            else:
                t = tolk.get((workload, size, stage))
            if t is None:
                continue
            g = tg.get((workload, size, tg_stage)) if tg_stage else None
            tolk_ms = t["ms_median"]
            tg_ms = g["ms_median"] if g else None
            ratio = (tolk_ms / tg_ms) if tg_ms else None
            table.append({
                "workload": workload, "size": size, "stage": stage,
                "tolk_ms": tolk_ms, "tolk_min": t["ms_min"],
                "tg_ms": tg_ms, "tg_min": g["ms_min"] if g else None,
                "ratio": ratio, "n_kern": t["n_kernels"],
            })
    return table


def fmt_ms(x):
    return f"{x:.3f}" if x is not None else "-"


def fmt_ratio(x):
    return f"{x:.2f}x" if x is not None else "-"


def print_table(table):
    header = ["workload", "size", "stage", "tolk_ms", "tg_ms", "ratio",
              "n_kern"]
    rows = [header] + [[
        r["workload"], r["size"], r["stage"], fmt_ms(r["tolk_ms"]),
        fmt_ms(r["tg_ms"]), fmt_ratio(r["ratio"]), str(r["n_kern"]),
    ] for r in table]
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    for i, row in enumerate(rows):
        print("  ".join(c.ljust(widths[j]) for j, c in enumerate(row)))
        if i == 0:
            print("  ".join("-" * widths[j] for j in range(len(header))))


def write_tsv(table, path):
    cols = ["workload", "size", "stage", "tolk_ms_median", "tolk_ms_min",
            "tg_ms_median", "tg_ms_min", "ratio", "n_kern"]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in table:
            f.write("\t".join([
                r["workload"], r["size"], r["stage"],
                f"{r['tolk_ms']:.6f}", f"{r['tolk_min']:.6f}",
                f"{r['tg_ms']:.6f}" if r["tg_ms"] is not None else "",
                f"{r['tg_min']:.6f}" if r["tg_min"] is not None else "",
                f"{r['ratio']:.4f}" if r["ratio"] is not None else "",
                str(r["n_kern"]),
            ]) + "\n")


# Per-node growth over ~2.3x per doubling is the superlinearity threshold from
# the plan; normalised to a size ratio it flags at 1.15 (2.3 / 2).
SUPERLINEAR = 1.15


def numeric_size(size):
    m = re.search(r"(\d+)", size)
    return int(m.group(1)) if m else None


def growth(ms_j, ms_i, n_j, n_i):
    if ms_i in (None, 0) or ms_j is None or n_i == 0:
        return None
    return (ms_j / ms_i) / (n_j / n_i)


def scaling_curves(table):
    sizes = {}
    for r in table:
        sizes.setdefault(r["workload"], set()).add(r["size"])
    return [wl for wl in dict.fromkeys(r["workload"] for r in table)
            if len(sizes[wl]) > 1]


def print_scaling(table):
    laddered = scaling_curves(table)
    if not laddered:
        return []
    print("\nScaling curves (ms vs size; growth = per-node ms ratio over the "
          f"previous point, normalised to size ratio; SUPER = > {SUPERLINEAR}):")
    out = []
    for wl in laddered:
        rows = [r for r in table if r["workload"] == wl]
        stages = list(dict.fromkeys(r["stage"] for r in rows))
        for stage in stages:
            pts = sorted((r for r in rows if r["stage"] == stage),
                         key=lambda r: numeric_size(r["size"]))
            print(f"\n{wl} / {stage}")
            header = ["size", "n", "tolk_ms", "tg_ms", "tolk_grow",
                      "tg_grow", "flag"]
            lines = [header]
            prev = None
            for r in pts:
                n = numeric_size(r["size"])
                tg = growth(r["tolk_ms"], prev["tolk_ms"], n,
                            numeric_size(prev["size"])) if prev else None
                gg = growth(r["tg_ms"], prev["tg_ms"], n,
                            numeric_size(prev["size"])) if prev else None
                flag = "SUPER" if (tg and tg > SUPERLINEAR) or \
                    (gg and gg > SUPERLINEAR) else ""
                lines.append([
                    r["size"], str(n), fmt_ms(r["tolk_ms"]),
                    fmt_ms(r["tg_ms"]),
                    f"{tg:.2f}" if tg is not None else "-",
                    f"{gg:.2f}" if gg is not None else "-", flag])
                out.append({"workload": wl, "size": r["size"], "n": n,
                            "stage": stage, "tolk_ms": r["tolk_ms"],
                            "tg_ms": r["tg_ms"], "tolk_grow": tg,
                            "tg_grow": gg, "super": bool(flag)})
                prev = r
            widths = [max(len(row[i]) for row in lines)
                      for i in range(len(header))]
            for i, row in enumerate(lines):
                print("  " + "  ".join(c.ljust(widths[j])
                                       for j, c in enumerate(row)))
                if i == 0:
                    print("  " + "  ".join("-" * widths[j]
                                           for j in range(len(header))))
    return out


def write_scaling_tsv(rows, path):
    cols = ["workload", "size", "n", "stage", "tolk_ms", "tg_ms",
            "tolk_grow", "tg_grow", "super"]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join([
                r["workload"], r["size"], str(r["n"]), r["stage"],
                f"{r['tolk_ms']:.6f}",
                f"{r['tg_ms']:.6f}" if r["tg_ms"] is not None else "",
                f"{r['tolk_grow']:.4f}" if r["tolk_grow"] is not None else "",
                f"{r['tg_grow']:.4f}" if r["tg_grow"] is not None else "",
                "1" if r["super"] else "0",
            ]) + "\n")


def verify_same_graph(tolk_v, tg_v):
    ok = True
    print("\nSame-graph verification (tolk vs reference):")
    for workload in tolk_v:
        t = tolk_v[workload]
        g = tg_v.get(workload)
        if g is None:
            print(f"  {workload:14} FAIL  missing from reference")
            ok = False
            continue
        n_ok = t["n_kernels"] == g["n_kernels"]
        src_ok = t["first_kernel_src"] == g["first_kernel_src"]
        status = "PASS" if (n_ok and src_ok) else "FAIL"
        detail = ""
        if not n_ok:
            detail += f" n_kernels {t['n_kernels']} != {g['n_kernels']}"
        if not src_ok:
            detail += " first-kernel source differs"
        print(f"  {workload:14} {status}  n_kern={t['n_kernels']}"
              f"  src_bytes={len(t['first_kernel_src'])}{detail}")
        ok = ok and n_ok and src_ok
    return ok


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    tolk_rows = load(os.path.join(out_dir, "tolk.json"))
    tg_rows = load(os.path.join(out_dir, "tinygrad.json"))
    tolk_v = load(os.path.join(out_dir, "tolk.verify.json"))
    tg_v = load(os.path.join(out_dir, "tinygrad.verify.json"))

    table = build_table(tolk_rows, tg_rows)
    print_table(table)
    tsv_path = os.path.join(out_dir, "compare.tsv")
    write_tsv(table, tsv_path)
    print(f"\nwrote {tsv_path}")

    scaling_rows = print_scaling(table)
    if scaling_rows:
        scaling_path = os.path.join(out_dir, "scaling.tsv")
        write_scaling_tsv(scaling_rows, scaling_path)
        print(f"\nwrote {scaling_path}")

    ok = verify_same_graph(tolk_v, tg_v)
    if not ok:
        print("\nSAME-GRAPH CHECK FAILED: the two sides are not compiling the "
              "same graph; the comparison above is not meaningful.")
        sys.exit(1)
    print("\nSame-graph check passed: both sides compile identical kernels.")


if __name__ == "__main__":
    main()
