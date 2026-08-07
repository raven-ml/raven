# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""End-to-end PyTorch-vs-Raven comparison harness.

Discovers every model directory under models/, runs both sides of each on the
same weights and the same batches, checks that they compute the same thing,
and only then reports timings.

    uv run bench/migrate/run.py [model ...] [--steps N] [--device D] [--json F]

The harness knows nothing about any particular model. A model is a directory
under models/ holding

    spec.json   hyperparameters, opaque to the harness, passed to both sides
    model.py    the PyTorch side, and the owner of the shared fixture
    model.ml    the Raven side, built by dune into model.exe
    dune

and both sides answer the same three-command protocol:

    <runner> variants
        prints {"variants": [{"variant": V, "device": D}, ...]} — what this
        build of this side can actually run, probed rather than assumed.

    <runner> run --spec S --fixture F --variant V --device D --steps N
                 --cache {cold,warm}
        trains for N steps and prints
        {"losses": [...], "step_ms": [...], "version": "..."}
        with losses[i] the loss at step i before update i, and step_ms[i] the
        wall clock of the whole of step i. --cache cold additionally means no
        compiled artifact may be served from an earlier process; each side
        owns the knob that arranges that.

    model.py fixture --spec S --out F        (PyTorch side only)
        writes the shared weights and training data, deterministically.

Adding a comparison is adding such a directory. Nothing here is edited.

Three numbers per (side, variant, device), never averaged together:

    cold        step 0 of a process whose caches are empty: trace, compile,
                and the first execution. A PyTorch eager user pays nearly
                nothing here, so this is where a compiling stack can lose
                badly and must be shown doing so.
    warm start  step 0 of a second process, with the on-disk compile caches
                the first one populated. What a user sees on every run after
                the first.
    steady      median of the per-step wall clock after the warmup steps the
                spec names, from that same second process. Minimum is
                reported alongside, because a loaded machine inflates the
                median and only the minimum is a floor.
    peak rss    peak resident set size of the second process, measured by
                this harness with wait4 so it is the same measurement on both
                sides. Host memory only: it does not see GPU allocations.

Agreement gates timing. Every run's loss trajectory is compared against the
reference run — the PyTorch side, eager, on CPU — element by element, and a
run that fails |a - b| <= atol + rtol * |b| is reported as a disagreement and
has its timings withheld. A timing comparison between two programs computing
different things is worse than no data.
"""

import argparse
import json
import os
import statistics
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MODELS = os.path.join(HERE, "models")

REFERENCE = ("pytorch", "eager", "cpu")


# Running a side


def capture(argv, env=None):
    """Run argv to completion; return (stdout, stderr, exit code, peak rss).

    Forked rather than subprocess.run so that wait4 yields this child's own
    rusage: peak RSS has to be per measurement, and RUSAGE_CHILDREN
    accumulates a maximum over every child ever waited for.
    """
    out = tempfile.TemporaryFile()
    err = tempfile.TemporaryFile()
    pid = os.fork()
    if pid == 0:
        try:
            os.dup2(out.fileno(), 1)
            os.dup2(err.fileno(), 2)
            os.execvpe(argv[0], argv, {**os.environ, **(env or {})})
        finally:
            os._exit(127)
    _, status, usage = os.wait4(pid, 0)
    out.seek(0)
    err.seek(0)
    # ru_maxrss is bytes on Darwin, kilobytes on Linux.
    scale = 1 if sys.platform == "darwin" else 1024
    return (
        out.read().decode(errors="replace"),
        err.read().decode(errors="replace"),
        os.waitstatus_to_exitcode(status),
        usage.ru_maxrss * scale,
    )


class Side:
    def __init__(self, name, argv, model_dir):
        self.name = name
        self.argv = argv
        self.dir = model_dir

    def json_call(self, args, env=None):
        stdout, stderr, code, rss = capture(self.argv + args, env)
        if code != 0:
            raise RuntimeError(
                f"{self.name} {' '.join(args)} exited {code}\n"
                + "\n".join(stderr.strip().splitlines()[-15:])
            )
        lines = [ln for ln in stdout.splitlines() if ln.startswith("{")]
        if not lines:
            raise RuntimeError(
                f"{self.name} {' '.join(args)} printed no JSON\n"
                + "\n".join((stdout + stderr).strip().splitlines()[-15:])
            )
        return json.loads(lines[-1]), rss

    def variants(self):
        return self.json_call(["variants"])[0]["variants"]

    def run(self, spec_path, fixture, variant, device, steps, cache):
        return self.json_call(
            [
                "run",
                "--spec", spec_path,
                "--fixture", fixture,
                "--variant", variant,
                "--device", device,
                "--steps", str(steps),
                "--cache", cache,
            ]
        )


def sides(model_dir):
    name = os.path.basename(model_dir)
    exe = os.path.join(
        ROOT, "_build", "default", "bench", "migrate", "models", name, "model.exe"
    )
    if not os.path.exists(exe):
        raise SystemExit(
            f"{exe} is missing. Build it with\n"
            f"    dune build bench/migrate/models/{name}/model.exe"
        )
    return [
        Side("pytorch", ["uv", "run", os.path.join(model_dir, "model.py")], model_dir),
        Side("raven", [exe], model_dir),
    ]


# Measurement


def agrees(losses, reference, atol, rtol):
    """(within tolerance, largest absolute loss deviation) against reference."""
    if len(losses) != len(reference) or not losses:
        return False, float("inf")
    pairs = list(zip(losses, reference))
    ok = all(abs(a - b) <= atol + rtol * abs(b) for a, b in pairs)
    return ok, max(abs(a - b) for a, b in pairs)


def measure(side, spec_path, spec, fixture, variant, device, steps):
    """Two processes: one with empty caches, one with the caches it left."""
    cold, _ = side.run(spec_path, fixture, variant, device, steps, "cold")
    warm, rss = side.run(spec_path, fixture, variant, device, steps, "warm")
    tail = warm["step_ms"][spec["warmup"] :]
    return {
        "side": side.name,
        "variant": variant,
        "device": device,
        "version": warm.get("version"),
        "cold_ms": cold["step_ms"][0],
        "warm_start_ms": warm["step_ms"][0],
        "steady_median_ms": statistics.median(tail),
        "steady_min_ms": min(tail),
        "peak_rss_mb": rss / 1e6,
        "losses": warm["losses"],
        "cold_losses": cold["losses"],
    }


def run_model(model_dir, steps=None, only_device=None):
    spec_path = os.path.join(model_dir, "spec.json")
    spec = json.load(open(spec_path))
    steps = steps or spec["steps"]

    results, errors = [], []
    with tempfile.TemporaryDirectory() as tmp:
        fixture = os.path.join(tmp, "fixture.safetensors")
        torch_side, raven_side = sides(model_dir)
        stdout, stderr, code, _ = capture(
            torch_side.argv + ["fixture", "--spec", spec_path, "--out", fixture]
        )
        if code != 0:
            raise SystemExit(f"fixture generation failed:\n{stderr}")

        for side in (torch_side, raven_side):
            for v in side.variants():
                if only_device and v["device"] != only_device:
                    continue
                label = f"{side.name}/{v['variant']}/{v['device']}"
                print(f"  {label} ...", end="", flush=True)
                t0 = time.perf_counter()
                try:
                    r = measure(
                        side, spec_path, spec, fixture, v["variant"], v["device"], steps
                    )
                    results.append(r)
                    print(f" {time.perf_counter() - t0:.0f}s")
                except RuntimeError as e:
                    errors.append((label, str(e)))
                    print(" FAILED")

    reference = next(
        (
            r
            for r in results
            if (r["side"], r["variant"], r["device"]) == REFERENCE
        ),
        None,
    )
    if reference is None:
        raise SystemExit(
            "the reference run (pytorch/eager/cpu) did not complete; "
            "no comparison is possible"
        )
    for r in results:
        ok, worst = agrees(
            r["losses"], reference["losses"], spec["agree_atol"], spec["agree_rtol"]
        )
        r["agrees"], r["max_loss_dev"] = ok, worst

    return {"model": os.path.basename(model_dir), "spec": spec,
            "results": results, "errors": errors}


# Report


def table(rows, headers):
    widths = [
        max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
        for i, h in enumerate(headers)
    ]
    def line(cells):
        return "  ".join(
            str(c).ljust(w) if i == 0 else str(c).rjust(w)
            for i, (c, w) in enumerate(zip(cells, widths))
        )

    rule = ["-" * w for w in widths]
    return "\n".join([line(headers), line(rule)] + [line(r) for r in rows])


def report(report_data):
    spec = report_data["spec"]
    print(f"\n{report_data['model']}: {spec.get('description', '')}\n")

    disagree = [r for r in report_data["results"] if not r["agrees"]]
    ref = next(
        r for r in report_data["results"]
        if (r["side"], r["variant"], r["device"]) == REFERENCE
    )
    print(
        f"agreement vs {'/'.join(REFERENCE)} over {len(ref['losses'])} steps "
        f"(tolerance {spec['agree_atol']} + {spec['agree_rtol']}*|ref|):"
    )
    print(table(
        [[f"{r['side']}/{r['variant']}/{r['device']}",
          "ok" if r["agrees"] else "DISAGREES",
          f"{r['max_loss_dev']:.2e}"]
         for r in report_data["results"]],
        ["run", "", "max |dloss|"],
    ))
    print(f"\nloss trajectory: {ref['losses'][0]:.4f} -> {ref['losses'][-1]:.4f}")

    if disagree:
        print(
            "\nTIMINGS WITHHELD: "
            + ", ".join(f"{r['side']}/{r['variant']}/{r['device']}" for r in disagree)
            + " does not compute the same thing as the reference."
        )
        return 1

    print("\ntimings (ms):")
    print(table(
        [[f"{r['side']}/{r['variant']}/{r['device']}",
          f"{r['cold_ms']:.0f}", f"{r['warm_start_ms']:.0f}",
          f"{r['steady_median_ms']:.1f}", f"{r['steady_min_ms']:.1f}",
          f"{r['peak_rss_mb']:.0f}"]
         for r in report_data["results"]],
        ["run", "cold", "warm start", "steady med", "steady min", "peak rss MB"],
    ))

    base = next(
        r for r in report_data["results"]
        if (r["side"], r["variant"], r["device"]) == REFERENCE
    )["steady_median_ms"]
    print("\nsteady state relative to pytorch/eager/cpu (>1 is slower):")
    print(table(
        [[f"{r['side']}/{r['variant']}/{r['device']}",
          f"{r['steady_median_ms'] / base:.2f}x"]
         for r in report_data["results"]],
        ["run", "vs ref"],
    ))

    for label, err in report_data["errors"]:
        print(f"\n{label} FAILED:\n{err}")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="*", help="model names (default: all)")
    p.add_argument("--steps", type=int, help="override the spec's step count")
    p.add_argument("--device", help="restrict to one device")
    p.add_argument("--json", help="also write the raw results here")
    args = p.parse_args()

    names = args.models or sorted(
        d for d in os.listdir(MODELS)
        if os.path.isfile(os.path.join(MODELS, d, "spec.json"))
    )
    status, everything = 0, []
    for name in names:
        print(f"{name}:")
        data = run_model(os.path.join(MODELS, name), args.steps, args.device)
        everything.append(data)
        status |= report(data)
    if args.json:
        json.dump(everything, open(args.json, "w"), indent=2)
    return status


if __name__ == "__main__":
    sys.exit(main())
