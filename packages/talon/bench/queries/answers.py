# /// script
# requires-python = "==3.13.*"
# dependencies = ["duckdb==1.5.6", "numpy==2.5.3", "polars==1.44.2", "pyarrow==25.0.1"]
# [tool.uv]
# exclude-newer = "2026-10-01T00:00:00Z"
# ///
"""Check both engines against the committed answers at the CI size.

usage: uv run answers.py DATA [--record]

Runs every question once at the CI size (H2O at 1e6 rows, TPC-H at scale
factor 0.1, generated under DATA) on DuckDB and Polars, and compares each
engine's answer with the committed one in answers/: its row count, and the
rows the committed file keeps (see answer.py). A committed answer that no
question produces any more fails the check. With --record, it checks that
Polars' answers equal DuckDB's and, when every one does, writes DuckDB's,
keeping the committed files that still agree with them, so that a float
summed in another order does not rewrite its answer, and removing the files of
questions that are gone.
"""

import argparse
from pathlib import Path

import polars as pl

import answer
import h2o
import tpch
from workload import ENGINES, check_engines, check_tables

ANSWERS = Path(__file__).resolve().parent / "answers"


def ci_workloads(data: Path):
    return h2o.workloads(data, "1e6") + [tpch.workload(data, "0.1")]


def check(expected: pl.DataFrame, rows: int, stride: int, got: pl.DataFrame):
    if got.height != rows:
        return [f"{got.height} rows, expected {rows}"]
    return answer.compare(expected, answer.thin(got, stride))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data", type=Path)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    workloads = ci_workloads(args.data)
    for workload in workloads:
        check_tables(workload)
    check_engines()
    committed = answer.read_index(ANSWERS)
    ran, index, recorded = set(), {}, {}
    failures = 0
    for workload in workloads:
        engines = [engine_class(workload) for engine_class in ENGINES]
        for question in workload.questions:
            id = f"{workload.id}/{question.name}"
            ran.add(id)
            path = answer.path(ANSWERS, id)
            answers = {}
            for engine in engines:
                engine.run(question)
                answers[engine.name] = answer.canonical(
                    engine.answer(), workload.ordered
                )
                engine.clear()
            reference = answers["duckdb"]
            if args.record:
                problems = answer.compare(reference, answers["polars"])
                index[id] = (reference.height, answer.stride(reference.height))
                thinned = answer.thin(reference, index[id][1])
                if (
                    committed.get(id) != index[id]
                    or not path.exists()
                    or answer.compare(answer.read(path, reference.schema), thinned)
                ):
                    recorded[path] = thinned
            elif id in committed:
                expected = answer.read(path, reference.schema)
                problems = [
                    f"{name}: {problem}"
                    for name, got in answers.items()
                    for problem in check(expected, *committed[id], got)
                ]
            else:
                problems = ["no committed answer; record one with --record"]
            print(f"{id:<20} {'FAIL' if problems else 'ok'}", flush=True)
            for problem in problems:
                print(f"  {problem}")
            failures += bool(problems)
    gone = sorted(committed.keys() - ran)
    if failures:
        raise SystemExit(f"{failures} questions disagree")
    if args.record:
        for path, frame in recorded.items():
            answer.write(path, frame)
        for id in gone:
            answer.path(ANSWERS, id).unlink(missing_ok=True)
        answer.write_index(ANSWERS, index)
        print(f"rewrote {len(recorded)} answers, removed {len(gone)}")
    elif gone:
        raise SystemExit(
            "committed answers to no question (re-record with --record):\n  "
            + "\n  ".join(gone)
        )


if __name__ == "__main__":
    main()
