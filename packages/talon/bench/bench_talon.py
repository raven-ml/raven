# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas==3.1.0"]
# ///
"""The pandas side of talon's bench, over the fixtures write_fixtures.exe
writes: each row's time per call.

usage: uv run packages/talon/bench/bench_talon.py DIR
"""

from __future__ import annotations

import gc
import sys
import timeit
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import pandas as pd

REPEATS = 5

Row = Tuple[str, Callable[[], object]]


# Timing


def time_per_call(fn: Callable[[], object]) -> float:
    """Seconds per call of fn: the fastest of REPEATS batches, each of as many
    calls as take 0.2 s. The collector stays on, as in a program."""
    timer = timeit.Timer(fn, setup=gc.enable)
    number, _ = timer.autorange()
    return min(timer.repeat(repeat=REPEATS, number=number)) / number


def format_time(seconds: float) -> str:
    for scale, unit in ((1.0, "s"), (1e-3, "ms"), (1e-6, "µs")):
        if seconds >= scale:
            return f"{seconds / scale:.2f}{unit}"
    return f"{seconds * 1e9:.2f}ns"


def run(rows: Sequence[Row]) -> None:
    width = max(len(name) for name, _ in rows)
    for name, fn in rows:
        print(f"{name:<{width}}  {format_time(time_per_call(fn))}", flush=True)


# Rows


def load_data(data: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    transactions = pd.read_csv(
        data / "transactions.csv",
        dtype={
            "transaction_id": "int32",
            "customer_id": "int32",
            "region": "category",
            "category": "category",
            "channel": "category",
            "amount": "float64",
            "quantity": "int32",
            "discount": "float64",
            "promo": "category",
            "event_date": "string",
        },
    )
    customers = pd.read_csv(
        data / "customers.csv",
        dtype={
            "customer_id": "int32",
            "segment": "category",
            "region": "category",
            "status": "category",
            "loyalty_score": "float64",
            "tenure_years": "int32",
        },
    )
    return transactions, customers


def build_rows(transactions: pd.DataFrame, customers: pd.DataFrame) -> List[Row]:
    def bench_filter() -> None:
        filtered = transactions[
            (transactions["amount"] > 120.0)
            & (transactions["quantity"] >= 3)
            & (transactions["region"] == "EMEA")
        ]
        float(filtered["amount"].sum())

    def bench_group() -> None:
        grouped = transactions.groupby(["category", "region"])["amount"].sum()
        float(grouped.sum())

    def bench_join() -> None:
        joined = transactions.merge(customers, on="customer_id", how="left")
        float(joined["amount"].sum())

    def bench_sort() -> None:
        sorted_df = transactions.sort_values("amount", ascending=False)
        float(sorted_df["amount"].iloc[0])

    return [
        ("Filter/high_value (pandas)", bench_filter),
        ("Group/category_region (pandas)", bench_group),
        ("Join/customer_lookup (pandas)", bench_join),
        ("Sort/amount_desc (pandas)", bench_sort),
    ]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    run(build_rows(*load_data(Path(sys.argv[1]))))
