from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_UBENCH = _ROOT / "vendor" / "ubench"
if str(_UBENCH) not in sys.path:
    sys.path.insert(0, str(_UBENCH))

import ubench  # type: ignore


DATA_DIR = Path(__file__).resolve().parent / "data"


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    transactions = pd.read_csv(
        DATA_DIR / "transactions.csv",
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
        DATA_DIR / "customers.csv",
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


TRANSACTIONS, CUSTOMERS = _load_data()


def build_benchmarks() -> List[Any]:
    benches: List[Any] = []

    def bench_filter() -> None:
        filtered = TRANSACTIONS[
            (TRANSACTIONS["amount"] > 120.0)
            & (TRANSACTIONS["quantity"] >= 3)
            & (TRANSACTIONS["region"] == "EMEA")
        ]
        float(filtered["amount"].sum())

    benches.append(ubench.bench("Filter/high_value (pandas)", bench_filter))

    def bench_group() -> None:
        grouped = (
            TRANSACTIONS.groupby(["category", "region"])["amount"].sum()
        )
        float(grouped.sum())

    benches.append(ubench.bench("Group/category_region (pandas)", bench_group))

    def bench_join() -> None:
        joined = TRANSACTIONS.merge(CUSTOMERS, on="customer_id", how="left")
        float(joined["amount"].sum())

    benches.append(ubench.bench("Join/customer_lookup (pandas)", bench_join))

    def bench_sort() -> None:
        sorted_df = TRANSACTIONS.sort_values("amount", ascending=False)
        float(sorted_df["amount"].iloc[0])

    benches.append(ubench.bench("Sort/amount_desc (pandas)", bench_sort))

    return benches


def default_config() -> ubench.Config:
    return ubench.Config.default().build()


def main() -> None:
    benchmarks = build_benchmarks()
    config = default_config()
    ubench.run(benchmarks, config=config, output_format="pretty", verbose=False)


if __name__ == "__main__":
    main()
