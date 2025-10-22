"""Generate synthetic fixtures for the Talon dataframe benchmarks."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class Customer:
    customer_id: int
    region: str
    segment: str
    status: str
    loyalty_score: float
    tenure_years: int


def _create_customers(rng: random.Random, base_id: int = 1000, count: int = 1500) -> List[Customer]:
    regions = ["EMEA", "AMER", "APAC", "LATAM"]
    segments = ["Enterprise", "Growth", "SMB", "Consumer"]
    statuses = ["active", "at_risk", "inactive"]

    customers: List[Customer] = []
    for offset in range(count):
        customer_id = base_id + offset
        region = rng.choice(regions)
        segment = rng.choices(segments, weights=[0.25, 0.2, 0.3, 0.25])[0]
        status = rng.choices(statuses, weights=[0.65, 0.2, 0.15])[0]
        loyalty_score = round(rng.uniform(20.0, 98.0), 2)
        tenure_years = rng.randint(1, 12)
        customers.append(
            Customer(
                customer_id=customer_id,
                region=region,
                segment=segment,
                status=status,
                loyalty_score=loyalty_score,
                tenure_years=tenure_years,
            )
        )
    return customers


def _save_customers(path: Path, rows: Iterable[Customer]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "customer_id",
            "segment",
            "region",
            "status",
            "loyalty_score",
            "tenure_years",
        ])
        for row in rows:
            writer.writerow(
                [
                    row.customer_id,
                    row.segment,
                    row.region,
                    row.status,
                    f"{row.loyalty_score:.2f}",
                    row.tenure_years,
                ]
            )


def _write_transactions(path: Path, customers: List[Customer], rng: random.Random, count: int = 40000) -> None:
    categories = [
        "electronics",
        "grocery",
        "fashion",
        "home",
        "sports",
        "toys",
    ]
    channels = ["online", "retail", "mobile", "partner"]
    promo_flags = ["none", "coupon", "bundle", "loyalty"]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "transaction_id",
                "customer_id",
                "region",
                "category",
                "channel",
                "amount",
                "quantity",
                "discount",
                "promo",
                "event_date",
            ]
        )

        for tx_id in range(1, count + 1):
            customer = rng.choice(customers)
            category = rng.choices(categories, weights=[0.2, 0.25, 0.15, 0.18, 0.12, 0.1])[0]
            channel = rng.choices(channels, weights=[0.45, 0.35, 0.15, 0.05])[0]
            quantity = rng.randint(1, 8)
            base_amount = rng.lognormvariate(4.0, 0.5)
            seasonal_factor = 0.85 + rng.random() * 0.4
            amount = round(base_amount * seasonal_factor, 2)
            discount = round(rng.uniform(0.0, 0.25), 3)
            promo = rng.choices(promo_flags, weights=[0.55, 0.2, 0.15, 0.1])[0]
            month = rng.randint(1, 12)
            day = rng.randint(1, 28)
            event_date = f"2024-{month:02d}-{day:02d}"

            writer.writerow(
                [
                    tx_id,
                    customer.customer_id,
                    customer.region,
                    category,
                    channel,
                    f"{amount:.2f}",
                    quantity,
                    f"{discount:.3f}",
                    promo,
                    event_date,
                ]
            )


def main() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(20240127)

    customers = _create_customers(rng)
    _save_customers(data_dir / "customers.csv", customers)
    _write_transactions(data_dir / "transactions.csv", customers, rng)

    print(f"Generated Talon fixtures in {data_dir}")


if __name__ == "__main__":
    main()
