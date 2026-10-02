# /// script
# dependencies = ["pyarrow==25.0.1"]
# ///
"""Writes the example's data: flights.csv and carriers.parquet.

Run from this directory with `uv run data.py`. The files are committed, so the
example needs neither Python nor the network. The flights are made up; the
carriers are the airlines of the New York flights of 2013.
"""

import csv
import random

import pyarrow as pa
import pyarrow.parquet as pq

CARRIERS = [
    ("AA", "American Airlines Inc."),
    ("B6", "JetBlue Airways"),
    ("DL", "Delta Air Lines Inc."),
    ("F9", "Frontier Airlines Inc."),
    ("HA", "Hawaiian Airlines Inc."),
    ("OO", None),
    ("UA", "United Air Lines Inc."),
    ("WN", "Southwest Airlines Co."),
]

ROUTES = {
    "AA": [("JFK", "MIA", 1089), ("LGA", "ORD", 733)],
    "B6": [("JFK", "BOS", 187), ("JFK", "FLL", 1069)],
    "DL": [("LGA", "ATL", 762), ("JFK", "LAX", 2475)],
    "F9": [("LGA", "DEN", 1620)],
    "HA": [("JFK", "HNL", 4983)],
    "OO": [("LGA", "DTW", 502)],
    "UA": [("EWR", "SFO", 2565), ("EWR", "IAH", 1400)],
    "WN": [("LGA", "MDW", 725), ("EWR", "DEN", 1605)],
}

random.seed(2013)
rows = []
for i in range(60):
    carrier = random.choice([c for c, _ in CARRIERS])
    origin, dest, distance = random.choice(ROUTES[carrier])
    month, day = random.randint(1, 12), random.randint(1, 28)
    sched = random.choice([600, 745, 900, 1130, 1415, 1700, 1955])
    cancelled = random.random() < 0.08
    if cancelled:
        dep_time, dep_delay, arr_delay = "NA", "NA", "NA"
    else:
        dep_delay = round(random.expovariate(1 / 18) - 6)
        arr_delay = dep_delay + random.randint(-15, 15)
        minutes = sched // 100 * 60 + sched % 100 + dep_delay
        dep_time = minutes // 60 % 24 * 100 + minutes % 60
    rows.append(
        [2013, month, day, dep_time, carrier, 100 + 37 * i % 900, origin, dest,
         dep_delay, arr_delay, distance]
    )

with open("flights.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["year", "month", "day", "dep_time", "carrier", "flight",
                "origin", "dest", "dep_delay", "arr_delay", "distance"])
    w.writerows(rows)

pq.write_table(
    pa.table({
        "carrier": pa.array([c for c, _ in CARRIERS], pa.string()),
        "name": pa.array([n for _, n in CARRIERS], pa.string()),
    }),
    "carriers.parquet",
    compression="snappy",
)
