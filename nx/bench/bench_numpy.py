import numpy as np
import time
import statistics

# Measure in nanoseconds
def measure_time(f):
    t0 = time.perf_counter()
    f()
    t1 = time.perf_counter()
    return (t1 - t0) * 1_000_000_000

def run_benchmark(f, iterations):
    times = [measure_time(f) for _ in range(iterations)]
    return statistics.mean(times)

sizes = [50, 100, 500, 1000, 2000]
dtypes = [np.float32, np.float64]
iterations = 100

results = []
for size in sizes:
    for dtype in dtypes:
        a = np.random.rand(size, size).astype(dtype)
        b = np.random.rand(size, size).astype(dtype)

        ops = [
            ("Addition",       lambda: a + b),
            # ("Multiplication", lambda: a * b),
            # ("Subtraction",    lambda: a - b),
            # ("Division",       lambda: a / b),
            # ("Power",          lambda: np.power(a, b)),
            # ("Maximum",        lambda: np.maximum(a, b)),
            # ("Minimum",        lambda: np.minimum(a, b)),
        ]

        for name, func in ops:
            full_name = f"{name} on {size}x{size} {dtype.__name__}"
            mean_ns = run_benchmark(func, iterations)
            results.append((full_name, mean_ns))

# Find the fastest time to normalize against
fastest_ns = min(ns for _, ns in results)

def format_time(ns):
    return f"{ns:,.2f}ns".replace(",", "_")

formatted = [format_time(ns) for _, ns in results]
time_col_width = max(len(s) for s in formatted) + 2

header = f"┌─────────────────────────────────────┬{'─' * time_col_width}┬────────────┐"
divider = f"├─────────────────────────────────────┼{'─' * time_col_width}┼────────────┤"
footer = f"└─────────────────────────────────────┴{'─' * time_col_width}┴────────────┘"

print(header)
print(f"│ Name                                │ {'Time/Run (ns)':>{time_col_width-2}} │ Percentage │")
print(divider)

for name, ns in results:
    pct = (ns / fastest_ns) * 100
    tstr = format_time(ns)
    print(f"│ {name:<35} │ {tstr:>{time_col_width-2}} │ {pct:>9.2f}% │")

print(footer)
