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

sizes = [50, 100, 500]
dtypes = [np.float32, np.float64]
iterations = 3  # Reduced for consistency with OCaml benchmarks

results = []
for size in sizes:
    for dtype in dtypes:
        a = np.random.rand(size, size).astype(dtype)
        b = np.random.rand(size, size).astype(dtype)

        # Fixed order to match Nx benchmarks
        ops = [
            ("Addition",       lambda: a + b),
            ("Multiplication", lambda: a * b),
            ("Square",         lambda: np.square(a)),
        ]
        
        # Matrix operations - skip for large sizes
        if size <= 100:
            ops.append(("MatMul", lambda: np.matmul(a, b)))
        
        # Reduction operations
        ops.append(("Sum", lambda: np.sum(a)))
        
        # Float-specific operations
        if dtype in [np.float32, np.float64]:
            ops.extend([
                ("Sqrt",       lambda: np.sqrt(a)),
                ("Exp",        lambda: np.exp(a)),
            ])
        
        for name, func in ops:
            full_name = f"{name} on {size}x{size} {dtype.__name__}"
            mean_ns = run_benchmark(func, iterations)
            results.append((full_name, mean_ns))

# Sort results by time (fastest first)
results.sort(key=lambda x: x[1])

# Find the fastest time to normalize against
fastest_ns = min(ns for _, ns in results)

def format_time(ns):
    return f"{ns:,.2f}ns".replace(",", "_")

formatted = [format_time(ns) for _, ns in results]
time_col_width = max(len(s) for s in formatted) + 2

header = f"┌─────────────────────────────────────┬{'─' * time_col_width}┬────────────┐"
divider = f"├─────────────────────────────────────┼{'─' * time_col_width}┼────────────┤"
footer = f"└─────────────────────────────────────┴{'─' * time_col_width}┴────────────┘"

print("# NumPy Benchmarks")
print()
print(header)
print(f"│ Name                                │ {'Time/Run (ns)':>{time_col_width-2}} │ Percentage │")
print(divider)

for name, ns in results:
    pct = (ns / fastest_ns) * 100
    tstr = format_time(ns)
    print(f"│ {name:<35} │ {tstr:>{time_col_width-2}} │ {pct:>9.2f}% │")

print(footer)
