"""
Ubench - Micro-benchmarking library for Python.

This module mirrors the public surface of the OCaml `ubench` library,
providing comparable semantics while remaining idiomatic Python.
It focuses on high-resolution timing, light-weight statistical analysis, and
flexible output formats suitable for comparing backends such as Nx and NumPy.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

try:
    import resource
except ImportError:  # pragma: no cover - Windows fallback
    resource = None  # type: ignore

BenchmarkFn = Callable[[], Any]


# ---------------------------------------------------------------------------
# Core types


@dataclass(frozen=True)
class TimeLimit:
    seconds: float


@dataclass(frozen=True)
class IterationLimit:
    iterations: int


@dataclass(frozen=True)
class VarianceLimit:
    coefficient: float


Quota = Union[TimeLimit, IterationLimit, VarianceLimit]


class BenchmarkMode(str, Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"


@dataclass
class ProgressInfo:
    name: str
    current_measurement: int
    total_measurements: Optional[int]
    elapsed_time: float
    estimated_remaining: Optional[float]


class Predictor(str, Enum):
    ONE = "one"
    RUNS = "runs"
    TIME_NS = "time_ns"
    WALL_NS = "wall_ns"
    CYCLES = "cycles"
    USER_TIME = "user_time"
    SYSTEM_TIME = "system_time"
    CHILD_TIME = "child_time"
    MINOR_WORDS = "minor_words"
    MAJOR_WORDS = "major_words"
    PROMOTED_WORDS = "promoted_words"
    MINOR_COLLECTIONS = "minor_collections"
    MAJOR_COLLECTIONS = "major_collections"
    COMPACTIONS = "compactions"
    CUSTOM = "custom"


class Responder(str, Enum):
    TIME_PER_RUN = "time_per_run"
    WALL_PER_RUN = "wall_per_run"
    MEMORY_PER_RUN = "memory_per_run"
    TOTAL_TIME = "total_time"
    TOTAL_WALL = "total_wall"
    ALLOCATION_RATE = "allocation_rate"
    CUSTOM = "custom"


@dataclass
class Measurement:
    time_ns: float
    wall_ns: float
    utime_ns: float
    stime_ns: float
    cutime_ns: float
    cstime_ns: float
    cycles: float
    runs: int
    minor_words: float = 0.0
    major_words: float = 0.0
    promoted_words: float = 0.0
    minor_collections: int = 0
    major_collections: int = 0
    compactions: int = 0
    custom_predictors: Tuple[Tuple[str, float], ...] = field(default_factory=tuple)


@dataclass
class Statistics:
    avg: float
    min: float
    max: float
    std_dev: float
    ci95_lower: float
    ci95_upper: float


@dataclass
class RegressionResult:
    responder: Responder
    predictors: Tuple[Predictor, ...]
    coefficients: Tuple[float, ...]
    r_squared: float
    adjusted_r_squared: float
    intercept: Optional[float]
    confidence_intervals: Optional[Tuple[Tuple[float, float], ...]]


@dataclass
class BenchData:
    measurements: List[Measurement]
    time_stats: Statistics
    memory_stats: Statistics
    regressions: List[RegressionResult]
    total_time_ns: float
    total_runs: int


@dataclass
class AnalysisResult:
    name: str
    measurements: List[Measurement]
    time_stats: Statistics
    memory_stats: Statistics
    regressions: List[RegressionResult]
    total_time_ns: float
    total_runs: int


@dataclass(frozen=True)
class Config:
    mode: BenchmarkMode = BenchmarkMode.THROUGHPUT
    quota: Quota = field(default_factory=lambda: TimeLimit(1.0))
    warmup_iterations: int = 3
    min_measurements_required: int = 10
    stabilize_gc: bool = True
    geometric_scale_factor: float = 1.5
    fork_benchmarks: bool = False
    regressions_spec: Tuple[
        Tuple[Responder, Tuple[Predictor, ...], bool], ...
    ] = field(
        default_factory=lambda: (
            (Responder.TIME_PER_RUN, (Predictor.ONE, Predictor.RUNS), False),
            (Responder.MEMORY_PER_RUN, (Predictor.RUNS,), True),
        )
    )
    custom_measurer_fn: Optional[
        Callable[[Callable[[], None], int], Measurement]
    ] = None
    ascii_only_output: bool = False
    null_loop_subtraction: bool = True
    min_cpu_seconds: float = 0.4
    repeat: int = 1
    progress_callback_fn: Optional[Callable[[ProgressInfo], None]] = None

    @staticmethod
    def default() -> Config:
        return Config()

    def time_limit(self, seconds: float) -> Config:
        return replace(self, quota=TimeLimit(float(seconds)))

    def iteration_limit(self, iterations: int) -> Config:
        return replace(self, quota=IterationLimit(int(iterations)))

    def variance_limit(self, coefficient: float) -> Config:
        return replace(self, quota=VarianceLimit(float(coefficient)))

    def warmup(self, iterations: int) -> Config:
        return replace(self, warmup_iterations=int(iterations))

    def min_measurements(self, count: int) -> Config:
        return replace(self, min_measurements_required=int(count))

    def gc_stabilization(self, enabled: bool) -> Config:
        return replace(self, stabilize_gc=bool(enabled))

    def fork(self, enabled: bool) -> Config:
        return replace(self, fork_benchmarks=bool(enabled))

    def ascii_only(self, enabled: bool) -> Config:
        return replace(self, ascii_only_output=bool(enabled))

    def geometric_scale(self, factor: float) -> Config:
        if factor <= 1.0:
            raise ValueError("geometric_scale must be > 1.0")
        return replace(self, geometric_scale_factor=float(factor))

    def regressions(
        self, entries: Sequence[Tuple[Responder, Sequence[Predictor], bool]]
    ) -> Config:
        normalized = tuple(
            (resp, tuple(preds), bool(include_intercept))
            for resp, preds, include_intercept in entries
        )
        return replace(self, regressions_spec=normalized)

    def custom_measurer(
        self, measurer: Optional[Callable[[Callable[[], None], int], Measurement]]
    ) -> Config:
        return replace(self, custom_measurer_fn=measurer)

    def progress_callback(
        self, callback: Optional[Callable[[ProgressInfo], None]]
    ) -> Config:
        return replace(self, progress_callback_fn=callback)

    def null_loop(self, enabled: bool) -> Config:
        return replace(self, null_loop_subtraction=bool(enabled))

    def min_cpu(self, seconds: float) -> Config:
        return replace(self, min_cpu_seconds=float(seconds))

    def repeat_runs(self, count: int) -> Config:
        return replace(self, repeat=max(1, int(count)))

    def build(self) -> Config:
        return self


default_config = Config.default()


# ---------------------------------------------------------------------------
# Statistics utilities


def _mean(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _std_dev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def random_state() -> "random.Random":
    import random

    if not hasattr(random_state, "_rng"):
        random_state._rng = random.Random()
        random_state._rng.seed(int(time.time() * 1e9) ^ os.getpid())
    return random_state._rng  # type: ignore[attr-defined]


def _bootstrap_interval(
    values: List[float], confidence: float = 0.95
) -> Tuple[float, float]:
    if len(values) < 3:
        mean_val = _mean(values)
        return (mean_val, mean_val)

    rng = random_state()
    n = len(values)
    resamples = max(1000, 10 * n)
    stats = []
    for _ in range(resamples):
        sample = [values[rng.randrange(0, n)] for _ in range(n)]
        stats.append(_mean(sample))
    stats.sort()
    alpha = 1.0 - confidence
    lower_idx = int(resamples * (alpha / 2.0))
    upper_idx = min(resamples - 1, int(resamples * (1.0 - alpha / 2.0)))
    return (stats[lower_idx], stats[upper_idx])


def _confidence_interval(values: List[float]) -> Tuple[float, float]:
    if len(values) >= 20:
        return _bootstrap_interval(values)
    if len(values) < 3:
        mean_val = _mean(values)
        return (mean_val, mean_val)
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    lower_idx = max(0, n * 25 // 1000)
    upper_idx = min(n - 1, n * 975 // 1000)
    return (sorted_vals[lower_idx], sorted_vals[upper_idx])


def compute_statistics(values: List[float]) -> Statistics:
    if not values:
        return Statistics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    avg = _mean(values)
    std_dev = _std_dev(values)
    ci_lower, ci_upper = _confidence_interval(values)
    return Statistics(
        avg=avg,
        min=min(values),
        max=max(values),
        std_dev=std_dev,
        ci95_lower=ci_lower,
        ci95_upper=ci_upper,
    )


# ---------------------------------------------------------------------------
# Math helpers for statistical tests


def log_gamma(x: float) -> float:
    return math.lgamma(x)


def _max_tiny(x: float) -> float:
    return max(1e-30, x)


_BETAI_CF_EPS = sys.float_info.epsilon


def _betai_cf(x: float, a: float, b: float) -> float:
    apb = a + b
    ap1 = a + 1.0
    am1 = a - 1.0
    # Initialize Lentz's method
    d = 1.0 / _max_tiny(1.0 - (apb * x / ap1))
    c = 1.0
    f = d
    m = 1.0
    while True:
        m2 = 2.0 * m
        cf_d2m = m * (b - m) * x / ((am1 + m2) * (a + m2))
        d = 1.0 / _max_tiny(1.0 + (cf_d2m * d))
        c = _max_tiny(1.0 + (cf_d2m / c))
        f *= d * c

        cf_d2m1 = -(a + m) * (apb + m) * x / ((a + m2) * (ap1 + m2))
        d = 1.0 / _max_tiny(1.0 + (cf_d2m1 * d))
        c = _max_tiny(1.0 + (cf_d2m1 / c))
        delta = c * d
        f *= delta
        if abs(delta - 1.0) < _BETAI_CF_EPS:
            break
        m += 1.0
    return f


def betai(x: float, a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        raise ValueError("betai: a and b must be positive")
    if x < 0.0 or x > 1.0:
        raise ValueError("betai: x must be in [0, 1]")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    m = math.exp(
        log_gamma(a + b)
        - log_gamma(a)
        - log_gamma(b)
        + (a * math.log(x))
        + (b * math.log(1.0 - x))
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return m * _betai_cf(x, a, b) / a
    return 1.0 - (m * _betai_cf(1.0 - x, b, a) / b)


def cpl_student_t(t: float, nu: float) -> float:
    return betai(nu / (nu + (t * t)), 0.5 * nu, 0.5)


def different_rates(
    significance: float,
    n1: int,
    r1: float,
    var1: float,
    n2: int,
    r2: float,
    var2: float,
) -> bool:
    if n1 <= 0 or n2 <= 0:
        return False
    if n1 == 1 and n2 == 1:
        return True

    df = float(n1 + n2 - 2)
    n1f = float(n1)
    n2f = float(n2)
    pooled = (var1 + var2) / df
    if pooled <= 0.0:
        return False
    se = math.sqrt(pooled * ((1.0 / n1f) + (1.0 / n2f)))
    if se == 0.0:
        return False
    t_val = abs(r1 - r2) / se
    return cpl_student_t(t_val, df) <= significance


# ---------------------------------------------------------------------------
# Formatting helpers


def format_time_ns(ns: float) -> str:
    if ns < 0.0:
        return f"-{format_time_ns(-ns)}"
    units = [
        (1e9, "s"),
        (1e6, "ms"),
        (1e3, "µs"),
        (1.0, "ns"),
    ]
    for scale, suffix in units:
        if ns >= scale:
            value = ns / scale
            return f"{value:,.2f}{suffix}".replace(",", "_")
    return f"{ns:.2f}ns"


def format_words(words: float) -> str:
    units = [
        (1e9, "Gw"),
        (1e6, "Mw"),
        (1e3, "kw"),
        (1.0, "w"),
    ]
    value = abs(words)
    for scale, suffix in units:
        if value >= scale:
            res = value / scale
            return f"{res:,.2f}{suffix}".replace(",", "_")
    return f"{value:.2f}w"


def format_number(value: float) -> str:
    units = [
        (1e9, "G"),
        (1e6, "M"),
        (1e3, "k"),
    ]
    abs_value = abs(value)
    for scale, suffix in units:
        if abs_value >= scale:
            res = value / scale
            return f"{res:,.2f}{suffix}".replace(",", "_")
    return f"{value:.2f}"


# ---------------------------------------------------------------------------
# Measurement primitives


def _collect_times() -> Tuple[int, int, os.times_result]:
    return time.perf_counter_ns(), time.process_time_ns(), os.times()


def _to_measurement(
    before: Tuple[int, int, os.times_result],
    after: Tuple[int, int, os.times_result],
    runs: int,
) -> Measurement:
    wall_start, cpu_start, tms_start = before
    wall_end, cpu_end, tms_end = after
    utime = (tms_end.user - tms_start.user) * 1e9
    stime = (tms_end.system - tms_start.system) * 1e9
    cutime = (tms_end.children_user - tms_start.children_user) * 1e9
    cstime = (tms_end.children_system - tms_start.children_system) * 1e9
    wall_ns = float(wall_end - wall_start)
    time_ns = float(cpu_end - cpu_start)
    estimated_cycles = (wall_ns / 1e9) * 3e9
    measurement = Measurement(
        time_ns=time_ns,
        wall_ns=wall_ns,
        utime_ns=utime,
        stime_ns=stime,
        cutime_ns=cutime,
        cstime_ns=cstime,
        cycles=estimated_cycles,
        runs=runs,
    )
    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        measurement.minor_words = getattr(usage, "ru_minflt", 0.0)
        measurement.major_words = getattr(usage, "ru_majflt", 0.0)
    return measurement


def _subtract_measurements(target: Measurement, baseline: Measurement) -> Measurement:
    for attr in [
        "time_ns",
        "wall_ns",
        "utime_ns",
        "stime_ns",
        "cutime_ns",
        "cstime_ns",
        "cycles",
    ]:
        value = getattr(target, attr) - getattr(baseline, attr)
        setattr(target, attr, max(0.0, value))
    return target


def _measure_callable(func: Callable[[], Any], runs: int) -> Measurement:
    before = _collect_times()
    for _ in range(runs):
        func()
    after = _collect_times()
    return _to_measurement(before, after, runs)


def _measure_null_loop(runs: int) -> Measurement:
    return _measure_callable(lambda: None, runs)


def _measure_one_batch(
    func: Callable[[], Any],
    batch_size: int,
    *,
    null_loop_subtraction: bool,
) -> Measurement:
    for _ in range(3):
        func()
    measurement = _measure_callable(func, batch_size)
    if null_loop_subtraction and batch_size > 0:
        baseline = _measure_null_loop(batch_size)
        measurement = _subtract_measurements(measurement, baseline)
    return measurement


def stabilize_gc() -> None:
    gc.collect()


# ---------------------------------------------------------------------------
# Regression analysis


def _predictor_value(measurement: Measurement, predictor: Predictor) -> float:
    if predictor == Predictor.ONE:
        return 1.0
    if predictor == Predictor.RUNS:
        return float(measurement.runs)
    if predictor == Predictor.TIME_NS:
        return measurement.time_ns
    if predictor == Predictor.WALL_NS:
        return measurement.wall_ns
    if predictor == Predictor.CYCLES:
        return measurement.cycles
    if predictor == Predictor.USER_TIME:
        return measurement.utime_ns
    if predictor == Predictor.SYSTEM_TIME:
        return measurement.stime_ns
    if predictor == Predictor.CHILD_TIME:
        return measurement.cutime_ns + measurement.cstime_ns
    if predictor == Predictor.MINOR_WORDS:
        return measurement.minor_words
    if predictor == Predictor.MAJOR_WORDS:
        return measurement.major_words
    if predictor == Predictor.PROMOTED_WORDS:
        return measurement.promoted_words
    if predictor == Predictor.MINOR_COLLECTIONS:
        return float(measurement.minor_collections)
    if predictor == Predictor.MAJOR_COLLECTIONS:
        return float(measurement.major_collections)
    if predictor == Predictor.COMPACTIONS:
        return float(measurement.compactions)
    if predictor == Predictor.CUSTOM:
        # Fallback to first custom predictor, mirroring OCaml semantics.
        return measurement.custom_predictors[0][1] if measurement.custom_predictors else 0.0
    raise ValueError(f"Unsupported predictor: {predictor}")


def _responder_value(measurement: Measurement, responder: Responder) -> float:
    if responder == Responder.TIME_PER_RUN:
        return measurement.time_ns / max(1, measurement.runs)
    if responder == Responder.WALL_PER_RUN:
        return measurement.wall_ns / max(1, measurement.runs)
    if responder == Responder.MEMORY_PER_RUN:
        return measurement.minor_words / max(1, measurement.runs)
    if responder == Responder.TOTAL_TIME:
        return measurement.time_ns
    if responder == Responder.TOTAL_WALL:
        return measurement.wall_ns
    if responder == Responder.ALLOCATION_RATE:
        seconds = measurement.time_ns / 1e9
        return measurement.minor_words / seconds if seconds > 0 else 0.0
    if responder == Responder.CUSTOM:
        return measurement.custom_predictors[0][1] if measurement.custom_predictors else 0.0
    raise ValueError(f"Unsupported responder: {responder}")


def _transpose(matrix: List[List[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*matrix)]


def _matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    result = [[0.0 for _ in range(len(b[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for k in range(len(b)):
            aik = a[i][k]
            if aik == 0.0:
                continue
            for j in range(len(b[0])):
                result[i][j] += aik * b[k][j]
    return result


def _matvec_mul(matrix: List[List[float]], vector: List[float]) -> List[float]:
    return [sum(row[j] * vector[j] for j in range(len(vector))) for row in matrix]


def _solve_normal_equations(xtx: List[List[float]], xty: List[float]) -> List[float]:
    n = len(xtx)
    augmented = [row[:] + [value] for row, value in zip(xtx, xty)]

    for i in range(n):
        pivot = augmented[i][i]
        if abs(pivot) < 1e-12:
            for j in range(i + 1, n):
                if abs(augmented[j][i]) > abs(pivot):
                    augmented[i], augmented[j] = augmented[j], augmented[i]
                    pivot = augmented[i][i]
                    break
        if abs(pivot) < 1e-12:
            raise ValueError("Matrix is singular")
        pivot_inv = 1.0 / pivot
        for j in range(i, n + 1):
            augmented[i][j] *= pivot_inv
        for k in range(n):
            if k == i:
                continue
            factor = augmented[k][i]
            if factor == 0.0:
                continue
            for j in range(i, n + 1):
                augmented[k][j] -= factor * augmented[i][j]
    return [augmented[i][n] for i in range(n)]


def ordinary_least_squares(
    measurements: Sequence[Measurement],
    responder: Responder,
    predictors: Sequence[Predictor],
    include_intercept: bool,
) -> RegressionResult:
    if not measurements:
        return RegressionResult(
            responder=responder,
            predictors=tuple(predictors),
            coefficients=tuple(),
            r_squared=0.0,
            adjusted_r_squared=0.0,
            intercept=None,
            confidence_intervals=None,
        )

    y = [_responder_value(m, responder) for m in measurements]
    x_rows = []
    for m in measurements:
        row = [_predictor_value(m, p) for p in predictors]
        if include_intercept:
            row.insert(0, 1.0)
        x_rows.append(row)

    xt = _transpose(x_rows)
    xtx = _matmul(xt, x_rows)
    xty = _matvec_mul(xt, y)
    try:
        coeffs = _solve_normal_equations(xtx, xty)
    except ValueError:
        coeffs = [0.0 for _ in range(len(predictors) + (1 if include_intercept else 0))]

    predictions = [_matvec_mul([row], coeffs)[0] for row in x_rows]
    mean_y = _mean(y)
    ss_tot = sum((val - mean_y) ** 2 for val in y)
    ss_res = sum((y_i - y_hat) ** 2 for y_i, y_hat in zip(y, predictions))
    r_squared = 0.0 if ss_tot == 0 else max(0.0, 1.0 - (ss_res / ss_tot))
    n = len(measurements)
    p = len(coeffs)
    adjusted_r_squared = (
        1.0 - ((1.0 - r_squared) * (n - 1) / (n - p - 1)) if n > p + 1 else r_squared
    )

    intercept = coeffs[0] if include_intercept else None
    coeff_tuple = tuple(coeffs[1:]) if include_intercept else tuple(coeffs)

    return RegressionResult(
        responder=responder,
        predictors=tuple(predictors),
        coefficients=coeff_tuple,
        r_squared=r_squared,
        adjusted_r_squared=adjusted_r_squared,
        intercept=intercept,
        confidence_intervals=None,
    )


# ---------------------------------------------------------------------------
# Benchmark definitions


@dataclass
class _Benchmark:
    name: str
    fn: BenchmarkFn


@dataclass
class _BenchmarkGroup:
    name: str
    benchmarks: Sequence[Union["_Benchmark", "_BenchmarkGroup"]]


Benchmark = Union[_Benchmark, _BenchmarkGroup]


def bench(name: str, fn: Callable[[], Any]) -> _Benchmark:
    return _Benchmark(name=name, fn=lambda: fn())


def create(name: str, fn: Callable[[], Any]) -> _Benchmark:
    return bench(name, fn)


def group(name: str, benchmarks: Sequence[Benchmark]) -> _BenchmarkGroup:
    return _BenchmarkGroup(name=name, benchmarks=list(benchmarks))


def create_group(name: str, benchmarks: Sequence[Benchmark]) -> _BenchmarkGroup:
    return group(name, benchmarks)


def bench_with_setup(
    name: str,
    *,
    setup: Callable[[], Any],
    teardown: Callable[[Any], None],
    f: Callable[[Any], Any],
) -> _Benchmark:
    def wrapped() -> None:
        resource = setup()
        try:
            f(resource)
        finally:
            teardown(resource)

    return _Benchmark(name=name, fn=wrapped)


def create_with_setup(
    name: str,
    *,
    setup: Callable[[], Any],
    teardown: Callable[[Any], None],
    f: Callable[[Any], Any],
) -> _Benchmark:
    return bench_with_setup(name, setup=setup, teardown=teardown, f=f)


def bench_param(
    base_name: str,
    func: Callable[..., Any],
    *,
    params: Sequence[Tuple[str, Any]],
) -> List[_Benchmark]:
    benchmarks = []
    for label, value in params:
        name = f"{base_name}[{label}]"

        def wrapped(
            fn: Callable[..., Any] = func, param=value
        ) -> None:  # default values capture loop vars
            fn(param=param)

        benchmarks.append(_Benchmark(name=name, fn=wrapped))
    return benchmarks


def create_param(
    base_name: str,
    func: Callable[..., Any],
    *,
    params: Sequence[Tuple[str, Any]],
) -> List[_Benchmark]:
    return bench_param(base_name, func, params=params)


def _flatten(benchmark: Benchmark, prefix: str = "") -> List[_Benchmark]:
    if isinstance(benchmark, _Benchmark):
        full_name = benchmark.name if not prefix else f"{prefix}/{benchmark.name}"
        return [_Benchmark(name=full_name, fn=benchmark.fn)]
    new_prefix = benchmark.name if not prefix else f"{prefix}/{benchmark.name}"
    flattened: List[_Benchmark] = []
    for child in benchmark.benchmarks:
        flattened.extend(_flatten(child, new_prefix))
    return flattened


def flatten_benchmarks(benchmarks: Sequence[Benchmark]) -> List[_Benchmark]:
    flattened: List[_Benchmark] = []
    for benchmark in benchmarks:
        flattened.extend(_flatten(benchmark))
    return flattened


# ---------------------------------------------------------------------------
# Execution engine


def run_bench_with_config(config: Config, fn: Callable[[], None]) -> BenchData:
    if config.geometric_scale_factor <= 1.0:
        raise ValueError("geometric_scale must be > 1.0")

    measurements: List[Measurement] = []
    total_time_ns = 0.0
    total_runs = 0
    measurement_count = 0
    batch_size = 1
    start_cpu = time.process_time()
    samples: List[float] = []

    for _ in range(config.warmup_iterations):
        fn()

    if config.custom_measurer_fn is not None:
        measure_batch = lambda runs: config.custom_measurer_fn(fn, runs)
    else:
        measure_batch = lambda runs: _measure_one_batch(
            fn, runs, null_loop_subtraction=config.null_loop_subtraction
        )

    def should_continue(elapsed_cpu: float) -> bool:
        min_met = measurement_count >= config.min_measurements_required
        quota = config.quota
        if isinstance(quota, TimeLimit):
            return not min_met or elapsed_cpu < quota.seconds
        if isinstance(quota, IterationLimit):
            return total_runs < quota.iterations
        if isinstance(quota, VarianceLimit):
            if measurement_count < config.min_measurements_required:
                return True
            mean_val = _mean(samples)
            if mean_val == 0.0:
                return False
            std_val = _std_dev(samples)
            return (std_val / mean_val) > quota.coefficient
        return False

    while True:
        if config.stabilize_gc:
            stabilize_gc()
        measurement = measure_batch(batch_size)
        if measurement.time_ns / 1e9 < config.min_cpu_seconds:
            batch_size = max(batch_size + 1, int(batch_size * config.geometric_scale_factor))
            continue
        measurement_count += 1
        measurement_per_run = (
            measurement.time_ns / max(1, measurement.runs) if measurement.runs else 0.0
        )
        measurements.append(measurement)
        samples.append(measurement_per_run)
        total_time_ns += measurement.time_ns
        total_runs += measurement.runs

        elapsed_cpu = time.process_time() - start_cpu
        if not should_continue(elapsed_cpu):
            break

        next_batch = max(
            batch_size + 1, int(math.ceil(batch_size * config.geometric_scale_factor))
        )
        if isinstance(config.quota, IterationLimit):
            remaining = config.quota.iterations - total_runs
            next_batch = max(1, min(next_batch, remaining))
        batch_size = next_batch if next_batch > 0 else 1

    time_values = [m.time_ns / max(1, m.runs) for m in measurements]
    memory_values = [m.minor_words / max(1, m.runs) for m in measurements]

    regressions = [
        ordinary_least_squares(
            measurements,
            responder=resp,
            predictors=preds,
            include_intercept=include_intercept,
        )
        for resp, preds, include_intercept in config.regressions_spec
    ]

    return BenchData(
        measurements=measurements,
        time_stats=compute_statistics(time_values),
        memory_stats=compute_statistics(memory_values),
        regressions=regressions,
        total_time_ns=total_time_ns,
        total_runs=total_runs,
    )


def run_silent(
    benchmarks: Sequence[Benchmark],
    *,
    config: Config = default_config,
) -> List[AnalysisResult]:
    flattened = flatten_benchmarks(benchmarks)
    results: List[AnalysisResult] = []
    start_wall = time.perf_counter()

    for index, bench_impl in enumerate(flattened, start=1):
        print(f"[{index}/{len(flattened)}] Running {bench_impl.name}...", end="", flush=True)
        bench_data = run_bench_with_config(config, bench_impl.fn)
        print(" done.")

        if config.progress_callback_fn is not None:
            elapsed = time.perf_counter() - start_wall
            info = ProgressInfo(
                name=bench_impl.name,
                current_measurement=index,
                total_measurements=len(flattened),
                elapsed_time=elapsed,
                estimated_remaining=None,
            )
            config.progress_callback_fn(info)

        results.append(
            AnalysisResult(
                name=bench_impl.name,
                measurements=bench_data.measurements,
                time_stats=bench_data.time_stats,
                memory_stats=bench_data.memory_stats,
                regressions=bench_data.regressions,
                total_time_ns=bench_data.total_time_ns,
                total_runs=bench_data.total_runs,
            )
        )

    return results


def run_and_print(
    benchmarks: Sequence[Benchmark],
    *,
    config: Config = default_config,
    output_format: str = "pretty",
    verbose: bool = False,
) -> List[AnalysisResult]:
    results = run_silent(benchmarks, config=config)
    print("\nBenchmark Results:")
    fmt = output_format.lower()
    if fmt in {"pretty", "table"}:
        print_pretty_table(results, ascii_only=config.ascii_only_output)
    elif fmt == "json":
        print_json(results)
    elif fmt == "csv":
        print_csv(results)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    if verbose:
        print_regression_analysis(results)
    return results


def run(
    benchmarks: Sequence[Benchmark],
    *,
    config: Config = default_config,
    output_format: str = "pretty",
    verbose: bool = False,
) -> List[AnalysisResult]:
    return run_and_print(
        benchmarks, config=config, output_format=output_format, verbose=verbose
    )


# ---------------------------------------------------------------------------
# Output helpers


def print_pretty_table(
    results: Sequence[AnalysisResult],
    *,
    ascii_only: bool = False,
) -> None:
    if not results:
        print("No benchmark results to display.")
        return

    reset = "\x1b[0m"
    bold = "\x1b[1m"
    green = "\x1b[32m"
    cyan = "\x1b[36m"

    def colorize(code: str, text: str) -> str:
        return f"{code}{text}{reset}"

    def strip_ansi_codes(text: str) -> str:
        result_chars: List[str] = []
        i = 0
        while i < len(text):
            if text[i] == "\x1b" and i + 1 < len(text) and text[i + 1] == "[":
                end = text.find("m", i + 2)
                if end == -1:
                    result_chars.append(text[i])
                    i += 1
                else:
                    i = end + 1
            else:
                result_chars.append(text[i])
                i += 1
        return "".join(result_chars)

    def visual_width(text: str) -> int:
        stripped = strip_ansi_codes(text)
        return sum(1 for _ in stripped)

    def pad_left(text: str, width: int) -> str:
        length = visual_width(text)
        return text if length >= width else " " * (width - length) + text

    def pad_right(text: str, width: int) -> str:
        length = visual_width(text)
        return text if length >= width else text + " " * (width - length)

    fastest_time = min(r.time_stats.avg for r in results)
    lowest_memory = min(r.memory_stats.avg for r in results)

    sorted_results = sorted(results, key=lambda r: r.time_stats.avg)

    rows_data: List[Tuple[AnalysisResult, List[str]]] = []
    for entry in sorted_results:
        time_str = format_time_ns(entry.time_stats.avg)
        mem_str = format_words(entry.memory_stats.avg)
        speedup = (
            fastest_time / entry.time_stats.avg if entry.time_stats.avg > 0.0 else float("inf")
        )
        vs_fastest = (
            entry.time_stats.avg / fastest_time if fastest_time > 0.0 else float("inf")
        )
        row = [
            entry.name,
            time_str,
            mem_str,
            f"{speedup:.2f}x",
            f"{vs_fastest * 100.0:.0f}%",
        ]
        if entry.time_stats.avg == fastest_time:
            row[1] = colorize(green, row[1])
        if entry.memory_stats.avg == lowest_memory:
            row[2] = colorize(cyan, row[2])
        if speedup >= 1.0:
            row[3] = colorize(green, row[3])
        if math.isclose(vs_fastest, 1.0):
            row[4] = colorize(green, row[4])
        rows_data.append((entry, row))

    headers = ["Name", "Time/Run", "mWd/Run", "Speedup", "vs Fastest"]
    widths = [visual_width(h) for h in headers]
    for _, row in rows_data:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], visual_width(value))

    if ascii_only:
        top_left, top_mid, top_right = "+", "+", "+"
        mid_left, mid_mid, mid_right = "+", "+", "+"
        bot_left, bot_mid, bot_right = "+", "+", "+"
        hline = "-"
        vline = "|"
    else:
        top_left, top_mid, top_right = "┌", "┬", "┐"
        mid_left, mid_mid, mid_right = "├", "┼", "┤"
        bot_left, bot_mid, bot_right = "└", "┴", "┘"
        hline = "─"
        vline = "│"

    def repeat_str(char: str, count: int) -> str:
        return char * count

    def make_border(left: str, mid: str, right: str) -> str:
        segments = [repeat_str(hline, width + 2) for width in widths]
        joined = f"{mid}".join(segments)
        return f"{left}{joined}{right}"

    top_border = make_border(top_left, top_mid, top_right)
    separator = make_border(mid_left, mid_mid, mid_right)
    bottom_border = make_border(bot_left, bot_mid, bot_right)

    print(top_border)
    header_row = []
    for index, header in enumerate(headers):
        padded = pad_right(header, widths[index]) if index == 0 else pad_left(header, widths[index])
        header_row.append(colorize(bold, padded))
    header_str = f" {vline} ".join(header_row)
    print(f"{vline} {header_str} {vline}")
    print(separator)

    for _, row in rows_data:
        padded_row = []
        for index, value in enumerate(row):
            padded = pad_right(value, widths[index]) if index == 0 else pad_left(value, widths[index])
            padded_row.append(padded)
        row_str = f" {vline} ".join(padded_row)
        print(f"{vline} {row_str} {vline}")
    print(bottom_border)


def print_json(results: Sequence[AnalysisResult]) -> None:
    payload = []
    for result in results:
        payload.append(
            {
                "name": result.name,
                "time_stats": dataclasses.asdict(result.time_stats),
                "memory_stats": dataclasses.asdict(result.memory_stats),
                "total_time_ns": result.total_time_ns,
                "total_runs": result.total_runs,
                "regressions": [
                    {
                        "responder": reg.responder.value,
                        "predictors": [pred.value for pred in reg.predictors],
                        "coefficients": list(reg.coefficients),
                        "r_squared": reg.r_squared,
                        "adjusted_r_squared": reg.adjusted_r_squared,
                        "intercept": reg.intercept,
                    }
                    for reg in result.regressions
                ],
            }
        )
    print(json.dumps(payload, indent=2))


def print_csv(results: Sequence[AnalysisResult]) -> None:
    headers = [
        "name",
        "time_avg",
        "time_min",
        "time_max",
        "time_std_dev",
        "time_ci95_lower",
        "time_ci95_upper",
        "memory_avg",
        "memory_min",
        "memory_max",
        "memory_std_dev",
        "memory_ci95_lower",
        "memory_ci95_upper",
        "total_runs",
        "time_r_squared",
        "time_adjusted_r_squared",
    ]
    print(",".join(headers))
    for result in results:
        time_reg = next(
            (reg for reg in result.regressions if reg.responder == Responder.TIME_PER_RUN),
            RegressionResult(
                responder=Responder.TIME_PER_RUN,
                predictors=tuple(),
                coefficients=tuple(),
                r_squared=0.0,
                adjusted_r_squared=0.0,
                intercept=None,
                confidence_intervals=None,
            ),
        )
        row = [
            result.name,
            f"{result.time_stats.avg:.2f}",
            f"{result.time_stats.min:.2f}",
            f"{result.time_stats.max:.2f}",
            f"{result.time_stats.std_dev:.2f}",
            f"{result.time_stats.ci95_lower:.2f}",
            f"{result.time_stats.ci95_upper:.2f}",
            f"{result.memory_stats.avg:.2f}",
            f"{result.memory_stats.min:.2f}",
            f"{result.memory_stats.max:.2f}",
            f"{result.memory_stats.std_dev:.2f}",
            f"{result.memory_stats.ci95_lower:.2f}",
            f"{result.memory_stats.ci95_upper:.2f}",
            f"{result.total_runs}",
            f"{time_reg.r_squared:.4f}",
            f"{time_reg.adjusted_r_squared:.4f}",
        ]
        print(",".join(row))


def print_regression_analysis(results: Sequence[AnalysisResult]) -> None:
    if not results:
        return
    print("\nRegression analysis:")
    for result in results:
        print(f"\n{result.name}:")
        for reg in result.regressions:
            predictor_str = ", ".join(pred.value for pred in reg.predictors)
            coeffs = ", ".join(f"{coef:.4g}" for coef in reg.coefficients)
            intercept = f"{reg.intercept:.4g}" if reg.intercept is not None else "None"
            print(
                f"  {reg.responder.value}: intercept={intercept}; "
                f"predictors=({predictor_str}); coeffs=[{coeffs}]; "
                f"R²={reg.r_squared:.4f}; adj.R²={reg.adjusted_r_squared:.4f}"
            )


# ---------------------------------------------------------------------------
# Comparison utilities


@dataclass
class ComparisonResult:
    baseline: AnalysisResult
    compared: AnalysisResult
    speedup: float
    speedup_ci_lower: float
    speedup_ci_upper: float
    significant: bool
    p_value: Optional[float]


def compare(
    baseline: AnalysisResult,
    compared: AnalysisResult,
    *,
    confidence: float = 0.95,
) -> ComparisonResult:
    n1 = len(baseline.measurements)
    n2 = len(compared.measurements)
    if n1 == 0 or n2 == 0:
        nan = float("nan")
        return ComparisonResult(
            baseline=baseline,
            compared=compared,
            speedup=nan,
            speedup_ci_lower=nan,
            speedup_ci_upper=nan,
            significant=False,
            p_value=None,
        )

    rates1 = [
        float(m.runs) / (m.time_ns / 1e9) if m.time_ns > 0 else 0.0
        for m in baseline.measurements
    ]
    rates2 = [
        float(m.runs) / (m.time_ns / 1e9) if m.time_ns > 0 else 0.0
        for m in compared.measurements
    ]

    avg1 = _mean(rates1)
    avg2 = _mean(rates2)
    var1 = statistics.variance(rates1) if len(rates1) > 1 else 0.0
    var2 = statistics.variance(rates2) if len(rates2) > 1 else 0.0
    significance = 1.0 - confidence
    significant = different_rates(significance, n1, avg1, var1, n2, avg2, var2)
    speedup = avg2 / avg1 if avg1 else float("inf")

    if n1 > 1 and n2 > 1:
        se1 = math.sqrt(var1 / n1)
        se2 = math.sqrt(var2 / n2)
        z = 1.96  # Approximate 95% CI
        lower = (avg2 - (z * se2)) / (avg1 + (z * se1)) if (avg1 + z * se1) else speedup
        upper = (avg2 + (z * se2)) / (avg1 - (z * se1)) if (avg1 - z * se1) else speedup
    else:
        lower = upper = speedup

    return ComparisonResult(
        baseline=baseline,
        compared=compared,
        speedup=speedup,
        speedup_ci_lower=lower,
        speedup_ci_upper=upper,
        significant=significant,
        p_value=None,
    )


def print_comparison(comparison: ComparisonResult) -> None:
    print("\n=== Benchmark Comparison ===")
    print(f"Baseline: {comparison.baseline.name}")
    print(f"Compared: {comparison.compared.name}")
    if math.isnan(comparison.speedup):
        print("Insufficient data for comparison.")
        return
    adjective = "faster" if comparison.speedup > 1.0 else "slower"
    print(
        f"{comparison.compared.name} is {abs(comparison.speedup):.2f}x {adjective} "
        f"than {comparison.baseline.name}"
    )
    print(
        f"{comparison.baseline.name}: {format_time_ns(comparison.baseline.time_stats.avg)} "
        f"(±{comparison.baseline.time_stats.std_dev / max(1e-9, comparison.baseline.time_stats.avg) * 100.0:.2f}%)"
    )
    print(
        f"{comparison.compared.name}: {format_time_ns(comparison.compared.time_stats.avg)} "
        f"(±{comparison.compared.time_stats.std_dev / max(1e-9, comparison.compared.time_stats.avg) * 100.0:.2f}%)"
    )
    print(
        f"95% CI on speedup: [{comparison.speedup_ci_lower:.2f}x, "
        f"{comparison.speedup_ci_upper:.2f}x]"
    )
    print(
        "Difference is statistically significant."
        if comparison.significant
        else "Difference is not statistically significant."
    )


def tabulate(
    results: Sequence[AnalysisResult],
    *,
    confidence: float = 0.95,
    cpu_selector: str = "process",
) -> None:
    if not results:
        print("(no benchmarks)")
        return
    selector = cpu_selector.lower()

    def cpu_time(m: Measurement) -> float:
        if selector == "process":
            return m.time_ns
        if selector == "user":
            return m.utime_ns
        if selector == "system":
            return m.stime_ns
        if selector == "children":
            return m.cutime_ns + m.cstime_ns
        if selector == "all":
            return m.time_ns + m.cutime_ns + m.cstime_ns
        raise ValueError(f"Unsupported cpu_selector: {cpu_selector}")

    entries: List[Tuple[str, int, float, float]] = []
    for result in results:
        n = len(result.measurements)
        if n == 0:
            entries.append((result.name, 0, float("nan"), 0.0))
            continue
        rates = [
            float(meas.runs) / (cpu_time(meas) / 1e9) if cpu_time(meas) > 0 else 0.0
            for meas in result.measurements
        ]
        avg = _mean(rates)
        var = statistics.variance(rates) if len(rates) > 1 else 0.0
        entries.append((result.name, n, avg, var))

    entries.sort(key=lambda item: item[2], reverse=True)
    header = f"{'Benchmark':<30} {'Rate (runs/s)':>16} {'Vs fastest':>12}"
    print(header)
    print("-" * len(header))
    for name, count, rate, _ in entries:
        if count == 0 or math.isnan(rate):
            print(f"{name:<30} {'N/A':>16} {'N/A':>12}")
            continue
        pct = rate / entries[0][2] * 100.0 if entries[0][2] else 0.0
        print(f"{name:<30} {rate:>16.2f} {pct:>11.0f}%")

    print("\nPairwise comparison:")
    significance = 1.0 - confidence
    for row_name, row_n, row_rate, row_var in entries:
        print(f"{row_name:<30}", end=" ")
        for col_name, col_n, col_rate, col_var in entries:
            if row_name == col_name:
                cell = "--"
            elif row_n == 0 or col_n == 0:
                cell = "N/A"
            else:
                diff = (row_rate / col_rate - 1.0) * 100.0 if col_rate else 0.0
                sig = different_rates(significance, row_n, row_rate, row_var, col_n, col_rate, col_var)
                cell = f"{diff:>7.0f}%" if sig else f"[{diff:>5.0f}%]"
            print(f"{cell:>10}", end=" ")
        print()


# ---------------------------------------------------------------------------
# CLI


def parse_quota(text: str) -> Quota:
    text = text.strip()
    if text.endswith("s"):
        return TimeLimit(float(text[:-1]))
    if text.endswith("x"):
        return IterationLimit(int(text[:-1]))
    if text.endswith("%"):
        return VarianceLimit(float(text[:-1]) / 100.0)
    try:
        return TimeLimit(float(text))
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid quota: {text}") from exc


def parse_output(text: str) -> str:
    text = text.lower()
    if text in {"pretty", "table"}:
        return "pretty"
    if text in {"json", "csv"}:
        return text
    raise ValueError(f"Unsupported format: {text}")


def run_cli(benchmarks: Sequence[Benchmark]) -> None:
    parser = argparse.ArgumentParser(description="Ubench - Python microbenchmarking")
    parser.add_argument(
        "-q",
        "--quota",
        default="1s",
        help="Quota: e.g. '5s', '1000x', or '1%%' (variance)",
    )
    parser.add_argument(
        "-f",
        "--format",
        default="pretty",
        help="Output format: pretty, json, csv",
    )
    parser.add_argument(
        "--fork",
        action="store_true",
        help="(Ignored) Compatibility with OCaml fork mode",
    )
    parser.add_argument(
        "-w",
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--gc",
        action="store_true",
        help="Collect garbage between measurements",
    )
    parser.add_argument(
        "--ascii-only",
        action="store_true",
        help="Disable Unicode box drawing characters",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print regression analysis",
    )
    args = parser.parse_args()

    quota = parse_quota(args.quota)
    output_format = parse_output(args.format)

    config = (
        Config.default()
        .warmup(args.warmup)
        .gc_stabilization(args.gc)
        .ascii_only(args.ascii_only)
        .build()
    )
    if isinstance(quota, TimeLimit):
        config = config.time_limit(quota.seconds)
    elif isinstance(quota, IterationLimit):
        config = config.iteration_limit(quota.iterations)
    elif isinstance(quota, VarianceLimit):
        config = config.variance_limit(quota.coefficient)

    run_and_print(
        benchmarks,
        config=config,
        output_format=output_format,
        verbose=args.verbose,
    )


__all__ = [
    # Core API
    "Config",
    "default_config",
    "Quota",
    "TimeLimit",
    "IterationLimit",
    "VarianceLimit",
    "BenchmarkMode",
    "Measurement",
    "Statistics",
    "RegressionResult",
    "BenchData",
    "AnalysisResult",
    "ProgressInfo",
    # Benchmark creation
    "bench",
    "create",
    "group",
    "create_group",
    "bench_with_setup",
    "create_with_setup",
    "bench_param",
    "create_param",
    "flatten_benchmarks",
    # Execution
    "run",
    "run_silent",
    "run_and_print",
    "run_bench_with_config",
    "run_cli",
    # Output and utilities
    "print_pretty_table",
    "print_json",
    "print_csv",
    "print_regression_analysis",
    "compare",
    "print_comparison",
    "tabulate",
    "format_time_ns",
    "format_words",
    "format_number",
    "different_rates",
]
