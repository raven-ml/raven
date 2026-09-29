"""Goldens of tinygrad/dtype.py: the dtypes, their promotion and their casts."""

from golden import table
from tinygrad.dtype import can_lossless_cast, dtypes, least_upper_dtype

# Every dtype that takes part in promotion, in priority order.
SCALARS = sorted([*dtypes.all, *dtypes.weaks])


@table
def properties():
    columns = ["dtype", "priority", "bitsize", "itemsize", "name", "fmt", "min", "max",
               "is_int", "is_float", "is_unsigned", "is_bool"]
    return columns, [
        (dt, dt.priority, dt.bitsize, dt.itemsize, dt.name, dt.fmt, dt.min, dt.max,
         dtypes.is_int(dt), dtypes.is_float(dt), dtypes.is_unsigned(dt), dtypes.is_bool(dt))
        for dt in [dtypes.void, *SCALARS]]


@table
def least_upper():
    return ["a", "b", "least_upper"], [(a, b, least_upper_dtype(a, b)) for a in SCALARS for b in SCALARS]


@table
def lossless_cast():
    return ["from", "to", "lossless"], [(a, b, can_lossless_cast(a, b)) for a in SCALARS for b in SCALARS]


@table
def finfo():
    return ["dtype", "exponent", "mantissa"], [(dt, *dtypes.finfo(dt)) for dt in dtypes.floats]
