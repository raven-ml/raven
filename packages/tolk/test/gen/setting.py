"""Goldens of tinygrad/helpers.py's settings: their defaults, and how a
variable's text reads as a number.

A column that holds arbitrary text, control characters included, holds it as
an OCaml string literal, which the suite reads back with Scanf's "%S".
"""

import os

from golden import table
from tinygrad import helpers
from tinygrad.helpers import DEV, ContextVar, getenv


def literal(s):
    """`s` as an OCaml string literal of its bytes, which a byte undecodable as
    UTF-8 holds as a surrogate escape. A character that does not print is
    written as the escapes of its UTF-8 bytes."""
    escapes = {'"': '\\"', "\\": "\\\\", "\n": "\\n", "\t": "\\t", "\r": "\\r"}

    def char(c):
        if c in escapes:
            return escapes[c]
        if c.isprintable():
            return c
        return "".join(f"\\x{b:02x}" for b in c.encode("utf-8", "surrogateescape"))
    return '"' + "".join(map(char, s)) + '"'


def outcome(fn, *args):
    """`fn(*args)`, or the name of the exception it raises."""
    try:
        return fn(*args)
    except Exception as e:
        return type(e).__name__


# Settings

# PARALLEL defaults to the number of CPUs of the machine that generates, and
# generate.py sets NO_COLOR.
UNRECORDED = {"PARALLEL", "NO_COLOR"}


@table
def settings():
    # ContextVar's == compares values, so the variables are told apart by id.
    declared = {id(var) for var in vars(helpers).values() if isinstance(var, ContextVar)}
    return ["key", "default"], [(var.key, str(var) if var is DEV else var.value) for var in ContextVar._cache.values()
                                if id(var) in declared and var.key not in UNRECORDED]


def parse_env(text, default):
    os.environ["TOLK_GOLDEN"] = text
    getenv.cache_clear()
    return getenv("TOLK_GOLDEN", default)


ENV_VALUES = [
    "0", "1", "-1", "+7", "007", "-0", "00", "0_0", "1_000", "-1_000", "1__000", "_1", "1_", "4611686018427387903",
    "-4611686018427387904", " 12 ", "\t12\n", "\x0b12\x0c", "\r12\r", "1 2", "- 1", "--1", "+-1", "12abc", "",
    "  ", "0x10", "0b1", "0o7", "1.5", ".5", "5.", "+.5", "-0.0", "1e3", "1E-3", "1.5e", "1_0.5", "1_000.000_1",
    "1__0.5", "_1.5", "1._5", "1.5_", "1e1_0", ".", "e3", "inf", "-inf", "+inf", "INF", "infinity", "-Infinity",
    "infinit", "nan", "NaN", "-nan", "1e400", "-1e400", "1e-400", "0x1p3", " 3.25 ",
    "\x1c12\x1f", "\x1d1\x1e", "\x1b12", "\udcc212", "12\udcc2", "1e", "1e+", "1_0.5_0", "1.5e1_0", "1e_5", "NAN",
    "+nan", "nan(1)", "1.5.", "e5", "-.5e-3", "\t\n\x0b\x0c\r ", "+", "-", "+.", ".e1", "-e5", "+_1",
]


@table
def env():
    return ["input", "int", "float"], [(literal(s), outcome(parse_env, s, 0), outcome(parse_env, s, 0.0))
                                       for s in ENV_VALUES]
