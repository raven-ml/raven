"""Goldens of tinygrad/helpers.py: settings, number parsing, division, targets
and terminal text.

A column that holds arbitrary text, control characters included, holds it as
an OCaml string literal, which the suite reads back with Scanf's "%S".
"""

import os
import re
import random
import string

from golden import table
from tinygrad import helpers
from tinygrad.helpers import (DEV, Context, ContextVar, Target, ansilen, ansistrip, cdiv, ceildiv, cmod, colored,
                              floordiv, floormod, getenv, round_up, select_by_name, size_to_str, strip_parens,
                              time_to_str, to_function_name)


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


# Integers

NUMERATORS = [-9, -7, -5, -4, -1, 0, 1, 4, 5, 7, 9, 10**12 + 7, -(10**12 + 7)]
DENOMINATORS = [-5, -4, -2, -1, 0, 1, 2, 4, 5, 1000]


@table
def division():
    columns = ["x", "y", "cdiv", "cmod", "floordiv", "floormod", "ceildiv", "round_up"]
    # round_up is specified for positive and zero divisors only.
    return columns, [(x, y, cdiv(x, y), cmod(x, y), floordiv(x, y), floormod(x, y), outcome(ceildiv, x, y),
                      outcome(round_up, x, y) if y >= 0 else None)
                     for x in NUMERATORS for y in DENOMINATORS]


# Targets

TARGETS = [
    "", "AMD", "amd", "AMD:LLVM", ":LLVM", "AMD::gfx1100", "AMD:LLVM:gfx1100", "::gfx1100", "amd:llvm:GFX1100",
    "USB+", "USB+AMD", "usb+amd", "PCI:0+AMD", ":0+AMD", "PCI:0,1+AMD", "remote:host:2+nv:cuda:sm_89", "cpu::",
    "+nv:cuda:", "PCI:+", ":0,2+nv", "::Apple7", "AMD:", "AMD::", "a:b:+X", "PCI+NV+CUDA", "a+b+", "++",
    "CPU:CLANG:arm64:extra", ":::", "PCI:0+AMD:LLVM:gfx1100:x",
]


def fields(s):
    """The target's fields and printed form, or the exception and the text its
    message quotes."""
    try:
        t = Target.parse(s)
    except RuntimeError as e:
        return ("RuntimeError",) * 6 + (re.search(r": '(.*)'$", str(e)).group(1),)
    return (t.device, t.renderer, t.arch, t.interface, t.indices, repr(t), "")


@table
def targets():
    return ["input", "device", "renderer", "arch", "interface", "indices", "to_string", "names"], [
        (s, *fields(s)) for s in TARGETS]


DEVS = ["", "CPU", "CPU:LLVM", ":LLVM", "AMD:LLVM", "QCOM:IR3;AMD:LLVM", "PCI+;NV:CUDA", "NV:CUDA;PCI+",
        "CPU:CLANG;PCI:2,0+NV:CUDA:sm_89", "AMD::gfx1100", "::gfx90a", "USB+AMD:LLVM;:LLVM:x86"]


def target(dev, device, arch):
    with Context(DEV=dev):
        return repr(DEV.target(device, arch=arch))


@table
def device_targets():
    return ["dev", "device", "arch", "target"], [(dev, device, arch, target(dev, device, arch))
                                                 for dev in DEVS for device in ["CPU", "AMD", "NV", "QCOM"]
                                                 for arch in ["", "arm64"]]


# Selection

CANDIDATES = [
    [], ["CLANG", "LLVM"], ["CLANG", "LLVM", "CLANG"], ["PCI", "USB", "KFD", "AM", "MOCK"],
    ["CPU", "CUDA", "METAL", "AMD", "NV", "NULL", "DISK"], ["AC", "AD", "AB"], ["ab", "ba"],
]
QUERIES = ["", "CLANG", "LLVM", "CLANGJIT", "clang", "LVM", "USA", "PCIE", "NVK", "AM", "AX", "ADC", "X", "CUDAA", "a"]
# Queries that match candidates in several blocks, score exactly the cutoff,
# or are long enough, at 200 characters, for popular characters to be junk.
PAIRS = [
    (["METAL", "MOCK", "DISK", "ABCDE"], q) for q in ["MXTAL", "TEMAL", "MEAL", "DSIK", "ABCXY", "XABCDEX", "AMETXL",
                                                      "EDCBA", "KCOM", "DIKS", "MMETAL", "METALM"]
] + [(["A" * (n - 2), "QB"], "Z" + "A" * (n - 2) + "B") for n in [199, 200]] + [
    (["A" * 190 + "BCD", "BCD" + "A" * 150, "XBCD"], "A" * 197 + "BCD"),
]

# A 201-character query whose every character occurs three times, the most
# that is not yet popular, and a candidate that differs from its first
# character on, so that no match extends over popular characters.
_THRICE = (string.ascii_letters + string.digits + "!#$%&") * 3
PAIRS.append((["~" + _THRICE[1:195]], _THRICE))


def words(rng, count):
    return ["".join(rng.choice("ABCD") for _ in range(rng.randint(3, 9))) for _ in range(count)]


# Random words over four letters, which share many blocks.
_rng = random.Random(0)
PAIRS += [(words(_rng, 4), words(_rng, 1)[0]) for _ in range(120)]


def select(candidates, query):
    try:
        return ",".join(str(i) for i, _ in select_by_name(list(enumerate(candidates)), lambda c: c[1], query,
                                                              "no match")), ""
    except RuntimeError as e:
        return "", str(e)


@table
def selection():
    return ["candidates", "query", "selected", "error"], [(",".join(cs), q, *select(cs, q))
                                                          for cs, q in [(cs, q) for cs in CANDIDATES for q in QUERIES] + PAIRS]


# Terminal text

TEXTS = [
    "", "abc", "\x1b[31mx\x1b[0m", "a\x1b[Kb", "\x1b[1;31mbold\x1b[0m", "\x1b[", "\x1b[31", "x\x1b[Km",
    "\x1b[abc\nm", "\x1b[abc\nmn", "\x1b[31m\x1b[0m", "\x1bx", "\x1bxm", "a[31mb", "\x1b[xyzm plain", "café", "█▏",
    "\x1b[36m日本\x1b[0m", "\U0001f600", "E_4_4", "r_16_2n1", "a b-c.d", "\x01\x7f", "(1+2)", "(1+(2+3))",
    "(int)(1+2)", "((c35+c39>>23&255)+-127).cast(dtypes.float)", "(abc", "abc)", "(a+(b))*(c)", "()", "(()",
    "())", ")(", "(a)(b)", "((a))", "(", ")",
]


@table
def text():
    return ["input", "ansistrip", "ansilen", "to_function_name", "strip_parens"], [
        (literal(s), literal(ansistrip(s)), ansilen(s), to_function_name(s), literal(strip_parens(s)))
        for s in TEXTS]


COLORS = ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"]


@table
def colors():
    with Context(NO_COLOR=0):
        return ["color", "background", "colored"], [(c, bg, literal(colored("x", c, bg)))
                                                    for c in COLORS + [c.upper() for c in COLORS]
                                                    for bg in [False, True]]


DURATIONS = [
    (10.01, 8), (10, 8), (0.5, 8), (0.01, 8), (0.0001, 8), (0, 8), (10.01, 6), (12.5, 8), (0.05, 8), (5e-6, 9),
    (0.0123, 9), (10.000001, 8), (0.0100001, 8), (1e-7, 8), (-1, 8), (1e10, 8), (123456.789, 8), (0.5, 0),
    (0.5, 12), (float("inf"), 8), (float("nan"), 8),
]


@table
def durations():
    return ["seconds", "w", "time_to_str"], [(t, w, time_to_str(t, w)) for t, w in DURATIONS]


SIZES = [0, 1, 5, 1023, 1024, 1025, 1536, (1 << 20) - 1, 1 << 20, (1 << 30) - 1, 1 << 30, (1 << 30) + (1 << 29),
         3 << 30, 1 << 40, 1 << 50, -5, -2048]


@table
def sizes():
    return ["bytes", "size_to_str"], [(s, size_to_str(s)) for s in SIZES]
