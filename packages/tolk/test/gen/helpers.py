"""Goldens of tinygrad/helpers.py: division, targets and terminal text.

A column that holds arbitrary text, control characters included, holds it as
an OCaml string literal, which the suite reads back with Scanf's "%S".
"""

import re

from golden import table
from tinygrad.helpers import (Context, Target, ansilen, ansistrip, ceildiv, colored, floordiv, floormod, round_up,
                              size_to_str, strip_parens, time_to_str, to_function_name)


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


# Integers

NUMERATORS = [-9, -7, -5, -4, -1, 0, 1, 4, 5, 7, 9, 10**12 + 7, -(10**12 + 7)]
DENOMINATORS = [-5, -4, -2, -1, 0, 1, 2, 4, 5, 1000]


@table
def division():
    columns = ["x", "y", "floordiv", "floormod", "ceildiv", "round_up"]
    # round_up is specified for positive and zero divisors only.
    return columns, [(x, y, floordiv(x, y), floormod(x, y), outcome(ceildiv, x, y),
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
