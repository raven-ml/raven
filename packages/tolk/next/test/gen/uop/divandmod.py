"""Goldens of tinygrad/uop/divandmod.py: divisions and remainders, each
followed by its rewrite by div_and_mod_symbolic, applied once, or by itself
where no rule applies.

A golden named after a claim holds one case, which the suite builds with
tolk.next's constructors. The other goldens come from tinygrad's own tests,
which test div_and_mod_symbolic through the whole symbolic rule set: the
generator runs them, records every expression they simplify, and rewrites each
division and remainder in it. `<test>.golden` holds those of the test of that
name. A rewrite that simplifies anything but constants on its way depends on
the symbolic rules, and goes to `<test>_needs_symbolic.golden` instead.
"""

import sys
import types
import unittest

# The tests prove their results with z3, which the capture does not need: it
# only reads the expressions they build.
z3 = types.ModuleType("z3")
z3.get_version = lambda: (4, 12, 4, 0)
z3.__getattr__ = lambda name: object
sys.modules["z3"] = z3

import golden
from golden import graph
from graph import write
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, Ops
from tinygrad.uop.divandmod import div_and_mod_symbolic, fold_divmod_general
import test.null.test_uop_symbolic as symbolic_tests
import test.null.test_symbolic_failures as failure_tests

DIVMOD = (Ops.FLOORDIV, Ops.FLOORMOD)


def captured_divmods():
    """Each test's divisions and remainders, in the order the test simplifies
    them, a node shared with an earlier test kept by the earlier one."""
    by_test, seen, ran, current = {}, set(), set(), [None]
    simplify, rewrite = UOp.simplify, symbolic_tests.graph_rewrite

    def record(u):
        for d in u.toposort():
            if d.op in DIVMOD and d not in seen:
                seen.add(d)
                by_test.setdefault(current[0], []).append(d)

    def recording_simplify(self, tracked=False):
        record(self)
        return simplify(self, tracked)

    def recording_rewrite(u, *args, **kwargs):
        record(u)
        return rewrite(u, *args, **kwargs)

    symbolic_tests.graph_rewrite = recording_rewrite
    symbolic_tests.TestSymbolic.check_equal_z3 = lambda *args: None
    UOp.simplify = recording_simplify
    try:
        for module in (symbolic_tests, failure_tests):
            for cls in vars(module).values():
                if not (isinstance(cls, type) and issubclass(cls, unittest.TestCase)): continue
                for name in sorted(n for n in dir(cls) if n.startswith("test")):
                    if name in ran: continue  # test_symbolic_failures repeats test_fuzz_failure1
                    ran.add(name)
                    current[0] = name
                    try: getattr(cls(name), name)()
                    except Exception: pass  # an assertion of the whole rule set: the expressions are recorded
    finally:
        UOp.simplify, symbolic_tests.graph_rewrite = simplify, rewrite
    return by_test


def constant(u):
    """Whether simplifying u needs no rule: u is a constant, or a sink of
    constants and of stacks of constants."""
    def constants(s): return s.op is Ops.CONST or (s.op is Ops.STACK and all(c.op is Ops.CONST for c in s.src))
    return u.op is Ops.CONST or (u.op is Ops.SINK and all(constants(s) for s in u.src))


def rewrite_once(d):
    """d's rewrite, or d, and whether the rewrite simplified anything but
    constants."""
    simplified, simplify = [], UOp.simplify

    def watched(self, tracked=False):
        if not constant(self): simplified.append(self)
        return simplify(self, tracked)

    # the tests filled fold_divmod_general's cache: a cached result would hide its simplifications
    fold_divmod_general.cache_clear()
    UOp.simplify = watched
    try: r = div_and_mod_symbolic.rewrite(d)
    finally: UOp.simplify = simplify
    return (d if r is None else r), bool(simplified)


def once(d):
    r = div_and_mod_symbolic.rewrite(d)
    return UOp.sink(d, d if r is None else r)


def v(name, lo, hi, **kwargs):
    return UOp.variable(name, lo, hi, **kwargs)


x, a, b = v("x", 0, 100), v("a", 0, 99), v("b", 0, 99)
signed = v("x", -100, 100)


# (x//c + a)//d

@graph
def merge_nested_divisions():
    return once((x // 2 + 3) // 4)


@graph
def merge_nested_divisions_by_a_negative_inner_divisor():
    return once((signed // -2 + 3) // 4)


@graph
def merge_nested_divisions_of_committed_integers():
    return once((v("x", 0, 100, dtype=dtypes.int32) // 2 + 3) // 4)


@graph
def split_nested_divisions_by_a_negative_divisor():
    return once((x // 2 + 3) // -4)


# (x + c)//d and (x + c)%d

@graph
def split_the_constant_out_of_a_division():
    return once((x + 7) // 4)


@graph
def split_the_constant_out_of_a_remainder():
    return once((x + 7) % 4)


@graph
def split_a_negative_constant_out_of_a_division():
    return once((x + UOp.const(-3)) // 4)


@graph
def split_the_constant_out_of_a_division_by_a_negative_divisor():
    return once((x + 7) // -4)


@graph
def keep_a_constant_smaller_than_the_divisor():
    return once((x + 3) // 4)


@graph
def keep_a_committed_integer_division():
    return once((v("x", 0, 100, dtype=dtypes.int32) + 7) // 4)


# One possible quotient

@graph
def fold_a_division_with_one_quotient():
    return once(v("x", 10, 14) // 5)


@graph
def fold_a_remainder_with_one_quotient():
    return once(v("x", 10, 14) % 5)


@graph
def fold_a_division_by_a_variable_with_one_quotient():
    return once(v("x", 0, 2) // v("y", 3, 10**12))


@graph
def fold_a_remainder_by_a_variable_with_one_quotient():
    return once(v("x", 0, 2) % v("y", 3, 10**12))


# Declared multiples

@graph
def fold_the_remainder_of_a_declared_multiple():
    return once(v("m", 0, 100, multiple_of=4) % 4)


@graph
def fold_the_remainder_of_a_declared_multiple_by_a_divisor_of_it():
    return once(v("m", 0, 100, multiple_of=4) % 2)


@graph
def keep_the_division_of_a_declared_multiple():
    return once(v("m", 0, 100, multiple_of=4) // 4)


@graph
def rewrite_the_remainder_of_a_declared_multiple_by_another_divisor():
    return once(v("m", 0, 100, multiple_of=4) % 3)


# Constant divisors

@graph
def nest_the_division_of_a_remainder():
    return once(a % 12 // 3)


@graph
def drop_a_nested_remainder():
    return once(a % 12 % 3)


@graph
def drop_a_nested_remainder_from_a_sum():
    return once((a % 4 + b) % 2)


@graph
def fold_a_remainder_by_congruence():
    return once((a * 5 + 3) % 4)


@graph
def fold_a_division_by_congruence():
    return once((a * 5 + 3) // 4)


@graph
def divide_a_common_factor_out_of_a_division():
    return once((a * 2 + 3) // 4)


@graph
def divide_a_common_factor_out_of_a_remainder():
    return once((a * 2 + 3) % 4)


@graph
def nest_a_division_by_a_factor_of_a_term():
    return once((a * 6 + b * 2 + 1) // 12)


@graph
def nest_a_remainder_by_a_factor_of_a_term():
    return once((a * 6 + b * 2 + 1) % 12)


@graph
def keep_a_division_of_huge_coefficients_exact():
    return once((a * 2**100) // (2**101 - 1))


@graph
def keep_a_remainder_whose_nested_part_reaches_the_factor():
    return once((v("a", 0, 5) * 4 + v("t", 0, 4)) % 12)


@graph
def keep_a_plain_division():
    return once(x // 5)


@graph
def keep_a_plain_remainder():
    return once(x % 5)


# Other divisors

@graph
def divide_a_common_divisor_out_of_a_division_by_a_variable():
    return once((a * 4 + b * 6) // (v("q", 0, 10) * 2))


@graph
def divide_a_common_divisor_out_of_a_remainder_by_a_variable():
    return once((a * 4 + b * 6) % (v("q", 0, 10) * 2))


@graph
def take_the_multiples_of_a_variable_divisor_out_of_a_division():
    d = v("d", 2, 5)
    return once((d * v("q", 0, 10) + 100) // d)


@graph
def take_the_multiples_of_a_variable_divisor_out_of_a_remainder():
    d = v("d", 2, 5)
    return once((d * v("q", 0, 10) + 100) % d)


@graph
def take_the_multiples_of_a_divisor_that_can_be_zero_out_of_a_division():
    d = v("d", 0, 5)
    return once((d * v("q", 0, 10) + 100) // d)


@graph
def keep_a_division_by_a_variable_that_can_be_negative():
    d = v("d", -2, 3)
    return once((d * v("q", 0, 10) + 100) // d)


@graph
def keep_a_remainder_by_a_variable_that_can_be_negative():
    d = v("d", -2, 3)
    return once((d * v("q", 0, 10) + 100) % d)


@graph
def divide_zero_by_a_divisor_that_can_be_zero():
    return once(v("x", 0, 0) // v("y", -5 * 10**9, 5 * 10**9))


@graph
def split_a_constant_factor_out_of_a_remainder():
    return once((a * 3 + b) % 2)


@graph
def split_constant_factors_out_of_a_remainder():
    return once((a * 3 + b * 5 + x) % 2)


def declare(name, pairs):
    golden.GOLDENS.append((name, lambda: write(UOp.sink(*(u for pair in pairs for u in pair)))))


for test, divmods in captured_divmods().items():
    alone, symbolic = [], []
    for d in divmods:
        try: r, needs_symbolic = rewrite_once(d)
        except ZeroDivisionError: continue  # a divisor that is always 0: the suite states it
        except AssertionError: continue  # a variable bounded by a node, which a variable of tolk.next cannot be
        (symbolic if needs_symbolic else alone).append((d, r))
    if alone: declare(test, alone)
    if symbolic: declare(f"{test}_needs_symbolic", symbolic)
