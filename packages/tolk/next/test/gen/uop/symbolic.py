"""Goldens of tinygrad/uop/symbolic.py: every simplification tinygrad's own
tests make, with its result, and `sym` on random integer expressions.

The generator runs the tests of symbolic.py, and the tests of other files that
test its rules, and records each simplification a test asks for: a rewrite by
one of symbolic.py's matchers, `UOp.simplify`, or `simplify_valid`. A
simplification is recorded when a test asks for it, directly, through a method
of `UOp` (`ssimplify`, `resolve`, `float`, `bool`) or through an assertion, and
not when tinygrad makes it on its own way, inside another simplification or a
pass. `<class>.<test>.golden` holds the records of that test, in the order the
test makes them, each once, and `tests.golden` lists the tests that have
records, with the file that declares them.

A record is a sink, tagged with its kind, whose sources are the input and the
result: `(in, out)` for a rewrite and for `simplify`, `(valid, out)` for
`simplify_valid`, and `(valid,)` when it is `None`. The kind names the matcher,
`sym`, `symbolic`, ..., or `simplify` or `simplify_valid`, with ` bottom_up`
for a bottom-up rewrite. When the result is an integer or a boolean whose bounds
are integers of 62 bits, the tag is `(kind, vmin, vmax)`, the bounds tinygrad
computes for it.
"""

import random
import sys
import types
import unittest


# The tests import what the recording does not need: z3 proves their results,
# numpy, hypothesis and pytest serve other tests of their files, and
# test.helpers runs the code generation pipeline, which is not symbolic.py's.
# Stand-ins let them load; a test that needs one of them fails, and records
# what it asked for before.
def stand_in(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    return module


def unavailable(*args, **kwargs):
    raise NotImplementedError("not needed to record simplifications")


def keep(*args, **kwargs):
    return lambda f: f


stand_in("z3", get_version=lambda: (4, 12, 4, 0), __getattr__=lambda name: object)
stand_in("numpy", __getattr__=lambda name: object)
stand_in("pytest", mark=types.SimpleNamespace(xfail=keep, skip=keep, skipif=keep))
stand_in("hypothesis", given=keep, settings=keep,
         strategies=stand_in("hypothesis.strategies", __getattr__=lambda name: lambda *args, **kwargs: None))
stand_in("test.helpers", full_rewrite=unavailable, to_uops_list=unavailable, eval_uop=unavailable)

from golden import graph, table
from tinygrad.dtype import dtypes
from tinygrad.helpers import DEV
from tinygrad.uop.ops import UOp, Ops, graph_rewrite
from tinygrad.uop.weak import pm_commit_weak
import tinygrad.uop.symbolic as symbolic
import test.null.test_uop_symbolic as test_uop_symbolic
import test.null.test_symbolic_failures as test_symbolic_failures
import test.null.test_simplify_valid_idx as test_simplify_valid_idx
import test.null.test_const_folding as test_const_folding
import test.null.test_uop_graph as test_uop_graph
import test.null.test_uop_resolve as test_uop_resolve
import test.null.test_uops as test_uops
import test.null.test_graph_rewrite as test_graph_rewrite
import test.null.test_dtype_weak as test_dtype_weak

# The files and, where not every test is symbolic.py's, the tests to run: those
# that the sections of other modules leave to Symbolic's.
FILES = {
    "null/test_uop_symbolic.py": (test_uop_symbolic, None),
    "null/test_symbolic_failures.py": (test_symbolic_failures, None),
    "null/test_simplify_valid_idx.py": (test_simplify_valid_idx, None),
    "null/test_const_folding.py": (test_const_folding, None),
    "null/test_uop_graph.py": (test_uop_graph, {
        "test_gep_const", "test_add_const", "test_cast", "test_mul", "test_div", "test_neg", "test_neg_min_int",
        "test_payne_hanek_reduction_bug", "test_commutative_work", "test_consts_go_last_right_away",
        "test_consts_go_last", "test_where_same_fold", "test_where_const_fold", "test_depth_2_const_fold"}),
    "null/test_uop_resolve.py": (test_uop_resolve, {
        "test_rtruediv", "test_float_direct", "test_ssimplify", "test_x_lt_x", "test_plus_ordering_lt"}),
    "null/test_uops.py": (test_uops, {
        "test_remove_invalid_stack_lanes", "test_cast_folds", "test_remove_intermediate_cast",
        "test_safe_cast_using_bounds", "test_cmp_self_folding_multidim"}),
    "null/test_graph_rewrite.py": (test_graph_rewrite, {"test_const_folding"}),
    "null/test_dtype_weak.py": (test_dtype_weak, {
        "test_committed_const_conversion_folds", "test_derivable_const_rounds_at_the_derived_width"}),
}

# A test's Tensor.empty places its buffer on the default device, which would
# otherwise be the machine's own.
DEV.value = "CPU"

MATCHERS = {name: getattr(symbolic, name) for name in
            ("sym", "symbolic", "symbolic_simple", "commutative", "pm_simplify_valid", "pm_move_where_on_load",
             "pm_drop_and_clauses", "pm_remove_invalid", "pm_clean_up_group_sink")}
MATCHERS["sym+pm_move_where_on_load"] = symbolic.sym + symbolic.pm_move_where_on_load
MATCHERS["symbolic_simple+pm_commit_weak"] = symbolic.symbolic_simple + pm_commit_weak

# test_variable_divmod bounds a variable by another, which tolk.next does not
# allow: a variable's bounds are numbers. Images are out of tolk.next's scope.
DROPPED = {"test_variable_divmod"}
DROPPED_CLASSES = {"TestImageSimplification", "TestImageStore"}

BOUND = 2**62


def matcher_name(pm):
    return next((name for name, m in MATCHERS.items() if m is pm), None)


def kind_tag(kind, out):
    if out.dtype in (dtypes.bool, dtypes.weakint) or dtypes.is_int(out.dtype):
        lo, hi = out.vmin, out.vmax
        if all(isinstance(v, (bool, int)) and -BOUND < v < BOUND for v in (lo, hi)): return (kind, lo, hi)
    return kind


def make_record(kind, srcs, out):
    return UOp(Ops.SINK, src=tuple(srcs), tag=kind_tag(kind, out) if out is not None else kind)


def test_cases():
    """Each test to run, as (file, class, name)."""
    for file, (module, selected) in FILES.items():
        for cls in vars(module).values():
            if not (isinstance(cls, type) and issubclass(cls, unittest.TestCase)) or cls.__module__ != module.__name__ \
               or cls.__name__ in DROPPED_CLASSES: continue
            for name in sorted(n for n in dir(cls) if n.startswith("test")):
                if name not in DROPPED and (selected is None or name in selected): yield file, cls, name


def record_all():
    """Each test's records, and the tests that have some, in the order they
    ran, with their file and class."""
    records, tests, files, current, depth = {}, [], {}, [None], [0]
    modules = {m.__name__ for m, _ in FILES.values()}

    def asked_by_a_test():
        frame = sys._getframe(2)
        while frame is not None and frame.f_globals.get("__name__") in ("tinygrad.uop.ops", "unittest.case"):
            frame = frame.f_back
        return frame is not None and frame.f_globals.get("__name__") in modules

    def record(kind, srcs, out):
        rec = make_record(kind, srcs, out)
        recs = records.setdefault(current[0], [])
        if rec not in recs: recs.append(rec)

    def outermost(fn, kind_of, srcs_of):
        def wrapped(*args, **kwargs):
            if depth[0] or current[0] is None or not asked_by_a_test(): return fn(*args, **kwargs)
            depth[0] += 1
            try: out = fn(*args, **kwargs)
            finally: depth[0] -= 1
            if (kind := kind_of(*args, **kwargs)) is not None: record(kind, srcs_of(out, *args, **kwargs), out)
            return out
        return wrapped

    def rewrite_kind(sink, pm, ctx=None, bottom_up=False, **_):
        name = matcher_name(pm)
        if name is None or ctx is not None: return None
        return name + (" bottom_up" if bottom_up else "")

    simplify = UOp.simplify
    rewrites = {m: m.graph_rewrite for m, _ in FILES.values() if hasattr(m, "graph_rewrite")}
    simplify_valid = test_simplify_valid_idx.simplify_valid
    UOp.simplify = outermost(simplify, lambda self, tracked=False: "simplify", lambda out, self, tracked=False: (self, out))
    for m, rewrite in rewrites.items():
        m.graph_rewrite = outermost(rewrite, rewrite_kind, lambda out, sink, *a, **k: (sink, out))
    test_simplify_valid_idx.simplify_valid = outermost(
        simplify_valid, lambda valid: "simplify_valid",
        lambda out, valid: (valid,) if out is None else (valid, out))
    test_uop_symbolic.TestSymbolic.check_equal_z3 = lambda *args: None
    try:
        for file, cls, name in test_cases():
            key = current[0] = f"{cls.__name__}.{name}"
            before = list(records.get(key, []))
            try: getattr(cls(name), name)()
            except Exception: pass  # an expected failure or a stand-in: its records are tinygrad's behaviour
            if any(k == key for k, _, _ in tests):
                if records.get(key, []) != before: sys.exit(f"two tests {key} record different things")
            elif key in records: tests.append((key, cls.__name__, name))
            files.setdefault(key, file)
            current[0] = None
    finally:
        UOp.simplify = simplify
        for m, rewrite in rewrites.items(): m.graph_rewrite = rewrite
        test_simplify_valid_idx.simplify_valid = simplify_valid
    return records, [(files[key], cls, name) for key, cls, name in tests]


RECORDS, TESTS = record_all()


def declare(key):
    def body(): return UOp.sink(*RECORDS[key])
    body.__name__ = key
    graph(body)


for _, cls, name in TESTS: declare(f"{cls}.{name}")


@table
def tests():
    return ["file", "class", "test"], TESTS


# Random expressions

def expression(rng, variables, divisors, depth):
    """An integer expression over `variables`, dividing by constants and by
    `divisors`, whose bounds exclude 0."""
    if depth == 0 or rng.random() < 0.25:
        return rng.choice(variables) if rng.random() < 0.7 else UOp.const(rng.choice([0, 1, -1, 2, 3, 4, 5, 7, 8, -3, 16]))
    op = rng.choice(["add", "mul", "div", "mod", "vdiv", "vmod", "max", "where", "neg", "mulv"])
    a = expression(rng, variables, divisors, depth - 1)
    if op == "neg": return -a
    if op in ("div", "mod"):
        c = rng.choice([1, 2, 3, 4, 5, 8, -2, -3, 7, 12])
        return a // c if op == "div" else a % c
    if op in ("vdiv", "vmod"):
        d = rng.choice(divisors)
        return a // d if op == "vdiv" else a % d
    if op == "mul": return a * rng.choice([0, 1, -1, 2, 3, 4, -2, 5])
    b = expression(rng, variables, divisors, depth - 1)
    if op == "mulv": return a * b
    if op == "add": return a + b
    if op == "max": return a.maximum(b)
    return (a < expression(rng, variables, divisors, depth - 1)).where(b, a)


@graph
def random_expressions():
    """`sym` on random integer expressions, recorded as the tests' rewrites."""
    rng, recs = random.Random(0), []
    while len(recs) < 200:
        variables = [UOp.variable(n, lo, lo + rng.randint(0, 20)) for n in "abc" for lo in [rng.randint(-10, 10)]]
        divisors = []
        for n in "de":
            lo = rng.randint(1, 6)
            hi = lo + rng.randint(0, 6)
            divisors.append(UOp.variable(n, *((-hi, -lo) if rng.random() < 0.3 else (lo, hi))))
        e = expression(rng, variables, divisors, 4)
        rec = make_record("sym", (e, out := graph_rewrite(e, symbolic.sym)), out)
        if rec not in recs: recs.append(rec)
    return UOp.sink(*recs)
