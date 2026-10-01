"""Goldens of tinygrad/uop/weak.py: graphs before and after each pass.

A golden `<name>.golden` is the sink of an input graph and of its rewrite to a
fixed point by one pass, in that order. The suite builds the same input with
tolk's constructors and compares its rewrite with the golden.
"""

from golden import graph
from tinygrad.dtype import AddrSpace, Invalid, dtypes
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops, ParamArg, UOp, graph_rewrite
from tinygrad.uop.weak import commit_weak_consts, pm_cast_const, pm_commit_weak, pm_lower_weak, pm_uncast_const


def rewritten(u, pm):
    return UOp.sink(u, graph_rewrite(u, pm))


def small(name="x", dtype=dtypes.int8):
    return UOp.variable(name, 0, 10, dtype)


def weak(name="w"):
    return UOp.variable(name, 0, 4)


def flag():
    return UOp.variable("c", False, True, dtypes.bool)


def loaded_float():
    return UOp.param(0, dtypes.float32, 1).index(UOp.const(0).cast(dtypes.int32)).load()


# commit_weak_consts

@graph
def commit_consts_at_a_stated_width():
    u = UOp.variable("h", 0.0, 1.0, dtypes.float16) * 2.0
    return UOp.sink(u, commit_weak_consts(u, dtypes.float16))


@graph
def commit_consts_leaves_weak_expressions():
    u = UOp(Ops.ADD, src=(weak() + 1, UOp.const(3)))
    return UOp.sink(u, commit_weak_consts(u, dtypes.int32))


# pm_commit_weak

@graph
def peer_keeps_a_derivable_literal_bare():
    return rewritten(small() + 3, pm_commit_weak)


@graph
def peer_rounds_a_derivable_literal():
    return rewritten(loaded_float() * UOp.const(-0.9999999893980771), pm_commit_weak)


@graph
def peer_commits_a_weak_expression():
    return rewritten(small() + (weak() + 1), pm_commit_weak)


@graph
def weak_sources_stay_weak_without_a_committed_peer():
    return rewritten(UOp(Ops.ADD, src=(UOp.const(1), UOp.const(1.0))), pm_commit_weak)


@graph
def where_keeps_a_weak_arm_bare():
    concrete = UOp.const(2.0).cast(dtypes.float16)
    return rewritten(UOp(Ops.WHERE, src=(UOp.const(True), concrete, UOp.const(1.0))), pm_commit_weak)


@graph
def shift_commits_its_weak_operand():
    return rewritten(UOp.const(0xFFFF) << UOp.variable("s", 0, 16, dtypes.uint32), pm_commit_weak)


@graph
def store_commits_its_value_at_the_destination():
    dst = UOp.param(0, dtypes.bfloat16, 1).index(UOp.const(0).cast(dtypes.int32))
    with Context(DEFAULT_FLOAT=dtypes.float16):
        return rewritten(dst.store(UOp.const(5.0), UOp.const(True)), pm_commit_weak)


@graph
def cast_widens_a_weak_expression():
    return rewritten((weak() + 1).cast(dtypes.int64), pm_commit_weak)


@graph
def cast_never_narrows_below_the_bounds():
    return rewritten((UOp.const(2**32) + 1).cast(dtypes.int8), pm_commit_weak)


@graph
def cast_never_narrows_below_the_default_float():
    with Context(DEFAULT_FLOAT=dtypes.float32):
        return rewritten((UOp.const(1.0) + UOp.const(2.0)).cast(dtypes.float16), pm_commit_weak)


@graph
def cast_to_another_kind_commits_nothing():
    return rewritten((weak() + 1).cast(dtypes.float32), pm_commit_weak)


@graph
def cast_to_a_weak_type_commits_nothing():
    return rewritten((weak() + 1).cast(dtypes.weakfloat), pm_commit_weak)


@graph
def cast_commits_operands_at_their_own_bounds():
    quotient = UOp(Ops.CDIV, src=(UOp.variable("n", 0, 2**40), UOp.const(2**40)))
    return rewritten(quotient.cast(dtypes.int32), pm_commit_weak)


@graph
def cast_anchors_a_mixed_expression_at_the_cast():
    with Context(DEFAULT_FLOAT=dtypes.float16):
        return rewritten((UOp.const(1).cast(dtypes.int32) + UOp.const(1.0)).cast(dtypes.float32), pm_commit_weak)


# pm_lower_weak

@graph
def lower_an_int_to_int32():
    return rewritten(UOp.sink(UOp.const(1) + UOp.const(2)), pm_lower_weak)


@graph
def lower_an_int_beyond_int32_to_int64():
    return rewritten(UOp.sink(UOp.const(2**32) + UOp.const(1)), pm_lower_weak)


@graph
def lower_an_int_beyond_int64_to_uint64():
    return rewritten(UOp.sink(UOp.const(2**64 - 1)), pm_lower_weak)


@graph
def lower_a_float_to_the_default_float():
    with Context(DEFAULT_FLOAT=dtypes.float16):
        return rewritten(UOp.sink(UOp.const(1.5) * UOp.const(2.0)), pm_lower_weak)


@graph
def lower_range_arithmetic():
    return rewritten(UOp.sink(UOp.range(16, 0) * 4), pm_lower_weak)


@graph
def lower_a_comparison():
    return rewritten(UOp.sink(UOp.range(16, 0) < 8), pm_lower_weak)


@graph
def lower_a_where():
    return rewritten(UOp.sink(flag().where(weak(), 3)), pm_lower_weak)


@graph
def lower_a_stack():
    return rewritten(UOp.sink(UOp.stack(weak(), UOp.const(3))), pm_lower_weak)


@graph
def lower_a_special():
    return rewritten(UOp.sink(UOp.special(8, "gidx0") * 2), pm_lower_weak)


@graph
def lower_a_unary_float():
    return rewritten(UOp.sink(UOp.const(2.0).exp2()), pm_lower_weak)


@graph
def lower_a_weak_int_cast_of_a_bool_as_a_conversion():
    return rewritten(UOp.sink(flag().cast(dtypes.weakint) + 1), pm_lower_weak)


@graph
def lower_a_weak_int_cast_of_a_float_as_a_conversion():
    return rewritten(UOp.sink(UOp.variable("f", 0.0, 9.0, dtypes.float32).cast(dtypes.weakint) + 1), pm_lower_weak)


@graph
def lower_a_weak_int_cast_of_an_int_as_a_restatement():
    return rewritten(UOp.sink(small(dtype=dtypes.int16).cast(dtypes.weakint) + 1), pm_lower_weak)


@graph
def lower_a_weak_float_cast_of_a_float_as_a_restatement():
    return rewritten(UOp.sink(UOp.variable("f", 0.0, 9.0, dtypes.float16).cast(dtypes.weakfloat).exp2()), pm_lower_weak)


@graph
def lower_stacked_weak_casts_as_two_conversions():
    return rewritten(UOp.sink(UOp.const(1.5, dtypes.float32).cast(dtypes.weakint).cast(dtypes.weakfloat)), pm_lower_weak)


@graph
def lower_stacked_weak_casts_of_a_weak_value_one_at_a_time():
    return rewritten(UOp.sink(weak().cast(dtypes.weakfloat).cast(dtypes.weakint) + 1), pm_lower_weak)


@graph
def lower_around_a_weak_node_it_does_not_lower():
    return rewritten(UOp.sink(UOp(Ops.CUSTOM, arg=("n", dtypes.weakint)) + 1), pm_lower_weak)


@graph
def lower_stacked_weak_casts_of_a_node_it_does_not_lower():
    custom = UOp(Ops.CUSTOM, arg=("n", dtypes.weakint))
    return rewritten(UOp.sink(custom.cast(dtypes.weakfloat).cast(dtypes.weakint)), pm_lower_weak)


@graph
def lower_keeps_weak_storage_outside_registers_of_scalars():
    return rewritten(UOp.sink(UOp(Ops.PARAM, arg=ParamArg(0, dtypes.weakint, size=4))), pm_lower_weak)


@graph
def lower_a_weak_variable_to_its_default():
    return rewritten(UOp.sink(weak() + 1), pm_lower_weak)


@graph
def lower_a_weak_expression_under_a_committed_consumer():
    return rewritten(UOp.sink(small(dtype=dtypes.int32) * (weak() + 1)), pm_lower_weak)


@graph
def lower_a_gated_long_index_into_a_small_buffer_to_int32():
    idx = UOp.variable("i", 0, 7, dtypes.int64)
    return rewritten(UOp.sink(UOp.param(0, dtypes.float32, 8).index(idx.valid(flag()))), pm_lower_weak)


@graph
def lower_a_gated_long_index_into_the_largest_int32_buffer_to_int32():
    idx = UOp.variable("i", 0, 2**31 - 1, dtypes.int64)
    return rewritten(UOp.sink(UOp.param(0, dtypes.float32, 2**31).index(idx.valid(flag()))), pm_lower_weak)


@graph
def lower_a_gated_long_index_into_a_buffer_past_int32_keeps_int64():
    idx = UOp.variable("i", 0, 2**31, dtypes.int64)
    return rewritten(UOp.sink(UOp.param(0, dtypes.float32, 2**31 + 1).index(idx.valid(flag()))), pm_lower_weak)


@graph
def lower_a_gated_long_index_into_a_huge_buffer_keeps_int64():
    idx = UOp.variable("i", 0, 2**32, dtypes.int64)
    return rewritten(UOp.sink(UOp.param(0, dtypes.float32, 2**33).index(idx.valid(flag()))), pm_lower_weak)


@graph
def lower_a_gated_shrink_to_the_width_its_bounds_need():
    buf = UOp.param(0, dtypes.float, 2**31 + 64)
    i = UOp.variable("i", 0, 2**28)
    return rewritten(UOp(Ops.SHRINK, src=(buf, (i * 24).valid(i < 2**28), UOp.const(4))).sink(), pm_lower_weak)


@graph
def lower_a_register_buffer_size():
    return rewritten(UOp.placeholder((4,), dtypes.float, 0, addrspace=AddrSpace.REG).sink(), pm_lower_weak)


# pm_uncast_const

@graph
def uncast_a_committed_literal():
    return rewritten(small(dtype=dtypes.int32) + UOp.const(1, dtypes.int32), pm_uncast_const)


@graph
def uncast_keeps_literals_with_no_committed_peer():
    return rewritten(UOp.const(1, dtypes.int32) + UOp.const(2, dtypes.int32), pm_uncast_const)


@graph
def uncast_keeps_a_shifted_literal():
    count = UOp.variable("s", 0, 31, dtypes.uint32)
    return rewritten(UOp(Ops.SHL, src=(UOp.const(1, dtypes.int32), count)), pm_uncast_const)


@graph
def uncast_keeps_a_cast_of_an_expression():
    return rewritten(small(dtype=dtypes.int32) + (weak() + 1).cast(dtypes.int32), pm_uncast_const)


@graph
def uncast_keeps_a_shifted_literal_of_its_peer_s_type():
    count = UOp.variable("s", 0, 31, dtypes.int32)
    return rewritten(UOp(Ops.SHL, src=(UOp.const(1, dtypes.int32), count)), pm_uncast_const)


@graph
def uncast_keeps_a_literal_that_widens_a_comparison():
    return rewritten(UOp(Ops.CMPLT, src=(small(dtype=dtypes.int16), UOp.const(1, dtypes.int32))), pm_uncast_const)


@graph
def uncast_drops_a_cast_that_fits():
    return rewritten(small("u", dtypes.uint8) < UOp.cconst(44, dtypes.uint8), pm_uncast_const)


# pm_cast_const

@graph
def cast_consts_state_each_edge_width():
    i, f = small("i", dtypes.int32), UOp.variable("f", 0.0, 10.0, dtypes.float32)
    return rewritten(UOp.sink(i + 1, f + 1, UOp.const(True)), pm_cast_const)


@graph
def cast_consts_give_underivable_literals_their_default():
    return rewritten(UOp.sink(UOp.const(1) + UOp.const(2**40)), pm_cast_const)


@graph
def cast_consts_leave_invalid_bare():
    return rewritten(UOp.sink(weak().cast(dtypes.int32).valid(flag())), pm_cast_const)
