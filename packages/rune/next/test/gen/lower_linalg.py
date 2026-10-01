"""The kernels tinygrad schedules for the linear algebra whose lowering agrees
with it: a float32 matrix product with the +0 the lowering adds to every float
sum, and the factors of tinygrad's Householder QR as the lowering builds them,
with LAPACK's reflectors and R with the zeros below its diagonal that it
selects. Each operand is a buffer
on the CPU."""

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops, UOp

from golden import graph


def operand(*shape):
    return Tensor.empty(*shape, dtype=dtypes.float32, device="CPU")


def kernels(t):
    """The kernel sinks of the schedule that realizes `t` into a buffer of its
    own shape, in order."""
    linear, _ = t.contiguous().linear_with_vars()
    return UOp.sink(*[call.src[0] for call in linear.src])


def fdiv(a, b):
    """`a / b` rounded once, where tinygrad's division multiplies by `1 / b`:
    the lowering divides operands of one shape, `b` expanded to `a`'s."""
    return a.alu(Ops.FDIV, b.expand(a.shape))


def qr(a):
    """tinygrad's `qr` (`mixin/op.py:1799`) as the lowering builds it: each
    quotient rounded once (`fdiv`), the sign of the first element written -1
    below zero and 1 elsewhere, which is `x0.ne(0).where(x0.sign(), 1)` at every
    value, NaN included, a column already zero below the diagonal not
    reflected, as LAPACK's reflectors are not, its update of Q and R skipped so
    that what it holds reaches neither, and the reflector built from the
    column divided by its largest magnitude, applied to the rows from the
    diagonal on, with the column itself written as its diagonal element and
    zeros below it, as LAPACK's are."""
    m, n = a.shape[-2:]
    R, Q = a, Tensor.eye(m, dtype=a.dtype)
    idx = Tensor.arange(m)
    rows, columns = idx.reshape(m, 1), Tensor.arange(n).reshape(1, n)
    for i in range(min(m, n)):
        at_i, x = idx.eq(i), (idx >= i).where(R[..., :, i], 0)
        magnitude = (x < 0).where(-x, x)
        largest = magnitude.max(-1, keepdim=True)
        scale = largest.ne(0).where(largest, 1)
        scaled = fdiv(x, scale)
        norm = (scaled * scaled).sum(-1, keepdim=True).sqrt()
        x0 = at_i.where(x, 0).sum(-1, keepdim=True)
        s0 = at_i.where(scaled, 0).sum(-1, keepdim=True)
        below = (idx > i).where(magnitude, 0).sum(-1, keepdim=True)
        sgn, active = (x0 < 0).where(x0.const_like(-1), x0.const_like(1)), below.ne(0)
        u0 = s0 + sgn * norm
        v = fdiv(at_i.where(u0, scaled), active.where(u0, 1)).unsqueeze(-1)
        w = active.where(fdiv(sgn * u0, active.where(norm, 1)), 0).unsqueeze(-1) * v
        diagonal = active.where((sgn * -1) * (scale * norm), x0)
        reflected = rows.eq(i).where(diagonal.unsqueeze(-1), (rows < i).where(R, 0))
        applied = R - w @ (v.transpose(-2, -1) @ R)
        on = active.unsqueeze(-1)
        Q = on.where(Q - (Q @ v) @ w.transpose(-2, -1), Q)
        R = columns.eq(i).where(reflected, (rows >= i).where(on.where(applied, R), R))
    return Q, R


@graph
def matmul(): return kernels(operand(4, 3) @ operand(3, 5) + 0.0)
@graph
def matmul_batched(): return kernels(operand(2, 1, 4, 3) @ operand(3, 3, 5) + 0.0)
@graph
def qr_q(): return kernels(qr(operand(4, 3))[0][:, :3])
@graph
def qr_r(): return kernels(qr(operand(4, 3))[1].triu()[:3])
