"""The kernels tinygrad schedules for the linear algebra whose lowering agrees
with it: a float32 matrix product with the +0 the lowering adds to every float
sum, and the factors of tinygrad's Householder QR as the lowering builds them,
R with the zeros below its diagonal that it selects. Each operand is a buffer
on the CPU."""

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp

from golden import graph


def operand(*shape):
    return Tensor.empty(*shape, dtype=dtypes.float32, device="CPU")


def kernels(t):
    """The kernel sinks of the schedule that realizes `t` into a buffer of its
    own shape, in order."""
    linear, _ = t.contiguous().linear_with_vars()
    return UOp.sink(*[call.src[0] for call in linear.src])


def qr(a):
    """tinygrad's `qr` (`mixin/op.py:1799`), with the sign of the first element
    written as the lowering writes it: -1 below zero and 1 elsewhere, which is
    `x0.ne(0).where(x0.sign(), 1)` at every value, NaN included."""
    m, n = a.shape[-2:]
    R, Q = a, Tensor.eye(m, dtype=a.dtype)
    idx = Tensor.arange(m)
    for i in range(min(m, n)):
        at_i, x = idx.eq(i), (idx >= i).where(R[..., :, i], 0)
        norm = x.square().sum(-1, keepdim=True).sqrt()
        x0 = at_i.where(x, 0).sum(-1, keepdim=True)
        sgn, active = (x0 < 0).where(x0.const_like(-1), x0.const_like(1)), norm.ne(0)
        u0 = x0 + sgn * norm
        v = (at_i.where(u0, x) / active.where(u0, 1)).unsqueeze(-1)
        w = active.where(sgn * u0 / active.where(norm, 1), 0).unsqueeze(-1) * v
        R = R - w @ (v.transpose(-2, -1) @ R)
        Q = Q - (Q @ v) @ w.transpose(-2, -1)
    return Q, R


@graph
def matmul(): return kernels(operand(4, 3) @ operand(3, 5) + 0.0)
@graph
def matmul_batched(): return kernels(operand(2, 1, 4, 3) @ operand(3, 3, 5) + 0.0)
@graph
def qr_q(): return kernels(qr(operand(4, 3))[0][:, :3])
@graph
def qr_r(): return kernels(qr(operand(4, 3))[1].triu()[:3])
