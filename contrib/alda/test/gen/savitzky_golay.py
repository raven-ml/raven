# golden/savitzky_golay.npz for test_taps.ml: Savitzky-Golay taps in exact
# rational arithmetic, rounded once. Row r(i) maps the window's samples to the
# deriv-th derivative at position `at` of the least-squares polynomial of the
# given order; the taps are r reversed (fir's orientation).
# Run from this directory: uv run --locked savitzky_golay.py ../golden/savitzky_golay.npz
import sys, numpy as np
from fractions import Fraction as F
from math import factorial

def row(w, order, deriv, at):
    at = F(at)
    n = order + 1
    G = [[F(sum(i ** (k + l) for i in range(w))) for l in range(n)] for k in range(n)]
    e = [F(0)] * n
    for k in range(deriv, n):
        e[k] = F(factorial(k), factorial(k - deriv)) * at ** (k - deriv)
    # solve G g = e (G symmetric positive definite), exact Gaussian elimination
    A = [G[k][:] + [e[k]] for k in range(n)]
    for c in range(n):
        p = next(r for r in range(c, n) if A[r][c] != 0)
        A[c], A[p] = A[p], A[c]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c] / A[c][c]
                A[r] = [x - f * y for x, y in zip(A[r], A[c])]
    g = [A[k][n] / A[k][k] for k in range(n)]
    return [sum(g[k] * i ** k for k in range(n)) for i in range(w)]

cases = {
    "w101o12d0": (101, 12, 0, 50), "w101o12d1a0": (101, 12, 1, 0),
    "w501o20d0": (501, 20, 0, 250), "w501o20d2": (501, 20, 2, F(7, 2)),
    "w2001o30d0": (2001, 30, 0, 1000), "w2001o30d0a0": (2001, 30, 0, 0),
    "w8o3d1": (8, 3, 1, F(7, 2)),
}
out = {}
for name, (w, o, d, a) in cases.items():
    r = row(w, o, d, a)
    out[name] = np.array([float(x) for x in reversed(r)])
    print(name, flush=True)
np.savez(sys.argv[1], **out)
