# golden/least_squares.npz for test_taps.ml: the taps minimising the integral
# of the squared error in units of the tolerance, from the closed-form Gram
# matrix and right side solved in Arb (python-flint) to 1e-30 relative.
# Bands (lo, hi, (g0, g1), (t0, t1)) in cycles a sample (rate 1); targets and
# tolerances linear in frequency.
# Run from this directory: uv run --locked least_squares.py ../golden/least_squares.npz
import sys, numpy as np
from flint import arb, arb_mat, ctx

def basis(sym, m):
    if m % 2:
        h = m // 2
        return (1, list(range(h + 1)) if sym else list(range(1, h + 1)))
    return (2, [2 * k + 1 for k in range(m // 2)])

def exact(sym, m, bands, prec=512):
    ctx.prec = prec
    u, f = basis(sym, m); L = len(f)
    pi = arb.pi()
    G = arb_mat(L, L); r = arb_mat(L, 1)
    # Integrals over [a, b] of cos(c w), sin(c w), w^j cos/sin(c w), j <= 3, by
    # Gauss-Legendre in Arb would be heavy; the weight 1/tau^2 with tau linear
    # makes closed forms awkward, so integrate numerically in Arb with a high
    # order Gauss-Legendre rule per band, whose error is checked by doubling.
    def integral(fn, a, b, n):
        xs, ws = gl(n)
        s = arb(0)
        for x, w in zip(xs, ws):
            t = (b - a) / 2 * x + (a + b) / 2
            s += w * fn(t)
        return s * (b - a) / 2
    for (lo, hi, (g0, g1), (t0, t1)) in bands:
        a = 2 * pi * arb(lo); b = 2 * pi * arb(hi)
        tg = lambda w: arb(g0) + (arb(g1) - g0) * (w - a) / (b - a)
        tt = lambda w: arb(t0) + (arb(t1) - t0) * (w - a) / (b - a)
        phi = (lambda k, w: (f[k] * w / u).cos()) if sym else (lambda k, w: (f[k] * w / u).sin())
        n = NODES
        xs, ws = gl(n)
        pts = [((b - a) / 2 * x + (a + b) / 2, w * (b - a) / 2) for x, w in zip(xs, ws)]
        for (t, w) in pts:
            W = w / tt(t) ** 2
            ph = [phi(k, t) for k in range(L)]
            for k in range(L):
                for l in range(k, L):
                    G[k, l] += W * ph[k] * ph[l]
                r[k, 0] += W * ph[k] * tg(t)
    for k in range(L):
        for l in range(k):
            G[k, l] = G[l, k]
    try:
        c = G.solve(r)
    except ZeroDivisionError:  # not certified nonsingular at this precision
        return exact(sym, m, bands, 2 * prec)
    return u, f, [c[k, 0] for k in range(L)]

_gl = {}
def gl(n):
    # Gauss-Legendre nodes and weights in Arb by Newton on P_n
    if n in _gl: return _gl[n]
    xs = []; ws = []
    for i in range(1, n + 1):
        x = arb(np.cos(np.pi * (i - 0.25) / (n + 0.5)))
        for _ in range(100):
            p0 = arb(1); p1 = x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1)
            dx = p1 / dp
            x = (x - dx).mid()
            x = arb(x)
            if abs(float(dx.mid())) < 1e-60: break
        p0 = arb(1); p1 = x
        for k in range(2, n + 1):
            p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        dp = n * (x * p1 - p0) / (x * x - 1)
        xs.append(x); ws.append(2 / ((1 - x * x) * dp * dp))
    _gl[n] = (xs, ws)
    return xs, ws

def taps(sym, m, u, f, c):
    h = np.zeros(m)
    for k, fk in enumerate(f):
        ck = float(c[k].mid())
        if m % 2:
            before, after = m // 2 - fk, m // 2 + fk
        else:
            before, after = m // 2 - (fk + 1) // 2, m // 2 - 1 + (fk + 1) // 2
        if fk == 0: h[before] = ck
        else:
            h[before] = ck / 2; h[after] = ck / 2 if sym else -ck / 2
    return h

cases = {
    "lp31": (True, 31, [(0, 0.2, (1, 1), (0.01, 0.01)), (0.25, 0.5, (0, 0), (0.001, 0.001))]),
    "lp30": (True, 30, [(0, 0.2, (1, 1), (0.01, 0.01)), (0.25, 0.5, (0, 0), (0.001, 0.001))]),
    "hilbert31": (False, 31, [(0.05, 0.45, (1, 1), (1, 1))]),
    "hilbert30": (False, 30, [(0.05, 0.5, (1, 1), (1, 1))]),
    "diff30": (False, 30, [(0, 0.4, (0, 2 * np.pi * 0.4), (1, 1))]),
    "relative31": (False, 31, [(0.02, 0.4, (2 * np.pi * 0.02, 2 * np.pi * 0.4), (2 * np.pi * 0.02, 2 * np.pi * 0.4))]),
}
out = {}
for name, (sym, m, bands) in cases.items():
    res = []
    for NODES in (2 * m + 40, 4 * m + 80):
        u, f, c = exact(sym, m, bands)
        res.append(taps(sym, m, u, f, c))
    d = np.max(np.abs(res[0] - res[1])) / np.max(np.abs(res[1]))
    print(name, "quadrature check", d, "max|h|", np.max(np.abs(res[1])), flush=True)
    out[name] = res[1]
np.savez(sys.argv[1], **out)
