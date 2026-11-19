# Orders SciPy 1.16.3's buttord, cheb1ord, cheb2ord and ellipord give for the
# specifications of test_designs.ml's Iir.meet cases (rate 1), printed as OCaml.
# Run from this directory: uv run --locked meet_orders.py
from scipy import signal
specs = [
    ("lp", (0.1,), (0.15,), 1.0, 60.0),
    ("lp", (0.001,), (0.0015,), 0.1, 80.0),
    ("lp", (1e-4,), (1.1e-4,), 0.1, 120.0),
    ("hp", (0.3,), (0.25,), 0.5, 50.0),
    ("hp", (0.45,), (0.44,), 0.1, 70.0),
    ("bp", (0.15, 0.25), (0.1, 0.32), 1.0, 40.0),
    ("bs", (0.08, 0.3), (0.12, 0.2), 0.5, 45.0),
    ("bs", (0.1, 0.4), (0.2, 0.22), 1.0, 30.0),
]
fns = [("`Butterworth", signal.buttord), ("`Chebyshev1", signal.cheb1ord),
       ("`Chebyshev2", signal.cheb2ord), ("`Elliptic", signal.ellipord)]
for kind, wp, ws, gp, gs in specs:
    row = []
    for name, f in fns:
        n, _ = f(list(wp) if len(wp) > 1 else wp[0], list(ws) if len(ws) > 1 else ws[0], gp, gs, fs=1.0)
        row.append(str(int(n)))
    print(kind, wp, ws, gp, gs, "->", " ".join(row))
