# Goldens for Alda.Window: scipy.signal.windows at float64.
# Run from this directory: uv run --locked windows.py ../golden/windows.npz
# Keys: "<kind>.<periodic|symmetric>.<n>", kind one of hann, hamming,
# blackman, kaiser<beta>, gaussian<sigma>, cosine (a = 0.3, 0.4, 0.2, 0.1).
import sys

import numpy as np
import scipy
from scipy.signal import windows as w

out = sys.argv[1]
sizes = [0, 1, 2, 3, 4, 5, 16, 17, 400, 401]
kinds = {
    "hann": lambda n, sym: w.hann(n, sym=sym),
    "hamming": lambda n, sym: w.hamming(n, sym=sym),
    "blackman": lambda n, sym: w.blackman(n, sym=sym),
    "cosine": lambda n, sym: w.general_cosine(n, [0.3, 0.4, 0.2, 0.1], sym=sym),
    "kaiser0": lambda n, sym: w.kaiser(n, 0.0, sym=sym),
    "kaiser8.6": lambda n, sym: w.kaiser(n, 8.6, sym=sym),
    "kaiser14": lambda n, sym: w.kaiser(n, 14.0, sym=sym),
    "gaussian1.5": lambda n, sym: w.gaussian(n, 1.5, sym=sym),
    "gaussian7": lambda n, sym: w.gaussian(n, 7.0, sym=sym),
}
arrays = {}
for k, f in kinds.items():
    for sym, name in ((False, "periodic"), (True, "symmetric")):
        for n in sizes:
            a = np.asarray(f(n, sym), dtype=np.float64)
            assert a.shape == (n,)
            arrays[f"{k}.{name}.{n}"] = a
np.savez(out, **arrays)
print("scipy", scipy.__version__, len(arrays), "arrays")
