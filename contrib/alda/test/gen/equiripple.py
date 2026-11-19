# golden/equiripple.npz for test_taps.ml: SciPy 1.16.3's remez (Parks and
# McClellan's program, grid density 16) on designs it converges on; weights
# are the reciprocals of alda's tolerances.
# Run from this directory: uv run --locked equiripple.py ../golden/equiripple.npz
import sys, numpy as np
from scipy.signal import remez
out = {}
out["lp51"] = remez(51, [0, 0.1, 0.15, 0.5], [1, 0], weight=[1, 10], fs=1.0)
out["lp50"] = remez(50, [0, 0.1, 0.15, 0.5], [1, 0], weight=[1, 10], fs=1.0)
out["hilbert31"] = remez(31, [0.05, 0.45], [1], type="hilbert", fs=1.0)
out["hilbert30"] = remez(30, [0.05, 0.5], [1], type="hilbert", fs=1.0)
for k, v in out.items(): print(k, len(v), flush=True)
np.savez(sys.argv[1], **out)
