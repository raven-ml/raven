# golden/digital.npz for test_designs.ml: SciPy 1.16.3's zpk2sos (nearest
# pairing) of digital zeros, poles and gains.
# Run from this directory: uv run --locked digital.py ../golden/digital.npz
import sys, numpy as np
from scipy.signal import zpk2sos
cases = {
    # an all-pole filter, poles inside and two outside the unit circle
    "allpole": ([], [0.9 * np.exp(0.3j), 0.9 * np.exp(-0.3j), 0.5, -0.7, 1.2 * np.exp(2.0j), 1.2 * np.exp(-2.0j)], 0.25),
    # more zeros than poles, zeros on the circle
    "fir_like": ([np.exp(1.0j), np.exp(-1.0j), -1.0, 0.3, 2.0], [0.6, 0.2 + 0.4j, 0.2 - 0.4j], -1.5),
    # odd counts
    "odd": ([0.1, -0.9 + 0.1j, -0.9 - 0.1j], [0.95, 0.5 + 0.5j, 0.5 - 0.5j], 3.0),
}
out = {}
for name, (z, p, k) in cases.items():
    out[name + ".zeros"] = np.array(z, dtype=complex)
    out[name + ".poles"] = np.array(p, dtype=complex)
    out[name + ".gain"] = np.array([k])
    out[name + ".sos"] = zpk2sos(np.array(z, dtype=complex), np.array(p, dtype=complex), k, pairing="nearest")
    print(name, out[name + ".sos"].shape, flush=True)
np.savez(sys.argv[1], **out)
