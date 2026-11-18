# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.4.1", "pyproj==3.8.0"]
# ///
"""Print the reference values test_crs.ml holds in its tables.

    uv run contrib/jord/test/gen/crs_values.py

Each section is named after its table and prints its rows as the test
writes them:

origins
    k0 M(φ₀), M the meridian arc to φ₀ on the ellipsoid's double flattening
    (1 / 298.257223563 and 1 / 298.257222101 as float64), at the double φ₀
    and k0, by mpmath at 50 digits: M(φ) = a (1 - e²) ∫₀^φ (1 - e² sin² t)^(-3/2) dt.
proj_datums, proj_albers
    PROJ's values (pyproj's Transformer.from_crs(src, dst, always_xy=True)),
    each from the system's own geographic system (4269, 4167, 4283), so no
    datum step is taken.
k0_a
    k0 A for each k0, A Karney's eq. (29) to n¹⁰ in exact rationals from the
    double flattening, times the double k0, rounded once.
"""

from fractions import Fraction as F

import mpmath as mp
import pyproj
from pyproj import Transformer

ELLIPSOIDS = (("Crs.Wgs84", 298.257223563), ("Crs.Nad83", 298.257222101))


def meridian_arc(invf, lat):
    mp.mp.dps = 50
    f = mp.mpf(1 / invf)  # the float64 flattening, as jord computes it
    e2 = f * (2 - f)
    phi = mp.mpf(lat) * mp.pi / 180  # radians of the double latitude
    g = lambda t: (1 - e2 * mp.sin(t) ** 2) ** mp.mpf(-1.5)
    return mp.mpf(6378137) * (1 - e2) * mp.quad(g, [0, phi])


def origins():
    print("origins")
    for name, invf in ELLIPSOIDS:
        for lat, k0 in ((49.0, 0.9996012717), (-41.0, 0.9999), (89.5, 1.0)):
            m = meridian_arc(invf, lat) * mp.mpf(k0)
            print(f"    ({name}, {lat!r}, {k0!r}, {mp.nstr(m, 25)});")


def proj(name, cases):
    print(name, f"(PROJ {pyproj.__proj_version__}, pyproj {pyproj.__version__})")
    for (src, dst), pts in cases:
        print(f"  {dst} from {src}")
        t = Transformer.from_crs(src, dst, always_xy=True)
        for lon, lat in pts:
            x, y = t.transform(lon, lat)
            print(f"    ({lon!r}, {lat!r}, {x!r}, {y!r});")


def karney_a(invf):
    f = F(1.0 / invf)
    n = f / (2 - f)
    n2 = n * n
    s = 1 + n2 / 4 + n2**2 / 64 + n2**3 / 256 + 25 * n2**4 / 16384 + 49 * n2**5 / 65536
    return F(6378137) / (1 + n) * s


def k0_a():
    print("k0_a")
    for name, invf in ELLIPSOIDS:
        a = karney_a(invf)
        print(f"  {name}")
        for k0 in (0.9996, 0.9999, 1.0, 0.9996012717):
            print(f"    ({k0!r}, {float(F(k0) * a).hex()});")


def main():
    origins()
    proj("proj_datums", [
        ((4269, 26917), [(-81.0, 0.0), (-81.0, 40.0), (-80.34, 26.03),
                         (-75.0, 45.0), (-87.0, 30.0), (-60.0, 60.0)]),
        ((4269, 26904), [(-157.86, 21.31), (-159.0, 22.0)]),
        ((4167, 2193), [(173.0, -41.0), (174.78, -41.29), (166.5, -46.0),
                        (178.5, -37.7), (-176.5, -44.0), (172.0, -34.4)]),
    ])
    k0_a()
    proj("proj_albers", [
        ((4269, 5070), [(-96.0, 23.0), (-77.04, 38.9), (-122.42, 37.77),
                        (-70.0, 45.0), (-150.0, 61.2), (-80.34, 26.03)]),
        ((4283, 3577), [(132.0, 0.0), (151.21, -33.87), (115.86, -31.95),
                        (147.3, -42.88), (130.84, -12.46), (132.0, -27.0)]),
    ])


if __name__ == "__main__":
    main()
