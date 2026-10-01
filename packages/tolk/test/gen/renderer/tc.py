"""Goldens of tinygrad/renderer/tc.py: the tensor cores of each target, what
each derives from its fragments, and the fragments a tensor core refuses.

A list of tile bits is written as its bits separated by spaces (`k1 m0 m1`), a
fragment as its lanes and its elements separated by a slash (`k1 m0/k0`), a
relabelling as its pairs (`k1>n1 m0>m0`), and the coordinates of a fragment as
its lanes separated by ` | `, each the coordinates of its elements (`0,0 0,1`).
"""

import json

from golden import table
from tinygrad.dtype import dtypes
from tinygrad.renderer import tc
from tinygrad.renderer.tc import TensorCore

TARGETS = ["cuda_sm75", "cuda_sm80", "cuda_sm89", "amd_rdna3", "amd_rdna4", "amd_cdna3", "amd_cdna4", "metal"]


def bits(bs):
    return " ".join(bs)


def fragment(f):
    return f"{bits(f[0])}/{bits(f[1])}"


def relabelling(r):
    return " ".join(f"{c}>{y}" for c, y in r.items())


def coords(lanes):
    return " | ".join(" ".join(f"{a},{b}" for a, b in elements) for elements in lanes)


def cores():
    return [(target, i, core) for target in TARGETS for i, core in enumerate(getattr(tc, target))]


@table
def tensor_cores():
    columns = ["core", "repr", "dtype_in", "dtype_out", "frag_a", "frag_b", "frag_c", "dims", "threads",
               "axis_coords", "base_upcast_axes", "relabel_a", "relabel_b"]
    rows = []
    for target, i, core in cores():
        relabel_a, relabel_b = core.relabel()
        rows.append((f"{target}[{i}]", repr(core), core.dtype_in, core.dtype_out, fragment(core.frag_a),
                     fragment(core.frag_b), fragment(core.frag_c), " ".join(map(str, core.dims)), core.threads,
                     bits(core.axis_coords()), bits(core.base_upcast_axes()), relabelling(relabel_a),
                     relabelling(relabel_b)))
    return columns, rows


@table
def frag_coords():
    columns = ["core", "operand", "coords"]
    rows = [(f"{target}[{i}]", operand, coords(lanes))
            for target, i, core in cores() for operand, lanes in zip("abc", core.frag_coords())]
    return columns, rows


def target_of(cores):
    return next(target for target in TARGETS if getattr(tc, target) is cores)


def outcome(get, arch):
    """The name of the list `get(arch)` is, `none` for no tensor core, or the
    exception it raises."""
    try:
        cores = get(arch)
    except Exception as e:
        return f"raises {type(e).__name__}"
    return target_of(cores) if cores else "none"


@table
def cuda():
    archs = ["sm_", "sm_x", "sm_7", "sm_70", "sm_74", "sm_75", "sm_79", "sm_80", "sm_86", "sm_88", "sm_89", "sm_90", "sm_90a",
             "sm_120", "gfx89"]
    return ["arch", "cores"], [(json.dumps(arch), outcome(tc.get_cuda, arch)) for arch in archs]


@table
def amd():
    archs = ["gfx942", "gfx950", "gfx1200", "gfx1201", "gfx1100", "gfx1151", "gfx90a", "gfx94", "gfx9420", ""]
    return ["arch", "cores"], [(json.dumps(arch), outcome(tc.get_amd, arch)) for arch in archs]


def variant(name, base, **frags):
    """The fragments of `base` with some replaced, each a pair of tuples."""
    return (name, *(frags.get(k, getattr(base, k)) for k in ("frag_a", "frag_b", "frag_c")))


def refusals():
    metal, rdna3, cdna4 = tc.metal[0], tc.amd_rdna3[0], tc.amd_cdna4[0]
    a, b, c = metal.frag_a, metal.frag_b, metal.frag_c
    return [
        variant("metal as it is", metal),
        variant("rdna3 broadcasts an N bit across A's lanes", rdna3),
        variant("cdna4 holds a high K bit in an element", cdna4),
        variant("A has a lane fewer than C", metal, frag_a=(a[0][1:], a[1])),
        variant("B has a lane more than C", metal, frag_b=(b[0] + ("m3",), b[1])),
        variant("A holds a bit twice", metal, frag_a=(("k1", "m0", "m0", "k2", "m2"), a[1])),
        variant("A holds a bit twice besides every bit of its tile", metal, frag_a=(a[0], ("k0", "m0"))),
        variant("A holds a bit as a lane and as an element", metal, frag_a=(("k1", "m0", "m1", "k0", "m2"), a[1])),
        variant("A lacks an element", metal, frag_a=(a[0], ())),
        variant("A's element is an N bit", metal, frag_a=(a[0], ("n0",))),
        variant("B's element is an M bit", metal, frag_b=(b[0], ("m0",))),
        variant("C's element is a K bit", metal, frag_c=(c[0], ("k0",))),
        variant("C's lane is a K bit", metal, frag_c=(("n1", "m0", "m1", "k2", "m2"), c[1])),
        variant("B names an N bit beyond the tile", metal, frag_b=(("n1", "k0", "k1", "n3", "k2"), b[1])),
        variant("A names a negative bit", metal, frag_a=(("k1", "m0", "m1", "k2", "m-1"), a[1])),
        variant("A and B order K differently", metal, frag_a=(("k2", "m0", "m1", "k1", "m2"), a[1])),
        variant("B trades an N bit for an M bit", metal, frag_b=(("n1", "k0", "k1", "m0", "k2"), b[1])),
        variant("A broadcasts an N bit that C lacks", metal, frag_a=(("k1", "m0", "m1", "k2", "n3"), ("k0", "m2"))),
    ]


@table
def refused():
    rows = []
    for name, fa, fb, fc in refusals():
        try:
            TensorCore(dtype_in=dtypes.half, dtype_out=dtypes.float, frag_a=fa, frag_b=fb, frag_c=fc)
            accepted = True
        except AssertionError:
            accepted = False
        rows.append((name, fragment(fa), fragment(fb), fragment(fc), accepted))
    return ["case", "frag_a", "frag_b", "frag_c", "accepted"], rows
