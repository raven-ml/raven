# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "numpy==2.5.5",
#   "safetensors==0.8.0",
# ]
# ///
"""Fixtures for test_parity: the scalar codecs' codebooks and upstream's goldens.

    uv run test/gen/scalars.py [--whole] CODECS GOLDENS_ST test/upstream

CODECS is polymathic-ai/aion-base's codecs/ directory at 40541618 (the HF
snapshot), GOLDENS_ST upstream's tests/test_data goldens converted to
safetensors by goldens.py. Writes:

- OUT/edges.safetensors: each scalar codec's codebooks as one float32
  [c; 1024] entry under its codec name, for the XP codecs those of the kept
  coefficients alone. Only the codebooks: the reservoirs, 100,000 floats per
  channel, serve training alone.
- OUT/goldens.safetensors: for each scalar codec, <name>.input,
  <name>.encoded and <name>.decoded as upstream stored them. For
  bp_coefficients and rp_coefficients, only the rows <name>.rows (their
  indices in upstream's 1024) and the coefficients <name>.coefficients.

An XP codec computes each coefficient alike, so its fixture keeps the
coefficients where the goldens can tell a wrong edge from rounding:
  - near: a codebook edge lies where nx's float32 asinh(asinh x) may land
    (each asinh within 3 floats of the exact value of its input, rounded, as
    nx.mli states and apps/aion's parity test assumes), where a token may
    flip;
and the lowest coefficient whose token is 0 (first) and 1023 (last) in some
row, and coefficient 5: upstream's float32 asinh put coefficient 5 of bp's
row 20 one bin above the correctly rounded value. Of those, each
(coefficient, label) pair that some row reaches must be reached by a kept
row; row 20 is kept whatever it covers. Rows are added greedily, the one
reaching most uncovered pairs first, lowest index on ties.

With --whole the XP goldens and codebooks are kept whole: test_parity.exe
run from a directory whose upstream/ holds those files checks every row and
coefficient (2.2 MB more, outside the repository).

Each output's metadata names this script and the SHA-256 of every input.
"""

import hashlib
import json
import os
import sys

import numpy as np
from safetensors.numpy import load_file, save_file

CODECS = [
    "SHAPE_E1", "SHAPE_E2", "EBV", "Z", "a_g", "a_r", "a_i", "a_z", "a_y",
    "g_cmodel_mag", "r_cmodel_mag", "i_cmodel_mag", "z_cmodel_mag",
    "y_cmodel_mag", "i_sdssshape_shape11", "i_sdssshape_shape22",
    "i_sdssshape_shape12", "ra", "dec", "FLUX_G", "FLUX_R", "FLUX_I",
    "FLUX_Z", "FLUX_W1", "FLUX_W2", "FLUX_W3", "FLUX_W4", "SHAPE_R",
    "phot_g_mean_flux", "phot_bp_mean_flux", "phot_rp_mean_flux", "parallax",
    "bp_coefficients", "rp_coefficients",
]
XP = ("bp_coefficients", "rp_coefficients")
KEEP_ROWS = [20]
KEEP_COEFFICIENT = 5
SCRIPT = "apps/aion/test/gen/scalars.py"


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def codebooks(path):
    d = load_file(path)
    if "_quantizer._index_to_val" in d:
        return d["_quantizer._index_to_val"][None, :]
    if "_quantizer.buckets" in d:
        return d["_quantizer.buckets"][None, :]
    rows, i = [], 0
    while f"_quantizer.quantizers.{i}._index_to_val" in d:
        rows.append(d[f"_quantizer.quantizers.{i}._index_to_val"])
        i += 1
    return np.stack(rows)


def move(k, v):
    """The float32 k floats from the float32 v."""
    v = np.float32(v)
    target = np.float32(np.inf if k > 0 else -np.inf)
    for _ in range(abs(k)):
        v = np.nextafter(v, target)
    return float(v)


def landing(x):
    """Where nx's float32 asinh(asinh x) may land: [lo, hi]."""
    lo1, hi1 = move(-3, np.arcsinh(x)), move(3, np.arcsinh(x))
    return move(-3, np.arcsinh(lo1)), move(3, np.arcsinh(hi1))


def labels(x, edges, tokens):
    """The (coefficient, label) pairs of each row of an XP golden."""
    rows = []
    for r in range(x.shape[0]):
        s = set()
        for j in range(x.shape[1]):
            e = edges[j].astype(np.float64)
            lo, hi = landing(float(x[r, j]))
            if np.any((e >= lo) & (e <= hi)):
                s.add((j, "near"))
            if tokens[r, j] == 0:
                s.add((j, "first"))
            if tokens[r, j] == edges.shape[1] - 1:
                s.add((j, "last"))
        rows.append(s)
    return rows


def coefficients(rows):
    """The kept coefficients: those near an edge, the first reaching each end, and
    KEEP_COEFFICIENT."""
    pairs = set().union(*rows)
    keep = {j for j, label in pairs if label == "near"} | {KEEP_COEFFICIENT}
    for end in ("first", "last"):
        ends = sorted(j for j, label in pairs if label == end)
        if ends and not keep.intersection(ends):
            keep.add(ends[0])
    return sorted(keep)


def select(rows):
    keep = list(KEEP_ROWS)
    covered = set().union(*(rows[r] for r in keep))
    universe = set().union(*rows)
    while covered != universe:
        best = max(range(len(rows)), key=lambda r: (len(rows[r] - covered), -r))
        keep.append(best)
        covered |= rows[best]
    return sorted(keep), universe


def main(codecs_dir, goldens_dir, out, whole=False):
    os.makedirs(out, exist_ok=True)
    edges, edge_inputs = {}, []
    for name in CODECS:
        path = os.path.join(codecs_dir, name, "model.safetensors")
        edges[name] = np.ascontiguousarray(codebooks(path), dtype=np.float32)
        edge_inputs.append(f"{sha(path)}  codecs/{name}/model.safetensors")

    goldens, golden_inputs = {}, []
    for name in CODECS:
        parts = {}
        for part in ("input", "encoded", "decoded"):
            file = f"{name}_codec_{part}_batch.safetensors"
            path = os.path.join(goldens_dir, file)
            parts[part] = load_file(path)["value"]
            golden_inputs.append(f"{sha(path)}  {file}")
        if name in XP and not whole:
            cols = coefficients(labels(parts["input"], edges[name], parts["encoded"]))
            edges[name] = np.ascontiguousarray(edges[name][cols])
            for part in parts:
                parts[part] = parts[part][:, cols]
            rows = labels(parts["input"], edges[name], parts["encoded"])
            keep, universe = select(rows)
            print(f"{name}: coefficients {cols}; {len(keep)} rows of {len(rows)}"
                  f" reach {len(universe)} pairs: {keep}")
            for part in parts:
                parts[part] = parts[part][keep]
            goldens[f"{name}.rows"] = np.array(keep, dtype=np.int64)
            goldens[f"{name}.coefficients"] = np.array(cols, dtype=np.int64)
        for part, v in parts.items():
            goldens[f"{name}.{part}"] = np.ascontiguousarray(v)
    meta = {"generator": SCRIPT, "inputs": "\n".join(edge_inputs)}
    save_file(edges, os.path.join(out, "edges.safetensors"), metadata=meta)
    meta = {"generator": SCRIPT, "inputs": "\n".join(golden_inputs)}
    save_file(goldens, os.path.join(out, "goldens.safetensors"), metadata=meta)
    for f in ("edges.safetensors", "goldens.safetensors"):
        p = os.path.join(out, f)
        print(f, os.path.getsize(p), sha(p))


if __name__ == "__main__":
    args = sys.argv[1:]
    whole = "--whole" in args
    main(*[a for a in args if a != "--whole"], whole=whole)
