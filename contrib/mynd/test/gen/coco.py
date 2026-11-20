# /// script
# requires-python = "==3.13.*"
# dependencies = ["numpy==2.5.5", "pycocotools==2.0.11"]
# ///
"""pycocotools 2.0.11's evaluation and polygon masks, as goldens for
mynd.coco (contrib/mynd/test/coco).

Run from this directory, with the script's own lock:

    uv run --locked --script coco.py --annotations FILE ../coco/golden
    uv run --locked --script coco.py --annotations FILE --check EXE --work DIR

FILE is COCO's instances_val2017.json, checked against its sha256; without
--annotations the script fetches COCO's 2017 annotations archive into
$XDG_CACHE_HOME/mynd-coco and extracts it there.

The first form writes the goldens:

- val24.json: the annotations of the first 24 val2017 images with a CC BY
  or CC BY-SA licence and at least one annotation (CC BY 4.0, the COCO
  Consortium), with COCO's info and licences;
- dets24.json: detections drawn near them, as a results file holding both
  bbox and segmentation (ties within and across images, duplicates, wrong
  categories, 130 detections of one category in one image);
- eval24.npz: pycocotools' precision and recall for boxes and masks (-1
  where undefined), the twelve means as math.fsum of the defined entries
  over their count, and each annotation's runs (annToRLE);
- polygons.npz: 1000 drawn and 15 named polygons, the val2017 polygons
  and drawn ones whose mask a fused multiply-add changes, and their masks'
  runs by rleFrPoly transcribed with each operation rounded on its own
  (frpoly), which on x86-64 the script asserts equal to pycocotools' before
  writing;
- tie.npz: a dataset whose AR@100 pycocotools prints as 0.313 and whose
  exactly rounded mean is 0.3125.

Before writing, it evaluates the fixture again under numpy 2.2.6, whose
reductions sum in another order, and asserts the tables bit-equal.

--check runs the evaluation on the whole of val2017 with detections drawn
the same way, through pycocotools and through EXE (test/coco/coco_eval.exe),
and compares every entry of the tables bit for bit; each --results FILE, a
model's results on val2017, is compared as well. It needs the 20 MB
annotation file, so it lives outside runtest.
"""
import argparse, contextlib, copy, hashlib, io, json, math, os, platform
import subprocess, sys, tempfile, urllib.request, zipfile
import numpy as np
from pycocotools import mask as mu
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

VAL_SHA256 = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
ARCHIVE = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
MEMBER = "annotations/instances_val2017.json"
FIXTURE_IMAGES = 24
OTHER_NUMPY = "2.2.6"
NAMED, DRAWN = 15, 1000


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def annotations(path):
    if path is None:
        cache = os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
            "mynd-coco")
        os.makedirs(cache, exist_ok=True)
        path = os.path.join(cache, "instances_val2017.json")
        if not os.path.exists(path):
            z = os.path.join(cache, "annotations_trainval2017.zip")
            if not os.path.exists(z):
                urllib.request.urlretrieve(ARCHIVE, z + ".part")
                os.rename(z + ".part", z)
            with zipfile.ZipFile(z) as zf, zf.open(MEMBER) as src, \
                    open(path + ".part", "wb") as dst:
                dst.write(src.read())
            os.rename(path + ".part", path)
    got = sha256(path)
    if got != VAL_SHA256:
        sys.exit(f"{path}: sha256 {got}, expected {VAL_SHA256}")
    with open(path) as f:
        return path, json.load(f)


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


# The fixture


def fixture(val):
    by_img = {}
    for a in val["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    ids = sorted(i["id"] for i in val["images"]
                 if i["license"] in (4, 5) and i["id"] in by_img)
    ids = set(ids[:FIXTURE_IMAGES])
    images = [i for i in val["images"] if i["id"] in ids]
    anns = [a for a in val["annotations"] if a["image_id"] in ids]
    cats = {a["category_id"] for a in anns}
    return {
        "info": val["info"],
        "licenses": val["licenses"],
        "images": images,
        "categories": [c for c in val["categories"] if c["id"] in cats],
        "annotations": anns,
    }


# Detections near the annotations, as design b's probe p3_gen draws them,
# with wrong categories among the dataset's own.


def f32(x):
    return float(np.float32(x))


def detections(data, seed, crowded):
    rng = np.random.default_rng(seed)
    imgs = {i["id"]: i for i in data["images"]}
    cats = sorted(c["id"] for c in data["categories"])
    by_img = {}
    for a in data["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    def score():
        if rng.random() < .5:
            return f32(np.round(rng.random(), 2))
        return f32(rng.random())

    dts = []
    for iid in sorted(imgs):
        h, w = imgs[iid]["height"], imgs[iid]["width"]
        for a in by_img.get(iid, []):
            if rng.random() < .15:
                continue
            x, y, bw, bh = a["bbox"]
            j = rng.normal(0, .08, 4) * [bw, bh, bw, bh]
            b = [f32(x + j[0]), f32(y + j[1]), f32(max(bw + j[2], 0)),
                 f32(max(bh + j[3], 0))]
            c = a["category_id"] if rng.random() < .9 else int(rng.choice(cats))
            d = {"image_id": iid, "category_id": c, "bbox": b, "score": score()}
            if isinstance(a["segmentation"], list) and rng.random() < .8:
                parts = [list(np.array(p) + rng.normal(0, 1.5, len(p)))
                         for p in a["segmentation"]]
                d["segmentation"] = mu.merge(mu.frPyObjects(parts, h, w))
            else:
                d["segmentation"] = mu.frPyObjects(np.array([b]), h, w)[0]
            dts.append(d)
            if rng.random() < .1:
                dts.append(copy.deepcopy(d))
        for _ in range(rng.integers(0, 12)):
            bw, bh = rng.uniform(1, w / 2), rng.uniform(1, h / 2)
            b = [f32(rng.uniform(0, w - bw)), f32(rng.uniform(0, h - bh)),
                 f32(bw), f32(bh)]
            d = {"image_id": iid, "category_id": int(rng.choice(cats)),
                 "bbox": b, "score": score()}
            d["segmentation"] = mu.frPyObjects(np.array([b]), h, w)[0]
            dts.append(d)
        if crowded(iid):
            c = int(rng.choice(cats))
            for _ in range(130):
                b = [f32(rng.uniform(0, w - 20)), f32(rng.uniform(0, h - 20)),
                     f32(20), f32(20)]
                dts.append({"image_id": iid, "category_id": c, "bbox": b,
                            "score": score(),
                            "segmentation": mu.frPyObjects(np.array([b]), h, w)[0]})
    for d in dts:
        d["segmentation"]["counts"] = d["segmentation"]["counts"].decode()
    return dts


def coco(data):
    with quiet():
        g = COCO()
        g.dataset = copy.deepcopy(data)
        g.createIndex()
    return g


# pycocotools' loadRes reads a results file as boxes when its first entry
# holds a bbox, and then ranges a mask by its box's area; mynd.coco ranges a
# mask by its pixel count, as pycocotools does for masks without a box. So
# the masks' bbox is dropped before pycocotools runs, here and for every
# --results file, and the tables compare under one rule.
def evaluate(data, dts, kind):
    drop = "segmentation" if kind == "bbox" else "bbox"
    res = [{k: v for k, v in d.items() if k != drop} for d in dts]
    g = coco(data)
    with quiet():
        dt = g.loadRes(copy.deepcopy(res))
        e = COCOeval(g, dt, kind)
        e.evaluate()
        e.accumulate()
        e.summarize()
    return e.eval["precision"], e.eval["recall"], e.stats


# The twelve means as exactly rounded sums of the defined entries.
def means(p, r):
    def mean(s):
        s = s[s > -1]
        return math.fsum(s.tolist()) / s.size if s.size else math.nan
    return np.array([
        mean(p[:, :, :, 0, 2]), mean(p[0, :, :, 0, 2]), mean(p[5, :, :, 0, 2]),
        mean(p[:, :, :, 1, 2]), mean(p[:, :, :, 2, 2]), mean(p[:, :, :, 3, 2]),
        mean(r[:, :, 0, 0]), mean(r[:, :, 0, 1]), mean(r[:, :, 0, 2]),
        mean(r[:, :, 1, 2]), mean(r[:, :, 2, 2]), mean(r[:, :, 3, 2])])


def tables(gt_file, dt_file, out):
    with open(gt_file) as f:
        data = json.load(f)
    with open(dt_file) as f:
        dts = json.load(f)
    t = {}
    for kind in ("bbox", "segm"):
        p, r, _ = evaluate(data, dts, kind)
        t[f"{kind}.precision"], t[f"{kind}.recall"] = p, r
    np.savez(out, **t)


# rleFrPoly (pycocotools 2.0.11, common/maskApi.c), each operation rounded
# on its own, or with each a * b + c fused as the arm64 wheels compile it.


def frpoly(xy, h, w, fused=False):
    k = len(xy) // 2

    def mad(a, b, c):
        return math.fma(a, b, c) if fused else a * b + c

    x = [int(mad(5.0, xy[2 * j], .5)) for j in range(k)]
    y = [int(mad(5.0, xy[2 * j + 1], .5)) for j in range(k)]
    x.append(x[0])
    y.append(y[0])
    u, v = [], []
    for j in range(k):
        xs, xe, ys, ye = x[j], x[j + 1], y[j], y[j + 1]
        dx, dy = abs(xe - xs), abs(ys - ye)
        flip = (dx >= dy and xs > xe) or (dx < dy and ys > ye)
        if flip:
            xs, xe, ys, ye = xe, xs, ye, ys
        if dx >= dy:
            s = (ye - ys) / dx if dx else 0.0
            for d in range(dx + 1):
                t = dx - d if flip else d
                u.append(t + xs)
                v.append(ys if dx == 0 else int(mad(s, t, ys) + .5))
        else:
            s = (xe - xs) / dy
            for d in range(dy + 1):
                t = dy - d if flip else d
                v.append(t + ys)
                u.append(int(mad(s, t, xs) + .5))
    pts = []
    for j in range(1, len(u)):
        if u[j] != u[j - 1]:
            xd = float(u[j] if u[j] < u[j - 1] else u[j] - 1)
            xd = (xd + .5) / 5.0 - .5
            if math.floor(xd) != xd or xd < 0 or xd > w - 1:
                continue
            yd = float(v[j] if v[j] < v[j - 1] else v[j - 1])
            yd = (yd + .5) / 5.0 - .5
            yd = 0 if yd < 0 else h if yd > h else yd
            pts.append(int(xd) * h + math.ceil(yd))
    pts.append(h * w)
    pts.sort()
    a, p = [], 0
    for t in pts:
        a.append(t - p)
        p = t
    out = [a[0]]
    i = 1
    while i < len(a):
        if a[i] > 0:
            out.append(a[i])
            i += 1
        else:
            i += 1
            if i < len(a):
                out[-1] += a[i]
                i += 1
    return out


def decode(counts, h, w):
    m = np.zeros(h * w, bool)
    at, val = 0, False
    for c in counts:
        m[at:at + c] = val
        at += c
        val = not val
    return m


def encode(m):
    counts, run, cur = [], 0, False
    for p in m:
        if p == cur:
            run += 1
        else:
            counts.append(run)
            run, cur = 1, p
    counts.append(run)
    return counts


# The counts of pycocotools' compressed RLE (rleFrString).
def runs_of(rle):
    s = rle["counts"]
    s = s.decode() if isinstance(s, bytes) else s
    if not isinstance(s, str):
        return list(s)
    cnts, p = [], 0
    while p < len(s):
        x, k, more = 0, 0, True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1f) << 5 * k
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << 5 * k
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    return cnts


def union_counts(parts, h, w, fused=False):
    m = np.zeros(h * w, bool)
    for p in parts:
        m |= decode(frpoly(p, h, w, fused), h, w)
    return encode(m)


def polygons(val):
    rng = np.random.default_rng(3)
    x86 = platform.machine() in ("x86_64", "AMD64")
    cases = []

    def add(parts, h, w):
        cases.append((parts, h, w))

    H, W = 23, 31
    add([[2.2, 3.1, 20.7, 5.5, 9.9, 19.4]], H, W)
    add([[2, 2, 20, 18, 20, 2, 2, 18]], H, W)                      # bowtie
    add([[1, 1, 10, 10, 20, 20]], H, W)                            # collinear
    add([[3, 3, 3, 3, 15, 4, 8, 14]], H, W)                        # repeated
    add([[5, 5, 5, 5, 5, 5]], H, W)                                # one point
    add([[-0.15, -0.15, 12, -0.05, 12, 9, -0.19, 9]], H, W)        # trunc at 0
    add([[-50, -40, 80, -30, 90, 70, -20, 60]], H, W)              # outside
    add([[2, 2, 14, 2, 14, 14, 2, 14], [8, 8, 20, 8, 20, 20, 8, 20]], H, W)
    add([[2.5, 2.5, 10.5, 2.5, 10.5, 7.5, 2.5, 7.5]], H, W)
    add([[2.1, 2.3, 10.3, 2.1, 10.1, 7.3, 2.3, 7.1]], H, W)
    add([[0, 0, 1, 0, 1, 1, 0, 1]], 1, 1)
    add([[2, 2, 14, 2, 14, 14], [3, 3, 9, 9]], H, W)               # 2 vertices
    add([[0.3, 0.3, 0.5, 0.3, 0.4, 0.6]], H, W)                    # sub-pixel
    add([[-0.3, 4, 6, -0.25, 6, 8]], H, W)          # trunc and floor differ
    add([[1, 1, 30, 1, 30, 1.2, 1, 1.2]], H, W)                    # thin
    for t in range(DRAWN):
        k = int(rng.integers(3, 12))
        h, w = int(rng.integers(1, 40)), int(rng.integers(1, 40))
        kind = t % 5
        if kind == 0:     # often self-intersecting, partly outside
            xy = rng.uniform([-5, -5], [w + 5, h + 5], (k, 2))
        elif kind == 1:   # star-shaped
            c = rng.uniform([0, 0], [w, h])
            r = rng.uniform(0.2, 12, k)
            a = np.sort(rng.uniform(0, 2 * np.pi, k))
            xy = np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)], 1)
        elif kind == 2:   # the 1/10 grid: 5x + 1/2 lands on halves
            xy = np.round(rng.uniform([-2, -2], [w + 2, h + 2], (k, 2)) * 10) / 10
        elif kind == 3:   # integer and half-integer vertices
            xy = np.round(rng.uniform([-1, -1], [w + 1, h + 1], (k, 2)) * 2) / 2
        else:             # near zero, where truncation and floor differ
            xy = rng.uniform(-0.5, 3, (k, 2))
        add([[float(v) for v in xy.ravel()]], h, w)
    # val2017 polygons whose mask a fused multiply-add changes.
    imgs = {i["id"]: i for i in val["images"]}
    for a in val["annotations"]:
        if a["id"] in (1213103, 1492903):
            im = imgs[a["image_id"]]
            for p in a["segmentation"]:
                if frpoly(p, im["height"], im["width"]) != \
                        frpoly(p, im["height"], im["width"], True):
                    add([p], im["height"], im["width"])
    # Drawn polygons of two-decimal coordinates whose mask fusion changes.
    found = 0
    while found < 4:
        k = int(rng.integers(3, 12))
        xy = [round(float(rng.uniform(-5, 650)), 2) if i % 2 == 0 else
              round(float(rng.uniform(-5, 490)), 2) for i in range(2 * k)]
        if frpoly(xy, 480, 640) != frpoly(xy, 480, 640, True):
            add([xy], 480, 640)
            found += 1
    fused_cases = len(cases) - DRAWN - NAMED
    assert fused_cases >= 6, fused_cases
    out = {"height": [], "width": [], "parts": [], "lengths": [], "coords": [],
           "counts_lengths": [], "counts": []}
    for parts, h, w in cases:
        expect = union_counts(parts, h, w)
        if x86:
            r = mu.merge(mu.frPyObjects([p for p in parts if len(p) >= 6] or
                                        [[0, 0, 0, 0, 0, 0]], h, w))
            assert encode(decode(runs_of(r), h, w)) == expect, parts
        out["height"].append(h)
        out["width"].append(w)
        out["parts"].append(len(parts))
        for p in parts:
            out["lengths"].append(len(p))
            out["coords"].extend(float(v) for v in p)
        out["counts_lengths"].append(len(expect))
        out["counts"].extend(expect)
    # A polygon of fewer than three vertices covers nothing.
    for _ in range(200):
        k = int(rng.integers(1, 3))
        xy = [float(v) for v in rng.uniform(-3, 30, 2 * k)]
        assert sum(frpoly(xy, H, W)[1::2]) == 0, xy
    res = {k: np.array(v, np.float64 if k == "coords" else np.int64)
           for k, v in out.items()}
    res["fused"] = np.array([fused_cases], np.int64)
    return res


# Skeptic probe s3: 8 categories of 5 annotations, one box detection each,
# whose AR@100 has entries summing to 25/80 = 0.3125 exactly in rationals.


def tie():
    import random
    K, N, T = 8, 5, 10
    random.seed(3)

    def entries(L):
        r = np.zeros((T, K))
        for c in range(K):
            for t in range(T):
                r[t, c] = sum(1 for j in range(N) if L[c][j] > t) / N
        return r

    found = None
    for _ in range(200000):
        L = [[random.randrange(0, 7) for _ in range(N)] for _ in range(K)]
        if sum(map(sum, L)) != 125:
            continue
        s = entries(L).ravel()
        a, b = float(np.mean(s)), math.fsum(s.tolist()) / s.size
        if f"{a:0.3f}" != f"{b:0.3f}":
            found = L
            break
    assert found
    thr = np.linspace(.5, 0.95, 10)
    images, anns, dets, dxs = [], [], [], []
    for c in range(K):
        for j in range(N):
            iid = c * N + j + 1
            images.append({"id": iid, "height": 200, "width": 300,
                           "file_name": f"{iid}.jpg"})
            anns.append({"id": iid, "image_id": iid, "category_id": c + 1,
                         "bbox": [0, 0, 100, 100], "area": 10000, "iscrowd": 0})
            L = found[c][j]
            q = 0.3 if L == 0 else (thr[L - 1] + (thr[L] if L < 10 else 1.0)) / 2
            dx = float(100 * (1 - q) / (1 + q))
            dxs.append(dx)
            dets.append({"image_id": iid, "category_id": c + 1,
                         "bbox": [dx, 0, 100, 100], "score": 0.9})
    data = {"images": images, "annotations": anns,
            "categories": [{"id": c + 1, "name": str(c)} for c in range(K)]}
    p, r, stats = evaluate(data, dets, "bbox")
    m = means(p, r)
    assert f"{stats[8]:0.3f}" == "0.313" and f"{m[8]:0.3f}" == "0.312"
    return {"dx": np.array(dxs), "recall": r, "pycocotools_ar100":
            np.array([stats[8]]), "ar100": np.array([m[8]])}


def dump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
        f.write("\n")


def write(val, out):
    data = fixture(val)
    first = min(i["id"] for i in data["images"])
    dts = detections(data, 7, lambda iid: iid == first)
    dump(os.path.join(out, "val24.json"), data)
    dump(os.path.join(out, "dets24.json"), dts)
    g = coco(data)
    res = {}
    for kind in ("bbox", "segm"):
        p, r, _ = evaluate(data, dts, kind)
        res[f"{kind}.precision"], res[f"{kind}.recall"] = p, r
        res[f"{kind}.means"] = means(p, r)
    # The same tables under a numpy that sums in another order.
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        other = os.path.join(tmp, "other.npz")
        subprocess.run(
            ["uv", "run", "-q", "--no-project", "--python", "3.13",
             "--with", f"numpy=={OTHER_NUMPY}", "--with", "pycocotools==2.0.11",
             "python", "-I", os.path.abspath(__file__), "--tables",
             os.path.join(out, "val24.json"), os.path.join(out, "dets24.json"),
             other], check=True, stdin=subprocess.DEVNULL)
        o = np.load(other)
        for k in o.files:
            assert o[k].tobytes() == res[k].tobytes(), k
        print(f"numpy {OTHER_NUMPY}: {len(o.files)} tables bit-equal",
              file=sys.stderr)
    lengths, counts = [], []
    for a in data["annotations"]:
        c = runs_of(g.annToRLE(a))
        lengths.append(len(c))
        counts.extend(c)
    res["runs.lengths"] = np.array(lengths, np.int64)
    res["runs.counts"] = np.array(counts, np.int64)
    np.savez_compressed(os.path.join(out, "eval24.npz"), **res)
    np.savez_compressed(os.path.join(out, "polygons.npz"), **polygons(val))
    np.savez_compressed(os.path.join(out, "tie.npz"), **tie())


def check(gt, val, exe, work, results):
    os.makedirs(work, exist_ok=True)
    dts = detections(val, 7, lambda iid: iid % 97 == 0)
    dt = os.path.join(work, "dets_val2017.json")
    dump(dt, dts)
    print(f"val2017: {len(val['images'])} images, {len(dts)} detections",
          file=sys.stderr)
    runs = [(dt, dts, "bbox"), (dt, dts, "segm")]
    for f in results:
        with open(f) as h:
            res = json.load(h)
        runs.append((f, res, "segm" if "segmentation" in res[0] else "bbox"))
    bad = 0
    for dt, dts, kind in runs:
        print(f"{os.path.basename(dt)}, {kind}:", file=sys.stderr)
        p, r, _ = evaluate(val, dts, kind)
        out = os.path.join(work, f"ocaml_{kind}.npz")
        subprocess.run([exe, gt, dt, kind, out], check=True,
                       stdin=subprocess.DEVNULL)
        o = np.load(out)
        for name, want in (("precision", p), ("recall", r)):
            want = np.where(want == -1, np.nan, want)
            got = o[name]
            assert got.shape == want.shape, (name, got.shape, want.shape)
            # Equal when both are NaN, whatever their payloads, and
            # otherwise when their bits are.
            nan = np.isnan(got) & np.isnan(want)
            bits = got.view(np.uint64) != want.view(np.uint64)
            differ = int(np.sum(bits & ~nan))
            print(f"{kind} {name}: {differ} of {want.size} entries differ in "
                  f"bits", file=sys.stderr)
            bad += differ
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations")
    ap.add_argument("--check", metavar="EXE")
    ap.add_argument("--work")
    ap.add_argument("--results", action="append", default=[],
                    help="a model's results file to compare as well")
    ap.add_argument("--tables", nargs=3, metavar=("GT", "DT", "OUT"))
    ap.add_argument("out", nargs="?")
    a = ap.parse_args()
    if a.tables:
        tables(*a.tables)
        return
    path, val = annotations(a.annotations)
    if a.check:
        if not a.work:
            sys.exit("--check needs --work DIR")
        check(path, val, a.check, a.work, a.results)
    else:
        write(val, a.out)


if __name__ == "__main__":
    main()
