# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.15.0", "numpy==2.5.4"]
# ///
"""Record the released models' embeddings of tiles with TESSERA's own code.

    uv run test/gen/release.py UPSTREAM WEIGHTS FIXTURES

UPSTREAM is github.com/ucam-eo/tessera at e368539f7c4711e5f16fdd09801573830e34baf9,
WEIGHTS holds student_{nano,small,medium,large}.pt and tessera_v2_2B_teacher.pt
from huggingface.co/geotessera, and FIXTURES/tiles/* are tiles in TESSERA's
files. For each tile and each model present, tessera_infer_v2's encode_tile on
the CPU in float32 writes FIXTURES/emb/TILE/MODEL.npy, NaN where a pixel has no
observation, and for a student its int8 codes and scales of those pixels,
MODEL_int8.npy and MODEL_scales.npy. The teacher takes batches of 512 pixels,
the students 4096, as infer_v2.py does.
"""
import os, sys
import numpy as np
import torch

UPSTREAM, WEIGHTS, FIXTURES = sys.argv[1:4]
sys.path.insert(0, os.path.join(UPSTREAM, "tessera_infer_v2"))
import infer_v2
import student.model as smodel, student.infer as sinfer, student.quantize as squant
import teacher.model as tmodel, teacher.infer as tinfer

MODELS = {m: (f"student_{m}.pt", smodel, sinfer, 4096) for m in ("nano", "small", "medium", "large")}
MODELS["teacher"] = ("tessera_v2_2B_teacher.pt", tmodel, tinfer, 512)
cpu = torch.device("cpu")
for tile in sorted(os.listdir(os.path.join(FIXTURES, "tiles"))):
    d = infer_v2.load_tile(os.path.join(FIXTURES, "tiles", tile))
    observed = ((d["s2_masks"].sum(0) > 0) | (d["s1_asc_bands"] != 0).any(-1).any(0)
                | (d["s1_desc_bands"] != 0).any(-1).any(0))
    out = os.path.join(FIXTURES, "emb", tile)
    os.makedirs(out, exist_ok=True)
    for name, (file, model, infer, batch) in MODELS.items():
        if not os.path.exists(os.path.join(WEIGHTS, file)):
            continue
        m = model.load_model(os.path.join(WEIGHTS, file), cpu)
        with torch.no_grad():
            e = infer.encode_tile(m, **d, batch_pixels=batch, device=cpu)
        e[~observed] = np.nan
        np.save(os.path.join(out, name + ".npy"), e)
        if name != "teacher":
            codes, scales = squant.quantize(e[observed])
            np.save(os.path.join(out, name + "_int8.npy"), codes)
            np.save(os.path.join(out, name + "_scales.npy"), scales)
        print(tile, name, flush=True)
