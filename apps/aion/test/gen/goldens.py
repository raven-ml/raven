"""Convert upstream's tests/test_data/*.pt goldens to safetensors, one file per golden.

    uv run --locked goldens.py SRC_DIR DST_DIR

SRC_DIR is upstream's tests/test_data (PolymathicAI/AION at db7fba6d).

Each .pt holds a tensor or a nested dict of tensors. Nested keys are joined with '.'
("image.array"); a bare tensor is stored under "value". Booleans are stored as uint8,
since nx.io's safetensors reader maps BOOL; the original dtype is kept in the file's
metadata under "dtype.<key>". Non-tensor leaves (band name lists) go to metadata as text.
"""
import hashlib, json, os, sys

import numpy as np
import torch
from safetensors.numpy import save_file

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)


def flatten(prefix, x, arrays, meta):
    if isinstance(x, torch.Tensor):
        a = x.detach().cpu().numpy()
        meta[f"dtype.{prefix}"] = str(x.dtype)
        arrays[prefix] = np.ascontiguousarray(a.astype(np.uint8) if a.dtype == np.bool_ else a)
    elif isinstance(x, dict):
        for k, v in x.items():
            flatten(f"{prefix}.{k}" if prefix else str(k), v, arrays, meta)
    else:
        meta[f"value.{prefix}"] = json.dumps(x, default=str)


rows = []
for name in sorted(os.listdir(src)):
    if not name.endswith(".pt"):
        continue
    path = os.path.join(src, name)
    # weights_only: the goldens are downloads; they hold only tensors, dicts and lists.
    obj = torch.load(path, weights_only=True, map_location="cpu")
    arrays, meta = {}, {}
    flatten("" if isinstance(obj, dict) else "value", obj, arrays, meta)
    meta["source"] = name
    meta["source_sha256"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    out = os.path.join(dst, name[:-3] + ".safetensors")
    save_file(arrays, out, metadata=meta)
    rows.append((name, {k: (str(v.dtype), v.shape) for k, v in arrays.items()}))

for name, shapes in rows:
    print(name, shapes)
print("converted", len(rows))
