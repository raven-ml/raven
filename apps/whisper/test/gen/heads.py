# /// script
# requires-python = "==3.13.*"
# dependencies = [
#   "openai-whisper @ git+https://github.com/openai/whisper.git@86098128c0b4f24f0e2aa2994de830614b474227",
#   "torch==2.15.0",
#   "numpy==2.5.5",
#   "tiktoken==0.14.0",
# ]
#
# [tool.uv]
# # whisper declares triton on Linux x86-64 for its CUDA kernels; the
# # generators run on the CPU.
# override-dependencies = ["triton; sys_platform == 'never'"]
# sources = { torch = { index = "pytorch-cpu" } }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Alignment heads of every released checkpoint, for test_model, decoded as
upstream decodes them (whisper/model.py set_alignment_heads: gzip of base85),
with each checkpoint's decoder layers and heads from its dims.

    uv run apps/whisper/test/gen/heads.py OUT

OUT is apps/whisper/test/upstream.

Writes OUT/heads.json.
"""
import base64, gzip, json, os, sys

import numpy as np
from whisper import _ALIGNMENT_HEADS

DIMS = {"tiny.en": (4, 6), "tiny": (4, 6), "base.en": (6, 8), "base": (6, 8),
        "small.en": (12, 12), "small": (12, 12), "medium.en": (24, 16), "medium": (24, 16),
        "large-v1": (32, 20), "large-v2": (32, 20), "large-v3": (32, 20),
        "large-v3-turbo": (4, 20)}
table = []
for name, (layers, heads) in DIMS.items():
    a = np.frombuffer(gzip.decompress(base64.b85decode(_ALIGNMENT_HEADS[name])), dtype=bool)
    a = a.reshape(layers, heads)
    table.append({"name": name, "layers": layers, "heads": heads,
                  "alignment_heads": [[int(l), int(h)] for l, h in np.argwhere(a)]})
for alias, target in (("large", "large-v3"), ("turbo", "large-v3-turbo")):
    assert _ALIGNMENT_HEADS[alias] == _ALIGNMENT_HEADS[target]
json.dump({"generator": "apps/whisper/test/gen/heads.py", "models": table},
          open(os.path.join(sys.argv[1], "heads.json"), "w"))
print({t["name"]: len(t["alignment_heads"]) for t in table})
