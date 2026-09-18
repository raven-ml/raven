# Reference logits for kaun's Llama example, from the transformers
# implementation at float32. Writes fixture.json next to this script.
#
#   uv run --with torch --with transformers --with safetensors --with huggingface_hub reference.py
import json, sys, hashlib, os
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM

repo = "NousResearch/Llama-3.2-1B"
weights = hf_hub_download(repo, "model.safetensors")
h = hashlib.sha256()
with open(weights, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
        h.update(chunk)

model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32)
model.eval()
# <|begin_of_text|> then arbitrary in-vocabulary ids: no tokenizer involved.
ids = [128000, 791, 6864, 315, 9822, 374, 279, 3363, 1174, 323, 433, 596]
with torch.no_grad():
    out = model(torch.tensor([ids]), output_hidden_states=True)
logits = out.logits[0]            # [seq; vocab]
last = logits[-1]
top = torch.topk(last, 8)
fixture = {
    "repo": repo,
    "weights_sha256": h.hexdigest(),
    "ids": ids,
    "argmax_per_position": logits.argmax(-1).tolist(),
    "last_top_ids": top.indices.tolist(),
    "last_top_values": [float(v) for v in top.values],
    "last_first_values": [float(v) for v in last[:8]],
    # The residual stream after blocks 1, 8 and 16, last position, first 8 features.
    "hidden": {str(k): [float(v) for v in out.hidden_states[k][0, -1, :8]] for k in (1, 8, 16)},
}
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixture.json")
json.dump(fixture, open(path, "w"), indent=1)
print("wrote", path, "argmax last =", int(last.argmax()))
