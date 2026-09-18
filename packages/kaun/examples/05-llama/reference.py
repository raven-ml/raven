# Reference values for kaun's Llama example, from the transformers
# implementation at float32. Writes fixtures/<name>.json next to this script.
#
#   uv run --with torch --with transformers --with safetensors \
#          --with huggingface_hub --with accelerate reference.py
import hashlib, json, os
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM

# Token ids are fixed and in-vocabulary: no tokenizer is involved. Each model
# gets a prompt and a shorter one, fed together as a ragged batch by the
# validator and one at a time here.
MODELS = {
    "llama-3.2-1b": {
        "repo": "NousResearch/Llama-3.2-1B",
        "ids": [128000, 791, 6864, 315, 9822, 374, 279, 3363, 1174, 323, 433, 596],
        "short": [128000, 9906, 1917, 11, 420],
    },
    "tinyllama-1.1b": {
        "repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "ids": [1, 450, 7483, 310, 3444, 338, 278, 4234, 29892, 322, 372, 338],
        "short": [1, 15043, 3186, 29892, 445],
    },
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def top(logits, k=8):
    t = torch.topk(logits, k)
    return t.indices.tolist(), [float(v) for v in t.values]


here = os.path.dirname(os.path.abspath(__file__))
for name, spec in MODELS.items():
    weights = hf_hub_download(spec["repo"], "model.safetensors")
    model = AutoModelForCausalLM.from_pretrained(spec["repo"], torch_dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor([spec["ids"]]), output_hidden_states=True)
        short = model(torch.tensor([spec["short"]])).logits[0, -1]
    logits = out.logits[0]
    top_ids, top_values = top(logits[-1])
    short_ids, short_values = top(short)
    n = len(out.hidden_states) - 1
    fixture = {
        "repo": spec["repo"],
        "weights_sha256": sha256(weights),
        "ids": spec["ids"],
        "argmax_per_position": logits.argmax(-1).tolist(),
        "last_top_ids": top_ids,
        "last_top_values": top_values,
        "last_first_values": [float(v) for v in logits[-1][:8]],
        # The residual stream after every block, last position, first 8
        # features. The last entry is after the final norm.
        "hidden": {
            str(k): [float(v) for v in out.hidden_states[k][0, -1, :8]]
            for k in range(1, n + 1)
        },
        "short_ids": spec["short"],
        "short_top_ids": short_ids,
        "short_top_values": short_values,
    }
    path = os.path.join(here, "fixtures", name + ".json")
    with open(path, "w") as f:
        json.dump(fixture, f, indent=1)
        f.write("\n")
    print("wrote", path, "blocks =", n, "argmax last =", int(logits[-1].argmax()))
