# Reference values for kaun's Llama example on GGUF checkpoints, from the
# transformers implementation at float32 over the checkpoint's weights as the
# gguf package dequantises them. Writes fixtures/<name>.json next to this
# script, in reference.py's format, with the file's name under "file".
#
#   uv run --with torch==2.14.1 --with transformers==5.22.1 \
#          --with gguf==0.17.1 --with numpy==2.2.6 \
#          --with accelerate==1.15.0 reference_gguf.py [DIR]
#
# DIR holds the GGUF files; without it they are downloaded.
import hashlib, json, os, sys
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM

REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
# The prompts of reference.py's TinyLlama fixture.
IDS = [1, 450, 7483, 310, 3444, 338, 278, 4234, 29892, 322, 372, 338]
SHORT = [1, 15043, 3186, 29892, 445]
# Q4_K_M stores most matrices as Q4_K and the rest as Q6_K.
MODELS = {
    "tinyllama-1.1b-q8_0": "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
    "tinyllama-1.1b-q6_k": "tinyllama-1.1b-chat-v1.0.Q6_K.gguf",
    "tinyllama-1.1b-q4_k_m": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
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
for name, file in MODELS.items():
    if len(sys.argv) > 1:
        path = os.path.join(sys.argv[1], file)
    else:
        path = hf_hub_download(REPO, file)
    model = AutoModelForCausalLM.from_pretrained(
        os.path.dirname(path), gguf_file=os.path.basename(path), dtype=torch.float32
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor([IDS]), output_hidden_states=True)
        short = model(torch.tensor([SHORT])).logits[0, -1]
    logits = out.logits[0]
    top_ids, top_values = top(logits[-1])
    short_ids, short_values = top(short)
    n = len(out.hidden_states) - 1
    fixture = {
        "repo": REPO,
        "file": file,
        "weights_sha256": sha256(path),
        "ids": IDS,
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
        "short_ids": SHORT,
        "short_top_ids": short_ids,
        "short_top_values": short_values,
    }
    out_path = os.path.join(here, "fixtures", name + ".json")
    with open(out_path, "w") as f:
        json.dump(fixture, f, indent=1)
        f.write("\n")
    print("wrote", out_path, "blocks =", n, "argmax last =", int(logits[-1].argmax()))
