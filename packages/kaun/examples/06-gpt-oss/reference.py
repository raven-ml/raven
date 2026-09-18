# Reference values for kaun's gpt-oss example, from the transformers
# implementation at float32 on CPU. Writes fixtures/<name>.json, the MoE block
# and the dequantiser, and fixtures/<name>-model.json, the whole model, next to
# this script. Only the tiny random checkpoints are read, a few MB each.
#
#   uv run --with torch==2.14.0 --with transformers==5.17.0 \
#          --with safetensors==0.8.0 --with huggingface_hub==1.32.0 \
#          --with accelerate==1.15.0 reference.py
import hashlib, json, os, sys
import torch, transformers, safetensors, huggingface_hub
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM
from transformers.integrations.mxfp4 import convert_moe_packed_tensors

LAYER = 0
# Token ids are fixed and in-vocabulary: no tokenizer is involved. The ragged
# pair is the long prompt and a shorter one, left-padded.
IDS = [200006, 17360, 200008, 3575, 553, 17554, 162016, 11, 261, 4410, 6439, 2359]
SHORT = [200006, 1428, 200008, 13225, 2375]
PAD = 199999

# The scales of tiny-random/gpt-oss-mxfp4 are drawn from 0..3, so its expert
# weights are of the order of 1e-38 and the experts reduce to their biases. A
# second set of cases adds SCALE_OFFSET to every scale byte before
# dequantising, which puts the weights at the magnitude of trained ones.
SCALE_OFFSET = 118
# Natural activations never reach the limit of the clamped activation. The
# "clamped" case feeds the block its recorded input times this factor.
CLAMP_FACTOR = 200.0
# The checkpoints' window of 128 never binds on a 12-token prompt. The whole
# model is recorded with this window instead, so that the sliding layer hides
# tokens and a chunk boundary falls inside a window.
WINDOW = 4
# Positions of the recorded rotary tables. Frequencies formed in float32 and in
# float64 differ by a unit in the last place, which the position multiplies.
TABLE_POSITIONS = [0, 1, 11, 127, 4095]

MODELS = {
    "gpt-oss-bf16": {"repo": "tiny-random/gpt-oss-bf16", "packed": False},
    "gpt-oss-mxfp4": {"repo": "tiny-random/gpt-oss-mxfp4", "packed": True},
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def flat(t):
    return [float(v) for v in t.detach().to(torch.float64).flatten()]


def ints(t):
    return [int(v) for v in t.detach().flatten()]


def clamp_hits(experts, seen):
    """How many pre-activations of the selected experts the limit changes."""
    x = seen["hidden"].reshape(-1, seen["hidden"].shape[-1])
    e = seen["experts"]
    w, b = experts.gate_up_proj[e], experts.gate_up_proj_bias[e]
    gate_up = torch.einsum("td,tkdf->tkf", x, w) + b
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    return int((gate > experts.limit).sum()), int((up.abs() > experts.limit).sum())


def record(model, run):
    """The MoE block of layer LAYER on one batch: what enters, what the router
    decides and what leaves."""
    mlp = model.model.layers[LAYER].mlp
    seen = {}
    hooks = [
        mlp.register_forward_hook(
            lambda m, args, out: seen.update(hidden=args[0], output=out[0])
        ),
        mlp.router.register_forward_hook(
            lambda m, args, out: seen.update(logits=out[0], weights=out[1], experts=out[2])
        ),
    ]
    with torch.no_grad():
        run(mlp)
        gate_hits, linear_hits = clamp_hits(mlp.experts, seen)
    for h in hooks:
        h.remove()
    return {
        "clamped_gates": gate_hits,
        "clamped_linears": linear_hits,
        "shape": list(seen["hidden"].shape),
        "hidden": flat(seen["hidden"]),
        "router_logits": flat(seen["logits"]),
        "experts": ints(seen["experts"]),
        "expert_weights": flat(seen["weights"]),
        "output": flat(seen["output"]),
    }


def cases(model):
    n, m = len(IDS), len(SHORT)

    def through_model(ids, mask):
        return lambda mlp: model(torch.tensor(ids), attention_mask=torch.tensor(mask))

    batch = record(model, through_model([IDS], [[1] * n]))
    ragged = record(
        model,
        through_model([IDS, [PAD] * (n - m) + SHORT], [[1] * n, [0] * (n - m) + [1] * m]),
    )
    hidden = torch.tensor(batch["hidden"], dtype=torch.float32).reshape(batch["shape"])
    clamped = record(model, lambda mlp: mlp(hidden * CLAMP_FACTOR))
    return {"batch": batch, "ragged": ragged, "clamped": clamped}


def packed(path, name):
    with safe_open(path, "pt") as f:
        prefix = f"model.layers.{LAYER}.mlp.experts.{name}"
        return f.get_tensor(prefix + "_blocks"), f.get_tensor(prefix + "_scales")


def dequantised(blocks, scales):
    # convert_moe_packed_tensors ends with a transpose to the module's
    # [experts, in, out]; the fixture keeps the checkpoint's [.., out, in].
    w = convert_moe_packed_tensors(blocks, scales, dtype=torch.float32)
    return w.transpose(1, 2).contiguous()


def dequant_case(blocks, scales):
    return {
        "blocks_shape": list(blocks.shape),
        "blocks": ints(blocks),
        "scales": ints(scales),
        "values": flat(dequantised(blocks, scales)),
    }


def ties():
    """torch.topk on rows with equal entries, and the softmax of the values."""
    rows = torch.zeros(3, 32)
    rows[1] = torch.arange(32, dtype=torch.float32).remainder(7) * 0.25
    rows[2, 5], rows[2, 9], rows[2, 20], rows[2, 21], rows[2, 30] = 3.0, 3.0, 1.5, 1.5, 1.5
    values, experts = torch.topk(rows, 4, dim=-1)
    return {
        "logits": flat(rows),
        "experts": ints(experts),
        "values": flat(values),
        "expert_weights": flat(torch.softmax(values, dim=1)),
    }


def top(logits, k=8):
    t = torch.topk(logits, k)
    return t.indices.tolist(), flat(t.values)


def record_model(model):
    """The whole model on the long prompt, as 05-llama records it, the two
    attention layers' inputs and outputs, and the short prompt alone."""
    layers = model.model.layers
    seen = {}
    hooks = [
        layer.self_attn.register_forward_hook(
            lambda m, args, kwargs, out, i=i: seen.update(
                {i: (kwargs["hidden_states"], out[0])}
            ),
            with_kwargs=True,
        )
        for i, layer in enumerate(layers)
    ]
    with torch.no_grad():
        out = model(torch.tensor([IDS]), output_hidden_states=True)
    for h in hooks:
        h.remove()
    with torch.no_grad():
        short = model(torch.tensor([SHORT])).logits[0, -1]
    logits = out.logits[0]
    top_ids, top_values = top(logits[-1])
    short_ids, short_values = top(short)
    n = len(out.hidden_states) - 1
    return {
        "argmax_per_position": logits.argmax(-1).tolist(),
        "last_top_ids": top_ids,
        "last_top_values": top_values,
        "last_first_values": flat(logits[-1][:8]),
        # The residual stream after every block, last position, first 8
        # features. The last entry is after the final norm.
        "hidden": {str(k): flat(out.hidden_states[k][0, -1, :8]) for k in range(1, n + 1)},
        "attention": {
            str(i): {
                "kind": model.config.layer_types[i],
                "input": flat(seen[i][0]),
                "output": flat(seen[i][1]),
            }
            for i in range(len(layers))
        },
        "short_top_ids": short_ids,
        "short_top_values": short_values,
    }


def rotary(model):
    rot = model.model.rotary_emb
    pos = torch.tensor([TABLE_POSITIONS])
    cos, sin = rot(torch.zeros(1, dtype=torch.float32), pos)
    return {
        "inv_freq": flat(rot.inv_freq),
        "attention_scaling": float(rot.attention_scaling),
        "positions": TABLE_POSITIONS,
        "cos": flat(cos),
        "sin": flat(sin),
    }


def set_scales(model, weights, offset):
    for i, layer in enumerate(model.model.layers):
        with safe_open(weights, "pt") as f:
            for param in ("gate_up_proj", "down_proj"):
                prefix = f"model.layers.{i}.mlp.experts.{param}"
                blocks, scales = f.get_tensor(prefix + "_blocks"), f.get_tensor(prefix + "_scales")
                w = convert_moe_packed_tensors(blocks, scales + offset, dtype=torch.float32)
                getattr(layer.mlp.experts, param).data = w


def model_fixture(spec, weights, provenance):
    model = AutoModelForCausalLM.from_pretrained(
        spec["repo"], dtype=torch.float32, experts_implementation="eager", sliding_window=WINDOW
    )
    model = model.float().eval()
    assert model.config._experts_implementation == "eager"
    assert model.config._attn_implementation == "eager"
    assert model.config.sliding_window == WINDOW
    fixture = dict(provenance)
    fixture.update(
        {
            "sliding_window": WINDOW,
            "ids": IDS,
            "short_ids": SHORT,
            "rotary": rotary(model),
            "sinks": {str(i): flat(l.self_attn.sinks) for i, l in enumerate(model.model.layers)},
        }
    )
    if not spec["packed"]:
        fixture["cases"] = {"0": record_model(model)}
    else:
        fixture["cases"] = {}
        for offset in (0, SCALE_OFFSET):
            set_scales(model, weights, offset)
            fixture["cases"][str(offset)] = record_model(model)
    return fixture


def write(name, fixture):
    path = os.path.join(here, "fixtures", name + ".json")
    with open(path, "w") as f:
        json.dump(fixture, f)
        f.write("\n")
    print("wrote", path, os.path.getsize(path), "bytes")


here = os.path.dirname(os.path.abspath(__file__))
for name, spec in MODELS.items():
    weights = hf_hub_download(spec["repo"], "model.safetensors")
    config = hf_hub_download(spec["repo"], "config.json")
    model = AutoModelForCausalLM.from_pretrained(
        spec["repo"], dtype=torch.float32, experts_implementation="eager"
    )
    # Without triton the MXFP4 loader dequantises the experts to bfloat16
    # whatever dtype is asked for.
    model = model.float().eval()
    assert model.config._experts_implementation == "eager"
    cfg = model.config
    provenance = {
        "repo": spec["repo"],
        "weights_sha256": sha256(weights),
        "config_sha256": sha256(config),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "safetensors": safetensors.__version__,
            "huggingface_hub": huggingface_hub.__version__,
        },
    }
    fixture = dict(provenance)
    fixture.update({
        "config": {
            "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "num_local_experts": cfg.num_local_experts,
            "num_experts_per_tok": cfg.num_experts_per_tok,
            "swiglu_limit": cfg.swiglu_limit,
        },
        "layer": LAYER,
        "ids": IDS,
        "short_ids": SHORT,
        "pad_id": PAD,
        "ties": ties(),
    })
    if not spec["packed"]:
        fixture["cases"] = {"0": cases(model)}
    else:
        experts = model.model.layers[LAYER].mlp.experts
        gate_up, down = packed(weights, "gate_up_proj"), packed(weights, "down_proj")
        fixture["cases"] = {}
        for offset in (0, SCALE_OFFSET):
            for param, (blocks, scales) in (("gate_up_proj", gate_up), ("down_proj", down)):
                w = convert_moe_packed_tensors(blocks, scales + offset, dtype=torch.float32)
                getattr(experts, param).data = w
            fixture["cases"][str(offset)] = cases(model)
        # Every scale byte but 253 and up, whose products leave float32.
        g = torch.Generator().manual_seed(0)
        sweep = torch.randint(0, 256, (1, 23, 11, 16), dtype=torch.uint8, generator=g)
        fixture["dequant"] = {
            "gate_up_proj": dequant_case(gate_up[0][:2, :3], gate_up[1][:2, :3]),
            "down_proj": dequant_case(down[0][30:, :3], down[1][30:, :3]),
            "every_scale": dequant_case(
                sweep, torch.arange(253, dtype=torch.uint8).reshape(1, 23, 11)
            ),
        }
    write(name, fixture)
    write(name + "-model", model_fixture(spec, weights, provenance))
