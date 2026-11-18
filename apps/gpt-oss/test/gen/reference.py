# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.14.0",
#   "transformers==5.17.0",
#   "safetensors==0.8.0",
#   "accelerate==1.15.0",
# ]
# ///
"""Write gpt-oss's small checkpoints and record transformers on them, at
float32 on CPU.

    uv run --python 3.12 apps/gpt-oss/test/gen/reference.py

Writes test/checkpoints/<name>/, a two-layer gpt-oss of random weights drawn
from SEED, with float or MXFP4 experts, and test/fixtures/<name>.json, what
transformers computes from it: the MoE block of layer LAYER and the whole
model.
"""
import json, os, sys
import torch, transformers, safetensors
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM
from transformers.integrations.mxfp4 import convert_moe_packed_tensors

LAYER = 0
SEED = 20261118
# The shapes of tiny-random/gpt-oss-mxfp4, on a vocabulary of VOCAB and a
# sliding window of 4, which hides tokens of a 12-token prompt.
VOCAB = 1024
DIM, HIDDEN, EXPERTS, HEADS, KV_HEADS, HEAD_DIM = 32, 64, 32, 2, 1, 32
CONFIG = {
    "architectures": ["GptOssForCausalLM"],
    "model_type": "gpt_oss",
    "attention_bias": True,
    "attention_dropout": 0.0,
    "experts_per_token": 4,
    "head_dim": HEAD_DIM,
    "hidden_act": "silu",
    "hidden_size": DIM,
    "initial_context_length": 4096,
    "initializer_range": 0.02,
    "intermediate_size": HIDDEN,
    "layer_types": ["sliding_attention", "full_attention"],
    "max_position_embeddings": 131072,
    "num_attention_heads": HEADS,
    "num_experts_per_tok": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": KV_HEADS,
    "num_local_experts": EXPERTS,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "factor": 32.0,
        "original_max_position_embeddings": 4096,
        "rope_type": "yarn",
        "truncate": False,
    },
    "rope_theta": 150000,
    "sliding_window": 4,
    "swiglu_limit": 7.0,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "vocab_size": VOCAB,
}
MXFP4 = {
    "modules_to_not_convert": [
        "model.layers.*.self_attn",
        "model.layers.*.mlp.router",
        "model.embed_tokens",
        "lm_head",
    ],
    "quant_method": "mxfp4",
}
# Token ids are fixed: no tokenizer is involved. The ragged pair is the long
# prompt and a shorter one, left-padded.
IDS = [6, 517, 8, 1000, 553, 704, 162, 11, 261, 441, 643, 235]
SHORT = [6, 142, 8, 132, 237]
PAD = 0
# Natural activations never reach the limit of the clamped activation. The
# "clamped" case feeds the block its recorded input times this factor.
CLAMP_FACTOR = 200.0

MODELS = {"gpt-oss-bf16": False, "gpt-oss-mxfp4": True}


def weights(packed, g):
    """The checkpoint's tensors, named, shaped and stored as gpt-oss's are.
    Packed expert scales are the bytes 118 to 121, the magnitude of trained
    weights."""
    def normal(*shape, scale=0.5, mean=0.0):
        return (mean + scale * torch.randn(*shape, generator=g)).to(torch.bfloat16)

    t = {"model.embed_tokens.weight": normal(VOCAB, DIM), "model.norm.weight": normal(DIM, mean=1.0)}
    for i in range(CONFIG["num_hidden_layers"]):
        p = f"model.layers.{i}."
        t[p + "input_layernorm.weight"] = normal(DIM, mean=1.0)
        t[p + "post_attention_layernorm.weight"] = normal(DIM, mean=1.0)
        q, kv = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
        for name, outputs, inputs in (("q_proj", q, DIM), ("k_proj", kv, DIM), ("v_proj", kv, DIM), ("o_proj", DIM, q)):
            t[p + f"self_attn.{name}.weight"] = normal(outputs, inputs, scale=0.2)
            t[p + f"self_attn.{name}.bias"] = normal(outputs, scale=0.2)
        t[p + "self_attn.sinks"] = normal(HEADS)
        t[p + "mlp.router.weight"] = normal(EXPERTS, DIM)
        t[p + "mlp.router.bias"] = normal(EXPERTS)
        for name, outputs, inputs in (("gate_up_proj", 2 * HIDDEN, DIM), ("down_proj", DIM, HIDDEN)):
            e = p + "mlp.experts." + name
            t[e + "_bias"] = normal(EXPERTS, outputs, scale=0.2)
            if packed:
                t[e + "_blocks"] = torch.randint(0, 256, (EXPERTS, outputs, inputs // 32, 16), generator=g, dtype=torch.uint8)
                t[e + "_scales"] = torch.randint(118, 122, (EXPERTS, outputs, inputs // 32), generator=g, dtype=torch.uint8)
            else:
                t[e] = normal(EXPERTS, inputs, outputs, scale=0.2)
    return t


def checkpoint(name, packed):
    """test/checkpoints/<name>/: config.json and model.safetensors."""
    out = os.path.join(here, "..", "checkpoints", name)
    os.makedirs(out, exist_ok=True)
    config = dict(CONFIG, **({"quantization_config": MXFP4} if packed else {}))
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    g = torch.Generator().manual_seed(SEED + packed)
    save_file(weights(packed, g), os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    return out


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


def write(name, fixture):
    path = os.path.join(here, "..", "fixtures", name + ".json")
    with open(path, "w") as f:
        json.dump(fixture, f)
        f.write("\n")
    print("wrote", path, os.path.getsize(path), "bytes")


def load(directory):
    """transformers' model of [directory] at float32, MXFP4 experts decoded at
    float32: without triton its loader decodes them to bfloat16."""
    model = AutoModelForCausalLM.from_pretrained(
        directory, dtype=torch.float32, experts_implementation="eager"
    )
    model = model.float().eval()
    assert model.config._experts_implementation == "eager"
    assert model.config._attn_implementation == "eager"
    with safe_open(os.path.join(directory, "model.safetensors"), "pt") as f:
        for i, layer in enumerate(model.model.layers):
            for param in ("gate_up_proj", "down_proj"):
                prefix = f"model.layers.{i}.mlp.experts.{param}"
                if prefix + "_blocks" in f.keys():
                    blocks, scales = f.get_tensor(prefix + "_blocks"), f.get_tensor(prefix + "_scales")
                    w = convert_moe_packed_tensors(blocks, scales, dtype=torch.float32)
                    getattr(layer.mlp.experts, param).data = w
    return model


here = os.path.dirname(os.path.abspath(__file__))
for name, packed in MODELS.items():
    model = load(checkpoint(name, packed))
    fixture = {
        "checkpoint": name,
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "safetensors": safetensors.__version__,
        },
        "layer": LAYER,
        "ids": IDS,
        "short_ids": SHORT,
        "block": cases(model),
    }
    fixture.update(record_model(model))
    write(name, fixture)
