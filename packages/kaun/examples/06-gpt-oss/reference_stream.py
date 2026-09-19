# Reference values for kaun's gpt-oss example at sizes that do not fit in
# memory: the transformers implementation at float32 on CPU, one block at a
# time. transformers dequantises every expert when it loads the model, about
# 42 GB for gpt-oss-20b, so this recorder never builds the model. It reads a
# block's tensors from the safetensors files by name, dequantises that block's
# MXFP4 experts with transformers' own function, loads them into transformers'
# own GptOssDecoderLayer, runs the block on the residual stream of every
# prompt, records and frees it. The rotary tables, the two masks, the final
# norm and the head are transformers' too. Peak memory is one block's float32
# experts and the dequantiser's temporaries, about 6 GB for gpt-oss-20b.
#
#   uv run --with torch==2.14.0 --with transformers==5.17.0 \
#          --with safetensors==0.8.0 --with huggingface_hub==1.32.0 \
#          --with accelerate==1.15.0 reference_stream.py openai/gpt-oss-20b
#
# writes fixtures/<name>-stream.json. The checkpoint is read from kaun's hub
# cache, where the example's importer reads it, or from --dir, and is
# downloaded with huggingface_hub when neither has it.
#
# --whole also runs transformers' whole model in this process and compares
# every residual stream, every router decision and every logit with the
# streamed ones. It is the check of this recorder, for checkpoints that fit:
#
#   ... reference_stream.py tiny-random/gpt-oss-mxfp4 --whole --window 4 --scale-offset 118
#
# --dtype bfloat16 runs transformers at the precision the model is served at,
# experts dequantised to bfloat16 as its loader does. float32 over the same
# dequantised weights is the reference of a float32 implementation; the gap
# between the two recordings is what a bfloat16 implementation may differ by.
# On CPU it is slow: 8 minutes of processor time for gpt-oss-20b.
import argparse, hashlib, json, os, resource, sys, time
import torch, transformers, safetensors, huggingface_hub
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.integrations.mxfp4 import convert_moe_packed_tensors
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssDecoderLayer,
    GptOssRMSNorm,
    GptOssRotaryEmbedding,
)

# Token ids are fixed: no tokenizer is involved. "short" is the prompt of
# reference.py. "medium" and "long" are English prose encoded once with the
# checkpoint's tokenizer; "long" is longer than the sliding window of 128.
PROMPTS = {
    "short": [200006, 17360, 200008, 3575, 553, 17554, 162016, 11, 261, 4410, 6439, 2359],
    "medium": [976, 27467, 328, 13071, 306, 495, 2359, 19749, 1753, 6602, 316, 4242, 328, 37650,
               75351, 13071, 13, 11555, 8333, 382, 261, 114973, 7685, 95766, 5402, 12119, 28460,
               553, 16240, 472, 4242, 33736, 101868, 483, 261, 9599, 51159, 777, 3566, 328, 37650,
               75351, 13],
    "long": [32, 53853, 5734, 328, 1001, 11779, 326, 25565, 147647, 15897, 4748, 484, 261, 6602,
             306, 448, 1952, 10261, 1606, 134910, 316, 290, 1645, 7178, 12994, 11, 2049, 290,
             14210, 22119, 13384, 316, 290, 6062, 16281, 13, 2514, 13526, 484, 5734, 290, 15226,
             853, 316, 413, 7411, 1572, 290, 5734, 8807, 11, 1118, 382, 4436, 495, 27853, 18295,
             2966, 395, 261, 2049, 13, 623, 54782, 6855, 382, 19460, 1934, 1753, 4355, 11, 290,
             15288, 885, 16180, 13071, 553, 19460, 395, 1753, 6602, 11, 326, 290, 1721, 148063,
             553, 43991, 5761, 656, 1043, 25565, 10574, 18614, 13, 23207, 2105, 20102, 402, 261,
             99665, 540, 2461, 1058, 25, 290, 27380, 4895, 1504, 12790, 4730, 326, 553, 16240,
             472, 21402, 54912, 13, 1843, 290, 124475, 31523, 261, 33686, 483, 290, 8201, 10557,
             11, 503, 290, 91894, 64338, 553, 1277, 11, 503, 290, 22834, 148063, 553, 60816, 1261,
             1023, 1757, 625, 413, 11, 290, 19211, 28336, 540, 290, 1577, 4355, 1919, 290, 9809,
             14518, 11, 326, 290, 3019, 5003, 1118, 1001, 13],
}
# The first, middle and last blocks' residual streams are recorded whole at
# the last position. Every block is summarised at the positions [positions]
# gives.
FIRST = 8
PROJECTIONS = 16
TOP = 20
THREADS = 4


def positions(n):
    return sorted({0, 1, n // 2, n - 1})


def signs(rows, dim):
    """A fixed matrix of +1 and -1 from a linear congruential generator, so
    that the validator forms the same one."""
    state, out = 12345, []
    for _ in range(rows * dim):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(1.0 if (state >> 16) & 1 else -1.0)
    return torch.tensor(out, dtype=torch.float64).reshape(rows, dim)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def flat(t):
    # Nine significant digits identify a float32.
    return [float(f"{v:.9g}") for v in t.detach().to(torch.float64).flatten().tolist()]


def raven_cache(repo):
    root = os.environ.get("RAVEN_CACHE_ROOT") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "raven"
    )
    return os.path.join(root, "huggingface", repo.replace("/", "-"), "main")


class Checkpoint:
    """Tensors by name over one safetensors file or an index of shards. Files
    are memory-mapped and a tensor is read when asked for."""

    def __init__(self, directory):
        index = os.path.join(directory, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as f:
                self.file_of = json.load(f)["weight_map"]
            self.files = sorted(set(self.file_of.values()))
        else:
            self.files = ["model.safetensors"]
            with safe_open(os.path.join(directory, self.files[0]), "pt") as f:
                self.file_of = {name: self.files[0] for name in f.keys()}
        self.handles = {f: safe_open(os.path.join(directory, f), "pt") for f in self.files}

    def __contains__(self, name):
        return name in self.file_of

    def get(self, name):
        return self.handles[self.file_of[name]].get_tensor(name)

    def rows(self, name, ids):
        table = self.handles[self.file_of[name]].get_slice(name)
        return torch.stack([table[i] for i in ids])


def experts(ckpt, name, offset, dtype):
    """One projection of every expert at [dtype], in the module's layout
    [experts, inputs, outputs]. Float checkpoints store that layout; packed
    ones are dequantised by transformers, which also transposes them."""
    if name in ckpt:
        return ckpt.get(name).to(dtype)
    blocks, scales = ckpt.get(name + "_blocks"), ckpt.get(name + "_scales")
    return convert_moe_packed_tensors(
        blocks, scales + offset, dtype=dtype, rows_per_chunk=1 << 20
    )


def block(config, ckpt, i, offset, dtype):
    prefix = f"model.layers.{i}."
    with torch.device("meta"):
        layer = GptOssDecoderLayer(config, i)
    state = {}
    for name in layer.state_dict():
        if name in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"):
            state[name] = experts(ckpt, prefix + name, offset, dtype)
        else:
            state[name] = ckpt.get(prefix + name).to(dtype)
    layer.load_state_dict(state, strict=True, assign=True)
    return layer.eval()


def stream(config, ckpt, offset, dtype, log=lambda s: None):
    """The forward pass of every prompt, a block at a time. Per prompt: the
    residual stream entering block 0 and leaving every block, the stream
    between a block's attention and its experts, the router's logits and
    choices, the normalised last stream and the logits."""
    out = {
        p: {"residual": [], "middle": [], "router_logits": [], "experts": [], "expert_weights": []}
        for p in PROMPTS
    }
    rotary = GptOssRotaryEmbedding(config)
    state = {}
    for p, ids in PROMPTS.items():
        x = ckpt.rows("model.embed_tokens.weight", ids).to(dtype).unsqueeze(0)
        position_ids = torch.arange(len(ids)).unsqueeze(0)
        kwargs = dict(config=config, inputs_embeds=x, attention_mask=None, past_key_values=None)
        state[p] = {
            "x": x,
            "position_ids": position_ids,
            "rotary": rotary(x, position_ids),
            "masks": {
                "full_attention": create_causal_mask(**kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**kwargs),
            },
        }
        out[p]["residual"].append(x[0])
    for i in range(config.num_hidden_layers):
        t0 = time.time()
        layer = block(config, ckpt, i, offset, dtype)
        seen = {}
        hooks = [
            layer.self_attn.register_forward_hook(lambda m, a, o: seen.update(attention=o[0])),
            layer.mlp.router.register_forward_hook(
                lambda m, a, o: seen.update(logits=o[0], weights=o[1], experts=o[2])
            ),
        ]
        for p, s in state.items():
            with torch.no_grad():
                y = layer(
                    s["x"],
                    attention_mask=s["masks"][config.layer_types[i]],
                    position_ids=s["position_ids"],
                    position_embeddings=s["rotary"],
                )
            o = out[p]
            o["middle"].append((s["x"] + seen["attention"])[0])
            o["router_logits"].append(seen["logits"])
            o["experts"].append(seen["experts"])
            o["expert_weights"].append(seen["weights"])
            o["residual"].append(y[0])
            s["x"] = y
        for h in hooks:
            h.remove()
        del layer, seen
        log(f"block {i} ({config.layer_types[i]}): {time.time() - t0:.1f} s")
    norm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    norm.weight.data = ckpt.get("model.norm.weight").to(dtype)
    tied = "lm_head.weight" not in ckpt
    head = ckpt.get("model.embed_tokens.weight" if tied else "lm_head.weight").to(dtype)
    for p, s in state.items():
        with torch.no_grad():
            out[p]["normed"] = norm(s["x"])[0]
            out[p]["logits"] = out[p]["normed"] @ head.T
    return out, tied


def whole(repo, directory, config, offset, dtype):
    """transformers' whole model on the same prompts, as reference.py runs it."""
    model = AutoModelForCausalLM.from_pretrained(
        directory,
        dtype=dtype,
        experts_implementation="eager",
        sliding_window=config.sliding_window,
    )
    # Without triton the MXFP4 loader dequantises the experts to bfloat16
    # whatever dtype is asked for. At bfloat16 the model is left as loaded:
    # a conversion would also round the rotary frequencies, which the loader
    # keeps at float32.
    if dtype == torch.float32:
        model = model.float()
    model = model.eval()
    assert model.config._experts_implementation == "eager"
    assert model.config._attn_implementation == "eager"
    ckpt = Checkpoint(directory)
    for i, layer in enumerate(model.model.layers):
        for param in ("gate_up_proj", "down_proj"):
            name = f"model.layers.{i}.mlp.experts.{param}"
            if name not in ckpt:
                getattr(layer.mlp.experts, param).data = experts(ckpt, name, offset, dtype)
    out = {}
    for p, ids in PROMPTS.items():
        seen = {"logits": [], "experts": []}

        def routed(module, args, output):
            seen["logits"].append(output[0])
            seen["experts"].append(output[2])

        hooks = [layer.mlp.router.register_forward_hook(routed) for layer in model.model.layers]
        with torch.no_grad():
            r = model(torch.tensor([ids]), output_hidden_states=True)
        for h in hooks:
            h.remove()
        # hidden_states holds what enters every block, then the normalised
        # output of the last one.
        out[p] = {
            "entering": [h[0] for h in r.hidden_states[:-1]],
            "normed": r.hidden_states[-1][0],
            "logits": r.logits[0],
            "router_logits": seen["logits"],
            "experts": seen["experts"],
        }
    return out


def compare(streamed, reference):
    worst, equal = 0.0, True

    def check(what, a, b):
        nonlocal worst, equal
        assert a.shape == b.shape, (what, a.shape, b.shape)
        d = float((a - b).abs().max())
        scale = float(b.abs().max())
        worst = max(worst, d / scale if scale > 0 else d)
        equal = equal and torch.equal(a, b)
        return d

    for p, ref in reference.items():
        s = streamed[p]
        n = len(ref["entering"])
        ds = [check(f"{p} entering {k}", s["residual"][k], ref["entering"][k]) for k in range(n)]
        dn = check(f"{p} normed", s["normed"], ref["normed"])
        dl = check(f"{p} logits", s["logits"], ref["logits"])
        dr = max(
            check(f"{p} router {k}", s["router_logits"][k], ref["router_logits"][k])
            for k in range(n)
        )
        routed = all(torch.equal(a, b) for a, b in zip(s["experts"], ref["experts"]))
        argmax = torch.equal(s["logits"].argmax(-1), ref["logits"].argmax(-1))
        equal = equal and routed
        print(
            f"{p}: {len(PROMPTS[p])} tokens, max abs difference: residual streams {max(ds):.3g}, "
            f"normed {dn:.3g}, logits {dl:.3g}, router logits {dr:.3g}; "
            f"experts {'equal' if routed else 'DIFFER'}, argmax {'equal' if argmax else 'DIFFERS'}"
        )
    print(f"worst relative difference {worst:.3g}; bit-identical: {equal}")
    return worst


def summary(v, projection):
    v = v.to(torch.float64)
    return {
        "mean": float(v.mean()),
        "rms": float(v.pow(2).mean().sqrt()),
        "absmax": float(v.abs().max()),
        "first": flat(v[:FIRST]),
        "projection": flat(projection @ v),
    }


def fixture(config, streamed):
    layers = config.num_hidden_layers
    whole_blocks = {0, layers // 2, layers - 1}
    projection = signs(PROJECTIONS, config.hidden_size)
    prompts = {}
    for p, s in streamed.items():
        n = len(PROMPTS[p])
        at = positions(n)
        blocks = []
        for i in range(layers):
            logits = s["router_logits"][i]
            ranked = logits.sort(dim=-1, descending=True).values
            k = config.num_experts_per_tok
            b = {
                "experts": [int(v) for v in s["experts"][i].flatten()],
                # The least gap between the last selected logit and the first
                # rejected one: a selection is only as certain as this.
                "margin": float((ranked[:, k - 1] - ranked[:, k]).min()),
                "last_expert_weights": flat(s["expert_weights"][i][-1]),
                "after_attention": [summary(s["middle"][i][t], projection) for t in at],
                "after_block": [summary(s["residual"][i + 1][t], projection) for t in at],
            }
            if i in whole_blocks:
                b["after_block_last"] = flat(s["residual"][i + 1][-1])
            blocks.append(b)
        top = torch.topk(s["logits"], TOP, dim=-1)
        prompts[p] = {
            "ids": PROMPTS[p],
            "positions": at,
            "embedded": [summary(s["residual"][0][t], projection) for t in at],
            "blocks": blocks,
            "normed": [summary(s["normed"][t], projection) for t in at],
            "normed_last": flat(s["normed"][-1]),
            "argmax_per_position": s["logits"].argmax(-1).tolist(),
            "top_ids": top.indices.tolist(),
            "top_values": [flat(row) for row in top.values],
        }
    return {
        "first": FIRST,
        "projections": PROJECTIONS,
        "projection": "rows of +1 and -1, row-major from the generator "
        "s <- (1103515245 s + 12345) mod 2^31 started at 12345, +1 where bit 16 of s is set",
        "prompts": prompts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--dir", help="a directory holding config.json and the safetensors files")
    ap.add_argument("--window", type=int, help="override the configuration's sliding window")
    ap.add_argument("--scale-offset", type=int, default=0, help="added to every MXFP4 scale byte")
    ap.add_argument("--whole", action="store_true", help="compare with the whole model; writes nothing")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--out", help="the fixture's path; fixtures/<name>-stream.json by default")
    args = ap.parse_args()
    torch.set_num_threads(THREADS)
    dtype = getattr(torch, args.dtype)
    started = time.time()

    directory = args.dir or raven_cache(args.repo)
    if not os.path.exists(os.path.join(directory, "config.json")):
        directory = huggingface_hub.snapshot_download(
            args.repo, allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json"]
        )
    config = AutoConfig.from_pretrained(directory)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    if args.window is not None:
        config.sliding_window = args.window
    ckpt = Checkpoint(directory)

    streamed, tied = stream(
        config, ckpt, args.scale_offset, dtype, log=lambda s: print(s, flush=True)
    )
    if args.whole:
        worst = compare(streamed, whole(args.repo, directory, config, args.scale_offset, dtype))
        sys.exit(0 if worst < 1e-6 else 1)

    out = {
        "repo": args.repo,
        "files_sha256": {f: sha256(os.path.join(directory, f)) for f in ["config.json"] + ckpt.files},
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "safetensors": safetensors.__version__,
            "huggingface_hub": huggingface_hub.__version__,
        },
        "dtype": args.dtype,
        "threads": THREADS,
        "sliding_window": config.sliding_window,
        "scale_offset": args.scale_offset,
        "tied": tied,
    }
    out.update(fixture(config, streamed))
    here = os.path.dirname(os.path.abspath(__file__))
    path = args.out or os.path.join(here, "fixtures", args.repo.split("/")[-1] + "-stream.json")
    with open(path, "w") as f:
        json.dump(out, f)
        f.write("\n")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 30 if sys.platform == "darwin" else 1 << 20)
    print(f"wrote {path}, {os.path.getsize(path)} bytes, {time.time() - started:.0f} s, peak {peak:.2f} GB")


main()
